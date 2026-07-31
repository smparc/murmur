"""
Prometheus metrics for the Murmur services.

Every metric defined here is written to by something. That is a deliberate
constraint: a dashboard panel wired to a counter nobody increments reads as
"zero anomalies", which is indistinguishable from "healthy" and is the worst
possible failure mode for a monitoring system.

Multiprocess note
-----------------
Running uvicorn with more than one worker gives each process its own registry,
so ``/metrics`` returns whichever worker happened to serve the scrape. Setting
``PROMETHEUS_MULTIPROC_DIR`` switches this module to the multiprocess collector
so counters aggregate across workers.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    multiprocess,
)

log = logging.getLogger(__name__)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


# ---------------------------------------------------------------------------
# Request-level
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "murmur_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "murmur_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ---------------------------------------------------------------------------
# Model-level
# ---------------------------------------------------------------------------

MODEL_INFERENCE_TIME = Summary(
    "murmur_model_inference_seconds",
    "Model inference time",
    ["model"],
)

MODEL_LOADED = Gauge(
    "murmur_model_loaded",
    "1 when a model is resident and serving, 0 otherwise",
    ["model"],
)

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

ANOMALY_COUNT = Counter(
    "murmur_anomalies_total",
    "Total anomalies detected",
    ["node_id", "severity"],
)

ANOMALY_SCORE = Gauge(
    "murmur_anomaly_score",
    "Latest normalized anomaly score per node",
    ["node_id"],
)

ANOMALY_Z_SCORE = Gauge(
    "murmur_anomaly_z_score",
    "Latest robust z-score per node",
    ["node_id"],
)

TTF_PREDICTION = Gauge(
    "murmur_ttf_prediction",
    "Latest TTF prediction per node (0=healthy, 1=imminent failure)",
    ["node_id"],
)

# ---------------------------------------------------------------------------
# Pipeline / system
# ---------------------------------------------------------------------------

ACTIVE_WS_CLIENTS = Gauge(
    "murmur_active_websocket_clients",
    "Number of active WebSocket connections",
)

FRAMES_PROCESSED = Counter(
    "murmur_frames_processed_total",
    "Total spectrogram frames processed",
    ["node_id"],
)

FRAMES_DROPPED = Counter(
    "murmur_frames_dropped_total",
    "Frames discarded before processing",
    ["node_id", "reason"],
)

KAFKA_CONSUMER_LAG = Gauge(
    "murmur_kafka_consumer_lag",
    "Kafka consumer lag in messages",
    ["topic", "partition"],
)

# -- Array liveness --

ARRAY_NODES_REPORTING = Gauge(
    "murmur_array_nodes_reporting",
    "Microphones contributing to the most recent graph snapshot",
)
"""
Alert on this dropping below the configured array size.

A snapshot assembled from a subset of the array is still useful, but the
spatial model is reasoning over a graph with holes in it, and the operator has
to know that before trusting a per-node score.
"""

SNAPSHOTS_EMITTED = Counter(
    "murmur_snapshots_total",
    "Graph snapshots released for inference",
    ["mode"],  # complete | degraded
)

NODE_DROPPED = Counter(
    "murmur_node_dropped_total",
    "Microphones evicted from the assembler for exceeding the staleness bound",
    ["node_id"],
)

TELEMETRY_DROPPED = Counter(
    "murmur_telemetry_dropped_total",
    "Scored payloads the telemetry API refused or was unreachable for",
    ["node_id"],
)

# -- Spatial acoustics --

ACOUSTIC_COHERENCE = Gauge(
    "murmur_acoustic_coherence",
    "Mean GCC-PHAT correlation peak across microphone pairs (0-1)",
)

ARRAY_CLOCK_SPREAD = Gauge(
    "murmur_array_clock_spread_seconds",
    "Timestamp spread across the microphone array for one acoustic instant",
)
"""
Alert on this. Inter-microphone delays span ~15 ms on a 5 m array, so once edge
clock skew approaches that, every TDOA estimate and every source position
derived from it is noise wearing a confident number.
"""

SOURCE_LOCALIZED = Counter(
    "murmur_source_localizations_total",
    "Acoustic instants that yielded a source position",
)

PIPELINE_ERRORS = Counter(
    "murmur_pipeline_errors_total",
    "Unrecoverable errors by stage",
    ["stage"],
)

# End-to-end age of a frame when telemetry is emitted — the number an operator
# actually cares about, since it bounds how stale an alert can be.
END_TO_END_LATENCY = Histogram(
    "murmur_end_to_end_latency_seconds",
    "Seconds from edge capture to telemetry emission",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def track_latency(endpoint: str):
    """
    Decorate an async endpoint to record its count, status and latency.

    Applying this is what makes ``murmur_requests_total`` and
    ``murmur_request_latency_seconds`` non-zero; previously it was defined and
    never used, so both series reported nothing forever.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            status = "success"
            try:
                return await func(*args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
                REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()

        return wrapper

    return decorator


def track_inference(model_name: str):
    """Context manager timing a model forward pass."""
    return MODEL_INFERENCE_TIME.labels(model=model_name).time()


@contextmanager
def track_stage(stage: str) -> Iterator[None]:
    """Count unrecoverable errors raised inside a named pipeline stage."""
    try:
        yield
    except Exception:
        PIPELINE_ERRORS.labels(stage=stage).inc()
        raise


def record_consumer_lag(consumer, topics: list[str]) -> None:
    """
    Publish per-partition consumer lag.

    Lag is the single best early warning that ingestion is falling behind the
    edge devices; on a monitoring system, silent backpressure means alerts
    arrive long after the event they describe.
    """
    try:
        from confluent_kafka import TopicPartition

        assignment = consumer.assignment()
        if not assignment:
            return
        for tp in assignment:
            if tp.topic not in topics:
                continue
            _low, high = consumer.get_watermark_offsets(tp, timeout=1.0, cached=True)
            position = consumer.position([TopicPartition(tp.topic, tp.partition)])
            if not position:
                continue
            current = position[0].offset
            if current is None or current < 0:
                continue
            KAFKA_CONSUMER_LAG.labels(topic=tp.topic, partition=str(tp.partition)).set(
                max(0, high - current)
            )
    except Exception:  # pragma: no cover - telemetry must never break the pipeline
        log.debug("Could not sample consumer lag", exc_info=True)


def render() -> bytes:
    """Serialize the registry, aggregating across workers when configured."""
    multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest()


__all__ = [
    "ACOUSTIC_COHERENCE",
    "ACTIVE_WS_CLIENTS",
    "ANOMALY_COUNT",
    "ANOMALY_SCORE",
    "ANOMALY_Z_SCORE",
    "ARRAY_CLOCK_SPREAD",
    "ARRAY_NODES_REPORTING",
    "CONTENT_TYPE",
    "END_TO_END_LATENCY",
    "FRAMES_DROPPED",
    "FRAMES_PROCESSED",
    "KAFKA_CONSUMER_LAG",
    "MODEL_INFERENCE_TIME",
    "MODEL_LOADED",
    "NODE_DROPPED",
    "PIPELINE_ERRORS",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "SNAPSHOTS_EMITTED",
    "SOURCE_LOCALIZED",
    "TELEMETRY_DROPPED",
    "TTF_PREDICTION",
    "record_consumer_lag",
    "render",
    "track_inference",
    "track_latency",
    "track_stage",
]
