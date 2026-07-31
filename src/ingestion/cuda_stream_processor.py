"""
GPU stream processor: Kafka -> mel-spectrogram -> sliding window -> Kafka.

Consumes raw audio, computes log-mel spectrograms on the GPU, maintains a
per-node temporal buffer of ``SEQ_LENGTH`` frames, and republishes both single
frames (for low-latency anomaly scoring) and complete temporal windows (for the
ST-GNN and Liquid Network).

Two details carry most of the throughput:

- **Micro-batching.** Chunks are consumed in batches and the mel transform runs
  once across all of them. Per-message GPU calls on an 8000-sample chunk are
  dominated by kernel-launch and transfer overhead, so a "GPU-accelerated"
  pipeline that processes one message at a time is barely faster than CPU.
- **Manual offset commits.** Offsets advance only after downstream publish
  succeeds. Auto-commit acknowledges data the moment it is polled, so a crash
  mid-batch silently loses frames — unacceptable for a system whose value
  proposition is not missing events.
"""

from __future__ import annotations

import logging
import signal
import time
from collections import deque
from types import FrameType

import msgpack
import numpy as np
import torch
import torchaudio
from confluent_kafka import Consumer, KafkaError, Message, Producer

from src.ingestion.spatial_probe import SpatialProbe
from src.observability.metrics import (
    ACOUSTIC_COHERENCE,
    ARRAY_CLOCK_SPREAD,
    FRAMES_DROPPED,
    FRAMES_PROCESSED,
    PIPELINE_ERRORS,
    SOURCE_LOCALIZED,
    record_consumer_lag,
)
from src.settings import settings

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upper bound on messages fused into a single GPU call.
CONSUME_BATCH_SIZE = 32
CONSUME_TIMEOUT_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Sliding window buffer
# ---------------------------------------------------------------------------


class SlidingWindowBuffer:
    """
    Per-node circular buffer of spectrogram frames.

    The ST-GNN and LNN need ``SEQ_LENGTH`` frames of context to say anything
    meaningful; fed single frames, their temporal machinery is inert.

    Note this state is *in-process*. Two replicas in one consumer group each
    hold a partial view, so partitioning must keep a given node pinned to one
    consumer — see the key used when producing, and the deployment notes.
    """

    def __init__(self, window_size: int, num_nodes: int):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.buffers: dict[int, deque] = {i: deque(maxlen=window_size) for i in range(num_nodes)}
        self.timestamps: dict[int, deque] = {i: deque(maxlen=window_size) for i in range(num_nodes)}

    def push(self, node_id: int, spectrogram: np.ndarray, timestamp: float) -> None:
        if node_id not in self.buffers:
            self.buffers[node_id] = deque(maxlen=self.window_size)
            self.timestamps[node_id] = deque(maxlen=self.window_size)
        self.buffers[node_id].append(spectrogram)
        self.timestamps[node_id].append(timestamp)

    def is_ready(self, node_id: int) -> bool:
        return len(self.buffers.get(node_id, ())) >= self.window_size

    def get_window(self, node_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns ``(spectrograms, timespans)``.

        ``spectrograms`` is ``(window_size, n_mels, frames)``; ``timespans`` is
        the inter-frame interval in seconds, which the CfC integrates over.
        """
        frames = list(self.buffers[node_id])
        stamps = list(self.timestamps[node_id])
        spectrograms = np.stack(frames, axis=0)

        deltas = np.diff(stamps, prepend=stamps[0])
        # The first delta is unknowable; reuse the second so the CfC does not
        # integrate over a spurious zero interval.
        if len(deltas) > 1:
            deltas[0] = deltas[1]
        else:
            deltas[0] = settings.CHUNK_DURATION

        # Clock skew between edge devices, or a device whose clock steps
        # backwards, otherwise yields negative intervals.
        deltas = np.clip(deltas, 1e-3, 60.0)
        return spectrograms, deltas.astype(np.float32)

    def get_latest(self, node_id: int) -> np.ndarray | None:
        buf = self.buffers.get(node_id)
        return buf[-1] if buf else None


# ---------------------------------------------------------------------------
# Kafka + transform setup
# ---------------------------------------------------------------------------


def create_kafka_clients() -> tuple[Consumer, Producer]:
    """Consumer/producer pair configured for at-least-once delivery."""
    consumer_conf = {
        "bootstrap.servers": settings.KAFKA_BROKER,
        "group.id": settings.KAFKA_GROUP_ID,
        "auto.offset.reset": "latest",
        # Offsets are committed explicitly, after a successful publish.
        "enable.auto.commit": False,
        "session.timeout.ms": 30_000,
        "max.poll.interval.ms": 300_000,
    }
    producer_conf = {
        "bootstrap.servers": settings.KAFKA_BROKER,
        "compression.type": settings.KAFKA_COMPRESSION,
        "linger.ms": 5,
        "batch.num.messages": 100,
        "enable.idempotence": True,
        "acks": "all",
    }
    return Consumer(consumer_conf), Producer(producer_conf)


def get_mel_spectrogram_transform() -> torch.nn.Module:
    """
    Log-mel transform on the target device.

    Decibel scaling matters: raw mel power spans many orders of magnitude, so a
    linear-power spectrogram lets a handful of loud low-frequency bins dominate
    both the autoencoder's reconstruction loss and the GCN's activations.
    """
    return torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=settings.SAMPLE_RATE,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH,
            n_mels=settings.N_MELS,
        ),
        torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0),
    ).to(device)


def _delivery_callback(err, msg) -> None:
    if err is not None:
        PIPELINE_ERRORS.labels(stage="kafka_produce").inc()
        log.error("Delivery failed for key=%s: %s", msg.key(), err)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_message(msg: Message) -> tuple[int, float, np.ndarray] | None:
    """Validate and decode one raw-audio message, or ``None`` if unusable."""
    try:
        payload = msgpack.unpackb(msg.value(), raw=False)
        node_id = int(payload["node_id"])
        timestamp = float(payload["timestamp"])
        audio_bytes = payload["audio"]
    except Exception:
        PIPELINE_ERRORS.labels(stage="decode").inc()
        log.warning("Undecodable message on %s", msg.topic(), exc_info=True)
        return None

    node_label = str(node_id)
    if not audio_bytes or len(audio_bytes) < 4:
        FRAMES_DROPPED.labels(node_id=node_label, reason="empty").inc()
        return None
    if len(audio_bytes) % 4:
        FRAMES_DROPPED.labels(node_id=node_label, reason="truncated").inc()
        return None

    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if not np.isfinite(audio).all():
        FRAMES_DROPPED.labels(node_id=node_label, reason="non_finite").inc()
        return None
    if audio.size != settings.SAMPLES_PER_CHUNK:
        FRAMES_DROPPED.labels(node_id=node_label, reason="wrong_length").inc()
        log.warning(
            "Node %d sent %d samples, expected %d",
            node_id,
            audio.size,
            settings.SAMPLES_PER_CHUNK,
        )
        return None

    return node_id, timestamp, audio


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class _Shutdown:
    """Flips on SIGTERM/SIGINT so Kubernetes gets a clean drain."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        log.info("Received signal %s — draining", signum)
        self.requested = True


def process_stream(max_batches: int | None = None) -> int:
    """
    Run the ingestion loop.

    ``max_batches`` bounds the number of iterations, which makes the loop
    testable without a background thread.

    Returns the number of frames processed.
    """
    consumer, producer = create_kafka_clients()
    consumer.subscribe([settings.RAW_TOPIC])

    mel_transform = get_mel_spectrogram_transform()
    buffer = SlidingWindowBuffer(settings.SEQ_LENGTH, settings.NUM_NODES)
    shutdown = _Shutdown() if max_batches is None else None

    # TDOA has to be measured here, on raw waveforms: the mel transform below
    # discards phase, and phase is where the inter-channel delay lives.
    probe = (
        SpatialProbe(
            mic_coords=np.asarray(settings.MIC_COORDS, dtype=np.float64),
            sample_rate=settings.SAMPLE_RATE,
            staleness_tolerance=settings.TDOA_STALENESS_TOLERANCE,
            min_coherence=settings.TDOA_MIN_COHERENCE,
            interp=settings.TDOA_INTERP,
        )
        if settings.TDOA_ENABLED
        else None
    )

    consecutive_errors = 0
    max_errors = 50
    frames_processed = 0
    batches = 0
    last_lag_sample = 0.0

    log.info(
        "Ingestion listening on %s (device=%s, window=%d frames, batch<=%d)",
        settings.RAW_TOPIC,
        device,
        settings.SEQ_LENGTH,
        CONSUME_BATCH_SIZE,
    )

    try:
        while True:
            if shutdown is not None and shutdown.requested:
                break
            if max_batches is not None and batches >= max_batches:
                break
            batches += 1

            messages = consumer.consume(CONSUME_BATCH_SIZE, timeout=CONSUME_TIMEOUT_SECONDS)
            if not messages:
                continue

            decoded: list[tuple[int, float, np.ndarray]] = []
            fatal = False
            for msg in messages:
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    consecutive_errors += 1
                    log.warning(
                        "Consumer error (%d/%d): %s",
                        consecutive_errors,
                        max_errors,
                        msg.error(),
                    )
                    if consecutive_errors >= max_errors:
                        log.critical("Too many consecutive consumer errors — shutting down")
                        fatal = True
                    continue
                record = decode_message(msg)
                if record is not None:
                    decoded.append(record)

            if fatal:
                break
            if not decoded:
                continue
            consecutive_errors = 0

            try:
                # One transfer and one kernel launch for the whole batch.
                waveforms = torch.from_numpy(np.stack([d[2] for d in decoded])).to(
                    device, non_blocking=True
                )
                with torch.no_grad():
                    spectrograms = mel_transform(waveforms).cpu().numpy()
            except Exception:
                PIPELINE_ERRORS.labels(stage="mel_transform").inc()
                log.exception("Mel transform failed for a batch of %d", len(decoded))
                continue

            for (node_id, timestamp, raw), spec in zip(decoded, spectrograms, strict=True):
                _publish(producer, buffer, node_id, timestamp, spec)
                frames_processed += 1
                FRAMES_PROCESSED.labels(node_id=str(node_id)).inc()

                if probe is not None:
                    probe.push(node_id, timestamp, raw)

            if probe is not None:
                _publish_spatial(producer, probe)

            producer.poll(0)

            # Offsets advance only now that the derived frames are queued.
            try:
                consumer.commit(asynchronous=True)
            except Exception:
                PIPELINE_ERRORS.labels(stage="commit").inc()
                log.warning("Offset commit failed", exc_info=True)

            now = time.monotonic()
            if now - last_lag_sample > 10.0:
                record_consumer_lag(consumer, [settings.RAW_TOPIC])
                last_lag_sample = now

            if frames_processed and frames_processed % 100 < len(decoded):
                log.info("Processed %d frames", frames_processed)

    except KeyboardInterrupt:  # pragma: no cover
        log.info("Interrupted — terminating ingestion stream")
    finally:
        log.info("Flushing producer and closing consumer ...")
        producer.flush(timeout=10)
        consumer.close()

    return frames_processed


def _publish_spatial(producer: Producer, probe: SpatialProbe) -> None:
    """
    Solve the array and emit acoustic geometry, if a full instant is available.

    Failure here must never stall ingestion: localization is an enrichment, and
    the pipeline has to keep delivering spectrograms if the array is incomplete
    or the geometry is degenerate.
    """
    try:
        snapshot = probe.solve()
    except Exception:
        PIPELINE_ERRORS.labels(stage="spatial_probe").inc()
        log.warning("Spatial probe failed", exc_info=True)
        probe.reset()
        return

    if snapshot is None:
        return

    ACOUSTIC_COHERENCE.set(snapshot.mean_coherence)
    ARRAY_CLOCK_SPREAD.set(snapshot.clock_spread)
    if snapshot.localized:
        SOURCE_LOCALIZED.inc()

    # Unkeyed: a spatial snapshot describes the whole array, not one node.
    producer.produce(
        settings.SPATIAL_TOPIC,
        value=msgpack.packb(snapshot.to_payload(), use_bin_type=True),
        callback=_delivery_callback,
    )


def _publish(
    producer: Producer,
    buffer: SlidingWindowBuffer,
    node_id: int,
    timestamp: float,
    spec: np.ndarray,
) -> None:
    """Emit the single frame, plus the full window once one is available."""
    buffer.push(node_id, spec, timestamp)
    key = str(node_id).encode("utf-8")

    # Keying by node keeps a node's frames on one partition, which is what lets
    # the in-process window buffer stay coherent across a consumer group.
    producer.produce(
        settings.PROCESSED_TOPIC,
        key=key,
        value=msgpack.packb(
            {
                "node_id": node_id,
                "timestamp": timestamp,
                "spectrogram_shape": list(spec.shape),
                "spectrogram": spec.astype(np.float32).tobytes(),
                "window_ready": buffer.is_ready(node_id),
            },
            use_bin_type=True,
        ),
        callback=_delivery_callback,
    )

    if buffer.is_ready(node_id):
        window, timespans = buffer.get_window(node_id)
        producer.produce(
            settings.WINDOWED_TOPIC,
            key=key,
            value=msgpack.packb(
                {
                    "node_id": node_id,
                    "timestamp": timestamp,
                    "window_shape": list(window.shape),
                    "window": window.astype(np.float32).tobytes(),
                    "timespans": timespans.tobytes(),
                },
                use_bin_type=True,
            ),
            callback=_delivery_callback,
        )


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not torch.cuda.is_available():
        log.warning("CUDA unavailable — running on CPU (not recommended for production)")
    process_stream()


if __name__ == "__main__":  # pragma: no cover
    main()
