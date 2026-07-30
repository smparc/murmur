"""
Streaming inference worker — the link between ingestion and serving.

The ingestion service publishes spectrogram windows; the telemetry API consumes
scored embeddings. Nothing previously joined the two, so the topics were
written and never read and the dashboard could never display anything. This
module is that join.

Per window::

    windowed spectrograms
        -> assemble all microphones into one graph snapshot
        -> ST-GNN  (per-timestep embedding sequence)
        -> Liquid Network (TTF forecast over real inter-frame intervals)
        -> AnomalyScorer  (per-node robust z-score)
        -> POST /generate_telemetry

Two design points worth stating:

- **The graph needs every node at once.** A GCN over a single microphone is
  meaningless, so windows are buffered per node and only released when the
  whole array has contributed a recent frame.
- **Detection happens here, not in the API.** The API previously accepted
  ``anomaly_score`` and ``is_anomaly`` from its caller and forwarded them to
  Prometheus, so nothing in the system actually detected anything. Scores are
  computed here from model output.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from types import FrameType

import httpx
import msgpack
import numpy as np
import torch

from src.detection.anomaly_detector import AnomalyScorer, ScoreResult, SpectrogramAutoencoder
from src.forecasting.liquid_network import AcousticForecastingLNN
from src.mapping.st_gnn_model import SpatioTemporalGNN
from src.mapping.topology_graph import build_acoustic_topology
from src.observability.metrics import (
    PIPELINE_ERRORS,
    record_consumer_lag,
    track_inference,
    track_stage,
)
from src.settings import settings

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONSUME_BATCH_SIZE = 16
CONSUME_TIMEOUT_SECONDS = 0.5


@dataclass
class NodeWindow:
    """One microphone's most recent temporal window."""

    node_id: int
    features: np.ndarray  # (seq_len, n_mels)
    timespans: np.ndarray  # (seq_len,)
    timestamp: float
    latest_frame: np.ndarray  # (n_mels, mel_frames) — for anomaly scoring


class WindowAssembler:
    """
    Collects per-node windows into a single graph snapshot.

    Microphones do not report in lockstep: their windows complete at slightly
    different moments and any one of them can drop out. A snapshot is released
    once every node has reported within ``staleness_tolerance`` seconds of the
    newest arrival, so the GCN always sees a coherent instant rather than
    stitching together frames minutes apart.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_length: int,
        n_mels: int,
        staleness_tolerance: float = 5.0,
    ):
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.n_mels = n_mels
        self.staleness_tolerance = staleness_tolerance
        self._windows: dict[int, NodeWindow] = {}

    def push(self, window: NodeWindow) -> None:
        self._windows[window.node_id] = window

    def is_complete(self) -> bool:
        """True when every node has a window and none is stale."""
        if len(self._windows) < self.num_nodes:
            return False
        if not all(i in self._windows for i in range(self.num_nodes)):
            return False
        newest = max(w.timestamp for w in self._windows.values())
        oldest = min(w.timestamp for w in self._windows.values())
        return (newest - oldest) <= self.staleness_tolerance

    def assemble(self) -> tuple[torch.Tensor, torch.Tensor, dict[int, NodeWindow]]:
        """
        Build the ST-GNN input.

        Returns ``(x, timespans, windows)`` where ``x`` is
        ``(1, seq_len, num_nodes * n_mels)`` and ``timespans`` is
        ``(1, seq_len)``.
        """
        ordered = [self._windows[i] for i in range(self.num_nodes)]

        # (num_nodes, seq, mels) -> (seq, num_nodes, mels) -> (seq, nodes*mels)
        stacked = np.stack([w.features for w in ordered], axis=0)
        interleaved = stacked.transpose(1, 0, 2).reshape(self.seq_length, -1)

        x = torch.from_numpy(interleaved).float().unsqueeze(0)
        # Nodes share a clock closely enough that the mean interval is the right
        # integration step for the graph-level sequence.
        timespans = torch.from_numpy(
            np.mean([w.timespans for w in ordered], axis=0)
        ).float().unsqueeze(0)

        snapshot = dict(self._windows)
        return x, timespans, snapshot

    def clear(self) -> None:
        self._windows.clear()


def decode_window(raw: bytes) -> NodeWindow | None:
    """
    Decode a windowed-spectrogram message.

    Each buffered frame is ``(n_mels, mel_frames)``; the ST-GNN wants one
    feature vector per timestep, so the intra-chunk time axis is averaged away.
    Defining that reduction here is what makes the ingestion/model shape
    contract concrete — it was previously left unspecified because nothing
    consumed these messages.
    """
    try:
        payload = msgpack.unpackb(raw, raw=False)
        node_id = int(payload["node_id"])
        timestamp = float(payload["timestamp"])
        shape = tuple(payload["window_shape"])
        window = np.frombuffer(payload["window"], dtype=np.float32).reshape(shape)
        timespans = np.frombuffer(payload["timespans"], dtype=np.float32)
    except Exception:
        PIPELINE_ERRORS.labels(stage="worker_decode").inc()
        log.warning("Undecodable window message", exc_info=True)
        return None

    if window.ndim != 3:
        log.warning("Expected a 3-D window (seq, mels, frames), got %s", (window.shape,))
        return None
    if not np.isfinite(window).all():
        PIPELINE_ERRORS.labels(stage="worker_decode").inc()
        return None

    features = window.mean(axis=2)  # (seq, n_mels)
    return NodeWindow(
        node_id=node_id,
        features=np.ascontiguousarray(features),
        timespans=np.ascontiguousarray(timespans),
        timestamp=timestamp,
        latest_frame=np.ascontiguousarray(window[-1]),
    )


class InferenceWorker:
    """Holds the models and turns graph snapshots into telemetry payloads."""

    def __init__(
        self,
        inference_url: str | None = None,
        http_client: httpx.Client | None = None,
        load_weights: bool = True,
    ):
        self.inference_url = (inference_url or settings.INFERENCE_URL).rstrip("/")
        self.num_nodes = settings.NUM_NODES

        edge_index, edge_weight = build_acoustic_topology(
            settings.MIC_COORDS,
            settings.DISTANCE_THRESHOLD,
            decay_exponent=settings.DISTANCE_DECAY_EXPONENT,
        )
        self.edge_index = edge_index.to(DEVICE)
        self.edge_weight = edge_weight.to(DEVICE)

        self.st_gnn = SpatioTemporalGNN(
            in_channels=settings.N_MELS,
            hidden_channels=settings.GNN_HIDDEN_CHANNELS,
            embedding_dim=settings.GNN_EMBEDDING_DIM,
            num_nodes=self.num_nodes,
            num_heads=settings.GNN_NUM_HEADS,
        ).to(DEVICE)

        self.lnn = AcousticForecastingLNN(
            input_dim=settings.GNN_EMBEDDING_DIM,
            hidden_neurons=settings.LNN_HIDDEN_NEURONS,
        ).to(DEVICE)

        self.autoencoder = SpectrogramAutoencoder(
            n_mels=settings.N_MELS, latent_dim=settings.AE_LATENT_DIM
        ).to(DEVICE)

        self.weights_loaded = self._load_weights() if load_weights else False

        self.st_gnn.eval()
        self.lnn.eval()
        self.autoencoder.eval()

        self.scorer = AnomalyScorer(
            autoencoder=self.autoencoder if self.weights_loaded else None,
            num_nodes=self.num_nodes,
            warmup_frames=settings.ANOMALY_WARMUP_FRAMES,
            z_threshold=settings.ANOMALY_Z_THRESHOLD,
            window=settings.ANOMALY_WINDOW,
            device=DEVICE,
        )

        self.assembler = WindowAssembler(
            num_nodes=self.num_nodes,
            seq_length=settings.SEQ_LENGTH,
            n_mels=settings.N_MELS,
        )

        self._client = http_client or httpx.Client(timeout=30.0)
        self._owns_client = http_client is None
        self._throttled = 0

    def _load_weights(self) -> bool:
        """Load trained weights; fall back to energy-based scoring if absent."""
        paths = {
            "st_gnn": os.path.join(settings.MODEL_DIR, "st_gnn_weights.pth"),
            "lnn": os.path.join(settings.MODEL_DIR, "lnn_weights.pth"),
            "autoencoder": os.path.join(settings.MODEL_DIR, "autoencoder_weights.pth"),
        }
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            log.warning(
                "Missing weights for %s in %s — running with random initialisation. "
                "Anomaly scoring falls back to frame energy; run `murmur-train` first.",
                ", ".join(missing), settings.MODEL_DIR,
            )
            return False

        try:
            for name, module in (
                ("st_gnn", self.st_gnn),
                ("lnn", self.lnn),
                ("autoencoder", self.autoencoder),
            ):
                module.load_state_dict(
                    torch.load(paths[name], map_location=DEVICE, weights_only=True)
                )
            log.info("Loaded trained weights from %s", settings.MODEL_DIR)
            return True
        except Exception:
            log.exception("Failed to load weights — continuing with random initialisation")
            return False

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def infer(
        self, x: torch.Tensor, timespans: torch.Tensor, windows: dict[int, NodeWindow]
    ) -> list[dict]:
        """Run the model chain over one graph snapshot; returns per-node payloads."""
        x = x.to(DEVICE)
        timespans = timespans.to(DEVICE)

        with track_inference("st_gnn"):
            # The sequence form is what gives the Liquid Network genuine
            # temporal structure to integrate.
            embedding_sequence = self.st_gnn(
                x, self.edge_index, self.edge_weight, return_sequence=True
            )

        with track_inference("lnn"):
            ttf = self.lnn(embedding_sequence, timespans=timespans)

        graph_embedding = embedding_sequence.mean(dim=1).squeeze(0).cpu().tolist()
        ttf_value = float(ttf.squeeze().item())

        payloads: list[dict] = []
        for node_id in range(self.num_nodes):
            window = windows.get(node_id)
            if window is None:
                continue

            frame = torch.from_numpy(window.latest_frame).float()
            result: ScoreResult = self.scorer.score(node_id, frame)

            payloads.append(
                {
                    "node_id": node_id,
                    "timestamp": window.timestamp,
                    "gnn_embedding": graph_embedding,
                    "anomaly_score": round(result.normalized_score, 6),
                    "anomaly_severity": result.severity,
                    "ttf_prediction": round(ttf_value, 6),
                    "is_anomaly": result.is_anomaly,
                    "z_score": round(result.z_score, 4),
                }
            )
        return payloads

    def submit(self, payload: dict) -> bool:
        """POST one payload to the telemetry API."""
        headers = {"X-API-Key": settings.API_KEY} if settings.AUTH_ENABLED else {}
        try:
            response = self._client.post(
                f"{self.inference_url}/generate_telemetry",
                json=payload,
                headers=headers,
            )
            if response.status_code == 429:
                # Backpressure, not a defect. Logged sparsely because the
                # limiter rejects a whole snapshot at once and per-node warnings
                # would bury everything else in the log.
                PIPELINE_ERRORS.labels(stage="submit_throttled").inc()
                self._throttled += 1
                if self._throttled % 100 == 1:
                    log.warning(
                        "Telemetry API is rate limiting (%d rejected so far). "
                        "Raise RATE_LIMIT_PER_MINUTE above the pipeline's "
                        "steady-state rate of NUM_NODES per CHUNK_DURATION.",
                        self._throttled,
                    )
                return False
            if response.status_code >= 400:
                PIPELINE_ERRORS.labels(stage="submit").inc()
                log.warning(
                    "Telemetry API returned %s for node %s: %s",
                    response.status_code, payload["node_id"], response.text[:200],
                )
                return False
            return True
        except httpx.HTTPError:
            PIPELINE_ERRORS.labels(stage="submit").inc()
            log.warning("Could not reach telemetry API at %s", self.inference_url, exc_info=True)
            return False

    def handle_window(self, window: NodeWindow) -> list[dict]:
        """Buffer a window and, once the array is complete, infer and submit."""
        self.assembler.push(window)
        if not self.assembler.is_complete():
            return []

        x, timespans, snapshot = self.assembler.assemble()
        self.assembler.clear()

        with track_stage("inference"):
            payloads = self.infer(x, timespans, snapshot)

        for payload in payloads:
            self.submit(payload)
        return payloads

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


# ---------------------------------------------------------------------------
# Kafka loop
# ---------------------------------------------------------------------------


class _Shutdown:
    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        log.info("Received signal %s — draining", signum)
        self.requested = True


def run_worker(max_batches: int | None = None) -> int:
    """Consume windowed spectrograms and emit telemetry. Returns snapshots processed."""
    from confluent_kafka import Consumer, KafkaError

    consumer = Consumer(
        {
            "bootstrap.servers": settings.KAFKA_BROKER,
            "group.id": settings.WORKER_GROUP_ID,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "session.timeout.ms": 30_000,
            "max.poll.interval.ms": 300_000,
        }
    )
    consumer.subscribe([settings.WINDOWED_TOPIC])

    worker = InferenceWorker()
    shutdown = _Shutdown() if max_batches is None else None

    snapshots = 0
    batches = 0
    last_lag_sample = 0.0

    log.info(
        "Inference worker listening on %s -> %s (device=%s, weights=%s)",
        settings.WINDOWED_TOPIC,
        worker.inference_url,
        DEVICE,
        "trained" if worker.weights_loaded else "random",
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

            for msg in messages:
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        log.warning("Consumer error: %s", msg.error())
                    continue
                window = decode_window(msg.value())
                if window is None:
                    continue
                try:
                    if worker.handle_window(window):
                        snapshots += 1
                except Exception:
                    PIPELINE_ERRORS.labels(stage="worker").inc()
                    log.exception("Failed to process window from node %s", window.node_id)

            try:
                consumer.commit(asynchronous=True)
            except Exception:
                log.warning("Offset commit failed", exc_info=True)

            now = time.monotonic()
            if now - last_lag_sample > 10.0:
                record_consumer_lag(consumer, [settings.WINDOWED_TOPIC])
                last_lag_sample = now

    except KeyboardInterrupt:  # pragma: no cover
        log.info("Interrupted — stopping worker")
    finally:
        consumer.close()
        worker.close()

    return snapshots


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_worker()


if __name__ == "__main__":  # pragma: no cover
    main()
