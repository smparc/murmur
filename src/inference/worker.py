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
        -> Liquid Network (per-node degradation score over real inter-frame intervals)
        -> AnomalyScorer  (per-node robust z-score)
        -> POST /generate_telemetry

Design points worth stating:

- **The graph needs every node at once.** A GCN over a single microphone is
  meaningless, so windows are buffered per node and only released when the
  whole array has contributed a recent frame. That makes an array the unit of
  consumption: windows are keyed by ``ARRAY_ID``, so every window of one array
  lands on one partition and therefore one worker. Keyed by microphone, a
  second worker would split the array and neither could ever complete a
  snapshot.
- **Detection happens here, not in the API.** The API previously accepted
  ``anomaly_score`` and ``is_anomaly`` from its caller and forwarded them to
  Prometheus, so nothing in the system actually detected anything. Scores are
  computed here from model output.
- **Telemetry delivery is at-least-once to a record, best-effort to the feed.**
  The API is the live dashboard feed; alerting does not route through it. A
  failed submission is retried briefly for transient errors, then written to a
  dead-letter topic with the reason. Offsets advance only once those writes are
  acknowledged, so a scored result is never silently lost — but it is also not
  replayed into the live feed later, where it would be shown as current.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

import httpx
import msgpack
import numpy as np
import torch

from src.alerting.webhook import (
    Alert,
    AlertRouter,
    AlertSink,
    GenericWebhookSink,
    PagerDutySink,
    SlackSink,
)
from src.detection.anomaly_detector import AnomalyScorer, ScoreResult, SpectrogramAutoencoder
from src.explain.saliency import explain_anomaly
from src.forecasting.conformal import ConformalCalibrator, severity_bucket
from src.forecasting.liquid_network import AcousticForecastingLNN, per_node_forecast
from src.mapping.st_gnn_model import SpatioTemporalGNN
from src.mapping.tdoa import TDOAEstimate, tdoa_edge_weights
from src.mapping.topology_graph import build_acoustic_topology
from src.observability.metrics import (
    ARRAY_NODES_REPORTING,
    NODE_DROPPED,
    PIPELINE_ERRORS,
    SNAPSHOTS_EMITTED,
    TELEMETRY_DEAD_LETTERED,
    TELEMETRY_DROPPED,
    record_consumer_lag,
    track_inference,
    track_stage,
)
from src.settings import settings
from src.translation.taxonomy import FaultTaxonomy

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONSUME_BATCH_SIZE = 16
CONSUME_TIMEOUT_SECONDS = 0.5
DEAD_LETTER_FLUSH_TIMEOUT_SECONDS = 10.0

# Failure reasons from InferenceWorker.deliver. Only these are worth retrying
# within a snapshot: the API is momentarily unreachable or erroring. A 429 is the
# limiter's 60-second window, which a sub-second retry cannot outwait, and any
# other 4xx is a payload the API will reject however many times it is sent.
_RETRYABLE = frozenset({"unreachable", "server_error"})

DeadLetterSink = Callable[[dict, str], None]


def _build_alert_router() -> AlertRouter:
    """
    Assemble the alert router from configuration.

    Every sink is opt-in and all are unset by default, so an unconfigured
    deployment gets a router with no sinks — which is a no-op, not an error. A
    monitoring system paging somebody because a URL happened to be lying around
    in the environment is worse than one that stays quiet until asked.
    """
    sinks: list[AlertSink] = []
    if settings.SLACK_WEBHOOK_URL:
        sinks.append(SlackSink(settings.SLACK_WEBHOOK_URL))
    if settings.PAGERDUTY_ROUTING_KEY:
        sinks.append(PagerDutySink(settings.PAGERDUTY_ROUTING_KEY))
    if settings.ALERT_WEBHOOK_URL:
        sinks.append(GenericWebhookSink(settings.ALERT_WEBHOOK_URL))

    if sinks:
        log.info("Alert routing enabled: %s", ", ".join(s.name for s in sinks))
    return AlertRouter(
        sinks=sinks,
        cooldown_s=settings.ALERT_COOLDOWN_SECONDS,
        min_severity=settings.ALERT_MIN_SEVERITY,
    )


@dataclass
class NodeWindow:
    """One microphone's most recent temporal window."""

    node_id: int
    features: np.ndarray  # (seq_len, n_mels)
    timespans: np.ndarray  # (seq_len,)
    timestamp: float
    latest_frame: np.ndarray  # (n_mels, mel_frames) — for anomaly scoring
    array_id: str = ""


class WindowAssembler:
    """
    Collects per-node windows into a single graph snapshot.

    Microphones do not report in lockstep: their windows complete at slightly
    different moments and any one of them can drop out. A snapshot is released
    once every node has reported within ``staleness_tolerance`` seconds of the
    newest arrival, so the GCN always sees a coherent instant rather than
    stitching together frames minutes apart.

    Degraded release
    ----------------
    Requiring the full array unconditionally means a single dead microphone
    stalls the assembler forever: the remaining nodes stream indefinitely and
    nothing is ever emitted. On a system whose entire purpose is not missing
    events, going silent is the worst available failure mode — it is visually
    identical to a healthy, quiet plant.

    So after ``max_wait`` seconds the snapshot is released without the absent
    microphones, provided at least ``min_nodes`` are still reporting. Absent
    nodes are zero-filled in the tensor (the graph keeps a fixed shape so the
    topology stays aligned) and omitted from the returned snapshot, so no
    telemetry is fabricated for a microphone that said nothing.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_length: int,
        n_mels: int,
        staleness_tolerance: float | None = None,
        max_wait: float | None = None,
        min_nodes: int | None = None,
    ):
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.n_mels = n_mels
        self.staleness_tolerance = (
            settings.WINDOW_STALENESS_TOLERANCE
            if staleness_tolerance is None
            else staleness_tolerance
        )
        self.max_wait = settings.ARRAY_MAX_WAIT if max_wait is None else max_wait
        self.min_nodes = settings.ARRAY_MIN_NODES if min_nodes is None else min_nodes
        self._windows: dict[int, NodeWindow] = {}
        self._cycle_started: float | None = None
        # Reported once per node rather than per frame: a permanently misconfigured
        # SEQ_LENGTH would otherwise emit an error for every window forever.
        self._bad_shape_logged: set[int] = set()
        self.last_missing: set[int] = set()

    def push(self, window: NodeWindow) -> bool:
        """
        Buffer one node's window. Returns False if it was rejected.

        A window whose sequence length disagrees with ``seq_length`` cannot be
        reshaped into the graph tensor. Rejecting it here, loudly and once,
        beats letting ``assemble`` raise on every subsequent snapshot — that
        turns a fixable configuration error into an unbounded error log.
        """
        # Both arrays are checked, not just the features. `assemble` averages
        # the timespans across reporting nodes, and numpy raises on a ragged
        # stack — so a node whose timespans length disagrees would break every
        # subsequent snapshot with a traceback from deep inside assemble,
        # instead of the one clear diagnostic the features path already gives.
        for field, actual in (
            ("features", window.features.shape[0]),
            ("timespans", window.timespans.shape[0]),
        ):
            if actual == self.seq_length:
                continue
            if window.node_id not in self._bad_shape_logged:
                self._bad_shape_logged.add(window.node_id)
                log.error(
                    "Node %s sent %s of length %d but SEQ_LENGTH is %d. The ingestion "
                    "service and the worker disagree on window length; align SEQ_LENGTH "
                    "across both. Further windows from this node are dropped silently.",
                    window.node_id,
                    field,
                    actual,
                    self.seq_length,
                )
            PIPELINE_ERRORS.labels(stage="window_shape").inc()
            return False

        if not self._windows:
            self._cycle_started = time.monotonic()
        self._windows[window.node_id] = window
        return True

    def _fresh_nodes(self) -> set[int]:
        """Nodes whose window falls inside the staleness bound of the newest."""
        if not self._windows:
            return set()
        newest = max(w.timestamp for w in self._windows.values())
        return {
            node_id
            for node_id, w in self._windows.items()
            if newest - w.timestamp <= self.staleness_tolerance
        }

    def _evict_stale(self) -> None:
        """
        Drop windows that fell outside the staleness bound.

        Without this a node that stops reporting keeps its last window in the
        buffer forever, and the spread between newest and oldest never returns
        below tolerance — so even the surviving microphones stop producing.
        """
        fresh = self._fresh_nodes()
        for node_id in [n for n in self._windows if n not in fresh]:
            del self._windows[node_id]
            NODE_DROPPED.labels(node_id=str(node_id)).inc()
            log.warning(
                "Node %s exceeded the %.1fs staleness bound and was evicted from the "
                "current snapshot; the array is now running degraded.",
                node_id,
                self.staleness_tolerance,
            )

    def is_complete(self, now: float | None = None) -> bool:
        """
        True when the snapshot is ready — either whole, or degraded past
        ``max_wait`` with a quorum still reporting.
        """
        self._evict_stale()
        if not self._windows:
            return False

        if len(self._windows) == self.num_nodes:
            return True

        now = time.monotonic() if now is None else now
        waited = now - (self._cycle_started or now)
        return waited >= self.max_wait and len(self._windows) >= self.min_nodes

    def assemble(self) -> tuple[torch.Tensor, torch.Tensor, dict[int, NodeWindow]]:
        """
        Build the ST-GNN input.

        Returns ``(x, timespans, windows)`` where ``x`` is
        ``(1, seq_len, num_nodes * n_mels)`` and ``timespans`` is
        ``(1, seq_len)``. ``windows`` contains only the nodes that actually
        reported; absent nodes are zero-filled in ``x`` so the tensor keeps the
        fixed shape the topology is indexed against.
        """
        present = [self._windows.get(i) for i in range(self.num_nodes)]
        self.last_missing = {i for i, w in enumerate(present) if w is None}

        silence = np.zeros((self.seq_length, self.n_mels), dtype=np.float32)
        stacked = np.stack([silence if w is None else w.features for w in present], axis=0)
        # (num_nodes, seq, mels) -> (seq, num_nodes, mels) -> (seq, nodes*mels)
        interleaved = stacked.transpose(1, 0, 2).reshape(self.seq_length, -1)

        x = torch.from_numpy(np.ascontiguousarray(interleaved)).float().unsqueeze(0)

        # Nodes share a clock closely enough that the mean interval is the right
        # integration step for the graph-level sequence. Absent nodes contribute
        # no interval — averaging in zeros would compress the LNN's time axis and
        # silently shorten every forecast horizon.
        reporting = [w for w in present if w is not None]
        timespans = (
            torch.from_numpy(np.mean([w.timespans for w in reporting], axis=0)).float().unsqueeze(0)
        )

        snapshot = dict(self._windows)
        return x, timespans, snapshot

    def clear(self) -> None:
        self._windows.clear()
        self._cycle_started = None


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
        # Absent on windows produced before array keying; those can only have
        # come from this deployment's own array.
        array_id = str(payload.get("array_id") or settings.ARRAY_ID)
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
        array_id=array_id,
    )


class InferenceWorker:
    """Holds the models and turns graph snapshots into telemetry payloads."""

    def __init__(
        self,
        inference_url: str | None = None,
        http_client: httpx.Client | None = None,
        load_weights: bool = True,
        dead_letter: DeadLetterSink | None = None,
    ):
        self.inference_url = (inference_url or settings.INFERENCE_URL).rstrip("/")
        self.num_nodes = settings.NUM_NODES
        self.array_id = settings.ARRAY_ID
        # Where payloads the API would not accept are kept. None drops them
        # (counted), which is only appropriate in tests and local development.
        self.dead_letter = dead_letter

        edge_index, edge_weight = build_acoustic_topology(
            settings.MIC_COORDS,
            settings.DISTANCE_THRESHOLD,
            decay_exponent=settings.DISTANCE_DECAY_EXPONENT,
        )
        self.edge_index = edge_index.to(DEVICE)
        self.edge_weight = edge_weight.to(DEVICE)

        # The geometric weights above describe the building. TDOA coherence,
        # applied below, describes what the building currently sounds like; the
        # effective weights are the product. Until a spatial snapshot arrives
        # the graph falls back to pure geometry, which is the old behaviour.
        self._static_edge_index = edge_index.cpu().numpy()
        self._static_edge_weight = edge_weight.cpu().numpy()
        self.effective_edge_weight = self.edge_weight
        self.spatial: dict | None = None

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
            drift_guard=settings.ANOMALY_DRIFT_GUARD,
            drift_window=settings.ANOMALY_DRIFT_WINDOW,
        )

        self.assembler = WindowAssembler(
            num_nodes=self.num_nodes,
            seq_length=settings.SEQ_LENGTH,
            n_mels=settings.N_MELS,
        )

        self.calibrator = self._load_calibrator()

        # Spectral attribution and catalogue matching. Both are cheap relative to
        # the model chain but not free, so they run only for frames that actually
        # flagged — see infer().
        self.taxonomy = FaultTaxonomy()
        self.alerts = _build_alert_router()

        # Bounded well below max.poll.interval.ms even with retries: a hung API
        # must not stall the consumer long enough to be evicted from the group.
        self._client = http_client or httpx.Client(timeout=settings.TELEMETRY_TIMEOUT)
        self._owns_client = http_client is None
        self._throttled = 0

    def _load_calibrator(self) -> ConformalCalibrator | None:
        """
        Load conformal calibration, if the training run produced any.

        Absent calibration means forecasts ship as bare point estimates. That is
        a downgrade, not a failure — but it is logged, because an uncalibrated
        sigmoid read as a failure probability is exactly the kind of number
        that gets acted on and should not be.
        """
        path = os.path.join(settings.MODEL_DIR, "conformal.json")
        if not os.path.exists(path):
            log.warning(
                "No conformal calibration at %s — degradation scores will carry no "
                "uncertainty bounds. Run `murmur-train` to produce one.",
                path,
            )
            return None
        try:
            calibrator = ConformalCalibrator.load(path)
            log.info(
                "Loaded conformal calibration (alpha=%.3f, n=%d, groups=%s)",
                calibrator.alpha,
                calibrator.n_calibration,
                sorted(calibrator.group_radii) or "none",
            )
            return calibrator
        except Exception:
            log.exception("Could not load conformal calibration from %s", path)
            return None

    # -- spatial acoustics --------------------------------------------------

    def apply_spatial(self, payload: dict) -> None:
        """
        Adopt a TDOA snapshot, reweighting the graph by measured coherence.

        Pairs that have fallen out of correlation stop propagating, so the graph
        effectively re-partitions itself around whatever is actually making
        noise instead of around the floorplan.
        """
        pairs = payload.get("pairs") or []
        estimates = [
            TDOAEstimate(
                i=int(p["i"]),
                j=int(p["j"]),
                tau=float(p["tau"]),
                coherence=float(p["coherence"]),
                max_tau=float("inf"),
            )
            for p in pairs
        ]

        weights = tdoa_edge_weights(
            self._static_edge_index,
            estimates,
            self._static_edge_weight,
            gamma=settings.TDOA_EDGE_GAMMA,
            floor=settings.TDOA_EDGE_FLOOR,
        )
        self.effective_edge_weight = torch.from_numpy(weights).float().to(DEVICE)
        self.spatial = payload

    # -- operator control ---------------------------------------------------

    def apply_control(self, command: dict) -> bool:
        """
        Execute an operator command from the control topic.

        The only command is ``rebaseline``: accept a node's current sound as its
        new normal. Persistent departures are never absorbed automatically (see
        ``src/detection/anomaly_detector.py``), so this is how an engineer who
        has confirmed a legitimate change clears one. Logged at WARNING with who
        asked and why, because it silences an active detection.
        """
        if command.get("command") != "rebaseline":
            log.warning("Ignoring unknown control command %r", command.get("command"))
            return False
        if str(command.get("array_id", self.array_id)) != self.array_id:
            return False

        node_id = command.get("node_id")
        targets = range(self.num_nodes) if node_id is None else [int(node_id)]
        for nid in targets:
            adopted = self.scorer.rebaseline(nid)
            log.warning(
                "Node %s re-baselined by %s (%s): %s",
                nid,
                command.get("requested_by", "unknown"),
                command.get("reason", "no reason given"),
                "adopted recent audio as the new normal"
                if adopted
                else "too little recent history, restarting warmup",
            )
        return True

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
                ", ".join(missing),
                settings.MODEL_DIR,
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
            # temporal structure to integrate. The per-node form is what lets a
            # forecast be attributed to one microphone.
            _graph_sequence, node_sequence = self.st_gnn(
                x,
                self.edge_index,
                self.effective_edge_weight,
                return_sequence=True,
                return_nodes=True,
            )

        with track_inference("lnn"):
            # One forecast per microphone, from that microphone's own embedding
            # trajectory. Previously a single facility-level TTF was computed
            # from the pooled graph readout and copied into every node's
            # payload, so the dashboard's per-node cards were identical by
            # construction — an operator reading them as four independent
            # machine forecasts was reading the same number four times.
            #
            # Training and conformal calibration go through this same
            # `per_node_forecast`, so the LNN weights and the interval radii
            # were fitted on exactly the quantity served here.
            node_degradation = per_node_forecast(self.lnn, node_sequence, timespans)

        node_embeddings = node_sequence.mean(dim=1).squeeze(0).cpu()  # (N, E)
        degradation_by_node = node_degradation.squeeze(0).cpu().tolist()

        # A position is attached only when the array geometry supports it.
        #
        # `source_position` becomes the alert's `location` — the coordinates an
        # engineer walks to. Near the array centroid the hyperbolas are nearly
        # parallel, so the solve is ill-conditioned and the fix can be over a
        # metre out while reporting a residual of ~0; the residual cannot detect
        # this because the system is exactly determined and fits perfectly
        # either way. GDOP can, so an unreliable fix is dropped from the payload
        # rather than sent to the shop floor as a location.
        source_position = None
        if self.spatial is not None and self.spatial.get("position_reliable", True):
            source_position = self.spatial.get("position")

        payloads: list[dict] = []
        for node_id in range(self.num_nodes):
            window = windows.get(node_id)
            if window is None:
                continue

            frame = torch.from_numpy(window.latest_frame).float()
            result: ScoreResult = self.scorer.score(node_id, frame)
            degradation = float(degradation_by_node[node_id])

            # A calibrated band around the point estimate. Grouped by severity so
            # coverage holds within the high-risk stratum rather than only on
            # average across a population dominated by healthy machines.
            interval = None
            if self.calibrator is not None:
                interval = self.calibrator.interval(
                    degradation, severity_bucket(degradation)
                ).as_dict()

            payload = {
                "node_id": node_id,
                "timestamp": window.timestamp,
                "gnn_embedding": node_embeddings[node_id].tolist(),
                "anomaly_score": round(result.normalized_score, 6),
                "anomaly_severity": result.severity,
                "degradation_score": round(degradation, 6),
                "is_anomaly": result.is_anomaly,
                "z_score": round(result.z_score, 4),
                "drift_z": round(result.drift_z, 4),
            }
            if interval is not None:
                payload["degradation_interval"] = interval
            if source_position is not None:
                payload["source_position"] = source_position

            explanation = self._explain(frame, result)
            if explanation is not None:
                payload["explanation"] = explanation.as_dict()
                payload["diagnosis"] = self.taxonomy.best(explanation).as_dict()

            payloads.append(payload)
        return payloads

    def _explain(self, frame: torch.Tensor, result: ScoreResult):
        """
        Attribute a flagged frame's score across frequency bands.

        Only for frames that actually flagged, and only when a trained
        autoencoder is resident. The attribution is an exact decomposition of the
        reconstruction error, so without that autoencoder there is no error map
        to decompose — the scorer is falling back to frame energy and any
        "explanation" would be invented.

        Restricting this to anomalies keeps a per-node autoencoder forward off
        the steady-state path, where the overwhelming majority of frames are
        normal and nobody reads the attribution.
        """
        if not (result.is_anomaly and self.weights_loaded):
            return None
        try:
            return explain_anomaly(self.autoencoder, frame.to(DEVICE), settings.SAMPLE_RATE)
        except Exception:
            # An alert with no attribution is worth strictly more than no alert.
            PIPELINE_ERRORS.labels(stage="explain").inc()
            log.warning("Could not attribute anomaly score", exc_info=True)
            return None

    def submit(self, payload: dict, retry: bool = True) -> bool:
        """POST one payload to the telemetry API. True when it was accepted."""
        return self.deliver(payload, retry=retry) is None

    def deliver(self, payload: dict, retry: bool = True) -> str | None:
        """
        POST one payload, retrying transient failures.

        Returns ``None`` on success, otherwise the reason it was not accepted
        (``unreachable``, ``server_error``, ``throttled`` or ``rejected``). Only
        ``unreachable`` and ``server_error`` are retried, with exponential
        backoff, up to ``TELEMETRY_MAX_RETRIES`` times.
        """
        attempts = 1 + (settings.TELEMETRY_MAX_RETRIES if retry else 0)
        reason: str | None = None
        for attempt in range(attempts):
            reason = self._post(payload)
            if reason is None or reason not in _RETRYABLE:
                return reason
            if attempt + 1 < attempts:
                time.sleep(settings.TELEMETRY_RETRY_BACKOFF * 2**attempt)
        return reason

    def _post(self, payload: dict) -> str | None:
        """One attempt. ``None`` on success, else a failure reason."""
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
                return "throttled"
            if response.status_code >= 400:
                PIPELINE_ERRORS.labels(stage="submit").inc()
                log.warning(
                    "Telemetry API returned %s for node %s: %s",
                    response.status_code,
                    payload.get("node_id"),
                    response.text[:200],
                )
                return "server_error" if response.status_code >= 500 else "rejected"
            return None
        except httpx.HTTPError:
            PIPELINE_ERRORS.labels(stage="submit").inc()
            log.warning("Could not reach telemetry API at %s", self.inference_url, exc_info=True)
            return "unreachable"

    def _dead_letter(self, payload: dict, reason: str) -> None:
        """Keep a payload the API did not accept, or count it as lost."""
        node = str(payload["node_id"])
        if self.dead_letter is None:
            TELEMETRY_DROPPED.labels(node_id=node).inc()
            return
        try:
            self.dead_letter(payload, reason)
            TELEMETRY_DEAD_LETTERED.labels(reason=reason).inc()
        except Exception:
            TELEMETRY_DROPPED.labels(node_id=node).inc()
            PIPELINE_ERRORS.labels(stage="dead_letter").inc()
            log.warning("Could not dead-letter payload for node %s", node, exc_info=True)

    def handle_window(self, window: NodeWindow) -> list[dict]:
        """Buffer a window and, once the array is ready, infer and submit."""
        if not self.assembler.push(window):
            return []
        if not self.assembler.is_complete():
            return []

        x, timespans, snapshot = self.assembler.assemble()
        missing = set(self.assembler.last_missing)
        self.assembler.clear()

        ARRAY_NODES_REPORTING.set(len(snapshot))
        SNAPSHOTS_EMITTED.labels(mode="degraded" if missing else "complete").inc()
        if missing:
            log.warning(
                "Emitting a degraded snapshot: %d/%d microphones reporting (missing %s). "
                "Spatial inference is running over a graph with holes in it.",
                len(snapshot),
                self.assembler.num_nodes,
                sorted(missing),
            )

        with track_stage("inference"):
            payloads = self.infer(x, timespans, snapshot)

        # Once one payload has exhausted its retries against an API that is down,
        # the rest of this snapshot goes straight to the dead-letter topic rather
        # than paying the same retries again — an outage must not slow the
        # consumer to a crawl. The next snapshot tries afresh.
        api_down = False
        for payload in payloads:
            reason = self.deliver(payload, retry=not api_down)
            if reason is not None:
                api_down = api_down or reason in _RETRYABLE
                self._dead_letter(payload, reason)
            self._raise_alert(payload)
        return payloads

    def _raise_alert(self, payload: dict) -> None:
        """
        Route a scored frame to the configured alert sinks.

        Failure here must never propagate: a wedged webhook is not a reason to
        stop scoring the plant, and the telemetry has already been submitted by
        the time this runs.
        """
        if not self.alerts.sinks:
            return

        diagnosis = payload.get("diagnosis") or {}
        alert = Alert(
            node_id=payload["node_id"],
            severity=payload["anomaly_severity"],
            fault=diagnosis.get("fault", "Unrecognised acoustic anomaly"),
            confidence=float(diagnosis.get("confidence", 0.0)),
            anomaly_score=payload["anomaly_score"],
            degradation_score=payload["degradation_score"],
            evidence=tuple(diagnosis.get("evidence", ())),
            recommended_action=diagnosis.get("recommended_action", ""),
            location=tuple(payload["source_position"]) if payload.get("source_position") else None,
            timestamp=payload["timestamp"],
            resolved=not payload["is_anomaly"],
        )
        try:
            self.alerts.send(alert)
        except Exception:
            PIPELINE_ERRORS.labels(stage="alerting").inc()
            log.warning("Alert delivery failed for node %s", payload["node_id"], exc_info=True)

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


class _DeadLetterWriter:
    """
    Produces undelivered telemetry to the dead-letter topic.

    The worker only commits a batch's offsets once :meth:`settle` confirms every
    dead-letter write from that batch was acknowledged — otherwise a crash, or a
    broker that refused the write, would lose exactly the payloads the topic
    exists to keep.
    """

    def __init__(self, producer) -> None:
        self._producer = producer
        self._failed = 0

    def _on_delivery(self, err, _msg) -> None:
        if err is not None:
            self._failed += 1
            PIPELINE_ERRORS.labels(stage="dead_letter").inc()
            log.error("Dead-letter write failed: %s", err)

    def __call__(self, payload: dict, reason: str) -> None:
        record = {
            "reason": reason,
            "array_id": settings.ARRAY_ID,
            "dead_lettered_at": time.time(),
            "payload": payload,
        }
        self._producer.produce(
            settings.TELEMETRY_DLQ_TOPIC,
            key=str(payload["node_id"]).encode("utf-8"),
            value=json.dumps(record).encode("utf-8"),
            on_delivery=self._on_delivery,
        )
        self._producer.poll(0)

    def settle(self, timeout: float = DEAD_LETTER_FLUSH_TIMEOUT_SECONDS) -> bool:
        """Block until queued writes resolve. True only if all were acknowledged."""
        remaining = self._producer.flush(timeout)
        ok = remaining == 0 and self._failed == 0
        self._failed = 0
        return ok


def _rewind(consumer, first_offsets: dict[tuple[str, int], int]) -> None:
    """
    Seek back to the start of a batch so it is consumed again.

    Merely skipping a commit does not redeliver anything: the consumer's
    position has already moved past the batch, and the *next* successful commit
    acknowledges it along with everything after.
    """
    from confluent_kafka import TopicPartition

    for (topic, partition), offset in first_offsets.items():
        try:
            consumer.seek(TopicPartition(topic, partition, offset))
        except Exception:
            log.warning("Could not rewind %s[%d] to %d", topic, partition, offset, exc_info=True)


def run_worker(max_batches: int | None = None) -> int:
    """Consume windowed spectrograms and emit telemetry. Returns snapshots processed."""
    from confluent_kafka import Consumer, KafkaError, Producer

    consumer = Consumer(
        {
            "bootstrap.servers": settings.KAFKA_BROKER,
            "group.id": settings.WORKER_GROUP_ID,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "session.timeout.ms": 30_000,
            "max.poll.interval.ms": 300_000,
            # The control topic is written only when an operator re-baselines,
            # which may be never. Without this the subscription reports an
            # unknown topic on every metadata refresh until someone does.
            "allow.auto.create.topics": True,
        }
    )
    topics = [settings.WINDOWED_TOPIC, settings.CONTROL_TOPIC]
    if settings.TDOA_ENABLED:
        topics.append(settings.SPATIAL_TOPIC)
    consumer.subscribe(topics)

    dead_letter = _DeadLetterWriter(
        Producer(
            {
                "bootstrap.servers": settings.KAFKA_BROKER,
                "enable.idempotence": True,
                "acks": "all",
                "compression.type": settings.KAFKA_COMPRESSION,
            }
        )
    )
    worker = InferenceWorker(dead_letter=dead_letter)
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

            first_offsets: dict[tuple[str, int], int] = {}
            for msg in messages:
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        log.warning("Consumer error: %s", msg.error())
                    continue
                first_offsets.setdefault((msg.topic(), msg.partition()), msg.offset())

                if msg.topic() == settings.CONTROL_TOPIC:
                    try:
                        worker.apply_control(json.loads(msg.value()))
                    except Exception:
                        PIPELINE_ERRORS.labels(stage="control").inc()
                        log.warning("Could not apply control command", exc_info=True)
                    continue

                if msg.topic() == settings.SPATIAL_TOPIC:
                    # Enrichment only: a bad spatial frame must never stop
                    # telemetry, so the graph simply keeps its previous weights.
                    try:
                        spatial = msgpack.unpackb(msg.value(), raw=False)
                        if spatial.get("array_id", worker.array_id) == worker.array_id:
                            worker.apply_spatial(spatial)
                    except Exception:
                        PIPELINE_ERRORS.labels(stage="spatial_apply").inc()
                        log.warning("Could not apply spatial snapshot", exc_info=True)
                    continue

                window = decode_window(msg.value())
                if window is None:
                    continue
                if window.array_id != worker.array_id:
                    # Another array's window on our topic. Mixing it in would
                    # corrupt the snapshot; it belongs to that array's worker.
                    PIPELINE_ERRORS.labels(stage="foreign_array").inc()
                    continue
                try:
                    if worker.handle_window(window):
                        snapshots += 1
                except Exception:
                    PIPELINE_ERRORS.labels(stage="worker").inc()
                    log.exception("Failed to process window from node %s", window.node_id)

            if not dead_letter.settle():
                # A dead-letter write was not acknowledged. Committing would
                # acknowledge a payload that now exists nowhere, so the batch is
                # consumed again instead; the API may see duplicates, which is
                # the at-least-once trade.
                log.warning(
                    "Dead-letter writes unconfirmed — rewinding %d partition(s) for redelivery",
                    len(first_offsets),
                )
                _rewind(consumer, first_offsets)
                continue

            try:
                consumer.commit(asynchronous=True)
            except Exception:
                log.warning("Offset commit failed", exc_info=True)

            now = time.monotonic()
            if now - last_lag_sample > 10.0:
                record_consumer_lag(consumer, topics)
                last_lag_sample = now

    except KeyboardInterrupt:  # pragma: no cover
        log.info("Interrupted — stopping worker")
    finally:
        dead_letter.settle()
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
