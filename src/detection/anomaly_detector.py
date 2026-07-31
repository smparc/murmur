"""
Unsupervised acoustic anomaly detection.

Two cooperating pieces:

``SpectrogramAutoencoder``
    A small convolutional autoencoder trained only on *normal* machine noise.
    Reconstruction error becomes the anomaly signal: sounds the model has never
    been taught to compress reconstruct badly.

``AnomalyScorer``
    Turns that raw error into a decision. Raw reconstruction error is not
    comparable across microphones — a mic above a compressor sits at a very
    different noise floor than one in a corridor — so every node keeps its own
    rolling baseline and is judged against *itself*.

The scorer uses a median/MAD robust z-score rather than mean/std. That matters
here: a developing fault contaminates the very statistics used to detect it, and
the mean is far more easily dragged along by the anomaly than the median is.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Severity = Literal["normal", "warning", "critical"]

# 0.6745 is the 0.75 quantile of the standard normal; scaling MAD by it makes
# the robust z-score directly comparable to a conventional standard-deviation z.
_MAD_TO_SIGMA = 0.6745

# Reported when the rolling baseline has no spread at all. Comfortably above
# twice any sane ANOMALY_Z_THRESHOLD, so it classifies as critical, while
# staying a number a human can read on a dashboard.
_DEGENERATE_Z = 10.0


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class SpectrogramAutoencoder(nn.Module):
    """
    Convolutional autoencoder over mel-spectrogram patches.

    Input is ``(batch, 1, n_mels, time)``. The time axis is deliberately not
    fixed: chunk length varies with hop size and Kafka framing, so the encoder
    pools adaptively and the decoder resamples to whatever the caller asks for.

    GroupNorm is used throughout rather than BatchNorm because the streaming
    worker scores a single frame at a time, and BatchNorm's per-batch statistics
    are meaningless (and, in training mode, an outright error) at ``batch=1``.
    """

    def __init__(self, n_mels: int = 64, latent_dim: int = 32, base_channels: int = 16):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, c1), c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, c2), c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, c3), c3),
            nn.GELU(),
        )
        # Collapses any input resolution to a fixed 4x4 grid so `latent_dim` is
        # independent of the incoming time-axis length.
        self.encoder_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.to_latent = nn.Linear(c3 * 4 * 4, latent_dim)

        self.from_latent = nn.Linear(latent_dim, c3 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, c2), c2),
            nn.GELU(),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, c1), c1),
            nn.GELU(),
            nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
        )
        self._decoder_channels = c3

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, n_mels, T)`` -> ``(B, latent_dim)``."""
        h = self.encoder_conv(x)
        h = self.encoder_pool(h)
        return self.to_latent(h.flatten(1))

    def decode(self, z: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """``(B, latent_dim)`` -> ``(B, 1, *target_size)``."""
        h = self.from_latent(z)
        h = h.view(z.size(0), self._decoder_channels, 4, 4)
        h = self.decoder_conv(h)
        # The transposed convs land near the right resolution; interpolation
        # guarantees an exact match for arbitrary time-axis lengths.
        if h.shape[-2:] != torch.Size(target_size):
            h = F.interpolate(h, size=target_size, mode="bilinear", align_corners=False)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(reconstruction, latent)``."""
        z = self.encode(x)
        recon = self.decode(z, target_size=(x.size(-2), x.size(-1)))
        return recon, z

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction MSE. Shape ``(B,)``, always non-negative."""
        recon, _ = self.forward(x)
        return (recon - x).pow(2).flatten(1).mean(dim=1)


# ---------------------------------------------------------------------------
# Online scorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring one frame against a node's own history."""

    node_id: int
    raw_score: float
    z_score: float
    is_anomaly: bool
    is_warmup: bool
    severity: Severity
    baseline_median: float

    @property
    def normalized_score(self) -> float:
        """
        Robust z mapped into ``[0, 1]`` for transport over the API and for
        display. Saturating rather than clipping keeps large excursions ordered
        instead of flattening everything severe to exactly 1.0.
        """
        if self.z_score <= 0.0:
            return 0.0
        return float(1.0 - math.exp(-self.z_score / 6.0))


class _NodeState:
    """Rolling baseline for a single microphone."""

    __slots__ = ("anomaly_frames", "scores", "total_frames")

    def __init__(self, window: int):
        self.scores: deque[float] = deque(maxlen=window)
        self.total_frames: int = 0
        self.anomaly_frames: int = 0


class AnomalyScorer:
    """
    Per-node adaptive thresholding over a stream of spectrogram frames.

    Each node holds a bounded window of recent scores. The window is what makes
    this *adaptive*: as a machine's normal operating point drifts (seasonal
    temperature, load changes, a bearing bedding in), the baseline follows it,
    so the detector reports genuine departures rather than slow legitimate
    change.

    Parameters
    ----------
    autoencoder:
        Trained ``SpectrogramAutoencoder``. If ``None``, the scorer falls back
        to mean-square frame energy, which still catches gross events and lets
        the pipeline run before any model has been trained.
    num_nodes:
        Nodes to pre-register, so summaries report a complete picture even for
        microphones that have not yet produced traffic.
    warmup_frames:
        Frames observed per node before that node may raise an anomaly. Without
        this, the first few frames define a baseline of one sample and
        everything afterwards looks anomalous.
    z_threshold:
        Robust-z above which a frame is flagged. Twice this is "critical".
    window:
        Length of the rolling baseline, in frames.
    """

    def __init__(
        self,
        autoencoder: SpectrogramAutoencoder | None = None,
        num_nodes: int = 4,
        warmup_frames: int = 50,
        z_threshold: float = 3.0,
        window: int = 500,
        device: torch.device | str = "cpu",
    ):
        if warmup_frames < 1:
            raise ValueError("warmup_frames must be >= 1")
        if window < warmup_frames:
            raise ValueError("window must be >= warmup_frames")

        self.autoencoder = autoencoder
        self.warmup_frames = warmup_frames
        self.z_threshold = z_threshold
        self.window = window
        self.device = torch.device(device)

        if self.autoencoder is not None:
            self.autoencoder.to(self.device).eval()

        self._nodes: dict[int, _NodeState] = {i: _NodeState(window) for i in range(num_nodes)}

    # -- internals ----------------------------------------------------------

    def _state(self, node_id: int) -> _NodeState:
        state = self._nodes.get(node_id)
        if state is None:
            # A microphone we were not told about came online.
            state = _NodeState(self.window)
            self._nodes[node_id] = state
        return state

    def _raw_score(self, spectrogram: torch.Tensor) -> float:
        """Reconstruction error if a model is loaded, otherwise frame energy."""
        x = spectrogram.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        if self.autoencoder is None:
            return float(x.pow(2).mean().item())
        return float(self.autoencoder.anomaly_score(x).mean().item())

    @staticmethod
    def _robust_z(value: float, history: deque[float]) -> tuple[float, float]:
        """
        Modified z-score of ``value`` against ``history``.

        Returns ``(z, median)``. Falls back to standard deviation when the MAD
        collapses to zero — which happens with a perfectly constant signal, a
        case that is common in tests and in genuinely silent channels.
        """
        ordered = sorted(history)
        n = len(ordered)
        mid = n // 2
        median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])

        deviations = sorted(abs(s - median) for s in ordered)
        mad = deviations[mid] if n % 2 else 0.5 * (deviations[mid - 1] + deviations[mid])

        if mad > 1e-12:
            return _MAD_TO_SIGMA * (value - median) / mad, median

        mean = sum(ordered) / n
        variance = sum((s - mean) ** 2 for s in ordered) / n
        std = math.sqrt(variance)
        if std > 1e-12:
            return (value - mean) / std, median

        # Degenerate baseline: every observation so far is identical, so there
        # is no scale to measure a departure against. That a departure happened
        # is meaningful; its magnitude is not, because the ratio is taken
        # against an arbitrarily small median and can be astronomically large.
        #
        # Saturating at a fixed, unambiguously-critical value keeps that from
        # reaching a Prometheus gauge and an LLM prompt as a z-score of 40,000 —
        # a number an operator cannot interpret and a model will happily narrate.
        # Sign is preserved: a channel that went quiet is not a fault.
        scale = max(abs(median), 1e-9)
        relative = (value - median) / scale
        if abs(relative) <= 1e-9:
            return 0.0, median
        return math.copysign(_DEGENERATE_Z, relative), median

    def _severity(self, z: float) -> Severity:
        if z >= 2.0 * self.z_threshold:
            return "critical"
        if z >= self.z_threshold:
            return "warning"
        return "normal"

    # -- public API ---------------------------------------------------------

    def score(self, node_id: int, spectrogram: torch.Tensor) -> ScoreResult:
        """Score one frame and fold it into that node's rolling baseline."""
        state = self._state(node_id)
        raw = self._raw_score(spectrogram)

        state.total_frames += 1
        is_warmup = state.total_frames <= self.warmup_frames

        if is_warmup or not state.scores:
            z, median = 0.0, raw
        else:
            z, median = self._robust_z(raw, state.scores)

        is_anomaly = (not is_warmup) and z >= self.z_threshold
        severity = self._severity(z) if not is_warmup else "normal"

        if is_anomaly:
            state.anomaly_frames += 1

        # Anomalous frames are still admitted to the baseline. Excluding them
        # would freeze the window during a sustained fault and make a new
        # steady state permanently anomalous; the median keeps them in check.
        state.scores.append(raw)

        return ScoreResult(
            node_id=node_id,
            raw_score=raw,
            z_score=float(z),
            is_anomaly=bool(is_anomaly),
            is_warmup=bool(is_warmup),
            severity=severity,
            baseline_median=float(median),
        )

    def get_node_summary(self) -> dict[int, dict[str, float]]:
        """Per-node counters, for the ``/health`` payload and Dagster checks."""
        summary: dict[int, dict[str, float]] = {}
        for node_id, state in sorted(self._nodes.items()):
            total = state.total_frames
            summary[node_id] = {
                "total_frames": total,
                "anomaly_frames": state.anomaly_frames,
                "anomaly_rate": (state.anomaly_frames / total) if total else 0.0,
                "baseline_samples": len(state.scores),
                "is_warmed_up": total > self.warmup_frames,
            }
        return summary

    def reset(self, node_id: int | None = None) -> None:
        """Clear one node's baseline, or every node's."""
        targets = self._nodes.keys() if node_id is None else [node_id]
        for nid in list(targets):
            self._nodes[nid] = _NodeState(self.window)
