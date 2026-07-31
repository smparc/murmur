"""
Spatial probe — turns time-aligned raw audio into acoustic geometry.

This has to live in the ingestion service, and that is a constraint rather than
a preference. TDOA is recovered from the *phase* relationship between channels,
and the mel spectrogram discards phase entirely. By the time audio reaches the
inference worker the information is already gone, so the measurement has to
happen upstream, on the raw waveforms, before the transform.

The probe buffers the most recent chunk from every microphone, and once the
whole array has reported for the same acoustic instant it estimates pairwise
delays and solves for the source position. The result is published alongside the
spectrograms so the worker can use it to reweight the graph.

Clock synchronisation
---------------------
The delays being measured span roughly 15 ms across a 5 m array. Chunk
timestamps come from each edge device's own clock, so any skew between devices
larger than that swamps the physical delay completely and the estimates become
noise. This matters:

- The default staleness tolerance is one chunk duration, and ``clock_spread`` is
  reported on every snapshot so drift is observable rather than silent.
- A real deployment needs PTP, or NTP with a disciplined local clock, on the
  edge nodes. Unsynchronised devices will produce confident, wrong positions.
- ``max_tau`` in the GCC-PHAT search bounds each estimate by the pair's physical
  separation, which contains the damage but does not repair it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.mapping.tdoa import SPEED_OF_SOUND, TDOAEstimate, localize_source, pairwise_tdoa

log = logging.getLogger(__name__)


@dataclass
class SpatialSnapshot:
    """One coherent acoustic instant across the whole array."""

    timestamp: float
    estimates: list[TDOAEstimate]
    position: np.ndarray | None
    residual: float
    clock_spread: float
    """Seconds between the earliest and latest contributing chunk."""

    @property
    def mean_coherence(self) -> float:
        if not self.estimates:
            return 0.0
        return float(np.mean([e.coherence for e in self.estimates]))

    @property
    def localized(self) -> bool:
        return self.position is not None

    def coherence_map(self) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for e in self.estimates:
            out[(e.i, e.j)] = e.coherence
            out[(e.j, e.i)] = e.coherence
        return out

    def to_payload(self) -> dict:
        """MessagePack-friendly wire form."""
        return {
            "timestamp": self.timestamp,
            "clock_spread": float(self.clock_spread),
            "mean_coherence": self.mean_coherence,
            "residual": float(self.residual) if np.isfinite(self.residual) else None,
            "position": None if self.position is None else [float(v) for v in self.position],
            "pairs": [
                {
                    "i": e.i,
                    "j": e.j,
                    "tau": float(e.tau),
                    "coherence": float(e.coherence),
                }
                for e in self.estimates
            ],
        }


@dataclass
class _PendingChunk:
    timestamp: float
    waveform: np.ndarray


@dataclass
class SpatialProbe:
    """
    Buffers raw chunks per node and solves the array when it is complete.

    Parameters
    ----------
    mic_coords:
        ``(N, 3)`` microphone positions in metres.
    sample_rate:
        Hz.
    staleness_tolerance:
        Maximum clock spread, in seconds, still considered one instant. Chunks
        further apart than this describe different moments and must not be
        cross-correlated.
    min_coherence:
        Pairs below this are excluded from the position solve.
    plane_z:
        Height to constrain the solution to. Defaults to the mean microphone
        height, because the stock array is coplanar and cannot observe elevation
        (see ``localize_source``). Pass ``None`` for a full 3-D solve, which
        needs a non-coplanar array of at least five microphones.
    """

    mic_coords: np.ndarray
    sample_rate: int
    staleness_tolerance: float = 0.5
    min_coherence: float = 0.15
    interp: int = 8
    plane_z: float | None = field(default=None)
    speed: float = SPEED_OF_SOUND
    _solve_plane: bool = field(default=True, init=False)
    _pending: dict[int, _PendingChunk] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.mic_coords = np.asarray(self.mic_coords, dtype=np.float64)
        if self.mic_coords.ndim != 2 or self.mic_coords.shape[1] != 3:
            raise ValueError(f"mic_coords must be (N, 3), got {self.mic_coords.shape}")
        if self.plane_z is None and self._solve_plane:
            self.plane_z = float(self.mic_coords[:, 2].mean())

    @property
    def num_nodes(self) -> int:
        return int(self.mic_coords.shape[0])

    def push(self, node_id: int, timestamp: float, waveform: np.ndarray) -> None:
        """Buffer one node's chunk, replacing any older one it holds."""
        if not 0 <= node_id < self.num_nodes:
            # A node outside the configured topology has no position, so it
            # cannot contribute to a geometric solve.
            return
        self._pending[node_id] = _PendingChunk(timestamp, np.asarray(waveform, dtype=np.float64))

    def is_ready(self) -> bool:
        """True when every node has contributed a chunk from the same instant."""
        if len(self._pending) < self.num_nodes:
            return False
        spread = self._spread()
        return spread <= self.staleness_tolerance

    def _spread(self) -> float:
        stamps = [c.timestamp for c in self._pending.values()]
        return max(stamps) - min(stamps)

    def solve(self, clear: bool = True) -> SpatialSnapshot | None:
        """
        Estimate delays and source position, if the array is complete.

        Returns ``None`` when nodes are missing or their chunks describe
        different moments.
        """
        if not self.is_ready():
            return None

        ordered = [self._pending[i] for i in range(self.num_nodes)]
        spread = self._spread()

        # Channels must be equal length for a well-defined cross-correlation.
        length = min(c.waveform.size for c in ordered)
        if length < 2:
            if clear:
                self._pending.clear()
            return None
        channels = np.stack([c.waveform[:length] for c in ordered])

        estimates = pairwise_tdoa(
            channels,
            self.mic_coords,
            self.sample_rate,
            interp=self.interp,
            speed=self.speed,
        )
        position, residual = localize_source(
            estimates,
            self.mic_coords,
            speed=self.speed,
            plane_z=self.plane_z,
            min_coherence=self.min_coherence,
        )

        snapshot = SpatialSnapshot(
            timestamp=max(c.timestamp for c in ordered),
            estimates=estimates,
            position=position,
            residual=residual,
            clock_spread=spread,
        )

        if spread > self.staleness_tolerance / 2:
            log.debug(
                "Array clock spread %.1f ms is over half the tolerance — "
                "check edge time synchronisation",
                spread * 1e3,
            )

        if clear:
            self._pending.clear()
        return snapshot

    def reset(self) -> None:
        self._pending.clear()
