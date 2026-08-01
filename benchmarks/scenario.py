"""
Labelled degradation scenarios built on the edge simulator.

``src.ingestion.mock_edge_device`` already knows how to synthesise bearing
squeal, pump cavitation and rotating imbalance at a requested severity — and
then throws the parameters away, publishing only audio. That is correct for the
simulator's day job (leaking labels into the stream would invalidate the whole
pipeline) but it means the one thing the project has plenty of, known truth, was
going unused.

This module drives the same generator on a fixed schedule and *keeps* the
schedule. The result is a stream of frames that is byte-identical to what the
pipeline sees in normal operation, paired with a per-frame record of what was
actually wrong at that moment.

Faults ramp rather than switch. A bearing does not fail instantly, and a
detector evaluated against step-function faults will look far better than it is:
the interesting question is how far up the ramp it fires, which only a gradual
onset can ask.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from src.ingestion.mock_edge_device import FaultType, generate_mock_audio
from src.settings import settings

__all__ = ["FaultEvent", "Frame", "Scenario", "ScenarioConfig", "generate_scenario"]

_FAULTS = (FaultType.BEARING, FaultType.CAVITATION, FaultType.IMBALANCE)


@dataclass(frozen=True)
class FaultEvent:
    """One degradation episode on one microphone."""

    node_id: int
    fault: FaultType
    onset_frame: int
    failure_frame: int

    @property
    def duration_frames(self) -> int:
        return self.failure_frame - self.onset_frame

    def severity_at(self, frame: int) -> float:
        """Linear ramp from 0 at onset to 1.0 at failure; 0 outside the window."""
        if frame < self.onset_frame or frame >= self.failure_frame:
            return 0.0
        span = self.duration_frames
        return (frame - self.onset_frame) / span if span > 0 else 1.0


@dataclass(frozen=True)
class Frame:
    """One audio chunk plus the truth about it."""

    node_id: int
    frame_index: int
    audio: np.ndarray
    severity: float
    fault: FaultType

    @property
    def is_faulty(self) -> bool:
        return self.severity > 0.0


@dataclass
class ScenarioConfig:
    """
    Knobs for scenario generation.

    Defaults describe a short run that finishes in seconds on CPU while still
    containing enough healthy frames for the scorer to establish a baseline —
    ``AnomalyScorer`` ignores its first ``warmup_frames`` per node, so a scenario
    shorter than that would measure nothing at all.
    """

    num_nodes: int = 4
    frames_per_node: int = 400
    events_per_node: int = 1
    min_onset_frame: int = 120
    ramp_frames: tuple[int, int] = (60, 140)
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        if self.frames_per_node < 1:
            raise ValueError("frames_per_node must be >= 1")
        low, high = self.ramp_frames
        if low < 1 or high < low:
            raise ValueError("ramp_frames must be a valid (low, high) range with low >= 1")
        if self.min_onset_frame < 0:
            raise ValueError("min_onset_frame must be >= 0")


@dataclass
class Scenario:
    """A generated run: frames in emission order, plus the ground truth."""

    frames: list[Frame]
    events: list[FaultEvent]
    config: ScenarioConfig
    frame_interval_s: float = field(default_factory=lambda: settings.CHUNK_DURATION)

    def frames_for(self, node_id: int) -> list[Frame]:
        """Frames from one microphone, in order."""
        return [f for f in self.frames if f.node_id == node_id]

    def events_for(self, node_id: int) -> list[tuple[int, int]]:
        """``(onset, failure)`` pairs for one node, for the lead-time metrics."""
        return [
            (e.onset_frame, e.failure_frame) for e in self.events if e.node_id == node_id
        ]

    @property
    def duration_s(self) -> float:
        return self.config.frames_per_node * self.frame_interval_s

    @property
    def positive_rate(self) -> float:
        if not self.frames:
            return 0.0
        return sum(1 for f in self.frames if f.is_faulty) / len(self.frames)


def _plan_events(config: ScenarioConfig, rng: random.Random) -> list[FaultEvent]:
    """Lay out non-overlapping degradation episodes for every node."""
    events: list[FaultEvent] = []
    for node_id in range(config.num_nodes):
        cursor = config.min_onset_frame
        for _ in range(config.events_per_node):
            ramp = rng.randint(*config.ramp_frames)
            # Leave room for the ramp to complete inside the run.
            latest_onset = config.frames_per_node - ramp
            if cursor >= latest_onset:
                break
            onset = rng.randint(cursor, latest_onset)
            failure = onset + ramp
            events.append(
                FaultEvent(
                    node_id=node_id,
                    fault=rng.choice(_FAULTS),
                    onset_frame=onset,
                    failure_frame=failure,
                )
            )
            # Healthy gap before the next episode so baselines can recover.
            cursor = failure + config.min_onset_frame // 2
    return events


def generate_scenario(config: ScenarioConfig | None = None) -> Scenario:
    """
    Build a deterministic labelled run.

    The same ``seed`` always produces the same audio, so a benchmark number is
    reproducible and a regression is attributable to the model rather than to
    the draw.
    """
    config = config or ScenarioConfig()
    rng = random.Random(config.seed)
    np_rng_state = np.random.get_state()
    np.random.seed(config.seed)

    try:
        events = _plan_events(config, rng)
        by_node: dict[int, list[FaultEvent]] = {}
        for event in events:
            by_node.setdefault(event.node_id, []).append(event)

        frames: list[Frame] = []
        # Emit frame-major so the stream interleaves nodes exactly as Kafka
        # would deliver them.
        for frame_index in range(config.frames_per_node):
            for node_id in range(config.num_nodes):
                severity = 0.0
                fault = FaultType.NONE
                for event in by_node.get(node_id, ()):
                    s = event.severity_at(frame_index)
                    if s > 0.0:
                        severity, fault = s, event.fault
                        break

                audio = np.frombuffer(
                    generate_mock_audio(node_id=node_id, fault=fault, severity=severity),
                    dtype=np.float32,
                )
                frames.append(
                    Frame(
                        node_id=node_id,
                        frame_index=frame_index,
                        audio=audio,
                        severity=severity,
                        fault=fault,
                    )
                )
    finally:
        np.random.set_state(np_rng_state)

    return Scenario(frames=frames, events=events, config=config)
