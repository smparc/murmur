"""
Tests for labelled scenario generation.

The benchmark's credibility rests on two properties: the same seed must produce
the same run (or numbers are not comparable across commits), and the labels must
actually describe the audio (or the benchmark scores the detector against
fiction).
"""

import numpy as np
import pytest

from benchmarks.scenario import FaultEvent, ScenarioConfig, generate_scenario
from src.ingestion.mock_edge_device import FaultType


class TestFaultEvent:
    def test_severity_ramps_linearly_between_onset_and_failure(self):
        event = FaultEvent(0, FaultType.BEARING, onset_frame=10, failure_frame=20)
        assert event.severity_at(10) == pytest.approx(0.0)
        assert event.severity_at(15) == pytest.approx(0.5)
        assert event.severity_at(19) == pytest.approx(0.9)

    def test_severity_is_zero_outside_the_window(self):
        event = FaultEvent(0, FaultType.BEARING, onset_frame=10, failure_frame=20)
        assert event.severity_at(9) == 0.0
        assert event.severity_at(20) == 0.0
        assert event.severity_at(100) == 0.0

    def test_duration(self):
        assert FaultEvent(0, FaultType.NONE, 5, 25).duration_frames == 20


class TestScenarioConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_nodes": 0},
            {"frames_per_node": 0},
            {"ramp_frames": (0, 10)},
            {"ramp_frames": (50, 10)},
            {"min_onset_frame": -1},
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ScenarioConfig(**kwargs)


class TestGeneration:
    def test_frame_count_matches_configuration(self):
        config = ScenarioConfig(num_nodes=3, frames_per_node=60, min_onset_frame=20)
        scenario = generate_scenario(config)
        assert len(scenario.frames) == 3 * 60
        assert len(scenario.frames_for(0)) == 60

    def test_same_seed_reproduces_identical_audio(self):
        config = ScenarioConfig(num_nodes=2, frames_per_node=40, min_onset_frame=10, seed=99)
        a = generate_scenario(config)
        b = generate_scenario(config)

        assert [e.onset_frame for e in a.events] == [e.onset_frame for e in b.events]
        for fa, fb in zip(a.frames, b.frames, strict=True):
            assert np.array_equal(fa.audio, fb.audio)

    def test_different_seeds_diverge(self):
        base = ScenarioConfig(num_nodes=2, frames_per_node=40, min_onset_frame=10, seed=1)
        other = ScenarioConfig(num_nodes=2, frames_per_node=40, min_onset_frame=10, seed=2)
        a, b = generate_scenario(base), generate_scenario(other)
        assert any(
            not np.array_equal(fa.audio, fb.audio)
            for fa, fb in zip(a.frames, b.frames, strict=True)
        )

    def test_generation_does_not_leak_numpy_global_seed(self):
        # The generator seeds numpy to stay deterministic; it must put the
        # caller's RNG state back, or it silently makes the *rest* of the
        # process deterministic too.
        np.random.seed(4321)
        before = np.random.random()
        np.random.seed(4321)
        generate_scenario(ScenarioConfig(num_nodes=1, frames_per_node=20, min_onset_frame=5))
        after = np.random.random()
        assert before == pytest.approx(after)

    def test_labels_agree_with_the_planned_events(self):
        config = ScenarioConfig(num_nodes=2, frames_per_node=80, min_onset_frame=20, seed=7)
        scenario = generate_scenario(config)

        for frame in scenario.frames:
            expected = 0.0
            for event in scenario.events:
                if event.node_id == frame.node_id:
                    expected = max(expected, event.severity_at(frame.frame_index))
            assert frame.severity == pytest.approx(expected)
            assert frame.is_faulty == (expected > 0.0)

    def test_faulty_frames_carry_a_named_fault(self):
        scenario = generate_scenario(
            ScenarioConfig(num_nodes=2, frames_per_node=80, min_onset_frame=20, seed=3)
        )
        for frame in scenario.frames:
            if frame.is_faulty:
                assert frame.fault is not FaultType.NONE
            else:
                assert frame.fault is FaultType.NONE

    def test_audio_length_matches_the_configured_chunk(self):
        from src.settings import settings

        scenario = generate_scenario(
            ScenarioConfig(num_nodes=1, frames_per_node=5, min_onset_frame=1)
        )
        for frame in scenario.frames:
            assert frame.audio.shape == (settings.SAMPLES_PER_CHUNK,)
            assert frame.audio.dtype == np.float32

    def test_faults_raise_measurable_energy(self):
        # A severe fault must be distinguishable from silence-adjacent normal
        # operation, otherwise the benchmark is unwinnable by construction.
        scenario = generate_scenario(
            ScenarioConfig(num_nodes=1, frames_per_node=200, min_onset_frame=40, seed=11)
        )
        healthy = [f for f in scenario.frames if not f.is_faulty]
        severe = [f for f in scenario.frames if f.severity > 0.7]
        assert healthy and severe

        healthy_energy = np.mean([np.mean(f.audio**2) for f in healthy])
        severe_energy = np.mean([np.mean(f.audio**2) for f in severe])
        assert severe_energy > healthy_energy

    def test_events_stay_inside_the_run(self):
        scenario = generate_scenario(
            ScenarioConfig(num_nodes=4, frames_per_node=300, min_onset_frame=50, seed=5)
        )
        assert scenario.events
        for event in scenario.events:
            assert event.onset_frame >= 50
            assert event.failure_frame <= 300

    def test_positive_rate_is_a_sane_fraction(self):
        scenario = generate_scenario(
            ScenarioConfig(num_nodes=4, frames_per_node=300, min_onset_frame=50, seed=5)
        )
        assert 0.0 < scenario.positive_rate < 1.0

    def test_events_for_returns_node_scoped_pairs(self):
        scenario = generate_scenario(
            ScenarioConfig(num_nodes=3, frames_per_node=250, min_onset_frame=40, seed=13)
        )
        for node_id in range(3):
            for onset, failure in scenario.events_for(node_id):
                assert onset < failure
