"""
End-to-end tests for the two evaluation entry points.

Kept deliberately small — a handful of epochs on a handful of clips. The purpose
is to prove the wiring holds together (features -> training -> scoring ->
metrics) and that the reported numbers are structurally sound, not to measure
accuracy. Accuracy is what the CLI runs are for.
"""

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.baselines import DCASE2020_AUTOENCODER, lookup
from benchmarks.evaluate_dataset import evaluate, format_report
from benchmarks.evaluate_synthetic import evaluate as evaluate_synthetic
from benchmarks.scenario import ScenarioConfig


@pytest.fixture
def tiny_dcase(tmp_path):
    """A DCASE-shaped tree where anomalies are audibly louder than normals."""
    root = tmp_path / "dcase"
    rng = np.random.default_rng(0)

    def write(path, amplitude):
        path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(path, 16_000, rng.normal(0, amplitude, 8_000).astype(np.float32))

    for i in range(10):
        write(root / "fan" / "train" / f"normal_id_00_{i:08d}.wav", 0.05)
    for i in range(5):
        write(root / "fan" / "test" / f"normal_id_00_{i:08d}.wav", 0.05)
        write(root / "fan" / "test" / f"anomaly_id_00_{i:08d}.wav", 0.6)
    return root


class TestEvaluateDataset:
    def test_runs_end_to_end_and_reports_every_metric(self, tiny_dcase):
        results = evaluate("dcase", tiny_dcase, epochs=3, seed=0)

        assert results["groups_scored"] == 1
        row = results["per_group"]["fan/id_00"]
        for key in ("auc", "pauc_10", "average_precision"):
            assert 0.0 <= row[key] <= 1.0
        assert row["train_samples"] == 10
        assert row["test_samples"] == 10
        assert row["anomaly_rate"] == pytest.approx(0.5)

    def test_detects_an_obvious_amplitude_anomaly(self, tiny_dcase):
        # If the harness cannot separate 0.05 from 0.6 amplitude noise, the
        # plumbing is wrong — this is the easiest possible discrimination.
        results = evaluate("dcase", tiny_dcase, epochs=5, seed=0)
        assert results["per_group"]["fan/id_00"]["auc"] > 0.9

    def test_attaches_the_published_baseline_for_known_machines(self, tiny_dcase):
        results = evaluate("dcase", tiny_dcase, epochs=2, seed=0)
        row = results["per_group"]["fan/id_00"]
        assert row["baseline_auc"] == pytest.approx(DCASE2020_AUTOENCODER["fan"].auc)
        assert row["auc_vs_baseline"] == pytest.approx(row["auc"] - row["baseline_auc"])

    def test_report_renders_a_markdown_table(self, tiny_dcase):
        results = evaluate("dcase", tiny_dcase, epochs=2, seed=0)
        report = format_report(results)
        assert "| Machine | AUC | pAUC@10% |" in report
        assert "fan/id_00" in report
        assert "Mean AUC" in report

    def test_is_deterministic_for_a_seed(self, tiny_dcase):
        a = evaluate("dcase", tiny_dcase, epochs=3, seed=42)
        b = evaluate("dcase", tiny_dcase, epochs=3, seed=42)
        assert a["per_group"]["fan/id_00"]["auc"] == pytest.approx(
            b["per_group"]["fan/id_00"]["auc"]
        )

    def test_single_class_test_fold_is_skipped_not_fatal(self, tmp_path):
        root = tmp_path / "dcase"
        rng = np.random.default_rng(0)
        for split, count in (("train", 4), ("test", 3)):
            for i in range(count):
                path = root / "fan" / split / f"normal_id_00_{i:08d}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(path, 16_000, rng.normal(0, 0.05, 8_000).astype(np.float32))

        with pytest.raises(RuntimeError, match="No group could be scored"):
            evaluate("dcase", root, epochs=1)


class TestEvaluateSynthetic:
    def test_produces_a_complete_summary(self):
        results = evaluate_synthetic(
            ScenarioConfig(num_nodes=2, frames_per_node=200, min_onset_frame=80, seed=3)
        )
        summary = results["summary"]
        for key in ("roc_auc", "pauc_10", "mean_lead_time_s", "false_alarms_per_hour"):
            assert key in summary
        assert 0.0 <= summary["roc_auc"] <= 1.0
        assert summary["frames"] > 0

    def test_beats_chance_on_ramped_faults(self):
        results = evaluate_synthetic(
            ScenarioConfig(num_nodes=2, frames_per_node=250, min_onset_frame=100, seed=5)
        )
        assert results["summary"]["roc_auc"] > 0.6

    def test_reports_lead_time_per_node(self):
        results = evaluate_synthetic(
            ScenarioConfig(num_nodes=3, frames_per_node=250, min_onset_frame=100, seed=8)
        )
        assert set(results["per_node"]) == {"0", "1", "2"}
        for report in results["per_node"].values():
            assert report["events"] >= 0
            assert report["detected"] + report["missed"] == report["events"]

    def test_is_deterministic_for_a_seed(self):
        config = ScenarioConfig(num_nodes=2, frames_per_node=200, min_onset_frame=80, seed=11)
        a = evaluate_synthetic(config)
        b = evaluate_synthetic(config)
        assert a["summary"]["roc_auc"] == pytest.approx(b["summary"]["roc_auc"])


class TestBaselines:
    def test_lookup_is_insensitive_to_case_and_separators(self):
        assert lookup("fan") is not None
        assert lookup("ToyCar") is not None
        assert lookup("toy_car") is not None
        assert lookup("TOY-CAR") is not None

    def test_unknown_machine_returns_none(self):
        assert lookup("centrifuge") is None

    def test_scores_are_on_a_zero_to_one_scale(self):
        # The challenge publishes percentages; storing them as percentages here
        # would silently make every delta against our own metrics nonsense.
        for baseline in DCASE2020_AUTOENCODER.values():
            assert 0.0 < baseline.auc <= 1.0
            assert 0.0 < baseline.pauc <= 1.0
