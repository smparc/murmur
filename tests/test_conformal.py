"""Tests for split-conformal prediction intervals on the TTF forecast."""

from __future__ import annotations

import numpy as np
import pytest

from src.forecasting.conformal import (
    ConformalCalibrator,
    PredictionInterval,
    conformal_quantile,
    evaluate_coverage,
    severity_bucket,
)


class TestConformalQuantile:
    def test_finite_sample_correction(self):
        """
        The radius is the ceil((n+1)(1-alpha))-th smallest residual, not the
        plain empirical quantile — that +1 is the whole guarantee.
        """
        residuals = np.arange(1, 101, dtype=float)  # 1..100
        # ceil(101 * 0.9) = 91 -> the 91st smallest value.
        assert conformal_quantile(residuals, alpha=0.1) == 91.0

    def test_returns_inf_when_undercalibrated(self):
        """10 points cannot certify 99% coverage; say so rather than guess."""
        assert conformal_quantile(np.arange(10.0), alpha=0.01) == float("inf")

    def test_tighter_alpha_gives_wider_radius(self):
        residuals = np.abs(np.random.default_rng(0).normal(0, 1, 5000))
        assert conformal_quantile(residuals, 0.01) > conformal_quantile(residuals, 0.2)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            conformal_quantile(np.array([]), 0.1)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_alpha(self, alpha):
        with pytest.raises(ValueError):
            conformal_quantile(np.ones(10), alpha)


class TestCoverageGuarantee:
    """The property that justifies the whole module."""

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_marginal_coverage_holds(self, alpha):
        rng = np.random.default_rng(11)
        truth = rng.uniform(0, 1, 6000)
        predicted = np.clip(truth + rng.normal(0, 0.1, 6000), 0, 1)

        cal, test = slice(0, 3000), slice(3000, None)
        calibrator = ConformalCalibrator(alpha=alpha).fit(predicted[cal], truth[cal])
        report = evaluate_coverage(calibrator.intervals(predicted[test]), truth[test])

        # Conformal guarantees >= 1-alpha; allow a small finite-sample margin.
        assert report["coverage"] >= (1 - alpha) - 0.02

    def test_holds_under_a_badly_biased_model(self):
        """
        Coverage is distribution-free: it must hold even for a model with a
        large systematic offset, by widening the interval to compensate.
        """
        rng = np.random.default_rng(12)
        truth = rng.uniform(0, 1, 4000)
        predicted = np.clip(truth * 0.3 + 0.4, 0, 1)  # heavily shrunk toward the mean

        cal, test = slice(0, 2000), slice(2000, None)
        calibrator = ConformalCalibrator(alpha=0.1).fit(predicted[cal], truth[cal])
        report = evaluate_coverage(calibrator.intervals(predicted[test]), truth[test])
        assert report["coverage"] >= 0.88

    def test_heavy_tailed_noise_still_covered(self):
        rng = np.random.default_rng(13)
        truth = rng.uniform(0, 1, 4000)
        predicted = np.clip(truth + rng.standard_cauchy(4000) * 0.02, 0, 1)

        cal, test = slice(0, 2000), slice(2000, None)
        calibrator = ConformalCalibrator(alpha=0.1).fit(predicted[cal], truth[cal])
        report = evaluate_coverage(calibrator.intervals(predicted[test]), truth[test])
        assert report["coverage"] >= 0.88


class TestMondrianCalibration:
    @staticmethod
    def _heteroscedastic(n: int, seed: int):
        rng = np.random.default_rng(seed)
        truth = rng.uniform(0, 1, n)
        predicted = np.clip(truth + rng.normal(0, 0.02 + 0.2 * truth), 0, 1)
        groups = np.array([severity_bucket(p) for p in predicted])
        return truth, predicted, groups

    def test_improves_conditional_coverage_on_the_critical_bucket(self):
        """
        Marginal calibration systematically under-covers the high-risk stratum
        when errors grow with the prediction — which is exactly the machines
        anyone is monitoring for.
        """
        truth, predicted, groups = self._heteroscedastic(8000, seed=21)
        cal, test = slice(0, 4000), slice(4000, None)

        marginal = ConformalCalibrator(alpha=0.1).fit(predicted[cal], truth[cal])
        mondrian = ConformalCalibrator(alpha=0.1).fit(predicted[cal], truth[cal], groups[cal])

        mask = groups[test] == "critical"
        assert mask.sum() > 100, "need a populated critical bucket for this test"

        m_cov = evaluate_coverage(marginal.intervals(predicted[test][mask]), truth[test][mask])[
            "coverage"
        ]
        d_cov = evaluate_coverage(
            mondrian.intervals(predicted[test][mask], groups[test][mask]),
            truth[test][mask],
        )["coverage"]

        assert m_cov < 0.9, "expected marginal calibration to under-cover here"
        assert d_cov > m_cov

    def test_sparse_group_falls_back_to_global(self):
        calibrator = ConformalCalibrator(alpha=0.1, min_group_size=100)
        predicted = np.linspace(0, 1, 200)
        truth = predicted + 0.05
        groups = np.array(["common"] * 190 + ["rare"] * 10)
        calibrator.fit(predicted, truth, groups)

        assert "common" in calibrator.group_radii
        assert "rare" not in calibrator.group_radii
        assert calibrator.interval(0.5, "rare").group == "global"
        assert calibrator.radius_for("rare") == calibrator.global_radius

    def test_group_counts_recorded(self):
        calibrator = ConformalCalibrator(alpha=0.1, min_group_size=2)
        calibrator.fit(np.zeros(6), np.zeros(6), np.array(["a"] * 4 + ["b"] * 2))
        assert calibrator.group_counts == {"a": 4, "b": 2}


class TestIntervalBehaviour:
    def _fitted(self, alpha=0.1):
        rng = np.random.default_rng(31)
        truth = rng.uniform(0, 1, 500)
        predicted = np.clip(truth + rng.normal(0, 0.1, 500), 0, 1)
        return ConformalCalibrator(alpha=alpha).fit(predicted, truth)

    def test_clipped_to_probability_range(self):
        """TTF is a probability; an interval reaching past [0,1] is nonsense."""
        calibrator = self._fitted()
        assert calibrator.interval(0.0).lower == 0.0
        assert calibrator.interval(1.0).upper == 1.0

    def test_interval_brackets_the_point_estimate(self):
        calibrator = self._fitted()
        interval = calibrator.interval(0.5)
        assert interval.lower <= interval.point <= interval.upper

    def test_contains(self):
        interval = PredictionInterval(point=0.5, lower=0.4, upper=0.6, alpha=0.1)
        assert interval.contains(0.45)
        assert not interval.contains(0.7)
        assert interval.width == pytest.approx(0.2)
        assert interval.confidence == pytest.approx(0.9)

    def test_unfitted_calibrator_refuses(self):
        with pytest.raises(RuntimeError):
            ConformalCalibrator().interval(0.5)

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError):
            ConformalCalibrator().fit(np.zeros(5), np.zeros(6))

    def test_empty_fit_rejected(self):
        with pytest.raises(ValueError):
            ConformalCalibrator().fit(np.array([]), np.array([]))

    def test_as_dict_is_serializable(self):
        import json

        payload = self._fitted().interval(0.42).as_dict()
        assert json.loads(json.dumps(payload))["point"] == pytest.approx(0.42)


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        rng = np.random.default_rng(41)
        truth = rng.uniform(0, 1, 400)
        predicted = np.clip(truth + rng.normal(0, 0.08, 400), 0, 1)
        groups = np.array([severity_bucket(p) for p in predicted])

        original = ConformalCalibrator(alpha=0.1, min_group_size=20)
        original.fit(predicted, truth, groups)

        path = tmp_path / "conformal.json"
        original.save(str(path))
        restored = ConformalCalibrator.load(str(path))

        assert restored.alpha == original.alpha
        assert restored.global_radius == pytest.approx(original.global_radius)
        assert restored.group_radii.keys() == original.group_radii.keys()
        assert restored.interval(0.5).upper == pytest.approx(original.interval(0.5).upper)

    def test_infinite_radius_survives_json(self, tmp_path):
        """`inf` is not valid JSON; it must round-trip through the sentinel."""
        calibrator = ConformalCalibrator(alpha=0.001).fit(np.zeros(5), np.ones(5))
        assert calibrator.global_radius == float("inf")

        path = tmp_path / "c.json"
        calibrator.save(str(path))
        assert ConformalCalibrator.load(str(path)).global_radius == float("inf")


class TestEvaluateCoverage:
    def test_reports_gap_against_nominal(self):
        intervals = [PredictionInterval(0.5, 0.4, 0.6, alpha=0.1) for _ in range(10)]
        targets = np.array([0.45] * 9 + [0.99])
        report = evaluate_coverage(intervals, targets)
        assert report["coverage"] == pytest.approx(0.9)
        assert report["coverage_gap"] == pytest.approx(0.0)
        assert report["n"] == 10

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            evaluate_coverage([PredictionInterval(0.5, 0.4, 0.6, 0.1)], np.zeros(3))

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            evaluate_coverage([], np.array([]))


class TestSeverityBucket:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "normal"),
            (0.32, "normal"),
            (0.33, "warning"),
            (0.65, "warning"),
            (0.66, "critical"),
            (1.0, "critical"),
        ],
    )
    def test_boundaries(self, value, expected):
        assert severity_bucket(value) == expected
