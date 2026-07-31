"""
Distribution-free prediction intervals for the Time-to-Failure forecast.

The Liquid Network emits a sigmoid — a number in [0, 1] that looks like a
probability but is not calibrated to be one. Nothing in the training objective
forces 0.7 to mean "fails 70% of the time". A maintenance planner cannot act on
that: scheduling an outage is expensive, and the question is not "what is the
score" but "how wrong could this be".

Split conformal prediction answers exactly that, and does so without assuming
anything about the model, the noise distribution, or the data-generating
process. Hold out a calibration set the model never trained on, measure how far
its predictions actually landed from the truth, and take a quantile of those
errors. The resulting interval has *finite-sample marginal coverage*: for
exchangeable data, the true value falls inside at least ``1 - alpha`` of the
time. No asymptotics, no Gaussian assumption, no retraining.

The one thing it does assume is exchangeability between calibration and
production data. Acoustic drift breaks that, which is precisely what the Dagster
drift monitor exists to catch — a coverage collapse on live data is the signal
that the calibration is stale, and is far more actionable than a moved MAE.

References
----------
Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*.
Lei et al. (2018), "Distribution-Free Predictive Inference for Regression",
JASA 113(523).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

# TTF is a probability; intervals are clipped to it.
_LOWER_BOUND = 0.0
_UPPER_BOUND = 1.0


@dataclass(frozen=True)
class PredictionInterval:
    """A point forecast with a calibrated uncertainty band."""

    point: float
    lower: float
    upper: float
    alpha: float
    group: str = "global"

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def as_dict(self) -> dict[str, float | str]:
        return {
            "point": round(self.point, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "width": round(self.width, 6),
            "confidence": round(self.confidence, 4),
            "group": self.group,
        }


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    The split-conformal radius for a given miscoverage level.

    Takes the ``ceil((n + 1) * (1 - alpha))``-th smallest absolute residual
    rather than the plain empirical quantile. That ``+1`` is what converts an
    asymptotic statement into a finite-sample guarantee — it accounts for the
    unseen test point being exchangeable with the calibration points.

    Returns ``inf`` when the calibration set is too small to support the
    requested confidence at all. That is the honest answer: with 10 points you
    cannot certify 99% coverage, and silently returning the maximum residual
    would understate the interval while claiming a guarantee it does not have.
    """
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    n = residuals.size
    if n == 0:
        raise ValueError("cannot calibrate on an empty residual set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    # `rank` is 1-indexed.
    return float(np.partition(residuals, rank - 1)[rank - 1])


@dataclass
class ConformalCalibrator:
    """
    Split-conformal calibrator, optionally conditioned on a grouping variable.

    Marginal coverage is a weak promise: a calibrator can hit 90% overall while
    systematically under-covering the degrading machines — which are the only
    ones anybody cares about. Passing ``groups`` (severity bucket, machine type,
    site) fits a separate radius per group, a *Mondrian* conformal predictor,
    which restores coverage within each stratum at the cost of needing enough
    calibration points in each.

    Groups that are too sparse to calibrate fall back to the global radius
    rather than emitting an interval with no support behind it.
    """

    alpha: float = 0.1
    min_group_size: int = 30
    global_radius: float = float("inf")
    group_radii: dict[str, float] = field(default_factory=dict)
    n_calibration: int = 0
    group_counts: dict[str, int] = field(default_factory=dict)

    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray | list[str] | None = None,
    ) -> ConformalCalibrator:
        """
        Calibrate on held-out predictions.

        The calibration set must be disjoint from training data. Reusing
        training points makes residuals optimistically small and silently
        destroys the coverage guarantee — the failure mode is an interval that
        looks reassuringly tight and is wrong far more often than advertised.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        targets = np.asarray(targets, dtype=np.float64).ravel()
        if predictions.shape != targets.shape:
            raise ValueError(
                f"predictions {predictions.shape} and targets {targets.shape} must match"
            )
        if predictions.size == 0:
            raise ValueError("cannot calibrate on an empty set")

        residuals = np.abs(predictions - targets)
        self.global_radius = conformal_quantile(residuals, self.alpha)
        self.n_calibration = int(residuals.size)
        self.group_radii = {}
        self.group_counts = {}

        if groups is not None:
            labels = np.asarray(groups).ravel()
            if labels.shape != predictions.shape:
                raise ValueError(
                    f"groups {labels.shape} must match predictions {predictions.shape}"
                )
            for label in np.unique(labels):
                mask = labels == label
                count = int(mask.sum())
                self.group_counts[str(label)] = count
                if count >= self.min_group_size:
                    self.group_radii[str(label)] = conformal_quantile(residuals[mask], self.alpha)

        return self

    @property
    def is_fitted(self) -> bool:
        return self.n_calibration > 0

    def radius_for(self, group: str | None = None) -> float:
        if group is not None and group in self.group_radii:
            return self.group_radii[group]
        return self.global_radius

    def interval(self, prediction: float, group: str | None = None) -> PredictionInterval:
        """Wrap a point forecast in its calibrated band, clipped to [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("calibrator has not been fitted")

        radius = self.radius_for(group)
        resolved = group if (group and group in self.group_radii) else "global"
        return PredictionInterval(
            point=float(prediction),
            lower=float(max(_LOWER_BOUND, prediction - radius)),
            upper=float(min(_UPPER_BOUND, prediction + radius)),
            alpha=self.alpha,
            group=resolved,
        )

    def intervals(
        self, predictions: np.ndarray, groups: np.ndarray | list[str] | None = None
    ) -> list[PredictionInterval]:
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        if groups is None:
            return [self.interval(p) for p in predictions]
        labels = np.asarray(groups).ravel()
        return [self.interval(p, str(g)) for p, g in zip(predictions, labels, strict=True)]

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "min_group_size": self.min_group_size,
            "global_radius": self.global_radius,
            "group_radii": self.group_radii,
            "n_calibration": self.n_calibration,
            "group_counts": self.group_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ConformalCalibrator:
        return cls(
            alpha=float(payload["alpha"]),
            min_group_size=int(payload.get("min_group_size", 30)),
            global_radius=float(payload["global_radius"]),
            group_radii={k: float(v) for k, v in payload.get("group_radii", {}).items()},
            n_calibration=int(payload.get("n_calibration", 0)),
            group_counts={k: int(v) for k, v in payload.get("group_counts", {}).items()},
        )

    def save(self, path: str) -> None:
        """Persist alongside the model weights it was calibrated against."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # `inf` is not valid JSON; the loader maps the sentinel back.
        payload = self.to_dict()
        if math.isinf(payload["global_radius"]):
            payload["global_radius"] = None
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> ConformalCalibrator:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("global_radius") is None:
            payload["global_radius"] = float("inf")
        return cls.from_dict(payload)


def evaluate_coverage(
    intervals: list[PredictionInterval],
    targets: np.ndarray,
) -> dict[str, float]:
    """
    Measure realised coverage against the nominal level.

    Empirical coverage materially below nominal means the exchangeability
    assumption has broken — the live acoustic distribution has drifted away from
    the calibration set. Coverage far *above* nominal is not free either: it
    means the intervals are wider than they need to be and every forecast is
    less informative than it could be.
    """
    targets = np.asarray(targets, dtype=np.float64).ravel()
    if len(intervals) != targets.size:
        raise ValueError(f"got {len(intervals)} intervals for {targets.size} targets")
    if not intervals:
        raise ValueError("cannot evaluate coverage on an empty set")

    covered = np.array([iv.contains(t) for iv, t in zip(intervals, targets, strict=True)])
    widths = np.array([iv.width for iv in intervals])
    nominal = intervals[0].confidence

    return {
        "coverage": float(covered.mean()),
        "nominal_coverage": float(nominal),
        "coverage_gap": float(covered.mean() - nominal),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "n": int(targets.size),
    }


def severity_bucket(prediction: float) -> str:
    """
    Group forecasts for Mondrian calibration.

    Mirrors the severity bands the API and dashboard already use, so a
    per-group coverage report reads in the same terms as the alerts it explains.
    """
    if prediction >= 0.66:
        return "critical"
    if prediction >= 0.33:
        return "warning"
    return "normal"


def main() -> None:  # pragma: no cover - manual inspection helper
    rng = np.random.default_rng(7)

    # A deliberately heteroscedastic model: it is confident when healthy and
    # erratic near failure, which is exactly when marginal coverage misleads.
    truth = rng.uniform(0, 1, 4000)
    noise = 0.02 + 0.18 * truth
    predicted = np.clip(truth + rng.normal(0, noise), 0, 1)

    cal, test = slice(0, 2000), slice(2000, None)
    groups = np.array([severity_bucket(p) for p in predicted])

    marginal = ConformalCalibrator(alpha=0.1).fit(predicted[cal], truth[cal])
    mondrian = ConformalCalibrator(alpha=0.1).fit(predicted[cal], truth[cal], groups[cal])

    print("[*] Marginal:", evaluate_coverage(marginal.intervals(predicted[test]), truth[test]))
    print(
        "[*] Mondrian:",
        evaluate_coverage(mondrian.intervals(predicted[test], groups[test]), truth[test]),
    )
    print("\n[*] Per-group coverage (why marginal is not enough):")
    for bucket in ("normal", "warning", "critical"):
        mask = groups[test] == bucket
        if mask.sum() < 20:
            continue
        m_iv = marginal.intervals(predicted[test][mask])
        d_iv = mondrian.intervals(predicted[test][mask], groups[test][mask])
        m = evaluate_coverage(m_iv, truth[test][mask])
        d = evaluate_coverage(d_iv, truth[test][mask])
        print(
            f"    {bucket:>8}: marginal={m['coverage']:.3f} (w={m['mean_width']:.3f})  "
            f"mondrian={d['coverage']:.3f} (w={d['mean_width']:.3f})"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
