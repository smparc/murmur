"""
Scoring metrics for acoustic anomaly detection and failure forecasting.

Two families live here, and they answer different questions.

**Threshold-free ranking metrics** (``roc_auc``, ``partial_roc_auc``,
``average_precision``) ask whether the model orders faulty frames above healthy
ones. They are what the anomalous-sound-detection literature reports, so they
are what makes Murmur's numbers comparable to a published baseline.

**Operational metrics** (``lead_time``, ``false_alarm_rate_per_hour``) ask
whether the system would be useful on a factory floor. A detector with an
excellent AUC that only fires once the bearing has already seized has no
operational value, and a detector that pages a technician six times a shift
gets muted within a week. Neither failure is visible in an AUC.

Implemented directly rather than pulled from scikit-learn: the whole package is
~200 lines of well-understood statistics, and Murmur's runtime dependency set is
already heavy enough without adding another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "DetectionMetrics",
    "LeadTimeMetrics",
    "average_precision",
    "confusion_at_threshold",
    "false_alarm_rate_per_hour",
    "first_sustained_alarm",
    "format_table",
    "lead_time",
    "roc_auc",
    "partial_roc_auc",
    "summarize",
]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def _average_ranks(values: Sequence[float]) -> list[float]:
    """
    1-based ranks with ties averaged.

    Tie handling is not a detail here. Reconstruction error saturates, and a
    detector that assigns the *same* score to many frames would otherwise be
    credited or penalised depending purely on input ordering.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = 0.5 * ((i + 1) + (j + 1))  # mean of 1-based positions i..j
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    Area under the ROC curve, via the Mann-Whitney U identity.

    Returns 0.5 when either class is absent — an undefined AUC reported as
    "no better than chance" rather than raising, so a sweep over many machine
    types does not abort on the one that happens to be all-normal.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    n_pos = sum(1 for v in labels if v)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _average_ranks(scores)
    rank_sum = sum(r for r, v in zip(ranks, labels) if v)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _roc_curve(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    """ROC points as ``(fpr, tpr)``, swept high-score-first. Starts at (0, 0)."""
    n_pos = sum(1 for v in labels if v)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]

    paired = sorted(zip(scores, labels), key=lambda p: -p[0])
    fprs, tprs = [0.0], [0.0]
    tp = fp = 0
    i = 0
    while i < len(paired):
        threshold = paired[i][0]
        # Consume the whole tie group before emitting a point, otherwise the
        # curve records operating points that no threshold can actually select.
        while i < len(paired) and paired[i][0] == threshold:
            if paired[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
    return fprs, tprs


def partial_roc_auc(
    scores: Sequence[float], labels: Sequence[int], max_fpr: float = 0.1
) -> float:
    """
    ROC AUC restricted to ``fpr <= max_fpr``, normalised back to ``[0, 1]``.

    This is the DCASE anomalous-sound-detection headline metric, and it is the
    honest one for maintenance: the only region of the ROC curve a plant will
    ever operate in is the low-false-positive end. A model can buy a strong full
    AUC with behaviour at 60% FPR that nobody would ever deploy.
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")

    fprs, tprs = _roc_curve(scores, labels)

    area = 0.0
    for (f0, t0), (f1, t1) in zip(zip(fprs, tprs), zip(fprs[1:], tprs[1:])):
        if f0 >= max_fpr:
            break
        if f1 > max_fpr:
            # Clip the final trapezoid at the cutoff, interpolating the TPR.
            span = f1 - f0
            t_at_cut = t0 if span <= 0 else t0 + (t1 - t0) * (max_fpr - f0) / span
            area += 0.5 * (t0 + t_at_cut) * (max_fpr - f0)
            break
        area += 0.5 * (t0 + t1) * (f1 - f0)

    return area / max_fpr


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    Area under the precision-recall curve (average precision).

    Preferred over ROC AUC when faults are rare, which they are by construction:
    ROC AUC is insensitive to class imbalance in a way that flatters a detector
    facing 1% positives.
    """
    n_pos = sum(1 for v in labels if v)
    if n_pos == 0:
        return 0.0

    paired = sorted(zip(scores, labels), key=lambda p: -p[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(paired):
        threshold = paired[i][0]
        while i < len(paired) and paired[i][0] == threshold:
            if paired[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionMetrics:
    """Confusion-matrix derived scores at one operating threshold."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


def confusion_at_threshold(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> DetectionMetrics:
    """Confusion matrix for ``score >= threshold`` counting as a detection."""
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if label and predicted:
            tp += 1
        elif label and not predicted:
            fn += 1
        elif not label and predicted:
            fp += 1
        else:
            tn += 1
    return DetectionMetrics(tp, fp, tn, fn)


def false_alarm_rate_per_hour(
    predictions: Sequence[bool],
    labels: Sequence[int],
    frame_interval_s: float,
) -> float:
    """
    Alarms raised on healthy frames, per hour of healthy runtime.

    The metric that decides whether an alerting integration survives contact
    with the people it pages. Expressed per hour rather than as a rate because
    that is the unit an on-call rotation actually reasons about.
    """
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")
    healthy = [i for i, label in enumerate(labels) if not label]
    if not healthy:
        return 0.0
    false_alarms = sum(1 for i in healthy if predictions[i])
    healthy_hours = len(healthy) * frame_interval_s / 3600.0
    return false_alarms / healthy_hours if healthy_hours > 0 else 0.0


# ---------------------------------------------------------------------------
# Lead time
# ---------------------------------------------------------------------------


@dataclass
class LeadTimeMetrics:
    """
    How much warning the system delivered, per fault event.

    ``lead_time_s`` is the headline: seconds between the first sustained alarm
    and the moment the machine reached end-of-life. It is the number a
    maintenance planner cares about, because it is the number that determines
    whether the part could have been ordered in time.

    ``detection_delay_s`` is its complement — seconds between fault onset and
    that same alarm — and measures how much of the available warning the
    detector threw away.

    Events where no sustained alarm ever fired are counted in ``missed`` and
    contribute to neither mean.
    """

    lead_times_s: list[float] = field(default_factory=list)
    detection_delays_s: list[float] = field(default_factory=list)
    missed: int = 0

    @property
    def detected(self) -> int:
        return len(self.lead_times_s)

    @property
    def total_events(self) -> int:
        return self.detected + self.missed

    @property
    def detection_rate(self) -> float:
        return self.detected / self.total_events if self.total_events else 0.0

    @property
    def mean_lead_time_s(self) -> float:
        return sum(self.lead_times_s) / len(self.lead_times_s) if self.lead_times_s else 0.0

    @property
    def median_lead_time_s(self) -> float:
        if not self.lead_times_s:
            return 0.0
        ordered = sorted(self.lead_times_s)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @property
    def mean_detection_delay_s(self) -> float:
        if not self.detection_delays_s:
            return 0.0
        return sum(self.detection_delays_s) / len(self.detection_delays_s)

    def as_dict(self) -> dict[str, float]:
        return {
            "events": self.total_events,
            "detected": self.detected,
            "missed": self.missed,
            "detection_rate": self.detection_rate,
            "mean_lead_time_s": self.mean_lead_time_s,
            "median_lead_time_s": self.median_lead_time_s,
            "mean_detection_delay_s": self.mean_detection_delay_s,
        }


def first_sustained_alarm(
    predictions: Sequence[bool],
    start: int,
    end: int,
    consecutive: int = 3,
) -> int | None:
    """
    Index of the first alarm in ``[start, end)`` sustained for ``consecutive``
    frames.

    Requiring persistence is what separates a warning from a blip. A single
    frame crossing threshold is routinely just a dropped tool or a passing
    forklift; demanding a run of them costs a little lead time and removes most
    of the noise that would otherwise be paged out.
    """
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    run = 0
    for i in range(max(0, start), min(end, len(predictions))):
        if predictions[i]:
            run += 1
            if run >= consecutive:
                return i - consecutive + 1
        else:
            run = 0
    return None


def lead_time(
    predictions: Sequence[bool],
    events: Sequence[tuple[int, int]],
    frame_interval_s: float,
    consecutive: int = 3,
) -> LeadTimeMetrics:
    """
    Lead time and detection delay for each ``(onset_frame, failure_frame)``.

    Alarms are only credited inside the window between onset and failure. An
    alarm before onset is a false alarm, not prescience, and is accounted for by
    ``false_alarm_rate_per_hour`` instead.
    """
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")

    result = LeadTimeMetrics()
    for onset, failure in events:
        alarm = first_sustained_alarm(predictions, onset, failure, consecutive)
        if alarm is None:
            result.missed += 1
            continue
        result.lead_times_s.append((failure - alarm) * frame_interval_s)
        result.detection_delays_s.append((alarm - onset) * frame_interval_s)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(
    scores: Sequence[float],
    labels: Sequence[int],
    predictions: Sequence[bool],
    events: Sequence[tuple[int, int]],
    frame_interval_s: float,
    consecutive: int = 3,
) -> dict[str, float]:
    """Every metric in this module, flattened into one reportable dict."""
    confusion = confusion_at_threshold(
        [1.0 if p else 0.0 for p in predictions], labels, threshold=0.5
    )
    timing = lead_time(predictions, events, frame_interval_s, consecutive)

    summary: dict[str, float] = {
        "roc_auc": roc_auc(scores, labels),
        "pauc_10": partial_roc_auc(scores, labels, max_fpr=0.1),
        "average_precision": average_precision(scores, labels),
        "false_alarms_per_hour": false_alarm_rate_per_hour(
            predictions, labels, frame_interval_s
        ),
        "frames": len(scores),
        "positive_rate": (sum(1 for v in labels if v) / len(labels)) if labels else 0.0,
    }
    summary.update(confusion.as_dict())
    summary.update(timing.as_dict())
    return summary


_INTEGER_KEYS = frozenset(
    {
        "frames",
        "events",
        "detected",
        "missed",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    }
)


def format_table(summary: dict[str, float], title: str = "Results") -> str:
    """Render a summary as a fixed-width block for terminals and CI logs."""
    width = max((len(k) for k in summary), default=0)
    lines = [title, "=" * max(len(title), width + 14)]
    for key, value in summary.items():
        if key in _INTEGER_KEYS:
            rendered = f"{int(value):d}"
        elif abs(value) >= 1000:
            rendered = f"{value:,.1f}"
        else:
            rendered = f"{value:.4f}"
        lines.append(f"{key:<{width}}  {rendered:>10}")
    return "\n".join(lines)
