"""
Detection metrics for anomalous-sound benchmarks.

Implemented directly rather than pulled from scikit-learn: the project has no
scientific-Python dependency beyond NumPy/SciPy, and adding one for two rank
statistics would put an extra ~100 MB into two GPU container images.

The metric that matters here is **partial** AUC. Full AUC integrates over the
entire false-positive range, including the region where the detector fires on
90% of healthy machines — an operating point no factory would ever run. A
detector can post a respectable AUC while being useless in the only regime it
would actually be deployed in. DCASE's anomalous-sound task reports pAUC at
``p = 0.1`` for exactly this reason, and so does this module.
"""

from __future__ import annotations

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-indexed, ties shared. Equivalent to scipy's 'average'."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)

    # Average the ranks within each run of equal values.
    sorted_values = values[order]
    start = 0
    for end in range(1, values.size + 1):
        if end == values.size or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Area under the ROC curve, via the Mann-Whitney U statistic.

    ``labels`` is 1 for anomalous, 0 for normal. Ties contribute 0.5, which is
    what makes this exact rather than an approximation of the curve.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must match")

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative example")

    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr)`` at every distinct threshold, both starting at 0."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    # Collapse tied scores to a single operating point.
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, labels.size - 1]

    n_pos, n_neg = tps[-1], fps[-1]
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs at least one positive and one negative example")

    return np.r_[0.0, fps[idx] / n_neg], np.r_[0.0, tps[idx] / n_pos]


def partial_auc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.1) -> float:
    """
    AUC restricted to ``fpr <= max_fpr``, divided by ``max_fpr``.

    Normalisation convention
    ------------------------
    Dividing the partial area by its width makes this the **mean true-positive
    rate over the false-positive range [0, max_fpr]** — that is, average recall
    in the only alert regime a plant would tolerate. A perfect detector scores
    1.0; a detector that catches nothing before exceeding the false-alarm budget
    scores 0.0; a chance detector scores ``max_fpr / 2`` (0.05 at the default),
    because chance TPR equals FPR across that strip.

    Note that this is *not* the McClish standardisation, which rescales the same
    partial area onto [0.5, 1] so that chance reads 0.5. McClish is preferable
    when comparing against full AUC on one axis, but it compresses the entire
    useful range into the top half and — because its floor for a zero-recall
    detector is ~0.47 rather than 0 — makes a detector that is completely blind
    at low FPR nearly indistinguishable from an average one. Mean recall is the
    more honest number for a monitoring system, so it is what this returns.
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")

    fpr, tpr = roc_curve(scores, labels)

    # Interpolate the curve exactly at the cutoff so the area is not truncated
    # to whichever threshold happens to sit nearest.
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    fpr_c = np.r_[fpr[:stop], max_fpr]
    if stop == 0:
        tpr_c = np.r_[tpr[0], tpr[0]]
        fpr_c = np.r_[0.0, max_fpr]
    elif stop < len(fpr):
        span = fpr[stop] - fpr[stop - 1]
        frac = 0.0 if span == 0 else (max_fpr - fpr[stop - 1]) / span
        tpr_c = np.r_[tpr[:stop], tpr[stop - 1] + frac * (tpr[stop] - tpr[stop - 1])]
    else:
        tpr_c = np.r_[tpr[:stop], tpr[-1]]

    return float(np.trapezoid(tpr_c, fpr_c) / max_fpr)


def detection_report(
    scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.1
) -> dict[str, float]:
    """Full metric bundle for one machine or one aggregate."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    return {
        "auc": roc_auc(scores, labels),
        "pauc": partial_auc(scores, labels, max_fpr),
        "max_fpr": max_fpr,
        "n_normal": int((labels == 0).sum()),
        "n_anomalous": int((labels == 1).sum()),
    }
