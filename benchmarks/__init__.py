"""
Reproducible evaluation for Murmur.

The package exists to answer one question the rest of the repository cannot:
*does any of this actually work?* Every number Murmur reports at runtime —
anomaly scores, TTF probabilities, severity labels — is produced by a model
that, until something scores it against known truth, has no demonstrated skill
at all.

Three entry points, in increasing order of how much they prove:

``benchmarks.evaluate_synthetic``
    Scores the detector against the simulator's own ground truth. Cheap, fully
    deterministic, no downloads. Catches regressions.

``benchmarks.evaluate_dataset``
    Scores the detector against real recorded machine audio (MIMII, DCASE, IMS)
    and reports AUC / pAUC next to published baselines.

``benchmarks.perf``
    Measures the throughput and latency claims made in the README instead of
    asserting them.
"""

from benchmarks.metrics import (
    DetectionMetrics,
    LeadTimeMetrics,
    average_precision,
    partial_roc_auc,
    roc_auc,
)
from benchmarks.scenario import FaultEvent, Scenario, ScenarioConfig, generate_scenario

__all__ = [
    "DetectionMetrics",
    "FaultEvent",
    "LeadTimeMetrics",
    "Scenario",
    "ScenarioConfig",
    "average_precision",
    "generate_scenario",
    "partial_roc_auc",
    "roc_auc",
]
