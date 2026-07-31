"""
Evaluation against recorded machine sound.

Synthetic benchmarks measure whether a model can invert the generator that made
them. This package measures whether it detects real faults, using the same
feature transform the production pipeline uses.
"""

from src.evaluation.metrics import detection_report, partial_auc, roc_auc, roc_curve

__all__ = ["detection_report", "partial_auc", "roc_auc", "roc_curve"]
