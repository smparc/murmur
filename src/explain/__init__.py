"""
Explaining anomaly scores.

An alert reading "node 3, score 0.87" gets ignored. An alert reading "node 3,
energy concentrated at 2.1-3.4 kHz" can be acted on. This package decomposes the
autoencoder's reconstruction error across frequency so the alert carries its own
evidence.
"""

from src.explain.saliency import (
    AnomalyExplanation,
    BandContribution,
    explain_anomaly,
    mel_bin_frequencies,
    reconstruction_error_map,
)

__all__ = [
    "AnomalyExplanation",
    "BandContribution",
    "explain_anomaly",
    "mel_bin_frequencies",
    "reconstruction_error_map",
]
