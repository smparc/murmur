"""Observability: Prometheus metrics and structured logging."""
from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ANOMALY_COUNT,
    ANOMALY_SCORE,
    TTF_PREDICTION,
    ACTIVE_WS_CLIENTS,
    FRAMES_PROCESSED,
    track_latency,
    track_inference,
)


__all__ = [
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "ANOMALY_COUNT",
    "ANOMALY_SCORE",
    "TTF_PREDICTION",
    "ACTIVE_WS_CLIENTS",
    "FRAMES_PROCESSED",
    "track_latency",
    "track_inference",
]
