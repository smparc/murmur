"""
Prometheus metrics for the Murmur inference service.


Tracks request latency, throughput, error rates, anomaly counts,
and model inference timing — essential for production observability.
"""


import time
from functools import wraps


from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest



# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------


# Request-level metrics
REQUEST_COUNT = Counter(
    "murmur_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)


REQUEST_LATENCY = Histogram(
    "murmur_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


# Model-level metrics
MODEL_INFERENCE_TIME = Summary(
    "murmur_model_inference_seconds",
    "Model inference time",
    ["model"],
)


# Anomaly detection metrics
ANOMALY_COUNT = Counter(
    "murmur_anomalies_total",
    "Total anomalies detected",
    ["node_id", "severity"],
)


ANOMALY_SCORE = Gauge(
    "murmur_anomaly_score",
    "Latest anomaly score per node",
    ["node_id"],
)


TTF_PREDICTION = Gauge(
    "murmur_ttf_prediction",
    "Latest TTF prediction per node (0=healthy, 1=imminent failure)",
    ["node_id"],
)


# System metrics
ACTIVE_WS_CLIENTS = Gauge(
    "murmur_active_websocket_clients",
    "Number of active WebSocket connections",
)


FRAMES_PROCESSED = Counter(
    "murmur_frames_processed_total",
    "Total spectrogram frames processed",
    ["node_id"],
)


KAFKA_CONSUMER_LAG = Gauge(
    "murmur_kafka_consumer_lag",
    "Kafka consumer lag (estimated)",
    ["topic"],
)



# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def track_latency(endpoint: str):
    """Decorator to track endpoint latency."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
                return result
            except Exception:
                REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(
                    time.perf_counter() - start
                )
        return wrapper
    return decorator



def track_inference(model_name: str):
    """Context manager for tracking model inference time."""
    return MODEL_INFERENCE_TIME.labels(model=model_name).time()