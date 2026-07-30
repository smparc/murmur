"""
Dagster orchestration for Murmur's production monitoring pipeline.

Assets
------
``acoustic_topology``
    Validates the configured microphone geometry produces a usable graph.
``raw_acoustic_stream``
    Kafka broker and topic reachability.
``anomaly_detector_health``
    Autoencoder weights load and behave sanely on a known input.
``liquid_network_drift_check``
    Evaluates the saved forecaster against a fresh synthetic sample and
    compares it to a mean-predictor baseline.

Every asset returns a ``MaterializeResult`` carrying metadata, so the Dagster
UI shows the actual numbers rather than only a green tick.
"""

from __future__ import annotations

import logging
import os

import torch
from dagster import (
    AssetSelection,
    Definitions,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)

from src.settings import settings

log = logging.getLogger(__name__)

# Drift threshold: the forecaster should stay well inside the error a
# mean-predictor would achieve. Absolute MAE alone is not meaningful without
# that reference point.
DRIFT_MAE_THRESHOLD = 0.10


@asset(description="Validates the configured microphone geometry.")
def acoustic_topology() -> MaterializeResult:
    from src.mapping.topology_graph import build_acoustic_topology, topology_summary

    edge_index, edge_weight = build_acoustic_topology(
        settings.MIC_COORDS,
        settings.DISTANCE_THRESHOLD,
        decay_exponent=settings.DISTANCE_DECAY_EXPONENT,
    )
    summary = topology_summary(settings.MIC_COORDS, settings.DISTANCE_THRESHOLD)

    isolated = int(summary["isolated_nodes"])
    if isolated:
        # An isolated microphone contributes nothing through the GCN — its
        # readings reach the embedding only via global pooling.
        log.warning(
            "%d microphone(s) have no acoustic neighbours; consider raising "
            "DISTANCE_THRESHOLD or repositioning them.",
            isolated,
        )

    return MaterializeResult(
        metadata={
            "num_nodes": summary["num_nodes"],
            "num_edges": summary["num_edges"],
            "density": MetadataValue.float(round(summary["density"], 4)),
            "mean_degree": MetadataValue.float(round(summary["mean_degree"], 3)),
            "isolated_nodes": isolated,
            "edge_index_shape": str(tuple(edge_index.shape)),
            "mean_edge_weight": MetadataValue.float(round(float(edge_weight.mean()), 4))
            if edge_weight.numel()
            else 0.0,
        }
    )


@asset(description="Checks the Kafka broker is reachable and topics exist.")
def raw_acoustic_stream() -> MaterializeResult:
    reachable = False
    topics: list[str] = []
    detail = ""

    try:
        from confluent_kafka.admin import AdminClient

        admin = AdminClient({"bootstrap.servers": settings.KAFKA_BROKER})
        metadata = admin.list_topics(timeout=5.0)
        topics = sorted(metadata.topics)
        reachable = True
    except Exception as exc:  # pragma: no cover - depends on live infrastructure
        detail = str(exc)
        log.warning("Kafka unreachable at %s: %s", settings.KAFKA_BROKER, detail)

    expected = [settings.RAW_TOPIC, settings.PROCESSED_TOPIC, settings.WINDOWED_TOPIC]
    return MaterializeResult(
        metadata={
            "broker": settings.KAFKA_BROKER,
            "reachable": reachable,
            "status": "healthy" if reachable else "unreachable",
            "expected_topics": MetadataValue.json(expected),
            "missing_topics": MetadataValue.json(
                [t for t in expected if t not in topics] if reachable else expected
            ),
            "detail": detail,
        }
    )


@asset(description="Validates the autoencoder anomaly detector.")
def anomaly_detector_health() -> MaterializeResult:
    from src.detection.anomaly_detector import AnomalyScorer, SpectrogramAutoencoder

    autoencoder = SpectrogramAutoencoder(
        n_mels=settings.N_MELS, latent_dim=settings.AE_LATENT_DIM
    )
    weights_path = os.path.join(settings.MODEL_DIR, "autoencoder_weights.pth")
    loaded = os.path.exists(weights_path)

    if loaded:
        autoencoder.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded autoencoder from %s", weights_path)
    else:
        log.warning("No autoencoder weights at %s — using random init", weights_path)

    autoencoder.eval()
    frames = settings.MEL_FRAMES_PER_CHUNK

    quiet = torch.zeros(1, 1, settings.N_MELS, frames)
    loud = torch.randn(1, 1, settings.N_MELS, frames) * 10.0
    quiet_score = float(autoencoder.anomaly_score(quiet).item())
    loud_score = float(autoencoder.anomaly_score(loud).item())

    # A detector that does not rank a violent signal above silence is not
    # discriminating at all, whatever its absolute loss happens to be.
    ordering_ok = loud_score > quiet_score

    scorer = AnomalyScorer(
        autoencoder=autoencoder,
        num_nodes=settings.NUM_NODES,
        warmup_frames=5,
        z_threshold=settings.ANOMALY_Z_THRESHOLD,
        window=settings.ANOMALY_WINDOW,
    )
    for _ in range(10):
        scorer.score(0, torch.randn(1, 1, settings.N_MELS, frames) * 0.1)
    spike = scorer.score(0, torch.randn(1, 1, settings.N_MELS, frames) * 50.0)

    if not ordering_ok:
        log.error("Autoencoder scores silence above a loud transient — retraining required")

    return MaterializeResult(
        metadata={
            "autoencoder_loaded": loaded,
            "quiet_input_score": MetadataValue.float(round(quiet_score, 6)),
            "loud_input_score": MetadataValue.float(round(loud_score, 6)),
            "ordering_correct": ordering_ok,
            "spike_detected": spike.is_anomaly,
            "spike_z_score": MetadataValue.float(round(spike.z_score, 3)),
            "status": "healthy" if ordering_ok else "degraded",
        }
    )


@asset(
    deps=[acoustic_topology],
    description="Evaluates the saved forecaster for drift against a baseline.",
)
def liquid_network_drift_check() -> MaterializeResult:
    from src.forecasting.liquid_network import AcousticForecastingLNN
    from src.mapping.st_gnn_model import SpatioTemporalGNN
    from src.mapping.topology_graph import build_acoustic_topology
    from src.training.train_pipeline import compute_metrics, generate_degradation_data

    device = torch.device("cpu")
    mics = settings.MIC_COORDS
    num_nodes = len(mics)
    edge_index, edge_weight = build_acoustic_topology(
        mics, settings.DISTANCE_THRESHOLD, decay_exponent=settings.DISTANCE_DECAY_EXPONENT
    )

    st_gnn = SpatioTemporalGNN(
        in_channels=settings.GNN_IN_CHANNELS,
        hidden_channels=settings.GNN_HIDDEN_CHANNELS,
        embedding_dim=settings.GNN_EMBEDDING_DIM,
        num_nodes=num_nodes,
        num_heads=settings.GNN_NUM_HEADS,
    ).to(device)
    lnn = AcousticForecastingLNN(
        input_dim=settings.GNN_EMBEDDING_DIM,
        hidden_neurons=settings.LNN_HIDDEN_NEURONS,
    ).to(device)

    gnn_path = os.path.join(settings.MODEL_DIR, "st_gnn_weights.pth")
    lnn_path = os.path.join(settings.MODEL_DIR, "lnn_weights.pth")
    weights_loaded = os.path.exists(gnn_path) and os.path.exists(lnn_path)

    if weights_loaded:
        st_gnn.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
        lnn.load_state_dict(torch.load(lnn_path, map_location=device, weights_only=True))
        log.info("Loaded model weights for drift check")
    else:
        log.warning("Model weights not found — drift check runs on random init")

    # A fresh seed each run: reusing the training seed would measure memorisation.
    x_eval, y_eval, ts_eval = generate_degradation_data(
        num_sequences=100,
        seq_length=settings.SEQ_LENGTH,
        num_nodes=num_nodes,
        in_channels=settings.GNN_IN_CHANNELS,
        anomaly_ratio=0.30,
        mic_coords=mics,
    )

    st_gnn.eval()
    lnn.eval()
    with torch.no_grad():
        sequence = st_gnn(x_eval, edge_index, edge_weight, return_sequence=True)
        preds = lnn(sequence, timespans=ts_eval)
        metrics = compute_metrics(preds, y_eval)
        baseline = compute_metrics(torch.full_like(y_eval, float(y_eval.mean())), y_eval)

    drift_detected = weights_loaded and metrics["mae"] > DRIFT_MAE_THRESHOLD
    beats_baseline = metrics["mae"] < baseline["mae"]

    if drift_detected:
        log.warning(
            "DRIFT DETECTED: MAE=%.4f exceeds threshold %.2f", metrics["mae"], DRIFT_MAE_THRESHOLD
        )

    try:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("murmur_lnn_production_monitoring")
        with mlflow.start_run(run_name="daily_drift_evaluation"):
            mlflow.log_metrics({f"drift_{k}": v for k, v in metrics.items()})
            mlflow.log_metric("baseline_mae", baseline["mae"])
            mlflow.log_param("weights_loaded", weights_loaded)
    except Exception:
        log.warning("Could not log drift metrics to MLflow", exc_info=True)

    return MaterializeResult(
        metadata={
            **{k: MetadataValue.float(v) for k, v in metrics.items()},
            "baseline_mae": MetadataValue.float(baseline["mae"]),
            "beats_baseline": beats_baseline,
            "weights_loaded": weights_loaded,
            "drift_detected": drift_detected,
            "status": "drift_detected" if drift_detected else "within_tolerance",
        }
    )


monitoring_job = define_asset_job(
    name="murmur_daily_monitoring",
    selection=AssetSelection.all(),
)

daily_drift_schedule = ScheduleDefinition(
    name="daily_acoustic_drift_monitor",
    job=monitoring_job,
    cron_schedule="0 0 * * *",
)

defs = Definitions(
    assets=[
        acoustic_topology,
        raw_acoustic_stream,
        anomaly_detector_health,
        liquid_network_drift_check,
    ],
    jobs=[monitoring_job],
    schedules=[daily_drift_schedule],
)
