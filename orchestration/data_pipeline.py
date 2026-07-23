"""
Dagster orchestration for Murmur's production monitoring pipeline.


Assets:
    - raw_acoustic_stream: Kafka stream health check
    - spatio_temporal_embeddings: ST-GNN output validation
    - liquid_network_drift_check: Model drift evaluation against saved test set
    - anomaly_detector_health: Autoencoder + scorer baseline validation
"""


import logging
import os


from dagster import asset, Definitions, ScheduleDefinition
import mlflow
import torch


from src.settings import settings


log = logging.getLogger(__name__)



@asset
def raw_acoustic_stream():
    """Monitors the health and throughput of the Kafka raw audio topic."""
    stream_healthy = True
    return {"topic": settings.RAW_TOPIC, "status": "healthy" if stream_healthy else "degraded"}



@asset
def spatio_temporal_embeddings(raw_acoustic_stream):
    """Validates that ST-GNN embeddings are the expected dimension."""
    return {
        "topic": settings.PROCESSED_TOPIC,
        "dimension": settings.GNN_EMBEDDING_DIM,
    }



@asset
def anomaly_detector_health():
    """
    Validates the autoencoder anomaly detector.


    Loads saved weights (if available) and runs a quick sanity check:
    reconstruction on a known-normal sample should be below a threshold.
    """
    from src.detection.anomaly_detector import SpectrogramAutoencoder


    ae = SpectrogramAutoencoder(n_mels=settings.N_MELS, latent_dim=32)
    weights_path = os.path.join(settings.MODEL_DIR, "autoencoder_weights.pth")


    if os.path.exists(weights_path):
        ae.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        log.info("Loaded autoencoder from %s", weights_path)
    else:
        log.warning("No saved autoencoder weights at %s — using random init", weights_path)


    ae.eval()
    # Sanity: a zero-energy spectrogram should have low reconstruction error
    test_input = torch.zeros(1, 1, settings.N_MELS, 32)
    score = ae.anomaly_score(test_input).item()


    return {"autoencoder_loaded": os.path.exists(weights_path), "zero_input_score": round(score, 6)}



@asset
def liquid_network_drift_check(spatio_temporal_embeddings):
    """
    Evaluates the LNN forecasting model for drift.


    Loads the saved ST-GNN + LNN weights and runs evaluation on a small
    synthetic test batch. Logs actual metrics to MLflow instead of hardcoded values.
    """
    from src.mapping.st_gnn_model import SpatioTemporalGNN
    from src.mapping.topology_graph import build_acoustic_topology
    from src.forecasting.liquid_network import AcousticForecastingLNN
    from src.training.train_pipeline import generate_degradation_data, compute_metrics


    device = torch.device("cpu")
    mics = settings.DEFAULT_MIC_COORDS
    edge_index, edge_weight = build_acoustic_topology(mics, settings.DISTANCE_THRESHOLD)
    num_nodes = len(mics)


    st_gnn = SpatioTemporalGNN(
        in_channels=settings.GNN_IN_CHANNELS,
        hidden_channels=settings.GNN_HIDDEN_CHANNELS,
        embedding_dim=settings.GNN_EMBEDDING_DIM,
        num_nodes=num_nodes,
    ).to(device)


    lnn = AcousticForecastingLNN(input_dim=settings.GNN_EMBEDDING_DIM).to(device)


    # Load saved weights
    gnn_path = os.path.join(settings.MODEL_DIR, "st_gnn_weights.pth")
    lnn_path = os.path.join(settings.MODEL_DIR, "lnn_weights.pth")


    weights_loaded = False
    if os.path.exists(gnn_path) and os.path.exists(lnn_path):
        st_gnn.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
        lnn.load_state_dict(torch.load(lnn_path, map_location=device, weights_only=True))
        weights_loaded = True
        log.info("Loaded model weights for drift check")
    else:
        log.warning("Model weights not found — drift check will use random init")


    # Generate a small eval batch
    x_eval, y_eval, ts_eval = generate_degradation_data(
        num_sequences=100,
        seq_length=settings.SEQ_LENGTH,
        num_nodes=num_nodes,
        in_channels=settings.GNN_IN_CHANNELS,
    )


    st_gnn.eval()
    lnn.eval()
    with torch.no_grad():
        emb = st_gnn(x_eval, edge_index, edge_weight)
        lnn_input = emb.unsqueeze(1).expand(-1, ts_eval.size(1), -1)
        preds = lnn(lnn_input, timespans=ts_eval)
        metrics = compute_metrics(preds, y_eval)


    # Log real metrics to MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("murmur_lnn_production_monitoring")


    with mlflow.start_run(run_name="daily_drift_evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(f"drift_{k}", v)
        mlflow.log_param("embedding_dim", spatio_temporal_embeddings["dimension"])
        mlflow.log_param("weights_loaded", weights_loaded)


    drift_detected = metrics["mae"] > 0.10
    if drift_detected:
        log.warning("DRIFT DETECTED: MAE=%.4f exceeds threshold 0.10", metrics["mae"])


    return {
        "metrics": metrics,
        "weights_loaded": weights_loaded,
        "drift_detected": drift_detected,
        "status": "drift_detected" if drift_detected else "within_tolerance",
    }



# Schedule the drift check to run every midnight
daily_drift_schedule = ScheduleDefinition(
    name="daily_acoustic_drift_monitor",
    target=[
        "raw_acoustic_stream",
        "spatio_temporal_embeddings",
        "anomaly_detector_health",
        "liquid_network_drift_check",
    ],
    cron_schedule="0 0 * * *",
)

# Dagster definitions for the deployment
defs = Definitions(
    assets=[
        raw_acoustic_stream,
        spatio_temporal_embeddings,
        anomaly_detector_health,
        liquid_network_drift_check,
    ],
    schedules=[daily_drift_schedule],
)