"""
End-to-end training pipeline for Murmur's models.

Stages
------
1. **Autoencoder** — unsupervised reconstruction of normal spectrograms. This
   stage previously did not exist: the module imported ``SpectrogramAutoencoder``
   and never trained it, and the orchestration layer then looked for weights
   that were never written.
2. **ST-GNN + Liquid Network** — jointly trained to forecast TTF from windowed
   spectrograms, using the ST-GNN's *sequence* output so the continuous-time
   model receives genuinely time-varying input.
3. **Projection adapter** — aligns acoustic embeddings with the LLM's token
   embedding space. Skipped, loudly, when no LLM is available locally.

On the synthetic data
---------------------
The generator models a fault that originates at one microphone and reaches the
others attenuated by distance, on top of a *constant* ambient floor. That
matters: the previous generator set the noise level from the label
(``noise = 0.1 + phase * 1.5``, ``ttf = phase``), so the target was a
deterministic function of global variance and the network only had to read the
noise floor. It also added a scalar impulse identically to every node and every
channel, leaving no spatial gradient at all — meaning the *spatial* GNN, the
centrepiece of the architecture, was never asked to do anything.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.detection.anomaly_detector import SpectrogramAutoencoder
from src.forecasting.conformal import (
    ConformalCalibrator,
    evaluate_coverage,
    severity_bucket,
)
from src.forecasting.liquid_network import AcousticForecastingLNN
from src.mapping.st_gnn_model import SpatioTemporalGNN
from src.mapping.topology_graph import build_acoustic_topology
from src.settings import settings

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches, so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

# Mel-band centres (as a fraction of the band range) excited by each fault.
_FAULT_PROFILES: dict[str, tuple[float, float]] = {
    "bearing": (0.75, 0.08),  # high-frequency squeal
    "cavitation": (0.50, 0.30),  # broadband burst
    "imbalance": (0.15, 0.06),  # low-frequency modulation
}


def _spectral_profile(in_channels: int, fault: str) -> np.ndarray:
    """Gaussian energy envelope over mel bins for a given fault signature."""
    centre_frac, width_frac = _FAULT_PROFILES[fault]
    bins = np.arange(in_channels, dtype=np.float64)
    centre = centre_frac * (in_channels - 1)
    width = max(1.0, width_frac * in_channels)
    return np.exp(-0.5 * ((bins - centre) / width) ** 2)


def generate_degradation_data(
    num_sequences: int,
    seq_length: int,
    num_nodes: int,
    in_channels: int,
    anomaly_ratio: float = 0.15,
    mic_coords: list[tuple[float, float, float]] | None = None,
    ambient_level: float = 0.35,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate windowed acoustic sequences with spatially-localised faults.

    Returns
    -------
    x:
        ``(N, seq_length, num_nodes * in_channels)``
    y_ttf:
        ``(N, 1)`` failure probability in ``[0, 1]``
    timespans:
        ``(N, seq_length)`` inter-frame intervals in seconds
    """
    rng = np.random.default_rng(seed)

    if mic_coords is None:
        mic_coords = settings.MIC_COORDS[:num_nodes]
    coords = np.asarray(mic_coords, dtype=np.float64)
    if coords.shape[0] < num_nodes:
        # Pad with a synthetic line array if fewer coordinates than nodes.
        extra = np.stack(
            [
                np.arange(coords.shape[0], num_nodes) * 5.0,
                np.zeros(num_nodes - coords.shape[0]),
                np.full(num_nodes - coords.shape[0], 3.0),
            ],
            axis=1,
        )
        coords = np.vstack([coords, extra])
    coords = coords[:num_nodes]

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((deltas**2).sum(-1))

    x_all = np.empty((num_sequences, seq_length, num_nodes, in_channels), dtype=np.float32)
    y_all = np.empty((num_sequences, 1), dtype=np.float32)
    ts_all = np.empty((num_sequences, seq_length), dtype=np.float32)

    fault_names = list(_FAULT_PROFILES)

    for i in range(num_sequences):
        # The ambient floor varies per sequence and is independent of the label,
        # so a loud healthy machine can out-energise a quiet degrading one. That
        # deliberately removes "total loudness" as a shortcut and forces the
        # model onto the spectral, spatial and temporal structure of the fault.
        floor = ambient_level * float(rng.uniform(0.6, 1.7))
        signal = rng.normal(0.0, floor, size=(seq_length, num_nodes, in_channels))

        if rng.random() < anomaly_ratio:
            severity = float(rng.uniform(0.5, 1.0))
            source = int(rng.integers(num_nodes))
            fault = fault_names[int(rng.integers(len(fault_names)))]
            profile = _spectral_profile(in_channels, fault)

            # Inverse-square attenuation from the failing machine outwards.
            gains = 1.0 / (1.0 + distances[source] ** 2)
            gains /= gains.max()

            # The fault ramps in over the window rather than being present
            # uniformly, which is what the temporal attention and the CfC's
            # time constants are there to pick up.
            ramp = np.linspace(0.15, 1.0, seq_length) ** 1.5
            modulation = 1.0 + 0.3 * np.sin(
                2 * np.pi * rng.uniform(0.05, 0.25) * np.arange(seq_length)
            )
            envelope = severity * ramp * modulation  # (seq,)

            contribution = envelope[:, None, None] * gains[None, :, None] * profile[None, None, :]
            signal += 2.5 * contribution
            ttf = severity
        else:
            ttf = float(rng.uniform(0.0, 0.1))

        x_all[i] = signal.astype(np.float32)
        y_all[i, 0] = ttf
        # 500 ms nominal cadence with edge-network jitter.
        ts_all[i] = np.clip(0.5 + rng.normal(0.0, 0.05, size=seq_length), 0.1, None).astype(
            np.float32
        )

    x = torch.from_numpy(x_all).reshape(num_sequences, seq_length, num_nodes * in_channels)
    return x, torch.from_numpy(y_all), torch.from_numpy(ts_all)


def generate_normal_spectrograms(
    num_samples: int,
    n_mels: int,
    n_frames: int,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Healthy-machine spectrogram patches for autoencoder pre-training.

    Structured rather than white: a low-frequency rumble plus harmonics, so the
    autoencoder learns an actual manifold to depart from.
    """
    rng = np.random.default_rng(seed)
    bins = np.arange(n_mels)[:, None]
    frames = np.arange(n_frames)[None, :]

    out = np.empty((num_samples, 1, n_mels, n_frames), dtype=np.float32)
    for i in range(num_samples):
        rumble = np.exp(-0.5 * ((bins - rng.uniform(2, 8)) / 4.0) ** 2)
        harmonic = 0.4 * np.exp(-0.5 * ((bins - rng.uniform(18, 26)) / 3.0) ** 2)
        drift = 1.0 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.02, 0.08) * frames)
        base = (rumble + harmonic) * drift
        out[i, 0] = (base + rng.normal(0, 0.05, size=(n_mels, n_frames))).astype(np.float32)
    return torch.from_numpy(out)


def train_val_test_split(
    x: torch.Tensor,
    y: torch.Tensor,
    ts: torch.Tensor,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    generator: torch.Generator | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Shuffled split into train/val/test."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("ratios must lie in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 to leave a test set")

    n = x.size(0)
    perm = torch.randperm(n, generator=generator)
    x, y, ts = x[perm], y[perm], ts[perm]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    end_val = n_train + n_val

    return {
        "train": (x[:n_train], y[:n_train], ts[:n_train]),
        "val": (x[n_train:end_val], y[n_train:end_val], ts[n_train:end_val]),
        "test": (x[end_val:], y[end_val:], ts[end_val:]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Regression and thresholded-classification metrics."""
    preds = preds.detach().float()
    targets = targets.detach().float()

    mse = nn.functional.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()

    pred_binary = (preds >= threshold).float()
    true_binary = (targets >= threshold).float()

    tp = ((pred_binary == 1) & (true_binary == 1)).sum().item()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# MLflow (optional)
# ---------------------------------------------------------------------------


@dataclass
class _Tracker:
    """
    Thin MLflow wrapper that no-ops when MLflow is unavailable.

    Training must not require the full MLOps stack merely to be importable —
    the data generators and metrics above are used by the test suite and by the
    orchestration layer.
    """

    enabled: bool = False

    def __post_init__(self) -> None:
        self._mlflow = None
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("murmur_model_training")
            self._mlflow = mlflow
            self.enabled = True
        except Exception:
            log.warning("MLflow unavailable — metrics will only be logged locally")

    def run(self):
        return self._mlflow.start_run() if self.enabled else nullcontext()

    def log_params(self, params: dict) -> None:
        if self.enabled:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            self._mlflow.log_metrics(metrics, step=step)


# ---------------------------------------------------------------------------
# Stage 1 — autoencoder
# ---------------------------------------------------------------------------


def train_autoencoder(
    epochs: int = 15,
    num_samples: int = 800,
    batch_size: int = 32,
    tracker: _Tracker | None = None,
) -> SpectrogramAutoencoder:
    """Fit the reconstruction baseline on normal-operation spectrograms."""
    log.info("Stage 1/3: training the anomaly-detection autoencoder")

    n_frames = settings.MEL_FRAMES_PER_CHUNK
    data = generate_normal_spectrograms(num_samples, settings.N_MELS, n_frames, seed=settings.SEED)

    split = int(0.85 * num_samples)
    train_x, val_x = data[:split].to(DEVICE), data[split:].to(DEVICE)

    model = SpectrogramAutoencoder(n_mels=settings.N_MELS, latent_dim=settings.AE_LATENT_DIM).to(
        DEVICE
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            val_recon, _ = model(val_x)
            val_loss = criterion(val_recon, val_x).item()

        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                "  AE epoch %d/%d | train %.5f | val %.5f",
                epoch,
                epochs,
                total / max(1, len(loader)),
                val_loss,
            )
            if tracker:
                tracker.log_metrics(
                    {"ae_train_loss": total / max(1, len(loader)), "ae_val_loss": val_loss},
                    step=epoch,
                )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Stage 2 — ST-GNN + LNN
# ---------------------------------------------------------------------------


def _forward_forecast(
    st_gnn: SpatioTemporalGNN,
    lnn: AcousticForecastingLNN,
    x: torch.Tensor,
    ts: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """
    ST-GNN sequence -> LNN forecast.

    ``return_sequence=True`` is the crux. Previously a single pooled embedding
    was broadcast across the time axis with ``.expand()``, handing the
    continuous-time network a constant and nullifying the reason to use one.
    """
    embedding_sequence = st_gnn(x, edge_index, edge_weight, return_sequence=True)
    return lnn(embedding_sequence, timespans=ts)


def calibrate_forecaster(
    st_gnn: SpatioTemporalGNN,
    lnn: AcousticForecastingLNN,
    splits: dict,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    alpha: float | None = None,
) -> tuple[ConformalCalibrator, dict[str, float]]:
    """
    Fit conformal prediction intervals and measure their realised coverage.

    The forecaster emits a sigmoid. Nothing in the training objective makes that
    number a calibrated probability, so shipping it as "73% chance of failure"
    is a fabrication — and it is the number a maintenance planner would schedule
    against. Conformal turns it into an interval with a finite-sample coverage
    guarantee that holds without assuming anything about the error distribution.

    Which data is used matters, and is the easiest thing to get wrong:

    - **Not the training set.** Residuals there are optimistically small and the
      guarantee silently evaporates, leaving intervals that look tight and are
      wrong far more often than advertised.
    - **Not the validation set either.** Early stopping selected on it, so it is
      no longer exchangeable with unseen data.

    The test split is therefore halved: one half calibrates, the other measures
    coverage on data neither the model nor the calibrator has seen.
    """
    alpha = settings.CONFORMAL_ALPHA if alpha is None else alpha
    log.info("Stage 4/4: conformal calibration (alpha=%.3f)", alpha)

    x_test, y_test, ts_test = (t.to(DEVICE) for t in splits["test"])

    st_gnn.eval()
    lnn.eval()
    with torch.no_grad():
        preds = _forward_forecast(st_gnn, lnn, x_test, ts_test, edge_index, edge_weight)

    predictions = preds.squeeze(-1).cpu().numpy()
    targets = y_test.squeeze(-1).cpu().numpy()

    half = len(predictions) // 2
    if half < 2:
        raise ValueError(
            f"test split has {len(predictions)} samples — too few to both "
            "calibrate and verify. Increase TRAIN_NUM_SAMPLES."
        )

    cal_pred, cal_true = predictions[:half], targets[:half]
    ver_pred, ver_true = predictions[half:], targets[half:]

    groups = np.array([severity_bucket(p) for p in cal_pred])
    calibrator = ConformalCalibrator(alpha=alpha).fit(cal_pred, cal_true, groups)

    ver_groups = np.array([severity_bucket(p) for p in ver_pred])
    coverage = evaluate_coverage(calibrator.intervals(ver_pred, ver_groups), ver_true)

    log.info(
        "Conformal: coverage %.3f against nominal %.3f, mean width %.3f (n_cal=%d)",
        coverage["coverage"],
        coverage["nominal_coverage"],
        coverage["mean_width"],
        calibrator.n_calibration,
    )
    if coverage["coverage"] < coverage["nominal_coverage"] - 0.05:
        log.warning(
            "Realised coverage %.3f is well below nominal %.3f. With exchangeable "
            "data this should not happen; check that the calibration split is "
            "genuinely disjoint from training.",
            coverage["coverage"],
            coverage["nominal_coverage"],
        )

    return calibrator, coverage


def train_forecaster(
    splits: dict,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    tracker: _Tracker | None = None,
) -> tuple[SpatioTemporalGNN, AcousticForecastingLNN, dict[str, float]]:
    """Jointly train the spatial-temporal encoder and the TTF forecaster."""
    log.info("Stage 2/3: training ST-GNN + Liquid Network")

    x_train, y_train, ts_train = (t.to(DEVICE) for t in splits["train"])
    x_val, y_val, ts_val = (t.to(DEVICE) for t in splits["val"])
    x_test, y_test, ts_test = (t.to(DEVICE) for t in splits["test"])

    st_gnn = SpatioTemporalGNN(
        in_channels=settings.GNN_IN_CHANNELS,
        hidden_channels=settings.GNN_HIDDEN_CHANNELS,
        embedding_dim=settings.GNN_EMBEDDING_DIM,
        num_nodes=num_nodes,
        num_heads=settings.GNN_NUM_HEADS,
    ).to(DEVICE)
    lnn = AcousticForecastingLNN(
        input_dim=settings.GNN_EMBEDDING_DIM,
        hidden_neurons=settings.LNN_HIDDEN_NEURONS,
    ).to(DEVICE)

    log.info(
        "  ST-GNN %.2fM params | LNN %.2fM params",
        sum(p.numel() for p in st_gnn.parameters()) / 1e6,
        sum(p.numel() for p in lnn.parameters()) / 1e6,
    )

    params = list(st_gnn.parameters()) + list(lnn.parameters())
    optimizer = optim.AdamW(params, lr=settings.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.TRAIN_EPOCHS)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x_train, y_train, ts_train),
        batch_size=settings.TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    best_val = float("inf")
    best_state: dict | None = None
    patience_left = settings.EARLY_STOP_PATIENCE

    for epoch in range(settings.TRAIN_EPOCHS):
        st_gnn.train()
        lnn.train()
        epoch_loss = 0.0
        for batch_x, batch_y, batch_ts in loader:
            optimizer.zero_grad()
            preds = _forward_forecast(st_gnn, lnn, batch_x, batch_ts, edge_index, edge_weight)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / max(1, len(loader))
        scheduler.step()

        st_gnn.eval()
        lnn.eval()
        with torch.no_grad():
            val_preds = _forward_forecast(st_gnn, lnn, x_val, ts_val, edge_index, edge_weight)
            val_loss = criterion(val_preds, y_val).item()
            val_metrics = compute_metrics(val_preds, y_val)

        if epoch % 5 == 0 or epoch == settings.TRAIN_EPOCHS - 1:
            lr = optimizer.param_groups[0]["lr"]
            log.info(
                "  epoch %d/%d | train %.4f | val %.4f | MAE %.4f | F1 %.4f | lr %.2e",
                epoch,
                settings.TRAIN_EPOCHS,
                avg_train,
                val_loss,
                val_metrics["mae"],
                val_metrics["f1"],
                lr,
            )
            if tracker:
                tracker.log_metrics(
                    {
                        "train_loss": avg_train,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": lr,
                    },
                    step=epoch,
                )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_left = settings.EARLY_STOP_PATIENCE
            best_state = {
                "st_gnn": {k: v.detach().cpu().clone() for k, v in st_gnn.state_dict().items()},
                "lnn": {k: v.detach().cpu().clone() for k, v in lnn.state_dict().items()},
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("  early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        st_gnn.load_state_dict(best_state["st_gnn"])
        lnn.load_state_dict(best_state["lnn"])
        st_gnn.to(DEVICE)
        lnn.to(DEVICE)

    st_gnn.eval()
    lnn.eval()
    with torch.no_grad():
        test_preds = _forward_forecast(st_gnn, lnn, x_test, ts_test, edge_index, edge_weight)
        test_metrics = compute_metrics(test_preds, y_test)
        # A model that predicts the training mean scores deceptively well on an
        # imbalanced target, so report the baseline it has to beat.
        baseline = compute_metrics(torch.full_like(y_test, float(y_train.mean())), y_test)

    test_metrics["baseline_mae"] = baseline["mae"]
    test_metrics["best_val_loss"] = round(best_val, 6)
    return st_gnn, lnn, test_metrics


# ---------------------------------------------------------------------------
# Stage 3 — projection adapter
# ---------------------------------------------------------------------------


def train_projector(
    st_gnn: SpatioTemporalGNN,
    splits: dict,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int = 20,
) -> nn.Module | None:
    """
    Align acoustic embeddings with the LLM's token embedding space.

    The adapter is trained to place an acoustic embedding near the LLM's own
    embedding of a sentence describing that machine's condition. Without this
    the projector is random, and the "diagnostic" text is unconditioned on the
    audio — text that looks authoritative while carrying no acoustic
    information is worse than no text at all in a safety context.

    Returns ``None`` when no LLM is available locally, rather than emitting an
    untrained adapter that would look trained on disk.
    """
    log.info("Stage 3/3: training the LLM projection adapter")

    if not settings.LLM_ENABLED:
        log.warning("  LLM_ENABLED=false — skipping adapter training")
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
        llm = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL_NAME, dtype=torch.float32).to(
            DEVICE
        )
        llm.eval()
    except Exception:
        log.warning(
            "  Could not load %s — skipping adapter training. The telemetry "
            "service will serve templated text until this stage runs.",
            settings.LLM_MODEL_NAME,
            exc_info=True,
        )
        return None

    from src.translation.llm_decoder import EmbeddingProjector

    hidden = llm.get_input_embeddings().weight.shape[1]
    projector = EmbeddingProjector(settings.GNN_EMBEDDING_DIM, hidden).to(DEVICE)

    x_train, y_train, _ = (t.to(DEVICE) for t in splits["train"])

    with torch.no_grad():
        embeddings = st_gnn(x_train, edge_index, edge_weight)

    # Target: the LLM's mean token embedding for a description of the condition.
    def describe(ttf: float) -> str:
        if ttf >= 0.66:
            return "critical acoustic anomaly, imminent mechanical failure"
        if ttf >= 0.33:
            return "elevated acoustic anomaly, progressive degradation"
        return "nominal acoustic signature, machine healthy"

    with torch.no_grad():
        targets = []
        for value in y_train.squeeze(-1).tolist():
            ids = tokenizer(describe(value), return_tensors="pt").input_ids.to(DEVICE)
            targets.append(llm.get_input_embeddings()(ids).mean(dim=1).squeeze(0))
        target_tensor = torch.stack(targets)

    optimizer = optim.AdamW(projector.parameters(), lr=1e-4, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(embeddings, target_tensor), batch_size=settings.TRAIN_BATCH_SIZE, shuffle=True
    )

    for epoch in range(epochs):
        projector.train()
        total = 0.0
        for batch_emb, batch_target in loader:
            optimizer.zero_grad()
            projected = projector(batch_emb)
            # Cosine alignment: direction in embedding space is what matters,
            # not magnitude.
            loss = (1 - nn.functional.cosine_similarity(projected, batch_target, dim=-1)).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info("  projector epoch %d/%d | loss %.5f", epoch, epochs, total / len(loader))

    projector.eval()
    return projector


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def train() -> dict[str, float]:
    """Run all three stages and persist weights. Returns test metrics."""
    set_seed(settings.SEED)
    log.info("Murmur training pipeline starting on %s (seed=%d)", DEVICE, settings.SEED)

    tracker = _Tracker()

    mics = settings.MIC_COORDS
    num_nodes = len(mics)
    edge_index, edge_weight = build_acoustic_topology(
        mics, settings.DISTANCE_THRESHOLD, decay_exponent=settings.DISTANCE_DECAY_EXPONENT
    )
    edge_index, edge_weight = edge_index.to(DEVICE), edge_weight.to(DEVICE)

    num_samples = settings.TRAIN_NUM_SAMPLES
    log.info(
        "Generating %d sequences (seq_len=%d, nodes=%d)",
        num_samples,
        settings.SEQ_LENGTH,
        num_nodes,
    )
    x, y, ts = generate_degradation_data(
        num_sequences=num_samples,
        seq_length=settings.SEQ_LENGTH,
        num_nodes=num_nodes,
        in_channels=settings.GNN_IN_CHANNELS,
        anomaly_ratio=0.30,
        mic_coords=mics,
        seed=settings.SEED,
    )

    generator = torch.Generator().manual_seed(settings.SEED)
    splits = train_val_test_split(x, y, ts, generator=generator)
    log.info(
        "Split: train=%d val=%d test=%d",
        splits["train"][0].size(0),
        splits["val"][0].size(0),
        splits["test"][0].size(0),
    )

    os.makedirs(settings.MODEL_DIR, exist_ok=True)

    with tracker.run():
        tracker.log_params(
            {
                "epochs": settings.TRAIN_EPOCHS,
                "learning_rate": settings.LEARNING_RATE,
                "embedding_dim": settings.GNN_EMBEDDING_DIM,
                "hidden_channels": settings.GNN_HIDDEN_CHANNELS,
                "seq_length": settings.SEQ_LENGTH,
                "num_samples": num_samples,
                "batch_size": settings.TRAIN_BATCH_SIZE,
                "num_nodes": num_nodes,
                "seed": settings.SEED,
                "device": str(DEVICE),
            }
        )

        autoencoder = train_autoencoder(tracker=tracker)
        torch.save(
            autoencoder.state_dict(),
            os.path.join(settings.MODEL_DIR, "autoencoder_weights.pth"),
        )

        st_gnn, lnn, test_metrics = train_forecaster(
            splits, edge_index, edge_weight, num_nodes, tracker=tracker
        )
        torch.save(st_gnn.state_dict(), os.path.join(settings.MODEL_DIR, "st_gnn_weights.pth"))
        torch.save(lnn.state_dict(), os.path.join(settings.MODEL_DIR, "lnn_weights.pth"))

        projector = train_projector(st_gnn, splits, edge_index, edge_weight)
        if projector is not None:
            torch.save(
                projector.state_dict(),
                os.path.join(settings.MODEL_DIR, "projector_weights.pth"),
            )

        calibrator, coverage = calibrate_forecaster(st_gnn, lnn, splits, edge_index, edge_weight)
        calibrator.save(os.path.join(settings.MODEL_DIR, "conformal.json"))
        test_metrics.update({f"conformal_{k}": v for k, v in coverage.items()})

        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    log.info("TEST RESULTS: %s", test_metrics)
    log.info(
        "Test MAE %.4f vs mean-predictor baseline %.4f",
        test_metrics["mae"],
        test_metrics["baseline_mae"],
    )
    log.info("Weights written to %s/", settings.MODEL_DIR)
    return test_metrics


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()


if __name__ == "__main__":  # pragma: no cover
    main()
