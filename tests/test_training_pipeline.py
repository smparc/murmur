"""Tests for the training pipeline improvements."""

import pytest
import torch

from src.training.train_pipeline import (
    compute_metrics,
    generate_degradation_data,
    train_val_test_split,
)


class TestSyntheticDataGeneration:
    """Tests for realistic degradation data generation."""

    def test_output_shapes(self):
        x, y, ts = generate_degradation_data(
            num_sequences=50,
            seq_length=10,
            num_nodes=4,
            in_channels=64,
        )
        assert x.shape == (50, 10, 256), f"x shape: {x.shape}"
        # One degradation label per microphone — the quantity the worker serves.
        assert y.shape == (50, 4), f"y shape: {y.shape}"
        assert ts.shape == (50, 10), f"ts shape: {ts.shape}"

    def test_degradation_range(self):
        """Degradation labels should be in [0, 1]."""
        _, y, _ = generate_degradation_data(100, 10, 4, 64, anomaly_ratio=0.5)
        assert y.min() >= 0.0, f"degradation below 0: {y.min()}"
        assert y.max() <= 1.0, f"degradation above 1: {y.max()}"

    def test_anomaly_ratio(self):
        """Roughly the expected fraction of sequences should contain a degrading machine."""
        _, y, _ = generate_degradation_data(1000, 10, 4, 64, anomaly_ratio=0.2)
        degrading = (y > 0.1).any(dim=1).float().mean().item()
        # Should be roughly 20% (with some variance)
        assert 0.1 < degrading < 0.35, f"Anomaly ratio out of range: {degrading}"

    def test_only_the_source_microphone_carries_the_fault(self):
        """
        A fault is labelled at the machine it originates from. Neighbours hear it
        attenuated, but the machines they monitor are healthy, so their labels
        stay in the healthy band — otherwise a per-node score could not say which
        machine is failing.
        """
        _, y, _ = generate_degradation_data(500, 10, 4, 64, anomaly_ratio=1.0, seed=0)
        above_healthy = (y > 0.1).sum(dim=1)
        assert above_healthy.max().item() <= 1
        # Every sequence has exactly one source; its label is the row maximum.
        assert (y.max(dim=1).values >= 0.05).all()

    def test_timespans_positive(self):
        """All timespans should be positive."""
        _, _, ts = generate_degradation_data(50, 10, 4, 64)
        assert (ts > 0).all(), "All timespans must be positive"

    def test_normal_samples_low_energy(self):
        """Normal samples should have lower energy than degraded ones."""
        x, y, _ = generate_degradation_data(200, 10, 4, 64, anomaly_ratio=0.3)
        # Per microphone: (sequences, seq, nodes, mels) -> energy per (sequence, node).
        energy = x.reshape(200, 10, 4, 64).pow(2).mean(dim=(1, 3))
        normal_mask = y < 0.1
        anomaly_mask = y > 0.3

        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            normal_energy = energy[normal_mask].mean()
            anomaly_energy = energy[anomaly_mask].mean()
            assert anomaly_energy > normal_energy, "Anomalous data should have higher energy"


class TestTrainValTestSplit:
    """Tests for the data splitting logic."""

    def test_split_sizes(self):
        x = torch.randn(100, 10, 256)
        y = torch.randn(100, 1)
        ts = torch.randn(100, 10)

        splits = train_val_test_split(x, y, ts, train_ratio=0.7, val_ratio=0.15)

        assert splits["train"][0].size(0) == 70
        assert splits["val"][0].size(0) == 15
        assert splits["test"][0].size(0) == 15

    def test_no_data_loss(self):
        """All samples should be accounted for."""
        x = torch.randn(100, 10, 256)
        y = torch.randn(100, 1)
        ts = torch.randn(100, 10)

        splits = train_val_test_split(x, y, ts)
        total = sum(s[0].size(0) for s in splits.values())
        assert total == 100

    def test_shuffling(self):
        """Split should shuffle data (not just slice)."""
        torch.manual_seed(42)
        x = torch.arange(100).float().unsqueeze(1).unsqueeze(1)
        y = torch.zeros(100, 1)
        ts = torch.zeros(100, 1)

        splits = train_val_test_split(x, y, ts)
        train_vals = splits["train"][0].squeeze()
        # If shuffled, the first 70 values should not be 0-69 in order
        assert not torch.equal(train_vals, torch.arange(70).float())


class TestComputeMetrics:
    """Tests for the metrics computation."""

    def test_perfect_predictions(self):
        targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        metrics = compute_metrics(targets, targets, threshold=0.5)
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["precision"] > 0.99
        assert metrics["recall"] > 0.99

    def test_completely_wrong(self):
        preds = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        metrics = compute_metrics(preds, targets, threshold=0.5)
        assert metrics["mse"] > 0.9
        assert metrics["precision"] < 0.01  # All predictions are wrong

    def test_all_keys_present(self):
        preds = torch.randn(10, 1).sigmoid()
        targets = torch.randn(10, 1).sigmoid()
        metrics = compute_metrics(preds, targets)
        assert set(metrics.keys()) == {"mse", "mae", "precision", "recall", "f1"}


class TestPerNodeTrainServeParity:
    """
    The worker serves one degradation score per microphone. Training and
    conformal calibration used the pooled graph readout instead, so the LNN and
    the interval radii were fitted to a quantity that is never served.
    """

    @pytest.fixture
    def models(self, sample_edge_topology):
        from src.forecasting.liquid_network import AcousticForecastingLNN
        from src.mapping.st_gnn_model import SpatioTemporalGNN

        edge_index, edge_weight, num_nodes = sample_edge_topology
        st_gnn = SpatioTemporalGNN(
            in_channels=16, hidden_channels=16, embedding_dim=16, num_nodes=num_nodes, num_heads=2
        ).eval()
        lnn = AcousticForecastingLNN(input_dim=16, hidden_neurons=16).eval()
        return st_gnn, lnn, edge_index, edge_weight, num_nodes

    def test_forward_forecast_is_per_node(self, models):
        from src.training.train_pipeline import _forward_forecast

        st_gnn, lnn, edge_index, edge_weight, num_nodes = models
        x, _, ts = generate_degradation_data(6, 8, num_nodes, 16, seed=0)
        with torch.no_grad():
            preds = _forward_forecast(st_gnn, lnn, x, ts, edge_index, edge_weight)
        assert preds.shape == (6, num_nodes)

    def test_training_path_is_the_serving_path(self, models):
        """Each node's score is the LNN applied to that node's own trajectory."""
        from src.forecasting.liquid_network import per_node_forecast

        st_gnn, lnn, edge_index, edge_weight, num_nodes = models
        x, _, ts = generate_degradation_data(3, 8, num_nodes, 16, seed=1)
        with torch.no_grad():
            _, node_sequence = st_gnn(
                x, edge_index, edge_weight, return_sequence=True, return_nodes=True
            )
            batched = per_node_forecast(lnn, node_sequence, ts)
            for node in range(num_nodes):
                alone = lnn(node_sequence[:, :, node, :], timespans=ts).squeeze(-1)
                assert torch.allclose(batched[:, node], alone, atol=1e-5)

    def test_calibration_uses_every_nodes_forecast(self, models):
        from src.training.train_pipeline import calibrate_forecaster

        st_gnn, lnn, edge_index, edge_weight, num_nodes = models
        x, y, ts = generate_degradation_data(40, 8, num_nodes, 16, seed=2)
        calibrator, coverage = calibrate_forecaster(
            st_gnn, lnn, {"test": (x, y, ts)}, edge_index, edge_weight, alpha=0.1
        )
        # Half the sequences calibrate, and every microphone in each contributes.
        assert calibrator.n_calibration == 20 * num_nodes
        assert coverage["n"] == 20 * num_nodes
