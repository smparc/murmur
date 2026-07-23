"""Tests for the training pipeline improvements."""


import torch
import pytest


from src.training.train_pipeline import (
    generate_degradation_data,
    train_val_test_split,
    compute_metrics,
)
from src.settings import settings



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
        assert y.shape == (50, 1), f"y shape: {y.shape}"
        assert ts.shape == (50, 10), f"ts shape: {ts.shape}"


    def test_ttf_range(self):
        """TTF labels should be in [0, 1]."""
        _, y, _ = generate_degradation_data(100, 10, 4, 64, anomaly_ratio=0.5)
        assert y.min() >= 0.0, f"TTF below 0: {y.min()}"
        assert y.max() <= 1.0, f"TTF above 1: {y.max()}"


    def test_anomaly_ratio(self):
        """Roughly the expected fraction should be degrading (high TTF)."""
        _, y, _ = generate_degradation_data(1000, 10, 4, 64, anomaly_ratio=0.2)
        high_ttf = (y > 0.1).float().mean().item()
        # Should be roughly 20% (with some variance)
        assert 0.1 < high_ttf < 0.35, f"Anomaly ratio out of range: {high_ttf}"


    def test_timespans_positive(self):
        """All timespans should be positive."""
        _, _, ts = generate_degradation_data(50, 10, 4, 64)
        assert (ts > 0).all(), "All timespans must be positive"


    def test_normal_samples_low_energy(self):
        """Normal samples should have lower energy than degraded ones."""
        x, y, _ = generate_degradation_data(200, 10, 4, 64, anomaly_ratio=0.3)
        normal_mask = y.squeeze() < 0.1
        anomaly_mask = y.squeeze() > 0.3


        if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
            normal_energy = x[normal_mask].pow(2).mean()
            anomaly_energy = x[anomaly_mask].pow(2).mean()
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