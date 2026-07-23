"""Unit tests for the Liquid Neural Network forecasting module."""


import pytest
import torch


from src.forecasting.liquid_network import AcousticForecastingLNN



class TestAcousticForecastingLNN:
    def test_forward_shape(self):
        model = AcousticForecastingLNN(input_dim=256, hidden_neurons=64, output_dim=1)
        x = torch.randn(8, 50, 256)
        out = model(x)
        assert out.shape == (8, 1)


    def test_output_range(self):
        """Sigmoid activation should clamp output to [0, 1]."""
        model = AcousticForecastingLNN(input_dim=128, hidden_neurons=32, output_dim=1)
        x = torch.randn(4, 20, 128)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


    def test_with_timespans(self):
        """CfC should accept irregular time intervals."""
        model = AcousticForecastingLNN(input_dim=64, hidden_neurons=32, output_dim=1)
        x = torch.randn(4, 30, 64)
        ts = torch.rand(4, 30)
        out = model(x, timespans=ts)
        assert out.shape == (4, 1)


    def test_gradient_flow(self):
        model = AcousticForecastingLNN(input_dim=64, hidden_neurons=32, output_dim=1)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


    def test_multi_output(self):
        """Test with multiple output dimensions (e.g., TTF + severity)."""
        model = AcousticForecastingLNN(input_dim=64, hidden_neurons=32, output_dim=3)
        x = torch.randn(4, 20, 64)
        out = model(x)
        assert out.shape == (4, 3)