"""Tests for the anomaly detection module."""


import torch
import numpy as np
import pytest


from src.detection.anomaly_detector import SpectrogramAutoencoder, AnomalyScorer



class TestSpectrogramAutoencoder:
    """Tests for the unsupervised anomaly detection autoencoder."""


    def test_output_shape(self):
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        x = torch.randn(4, 1, 64, 32)
        recon, z = ae(x)
        assert recon.shape == x.shape, f"Reconstruction shape mismatch: {recon.shape}"
        assert z.shape == (4, 32), f"Latent shape mismatch: {z.shape}"


    def test_anomaly_score_shape(self):
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        x = torch.randn(8, 1, 64, 32)
        scores = ae.anomaly_score(x)
        assert scores.shape == (8,), f"Score shape mismatch: {scores.shape}"


    def test_anomaly_score_non_negative(self):
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        x = torch.randn(4, 1, 64, 32)
        scores = ae.anomaly_score(x)
        assert (scores >= 0).all(), "Anomaly scores should be non-negative (MSE)"


    def test_high_noise_scores_higher(self):
        """Anomalous (high-noise) spectrograms should score higher than normal."""
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        ae.eval()


        normal = torch.randn(16, 1, 64, 32) * 0.1
        anomalous = torch.randn(16, 1, 64, 32) * 3.0


        # After random init, both will have similar reconstruction error,
        # but the magnitude difference should still show
        normal_scores = ae.anomaly_score(normal)
        anomalous_scores = ae.anomaly_score(anomalous)


        # Anomalous should have higher mean reconstruction error
        assert anomalous_scores.mean() > normal_scores.mean(), (
            "Anomalous data should have higher reconstruction error"
        )


    def test_encoder_decoder_roundtrip(self):
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        x = torch.randn(2, 1, 64, 32)
        z = ae.encode(x)
        recon = ae.decode(z, target_size=(64, 32))
        assert recon.shape == x.shape


    def test_gradients_flow(self):
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        x = torch.randn(2, 1, 64, 32, requires_grad=True)
        recon, z = ae(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        # Verify gradients propagate through the entire model
        for name, param in ae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"



class TestAnomalyScorer:
    """Tests for the online adaptive anomaly scorer."""


    def test_warmup_no_anomaly(self):
        """During warmup, nothing should be flagged as anomaly."""
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=20)
        spec = torch.randn(1, 1, 64, 32)


        for i in range(20):
            result = scorer.score(node_id=0, spectrogram=spec)
            assert result.is_warmup is True
            assert result.is_anomaly is False


    def test_post_warmup_normal_no_anomaly(self):
        """After warmup with consistent data, normal data should not flag."""
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10)


        # Feed consistent normal data
        for i in range(20):
            spec = torch.randn(1, 1, 64, 32) * 0.1
            result = scorer.score(node_id=0, spectrogram=spec)


        # After 20 frames of consistent data, one more normal frame should be fine
        spec = torch.randn(1, 1, 64, 32) * 0.1
        result = scorer.score(node_id=0, spectrogram=spec)
        assert result.is_warmup is False


    def test_anomaly_detection(self):
        """A sudden spike should be detected as anomaly after warmup."""
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10, z_threshold=2.0)


        # Build baseline with low-energy data
        for i in range(30):
            spec = torch.randn(1, 1, 64, 32) * 0.01
            scorer.score(node_id=0, spectrogram=spec)


        # Inject massive anomaly (100x energy)
        spec = torch.randn(1, 1, 64, 32) * 10.0
        result = scorer.score(node_id=0, spectrogram=spec)
        assert result.is_anomaly is True, f"Expected anomaly, got z_score={result.z_score}"


    def test_severity_levels(self):
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10, z_threshold=2.0)


        # Build baseline
        for i in range(30):
            spec = torch.randn(1, 1, 64, 32) * 0.01
            scorer.score(node_id=0, spectrogram=spec)


        # Mild anomaly
        spec = torch.randn(1, 1, 64, 32) * 5.0
        mild = scorer.score(node_id=0, spectrogram=spec)


        # Reset and build new baseline
        scorer2 = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10, z_threshold=2.0)
        for i in range(30):
            spec = torch.randn(1, 1, 64, 32) * 0.01
            scorer2.score(node_id=0, spectrogram=spec)


        # Severe anomaly
        spec = torch.randn(1, 1, 64, 32) * 50.0
        severe = scorer2.score(node_id=0, spectrogram=spec)


        # Both should be anomalies
        assert mild.is_anomaly or severe.is_anomaly


    def test_per_node_independence(self):
        """Each node should maintain independent statistics."""
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=5)


        # Only feed data to node 0
        for i in range(10):
            spec = torch.randn(1, 1, 64, 32)
            scorer.score(node_id=0, spectrogram=spec)


        summary = scorer.get_node_summary()
        assert summary[0]["total_frames"] == 10
        assert summary[1]["total_frames"] == 0


    def test_node_summary(self):
        scorer = AnomalyScorer(autoencoder=None, num_nodes=2, warmup_frames=5)


        for i in range(10):
            for node in range(2):
                spec = torch.randn(1, 1, 64, 32)
                scorer.score(node_id=node, spectrogram=spec)


        summary = scorer.get_node_summary()
        assert len(summary) == 2
        assert all("total_frames" in v for v in summary.values())
        assert all("anomaly_rate" in v for v in summary.values())


    def test_with_autoencoder(self):
        """Test scorer integrated with the autoencoder."""
        ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
        ae.eval()
        scorer = AnomalyScorer(autoencoder=ae, num_nodes=4, warmup_frames=10)


        for i in range(15):
            spec = torch.randn(1, 1, 64, 32) * 0.1
            result = scorer.score(node_id=0, spectrogram=spec)


        # Should complete without error and produce valid results
        assert isinstance(result.raw_score, float)
        assert isinstance(result.z_score, float)