"""Tests for the anomaly detection module."""

import torch

from src.detection.anomaly_detector import _DEGENERATE_Z, AnomalyScorer, SpectrogramAutoencoder


class TestDegenerateBaseline:
    """
    A constant baseline has no scale, so the *magnitude* of a departure from it
    is not meaningful. The previous code multiplied the relative change by 1e3,
    producing z-scores in the tens of thousands that were written straight to a
    Prometheus gauge and interpolated into an LLM prompt.
    """

    def test_z_score_saturates_instead_of_exploding(self):
        scorer = AnomalyScorer(autoencoder=None, num_nodes=1, warmup_frames=1, window=50)

        constant = torch.full((64, 8), 0.001)
        for _ in range(10):
            scorer.score(0, constant)

        result = scorer.score(0, torch.full((64, 8), 5.0))

        assert result.z_score == _DEGENERATE_Z
        assert result.is_anomaly is True
        assert result.severity == "critical"

    def test_a_quieter_channel_is_not_a_fault(self):
        """Sign is preserved: silence is not a bearing failure."""
        scorer = AnomalyScorer(autoencoder=None, num_nodes=1, warmup_frames=1, window=50)

        constant = torch.full((64, 8), 1.0)
        for _ in range(10):
            scorer.score(0, constant)

        result = scorer.score(0, torch.zeros(64, 8))

        assert result.z_score == -_DEGENERATE_Z
        assert result.is_anomaly is False

    def test_an_identical_frame_scores_zero(self):
        scorer = AnomalyScorer(autoencoder=None, num_nodes=1, warmup_frames=1, window=50)
        constant = torch.full((64, 8), 0.25)
        for _ in range(10):
            scorer.score(0, constant)

        assert scorer.score(0, constant).z_score == 0.0


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
        recon, _ = ae(x)
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

        for _ in range(20):
            result = scorer.score(node_id=0, spectrogram=spec)
            assert result.is_warmup is True
            assert result.is_anomaly is False

    def test_post_warmup_normal_no_anomaly(self):
        """After warmup with consistent data, normal data should not flag."""
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10)

        # Feed consistent normal data
        for _ in range(20):
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
        for _ in range(30):
            spec = torch.randn(1, 1, 64, 32) * 0.01
            scorer.score(node_id=0, spectrogram=spec)

        # Inject massive anomaly (100x energy)
        spec = torch.randn(1, 1, 64, 32) * 10.0
        result = scorer.score(node_id=0, spectrogram=spec)
        assert result.is_anomaly is True, f"Expected anomaly, got z_score={result.z_score}"

    def test_severity_levels(self):
        scorer = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10, z_threshold=2.0)

        # Build baseline
        for _ in range(30):
            spec = torch.randn(1, 1, 64, 32) * 0.01
            scorer.score(node_id=0, spectrogram=spec)

        # Mild anomaly
        spec = torch.randn(1, 1, 64, 32) * 5.0
        mild = scorer.score(node_id=0, spectrogram=spec)

        # Reset and build new baseline
        scorer2 = AnomalyScorer(autoencoder=None, num_nodes=4, warmup_frames=10, z_threshold=2.0)
        for _ in range(30):
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
        for _ in range(10):
            spec = torch.randn(1, 1, 64, 32)
            scorer.score(node_id=0, spectrogram=spec)

        summary = scorer.get_node_summary()
        assert summary[0]["total_frames"] == 10
        assert summary[1]["total_frames"] == 0

    def test_node_summary(self):
        scorer = AnomalyScorer(autoencoder=None, num_nodes=2, warmup_frames=5)

        for _ in range(10):
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

        for _ in range(15):
            spec = torch.randn(1, 1, 64, 32) * 0.1
            result = scorer.score(node_id=0, spectrogram=spec)

        # Should complete without error and produce valid results
        assert isinstance(result.raw_score, float)
        assert isinstance(result.z_score, float)


def _frame(level: float, generator: torch.Generator, jitter: float = 0.05) -> torch.Tensor:
    """A frame whose energy score sits near ``level**2``, with frame-to-frame spread."""
    scale = level * (1.0 + jitter * torch.randn(1, generator=generator).item())
    return scale + 0.05 * torch.randn(64, 16, generator=generator)


class TestPersistentFaults:
    """
    Regression for the reviewer's isolated check: 500 frames near a raw score of
    1, then a sustained fault near 100, returned to "normal" at fault frame 251.
    Flagged frames were admitted to the rolling baseline, so once they filled
    half the window the median *was* the fault.
    """

    def _scorer(self, **kwargs) -> AnomalyScorer:
        return AnomalyScorer(autoencoder=None, num_nodes=1, warmup_frames=50, window=500, **kwargs)

    def test_a_sustained_fault_does_not_become_the_new_normal(self):
        g = torch.Generator().manual_seed(0)
        scorer = self._scorer()
        for _ in range(500):
            scorer.score(0, _frame(1.0, g))

        flagged = [scorer.score(0, _frame(10.0, g)).is_anomaly for _ in range(1000)]
        # Previously False from frame 251 onward.
        assert all(flagged)

    def test_a_slow_creep_is_caught_by_the_drift_guard(self):
        """
        Each step is far below the per-frame threshold, so the rolling baseline
        follows the ramp. Only the anchor frozen at warmup can see how far it
        has gone.
        """

        def run(drift_guard: bool) -> int:
            g = torch.Generator().manual_seed(1)
            scorer = self._scorer(drift_guard=drift_guard)
            for _ in range(500):
                scorer.score(0, _frame(1.0, g))
            ramp = [scorer.score(0, _frame(1.0 + 0.0005 * i, g)) for i in range(4000)]
            return sum(r.is_anomaly for r in ramp[-500:])

        assert run(drift_guard=False) < 50  # absorbed: the ramp became the baseline
        assert run(drift_guard=True) == 500

    def test_a_transient_clears_once_the_machine_recovers(self):
        g = torch.Generator().manual_seed(2)
        scorer = self._scorer()
        for _ in range(500):
            scorer.score(0, _frame(1.0, g))
        blip = [scorer.score(0, _frame(3.0, g)).is_anomaly for _ in range(10)]
        after = [scorer.score(0, _frame(1.0, g)).is_anomaly for _ in range(200)]
        assert all(blip)
        assert sum(after) <= 5  # background false-alarm rate, not a stuck flag

    def test_drift_flag_reports_a_non_zero_anomaly_score(self):
        """A drift-only flag must not reach the API as an anomaly score of ~0."""
        g = torch.Generator().manual_seed(3)
        scorer = self._scorer()
        for _ in range(500):
            scorer.score(0, _frame(1.0, g))
        for i in range(4000):
            result = scorer.score(0, _frame(1.0 + 0.0005 * i, g))
        assert result.is_anomaly
        assert result.drift_z >= scorer.z_threshold
        assert result.normalized_score > 0.3

    def test_rebaseline_accepts_a_new_operating_point(self):
        g = torch.Generator().manual_seed(4)
        scorer = self._scorer()
        for _ in range(500):
            scorer.score(0, _frame(1.0, g))
        for _ in range(100):
            assert scorer.score(0, _frame(2.0, g)).is_anomaly

        assert scorer.rebaseline(0) is True
        after = [scorer.score(0, _frame(2.0, g)).is_anomaly for _ in range(300)]
        assert sum(after) <= 5
        # And it still detects a genuine fault above the new normal.
        assert scorer.score(0, _frame(4.0, g)).is_anomaly

    def test_rebaseline_without_history_restarts_warmup(self):
        g = torch.Generator().manual_seed(5)
        scorer = self._scorer()
        for _ in range(10):
            scorer.score(0, _frame(1.0, g))
        assert scorer.rebaseline(0) is False
        assert scorer.get_node_summary()[0]["total_frames"] == 0

    def test_drift_guard_arms_at_end_of_warmup(self):
        g = torch.Generator().manual_seed(6)
        scorer = self._scorer()
        for _ in range(49):
            scorer.score(0, _frame(1.0, g))
        assert scorer.get_node_summary()[0]["drift_guard_armed"] is False
        scorer.score(0, _frame(1.0, g))
        assert scorer.get_node_summary()[0]["drift_guard_armed"] is True
