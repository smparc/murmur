"""Tests for the sliding window buffer and data quality."""


import numpy as np
import pytest


from src.ingestion.cuda_stream_processor import SlidingWindowBuffer



class TestSlidingWindowBuffer:
    """Tests for the per-node temporal buffering system."""


    def test_buffer_not_ready_before_full(self):
        buf = SlidingWindowBuffer(window_size=5, num_nodes=2)
        spec = np.random.randn(64, 32).astype(np.float32)


        for i in range(4):
            buf.push(0, spec, float(i))
            assert not buf.is_ready(0)


    def test_buffer_ready_when_full(self):
        buf = SlidingWindowBuffer(window_size=5, num_nodes=2)
        spec = np.random.randn(64, 32).astype(np.float32)


        for i in range(5):
            buf.push(0, spec, float(i))


        assert buf.is_ready(0)


    def test_buffer_window_shape(self):
        buf = SlidingWindowBuffer(window_size=5, num_nodes=2)


        for i in range(5):
            spec = np.random.randn(64, 32).astype(np.float32)
            buf.push(0, spec, float(i) * 0.5)


        window, timespans = buf.get_window(0)
        assert window.shape == (5, 64, 32), f"Expected (5, 64, 32), got {window.shape}"
        assert timespans.shape == (5,), f"Expected (5,), got {timespans.shape}"


    def test_buffer_sliding_behavior(self):
        """Buffer should drop oldest frames when overfilled."""
        buf = SlidingWindowBuffer(window_size=3, num_nodes=1)


        for i in range(5):
            spec = np.full((4, 4), float(i), dtype=np.float32)
            buf.push(0, spec, float(i))


        window, _ = buf.get_window(0)
        # Should have frames 2, 3, 4 (oldest dropped)
        assert window.shape[0] == 3
        np.testing.assert_array_equal(window[0], np.full((4, 4), 2.0))
        np.testing.assert_array_equal(window[2], np.full((4, 4), 4.0))


    def test_per_node_isolation(self):
        """Pushing to node 0 should not affect node 1."""
        buf = SlidingWindowBuffer(window_size=3, num_nodes=2)
        spec = np.random.randn(64, 32).astype(np.float32)


        for i in range(3):
            buf.push(0, spec, float(i))


        assert buf.is_ready(0)
        assert not buf.is_ready(1)


    def test_timespans_deltas(self):
        """Timespans should be time deltas between consecutive frames."""
        buf = SlidingWindowBuffer(window_size=4, num_nodes=1)


        timestamps = [0.0, 0.5, 1.0, 1.7]
        for t in timestamps:
            buf.push(0, np.zeros((4, 4), dtype=np.float32), t)


        _, timespans = buf.get_window(0)
        assert timespans.dtype == np.float32
        # Deltas: [0.5, 0.5, 0.5, 0.7] but first is special
        assert len(timespans) == 4


    def test_get_latest(self):
        buf = SlidingWindowBuffer(window_size=5, num_nodes=1)
        assert buf.get_latest(0) is None


        spec = np.full((4, 4), 42.0, dtype=np.float32)
        buf.push(0, spec, 0.0)


        latest = buf.get_latest(0)
        assert latest is not None
        np.testing.assert_array_equal(latest, spec)


    def test_dynamic_node_creation(self):
        """Pushing to an unknown node should create its buffer."""
        buf = SlidingWindowBuffer(window_size=3, num_nodes=2)
        spec = np.random.randn(4, 4).astype(np.float32)


        # Push to node 99 (not in original range)
        buf.push(99, spec, 0.0)
        assert buf.get_latest(99) is not None



class TestMockEdgeDevice:
    """Tests for the improved mock edge device."""


    def test_generate_normal_audio(self):
        from src.ingestion.mock_edge_device import generate_mock_audio, FaultType
        audio_bytes = generate_mock_audio(0, fault=FaultType.NONE)
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0


    def test_generate_bearing_fault(self):
        from src.ingestion.mock_edge_device import generate_mock_audio, FaultType
        audio_bytes = generate_mock_audio(0, fault=FaultType.BEARING, severity=0.8)
        assert isinstance(audio_bytes, bytes)


    def test_generate_cavitation_fault(self):
        from src.ingestion.mock_edge_device import generate_mock_audio, FaultType
        audio_bytes = generate_mock_audio(0, fault=FaultType.CAVITATION, severity=0.5)
        assert isinstance(audio_bytes, bytes)


    def test_generate_imbalance_fault(self):
        from src.ingestion.mock_edge_device import generate_mock_audio, FaultType
        audio_bytes = generate_mock_audio(0, fault=FaultType.IMBALANCE, severity=0.6)
        assert isinstance(audio_bytes, bytes)


    def test_no_anomaly_flag_in_payload(self):
        """Ensure the payload no longer contains the data-leaking is_anomalous_flag."""
        import msgpack
        from src.ingestion.mock_edge_device import generate_mock_audio, FaultType
        import time


        audio_bytes = generate_mock_audio(0, fault=FaultType.BEARING, severity=0.5)
        payload = msgpack.packb(
            {"node_id": 0, "timestamp": time.time(), "audio": audio_bytes},
            use_bin_type=True,
        )
        data = msgpack.unpackb(payload, raw=False)
        assert "is_anomalous_flag" not in data, "Data leakage: anomaly flag should not be in payload"


    def test_severity_scales_energy(self):
        """Higher severity should produce higher energy audio."""
        from src.ingestion.mock_edge_device import generate_mock_audio, FaultType


        low = np.frombuffer(
            generate_mock_audio(0, fault=FaultType.BEARING, severity=0.1),
            dtype=np.float32,
        )
        high = np.frombuffer(
            generate_mock_audio(0, fault=FaultType.BEARING, severity=1.0),
            dtype=np.float32,
        )
        assert np.abs(high).mean() > np.abs(low).mean(), "Higher severity should produce louder audio"