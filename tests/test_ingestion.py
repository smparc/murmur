"""Unit tests for ingestion: audio generation, decoding, serialization."""

from __future__ import annotations

import msgpack
import numpy as np
import pytest

from src.ingestion.mock_edge_device import FaultType, generate_mock_audio
from src.settings import settings


class _RecordingProducer:
    """Captures the topics a publish call actually writes to."""

    def __init__(self) -> None:
        self.topics: list[str] = []

    def produce(self, topic, key=None, value=None, callback=None) -> None:
        self.topics.append(topic)


class TestFrameTopicIsOptIn:
    """
    ``PROCESSED_TOPIC`` had no consumer anywhere in the tree — the worker reads
    ``WINDOWED_TOPIC``. Every frame from every node was still serialized,
    compressed and shipped to the broker, and retained, for no reader.
    """

    def _publish_once(self, buffer_len: int = 1):
        from src.ingestion.cuda_stream_processor import SlidingWindowBuffer, _publish

        producer = _RecordingProducer()
        buffer = SlidingWindowBuffer(window_size=buffer_len, num_nodes=settings.NUM_NODES)
        spec = np.zeros((settings.N_MELS, settings.MEL_FRAMES_PER_CHUNK), dtype=np.float32)
        _publish(producer, buffer, node_id=0, timestamp=1.0, spec=spec)
        return producer.topics

    def test_frame_topic_is_silent_by_default(self, override_settings):
        override_settings(PUBLISH_FRAME_TOPIC=False)
        topics = self._publish_once()
        assert settings.PROCESSED_TOPIC not in topics

    def test_windowed_topic_is_always_published(self, override_settings):
        """The worker depends on this one; it must never become optional."""
        override_settings(PUBLISH_FRAME_TOPIC=False)
        assert settings.WINDOWED_TOPIC in self._publish_once()

    def test_frame_topic_returns_when_asked_for(self, override_settings):
        override_settings(PUBLISH_FRAME_TOPIC=True)
        assert settings.PROCESSED_TOPIC in self._publish_once()


class TestMockAudioGeneration:
    """
    The previous version of this module called
    ``generate_mock_audio(node_id=0, anomaly=False)``. No such parameter has
    ever existed on the current signature (``fault`` / ``severity``), so every
    test here raised TypeError.
    """

    def test_output_is_bytes(self):
        assert isinstance(generate_mock_audio(0, fault=FaultType.NONE), bytes)

    def test_correct_length(self):
        audio = generate_mock_audio(0, fault=FaultType.NONE)
        assert len(audio) == settings.SAMPLES_PER_CHUNK * 4  # float32

    def test_decodes_to_finite_float32(self):
        audio = np.frombuffer(generate_mock_audio(0, fault=FaultType.NONE), dtype=np.float32)
        assert audio.shape == (settings.SAMPLES_PER_CHUNK,)
        assert np.isfinite(audio).all()

    @pytest.mark.parametrize(
        "fault",
        [FaultType.BEARING, FaultType.CAVITATION, FaultType.IMBALANCE],
    )
    def test_each_fault_type_generates(self, fault):
        audio = generate_mock_audio(0, fault=fault, severity=0.7)
        assert isinstance(audio, bytes)
        assert len(audio) == settings.SAMPLES_PER_CHUNK * 4

    @pytest.mark.parametrize(
        "fault",
        [FaultType.BEARING, FaultType.CAVITATION, FaultType.IMBALANCE],
    )
    def test_fault_raises_energy(self, fault):
        healthy = np.frombuffer(generate_mock_audio(0, fault=FaultType.NONE), dtype=np.float32)
        faulty = np.frombuffer(generate_mock_audio(0, fault=fault, severity=1.0), dtype=np.float32)
        assert np.abs(faulty).mean() > np.abs(healthy).mean()

    def test_severity_scales_energy(self):
        low = np.frombuffer(
            generate_mock_audio(0, fault=FaultType.BEARING, severity=0.1), dtype=np.float32
        )
        high = np.frombuffer(
            generate_mock_audio(0, fault=FaultType.BEARING, severity=1.0), dtype=np.float32
        )
        assert np.abs(high).mean() > np.abs(low).mean()

    def test_zero_severity_matches_healthy_energy(self):
        """A declared fault at zero severity must not inject any signal."""
        healthy = np.frombuffer(generate_mock_audio(1, fault=FaultType.NONE), dtype=np.float32)
        inert = np.frombuffer(
            generate_mock_audio(1, fault=FaultType.BEARING, severity=0.0), dtype=np.float32
        )
        assert abs(np.abs(inert).mean() - np.abs(healthy).mean()) < 0.05

    def test_node_id_does_not_affect_length(self):
        assert len(generate_mock_audio(0)) == len(generate_mock_audio(3))


class TestMessagePackSerialization:
    def test_roundtrip(self):
        payload = {
            "node_id": 2,
            "timestamp": 1234567890.123,
            "audio": generate_mock_audio(2),
        }
        unpacked = msgpack.unpackb(msgpack.packb(payload, use_bin_type=True), raw=False)

        assert unpacked["node_id"] == 2
        assert abs(unpacked["timestamp"] - 1234567890.123) < 1e-3
        assert isinstance(unpacked["audio"], bytes)
        assert len(unpacked["audio"]) == len(payload["audio"])

    def test_no_anomaly_label_leaks_into_payload(self):
        """
        The edge device must not ship ground truth. If it did, the detector
        could trivially "learn" to read the label instead of the audio.
        """
        payload = msgpack.unpackb(
            msgpack.packb(
                {
                    "node_id": 0,
                    "timestamp": 1.0,
                    "audio": generate_mock_audio(0, fault=FaultType.BEARING, severity=0.5),
                },
                use_bin_type=True,
            ),
            raw=False,
        )
        for leaky in ("is_anomalous_flag", "fault", "severity", "label"):
            assert leaky not in payload
