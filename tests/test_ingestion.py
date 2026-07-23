"""Unit tests for ingestion utilities (audio generation, serialization)."""


import struct


import msgpack
import numpy as np
import pytest


from src.ingestion.mock_edge_device import generate_mock_audio
from src.settings import settings



class TestMockAudioGeneration:
    def test_output_is_bytes(self):
        audio = generate_mock_audio(node_id=0, anomaly=False)
        assert isinstance(audio, bytes)


    def test_correct_length(self):
        audio = generate_mock_audio(node_id=0, anomaly=False)
        expected_samples = settings.SAMPLES_PER_CHUNK
        # float32 = 4 bytes per sample
        assert len(audio) == expected_samples * 4


    def test_anomaly_has_higher_energy(self):
        normal = np.frombuffer(generate_mock_audio(0, anomaly=False), dtype=np.float32)
        anomaly = np.frombuffer(generate_mock_audio(0, anomaly=True), dtype=np.float32)


        normal_rms = np.sqrt(np.mean(normal ** 2))
        anomaly_rms = np.sqrt(np.mean(anomaly ** 2))


        # Anomaly signal (2kHz squeal) should have significantly higher energy
        assert anomaly_rms > normal_rms * 1.5


    def test_node_id_does_not_affect_shape(self):
        a0 = generate_mock_audio(0)
        a3 = generate_mock_audio(3)
        assert len(a0) == len(a3)



class TestMessagePackSerialization:
    def test_roundtrip(self):
        payload = {
            "node_id": 2,
            "timestamp": 1234567890.123,
            "audio": generate_mock_audio(2),
        }
        packed = msgpack.packb(payload, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)


        assert unpacked["node_id"] == 2
        assert abs(unpacked["timestamp"] - 1234567890.123) < 1e-3
        assert isinstance(unpacked["audio"], bytes)
        assert len(unpacked["audio"]) == len(payload["audio"])