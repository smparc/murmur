"""Unit tests for the centralized settings module."""


import os
import pytest


from src.settings import settings



class TestSettings:
    def test_default_kafka_broker(self):
        assert settings.KAFKA_BROKER == os.getenv("KAFKA_BROKER", "localhost:9092")


    def test_samples_per_chunk(self):
        expected = int(settings.SAMPLE_RATE * settings.CHUNK_DURATION)
        assert settings.SAMPLES_PER_CHUNK == expected


    def test_num_nodes(self):
        assert settings.NUM_NODES == len(settings.DEFAULT_MIC_COORDS)


    def test_audio_params_sensible(self):
        assert settings.SAMPLE_RATE > 0
        assert settings.N_FFT > 0
        assert settings.HOP_LENGTH > 0
        assert settings.N_MELS > 0
        assert settings.N_FFT >= settings.HOP_LENGTH