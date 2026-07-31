"""Tests for configuration parsing and validation."""

from __future__ import annotations

import json

import pytest

from src.settings import ConfigError, Settings, settings


class TestDerivedValues:
    def test_samples_per_chunk(self):
        assert int(settings.SAMPLE_RATE * settings.CHUNK_DURATION) == settings.SAMPLES_PER_CHUNK

    def test_num_nodes_matches_topology(self):
        assert len(settings.MIC_COORDS) == settings.NUM_NODES

    def test_windowed_topic_derives_from_processed(self):
        assert f"{settings.PROCESSED_TOPIC}-windowed" == settings.WINDOWED_TOPIC

    def test_mel_frames_positive(self):
        assert settings.MEL_FRAMES_PER_CHUNK > 0

    def test_cors_origins_parse_to_list(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com ,")
        assert Settings().CORS_ORIGIN_LIST == ["http://a.com", "http://b.com"]

    def test_auth_disabled_when_key_blank(self, monkeypatch):
        monkeypatch.setenv("MURMUR_API_KEY", "")
        assert Settings().AUTH_ENABLED is False

    def test_auth_enabled_when_key_present(self, monkeypatch):
        monkeypatch.setenv("MURMUR_API_KEY", "s3cret")
        assert Settings().AUTH_ENABLED is True


class TestValidation:
    """
    The old module documented itself as providing "typed, validated
    configuration" but performed no validation at all — an impossible STFT
    setup produced silently misshapen spectrograms several services downstream.
    """

    def test_hop_larger_than_fft_rejected(self, monkeypatch):
        monkeypatch.setenv("HOP_LENGTH", "4096")
        monkeypatch.setenv("N_FFT", "1024")
        with pytest.raises(ConfigError, match="HOP_LENGTH"):
            Settings()

    def test_too_many_mels_rejected(self, monkeypatch):
        monkeypatch.setenv("N_MELS", "9999")
        with pytest.raises(ConfigError, match="N_MELS"):
            Settings()

    def test_chunk_shorter_than_fft_rejected(self, monkeypatch):
        monkeypatch.setenv("CHUNK_DURATION", "0.001")
        with pytest.raises(ConfigError, match="N_FFT"):
            Settings()

    def test_attention_head_divisibility_enforced(self, monkeypatch):
        monkeypatch.setenv("GNN_HIDDEN_CHANNELS", "130")
        monkeypatch.setenv("GNN_NUM_HEADS", "4")
        with pytest.raises(ConfigError, match="divisible"):
            Settings()

    def test_negative_dimension_rejected(self, monkeypatch):
        monkeypatch.setenv("GNN_EMBEDDING_DIM", "-1")
        with pytest.raises(ConfigError, match="GNN_EMBEDDING_DIM"):
            Settings()

    def test_anomaly_window_smaller_than_warmup_rejected(self, monkeypatch):
        monkeypatch.setenv("ANOMALY_WINDOW", "10")
        monkeypatch.setenv("ANOMALY_WARMUP_FRAMES", "100")
        with pytest.raises(ConfigError, match="ANOMALY_WINDOW"):
            Settings()

    def test_non_integer_env_reports_the_offending_key(self, monkeypatch):
        monkeypatch.setenv("SAMPLE_RATE", "not-a-number")
        with pytest.raises(ConfigError, match="SAMPLE_RATE"):
            Settings()

    def test_multiple_errors_reported_together(self, monkeypatch):
        monkeypatch.setenv("GNN_EMBEDDING_DIM", "-1")
        monkeypatch.setenv("SEQ_LENGTH", "-5")
        with pytest.raises(ConfigError) as exc:
            Settings()
        assert "GNN_EMBEDDING_DIM" in str(exc.value)
        assert "SEQ_LENGTH" in str(exc.value)


class TestTopologyFromEnvironment:
    """
    The microphone layout is the one thing the system cannot infer, and it was
    previously hardcoded — every site would have needed a source change.
    """

    def test_coords_parsed_from_json(self, monkeypatch):
        monkeypatch.setenv("MIC_COORDS", json.dumps([[0, 0, 3], [1, 1, 3], [2, 2, 3]]))
        assert Settings().NUM_NODES == 3

    def test_malformed_json_rejected(self, monkeypatch):
        monkeypatch.setenv("MIC_COORDS", "{not json")
        with pytest.raises(ConfigError, match="valid JSON"):
            Settings()

    def test_wrong_arity_rejected(self, monkeypatch):
        monkeypatch.setenv("MIC_COORDS", json.dumps([[0, 0], [1, 1]]))
        with pytest.raises(ConfigError, match=r"\[x, y, z\]"):
            Settings()

    def test_single_microphone_rejected(self, monkeypatch):
        monkeypatch.setenv("MIC_COORDS", json.dumps([[0, 0, 3]]))
        with pytest.raises(ConfigError, match="at least 2"):
            Settings()

    def test_non_numeric_coordinate_rejected(self, monkeypatch):
        monkeypatch.setenv("MIC_COORDS", json.dumps([[0, 0, 3], ["a", "b", "c"]]))
        with pytest.raises(ConfigError, match="non-numeric"):
            Settings()


class TestDescribe:
    def test_api_key_is_redacted(self, monkeypatch):
        monkeypatch.setenv("MURMUR_API_KEY", "super-secret-value")
        described = Settings().describe()
        assert described["API_KEY"] == "***redacted***"
        assert "super-secret-value" not in str(described)
