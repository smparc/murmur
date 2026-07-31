"""Tests for the FastAPI telemetry service."""

from __future__ import annotations

import time

import pytest

from src.settings import settings


def _valid_payload(**overrides) -> dict:
    payload = {
        "node_id": 0,
        "timestamp": time.time(),
        "gnn_embedding": [0.01] * settings.GNN_EMBEDDING_DIM,
        "anomaly_score": 0.5,
        "anomaly_severity": "warning",
        "ttf_prediction": 0.4,
        "is_anomaly": True,
        "z_score": 3.2,
    }
    payload.update(overrides)
    return payload


class TestHealth:
    def test_health_returns_200(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "ready"
        assert {"device", "uptime_seconds", "model_loaded", "llm_enabled"} <= data.keys()

    def test_ready_returns_200_once_loaded(self, api_client):
        assert api_client.get("/ready").status_code == 200


class TestValidation:
    """
    These guard a regression that shipped: the ``@field_validator`` decorators
    were stripped, leaving bare string expressions and undecorated methods, so
    pydantic performed none of these checks and a wrong-sized embedding reached
    the projector's matmul as a 500.
    """

    def test_wrong_embedding_dim_rejected(self, api_client):
        resp = api_client.post("/generate_telemetry", json=_valid_payload(gnn_embedding=[0.1, 0.2]))
        assert resp.status_code == 422
        assert "dim" in resp.text.lower()

    def test_negative_node_id_rejected(self, api_client):
        assert api_client.post(
            "/generate_telemetry", json=_valid_payload(node_id=-1)
        ).status_code == 422

    def test_missing_fields_rejected(self, api_client):
        assert api_client.post("/generate_telemetry", json={"node_id": 0}).status_code == 422

    def test_out_of_range_score_rejected(self, api_client):
        assert api_client.post(
            "/generate_telemetry", json=_valid_payload(anomaly_score=1.5)
        ).status_code == 422

    def test_unknown_severity_rejected(self, api_client):
        assert api_client.post(
            "/generate_telemetry", json=_valid_payload(anomaly_severity="catastrophic")
        ).status_code == 422


class TestGeneration:
    def test_valid_request_returns_structured_telemetry(self, api_client):
        body = api_client.post("/generate_telemetry", json=_valid_payload()).json()

        assert body["node_id"] == 0
        assert isinstance(body["telemetry"], str) and body["telemetry"]
        assert body["anomaly"]["severity"] == "warning"
        assert body["anomaly"]["is_anomaly"] is True
        assert 0.0 <= body["ttf_prediction"] <= 1.0

    def test_severity_is_structured_not_parsed_from_text(self, api_client):
        """The dashboard must never need to regex the generated prose."""
        for severity in ("normal", "warning", "critical"):
            body = api_client.post(
                "/generate_telemetry", json=_valid_payload(anomaly_severity=severity)
            ).json()
            assert body["anomaly"]["severity"] == severity

    def test_generated_flag_reports_templated_fallback(self, api_client):
        # LLM_ENABLED is false in the test environment.
        assert api_client.post("/generate_telemetry", json=_valid_payload()).json()[
            "generated"
        ] is False


class TestMetrics:
    def test_metrics_exposes_recorded_series(self, api_client):
        api_client.post("/generate_telemetry", json=_valid_payload())
        text = api_client.get("/metrics").text

        # track_latency was previously defined but never applied, leaving these
        # series permanently absent.
        assert "murmur_requests_total" in text
        assert "murmur_request_latency_seconds" in text
        assert "murmur_anomaly_score" in text


class TestWebSocket:
    def test_connect_and_disconnect(self, api_client):
        with api_client.websocket_connect("/ws/telemetry"):
            pass

    def test_client_receives_broadcast(self, api_client):
        with api_client.websocket_connect("/ws/telemetry") as ws:
            api_client.post("/generate_telemetry", json=_valid_payload(node_id=2))
            frame = ws.receive_json()
            assert frame["node_id"] == 2
            assert "anomaly" in frame and "ttf_prediction" in frame

    def test_new_client_receives_replay_history(self, api_client):
        """An operator opening the dashboard mid-shift should see context."""
        api_client.post("/generate_telemetry", json=_valid_payload(node_id=3))
        with api_client.websocket_connect("/ws/telemetry") as ws:
            assert ws.receive_json()["node_id"] == 3


class TestAuth:
    @pytest.fixture
    def secured_client(self, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(settings, "API_KEY", "test-key")
        from src.translation.llm_decoder import app

        with TestClient(app) as client:
            yield client

    def test_request_without_key_rejected(self, secured_client):
        assert secured_client.post(
            "/generate_telemetry", json=_valid_payload()
        ).status_code == 401

    def test_request_with_wrong_key_rejected(self, secured_client):
        assert secured_client.post(
            "/generate_telemetry", json=_valid_payload(), headers={"X-API-Key": "nope"}
        ).status_code == 401

    def test_request_with_correct_key_accepted(self, secured_client):
        assert secured_client.post(
            "/generate_telemetry", json=_valid_payload(), headers={"X-API-Key": "test-key"}
        ).status_code == 200

    def test_health_stays_open_for_probes(self, secured_client):
        """Kubernetes probes cannot present credentials."""
        assert secured_client.get("/health").status_code == 200


class TestRateLimit:
    def test_burst_beyond_limit_is_throttled(self, api_client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 5)
        from src.translation.llm_decoder import state

        state._rate_buckets.clear()

        codes = [
            api_client.post("/generate_telemetry", json=_valid_payload()).status_code
            for _ in range(8)
        ]
        assert 429 in codes
        assert codes.count(200) <= 5
