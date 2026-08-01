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
        assert (
            api_client.post("/generate_telemetry", json=_valid_payload(node_id=-1)).status_code
            == 422
        )

    def test_missing_fields_rejected(self, api_client):
        assert api_client.post("/generate_telemetry", json={"node_id": 0}).status_code == 422

    def test_out_of_range_score_rejected(self, api_client):
        assert (
            api_client.post(
                "/generate_telemetry", json=_valid_payload(anomaly_score=1.5)
            ).status_code
            == 422
        )

    def test_unknown_severity_rejected(self, api_client):
        assert (
            api_client.post(
                "/generate_telemetry", json=_valid_payload(anomaly_severity="catastrophic")
            ).status_code
            == 422
        )


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
        assert (
            api_client.post("/generate_telemetry", json=_valid_payload()).json()["generated"]
            is False
        )


class TestGroundedDiagnosis:
    """
    The projector that conditions generated text on audio is a random projection
    until it is trained, so prose derived from it is the model's prior wearing a
    diagnosis. Measured spectral evidence and a catalogue match give the output
    something true to rest on — and must survive to the client either way.
    """

    _EVIDENCE = {
        "explanation": {
            "total_error": 0.42,
            "summary": "energy concentrated at 2.1-3.4 kHz (46%)",
            "bands": [{"low_hz": 2100.0, "high_hz": 3400.0, "share": 0.46, "peak_frame": 3}],
        },
        "diagnosis": {
            "fault": "Bearing race defect",
            "confidence": 0.72,
            "urgency": "schedule",
            "recommended_action": "Schedule bearing inspection.",
        },
    }

    def test_evidence_is_echoed_to_the_client(self, api_client):
        body = api_client.post("/generate_telemetry", json=_valid_payload(**self._EVIDENCE)).json()

        assert body["diagnosis"]["fault"] == "Bearing race defect"
        assert body["explanation"]["bands"][0]["low_hz"] == 2100.0

    def test_templated_text_names_the_fault(self, api_client):
        """The LLM is optional; the templated path is what most sites run."""
        body = api_client.post("/generate_telemetry", json=_valid_payload(**self._EVIDENCE)).json()

        assert "Bearing race defect" in body["telemetry"]
        assert body["generated"] is False

    def test_evidence_is_optional(self, api_client):
        """A frame that never flagged has nothing to attribute."""
        body = api_client.post("/generate_telemetry", json=_valid_payload()).json()

        assert body["diagnosis"] is None
        assert body["explanation"] is None
        assert body["telemetry"]

    def test_prompt_states_the_evidence(self):
        from src.translation.llm_decoder import TelemetryRequest, _build_prompt

        prompt = _build_prompt(TelemetryRequest(**_valid_payload(**self._EVIDENCE)))

        assert "2.1-3.4 kHz" in prompt
        assert "Bearing race defect" in prompt
        assert "72% confidence" in prompt


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
    def secured_client(self, override_settings):
        from fastapi.testclient import TestClient

        override_settings(API_KEY="test-key")
        from src.translation.llm_decoder import app

        with TestClient(app) as client:
            yield client

    def test_request_without_key_rejected(self, secured_client):
        assert secured_client.post("/generate_telemetry", json=_valid_payload()).status_code == 401

    def test_request_with_wrong_key_rejected(self, secured_client):
        assert (
            secured_client.post(
                "/generate_telemetry", json=_valid_payload(), headers={"X-API-Key": "nope"}
            ).status_code
            == 401
        )

    def test_request_with_correct_key_accepted(self, secured_client):
        assert (
            secured_client.post(
                "/generate_telemetry", json=_valid_payload(), headers={"X-API-Key": "test-key"}
            ).status_code
            == 200
        )

    def test_health_stays_open_for_probes(self, secured_client):
        """Kubernetes probes cannot present credentials."""
        assert secured_client.get("/health").status_code == 200

    def test_metrics_requires_a_key(self, secured_client):
        """
        The exposition carries per-node anomaly counts, z-scores and TTF
        forecasts — a live map of which machines are failing. On a LoadBalancer
        Service that was public whenever an API key was configured.
        """
        assert secured_client.get("/metrics").status_code == 401
        assert secured_client.get("/metrics", headers={"X-API-Key": "test-key"}).status_code == 200

    def test_metrics_can_be_opened_for_in_cluster_scrapers(self, secured_client, override_settings):
        """A Prometheus that cannot present a key needs an escape hatch."""
        override_settings(METRICS_REQUIRE_AUTH=False)
        assert secured_client.get("/metrics").status_code == 200


class TestRateLimit:
    def test_burst_beyond_limit_is_throttled(self, api_client, override_settings):
        override_settings(RATE_LIMIT_PER_MINUTE=5)
        from src.translation.llm_decoder import state

        state._rate_buckets.clear()

        codes = [
            api_client.post("/generate_telemetry", json=_valid_payload()).status_code
            for _ in range(8)
        ]
        assert 429 in codes

    def test_bucket_map_is_bounded_across_distinct_keys(self, override_settings):
        """
        The limiter must not become the exhaustion vector it prevents.

        Buckets are keyed by API key or client address — both attacker-supplied
        on a public endpoint — and were only ever trimmed *within* a key, never
        across keys, and cleared only at shutdown. One request per spoofed
        source grew the map without bound.
        """
        override_settings(RATE_LIMIT_MAX_KEYS=100)
        from src.translation.llm_decoder import state

        state._rate_buckets.clear()
        state._last_sweep = 0.0

        now = time.monotonic()
        for i in range(5_000):
            state._rate_buckets[f"10.0.0.{i}"].append(now)
            state.sweep_rate_buckets(
                now,
                force=len(state._rate_buckets) >= 100,
            )

        assert len(state._rate_buckets) <= 100

    def test_idle_buckets_are_reclaimed(self, override_settings):
        """A caller that never returns must not hold memory forever."""
        from src.translation.llm_decoder import state

        state._rate_buckets.clear()
        state._last_sweep = 0.0

        now = time.monotonic()
        state._rate_buckets["stale-caller"].append(now - 3600.0)
        state._rate_buckets["live-caller"].append(now)

        state.sweep_rate_buckets(now, force=True)

        assert "stale-caller" not in state._rate_buckets
        assert "live-caller" in state._rate_buckets
