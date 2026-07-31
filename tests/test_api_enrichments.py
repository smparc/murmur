"""
Tests that the spatial and uncertainty enrichments survive the API boundary.

These exist because the failure mode is silent. Pydantic defaults to
``extra='ignore'``, so a worker can POST ``ttf_interval`` and
``source_position``, receive a cheerful 200, and have both fields dropped before
they ever reach a dashboard — with nothing in any log to say so.
"""

from __future__ import annotations

import time

from src.settings import settings


def _payload(**overrides) -> dict:
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


_INTERVAL = {
    "point": 0.4,
    "lower": 0.15,
    "upper": 0.65,
    "width": 0.5,
    "confidence": 0.9,
    "group": "warning",
}


class TestConformalInterval:
    def test_interval_is_echoed_not_dropped(self, api_client):
        resp = api_client.post("/generate_telemetry", json=_payload(ttf_interval=_INTERVAL))
        assert resp.status_code == 200

        interval = resp.json()["ttf_interval"]
        assert interval is not None, "the calibrated band was silently discarded"
        assert interval["lower"] == 0.15
        assert interval["upper"] == 0.65
        assert interval["confidence"] == 0.9

    def test_absent_interval_is_null_not_an_error(self, api_client):
        """No calibration is a degraded state, not a failure."""
        resp = api_client.post("/generate_telemetry", json=_payload())
        assert resp.status_code == 200
        assert resp.json()["ttf_interval"] is None

    def test_out_of_range_bounds_rejected(self, api_client):
        bad = dict(_INTERVAL, upper=1.7)
        assert (
            api_client.post("/generate_telemetry", json=_payload(ttf_interval=bad)).status_code
            == 422
        )


class TestSourceLocalization:
    def test_position_is_echoed(self, api_client):
        resp = api_client.post(
            "/generate_telemetry", json=_payload(source_position=[3.5, 7.0, 3.0])
        )
        assert resp.status_code == 200
        assert resp.json()["source_position"] == [3.5, 7.0, 3.0]

    def test_absent_position_is_null(self, api_client):
        assert (
            api_client.post("/generate_telemetry", json=_payload()).json()["source_position"]
            is None
        )

    def test_malformed_position_rejected(self, api_client):
        """A 2-element 'position' would silently mislocate every alert."""
        resp = api_client.post("/generate_telemetry", json=_payload(source_position=[1.0, 2.0]))
        assert resp.status_code == 422
        assert "triple" in resp.text.lower()


class TestBroadcast:
    def test_enrichments_reach_websocket_clients(self, api_client):
        """
        The dashboard reads the WebSocket, not the POST response. An enrichment
        that survives the response model but is stripped from the broadcast is
        invisible where it actually matters.
        """
        with api_client.websocket_connect("/ws/telemetry") as ws:
            api_client.post(
                "/generate_telemetry",
                json=_payload(ttf_interval=_INTERVAL, source_position=[3.5, 7.0, 3.0]),
            )

            for _ in range(10):
                frame = ws.receive_json()
                if frame.get("ttf_interval") is not None:
                    break
            else:  # pragma: no cover - only on regression
                raise AssertionError("no broadcast frame carried the interval")

            assert frame["ttf_interval"]["confidence"] == 0.9
            assert frame["source_position"] == [3.5, 7.0, 3.0]
