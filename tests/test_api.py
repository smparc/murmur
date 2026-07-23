"""Tests for the FastAPI inference server."""


import pytest
from fastapi.testclient import TestClient



@pytest.fixture
def client():
    """Create a test client — imports trigger model loading."""
    from src.translation.llm_decoder import app
    return TestClient(app)



class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "device" in data
        assert "uptime_seconds" in data



class TestTelemetryEndpoint:
    def test_invalid_embedding_dim_rejected(self, client):
        resp = client.post("/generate_telemetry", json={
            "node_id": 0,
            "timestamp": 1234567890.0,
            "gnn_embedding": [0.1, 0.2],  # Wrong dimension
        })
        assert resp.status_code == 422  # Pydantic validation error


    def test_missing_fields_rejected(self, client):
        resp = client.post("/generate_telemetry", json={
            "node_id": 0,
        })
        assert resp.status_code == 422



class TestWebSocket:
    def test_ws_connect_disconnect(self, client):
        with client.websocket_connect("/ws/telemetry") as ws:
            # Just verify connection succeeds
            pass