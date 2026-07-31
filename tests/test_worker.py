"""
Tests for the streaming inference worker.

This is the service that joins ingestion to serving. It did not exist before —
the spectrogram topics were produced and never consumed — so none of this path
had any coverage at all.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.inference.worker import InferenceWorker, NodeWindow, WindowAssembler, decode_window
from src.settings import settings


def _window(node_id: int, timestamp: float, amplitude: float = 1.0, seq: int = 8) -> NodeWindow:
    rng = np.random.default_rng(node_id * 31 + int(timestamp))
    frames = (rng.standard_normal((seq, settings.N_MELS, 6)) * amplitude).astype(np.float32)
    return NodeWindow(
        node_id=node_id,
        features=frames.mean(axis=2),
        timespans=np.full(seq, 0.5, dtype=np.float32),
        timestamp=timestamp,
        latest_frame=frames[-1],
    )


class TestWindowAssembler:
    def test_incomplete_until_every_node_reports(self):
        assembler = WindowAssembler(num_nodes=4, seq_length=8, n_mels=settings.N_MELS)
        for node in range(3):
            assembler.push(_window(node, 100.0))
            assert not assembler.is_complete()
        assembler.push(_window(3, 100.0))
        assert assembler.is_complete()

    def test_stale_node_blocks_completion(self):
        """
        A microphone that dropped out an hour ago must not be stitched into a
        snapshot alongside live ones — the GCN would convolve across time.
        """
        assembler = WindowAssembler(
            num_nodes=2, seq_length=8, n_mels=settings.N_MELS, staleness_tolerance=5.0
        )
        assembler.push(_window(0, 100.0))
        assembler.push(_window(1, 100.0 + 60.0))
        assert not assembler.is_complete()

    def test_assembled_shape_matches_gnn_contract(self):
        assembler = WindowAssembler(num_nodes=4, seq_length=8, n_mels=settings.N_MELS)
        for node in range(4):
            assembler.push(_window(node, 100.0))

        x, timespans, snapshot = assembler.assemble()
        assert x.shape == (1, 8, 4 * settings.N_MELS)
        assert timespans.shape == (1, 8)
        assert set(snapshot) == {0, 1, 2, 3}

    def test_node_ordering_is_stable(self):
        """Node k must always occupy the same feature slice, or edges misalign."""
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        w0, w1 = _window(0, 100.0, seq=4), _window(1, 100.0, seq=4)

        assembler.push(w1)
        assembler.push(w0)
        x_a, _, _ = assembler.assemble()
        assembler.clear()

        assembler.push(w0)
        assembler.push(w1)
        x_b, _, _ = assembler.assemble()

        assert np.allclose(x_a.numpy(), x_b.numpy())

    def test_clear_resets(self):
        assembler = WindowAssembler(num_nodes=2, seq_length=4, n_mels=settings.N_MELS)
        assembler.push(_window(0, 100.0, seq=4))
        assembler.clear()
        assert not assembler.is_complete()


class TestDecodeWindow:
    def test_roundtrip_from_producer_format(self):
        import msgpack

        seq, mels, frames = 6, settings.N_MELS, 5
        window = np.random.randn(seq, mels, frames).astype(np.float32)
        timespans = np.full(seq, 0.5, dtype=np.float32)

        raw = msgpack.packb(
            {
                "node_id": 2,
                "timestamp": 123.0,
                "window_shape": [seq, mels, frames],
                "window": window.tobytes(),
                "timespans": timespans.tobytes(),
            },
            use_bin_type=True,
        )

        decoded = decode_window(raw)
        assert decoded is not None
        assert decoded.node_id == 2
        # The intra-chunk time axis is averaged away to give one feature vector
        # per timestep, which is the contract the ST-GNN expects.
        assert decoded.features.shape == (seq, mels)
        assert decoded.latest_frame.shape == (mels, frames)
        assert np.allclose(decoded.features, window.mean(axis=2), atol=1e-5)

    def test_garbage_returns_none(self):
        assert decode_window(b"not msgpack at all") is None

    def test_non_finite_rejected(self):
        import msgpack

        seq, mels, frames = 4, settings.N_MELS, 3
        window = np.full((seq, mels, frames), np.nan, dtype=np.float32)
        raw = msgpack.packb(
            {
                "node_id": 0,
                "timestamp": 1.0,
                "window_shape": [seq, mels, frames],
                "window": window.tobytes(),
                "timespans": np.ones(seq, dtype=np.float32).tobytes(),
            },
            use_bin_type=True,
        )
        assert decode_window(raw) is None


@pytest.fixture
def worker(api_client, override_settings):
    override_settings(SEQ_LENGTH=8)
    w = InferenceWorker(
        inference_url="http://testserver", http_client=api_client, load_weights=False
    )
    w.assembler = WindowAssembler(
        num_nodes=settings.NUM_NODES, seq_length=8, n_mels=settings.N_MELS
    )
    yield w


class TestInferenceWorker:
    def test_partial_array_produces_nothing(self, worker):
        for node in range(settings.NUM_NODES - 1):
            assert worker.handle_window(_window(node, time.time())) == []

    def test_complete_array_emits_one_payload_per_node(self, worker):
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now)) or payloads

        assert len(payloads) == settings.NUM_NODES
        for payload in payloads:
            assert len(payload["gnn_embedding"]) == settings.GNN_EMBEDDING_DIM
            assert 0.0 <= payload["anomaly_score"] <= 1.0
            assert 0.0 <= payload["ttf_prediction"] <= 1.0
            assert payload["anomaly_severity"] in {"normal", "warning", "critical"}

    def test_detection_happens_in_the_worker(self, worker):
        """
        Scores must be computed here from model output. The API previously
        accepted them from its caller and forwarded them to Prometheus, so
        nothing in the system actually detected anything.
        """
        worker.scorer.warmup_frames = 5
        worker.scorer.z_threshold = 3.0

        for step in range(20):
            now = time.time() + step
            for node in range(settings.NUM_NODES):
                worker.handle_window(_window(node, now, amplitude=0.05))

        payloads: list[dict] = []
        final = time.time() + 100
        for node in range(settings.NUM_NODES):
            amplitude = 40.0 if node == 1 else 0.05
            result = worker.handle_window(_window(node, final, amplitude=amplitude))
            payloads = result or payloads

        by_node = {p["node_id"]: p for p in payloads}
        assert by_node[1]["is_anomaly"] is True
        assert by_node[1]["z_score"] > by_node[0]["z_score"]

    def test_payloads_are_accepted_by_the_api(self, worker, api_client):
        """The worker's output must satisfy the API's schema."""
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now)) or payloads

        for payload in payloads:
            assert api_client.post("/generate_telemetry", json=payload).status_code == 200

    def test_submit_survives_unreachable_api(self):
        w = InferenceWorker(inference_url="http://127.0.0.1:1", load_weights=False)
        try:
            assert w.submit({"node_id": 0}) is False
        finally:
            w.close()
