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
import torch

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


class TestDegradedArray:
    """
    One dead microphone must not silence the array.

    The assembler previously required every node with no timeout and no
    eviction, so a single dropout meant the healthy microphones streamed
    indefinitely and nothing was ever emitted. The dashboard then showed
    "waiting for model predictions", which is visually identical to a healthy,
    quiet plant — the worst available failure mode for a monitoring system.
    """

    def _assembler(self, **kwargs):
        defaults = {
            "num_nodes": 4,
            "seq_length": 8,
            "n_mels": settings.N_MELS,
            "staleness_tolerance": 5.0,
            "max_wait": 1.0,
            "min_nodes": 2,
        }
        return WindowAssembler(**{**defaults, **kwargs})

    def test_dead_microphone_releases_a_degraded_snapshot(self):
        assembler = self._assembler()
        for node in range(3):
            assembler.push(_window(node, 100.0))

        # Inside the grace period the array still waits for the fourth mic.
        assert not assembler.is_complete()

        # Past it, three healthy microphones are worth more than silence.
        assert assembler.is_complete(now=time.monotonic() + 5.0)

    def test_degraded_snapshot_zero_fills_and_omits_the_absent_node(self):
        assembler = self._assembler()
        for node in range(3):
            assembler.push(_window(node, 100.0))
        assert assembler.is_complete(now=time.monotonic() + 5.0)

        x, timespans, snapshot = assembler.assemble()

        # Fixed tensor shape: the topology is indexed by position, so a missing
        # node has to keep its slot or every edge after it misaligns.
        assert x.shape == (1, 8, 4 * settings.N_MELS)
        assert timespans.shape == (1, 8)
        mels = settings.N_MELS
        assert torch.allclose(x[0, :, 3 * mels : 4 * mels], torch.zeros(8, mels))

        # ...but no telemetry is fabricated for a microphone that said nothing.
        assert set(snapshot) == {0, 1, 2}
        assert assembler.last_missing == {3}

    def test_stale_node_is_evicted_so_the_array_recovers(self):
        """
        A stale window must be dropped, not retained. Left in place, the spread
        between newest and oldest never falls back inside tolerance, so even the
        surviving microphones stop producing snapshots — a permanent stall that
        no metric reported.
        """
        assembler = self._assembler(num_nodes=2, min_nodes=1)
        assembler.push(_window(0, 100.0))
        assembler.push(_window(1, 100.0))
        assert assembler.is_complete()
        assembler.clear()

        # Node 1 dies; node 0 keeps reporting well past the staleness bound.
        assembler.push(_window(1, 200.0))
        for step in range(5):
            assembler.push(_window(0, 260.0 + step))

        # The stale window is dropped rather than blocking forever, so the
        # surviving microphone can still produce a snapshot.
        assert assembler.is_complete(now=time.monotonic() + 5.0)
        assert 1 not in assembler._windows
        assert set(assembler.assemble()[2]) == {0}

    def test_below_quorum_emits_nothing(self):
        """A graph too sparse to convolve is worse than no answer."""
        assembler = self._assembler(min_nodes=3)
        for node in range(2):
            assembler.push(_window(node, 100.0))
        assert not assembler.is_complete(now=time.monotonic() + 60.0)

    def test_window_of_the_wrong_length_is_rejected_not_raised(self):
        """
        A SEQ_LENGTH mismatch between ingestion and the worker used to reach
        assemble() and raise on the reshape, once per snapshot, forever. It is
        a configuration error, so it is reported once per node and dropped.
        """
        assembler = self._assembler()
        assert assembler.push(_window(0, 100.0, seq=8)) is True
        assert assembler.push(_window(1, 100.0, seq=7)) is False
        assert 1 not in assembler._windows


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

    def test_forecast_and_embedding_are_per_node(self, worker):
        """
        Each card on the dashboard must describe its own machine.

        A single facility-level TTF and graph embedding were previously computed
        from the pooled graph readout and copied into every node's payload, so
        the per-node cards were identical by construction. An operator reading
        four "Node N — 47.3%" tiles was reading one number four times.
        """
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            # Give each microphone a genuinely different acoustic picture.
            amplitude = 0.05 * (node + 1) ** 3
            payloads = worker.handle_window(_window(node, now, amplitude=amplitude)) or payloads

        assert len(payloads) == settings.NUM_NODES

        embeddings = {tuple(p["gnn_embedding"]) for p in payloads}
        assert len(embeddings) == settings.NUM_NODES, "every node must get its own embedding"

        forecasts = {p["ttf_prediction"] for p in payloads}
        assert len(forecasts) > 1, "forecasts must not be one pooled number copied N times"

    def test_degraded_array_still_emits_telemetry(self, worker):
        """
        End to end: a dead microphone must not stop the other three.

        ``max_wait=0`` releases as soon as the quorum is met, which keeps the
        test deterministic without sleeping; the quorum is set to N-1 so the
        snapshot fires on exactly the three surviving microphones.
        """
        worker.assembler = WindowAssembler(
            num_nodes=settings.NUM_NODES,
            seq_length=8,
            n_mels=settings.N_MELS,
            max_wait=0.0,
            min_nodes=settings.NUM_NODES - 1,
        )

        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES - 1):
            payloads = worker.handle_window(_window(node, now)) or payloads

        reporting = {p["node_id"] for p in payloads}
        assert reporting == set(range(settings.NUM_NODES - 1))
        assert settings.NUM_NODES - 1 not in reporting

    def test_flagged_frames_carry_spectral_evidence(self, worker):
        """
        "Node 3, anomaly score 0.87" is not actionable. A technician needs to
        know what the system heard, and which catalogued fault that resembles.
        """
        # The autoencoder is resident (if untrained) — enough for the error map
        # to decompose, which is all the attribution needs.
        worker.weights_loaded = True
        worker.scorer.warmup_frames = 3

        payloads: list[dict] = []
        for step in range(8):
            now = time.time() + step
            for node in range(settings.NUM_NODES):
                worker.handle_window(_window(node, now, amplitude=0.05))

        final = time.time() + 100
        for node in range(settings.NUM_NODES):
            amplitude = 60.0 if node == 1 else 0.05
            payloads = worker.handle_window(_window(node, final, amplitude=amplitude)) or payloads

        flagged = [p for p in payloads if p["is_anomaly"]]
        assert flagged, "expected the loud node to flag"
        for payload in flagged:
            assert payload["explanation"]["bands"]
            assert payload["explanation"]["summary"]
            assert payload["diagnosis"]["fault"]
            assert payload["diagnosis"]["urgency"] in {"monitor", "schedule", "urgent"}

    def test_quiet_frames_pay_nothing_for_attribution(self, worker):
        """Attribution is off the steady-state path; most frames are normal."""
        worker.weights_loaded = True
        now = time.time()
        payloads: list[dict] = []
        for node in range(settings.NUM_NODES):
            payloads = worker.handle_window(_window(node, now, amplitude=0.05)) or payloads

        assert payloads
        assert all(not p["is_anomaly"] for p in payloads)
        assert all("explanation" not in p for p in payloads)

    def test_untrained_worker_does_not_invent_an_explanation(self, worker):
        """
        Without a trained autoencoder the scorer falls back to frame energy, so
        there is no reconstruction-error map to decompose and any attribution
        would be fabricated.
        """
        assert worker.weights_loaded is False
        worker.scorer.warmup_frames = 1

        payloads: list[dict] = []
        for step in range(4):
            now = time.time() + step
            for node in range(settings.NUM_NODES):
                amplitude = 60.0 if (node == 1 and step == 3) else 0.05
                result = worker.handle_window(_window(node, now, amplitude=amplitude))
                payloads = result or payloads

        assert all("explanation" not in p for p in payloads)

    def test_submit_survives_unreachable_api(self):
        w = InferenceWorker(inference_url="http://127.0.0.1:1", load_weights=False)
        try:
            assert w.submit({"node_id": 0}) is False
        finally:
            w.close()
