"""Tests for the ingestion-side spatial probe and its worker integration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ingestion.spatial_probe import SpatialProbe, SpatialSnapshot
from src.mapping.tdoa import SPEED_OF_SOUND, TDOAEstimate

FS = 16_000
COORDS = np.array([(0.0, 0.0, 3.0), (5.0, 0.0, 3.0), (0.0, 10.0, 3.0), (5.0, 10.0, 3.0)])


def _array_audio(source, n=8000, seed=5):
    """Waveforms as heard by each microphone from one broadband source."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n + 4096)
    out = []
    for p in COORDS:
        delay = round(np.linalg.norm(np.asarray(source) - p) / SPEED_OF_SOUND * FS)
        out.append(signal[delay : delay + n])
    return out


@pytest.fixture
def probe():
    return SpatialProbe(mic_coords=COORDS, sample_rate=FS, staleness_tolerance=0.5)


class TestReadiness:
    def test_not_ready_until_every_node_reports(self, probe):
        channels = _array_audio((3.5, 7.0, 3.0))
        for node in range(3):
            probe.push(node, 100.0, channels[node])
            assert not probe.is_ready()

        probe.push(3, 100.0, channels[3])
        assert probe.is_ready()

    def test_stale_array_is_not_ready(self, probe):
        """
        Chunks from different moments describe different acoustics. Correlating
        them would produce a confident, meaningless delay.
        """
        channels = _array_audio((3.5, 7.0, 3.0))
        for node in range(3):
            probe.push(node, 100.0, channels[node])
        probe.push(3, 100.0 + 5.0, channels[3])  # 5 s late, tolerance is 0.5

        assert not probe.is_ready()
        assert probe.solve() is None

    def test_newer_chunk_replaces_older_for_same_node(self, probe):
        channels = _array_audio((3.5, 7.0, 3.0))
        probe.push(0, 100.0, channels[0])
        probe.push(0, 100.1, channels[0])
        for node in range(1, 4):
            probe.push(node, 100.1, channels[node])
        assert probe.is_ready()

    def test_unknown_node_ignored(self, probe):
        probe.push(99, 100.0, np.zeros(8000))
        assert not probe.is_ready()

    def test_solve_clears_buffer(self, probe):
        channels = _array_audio((3.5, 7.0, 3.0))
        for node in range(4):
            probe.push(node, 100.0, channels[node])
        assert probe.solve() is not None
        assert not probe.is_ready()

    def test_reset(self, probe):
        probe.push(0, 100.0, np.zeros(8000))
        probe.reset()
        assert not probe.is_ready()


class TestSolve:
    def test_localizes_a_real_source(self, probe):
        source = (3.5, 7.0, 3.0)
        for node, wave in enumerate(_array_audio(source)):
            probe.push(node, 100.0, wave)

        snapshot = probe.solve()
        assert snapshot is not None
        assert snapshot.localized
        assert np.linalg.norm(snapshot.position - np.asarray(source)) < 0.5
        assert snapshot.mean_coherence > 0.5
        assert len(snapshot.estimates) == 6

    def test_clock_spread_is_reported(self, probe):
        for node, wave in enumerate(_array_audio((3.5, 7.0, 3.0))):
            probe.push(node, 100.0 + node * 0.05, wave)

        snapshot = probe.solve()
        assert snapshot is not None
        assert snapshot.clock_spread == pytest.approx(0.15, abs=1e-6)

    def test_defaults_to_the_array_plane(self, probe):
        """A coplanar array cannot observe elevation; z is pinned to the plane."""
        assert probe.plane_z == pytest.approx(3.0)
        for node, wave in enumerate(_array_audio((1.0, 2.0, 3.0))):
            probe.push(node, 100.0, wave)
        assert probe.solve().position[2] == pytest.approx(3.0)

    def test_incoherent_array_yields_no_position(self, probe):
        rng = np.random.default_rng(2)
        for node in range(4):
            probe.push(node, 100.0, rng.standard_normal(8000))

        snapshot = probe.solve()
        assert snapshot is not None
        assert not snapshot.localized
        assert snapshot.mean_coherence < 0.3

    def test_ragged_lengths_are_truncated_not_rejected(self, probe):
        channels = _array_audio((3.5, 7.0, 3.0))
        for node, wave in enumerate(channels):
            probe.push(node, 100.0, wave[: 8000 - node * 10])
        assert probe.solve() is not None

    def test_degenerate_short_chunks_return_none(self, probe):
        for node in range(4):
            probe.push(node, 100.0, np.zeros(1))
        assert probe.solve() is None

    def test_rejects_bad_coords(self):
        with pytest.raises(ValueError):
            SpatialProbe(mic_coords=np.zeros((4, 2)), sample_rate=FS)


class TestPayload:
    def test_roundtrips_through_msgpack(self, probe):
        import msgpack

        for node, wave in enumerate(_array_audio((3.5, 7.0, 3.0))):
            probe.push(node, 100.0, wave)
        payload = probe.solve().to_payload()

        decoded = msgpack.unpackb(msgpack.packb(payload, use_bin_type=True), raw=False)
        assert len(decoded["pairs"]) == 6
        assert len(decoded["position"]) == 3
        assert 0.0 <= decoded["mean_coherence"] <= 1.0

    def test_infinite_residual_serialized_as_null(self):
        """`inf` has no MessagePack representation that survives JSON bridges."""
        snapshot = SpatialSnapshot(
            timestamp=1.0,
            estimates=[],
            position=None,
            residual=float("inf"),
            clock_spread=0.0,
        )
        assert snapshot.to_payload()["residual"] is None

    def test_coherence_map_is_symmetric(self):
        snapshot = SpatialSnapshot(
            timestamp=1.0,
            estimates=[TDOAEstimate(0, 2, 0.001, 0.7, 0.02)],
            position=None,
            residual=0.0,
            clock_spread=0.0,
        )
        mapping = snapshot.coherence_map()
        assert mapping[(0, 2)] == mapping[(2, 0)] == pytest.approx(0.7)


class TestWorkerIntegration:
    """The probe's output must actually change what the graph does."""

    @pytest.fixture
    def worker(self):
        from src.inference.worker import InferenceWorker

        return InferenceWorker(load_weights=False, inference_url="http://test.invalid")

    def test_spatial_snapshot_reweights_the_graph(self, worker):
        before = worker.effective_edge_weight.clone()

        worker.apply_spatial(
            {
                "position": [2.0, 5.0, 3.0],
                "pairs": [
                    {"i": i, "j": j, "tau": 0.0, "coherence": 0.0}
                    for i in range(4)
                    for j in range(i + 1, 4)
                ],
            }
        )
        after = worker.effective_edge_weight

        # Every pair fully decorrelated -> every edge attenuated to the floor.
        assert not torch.allclose(before, after)
        assert (after < before).all()

    def test_full_coherence_restores_geometric_weights(self, worker):
        worker.apply_spatial(
            {
                "position": None,
                "pairs": [
                    {"i": i, "j": j, "tau": 0.0, "coherence": 1.0}
                    for i in range(4)
                    for j in range(i + 1, 4)
                ],
            }
        )
        assert torch.allclose(worker.effective_edge_weight, worker.edge_weight, atol=1e-6)

    def test_graph_never_fully_disconnects(self, worker):
        """A zero-weight graph collapses the GCN into a per-node MLP."""
        worker.apply_spatial(
            {
                "position": None,
                "pairs": [
                    {"i": i, "j": j, "tau": 0.0, "coherence": 0.0}
                    for i in range(4)
                    for j in range(i + 1, 4)
                ],
            }
        )
        assert (worker.effective_edge_weight > 0).all()

    def test_source_position_reaches_the_payload(self, worker):
        worker.apply_spatial({"position": [2.0, 5.0, 3.0], "pairs": []})

        window = _make_window(0)
        worker.assembler.push(window)
        for node in range(1, 4):
            worker.assembler.push(_make_window(node))

        x, timespans, snapshot = worker.assembler.assemble()
        payloads = worker.infer(x, timespans, snapshot)

        assert payloads
        assert all(p["source_position"] == [2.0, 5.0, 3.0] for p in payloads)

    def test_no_spatial_data_leaves_payload_clean(self, worker):
        worker.assembler.push(_make_window(0))
        for node in range(1, 4):
            worker.assembler.push(_make_window(node))

        x, timespans, snapshot = worker.assembler.assemble()
        payloads = worker.infer(x, timespans, snapshot)
        assert all("source_position" not in p for p in payloads)


def _make_window(node_id: int):
    """A minimal NodeWindow matching the configured pipeline shapes."""
    from src.inference.worker import NodeWindow
    from src.settings import settings

    rng = np.random.default_rng(node_id)
    return NodeWindow(
        node_id=node_id,
        features=rng.standard_normal((settings.SEQ_LENGTH, settings.N_MELS)).astype(np.float32),
        timespans=np.full(settings.SEQ_LENGTH, 0.5, dtype=np.float32),
        timestamp=1000.0,
        latest_frame=rng.standard_normal((settings.N_MELS, settings.MEL_FRAMES_PER_CHUNK)).astype(
            np.float32
        ),
    )
