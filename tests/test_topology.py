"""Tests for the acoustic topology graph."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.mapping.topology_graph import build_acoustic_topology, topology_summary


class TestEdgeConstruction:
    def test_shapes_are_consistent(self, mic_coords):
        edge_index, edge_weight = build_acoustic_topology(mic_coords)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_weight.shape[0]
        assert edge_index.dtype == torch.long
        assert edge_weight.dtype == torch.float32

    def test_no_self_loops(self, mic_coords):
        edge_index, _ = build_acoustic_topology(mic_coords)
        assert (edge_index[0] != edge_index[1]).all()

    def test_graph_is_symmetric(self, mic_coords):
        """Acoustic coupling is mutual; a directed edge has no physical meaning."""
        edge_index, _ = build_acoustic_topology(mic_coords)
        edges = {(int(s), int(t)) for s, t in edge_index.t().tolist()}
        assert all((t, s) in edges for s, t in edges)

    def test_distant_nodes_are_not_connected(self):
        edge_index, edge_weight = build_acoustic_topology(
            [(0, 0, 0), (100, 0, 0)], distance_threshold=15.0
        )
        assert edge_index.shape[1] == 0
        assert edge_weight.shape[0] == 0

    def test_weights_normalized_to_unit_max(self, mic_coords):
        _, edge_weight = build_acoustic_topology(mic_coords)
        assert pytest.approx(1.0) == float(edge_weight.max())
        assert float(edge_weight.min()) > 0.0

    def test_normalization_can_be_disabled(self, mic_coords):
        _, raw = build_acoustic_topology(mic_coords, normalize=False)
        assert float(raw.max()) != pytest.approx(1.0)

    def test_closer_pairs_weigh_more(self):
        # Node 0 is 1m from node 1 and 10m from node 2.
        coords = [(0, 0, 0), (1, 0, 0), (10, 0, 0)]
        edge_index, edge_weight = build_acoustic_topology(coords, distance_threshold=20.0)
        weights = {
            (int(s), int(t)): float(w)
            for (s, t), w in zip(edge_index.t().tolist(), edge_weight.tolist(), strict=True)
        }
        assert weights[(0, 1)] > weights[(0, 2)]

    def test_decay_exponent_changes_falloff(self):
        coords = [(0, 0, 0), (1, 0, 0), (4, 0, 0)]
        _, linear = build_acoustic_topology(coords, 20.0, decay_exponent=1.0)
        _, square = build_acoustic_topology(coords, 20.0, decay_exponent=2.0)
        # Inverse-square concentrates weight harder on the nearest neighbour.
        assert float(square.min()) < float(linear.min())

    def test_colocated_microphones_do_not_divide_by_zero(self):
        _, edge_weight = build_acoustic_topology([(0, 0, 0), (0, 0, 0)])
        assert torch.isfinite(edge_weight).all()


class TestNearestNeighbourLimit:
    def test_limits_degree(self):
        # Eight microphones on a line, all within threshold of each other.
        coords = [(float(i), 0.0, 0.0) for i in range(8)]
        full, _ = build_acoustic_topology(coords, distance_threshold=100.0)
        limited, _ = build_acoustic_topology(coords, distance_threshold=100.0, max_neighbours=2)
        assert limited.shape[1] < full.shape[1]

    def test_result_stays_symmetric(self):
        coords = [(float(i), 0.0, 0.0) for i in range(6)]
        edge_index, _ = build_acoustic_topology(coords, distance_threshold=100.0, max_neighbours=2)
        edges = {(int(s), int(t)) for s, t in edge_index.t().tolist()}
        assert all((t, s) in edges for s, t in edges)

    def test_dense_array_without_limit_becomes_complete(self):
        """
        A complete graph makes the GCN a plain mean, discarding exactly the
        spatial structure the architecture exists to exploit.
        """
        coords = [(float(i), 0.0, 0.0) for i in range(5)]
        summary = topology_summary(coords, distance_threshold=100.0)
        assert summary["density"] == pytest.approx(1.0)

    def test_zero_neighbours_rejected(self, mic_coords):
        with pytest.raises(ValueError, match="max_neighbours"):
            build_acoustic_topology(mic_coords, max_neighbours=0)


class TestValidation:
    def test_wrong_coordinate_arity_rejected(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            build_acoustic_topology([(0, 0), (1, 1)])

    def test_non_positive_threshold_rejected(self, mic_coords):
        with pytest.raises(ValueError, match="distance_threshold"):
            build_acoustic_topology(mic_coords, distance_threshold=0.0)

    def test_single_microphone_yields_empty_graph(self):
        edge_index, edge_weight = build_acoustic_topology([(0, 0, 0)])
        assert edge_index.shape == (2, 0)
        assert edge_weight.shape == (0,)

    def test_accepts_numpy_input(self, mic_coords):
        edge_index, _ = build_acoustic_topology(np.asarray(mic_coords))
        assert edge_index.shape[0] == 2


class TestSummary:
    def test_reports_expected_keys(self, mic_coords):
        summary = topology_summary(mic_coords)
        assert {
            "num_nodes",
            "num_edges",
            "density",
            "mean_degree",
            "isolated_nodes",
            "mean_weight",
        } == set(summary)

    def test_counts_isolated_nodes(self):
        coords = [(0, 0, 0), (1, 0, 0), (500, 0, 0)]
        summary = topology_summary(coords, distance_threshold=10.0)
        assert summary["isolated_nodes"] == 1

    def test_empty_graph_summarises_cleanly(self):
        summary = topology_summary([(0, 0, 0), (999, 0, 0)], distance_threshold=1.0)
        assert summary["num_edges"] == 0
        assert summary["mean_weight"] == 0.0
        assert summary["isolated_nodes"] == 2
