"""Unit tests for the Spatio-Temporal GNN and topology graph."""

import torch

from src.mapping.st_gnn_model import (
    _EDGE_CACHE_MAX_ENTRIES,
    SpatialGCNBlock,
    SpatioTemporalGNN,
    TemporalAttentionBlock,
)
from src.mapping.topology_graph import build_acoustic_topology


class TestTopologyGraph:
    def test_build_topology_shapes(self):
        mics = [(0, 0, 0), (5, 0, 0), (0, 10, 0), (5, 10, 0)]
        edge_index, edge_weight = build_acoustic_topology(mics)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_weight.shape[0]
        assert edge_weight.dtype == torch.float32

    def test_distance_threshold_filters_edges(self):
        mics = [(0, 0, 0), (100, 0, 0)]  # 100m apart
        edge_index, _ = build_acoustic_topology(mics, distance_threshold=15.0)

        # Should have no edges since mics are 100m apart
        assert edge_index.shape[1] == 0

    def test_edge_weights_normalized(self):
        mics = [(0, 0, 0), (3, 0, 0), (0, 4, 0)]
        _, edge_weight = build_acoustic_topology(mics)

        assert edge_weight.max() <= 1.0
        assert edge_weight.min() >= 0.0

    def test_self_loops_excluded(self):
        mics = [(0, 0, 0), (5, 0, 0)]
        edge_index, _ = build_acoustic_topology(mics)

        for i in range(edge_index.shape[1]):
            assert edge_index[0, i] != edge_index[1, i]


class TestBatchedTopologyCache:
    """
    The batched-topology cache must never return edges for a different graph.

    It was keyed partly on ``edge_index.data_ptr()``, which is unique only while
    the tensor is alive: once freed, the allocator may hand the same address to
    an unrelated tensor and the cache would serve a topology belonging to a
    graph that no longer exists. Silently wrong edges are the worst failure mode
    available to a spatial model — the output is plausible and unverifiable.
    """

    def _model(self, num_nodes: int) -> SpatioTemporalGNN:
        return SpatioTemporalGNN(
            in_channels=8,
            hidden_channels=16,
            embedding_dim=16,
            num_nodes=num_nodes,
            num_heads=2,
        )

    def test_a_different_topology_of_the_same_shape_is_not_a_cache_hit(self):
        mics = [(0, 0, 0), (5, 0, 0), (0, 10, 0), (5, 10, 0)]
        edge_index, edge_weight = build_acoustic_topology(mics)
        model = self._model(len(mics)).eval()
        x = torch.randn(1, 4, len(mics) * 8)

        with torch.no_grad():
            geometric = model(x, edge_index, edge_weight)
            # Same shape, same key, entirely different coupling.
            uniform = model(x, edge_index, torch.ones_like(edge_weight))

        assert not torch.allclose(geometric, uniform), (
            "the cache returned a stale topology for a different edge weighting"
        )

    def test_cache_is_bounded_and_keeps_the_hot_entry(self):
        """
        Eviction must be least-recently-used, not a full flush. Clearing the
        whole cache on overflow throws away the entry the training loop is about
        to request again, so an occasional odd batch size costs a full rebuild
        on the very next step.
        """
        mics = [(0, 0, 0), (5, 0, 0), (0, 10, 0)]
        edge_index, edge_weight = build_acoustic_topology(mics)
        model = self._model(len(mics))
        device = torch.device("cpu")

        last = _EDGE_CACHE_MAX_ENTRIES + 5
        for num_graphs in range(1, last + 1):
            model._batched_topology(edge_index, edge_weight, num_graphs, device)

        assert len(model._edge_cache) <= _EDGE_CACHE_MAX_ENTRIES
        assert (last, "cpu", edge_index.shape[1]) in model._edge_cache


class TestPerNodeEmbeddings:
    def test_return_nodes_yields_one_trajectory_per_microphone(self):
        """
        The graph readout pools the array into one vector, so anything shown to
        an operator as a per-machine reading has to come from the node-level
        embeddings instead.
        """
        mics = [(0, 0, 0), (5, 0, 0), (0, 10, 0), (5, 10, 0)]
        edge_index, edge_weight = build_acoustic_topology(mics)
        model = SpatioTemporalGNN(
            in_channels=8, hidden_channels=16, embedding_dim=16, num_nodes=4, num_heads=2
        ).eval()

        batch, seq = 2, 5
        x = torch.randn(batch, seq, 4 * 8)
        with torch.no_grad():
            graph, nodes = model(
                x, edge_index, edge_weight, return_sequence=True, return_nodes=True
            )

        assert graph.shape == (batch, seq, 16)
        assert nodes.shape == (batch, seq, 4, 16)
        # Distinct microphones must not collapse to the same representation.
        assert not torch.allclose(nodes[0, 0, 0], nodes[0, 0, 1])

    def test_default_call_is_unchanged(self):
        """Existing callers and trained weights must keep their contract."""
        mics = [(0, 0, 0), (5, 0, 0)]
        edge_index, edge_weight = build_acoustic_topology(mics)
        model = SpatioTemporalGNN(
            in_channels=8, hidden_channels=16, embedding_dim=16, num_nodes=2, num_heads=2
        ).eval()
        x = torch.randn(1, 4, 2 * 8)

        with torch.no_grad():
            out = model(x, edge_index, edge_weight)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 16)


class TestTemporalAttention:
    def test_output_shape(self):
        block = TemporalAttentionBlock(input_dim=64, num_heads=4)
        x = torch.randn(8, 50, 64)  # (batch*nodes, seq, features)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        block = TemporalAttentionBlock(input_dim=32, num_heads=2)
        x = torch.randn(4, 10, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestSpatialGCNBlock:
    def test_output_shape(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        block = SpatialGCNBlock(64, 128, 64)
        x = torch.randn(num_nodes, 64)
        out = block(x, edge_index, edge_weight)
        assert out.shape == (num_nodes, 64)

    def test_residual_projection(self):
        block = SpatialGCNBlock(32, 64, 128)  # in != out, should use projection
        assert not isinstance(block.residual, torch.nn.Identity)


class TestSpatioTemporalGNN:
    def test_forward_shape(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64,
            hidden_channels=128,
            embedding_dim=256,
            num_nodes=num_nodes,
            num_heads=4,
            num_gcn_layers=2,
        )
        x = torch.randn(4, 50, num_nodes * 64)  # (batch, seq, nodes*features)
        out = model(x, edge_index, edge_weight)
        assert out.shape == (4, 256)

    def test_batch_size_1(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64,
            hidden_channels=64,
            embedding_dim=128,
            num_nodes=num_nodes,
        )
        x = torch.randn(1, 10, num_nodes * 64)
        out = model(x, edge_index, edge_weight)
        assert out.shape == (1, 128)

    def test_gradient_flow(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64,
            hidden_channels=64,
            embedding_dim=128,
            num_nodes=num_nodes,
        )
        x = torch.randn(2, 10, num_nodes * 64, requires_grad=True)
        out = model(x, edge_index, edge_weight)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64,
            hidden_channels=64,
            embedding_dim=128,
            num_nodes=num_nodes,
        )
        model.eval()
        x = torch.randn(2, 10, num_nodes * 64)
        with torch.no_grad():
            out1 = model(x, edge_index, edge_weight)
            out2 = model(x, edge_index, edge_weight)
        assert torch.allclose(out1, out2)

    def test_param_count_reasonable(self, sample_edge_topology):
        _, _, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64,
            hidden_channels=128,
            embedding_dim=256,
            num_nodes=num_nodes,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert 10_000 < n_params < 10_000_000  # Between 10K and 10M
