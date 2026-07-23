"""Unit tests for the Spatio-Temporal GNN and topology graph."""


import pytest
import torch


from src.mapping.st_gnn_model import SpatioTemporalGNN, TemporalAttentionBlock, SpatialGCNBlock
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
        edge_index, edge_weight = build_acoustic_topology(mics, distance_threshold=15.0)


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
            in_channels=64, hidden_channels=128, embedding_dim=256,
            num_nodes=num_nodes, num_heads=4, num_gcn_layers=2,
        )
        x = torch.randn(4, 50, num_nodes * 64)  # (batch, seq, nodes*features)
        out = model(x, edge_index, edge_weight)
        assert out.shape == (4, 256)


    def test_batch_size_1(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64, hidden_channels=64, embedding_dim=128,
            num_nodes=num_nodes,
        )
        x = torch.randn(1, 10, num_nodes * 64)
        out = model(x, edge_index, edge_weight)
        assert out.shape == (1, 128)


    def test_gradient_flow(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64, hidden_channels=64, embedding_dim=128,
            num_nodes=num_nodes,
        )
        x = torch.randn(2, 10, num_nodes * 64, requires_grad=True)
        out = model(x, edge_index, edge_weight)
        out.sum().backward()
        assert x.grad is not None


    def test_deterministic(self, sample_edge_topology):
        edge_index, edge_weight, num_nodes = sample_edge_topology
        model = SpatioTemporalGNN(
            in_channels=64, hidden_channels=64, embedding_dim=128,
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
            in_channels=64, hidden_channels=128, embedding_dim=256,
            num_nodes=num_nodes,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert 10_000 < n_params < 10_000_000  # Between 10K and 10M