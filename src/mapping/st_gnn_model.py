"""
Spatio-Temporal Graph Neural Network (ST-GNN) for acoustic source localization
and feature extraction.


The network models the factory floor as a topological graph where:
- Nodes = microphone positions
- Edges = acoustic coupling between nearby sensors (inverse-distance weighted)


Architecture:
    1. Per-node temporal attention (captures frequency drift over time)
    2. Spatial GCN layers (propagates spatial acoustic correlations)
    3. Graph-level readout → dense embedding for downstream LLM / LNN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



class TemporalAttentionBlock(nn.Module):
    """
    Multi-head self-attention over the time axis of each node's spectrogram
    sequence. Captures temporal patterns like frequency drift, transient
    impulses, and periodic machinery signatures.
    """


    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * num_nodes, seq_len, features)
        Returns:
            (batch * num_nodes, seq_len, features)
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))



class SpatialGCNBlock(nn.Module):
    """
    Two-layer Graph Convolutional block that propagates acoustic information
    across physically connected microphone nodes.
    """


    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)


        # Residual projection if dimensions change
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (num_nodes_total, features)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,)
        Returns:
            (num_nodes_total, out_channels)
        """
        identity = self.residual(x)


        out = self.conv1(x, edge_index, edge_weight)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)


        out = self.conv2(out, edge_index, edge_weight)
        out = self.norm2(out)


        return F.gelu(out + identity)



class SpatioTemporalGNN(nn.Module):
    """
    Full ST-GNN pipeline:
        Raw spectrograms → Temporal Attention → Spatial GCN → Graph Readout → Embedding


    Parameters
    ----------
    in_channels : int
        Feature dimension per node per timestep (e.g., 64 mel bins).
    hidden_channels : int
        Width of GCN hidden layers.
    embedding_dim : int
        Output embedding size fed to downstream LLM / LNN.
    num_nodes : int
        Number of microphone nodes in the topology.
    num_heads : int
        Attention heads for temporal self-attention.
    num_gcn_layers : int
        Number of stacked spatial GCN blocks.
    dropout : float
        Dropout probability throughout the network.
    """


    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_nodes: int,
        num_heads: int = 4,
        num_gcn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim


        # -- Stage 1: Temporal feature extraction per node --
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.temporal_attention = TemporalAttentionBlock(
            hidden_channels, num_heads=num_heads, dropout=dropout
        )
        # Collapse the temporal dimension into a fixed-size vector
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)


        # -- Stage 2: Spatial GCN layers --
        self.gcn_blocks = nn.ModuleList()
        for i in range(num_gcn_layers):
            c_in = hidden_channels if i == 0 else hidden_channels
            self.gcn_blocks.append(
                SpatialGCNBlock(c_in, hidden_channels, hidden_channels, dropout=dropout)
            )


        # -- Stage 3: Graph readout → dense embedding --
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        batch_size: int = None,
        seq_length: int = None,
    ) -> torch.Tensor:
        """
        Forward pass.


        Args:
            x: (batch, seq_len, num_nodes * in_channels)  — flattened node features
               OR (batch, seq_len, num_nodes, in_channels) — structured
            edge_index: (2, num_edges) — graph topology
            edge_weight: (num_edges,) — edge weights
            batch_size: int — inferred from x if not provided
            seq_length: int — inferred from x if not provided


        Returns:
            embeddings: (batch, embedding_dim) — graph-level acoustic embedding
        """
        # -- Reshape input --
        if x.dim() == 3:
            # (batch, seq, nodes*features) → (batch, seq, nodes, features)
            B, S, _ = x.shape
            x = x.view(B, S, self.num_nodes, self.in_channels)
        else:
            B, S = x.shape[0], x.shape[1]


        # -- Stage 1: Temporal attention per node --
        # Reshape to (B * num_nodes, S, in_channels) so each node gets its own
        # temporal attention pass
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, nodes, S, C)
        x = x.view(B * self.num_nodes, S, self.in_channels)


        x = self.input_proj(x)                    # (B*nodes, S, hidden)
        x = self.temporal_attention(x)             # (B*nodes, S, hidden)


        # Pool over time: (B*nodes, S, hidden) → (B*nodes, hidden)
        x = x.permute(0, 2, 1)                    # (B*nodes, hidden, S)
        x = self.temporal_pool(x).squeeze(-1)      # (B*nodes, hidden)


        # -- Stage 2: Spatial GCN --
        # We need to expand edge_index for the full batch (B graphs in parallel).
        # PyG convention: offset node indices per graph in the batch.
        N = self.num_nodes
        device = x.device


        # Build batched edge_index by offsetting each graph's node indices
        batch_edge_indices = []
        for b in range(B):
            batch_edge_indices.append(edge_index + b * N)
        batched_edge_index = torch.cat(batch_edge_indices, dim=1).to(device)


        # Repeat edge weights for each graph in the batch
        if edge_weight is not None:
            batched_edge_weight = edge_weight.repeat(B).to(device)
        else:
            batched_edge_weight = None


        for gcn_block in self.gcn_blocks:
            x = gcn_block(x, batched_edge_index, batched_edge_weight)


        # -- Stage 3: Graph readout --
        # Create a batch vector for global_mean_pool: [0,0,0,0, 1,1,1,1, ...]
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)
        x = global_mean_pool(x, batch_vec)         # (B, hidden)


        embeddings = self.readout(x)               # (B, embedding_dim)
        return embeddings



if __name__ == "__main__":
    # Quick sanity test
    from topology_graph import build_acoustic_topology


    mics = [
        (0.0, 0.0, 3.0),
        (5.0, 0.0, 3.0),
        (0.0, 10.0, 3.0),
        (5.0, 10.0, 3.0),
    ]
    edge_index, edge_weight = build_acoustic_topology(mics)


    model = SpatioTemporalGNN(
        in_channels=64,
        hidden_channels=128,
        embedding_dim=256,
        num_nodes=4,
    )


    # Simulated input: 8 batches, 50 timesteps, 4 nodes * 64 mel bins
    x = torch.randn(8, 50, 4 * 64)
    out = model(x, edge_index, edge_weight)


    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[*] ST-GNN output: {out.shape}")       # (8, 256)
    print(f"[*] Parameters: {n_params:.2f}M")