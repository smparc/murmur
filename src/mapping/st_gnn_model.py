"""
Spatio-Temporal Graph Neural Network (ST-GNN) for acoustic feature extraction.

The factory floor is modelled as a graph:

- **Nodes** — microphone positions
- **Edges** — acoustic coupling between nearby sensors (distance-weighted)

Pipeline::

    spectrograms -> temporal attention -> spatial GCN -> readout -> embedding

The model can emit either a single graph-level embedding (``(B, E)``) or a full
per-timestep sequence (``(B, S, E)``). The sequence form exists because the
downstream Liquid Network is a *continuous-time* model: feeding it a single
pooled vector repeated across time gives it a constant signal and wastes the
entire reason for choosing that architecture.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Distinct (num_graphs, device, num_edges) combinations worth keeping. Training
# sees at most two (a full batch and the final short one); a serving process
# sees one.
_EDGE_CACHE_MAX_ENTRIES = 32


@dataclass
class _CachedTopology:
    """A batched topology plus the tensors it was derived from."""

    source_index: torch.Tensor
    source_weight: torch.Tensor | None
    batched_index: torch.Tensor
    batched_weight: torch.Tensor | None


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal position signal added before temporal self-attention.

    Self-attention is permutation-invariant: without this, reversing or
    shuffling a node's spectrogram frames yields a bit-identical embedding.
    For a model whose purpose is detecting *temporal* signatures — a bearing
    fault ramping up, a transient impulse train — that is a silent correctness
    bug, not a refinement.

    Sinusoidal rather than learned, so a model trained at one window length
    still behaves sensibly if ``SEQ_LENGTH`` is retuned in production.
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        if dim % 2:
            raise ValueError(f"positional encoding dim must be even, got {dim}")

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, S, D)`` -> ``(B, S, D)``."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds positional encoding capacity {self.max_len}"
            )
        return x + self.pe[:, :seq_len]


class TemporalAttentionBlock(nn.Module):
    """
    Pre-norm multi-head self-attention over a node's spectrogram sequence.

    Captures frequency drift, transient impulses and periodic machinery
    signatures. Pre-norm (normalise then attend, add residual) is used rather
    than post-norm because it trains stably without a warmup schedule.
    """

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if input_dim % num_heads:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B * num_nodes, seq_len, features)`` in and out."""
        h = self.norm(x)
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class SpatialGCNBlock(nn.Module):
    """Two-layer residual graph convolution across physically coupled nodes."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(num_nodes_total, features)`` -> ``(num_nodes_total, out_channels)``."""
        identity = self.residual(x)

        out = self.conv1(x, edge_index, edge_weight)
        out = F.gelu(self.norm1(out))
        out = self.dropout(out)

        out = self.conv2(out, edge_index, edge_weight)
        out = self.norm2(out)

        return F.gelu(out + identity)


class SpatioTemporalGNN(nn.Module):
    """
    Full ST-GNN.

    Parameters
    ----------
    in_channels:
        Features per node per timestep (mel bins).
    hidden_channels:
        Width of the attention and GCN stack.
    embedding_dim:
        Output embedding size consumed by the LLM and LNN.
    num_nodes:
        Microphones in the topology.
    num_heads:
        Temporal attention heads.
    num_gcn_layers:
        Stacked spatial GCN blocks.
    dropout:
        Dropout probability throughout.
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
        if num_gcn_layers < 1:
            raise ValueError(f"num_gcn_layers must be >= 1, got {num_gcn_layers}")

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim

        # -- Stage 1: temporal feature extraction, per node --
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_channels)
        self.temporal_attention = TemporalAttentionBlock(
            hidden_channels, num_heads=num_heads, dropout=dropout
        )

        # -- Stage 2: spatial propagation --
        self.gcn_blocks = nn.ModuleList(
            SpatialGCNBlock(hidden_channels, hidden_channels, hidden_channels, dropout=dropout)
            for _ in range(num_gcn_layers)
        )

        # -- Stage 3: graph readout --
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Expanding the topology to cover a batch of graphs is pure index
        # arithmetic that depends only on (num_graphs, device). Recomputing it
        # every forward pass — as the previous implementation did, with a Python
        # loop and a host-to-device copy — is wasted work on the hot path.
        # Keyed on shape, not on tensor addresses. An earlier version keyed on
        # ``data_ptr()``, which is only unique while the tensor is alive: once a
        # topology tensor is freed the allocator may hand the same address to a
        # different one, and the cache would then return a batched topology
        # belonging to a graph that no longer exists. The entry therefore holds a
        # reference to the source tensors and confirms identity on hit — the
        # reference also keeps the address from being recycled in the first place.
        self._edge_cache: OrderedDict[tuple, _CachedTopology] = OrderedDict()

    def _batched_topology(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        num_graphs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replicate the topology across ``num_graphs`` disjoint graphs."""
        key = (num_graphs, str(device), int(edge_index.shape[1]))

        cached = self._edge_cache.get(key)
        if (
            cached is not None
            and cached.source_index is edge_index
            and cached.source_weight is edge_weight
        ):
            self._edge_cache.move_to_end(key)
            return cached.batched_index, cached.batched_weight

        device_index = edge_index.to(device)
        # (2, E) -> (G, 2, E) via broadcast offsets -> (2, G*E)
        offsets = (torch.arange(num_graphs, device=device) * self.num_nodes).view(-1, 1, 1)
        batched_index = (device_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

        batched_weight = None
        if edge_weight is not None:
            batched_weight = edge_weight.to(device).repeat(num_graphs)

        self._edge_cache[key] = _CachedTopology(
            source_index=edge_index,
            source_weight=edge_weight,
            batched_index=batched_index,
            batched_weight=batched_weight,
        )
        self._edge_cache.move_to_end(key)
        # Least-recently-used eviction. Flushing the whole cache on overflow —
        # as this previously did — throws away the hot entry the training loop
        # is about to ask for again, so a service that occasionally sees an odd
        # batch size pays a full rebuild on the very next step.
        while len(self._edge_cache) > _EDGE_CACHE_MAX_ENTRIES:
            self._edge_cache.popitem(last=False)
        return batched_index, batched_weight

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_sequence: bool = False,
        return_nodes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            ``(B, S, num_nodes * in_channels)`` flattened, or
            ``(B, S, num_nodes, in_channels)`` structured.
        edge_index:
            ``(2, num_edges)`` topology.
        edge_weight:
            ``(num_edges,)`` coupling strengths.
        return_sequence:
            When ``True``, return a per-timestep embedding sequence
            ``(B, S, embedding_dim)`` for continuous-time consumers. When
            ``False``, mean-pool over time to a single ``(B, embedding_dim)``.
        return_nodes:
            Also return per-node embeddings ``(B, S, num_nodes, embedding_dim)``.
            The graph readout pools the array into one vector, so a forecast
            derived from it describes the *facility*; anything presented to an
            operator as a per-machine reading has to come from this instead.

        Returns
        -------
        ``(B, embedding_dim)`` or ``(B, S, embedding_dim)``, or a
        ``(graph, nodes)`` pair when ``return_nodes`` is set.

        Cost note
        ---------
        The spatial GCN runs over ``B * S`` disjoint graphs — one per timestep
        rather than one per sample. That is what makes the spatial convolution
        time-resolved, and it is the dominant term in training cost: a batch of
        16 over a 50-step window convolves 800 graphs per forward pass.
        """
        if x.dim() == 3:
            B, S, flat = x.shape
            expected = self.num_nodes * self.in_channels
            if flat != expected:
                raise ValueError(
                    f"expected {expected} features (num_nodes * in_channels), got {flat}"
                )
            x = x.view(B, S, self.num_nodes, self.in_channels)
        elif x.dim() == 4:
            B, S = x.shape[0], x.shape[1]
            if x.shape[2] != self.num_nodes or x.shape[3] != self.in_channels:
                raise ValueError(
                    f"expected (B, S, {self.num_nodes}, {self.in_channels}), got {tuple(x.shape)}"
                )
        else:
            raise ValueError(f"expected a 3D or 4D input, got {x.dim()}D")

        N, device = self.num_nodes, x.device

        # -- Stage 1: temporal attention, independently per node --
        # (B, S, N, C) -> (B*N, S, C) so each microphone attends over its own
        # history without leaking across the array.
        h = x.permute(0, 2, 1, 3).reshape(B * N, S, self.in_channels)
        h = self.input_proj(h)
        h = self.pos_encoding(h)
        h = self.temporal_attention(h)  # (B*N, S, hidden)

        # -- Stage 2: spatial GCN at every timestep --
        # Treat each (batch, timestep) pair as its own graph, so B*S disjoint
        # copies of the topology are convolved in a single call.
        h = h.view(B, N, S, self.hidden_channels).permute(0, 2, 1, 3)  # (B, S, N, hidden)
        h = h.reshape(B * S * N, self.hidden_channels)

        num_graphs = B * S
        batched_index, batched_weight = self._batched_topology(
            edge_index, edge_weight, num_graphs, device
        )
        for gcn_block in self.gcn_blocks:
            h = gcn_block(h, batched_index, batched_weight)

        # -- Stage 3: readout, one embedding per (batch, timestep) --
        batch_vec = torch.arange(num_graphs, device=device).repeat_interleave(N)
        pooled = global_mean_pool(h, batch_vec)  # (B*S, hidden)
        embeddings = self.readout(pooled).view(B, S, self.embedding_dim)

        graph_out = embeddings if return_sequence else embeddings.mean(dim=1)
        if not return_nodes:
            return graph_out

        # The same readout applied before pooling, so each microphone keeps its
        # own trajectory through the graph. h is laid out (B, S, N, hidden) by
        # the permute in stage 2, so this view preserves node ordering.
        node_embeddings = self.readout(h).view(B, S, N, self.embedding_dim)
        return graph_out, node_embeddings


def main() -> None:  # pragma: no cover - manual inspection helper
    from src.mapping.topology_graph import build_acoustic_topology
    from src.settings import settings

    edge_index, edge_weight = build_acoustic_topology(
        settings.MIC_COORDS, settings.DISTANCE_THRESHOLD
    )
    model = SpatioTemporalGNN(
        in_channels=settings.GNN_IN_CHANNELS,
        hidden_channels=settings.GNN_HIDDEN_CHANNELS,
        embedding_dim=settings.GNN_EMBEDDING_DIM,
        num_nodes=settings.NUM_NODES,
    )
    x = torch.randn(8, settings.SEQ_LENGTH, settings.NUM_NODES * settings.GNN_IN_CHANNELS)

    pooled = model(x, edge_index, edge_weight)
    sequence = model(x, edge_index, edge_weight, return_sequence=True)

    print(f"[*] Pooled embedding:   {tuple(pooled.shape)}")
    print(f"[*] Sequence embedding: {tuple(sequence.shape)}")
    print(f"[*] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":  # pragma: no cover
    main()
