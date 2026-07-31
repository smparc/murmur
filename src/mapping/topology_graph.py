"""
Physical graph topology for the ST-GNN.

The facility is modelled as a graph: microphones are nodes, and an edge means
two microphones are close enough to hear the same machine. Edge weight encodes
how strongly sound couples between them.

Acoustic intensity from a point source falls off as ``1/r^2``, so that is the
default. The exponent is configurable because on small, tightly-spaced arrays
an inverse-square weighting concentrates almost all mass on the nearest
neighbour and the GCN degenerates towards a one-hop copy.
"""

from __future__ import annotations

import numpy as np
import torch

# Guards against a singularity for co-located microphones.
_EPSILON = 1e-5


def build_acoustic_topology(
    mic_coordinates: list[tuple[float, float, float]] | np.ndarray,
    distance_threshold: float = 15.0,
    decay_exponent: float = 2.0,
    max_neighbours: int | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the acoustic coupling graph.

    Parameters
    ----------
    mic_coordinates:
        ``(x, y, z)`` positions in metres.
    distance_threshold:
        Maximum separation, in metres, that still counts as acoustic coupling.
    decay_exponent:
        Falloff power. ``2.0`` is the physical inverse-square law; ``1.0`` gives
        a flatter, often better-conditioned graph.
    max_neighbours:
        If set, keep only the ``k`` nearest neighbours per node. Without this a
        dense array within the threshold produces a complete graph, and a GCN
        over a complete graph is just a mean — the spatial structure the model
        is supposed to exploit vanishes.
    normalize:
        Scale weights so the strongest edge is 1.0, which keeps GCN activations
        in a stable range regardless of the facility's physical scale.

    Returns
    -------
    edge_index:
        ``(2, num_edges)`` long tensor.
    edge_weight:
        ``(num_edges,)`` float32 tensor.
    """
    coords = np.asarray(mic_coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"mic_coordinates must have shape (N, 3), got {coords.shape}")
    if distance_threshold <= 0:
        raise ValueError(f"distance_threshold must be > 0, got {distance_threshold}")

    num_nodes = coords.shape[0]
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    # Pairwise Euclidean distances, vectorized rather than the O(n^2) Python
    # double loop this replaces — it runs on every worker startup and on every
    # topology reconfiguration.
    deltas = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((deltas**2).sum(axis=-1))

    # A node is not its own neighbour.
    np.fill_diagonal(dist, np.inf)

    adjacency = dist <= distance_threshold

    if max_neighbours is not None:
        if max_neighbours < 1:
            raise ValueError(f"max_neighbours must be >= 1, got {max_neighbours}")
        keep = np.zeros_like(adjacency)
        k = min(max_neighbours, num_nodes - 1)
        nearest = np.argsort(dist, axis=1)[:, :k]
        rows = np.arange(num_nodes)[:, None]
        keep[rows, nearest] = True
        # Symmetrize: if A hears B, B hears A. Otherwise the graph is directed
        # in a way that has no physical meaning.
        keep = keep | keep.T
        adjacency &= keep

    sources, targets = np.nonzero(adjacency)
    if sources.size == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    weights = 1.0 / (dist[sources, targets] ** decay_exponent + _EPSILON)

    edge_index = torch.from_numpy(np.stack([sources, targets])).long()
    edge_weight = torch.from_numpy(weights).float()

    if normalize:
        edge_weight = edge_weight / edge_weight.max()

    return edge_index, edge_weight


def topology_summary(
    mic_coordinates: list[tuple[float, float, float]],
    distance_threshold: float = 15.0,
    **kwargs: object,
) -> dict[str, float]:
    """Diagnostics for logs and Dagster asset metadata."""
    edge_index, edge_weight = build_acoustic_topology(
        mic_coordinates,
        distance_threshold,
        **kwargs,  # type: ignore[arg-type]
    )
    num_nodes = len(mic_coordinates)
    num_edges = int(edge_index.shape[1])
    max_edges = num_nodes * (num_nodes - 1)

    degrees = (
        torch.bincount(edge_index[0], minlength=num_nodes) if num_edges else torch.zeros(num_nodes)
    )
    isolated = int((degrees == 0).sum())

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": (num_edges / max_edges) if max_edges else 0.0,
        "mean_degree": float(degrees.float().mean()),
        "isolated_nodes": isolated,
        "mean_weight": float(edge_weight.mean()) if num_edges else 0.0,
    }


def main() -> None:  # pragma: no cover - manual inspection helper
    from src.settings import settings

    edges, weights = build_acoustic_topology(
        settings.MIC_COORDS,
        settings.DISTANCE_THRESHOLD,
        decay_exponent=settings.DISTANCE_DECAY_EXPONENT,
    )
    print(f"[*] Topology for {settings.NUM_NODES} nodes")
    print(f"[*] Edge index shape: {tuple(edges.shape)}")
    print(f"[*] Edge weights: {weights}")
    for key, value in topology_summary(settings.MIC_COORDS, settings.DISTANCE_THRESHOLD).items():
        print(f"    {key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
