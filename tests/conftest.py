"""Shared pytest fixtures for the Murmur test suite."""


import pytest
import torch
import numpy as np



@pytest.fixture
def device():
    return torch.device("cpu")



@pytest.fixture
def sample_edge_topology():
    """4-node fully-connected acoustic topology."""
    from src.mapping.topology_graph import build_acoustic_topology


    mics = [
        (0.0, 0.0, 3.0),
        (5.0, 0.0, 3.0),
        (0.0, 10.0, 3.0),
        (5.0, 10.0, 3.0),
    ]
    edge_index, edge_weight = build_acoustic_topology(mics)
    return edge_index, edge_weight, len(mics)



@pytest.fixture
def mock_audio_chunk():
    """Half-second float32 audio waveform at 16 kHz."""
    return np.random.randn(8000).astype(np.float32)