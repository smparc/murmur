"""Shared pytest fixtures for the Murmur test suite."""

from __future__ import annotations

import os

# Set before any src import: the telemetry service reads this at startup, and
# without it the API tests would try to download a multi-gigabyte model from
# HuggingFace on every run.
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("MURMUR_API_KEY", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

SEED = 1337


@pytest.fixture(autouse=True)
def _deterministic():
    """
    Reseed every RNG before each test.

    Several assertions in this suite are statistical (does anomalous data score
    higher, does a split actually shuffle). Unseeded, they fail a few runs in a
    hundred and get dismissed as flakes.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def override_settings():
    """
    Temporarily override fields on the global settings object.

    ``Settings`` is a frozen dataclass — configuration should not mutate at
    runtime — so tests reach past that deliberately and restore afterwards
    rather than the production type being loosened for their convenience.
    """
    from src.settings import settings as live

    originals: dict[str, object] = {}

    def _apply(**overrides):
        for name, value in overrides.items():
            if not hasattr(live, name):
                raise AttributeError(f"Settings has no field {name!r}")
            originals.setdefault(name, getattr(live, name))
            object.__setattr__(live, name, value)
        return live

    yield _apply

    for name, value in originals.items():
        object.__setattr__(live, name, value)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def mic_coords() -> list[tuple[float, float, float]]:
    return [
        (0.0, 0.0, 3.0),
        (5.0, 0.0, 3.0),
        (0.0, 10.0, 3.0),
        (5.0, 10.0, 3.0),
    ]


@pytest.fixture
def sample_edge_topology(mic_coords):
    """4-node acoustic topology: ``(edge_index, edge_weight, num_nodes)``."""
    from src.mapping.topology_graph import build_acoustic_topology

    edge_index, edge_weight = build_acoustic_topology(mic_coords)
    return edge_index, edge_weight, len(mic_coords)


@pytest.fixture
def mock_audio_chunk() -> np.ndarray:
    """Half-second float32 waveform at 16 kHz."""
    return np.random.randn(8000).astype(np.float32)


@pytest.fixture
def api_client():
    """TestClient with the lifespan run, so models are actually initialised."""
    from fastapi.testclient import TestClient

    from src.translation.llm_decoder import app

    with TestClient(app) as client:
        yield client
