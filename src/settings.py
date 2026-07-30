"""
Centralized configuration for the Murmur system.

Everything is driven by environment variables with sensible defaults, and
*validated at import time*. A misconfigured acoustic pipeline tends to fail
quietly — a bad hop length just yields differently-shaped spectrograms, and the
first sign of trouble is a shape mismatch three services downstream. Failing
loudly at startup is much cheaper to diagnose.

Import ``settings`` anywhere to read configuration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields

MicCoord = tuple[float, float, float]

_DEFAULT_MIC_COORDS: list[MicCoord] = [
    (0.0, 0.0, 3.0),  # Node 0: Rack A entrance
    (5.0, 0.0, 3.0),  # Node 1: Rack A exit
    (0.0, 10.0, 3.0),  # Node 2: Rack B entrance
    (5.0, 10.0, 3.0),  # Node 3: Rack B exit
]


class ConfigError(ValueError):
    """Raised when the environment describes an unworkable configuration."""


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a float, got {raw!r}") from exc


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_mic_coords(key: str, default: list[MicCoord]) -> list[MicCoord]:
    """
    Parse the microphone layout from JSON.

    The physical topology is the one thing this system genuinely cannot infer,
    and baking it into source means every site needs a code change to deploy.
    Expects ``[[x, y, z], ...]`` in metres.
    """
    raw = os.getenv(key)
    if not raw:
        return list(default)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{key} must be valid JSON, got {raw!r}") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ConfigError(f"{key} must be a non-empty JSON array of [x, y, z] triples")

    coords: list[MicCoord] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ConfigError(f"{key}[{i}] must be a 3-element [x, y, z] array, got {item!r}")
        try:
            coords.append((float(item[0]), float(item[1]), float(item[2])))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key}[{i}] contains a non-numeric value: {item!r}") from exc
    return coords


@dataclass(frozen=True)
class Settings:
    """Typed, validated configuration snapshot."""

    # -- Kafka --
    KAFKA_BROKER: str = field(default_factory=lambda: _env_str("KAFKA_BROKER", "localhost:9092"))
    RAW_TOPIC: str = field(default_factory=lambda: _env_str("RAW_TOPIC", "raw-audio-stream"))
    PROCESSED_TOPIC: str = field(
        default_factory=lambda: _env_str("PROCESSED_TOPIC", "spectrogram-embeddings")
    )
    KAFKA_GROUP_ID: str = field(
        default_factory=lambda: _env_str("KAFKA_GROUP_ID", "gpu-preprocessing-group")
    )
    WORKER_GROUP_ID: str = field(
        default_factory=lambda: _env_str("WORKER_GROUP_ID", "inference-worker-group")
    )
    KAFKA_COMPRESSION: str = field(
        default_factory=lambda: _env_str("KAFKA_COMPRESSION", "lz4")
    )

    # -- Audio --
    SAMPLE_RATE: int = field(default_factory=lambda: _env_int("SAMPLE_RATE", 16_000))
    N_FFT: int = field(default_factory=lambda: _env_int("N_FFT", 1024))
    HOP_LENGTH: int = field(default_factory=lambda: _env_int("HOP_LENGTH", 512))
    N_MELS: int = field(default_factory=lambda: _env_int("N_MELS", 64))
    CHUNK_DURATION: float = field(default_factory=lambda: _env_float("CHUNK_DURATION", 0.5))

    # -- Models --
    GNN_EMBEDDING_DIM: int = field(default_factory=lambda: _env_int("GNN_EMBEDDING_DIM", 256))
    GNN_HIDDEN_CHANNELS: int = field(default_factory=lambda: _env_int("GNN_HIDDEN_CHANNELS", 128))
    GNN_IN_CHANNELS: int = field(default_factory=lambda: _env_int("GNN_IN_CHANNELS", 64))
    GNN_NUM_HEADS: int = field(default_factory=lambda: _env_int("GNN_NUM_HEADS", 4))
    LNN_HIDDEN_NEURONS: int = field(default_factory=lambda: _env_int("LNN_HIDDEN_NEURONS", 64))
    AE_LATENT_DIM: int = field(default_factory=lambda: _env_int("AE_LATENT_DIM", 32))
    SEQ_LENGTH: int = field(default_factory=lambda: _env_int("SEQ_LENGTH", 50))

    # -- Anomaly detection --
    ANOMALY_Z_THRESHOLD: float = field(
        default_factory=lambda: _env_float("ANOMALY_Z_THRESHOLD", 3.0)
    )
    ANOMALY_WARMUP_FRAMES: int = field(
        default_factory=lambda: _env_int("ANOMALY_WARMUP_FRAMES", 50)
    )
    ANOMALY_WINDOW: int = field(default_factory=lambda: _env_int("ANOMALY_WINDOW", 500))

    # -- LLM --
    LLM_MODEL_NAME: str = field(
        default_factory=lambda: _env_str("LLM_MODEL_NAME", "Qwen/Qwen1.5-1.8B")
    )
    LLM_HIDDEN_DIM: int = field(default_factory=lambda: _env_int("LLM_HIDDEN_DIM", 2048))
    LLM_MAX_NEW_TOKENS: int = field(default_factory=lambda: _env_int("LLM_MAX_NEW_TOKENS", 50))
    LLM_TEMPERATURE: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    # Skips the (multi-GB) model download; the service still serves structured
    # telemetry, with the narrative field templated instead of generated.
    LLM_ENABLED: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))

    # -- Training --
    TRAIN_EPOCHS: int = field(default_factory=lambda: _env_int("TRAIN_EPOCHS", 50))
    TRAIN_BATCH_SIZE: int = field(default_factory=lambda: _env_int("TRAIN_BATCH_SIZE", 16))
    LEARNING_RATE: float = field(default_factory=lambda: _env_float("LEARNING_RATE", 1e-3))
    TRAIN_NUM_SAMPLES: int = field(default_factory=lambda: _env_int("TRAIN_NUM_SAMPLES", 1000))
    EARLY_STOP_PATIENCE: int = field(default_factory=lambda: _env_int("EARLY_STOP_PATIENCE", 10))
    SEED: int = field(default_factory=lambda: _env_int("SEED", 1337))

    # -- Infrastructure --
    MODEL_DIR: str = field(default_factory=lambda: _env_str("MODEL_DIR", "models"))
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: _env_str("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    INFERENCE_HOST: str = field(default_factory=lambda: _env_str("INFERENCE_HOST", "0.0.0.0"))
    INFERENCE_PORT: int = field(default_factory=lambda: _env_int("INFERENCE_PORT", 8000))
    INFERENCE_URL: str = field(
        default_factory=lambda: _env_str("INFERENCE_URL", "http://localhost:8000")
    )
    WS_ENDPOINT: str = field(
        default_factory=lambda: _env_str("WS_ENDPOINT", "ws://localhost:8000/ws/telemetry")
    )

    # -- Security --
    # Empty means unauthenticated. Acceptable inside a private cluster; the
    # startup log warns loudly so it is never an accident in production.
    API_KEY: str = field(default_factory=lambda: _env_str("MURMUR_API_KEY", ""))
    CORS_ORIGINS: str = field(
        default_factory=lambda: _env_str("CORS_ORIGINS", "http://localhost:3000")
    )
    RATE_LIMIT_PER_MINUTE: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 120)
    )

    # -- Topology --
    MIC_COORDS: list[MicCoord] = field(
        default_factory=lambda: _env_mic_coords("MIC_COORDS", _DEFAULT_MIC_COORDS)
    )
    DISTANCE_THRESHOLD: float = field(
        default_factory=lambda: _env_float("DISTANCE_THRESHOLD", 15.0)
    )
    # Physical sound intensity falls off as 1/r^2; 1.0 (inverse-linear) is
    # gentler and often trains more stably on small graphs.
    DISTANCE_DECAY_EXPONENT: float = field(
        default_factory=lambda: _env_float("DISTANCE_DECAY_EXPONENT", 2.0)
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors: list[str] = []

        positive = [
            "SAMPLE_RATE", "N_FFT", "HOP_LENGTH", "N_MELS", "SEQ_LENGTH",
            "GNN_EMBEDDING_DIM", "GNN_HIDDEN_CHANNELS", "GNN_IN_CHANNELS",
            "GNN_NUM_HEADS", "LNN_HIDDEN_NEURONS", "AE_LATENT_DIM",
            "TRAIN_EPOCHS", "TRAIN_BATCH_SIZE", "TRAIN_NUM_SAMPLES",
            "LLM_MAX_NEW_TOKENS", "LLM_HIDDEN_DIM", "INFERENCE_PORT",
            "ANOMALY_WARMUP_FRAMES", "ANOMALY_WINDOW",
        ]
        for name in positive:
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be > 0, got {getattr(self, name)}")

        if self.CHUNK_DURATION <= 0:
            errors.append(f"CHUNK_DURATION must be > 0, got {self.CHUNK_DURATION}")
        if self.LEARNING_RATE <= 0:
            errors.append(f"LEARNING_RATE must be > 0, got {self.LEARNING_RATE}")
        if not 0 <= self.LLM_TEMPERATURE <= 2:
            errors.append(f"LLM_TEMPERATURE must be in [0, 2], got {self.LLM_TEMPERATURE}")

        # STFT invariants. Violating either produces silently wrong spectrograms
        # rather than an exception, which is exactly the kind of bug that
        # survives to production.
        if self.HOP_LENGTH > self.N_FFT:
            errors.append(
                f"HOP_LENGTH ({self.HOP_LENGTH}) must be <= N_FFT ({self.N_FFT}); "
                "a larger hop than window discards samples between frames"
            )
        if self.N_MELS > self.N_FFT // 2 + 1:
            errors.append(
                f"N_MELS ({self.N_MELS}) exceeds the {self.N_FFT // 2 + 1} available FFT bins; "
                "some mel filters would be empty"
            )
        if self.SAMPLES_PER_CHUNK < self.N_FFT:
            errors.append(
                f"CHUNK_DURATION yields {self.SAMPLES_PER_CHUNK} samples, fewer than "
                f"N_FFT ({self.N_FFT}); every chunk would need padding"
            )

        if self.GNN_HIDDEN_CHANNELS % self.GNN_NUM_HEADS:
            errors.append(
                f"GNN_HIDDEN_CHANNELS ({self.GNN_HIDDEN_CHANNELS}) must be divisible by "
                f"GNN_NUM_HEADS ({self.GNN_NUM_HEADS}) for multi-head attention"
            )

        if self.ANOMALY_WINDOW < self.ANOMALY_WARMUP_FRAMES:
            errors.append(
                f"ANOMALY_WINDOW ({self.ANOMALY_WINDOW}) must be >= "
                f"ANOMALY_WARMUP_FRAMES ({self.ANOMALY_WARMUP_FRAMES})"
            )

        if len(self.MIC_COORDS) < 2:
            errors.append(f"MIC_COORDS needs at least 2 microphones, got {len(self.MIC_COORDS)}")
        if self.DISTANCE_THRESHOLD <= 0:
            errors.append(f"DISTANCE_THRESHOLD must be > 0, got {self.DISTANCE_THRESHOLD}")

        if errors:
            raise ConfigError(
                "Invalid Murmur configuration:\n  - " + "\n  - ".join(errors)
            )

    # -- derived values -----------------------------------------------------

    @property
    def SAMPLES_PER_CHUNK(self) -> int:
        return int(self.SAMPLE_RATE * self.CHUNK_DURATION)

    @property
    def NUM_NODES(self) -> int:
        return len(self.MIC_COORDS)

    @property
    def WINDOWED_TOPIC(self) -> str:
        """Topic carrying full ``SEQ_LENGTH`` temporal windows."""
        return f"{self.PROCESSED_TOPIC}-windowed"

    @property
    def MEL_FRAMES_PER_CHUNK(self) -> int:
        """Time frames torchaudio produces per chunk (centred STFT)."""
        return self.SAMPLES_PER_CHUNK // self.HOP_LENGTH + 1

    @property
    def CORS_ORIGIN_LIST(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def AUTH_ENABLED(self) -> bool:
        return bool(self.API_KEY)

    def describe(self) -> dict[str, object]:
        """Loggable snapshot with secrets redacted."""
        out: dict[str, object] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = "***redacted***" if f.name == "API_KEY" and value else value
        return out


settings = Settings()

# Backwards-compatible alias: the topology used to be exposed under this name.
DEFAULT_MIC_COORDS = _DEFAULT_MIC_COORDS

__all__ = ["ConfigError", "Settings", "settings", "DEFAULT_MIC_COORDS"]
