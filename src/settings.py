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
    KAFKA_COMPRESSION: str = field(default_factory=lambda: _env_str("KAFKA_COMPRESSION", "lz4"))
    # The per-frame topic exists for external consumers; nothing inside Murmur
    # reads it (the worker consumes WINDOWED_TOPIC). Publishing it by default
    # costs a serialize + compress + broker round trip per frame per node for
    # nobody, so it is opt-in.
    PUBLISH_FRAME_TOPIC: bool = field(
        default_factory=lambda: _env_bool("PUBLISH_FRAME_TOPIC", False)
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

    # -- Array liveness --
    # A graph snapshot ideally contains every microphone. But requiring all of
    # them unconditionally means one dead sensor silences the entire array —
    # the system goes quiet, which on a monitoring product is indistinguishable
    # from "nothing is wrong". These three settings bound that failure.
    #
    # Spread across one snapshot still treated as a single acoustic instant.
    WINDOW_STALENESS_TOLERANCE: float = field(
        default_factory=lambda: _env_float("WINDOW_STALENESS_TOLERANCE", 5.0)
    )
    # How long to wait for absent microphones before releasing a partial
    # snapshot. Must exceed normal jitter or healthy arrays degrade needlessly.
    ARRAY_MAX_WAIT: float = field(default_factory=lambda: _env_float("ARRAY_MAX_WAIT", 15.0))
    # Below this many reporting microphones the graph is too sparse to convolve
    # meaningfully, so nothing is emitted and the operator is told why.
    ARRAY_MIN_NODES: int = field(default_factory=lambda: _env_int("ARRAY_MIN_NODES", 2))

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
    # Sized so per-stratum conformal calibration is actually estimable. Only
    # half the test split calibrates — 7.5% of this number — and it is then
    # partitioned by severity, so at 1000 the `warning` band received about
    # seven samples against a `min_group_size` of 30 and silently fell back to
    # the global radius. That defeats the point of grouping: coverage is then
    # guaranteed on a population dominated by healthy machines rather than
    # within the high-risk bands anyone would act on.
    TRAIN_NUM_SAMPLES: int = field(default_factory=lambda: _env_int("TRAIN_NUM_SAMPLES", 4000))
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
    # Must exceed the pipeline's own steady state or the limiter silently drops
    # monitoring data. One snapshot per CHUNK_DURATION emits NUM_NODES requests,
    # so a 4-mic array at 0.5s chunks sustains ~480/min; this leaves ~2.5x
    # headroom for bursts and backfill. Set 0 to disable.
    RATE_LIMIT_PER_MINUTE: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 1200)
    )
    # The limiter keeps one bucket per caller identity. On a public endpoint an
    # attacker controls that identity, so an unbounded map of buckets turns the
    # anti-DoS control into a memory-exhaustion vector of its own. Idle buckets
    # are swept; this caps what survives a burst of distinct keys.
    RATE_LIMIT_MAX_KEYS: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_MAX_KEYS", 10_000)
    )
    # /metrics publishes per-node anomaly counts and z-scores — an operational
    # map of which machines are failing. Authenticated by default whenever a key
    # is configured; set false for an in-cluster Prometheus that cannot send one.
    METRICS_REQUIRE_AUTH: bool = field(
        default_factory=lambda: _env_bool("METRICS_REQUIRE_AUTH", True)
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

    # -- Spatial acoustics (TDOA) --
    # Cross-correlating every microphone pair costs an FFT per pair per chunk.
    # It is cheap at 4 mics and quadratic thereafter, hence the switch.
    TDOA_ENABLED: bool = field(default_factory=lambda: _env_bool("TDOA_ENABLED", True))
    # Sub-sample interpolation factor. A 5 m array spans only ~230 samples of
    # delay at 16 kHz, so raw sample resolution is coarse relative to the signal.
    TDOA_INTERP: int = field(default_factory=lambda: _env_int("TDOA_INTERP", 8))
    # Pairs below this correlation are excluded from the position solve.
    TDOA_MIN_COHERENCE: float = field(
        default_factory=lambda: _env_float("TDOA_MIN_COHERENCE", 0.15)
    )
    # Maximum clock spread across the array still treated as one instant.
    # Delays being measured are ~15 ms, so edge clocks must be synchronised well
    # inside this or the estimates are noise. See src/ingestion/spatial_probe.py.
    TDOA_STALENESS_TOLERANCE: float = field(
        default_factory=lambda: _env_float("TDOA_STALENESS_TOLERANCE", 0.5)
    )
    # How sharply measured coherence attenuates a geometric edge weight.
    TDOA_EDGE_GAMMA: float = field(default_factory=lambda: _env_float("TDOA_EDGE_GAMMA", 1.0))
    # Floor on that attenuation. A graph with every edge at zero has no
    # propagation and the GCN collapses to a per-node MLP.
    TDOA_EDGE_FLOOR: float = field(default_factory=lambda: _env_float("TDOA_EDGE_FLOOR", 0.05))

    # -- Alerting --
    # A prediction nobody sees changes nothing. All three are empty by default:
    # a monitoring system must not page anyone until someone has deliberately
    # configured it to, so an unset deployment routes nowhere rather than
    # somewhere surprising.
    SLACK_WEBHOOK_URL: str = field(default_factory=lambda: _env_str("SLACK_WEBHOOK_URL", ""))
    PAGERDUTY_ROUTING_KEY: str = field(
        default_factory=lambda: _env_str("PAGERDUTY_ROUTING_KEY", "")
    )
    ALERT_WEBHOOK_URL: str = field(default_factory=lambda: _env_str("ALERT_WEBHOOK_URL", ""))
    # Silence per node and fault after a page. Escalation bypasses it — a fault
    # getting worse is new information — but a steady fault must not re-page
    # every half-second chunk, which is how alerting integrations get muted.
    ALERT_COOLDOWN_SECONDS: float = field(
        default_factory=lambda: _env_float("ALERT_COOLDOWN_SECONDS", 900.0)
    )
    ALERT_MIN_SEVERITY: str = field(
        default_factory=lambda: _env_str("ALERT_MIN_SEVERITY", "warning")
    )

    # -- Forecast uncertainty --
    # Target miscoverage for conformal prediction intervals: 0.1 gives 90%
    # coverage. See src/forecasting/conformal.py.
    CONFORMAL_ALPHA: float = field(default_factory=lambda: _env_float("CONFORMAL_ALPHA", 0.1))

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors: list[str] = []

        positive = [
            "SAMPLE_RATE",
            "N_FFT",
            "HOP_LENGTH",
            "N_MELS",
            "SEQ_LENGTH",
            "GNN_EMBEDDING_DIM",
            "GNN_HIDDEN_CHANNELS",
            "GNN_IN_CHANNELS",
            "GNN_NUM_HEADS",
            "LNN_HIDDEN_NEURONS",
            "AE_LATENT_DIM",
            "TRAIN_EPOCHS",
            "TRAIN_BATCH_SIZE",
            "TRAIN_NUM_SAMPLES",
            "LLM_MAX_NEW_TOKENS",
            "LLM_HIDDEN_DIM",
            "INFERENCE_PORT",
            "ANOMALY_WARMUP_FRAMES",
            "ANOMALY_WINDOW",
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

        if self.WINDOW_STALENESS_TOLERANCE <= 0:
            errors.append(
                f"WINDOW_STALENESS_TOLERANCE must be > 0, got {self.WINDOW_STALENESS_TOLERANCE}"
            )
        if self.ARRAY_MAX_WAIT < self.WINDOW_STALENESS_TOLERANCE:
            errors.append(
                f"ARRAY_MAX_WAIT ({self.ARRAY_MAX_WAIT}) must be >= "
                f"WINDOW_STALENESS_TOLERANCE ({self.WINDOW_STALENESS_TOLERANCE}); a shorter "
                "wait would release degraded snapshots before a healthy array can converge"
            )
        if self.ARRAY_MIN_NODES < 1:
            errors.append(f"ARRAY_MIN_NODES must be >= 1, got {self.ARRAY_MIN_NODES}")
        if len(self.MIC_COORDS) < self.ARRAY_MIN_NODES:
            errors.append(
                f"ARRAY_MIN_NODES ({self.ARRAY_MIN_NODES}) exceeds the number of "
                f"microphones ({len(self.MIC_COORDS)}); no snapshot could ever be released"
            )
        if self.RATE_LIMIT_MAX_KEYS <= 0:
            errors.append(f"RATE_LIMIT_MAX_KEYS must be > 0, got {self.RATE_LIMIT_MAX_KEYS}")

        if self.ANOMALY_WINDOW < self.ANOMALY_WARMUP_FRAMES:
            errors.append(
                f"ANOMALY_WINDOW ({self.ANOMALY_WINDOW}) must be >= "
                f"ANOMALY_WARMUP_FRAMES ({self.ANOMALY_WARMUP_FRAMES})"
            )

        if len(self.MIC_COORDS) < 2:
            errors.append(f"MIC_COORDS needs at least 2 microphones, got {len(self.MIC_COORDS)}")
        if self.DISTANCE_THRESHOLD <= 0:
            errors.append(f"DISTANCE_THRESHOLD must be > 0, got {self.DISTANCE_THRESHOLD}")

        if self.TDOA_INTERP < 1:
            errors.append(f"TDOA_INTERP must be >= 1, got {self.TDOA_INTERP}")
        if not 0 <= self.TDOA_MIN_COHERENCE <= 1:
            errors.append(f"TDOA_MIN_COHERENCE must be in [0, 1], got {self.TDOA_MIN_COHERENCE}")
        if not 0 <= self.TDOA_EDGE_FLOOR <= 1:
            errors.append(f"TDOA_EDGE_FLOOR must be in [0, 1], got {self.TDOA_EDGE_FLOOR}")
        if self.TDOA_STALENESS_TOLERANCE <= 0:
            errors.append(
                f"TDOA_STALENESS_TOLERANCE must be > 0, got {self.TDOA_STALENESS_TOLERANCE}"
            )
        if self.ALERT_COOLDOWN_SECONDS < 0:
            errors.append(f"ALERT_COOLDOWN_SECONDS must be >= 0, got {self.ALERT_COOLDOWN_SECONDS}")
        if self.ALERT_MIN_SEVERITY not in {"normal", "warning", "critical"}:
            errors.append(
                f"ALERT_MIN_SEVERITY must be normal, warning or critical, "
                f"got {self.ALERT_MIN_SEVERITY!r}"
            )

        if not 0 < self.CONFORMAL_ALPHA < 1:
            errors.append(f"CONFORMAL_ALPHA must be in (0, 1), got {self.CONFORMAL_ALPHA}")

        if errors:
            raise ConfigError("Invalid Murmur configuration:\n  - " + "\n  - ".join(errors))

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
    def SPATIAL_TOPIC(self) -> str:
        """Topic carrying TDOA delays and source localization per instant."""
        return f"{self.PROCESSED_TOPIC}-spatial"

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

__all__ = ["DEFAULT_MIC_COORDS", "ConfigError", "Settings", "settings"]
