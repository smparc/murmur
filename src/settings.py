"""
Centralized configuration for the Murmur system.


All settings are driven by environment variables with sensible defaults.
Import `settings` anywhere to access typed, validated configuration.
"""


import os



class _Settings:
    """Typed configuration populated from environment variables."""


    # -- Kafka --
    KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")
    RAW_TOPIC: str = os.getenv("RAW_TOPIC", "raw-audio-stream")
    PROCESSED_TOPIC: str = os.getenv("PROCESSED_TOPIC", "spectrogram-embeddings")
    KAFKA_GROUP_ID: str = os.getenv("KAFKA_GROUP_ID", "gpu-preprocessing-group")
    KAFKA_COMPRESSION: str = os.getenv("KAFKA_COMPRESSION", "lz4")


    # -- Audio --
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    N_FFT: int = int(os.getenv("N_FFT", "1024"))
    HOP_LENGTH: int = int(os.getenv("HOP_LENGTH", "512"))
    N_MELS: int = int(os.getenv("N_MELS", "64"))
    CHUNK_DURATION: float = float(os.getenv("CHUNK_DURATION", "0.5"))


    # -- Models --
    GNN_EMBEDDING_DIM: int = int(os.getenv("GNN_EMBEDDING_DIM", "256"))
    GNN_HIDDEN_CHANNELS: int = int(os.getenv("GNN_HIDDEN_CHANNELS", "128"))
    GNN_IN_CHANNELS: int = int(os.getenv("GNN_IN_CHANNELS", "64"))
    LNN_HIDDEN_NEURONS: int = int(os.getenv("LNN_HIDDEN_NEURONS", "64"))
    SEQ_LENGTH: int = int(os.getenv("SEQ_LENGTH", "50"))


    # -- LLM --
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen1.5-1.8B")
    LLM_HIDDEN_DIM: int = int(os.getenv("LLM_HIDDEN_DIM", "2048"))
    LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "50"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


    # -- Training --
    TRAIN_EPOCHS: int = int(os.getenv("TRAIN_EPOCHS", "50"))
    TRAIN_BATCH_SIZE: int = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "1e-3"))


    # -- Infrastructure --
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    INFERENCE_HOST: str = os.getenv("INFERENCE_HOST", "0.0.0.0")
    INFERENCE_PORT: int = int(os.getenv("INFERENCE_PORT", "8000"))
    WS_ENDPOINT: str = os.getenv("WS_ENDPOINT", "ws://localhost:8000/ws/telemetry")


    # -- Topology (default 4-mic server room) --
    DEFAULT_MIC_COORDS: list = [
        (0.0, 0.0, 3.0),
        (5.0, 0.0, 3.0),
        (0.0, 10.0, 3.0),
        (5.0, 10.0, 3.0),
    ]
    DISTANCE_THRESHOLD: float = float(os.getenv("DISTANCE_THRESHOLD", "15.0"))


    @property
    def SAMPLES_PER_CHUNK(self) -> int:
        return int(self.SAMPLE_RATE * self.CHUNK_DURATION)


    @property
    def NUM_NODES(self) -> int:
        return len(self.DEFAULT_MIC_COORDS)



settings = _Settings()