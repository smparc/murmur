"""
CUDA-accelerated streaming and audio preprocessing module.
"""
from .cuda_stream_processor import create_kafka_clients, get_mel_spectrogram_transform

__all__ = ["create_kafka_clients", "get_mel_spectrogram_transform"]