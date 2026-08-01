"""
Feature extraction for benchmark runs.

Deliberately a small reimplementation rather than an import of
``src.ingestion.cuda_stream_processor.get_mel_spectrogram_transform``: that
module constructs Kafka clients at import time, and a benchmark that cannot run
without a broker is a benchmark nobody runs. The transform parameters are read
from the same ``settings`` object, so the two cannot silently drift apart.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from src.settings import settings

__all__ = ["mel_transform", "to_log_mel"]


def mel_transform(device: torch.device | str = "cpu") -> torchaudio.transforms.MelSpectrogram:
    """Mel filterbank matching the streaming pipeline's configuration."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=settings.SAMPLE_RATE,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
        n_mels=settings.N_MELS,
    ).to(device)


def to_log_mel(
    audio: np.ndarray | torch.Tensor,
    transform: torchaudio.transforms.MelSpectrogram,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Waveform -> log-mel spectrogram, shaped ``(n_mels, time)``.

    Log scaling is not cosmetic. Acoustic energy spans several orders of
    magnitude, and a reconstruction loss computed on linear-power mel bins is
    dominated almost entirely by the loudest band, which makes the autoencoder
    blind to exactly the low-level spectral detail that distinguishes an early
    bearing fault from ordinary rumble.
    """
    if isinstance(audio, np.ndarray):
        # `.copy()` because frombuffer yields a read-only view, and torch
        # refuses to wrap non-writable memory without warning.
        waveform = torch.from_numpy(audio.copy())
    else:
        waveform = audio
    waveform = waveform.to(device=device, dtype=torch.float32)

    mel = transform(waveform)
    return torch.log1p(mel)
