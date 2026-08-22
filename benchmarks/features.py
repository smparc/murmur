"""
Feature extraction for benchmark runs.

This used to be a small reimplementation of
``src.ingestion.cuda_stream_processor.get_mel_spectrogram_transform`` — same
mel filterbank, but ``log1p`` in place of production's dB scaling — on the
theory that the production module "constructs Kafka clients at import time".
It does not: ``confluent_kafka.Consumer``/``Producer`` are only built inside
``create_kafka_clients()``, which nothing here calls, so importing the module
costs nothing but the (already mandatory) ``confluent-kafka`` package.
``src/evaluation/mimii.py`` already imports the production transform directly
for exactly this reason.

That the two transforms had quietly diverged was not cosmetic: ``log1p`` and
``AmplitudeToDB(top_db=80)`` compress dynamic range very differently, so an
autoencoder's reconstruction error has a different scale and a different
sensitivity to quiet detail under each. Every number in
``benchmarks/evaluate_dataset.py`` — including the DCASE2020 results recorded
under ``paper/results/`` — was therefore produced against a feature
representation the production pipeline does not actually compute, which is
precisely the train/serve skew this benchmark suite exists to catch. Importing
the real transform removes the possibility of the two drifting again.
"""

from __future__ import annotations

import numpy as np
import torch

from src.ingestion.cuda_stream_processor import get_mel_spectrogram_transform

__all__ = ["mel_transform", "to_log_mel"]


def mel_transform(device: torch.device | str = "cpu") -> torch.nn.Module:
    """The production log-mel transform, on the requested device."""
    return get_mel_spectrogram_transform().to(device)


def to_log_mel(
    audio: np.ndarray | torch.Tensor,
    transform: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Waveform -> log-mel spectrogram in dB, shaped ``(n_mels, time)``.

    ``transform`` already applies ``AmplitudeToDB`` — see
    ``get_mel_spectrogram_transform`` — so this is only the dtype/device
    handling a raw waveform needs before that transform can run.
    """
    # `.copy()` because frombuffer yields a read-only view, and torch refuses to
    # wrap non-writable memory without warning.
    waveform = torch.from_numpy(audio.copy()) if isinstance(audio, np.ndarray) else audio
    waveform = waveform.to(device=device, dtype=torch.float32)

    return transform(waveform)
