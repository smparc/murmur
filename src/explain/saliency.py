"""
Explaining *why* a frame was flagged.

A maintenance alert that says "node 3, anomaly score 0.87" is not actionable.
A technician needs to know what the system heard: a 3 kHz tone that was not there
yesterday points at a bearing race, broadband hiss points at cavitation or a
leak, and low-frequency modulation points at imbalance. Without that, the alert
is a number to be trusted or ignored on faith, and in practice it gets ignored.

The attribution here is not a saliency heuristic bolted onto an opaque score. For
a reconstruction-error detector the explanation is already exact: the anomaly
score *is* the mean squared reconstruction error, so the per-bin error map
decomposes it additively and with no approximation at all. Every bin's share of
the total is literally its contribution to the score.

That is worth preferring over the gradient-based attributions usually reached for
(saliency maps, Grad-CAM, integrated gradients). Those approximate a model's
sensitivity; this *is* the model's output, disaggregated. It cannot disagree with
the score it explains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "AnomalyExplanation",
    "BandContribution",
    "explain_anomaly",
    "mel_bin_frequencies",
    "reconstruction_error_map",
]

#: HTK mel scale — the default for ``torchaudio.transforms.MelSpectrogram``,
#: which is what the ingestion path uses. Slaney scaling would put the band
#: edges elsewhere and quietly mislabel every reported frequency.
_MEL_BREAK_FREQUENCY = 700.0
_MEL_SCALE_FACTOR = 2595.0


def _hz_to_mel(hz: float) -> float:
    return _MEL_SCALE_FACTOR * math.log10(1.0 + hz / _MEL_BREAK_FREQUENCY)


def _mel_to_hz(mel: float) -> float:
    return _MEL_BREAK_FREQUENCY * (10.0 ** (mel / _MEL_SCALE_FACTOR) - 1.0)


def mel_bin_frequencies(
    n_mels: int, sample_rate: int, f_min: float = 0.0, f_max: float | None = None
) -> list[float]:
    """
    Centre frequency, in Hz, of each mel filter.

    Needed to turn "bin 41 reconstructed badly" into "energy near 3.2 kHz",
    which is the only form a technician can act on. Mel filters are laid out at
    equal spacing on the mel scale using ``n_mels + 2`` edge points; the centres
    are the interior points.
    """
    if n_mels < 1:
        raise ValueError("n_mels must be >= 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if f_max is None:
        f_max = sample_rate / 2.0
    if not 0.0 <= f_min < f_max:
        raise ValueError("require 0 <= f_min < f_max")

    mel_min, mel_max = _hz_to_mel(f_min), _hz_to_mel(f_max)
    step = (mel_max - mel_min) / (n_mels + 1)
    return [_mel_to_hz(mel_min + step * (i + 1)) for i in range(n_mels)]


@dataclass(frozen=True)
class BandContribution:
    """One frequency band's share of the anomaly score."""

    low_hz: float
    high_hz: float
    share: float
    """Fraction of total reconstruction error, in ``[0, 1]``."""
    peak_frame: int
    """Index of the time frame where this band contributed most."""

    @property
    def centre_hz(self) -> float:
        return 0.5 * (self.low_hz + self.high_hz)

    def describe(self) -> str:
        """Short human-readable label, e.g. ``"2.1-3.4 kHz (46%)"``."""
        def fmt(hz: float) -> str:
            return f"{hz / 1000:.1f} kHz" if hz >= 1000 else f"{hz:.0f} Hz"

        return f"{fmt(self.low_hz)}-{fmt(self.high_hz)} ({self.share:.0%})"

    def as_dict(self) -> dict[str, float | int]:
        return {
            "low_hz": round(self.low_hz, 1),
            "high_hz": round(self.high_hz, 1),
            "centre_hz": round(self.centre_hz, 1),
            "share": round(self.share, 4),
            "peak_frame": self.peak_frame,
        }


@dataclass(frozen=True)
class AnomalyExplanation:
    """Where a frame's reconstruction error came from."""

    total_error: float
    bands: list[BandContribution]
    """Every band, ordered low frequency to high."""
    error_map: list[list[float]] | None = None
    """Per-bin error, ``(n_mels, time)``, for a dashboard heatmap overlay."""

    @property
    def dominant_band(self) -> BandContribution | None:
        return max(self.bands, key=lambda b: b.share) if self.bands else None

    def top_bands(self, limit: int = 3) -> list[BandContribution]:
        """The ``limit`` highest-contributing bands, most significant first."""
        return sorted(self.bands, key=lambda b: b.share, reverse=True)[:limit]

    def summary(self, limit: int = 2) -> str:
        """One-line description suitable for an alert body."""
        top = self.top_bands(limit)
        if not top:
            return "no spectral attribution available"
        return "energy concentrated at " + ", ".join(b.describe() for b in top)

    def as_dict(self, include_map: bool = False) -> dict:
        payload: dict = {
            "total_error": round(self.total_error, 6),
            "bands": [b.as_dict() for b in self.bands],
            "summary": self.summary(),
        }
        dominant = self.dominant_band
        if dominant is not None:
            payload["dominant_band"] = dominant.as_dict()
        if include_map and self.error_map is not None:
            payload["error_map"] = self.error_map
        return payload


@torch.no_grad()
def reconstruction_error_map(autoencoder, spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Per-bin squared reconstruction error, shaped ``(n_mels, time)``.

    Summing this map and dividing by its element count reproduces
    ``SpectrogramAutoencoder.anomaly_score`` exactly — which is the property that
    makes it an explanation rather than a plausible-looking picture.
    """
    x = spectrogram
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() != 4:
        raise ValueError(f"expected a 2-, 3- or 4-D spectrogram, got {spectrogram.dim()}-D")

    if x.size(0) != 1:
        raise ValueError("explain one frame at a time; got a batch of %d" % x.size(0))

    reconstruction, _ = autoencoder(x)
    return (reconstruction - x).pow(2).squeeze(0).squeeze(0)


def explain_anomaly(
    autoencoder,
    spectrogram: torch.Tensor,
    sample_rate: int,
    n_bands: int = 8,
    include_map: bool = False,
) -> AnomalyExplanation:
    """
    Attribute a frame's anomaly score across frequency bands.

    Parameters
    ----------
    n_bands:
        Mel bins are grouped into this many contiguous bands. Reporting all 64
        bins individually is precise and unreadable; eight bands is about the
        resolution a fault signature actually occupies, and each still carries
        its true Hz range rather than a bin index.
    include_map:
        Attach the full per-bin error map for a dashboard heatmap. Off by
        default — at 64x16 floats per frame per node it is far larger than the
        rest of the telemetry payload combined.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")

    error_map = reconstruction_error_map(autoencoder, spectrogram)
    n_mels, n_frames = error_map.shape
    n_bands = min(n_bands, n_mels)

    centres = mel_bin_frequencies(n_mels, sample_rate)
    total = float(error_map.sum().item())

    bands: list[BandContribution] = []
    # Distribute bins as evenly as possible; earlier bands absorb the remainder
    # so no band is empty and every bin is counted exactly once.
    base, remainder = divmod(n_mels, n_bands)
    start = 0
    for index in range(n_bands):
        width = base + (1 if index < remainder else 0)
        stop = start + width

        band_error = error_map[start:stop, :]
        band_total = float(band_error.sum().item())
        per_frame = band_error.sum(dim=0)
        peak_frame = int(torch.argmax(per_frame).item()) if n_frames else 0

        bands.append(
            BandContribution(
                low_hz=centres[start],
                high_hz=centres[stop - 1],
                # A silent frame reconstructs perfectly; report zero share
                # rather than dividing by zero.
                share=(band_total / total) if total > 0 else 0.0,
                peak_frame=peak_frame,
            )
        )
        start = stop

    return AnomalyExplanation(
        total_error=total,
        bands=bands,
        error_map=error_map.tolist() if include_map else None,
    )
