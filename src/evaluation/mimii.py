"""
Benchmark the anomaly detector against real machine sound.

Every number this project reports today comes from synthetic data whose labels
are a deterministic function of the signal variance it generates. A model can
score a perfect F1 on that and detect nothing whatsoever in a real plant room,
because the task is separable by a single scalar the generator put there
deliberately.

MIMII (Purohit et al., 2019) is 26 GB of recorded valves, pumps, fans and
sliders, each with genuine mechanical faults, mixed against real factory noise
at -6/0/+6 dB SNR. ToyADMOS is structured similarly. Running the *production*
detector over that, with the *production* feature transform, is the only claim
about detection quality this repository can honestly make.

Two things are deliberate here:

- **The mel transform is imported from the ingestion service, not
  reimplemented.** A benchmark that computes its own features measures a model
  that will never exist in production. Train/serve skew of exactly this kind is
  the most common reason offline metrics fail to survive deployment.
- **The corpus is optional.** Nobody should need a 26 GB download to run the
  test suite, so the harness is exercised end-to-end on a synthetic corpus with
  the same interface, and skips cleanly when the real one is absent.

Download
--------
MIMII: https://zenodo.org/records/3384388

Expected layout (the scanner is tolerant of variations)::

    <root>/<snr>_<machine>/<machine>/id_<nn>/{normal,abnormal}/*.wav

Reference
---------
Purohit et al. (2019), "MIMII Dataset: Sound Dataset for Malfunctioning
Industrial Machine Investigation and Inspection", DCASE2019 Workshop.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.evaluation.metrics import detection_report
from src.settings import settings

log = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"id[_-]?(\d+)", re.IGNORECASE)
_KNOWN_MACHINES = ("fan", "pump", "slider", "valve", "gearbox", "bearing", "ToyCar", "ToyTrain")


@dataclass(frozen=True)
class AudioSample:
    """One labelled recording."""

    path: Path
    machine: str
    machine_id: str
    label: int  # 1 = anomalous
    snr: str = "unknown"

    @property
    def group(self) -> str:
        return f"{self.machine}/{self.machine_id}"


def _infer_machine(parts: tuple[str, ...]) -> str:
    for part in reversed(parts):
        for known in _KNOWN_MACHINES:
            if known.lower() == part.lower():
                return known.lower()
    for part in reversed(parts):
        for known in _KNOWN_MACHINES:
            if known.lower() in part.lower():
                return known.lower()
    return "unknown"


def _infer_snr(parts: tuple[str, ...]) -> str:
    for part in parts:
        if "dB" in part or "db" in part:
            return part
    return "unknown"


def discover(root: str | Path, extensions: tuple[str, ...] = (".wav",)) -> list[AudioSample]:
    """
    Scan a MIMII/ToyADMOS tree for labelled clips.

    Labels come from the containing directory being ``normal`` or ``abnormal``,
    which is the convention both corpora share. Anything outside such a
    directory is ignored rather than guessed at.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus root does not exist: {root}")

    samples: list[AudioSample] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in extensions or not path.is_file():
            continue

        parents = tuple(p.name for p in path.parents)
        label_dir = path.parent.name.lower()
        if label_dir == "abnormal":
            label = 1
        elif label_dir == "normal":
            label = 0
        else:
            continue

        id_match = next((_ID_PATTERN.search(p) for p in parents if _ID_PATTERN.search(p)), None)
        samples.append(
            AudioSample(
                path=path,
                machine=_infer_machine(parents),
                machine_id=id_match.group(1) if id_match else "00",
                label=label,
                snr=_infer_snr(parents),
            )
        )

    if not samples:
        log.warning("No normal/abnormal directories found under %s", root)
    return samples


def _mel_transform(device: torch.device):
    """
    The production log-mel transform.

    Imported rather than reconstructed so this benchmark cannot silently drift
    from what the ingestion service actually computes — including the dB scaling,
    which changes the reconstruction error's dynamic range by orders of magnitude
    and would quietly invalidate every number below if it were omitted here.
    """
    from src.ingestion.cuda_stream_processor import get_mel_spectrogram_transform

    return get_mel_spectrogram_transform().to(device)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a PCM WAV as ``(channels, samples)`` float32 in [-1, 1].

    Uses the standard library rather than ``torchaudio.load``, which in recent
    releases delegates to TorchCodec and raises if that optional package is
    absent. MIMII and ToyADMOS are plain 16-bit PCM, so ``wave`` reads them
    natively and the benchmark gains no new dependency.
    """
    import wave

    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())

    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"unsupported WAV sample width {width * 8}-bit: {path}")

    data = np.frombuffer(frames, dtype=dtypes[width])
    if width == 1:
        # 8-bit PCM is unsigned with a 128 offset.
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32) / float(2 ** (8 * width - 1))

    return audio.reshape(-1, channels).T, rate


def _write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write mono float32 in [-1, 1] as 16-bit PCM."""
    import wave

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())


def load_waveform(path: Path, target_sr: int) -> np.ndarray:
    """Load a clip as mono float32 at ``target_sr``."""
    channels, sr = _read_wav(Path(path))
    mono = channels.mean(axis=0) if channels.shape[0] > 1 else channels[0]

    if sr != target_sr:
        # Resampling stays in torch: it is pure tensor maths with no codec
        # dependency, and matches what the ingestion path would do.
        import torchaudio

        resampled = torchaudio.functional.resample(
            torch.from_numpy(mono).unsqueeze(0), sr, target_sr
        )
        mono = resampled.squeeze(0).numpy()

    return np.ascontiguousarray(mono, dtype=np.float32)


def chunk_waveform(waveform: np.ndarray, samples_per_chunk: int) -> list[np.ndarray]:
    """
    Split a clip into production-sized chunks.

    A MIMII recording is 10 s; the pipeline reasons over 0.5 s. Scoring whole
    clips would measure a model that never sees a whole clip.
    """
    if samples_per_chunk <= 0:
        raise ValueError(f"samples_per_chunk must be > 0, got {samples_per_chunk}")
    n = waveform.size // samples_per_chunk
    if n == 0:
        return [waveform]
    return [waveform[i * samples_per_chunk : (i + 1) * samples_per_chunk] for i in range(n)]


def score_sample(
    waveform: np.ndarray,
    mel,
    score_fn,
    samples_per_chunk: int,
    aggregate: str = "mean",
    device: torch.device | None = None,
) -> float:
    """
    Reduce one clip to a single anomaly score.

    ``aggregate`` selects how per-chunk scores combine. ``mean`` matches the
    DCASE baseline; ``max`` is more sensitive to brief transients such as a
    single valve impact, at the cost of reacting to isolated noise.
    """
    device = device or torch.device("cpu")
    chunks = chunk_waveform(waveform, samples_per_chunk)

    batch = torch.from_numpy(np.stack(chunks)).float().to(device)
    with torch.no_grad():
        spectrograms = mel(batch)  # (chunks, n_mels, frames)
        scores = score_fn(spectrograms.unsqueeze(1))  # (chunks,)

    scores = scores.detach().cpu().numpy().ravel()
    if aggregate == "max":
        return float(scores.max())
    if aggregate == "mean":
        return float(scores.mean())
    raise ValueError(f"unknown aggregate {aggregate!r}, expected 'mean' or 'max'")


def evaluate(
    samples: list[AudioSample],
    score_fn,
    device: torch.device | None = None,
    aggregate: str = "mean",
    max_fpr: float = 0.1,
    progress_every: int = 200,
) -> dict:
    """
    Run the detector over a corpus and report AUC/pAUC overall and per machine.

    ``score_fn`` takes ``(batch, 1, n_mels, frames)`` and returns ``(batch,)``
    anomaly scores — the signature of
    ``SpectrogramAutoencoder.anomaly_score``, so the production detector drops
    straight in.

    Per-machine breakdown is not decoration. MIMII's difficulty varies enormously
    by machine type: valves are near-impossible for reconstruction-based
    detectors because their normal operation is itself impulsive, while fans are
    comparatively easy. A single pooled AUC hides that completely, and a model
    tuned against the pooled number optimises for whichever machine happens to
    dominate the sample count.
    """
    if not samples:
        raise ValueError("no samples to evaluate")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = _mel_transform(device)

    scores: list[float] = []
    labels: list[int] = []
    groups: list[str] = []

    for i, sample in enumerate(samples, start=1):
        try:
            waveform = load_waveform(sample.path, settings.SAMPLE_RATE)
        except Exception:
            log.warning("Unreadable clip, skipping: %s", sample.path, exc_info=True)
            continue

        scores.append(
            score_sample(
                waveform,
                mel,
                score_fn,
                settings.SAMPLES_PER_CHUNK,
                aggregate=aggregate,
                device=device,
            )
        )
        labels.append(sample.label)
        groups.append(sample.group)

        if progress_every and i % progress_every == 0:
            log.info("Scored %d/%d clips", i, len(samples))

    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    per_machine: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        buckets[group].append(idx)

    for group, indices in sorted(buckets.items()):
        idx = np.asarray(indices)
        group_labels = labels_arr[idx]
        # A group with only one class has no ROC; report it rather than crash.
        if len(np.unique(group_labels)) < 2:
            per_machine[group] = {
                "auc": float("nan"),
                "pauc": float("nan"),
                "n_normal": int((group_labels == 0).sum()),
                "n_anomalous": int((group_labels == 1).sum()),
                "note": "single-class group, ROC undefined",
            }
            continue
        per_machine[group] = detection_report(scores_arr[idx], group_labels, max_fpr)

    scored = [m for m in per_machine.values() if not np.isnan(m.get("auc", float("nan")))]
    return {
        "overall": detection_report(scores_arr, labels_arr, max_fpr),
        "per_machine": per_machine,
        # The mean over machines, not over clips: it weights every machine type
        # equally instead of by however many recordings the corpus happens to
        # contain, which is how DCASE ranks systems.
        "macro_auc": float(np.mean([m["auc"] for m in scored])) if scored else float("nan"),
        "macro_pauc": float(np.mean([m["pauc"] for m in scored])) if scored else float("nan"),
        "aggregate": aggregate,
        "n_clips": int(labels_arr.size),
    }


def synthetic_corpus(
    tmpdir: str | Path,
    machines: tuple[str, ...] = ("pump", "valve"),
    per_class: int = 6,
    duration: float = 2.0,
    seed: int = 0,
) -> list[AudioSample]:
    """
    Write a small labelled corpus to disk in MIMII layout.

    Exists so the whole path — discovery, loading, chunking, the production mel
    transform, scoring, ROC — is exercised in CI without a 26 GB dependency.
    Anomalous clips get a bearing-squeal tone plus impulses over the same
    ambient bed, which is detectable but not trivially so.
    """
    rng = np.random.default_rng(seed)
    root = Path(tmpdir)
    sr = settings.SAMPLE_RATE
    n = int(sr * duration)
    t = np.arange(n) / sr

    samples: list[AudioSample] = []
    for machine in machines:
        for label_name, label in (("normal", 0), ("abnormal", 1)):
            directory = root / f"0dB_{machine}" / machine / "id_00" / label_name
            directory.mkdir(parents=True, exist_ok=True)

            for k in range(per_class):
                audio = 0.3 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 0.05, n)
                if label:
                    audio += 0.25 * np.sin(2 * np.pi * 2500 * t)
                    for _ in range(8):
                        pos = rng.integers(0, max(1, n - 64))
                        audio[pos : pos + 64] += 0.6

                path = directory / f"{k:08d}.wav"
                _write_wav(path, audio.astype(np.float32), sr)
                samples.append(
                    AudioSample(
                        path=path,
                        machine=machine,
                        machine_id="00",
                        label=label,
                        snr="0dB",
                    )
                )
    return samples


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Benchmark the detector on MIMII/ToyADMOS.")
    parser.add_argument("root", help="Corpus root directory")
    parser.add_argument("--aggregate", choices=("mean", "max"), default="mean")
    parser.add_argument("--max-fpr", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N clips")
    parser.add_argument("--json", dest="json_out", default=None, help="Write the report here")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    from src.detection.anomaly_detector import SpectrogramAutoencoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = SpectrogramAutoencoder(
        n_mels=settings.N_MELS, latent_dim=settings.AE_LATENT_DIM
    ).to(device)

    weights = Path(settings.MODEL_DIR) / "autoencoder_weights.pth"
    if weights.exists():
        autoencoder.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        log.info("Loaded autoencoder from %s", weights)
    else:
        log.warning(
            "No trained autoencoder at %s — this measures a randomly initialised "
            "network and the numbers mean nothing. Run `murmur-train` first.",
            weights,
        )
    autoencoder.eval()

    samples = discover(args.root)
    if args.limit:
        samples = samples[: args.limit]
    log.info("Discovered %d clips", len(samples))

    report = evaluate(samples, autoencoder.anomaly_score, device=device, aggregate=args.aggregate)

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
