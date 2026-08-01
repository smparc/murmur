"""
Loaders for public machine-condition audio datasets.

Murmur's own numbers come from a simulator whose fault signatures were written
by hand. That is enough to catch regressions and nothing more: a detector tuned
to find a 2 kHz sine that the same repository injected has not been shown to
find anything. These loaders point the same detector at recordings of real
machinery so the results can be put next to published baselines.

None of the datasets are vendored — they are large and separately licensed. Each
loader takes a root directory and explains where to get the data if the
directory is missing. See ``benchmarks/README.md`` for download instructions.

Supported layouts
-----------------

``MIMIIDataset``
    MIMII (Purohit et al., 2019). Real recordings of industrial fans, pumps,
    sliders and valves, with genuine faults, at three SNRs.
    ``{root}/{machine}/id_{NN}/{normal,abnormal}/*.wav``

``DCASEDataset``
    DCASE Task 2 development sets, which repackage MIMII and ToyADMOS with a
    fixed train/test split. The label lives in the filename.
    ``{root}/{machine}/{train,test}/{normal,anomaly}_id_{NN}_{seq}.wav``

``IMSDataset``
    NASA/IMS bearing run-to-failure. ASCII vibration records rather than audio,
    and the only public set here that runs a bearing all the way to destruction
    — so it is the only one that can measure lead time on real hardware.
    ``{root}/{run}/{YYYY.MM.DD.HH.MM.SS}``
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "AudioSample",
    "DCASEDataset",
    "DatasetNotFound",
    "IMSDataset",
    "MIMIIDataset",
    "load_dataset",
]

log = logging.getLogger(__name__)


class DatasetNotFound(FileNotFoundError):
    """Raised when a dataset root is missing, with instructions attached."""


@dataclass(frozen=True)
class AudioSample:
    """One recording plus its label. ``label`` is 1 for anomalous."""

    path: Path
    label: int
    machine_type: str
    machine_id: str
    split: str = "test"

    @property
    def is_anomalous(self) -> bool:
        return self.label == 1

    @property
    def group(self) -> str:
        """Machine type and unit, the granularity DCASE reports AUC at."""
        return f"{self.machine_type}/{self.machine_id}"


class _BaseDataset:
    """Shared root handling and the iteration contract."""

    name = "dataset"
    download_hint = ""

    def __init__(self, root: str | Path, sample_rate: int = 16_000):
        self.root = Path(root)
        self.sample_rate = sample_rate
        if not self.root.exists():
            raise DatasetNotFound(
                f"{self.name} not found at {self.root}\n\n{self.download_hint}"
            )

    def iter_samples(self) -> Iterator[AudioSample]:
        raise NotImplementedError

    def samples(self, split: str | None = None) -> list[AudioSample]:
        """All samples, optionally restricted to one split."""
        found = [s for s in self.iter_samples() if split is None or s.split == split]
        if not found:
            log.warning("%s: no samples found under %s", self.name, self.root)
        return found

    def groups(self) -> list[str]:
        return sorted({s.group for s in self.iter_samples()})

    def load_audio(self, sample: AudioSample) -> np.ndarray:
        """Waveform as mono float32, resampled to ``self.sample_rate``."""
        return _load_wav(sample.path, self.sample_rate)

    def __len__(self) -> int:
        return len(self.samples())


# ---------------------------------------------------------------------------
# MIMII
# ---------------------------------------------------------------------------


class MIMIIDataset(_BaseDataset):
    """
    MIMII: ``{root}/{machine}/id_{NN}/{normal,abnormal}/*.wav``.

    MIMII ships no train/test split of its own. Everything is reported as
    ``test`` here; ``evaluate_dataset`` carves a normal-only training fold out of
    it, because an autoencoder trained on anomalies learns to reconstruct them
    and stops detecting them.
    """

    name = "MIMII"
    download_hint = (
        "Download MIMII from https://zenodo.org/record/3384388\n"
        "Extract so the tree looks like:\n"
        "    <root>/fan/id_00/normal/00000000.wav\n"
        "    <root>/fan/id_00/abnormal/00000000.wav"
    )

    def iter_samples(self) -> Iterator[AudioSample]:
        for machine_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for id_dir in sorted(p for p in machine_dir.iterdir() if p.is_dir()):
                for condition, label in (("normal", 0), ("abnormal", 1)):
                    condition_dir = id_dir / condition
                    if not condition_dir.is_dir():
                        continue
                    for wav in sorted(condition_dir.glob("*.wav")):
                        yield AudioSample(
                            path=wav,
                            label=label,
                            machine_type=machine_dir.name,
                            machine_id=id_dir.name,
                        )


# ---------------------------------------------------------------------------
# DCASE
# ---------------------------------------------------------------------------


_DCASE_NAME = re.compile(
    r"^(?P<condition>normal|anomaly)_(?P<id>id_\d+)_(?P<seq>\d+)\.wav$", re.IGNORECASE
)


class DCASEDataset(_BaseDataset):
    """
    DCASE Task 2 development data:
    ``{root}/{machine}/{train,test}/{normal,anomaly}_id_{NN}_{seq}.wav``.

    The official split is respected exactly. ``train`` is normal-only by
    construction, which is the point of the task: the challenge is unsupervised
    detection, and a system that peeked at labelled anomalies during training
    would not be reporting a comparable number.
    """

    name = "DCASE Task 2"
    download_hint = (
        "Download the DCASE Task 2 development set from\n"
        "  https://dcase.community/challenge2020/task-unsupervised-detection-"
        "of-anomalous-sounds\n"
        "Extract so the tree looks like:\n"
        "    <root>/fan/train/normal_id_00_00000000.wav\n"
        "    <root>/fan/test/anomaly_id_00_00000000.wav"
    )

    def iter_samples(self) -> Iterator[AudioSample]:
        for machine_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for split in ("train", "test"):
                split_dir = machine_dir / split
                if not split_dir.is_dir():
                    continue
                for wav in sorted(split_dir.glob("*.wav")):
                    matched = _DCASE_NAME.match(wav.name)
                    if matched is None:
                        log.debug("Skipping unrecognised filename: %s", wav.name)
                        continue
                    yield AudioSample(
                        path=wav,
                        label=0 if matched["condition"].lower() == "normal" else 1,
                        machine_type=machine_dir.name,
                        machine_id=matched["id"],
                        split=split,
                    )


# ---------------------------------------------------------------------------
# IMS bearing
# ---------------------------------------------------------------------------


_IMS_TIMESTAMP = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


class IMSDataset(_BaseDataset):
    """
    NASA/IMS bearing run-to-failure.

    Each file is one 1-second vibration snapshot, whitespace-separated ASCII with
    one column per channel, and the directory is a chronological sequence ending
    in bearing destruction.

    Labelling is positional rather than annotated: IMS ships no per-file labels,
    so the last ``failure_fraction`` of each run is treated as degraded. That is
    a convention, not ground truth, and it is the reason this loader is most
    useful for lead-time analysis — where the ordering is what matters — rather
    than for a headline AUC.
    """

    name = "IMS bearing"
    download_hint = (
        "Download the IMS bearing data set from the NASA Prognostics Data "
        "Repository:\n"
        "  https://www.nasa.gov/intelligent-systems-division/"
        "discovery-and-systems-health/pcoe/pcoe-data-set-repository/\n"
        "Extract so the tree looks like:\n"
        "    <root>/1st_test/2003.10.22.12.06.24"
    )

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 20_480,
        failure_fraction: float = 0.15,
        channel: int = 0,
    ):
        super().__init__(root, sample_rate)
        if not 0.0 < failure_fraction < 1.0:
            raise ValueError("failure_fraction must be in (0, 1)")
        self.failure_fraction = failure_fraction
        self.channel = channel

    def iter_samples(self) -> Iterator[AudioSample]:
        for run_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            records = sorted(
                (p for p in run_dir.iterdir() if _IMS_TIMESTAMP.match(p.name)),
                key=lambda p: p.name,
            )
            if not records:
                continue
            healthy_until = int(len(records) * (1.0 - self.failure_fraction))
            for position, record in enumerate(records):
                yield AudioSample(
                    path=record,
                    label=0 if position < healthy_until else 1,
                    machine_type="bearing",
                    machine_id=run_dir.name,
                )

    def load_audio(self, sample: AudioSample) -> np.ndarray:
        """Read one ASCII vibration record and return the selected channel."""
        data = np.loadtxt(sample.path, dtype=np.float32)
        if data.ndim == 1:
            return data
        if self.channel >= data.shape[1]:
            raise IndexError(
                f"channel {self.channel} out of range for {sample.path} "
                f"with {data.shape[1]} channels"
            )
        return np.ascontiguousarray(data[:, self.channel])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[_BaseDataset]] = {
    "mimii": MIMIIDataset,
    "dcase": DCASEDataset,
    "ims": IMSDataset,
}


def load_dataset(name: str, root: str | Path, **kwargs) -> _BaseDataset:
    """Construct a loader by name (``mimii``, ``dcase`` or ``ims``)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[key](root, **kwargs)


#: Full-scale value per PCM dtype, used to normalise into [-1, 1].
_PCM_FULL_SCALE: dict[str, float] = {
    "int16": 32768.0,
    "int32": 2147483648.0,
}


def _load_wav(path: Path, target_rate: int) -> np.ndarray:
    """
    Load a WAV as mono float32 in ``[-1, 1]`` at ``target_rate``.

    Read with ``scipy.io.wavfile`` rather than ``torchaudio.load``: as of
    torchaudio 2.9 the latter delegates to TorchCodec, which is a separate
    install and not one of this project's dependencies. Every dataset here is
    PCM WAV, which scipy reads natively — so the extra dependency would buy
    nothing and cost a failure at the point where someone finally has the data.
    """
    from scipy.io import wavfile

    source_rate, data = wavfile.read(path)

    scale = _PCM_FULL_SCALE.get(data.dtype.name)
    if scale is not None:
        data = data.astype(np.float32) / scale
    elif data.dtype == np.uint8:
        # 8-bit WAV is unsigned with a 128 midpoint.
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        # Mix to mono. MIMII ships 8-channel array recordings; the detector
        # consumes one stream per microphone node, not a beamformed array.
        data = data.mean(axis=1)

    if source_rate != target_rate:
        import torch
        import torchaudio

        # Resampling is pure tensor arithmetic and needs no codec backend.
        resampled = torchaudio.functional.resample(
            torch.from_numpy(data).unsqueeze(0), source_rate, target_rate
        )
        data = resampled.squeeze(0).numpy()

    return np.ascontiguousarray(data, dtype=np.float32)


def split_normal_train(
    samples: Sequence[AudioSample], train_fraction: float = 0.7, seed: int = 0
) -> tuple[list[AudioSample], list[AudioSample]]:
    """
    Carve a normal-only training fold out of an unsplit dataset.

    Anomalies are never placed in the training fold. Training the autoencoder on
    them would teach it to reconstruct exactly the sounds it is supposed to fail
    at, which quietly destroys the detector while leaving the loss curve looking
    healthy.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    normal = [s for s in samples if s.label == 0]
    anomalous = [s for s in samples if s.label == 1]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normal))
    cut = int(len(normal) * train_fraction)

    train = [normal[i] for i in order[:cut]]
    test = [normal[i] for i in order[cut:]] + anomalous
    return train, test
