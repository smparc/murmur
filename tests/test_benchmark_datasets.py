"""
Tests for the public-dataset loaders.

The real corpora are tens of gigabytes and separately licensed, so these tests
build miniature trees with the same directory and filename conventions. That
covers what actually breaks in practice — layout assumptions, label parsing, and
the train/test split — without a download.
"""

import numpy as np
import pytest
from scipy.io import wavfile

from benchmarks.datasets import (
    DatasetNotFound,
    DCASEDataset,
    IMSDataset,
    MIMIIDataset,
    load_dataset,
    split_normal_train,
)


def _write_wav(path, seconds=0.2, rate=16_000, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.1, int(rate * seconds)).astype(np.float32)
    wavfile.write(path, rate, samples)
    return path


@pytest.fixture
def mimii_root(tmp_path):
    root = tmp_path / "mimii"
    for machine in ("fan", "pump"):
        for unit in ("id_00", "id_02"):
            for condition, count in (("normal", 4), ("abnormal", 2)):
                for i in range(count):
                    _write_wav(root / machine / unit / condition / f"{i:08d}.wav", seed=i)
    return root


@pytest.fixture
def dcase_root(tmp_path):
    root = tmp_path / "dcase"
    for machine in ("fan", "valve"):
        for i in range(4):
            _write_wav(root / machine / "train" / f"normal_id_00_{i:08d}.wav", seed=i)
        for i in range(2):
            _write_wav(root / machine / "test" / f"normal_id_00_{i:08d}.wav", seed=i)
            _write_wav(root / machine / "test" / f"anomaly_id_00_{i:08d}.wav", seed=i + 50)
    return root


@pytest.fixture
def ims_root(tmp_path):
    root = tmp_path / "ims"
    run = root / "1st_test"
    run.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(20):
        stamp = f"2003.10.22.12.{i:02d}.24"
        np.savetxt(run / stamp, rng.normal(0, 0.1, (64, 4)))
    return root


class TestMissingRoot:
    def test_raises_with_download_instructions(self, tmp_path):
        with pytest.raises(DatasetNotFound) as excinfo:
            MIMIIDataset(tmp_path / "nope")
        message = str(excinfo.value)
        assert "zenodo" in message.lower()

    def test_every_loader_explains_where_to_get_the_data(self, tmp_path):
        for loader in (MIMIIDataset, DCASEDataset, IMSDataset):
            with pytest.raises(DatasetNotFound) as excinfo:
                loader(tmp_path / "missing")
            assert "http" in str(excinfo.value)


class TestMIMII:
    def test_discovers_every_recording(self, mimii_root):
        dataset = MIMIIDataset(mimii_root)
        # 2 machines x 2 units x (4 normal + 2 abnormal)
        assert len(dataset.samples()) == 24

    def test_labels_follow_the_condition_directory(self, mimii_root):
        samples = MIMIIDataset(mimii_root).samples()
        assert sum(s.label for s in samples) == 8
        for sample in samples:
            assert sample.is_anomalous == ("abnormal" in sample.path.parts)

    def test_groups_are_machine_and_unit(self, mimii_root):
        assert MIMIIDataset(mimii_root).groups() == [
            "fan/id_00",
            "fan/id_02",
            "pump/id_00",
            "pump/id_02",
        ]

    def test_audio_loads_as_mono_float32(self, mimii_root):
        dataset = MIMIIDataset(mimii_root)
        audio = dataset.load_audio(dataset.samples()[0])
        assert audio.ndim == 1
        assert audio.dtype == np.float32


class TestWavLoading:
    def test_int16_is_normalised_into_unit_range(self, tmp_path):
        path = tmp_path / "fan" / "id_00" / "normal" / "a.wav"
        path.parent.mkdir(parents=True)
        pcm = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        wavfile.write(path, 16_000, pcm)

        dataset = MIMIIDataset(tmp_path)
        audio = dataset.load_audio(dataset.samples()[0])
        assert audio.dtype == np.float32
        assert audio.min() >= -1.0 and audio.max() <= 1.0
        assert audio[0] == pytest.approx(-1.0)
        assert audio[2] == pytest.approx(0.0)

    def test_multichannel_is_mixed_to_mono(self, tmp_path):
        path = tmp_path / "fan" / "id_00" / "normal" / "a.wav"
        path.parent.mkdir(parents=True)
        stereo = np.stack(
            [np.full(64, 0.5, np.float32), np.full(64, -0.1, np.float32)], axis=1
        )
        wavfile.write(path, 16_000, stereo)

        dataset = MIMIIDataset(tmp_path)
        audio = dataset.load_audio(dataset.samples()[0])
        assert audio.ndim == 1
        assert audio[0] == pytest.approx(0.2)

    def test_resamples_to_the_requested_rate(self, tmp_path):
        path = tmp_path / "fan" / "id_00" / "normal" / "a.wav"
        path.parent.mkdir(parents=True)
        wavfile.write(path, 32_000, np.zeros(3200, dtype=np.float32))

        dataset = MIMIIDataset(tmp_path, sample_rate=16_000)
        audio = dataset.load_audio(dataset.samples()[0])
        assert audio.shape[0] == pytest.approx(1600, abs=8)


class TestDCASE:
    def test_respects_the_official_split(self, dcase_root):
        dataset = DCASEDataset(dcase_root)
        assert len(dataset.samples("train")) == 8
        assert len(dataset.samples("test")) == 8

    def test_training_fold_contains_no_anomalies(self, dcase_root):
        assert all(s.label == 0 for s in DCASEDataset(dcase_root).samples("train"))

    def test_label_is_parsed_from_the_filename(self, dcase_root):
        test_samples = DCASEDataset(dcase_root).samples("test")
        assert sum(s.label for s in test_samples) == 4
        for sample in test_samples:
            assert sample.is_anomalous == sample.path.name.startswith("anomaly")

    def test_machine_id_is_parsed_from_the_filename(self, dcase_root):
        assert all(s.machine_id == "id_00" for s in DCASEDataset(dcase_root).samples())

    def test_unrecognised_filenames_are_skipped_not_fatal(self, dcase_root):
        _write_wav(dcase_root / "fan" / "test" / "README_notes.wav")
        # Still 8: the stray file is ignored rather than mislabelled.
        assert len(DCASEDataset(dcase_root).samples("test")) == 8


class TestIMS:
    def test_reads_the_chronological_run(self, ims_root):
        assert len(IMSDataset(ims_root).samples()) == 20

    def test_final_fraction_is_labelled_degraded(self, ims_root):
        samples = IMSDataset(ims_root, failure_fraction=0.25).samples()
        assert sum(s.label for s in samples) == 5
        # And it must be the *last* five, since the run ends in failure.
        assert all(s.label == 1 for s in samples[-5:])
        assert all(s.label == 0 for s in samples[:-5])

    def test_loads_the_selected_channel(self, ims_root):
        dataset = IMSDataset(ims_root, channel=2)
        audio = dataset.load_audio(dataset.samples()[0])
        assert audio.shape == (64,)

    def test_out_of_range_channel_is_reported(self, ims_root):
        dataset = IMSDataset(ims_root, channel=9)
        with pytest.raises(IndexError):
            dataset.load_audio(dataset.samples()[0])

    def test_rejects_invalid_failure_fraction(self, ims_root):
        for bad in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError):
                IMSDataset(ims_root, failure_fraction=bad)


class TestSplitNormalTrain:
    def test_training_fold_never_contains_anomalies(self, mimii_root):
        samples = MIMIIDataset(mimii_root).samples()
        train, test = split_normal_train(samples, train_fraction=0.7)
        assert all(s.label == 0 for s in train)
        assert sum(s.label for s in test) == 8

    def test_split_is_a_partition(self, mimii_root):
        samples = MIMIIDataset(mimii_root).samples()
        train, test = split_normal_train(samples)
        assert len(train) + len(test) == len(samples)
        assert not ({s.path for s in train} & {s.path for s in test})

    def test_is_deterministic_for_a_seed(self, mimii_root):
        samples = MIMIIDataset(mimii_root).samples()
        a, _ = split_normal_train(samples, seed=3)
        b, _ = split_normal_train(samples, seed=3)
        assert [s.path for s in a] == [s.path for s in b]

    def test_rejects_invalid_fraction(self, mimii_root):
        samples = MIMIIDataset(mimii_root).samples()
        with pytest.raises(ValueError):
            split_normal_train(samples, train_fraction=1.0)


class TestRegistry:
    def test_load_dataset_by_name(self, mimii_root):
        assert isinstance(load_dataset("mimii", mimii_root), MIMIIDataset)
        assert isinstance(load_dataset("MIMII", mimii_root), MIMIIDataset)

    def test_unknown_name_lists_the_options(self, mimii_root):
        with pytest.raises(ValueError, match="dcase"):
            load_dataset("nonsense", mimii_root)
