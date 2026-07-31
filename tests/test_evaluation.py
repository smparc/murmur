"""Tests for the detection metrics and the MIMII benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.metrics import detection_report, partial_auc, roc_auc, roc_curve
from src.evaluation.mimii import (
    chunk_waveform,
    discover,
    evaluate,
    load_waveform,
    score_sample,
    synthetic_corpus,
)
from src.settings import settings


class TestRocAuc:
    def test_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(1.0)

    def test_inverted_separation(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.0)

    def test_all_ties_is_chance(self):
        """Every score identical means the detector conveys nothing."""
        assert roc_auc(np.ones(10), np.array([0] * 5 + [1] * 5)) == pytest.approx(0.5)

    def test_random_scores_near_chance(self):
        rng = np.random.default_rng(0)
        scores = rng.random(20_000)
        labels = rng.integers(0, 2, 20_000)
        assert roc_auc(scores, labels) == pytest.approx(0.5, abs=0.02)

    def test_matches_brute_force_definition(self):
        """AUC is P(score(anomaly) > score(normal)), ties counting a half."""
        rng = np.random.default_rng(5)
        scores = rng.integers(0, 5, 60).astype(float)  # many ties on purpose
        labels = rng.integers(0, 2, 60)
        pos, neg = scores[labels == 1], scores[labels == 0]

        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        assert roc_auc(scores, labels) == pytest.approx(wins / (pos.size * neg.size))

    def test_requires_both_classes(self):
        with pytest.raises(ValueError):
            roc_auc(np.array([0.1, 0.2]), np.array([1, 1]))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            roc_auc(np.zeros(3), np.zeros(4))


class TestRocCurve:
    def test_starts_at_origin_and_is_monotone(self):
        rng = np.random.default_rng(7)
        scores = rng.random(200)
        labels = rng.integers(0, 2, 200)
        fpr, tpr = roc_curve(scores, labels)

        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert np.all(np.diff(fpr) >= -1e-12)
        assert np.all(np.diff(tpr) >= -1e-12)
        assert fpr[-1] == pytest.approx(1.0)
        assert tpr[-1] == pytest.approx(1.0)


class TestPartialAuc:
    def test_perfect_detector_scores_one(self):
        scores = np.r_[np.zeros(50), np.ones(50)]
        labels = np.r_[np.zeros(50), np.ones(50)].astype(int)
        assert partial_auc(scores, labels, max_fpr=0.1) == pytest.approx(1.0)

    def test_chance_detector_scores_half_the_fpr_budget(self):
        """
        pAUC here is mean TPR over [0, max_fpr]. Chance has TPR == FPR across
        that strip, so it averages max_fpr/2 — not 0.5. See the normalisation
        note in `partial_auc`.
        """
        rng = np.random.default_rng(3)
        scores = rng.random(40_000)
        labels = rng.integers(0, 2, 40_000)
        assert partial_auc(scores, labels, max_fpr=0.1) == pytest.approx(0.05, abs=0.02)

    def test_full_range_equals_auc(self):
        rng = np.random.default_rng(9)
        scores = rng.random(500)
        labels = rng.integers(0, 2, 500)
        assert partial_auc(scores, labels, max_fpr=1.0) == pytest.approx(
            roc_auc(scores, labels), abs=1e-9
        )

    def test_penalises_a_detector_that_is_only_good_at_high_fpr(self):
        """
        The reason pAUC exists: a detector can post a decent overall AUC while
        being useless in the low-false-alarm regime anyone would actually run.
        """
        # 10% of normals outrank every anomaly. Global ordering stays good, but
        # the false-alarm budget is exhausted before a single fault is caught.
        normal = np.r_[np.linspace(0.9, 1.0, 20), np.linspace(0.0, 0.3, 180)]
        anomalous = np.linspace(0.4, 0.8, 200)
        scores = np.r_[normal, anomalous]
        labels = np.r_[np.zeros(200), np.ones(200)].astype(int)

        assert roc_auc(scores, labels) == pytest.approx(0.9, abs=0.01)
        assert partial_auc(scores, labels, max_fpr=0.1) < 0.05

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_invalid_max_fpr(self, bad):
        with pytest.raises(ValueError):
            partial_auc(np.array([0.0, 1.0]), np.array([0, 1]), max_fpr=bad)


class TestDetectionReport:
    def test_bundles_all_metrics(self):
        scores = np.r_[np.zeros(10), np.ones(10)]
        labels = np.r_[np.zeros(10), np.ones(10)].astype(int)
        report = detection_report(scores, labels)
        assert set(report) == {"auc", "pauc", "max_fpr", "n_normal", "n_anomalous"}
        assert report["n_normal"] == 10
        assert report["n_anomalous"] == 10


class TestChunking:
    def test_splits_into_whole_chunks(self):
        chunks = chunk_waveform(np.zeros(8000), 2000)
        assert len(chunks) == 4
        assert all(c.size == 2000 for c in chunks)

    def test_discards_trailing_partial_chunk(self):
        chunks = chunk_waveform(np.zeros(4500), 2000)
        assert len(chunks) == 2

    def test_short_clip_returned_whole(self):
        chunks = chunk_waveform(np.zeros(500), 2000)
        assert len(chunks) == 1 and chunks[0].size == 500

    def test_rejects_bad_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_waveform(np.zeros(10), 0)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A small MIMII-layout corpus written to disk once per module."""
    root = tmp_path_factory.mktemp("mimii")
    samples = synthetic_corpus(root, per_class=5, duration=1.5, seed=2)
    return root, samples


class TestDiscovery:
    def test_finds_every_clip(self, corpus):
        root, written = corpus
        found = discover(root)
        assert len(found) == len(written)

    def test_labels_from_directory_name(self, corpus):
        root, _ = corpus
        found = discover(root)
        assert {s.label for s in found} == {0, 1}
        assert all(s.path.parent.name == ("abnormal" if s.label else "normal") for s in found)

    def test_parses_machine_and_id(self, corpus):
        root, _ = corpus
        found = discover(root)
        assert {s.machine for s in found} == {"pump", "valve"}
        assert {s.machine_id for s in found} == {"00"}
        assert {s.group for s in found} == {"pump/00", "valve/00"}

    def test_ignores_unlabelled_files(self, corpus, tmp_path):
        root, _ = corpus
        stray = root / "loose.wav"
        stray.write_bytes(b"not audio")
        assert all(s.path != stray for s in discover(root))

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover(tmp_path / "nope")


class TestEndToEndBenchmark:
    def test_production_transform_is_used(self, corpus):
        """
        The harness must score through the same log-mel transform the ingestion
        service uses. Reimplementing features here would benchmark a model that
        never runs in production.
        """
        from src.ingestion.cuda_stream_processor import get_mel_spectrogram_transform

        root, _ = corpus
        waveform = load_waveform(discover(root)[0].path, settings.SAMPLE_RATE)
        mel = get_mel_spectrogram_transform().to("cpu")

        captured = {}

        def score_fn(x):
            captured["shape"] = tuple(x.shape)
            return x.flatten(1).mean(dim=1)

        score_sample(
            waveform, mel, score_fn, settings.SAMPLES_PER_CHUNK, device=torch.device("cpu")
        )
        # (chunks, 1, n_mels, frames) — the SpectrogramAutoencoder's signature.
        assert captured["shape"][1] == 1
        assert captured["shape"][2] == settings.N_MELS

    def test_energy_detector_separates_the_synthetic_corpus(self, corpus):
        """
        A full pass through discovery, loading, chunking, the production
        transform and ROC. The synthetic anomalies carry a 2.5 kHz squeal plus
        impulses, so a plain energy detector should find them comfortably.
        """
        root, _ = corpus
        report = evaluate(
            discover(root),
            score_fn=lambda x: x.flatten(1).mean(dim=1),
            device=torch.device("cpu"),
            progress_every=0,
        )

        assert report["overall"]["auc"] > 0.9
        assert report["n_clips"] == 20
        assert set(report["per_machine"]) == {"pump/00", "valve/00"}
        assert report["macro_auc"] > 0.9

    def test_real_autoencoder_runs_end_to_end(self, corpus):
        """Untrained, so no accuracy claim — this asserts the wiring holds."""
        from src.detection.anomaly_detector import SpectrogramAutoencoder

        root, _ = corpus
        autoencoder = SpectrogramAutoencoder(
            n_mels=settings.N_MELS, latent_dim=settings.AE_LATENT_DIM
        ).eval()

        report = evaluate(
            discover(root),
            score_fn=autoencoder.anomaly_score,
            device=torch.device("cpu"),
            progress_every=0,
        )
        assert 0.0 <= report["overall"]["auc"] <= 1.0
        assert 0.0 <= report["overall"]["pauc"] <= 1.0

    def test_max_aggregation_supported(self, corpus):
        root, _ = corpus
        report = evaluate(
            discover(root),
            score_fn=lambda x: x.flatten(1).mean(dim=1),
            device=torch.device("cpu"),
            aggregate="max",
            progress_every=0,
        )
        assert report["aggregate"] == "max"

    def test_unknown_aggregate_rejected(self, corpus):
        root, _ = corpus
        with pytest.raises(ValueError):
            evaluate(
                discover(root),
                score_fn=lambda x: x.flatten(1).mean(dim=1),
                device=torch.device("cpu"),
                aggregate="median",
                progress_every=0,
            )

    def test_single_class_group_reported_not_crashed(self, tmp_path):
        """A machine with only normal recordings has no ROC; say so."""
        samples = synthetic_corpus(tmp_path, machines=("fan",), per_class=3, duration=1.0)
        normals = [s for s in samples if s.label == 0]
        mixed = normals + [s for s in samples if s.label == 1][:1]

        # Force a second group that contains only normal clips.
        from dataclasses import replace

        lopsided = mixed + [replace(s, machine="pump") for s in normals]
        report = evaluate(
            lopsided,
            score_fn=lambda x: x.flatten(1).mean(dim=1),
            device=torch.device("cpu"),
            progress_every=0,
        )
        assert "note" in report["per_machine"]["pump/00"]
        assert np.isnan(report["per_machine"]["pump/00"]["auc"])

    def test_empty_sample_list_rejected(self):
        with pytest.raises(ValueError):
            evaluate([], score_fn=lambda x: x, device=torch.device("cpu"))
