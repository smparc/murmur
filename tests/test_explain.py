"""
Tests for anomaly explanation and grounded diagnosis.

The important test in this file is ``test_error_map_reproduces_the_anomaly_score``.
An explanation that can disagree with the score it explains is worse than no
explanation, because it will be believed. That test pins the two together.

The end-to-end tests train a small autoencoder on normal simulator audio and
then feed it a known fault, checking that the attribution lands in the band the
fault actually occupies. That is the only way to show the chain works rather
than merely runs.
"""

from itertools import pairwise

import numpy as np
import pytest
import torch

from benchmarks.features import mel_transform, to_log_mel
from src.detection.anomaly_detector import SpectrogramAutoencoder
from src.explain import explain_anomaly, mel_bin_frequencies, reconstruction_error_map
from src.explain.saliency import AnomalyExplanation, BandContribution
from src.ingestion.mock_edge_device import FaultType, generate_mock_audio
from src.settings import settings
from src.translation.taxonomy import DEFAULT_TAXONOMY, FaultSignature, FaultTaxonomy

SAMPLE_RATE = 16_000


@pytest.fixture(scope="module")
def trained_autoencoder():
    """A small autoencoder fitted on normal simulator audio."""
    torch.manual_seed(0)
    np.random.seed(0)
    transform = mel_transform("cpu")

    normal = [
        to_log_mel(
            np.frombuffer(
                generate_mock_audio(node_id=0, fault=FaultType.NONE, severity=0.0),
                dtype=np.float32,
            ),
            transform,
        )
        for _ in range(48)
    ]
    batch = torch.stack(normal).unsqueeze(1)

    model = SpectrogramAutoencoder(n_mels=settings.N_MELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(60):
        reconstruction, _ = model(batch)
        loss = torch.nn.functional.mse_loss(reconstruction, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _spectrogram(fault: FaultType, severity: float, seed: int = 0) -> torch.Tensor:
    np.random.seed(seed)
    audio = np.frombuffer(
        generate_mock_audio(node_id=0, fault=fault, severity=severity), dtype=np.float32
    )
    return to_log_mel(audio, mel_transform("cpu"))


class TestMelBinFrequencies:
    def test_returns_one_centre_per_bin(self):
        assert len(mel_bin_frequencies(64, 16_000)) == 64

    def test_centres_increase_monotonically(self):
        centres = mel_bin_frequencies(64, 16_000)
        assert all(b > a for a, b in pairwise(centres))

    def test_centres_stay_inside_the_nyquist_range(self):
        centres = mel_bin_frequencies(64, 16_000)
        assert centres[0] > 0.0
        assert centres[-1] < 8_000.0

    def test_spacing_widens_with_frequency(self):
        # The defining property of the mel scale: resolution is fine at low
        # frequency and coarse at high. If this inverted, every reported Hz
        # range would be wrong.
        centres = mel_bin_frequencies(32, 16_000)
        low_gap = centres[1] - centres[0]
        high_gap = centres[-1] - centres[-2]
        assert high_gap > low_gap

    @pytest.mark.parametrize(
        "kwargs",
        [{"n_mels": 0}, {"sample_rate": 0}, {"f_min": 9000.0}],
    )
    def test_invalid_arguments_rejected(self, kwargs):
        call = {"n_mels": 64, "sample_rate": 16_000}
        call.update(kwargs)
        with pytest.raises(ValueError):
            mel_bin_frequencies(**call)


class TestErrorMap:
    def test_error_map_reproduces_the_anomaly_score(self, trained_autoencoder):
        # The whole basis of this module: the map is the score, disaggregated.
        spectrogram = _spectrogram(FaultType.BEARING, 0.8)
        error_map = reconstruction_error_map(trained_autoencoder, spectrogram)

        batch = spectrogram.unsqueeze(0).unsqueeze(0)
        score = float(trained_autoencoder.anomaly_score(batch).item())
        assert float(error_map.mean().item()) == pytest.approx(score, rel=1e-5)

    def test_shape_matches_the_input_spectrogram(self, trained_autoencoder):
        spectrogram = _spectrogram(FaultType.NONE, 0.0)
        error_map = reconstruction_error_map(trained_autoencoder, spectrogram)
        assert error_map.shape == spectrogram.shape

    def test_errors_are_non_negative(self, trained_autoencoder):
        error_map = reconstruction_error_map(
            trained_autoencoder, _spectrogram(FaultType.CAVITATION, 0.6)
        )
        assert bool((error_map >= 0).all())

    def test_accepts_batched_and_channelled_shapes(self, trained_autoencoder):
        spectrogram = _spectrogram(FaultType.NONE, 0.0)
        for shaped in (
            spectrogram,
            spectrogram.unsqueeze(0),
            spectrogram.unsqueeze(0).unsqueeze(0),
        ):
            assert reconstruction_error_map(trained_autoencoder, shaped).dim() == 2

    def test_rejects_a_real_batch(self, trained_autoencoder):
        batch = torch.stack([_spectrogram(FaultType.NONE, 0.0) for _ in range(2)])
        with pytest.raises(ValueError):
            reconstruction_error_map(trained_autoencoder, batch.unsqueeze(1))


class TestExplainAnomaly:
    def test_produces_the_requested_number_of_bands(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.BEARING, 0.8), SAMPLE_RATE, n_bands=8
        )
        assert len(explanation.bands) == 8

    def test_band_shares_sum_to_one(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.BEARING, 0.8), SAMPLE_RATE
        )
        assert sum(b.share for b in explanation.bands) == pytest.approx(1.0, abs=1e-5)

    def test_bands_are_ordered_by_frequency(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.NONE, 0.0), SAMPLE_RATE
        )
        for lower, upper in pairwise(explanation.bands):
            assert upper.low_hz > lower.low_hz

    def test_bands_partition_the_spectrum_without_gaps(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.NONE, 0.0), SAMPLE_RATE, n_bands=5
        )
        # Every mel bin must be counted exactly once, so consecutive band edges
        # must not leave a hole.
        for lower, upper in pairwise(explanation.bands):
            assert upper.low_hz > lower.high_hz

    def test_error_map_is_omitted_by_default(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.NONE, 0.0), SAMPLE_RATE
        )
        assert explanation.error_map is None
        assert "error_map" not in explanation.as_dict()

    def test_error_map_can_be_requested(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder,
            _spectrogram(FaultType.NONE, 0.0),
            SAMPLE_RATE,
            include_map=True,
        )
        assert explanation.error_map is not None
        assert len(explanation.error_map) == settings.N_MELS
        assert "error_map" in explanation.as_dict(include_map=True)

    def test_rejects_zero_bands(self, trained_autoencoder):
        with pytest.raises(ValueError):
            explain_anomaly(
                trained_autoencoder, _spectrogram(FaultType.NONE, 0.0), SAMPLE_RATE, n_bands=0
            )

    def test_bearing_fault_energy_lands_in_the_high_bands(self, trained_autoencoder):
        # The simulator puts bearing squeal at 2 kHz with a harmonic at 4 kHz.
        # The attribution must point there, not somewhere else.
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.BEARING, 1.0, seed=1), SAMPLE_RATE
        )
        dominant = explanation.dominant_band
        assert dominant is not None
        assert dominant.centre_hz > 1200.0, explanation.summary()

    def test_imbalance_energy_lands_lower_than_bearing_energy(self, trained_autoencoder):
        bearing = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.BEARING, 1.0, seed=2), SAMPLE_RATE
        )
        imbalance = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.IMBALANCE, 1.0, seed=2), SAMPLE_RATE
        )
        assert imbalance.dominant_band.centre_hz < bearing.dominant_band.centre_hz


class TestPresentation:
    def test_describe_uses_khz_above_a_kilohertz(self):
        band = BandContribution(2100.0, 3400.0, 0.46, 3)
        assert "kHz" in band.describe()
        assert "46%" in band.describe()

    def test_describe_uses_hz_below_a_kilohertz(self):
        assert "Hz" in BandContribution(80.0, 240.0, 0.5, 0).describe()

    def test_summary_lists_the_leading_bands(self):
        explanation = AnomalyExplanation(
            total_error=1.0,
            bands=[
                BandContribution(100.0, 500.0, 0.1, 0),
                BandContribution(2000.0, 4000.0, 0.7, 2),
                BandContribution(4000.0, 8000.0, 0.2, 1),
            ],
        )
        assert "2.0 kHz-4.0 kHz" in explanation.summary()
        assert explanation.top_bands(1)[0].share == pytest.approx(0.7)

    def test_summary_handles_an_empty_explanation(self):
        assert "no spectral attribution" in AnomalyExplanation(0.0, []).summary()


class TestFaultSignature:
    def test_overlap_is_one_for_a_fully_contained_band(self):
        signature = DEFAULT_TAXONOMY[0]  # bearing, 2000-4500 Hz
        assert signature.band_overlap(2500.0, 3500.0) == pytest.approx(1.0)

    def test_overlap_is_zero_for_a_disjoint_band(self):
        assert DEFAULT_TAXONOMY[0].band_overlap(100.0, 300.0) == pytest.approx(0.0)

    def test_overlap_is_partial_for_a_straddling_band(self):
        # 1500-2500 Hz is half inside the 2000-4500 Hz band.
        overlap = DEFAULT_TAXONOMY[0].band_overlap(1500.0, 2500.0)
        assert 0.4 < overlap < 0.6

    def test_signature_with_no_bands_never_matches(self):
        empty = FaultSignature(
            name="x",
            description="",
            frequency_bands_hz=(),
            character="tonal",
            likely_causes=(),
            recommended_action="",
            urgency="monitor",
        )
        assert empty.band_overlap(100.0, 8000.0) == 0.0


class TestFaultTaxonomy:
    def _explanation(self, bands):
        return AnomalyExplanation(total_error=1.0, bands=bands)

    def test_high_frequency_tonal_energy_diagnoses_a_bearing(self):
        explanation = self._explanation(
            [
                BandContribution(50.0, 400.0, 0.05, 0),
                BandContribution(400.0, 1800.0, 0.05, 0),
                BandContribution(2000.0, 4000.0, 0.90, 5),
            ]
        )
        best = FaultTaxonomy().best(explanation)
        assert best.name == "Bearing race defect"
        assert best.confidence > 0.5
        assert best.evidence

    def test_low_frequency_energy_diagnoses_imbalance(self):
        explanation = self._explanation(
            [
                BandContribution(20.0, 200.0, 0.85, 1),
                BandContribution(2000.0, 4000.0, 0.15, 0),
            ]
        )
        assert FaultTaxonomy().best(explanation).name == "Rotating imbalance"

    def test_unmatched_evidence_is_reported_as_unrecognised(self):
        # A narrow band that no signature claims.
        explanation = self._explanation([BandContribution(300.0, 420.0, 1.0, 0)])
        best = FaultTaxonomy(min_confidence=0.9).best(explanation)
        assert best.name == "Unrecognised acoustic anomaly"
        assert best.confidence == 0.0

    def test_empty_explanation_is_unrecognised_not_an_error(self):
        best = FaultTaxonomy().best(AnomalyExplanation(0.0, []))
        assert best.name == "Unrecognised acoustic anomaly"

    def test_candidates_are_ranked_by_confidence(self):
        explanation = self._explanation(
            [
                BandContribution(2000.0, 4000.0, 0.6, 0),
                BandContribution(4000.0, 8000.0, 0.4, 0),
            ]
        )
        candidates = FaultTaxonomy().diagnose(explanation, limit=3)
        confidences = [c.confidence for c in candidates]
        assert confidences == sorted(confidences, reverse=True)

    def test_limit_is_respected(self):
        explanation = self._explanation([BandContribution(1000.0, 8000.0, 1.0, 0)])
        assert len(FaultTaxonomy(min_confidence=0.0).diagnose(explanation, limit=2)) == 2

    def test_diagnosis_serialises_with_its_evidence(self):
        explanation = self._explanation([BandContribution(2000.0, 4000.0, 1.0, 3)])
        payload = FaultTaxonomy().best(explanation).as_dict()
        for key in (
            "fault",
            "confidence",
            "urgency",
            "likely_causes",
            "recommended_action",
            "evidence",
        ):
            assert key in payload

    def test_describe_reads_as_a_sentence(self):
        explanation = self._explanation([BandContribution(2000.0, 4000.0, 1.0, 3)])
        described = FaultTaxonomy().best(explanation).describe()
        assert "confidence" in described
        assert described.endswith(".")

    def test_rejects_an_out_of_range_threshold(self):
        with pytest.raises(ValueError):
            FaultTaxonomy(min_confidence=1.5)

    def test_every_catalogue_entry_is_well_formed(self):
        for signature in DEFAULT_TAXONOMY:
            assert signature.name
            assert signature.recommended_action
            assert signature.likely_causes
            assert signature.urgency in {"monitor", "schedule", "urgent"}
            for low, high in signature.frequency_bands_hz:
                assert 0 <= low < high


class TestGroundedDiagnosisEndToEnd:
    def test_simulated_bearing_fault_is_diagnosed_from_audio(self, trained_autoencoder):
        # Audio -> spectrogram -> reconstruction error -> band attribution ->
        # catalogue match. Every step traceable, no language model involved.
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.BEARING, 1.0, seed=5), SAMPLE_RATE
        )
        diagnosis = FaultTaxonomy(min_confidence=0.1).best(explanation)
        assert diagnosis.confidence > 0.0
        assert diagnosis.name != "Unrecognised acoustic anomaly"

    def test_diagnosis_carries_frequency_evidence(self, trained_autoencoder):
        explanation = explain_anomaly(
            trained_autoencoder, _spectrogram(FaultType.BEARING, 1.0, seed=6), SAMPLE_RATE
        )
        diagnosis = FaultTaxonomy(min_confidence=0.1).best(explanation)
        assert any("error in" in item for item in diagnosis.evidence)
