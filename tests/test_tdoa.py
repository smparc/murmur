"""Tests for GCC-PHAT delay estimation and hyperbolic source localization."""

from __future__ import annotations

import numpy as np
import pytest

from src.mapping.tdoa import (
    GDOP_UNRELIABLE,
    SPEED_OF_SOUND,
    TDOAEstimate,
    gcc_phat,
    localize_source,
    max_delay_between,
    pairwise_tdoa,
    tdoa_edge_weights,
)

FS = 16_000


def _delayed_pair(
    delay_samples: int, n: int = 4096, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A broadband signal and a copy of it shifted by a known integer delay."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(n + 2 * abs(delay_samples) + 512)
    pad = abs(delay_samples) + 128
    reference = source[pad : pad + n]
    shifted = source[pad - delay_samples : pad - delay_samples + n]
    return shifted, reference


class TestGccPhat:
    def test_zero_delay(self):
        sig, ref = _delayed_pair(0)
        tau, coherence = gcc_phat(sig, ref, FS)
        assert abs(tau) < 1e-4
        assert coherence > 0.9

    @pytest.mark.parametrize("delay_samples", [-64, -13, 7, 55])
    def test_recovers_known_delay(self, delay_samples):
        sig, ref = _delayed_pair(delay_samples)
        tau, coherence = gcc_phat(sig, ref, FS)
        expected = delay_samples / FS
        # Sub-sample interpolation should land well inside one sample period.
        assert abs(tau - expected) < 1.0 / FS
        assert coherence > 0.8

    def test_identical_signals_give_unit_coherence(self):
        """PHAT normalisation makes a perfect match peak at exactly 1.0."""
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(2048)
        _, coherence = gcc_phat(sig, sig, FS)
        assert coherence == pytest.approx(1.0, abs=1e-6)

    def test_uncorrelated_signals_give_low_coherence(self):
        rng = np.random.default_rng(4)
        a = rng.standard_normal(4096)
        b = rng.standard_normal(4096)
        _, coherence = gcc_phat(a, b, FS)
        assert coherence < 0.2

    def test_scale_invariance(self):
        """Amplitude carries no delay information; PHAT must ignore it."""
        sig, ref = _delayed_pair(31)
        tau_a, _ = gcc_phat(sig, ref, FS)
        tau_b, _ = gcc_phat(sig * 100.0, ref * 0.01, FS)
        assert tau_a == pytest.approx(tau_b, abs=1e-9)

    def test_dominant_tone_does_not_capture_the_peak(self):
        """
        The property that motivates PHAT over plain cross-correlation: a loud
        shared mains hum must not drag the estimate toward zero lag.
        """
        n = 8192
        t = np.arange(n) / FS
        hum = 5.0 * np.sin(2 * np.pi * 60 * t)
        sig, ref = _delayed_pair(40, n=n, seed=11)
        tau, _ = gcc_phat(sig + hum, ref + hum, FS)
        assert abs(tau - 40 / FS) < 2.0 / FS

    def test_max_tau_bounds_the_search(self):
        sig, ref = _delayed_pair(200)
        bound = 50 / FS
        tau, _ = gcc_phat(sig, ref, FS, max_tau=bound)
        assert abs(tau) <= bound + 1e-9

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            gcc_phat(np.array([]), np.ones(10), FS)

    def test_rejects_bad_interp(self):
        with pytest.raises(ValueError):
            gcc_phat(np.ones(10), np.ones(10), FS, interp=0)


class TestMaxDelay:
    def test_matches_physics(self):
        delay = max_delay_between(np.array([0.0, 0, 0]), np.array([3.43, 0, 0]))
        assert delay == pytest.approx(0.01, rel=1e-6)

    def test_symmetric(self):
        a, b = np.array([1.0, 2, 3]), np.array([4.0, 5, 6])
        assert max_delay_between(a, b) == pytest.approx(max_delay_between(b, a))


def _simulate_array(source, coords, n=8000, seed=5):
    """Render what each microphone hears from a single broadband source."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n + 4096)
    channels = []
    for p in coords:
        delay = round(np.linalg.norm(np.asarray(source) - p) / SPEED_OF_SOUND * FS)
        channels.append(signal[delay : delay + n])
    return np.stack(channels)


class TestLocalization:
    COORDS = np.array([(0.0, 0.0, 3.0), (5.0, 0.0, 3.0), (0.0, 10.0, 3.0), (5.0, 10.0, 3.0)])

    @pytest.mark.parametrize("source", [(3.5, 7.0, 3.0), (1.0, 2.0, 3.0), (2.5, 5.0, 3.0)])
    def test_recovers_source_position(self, source):
        channels = _simulate_array(source, self.COORDS)
        estimates = pairwise_tdoa(channels, self.COORDS, FS)
        position, residual = localize_source(estimates, self.COORDS, plane_z=3.0)

        assert position is not None
        assert np.linalg.norm(position - np.asarray(source)) < 0.5
        assert residual < 1.0

    def test_pairwise_covers_all_pairs(self):
        channels = _simulate_array((2.5, 5.0, 3.0), self.COORDS)
        estimates = pairwise_tdoa(channels, self.COORDS, FS)
        assert len(estimates) == 6  # C(4, 2)
        assert all(e.coherence > 0.5 for e in estimates)

    def test_pairs_argument_restricts_work(self):
        channels = _simulate_array((2.5, 5.0, 3.0), self.COORDS)
        estimates = pairwise_tdoa(channels, self.COORDS, FS, pairs=[(0, 1)])
        assert len(estimates) == 1
        assert (estimates[0].i, estimates[0].j) == (0, 1)

    def test_delays_respect_physical_bounds(self):
        channels = _simulate_array((3.5, 7.0, 3.0), self.COORDS)
        for e in pairwise_tdoa(channels, self.COORDS, FS):
            assert abs(e.tau) <= e.max_tau + 1e-9

    def test_incoherent_array_declines_to_localize(self):
        """Independent noise at every mic must not yield a confident position."""
        rng = np.random.default_rng(9)
        channels = rng.standard_normal((4, 8000))
        estimates = pairwise_tdoa(channels, self.COORDS, FS)
        position, _ = localize_source(estimates, self.COORDS, plane_z=3.0, min_coherence=0.5)
        assert position is None

    def test_too_few_estimates_returns_none(self):
        estimates = [TDOAEstimate(i=0, j=1, tau=0.0, coherence=1.0, max_tau=0.02)]
        position, residual = localize_source(estimates, self.COORDS, plane_z=3.0)
        assert position is None
        assert residual == float("inf")

    def test_coplanar_array_cannot_solve_for_z(self):
        """
        The default array is flat, so height is unobservable. The solver must
        report that rather than return a confident, arbitrary z.
        """
        channels = _simulate_array((3.5, 7.0, 3.0), self.COORDS)
        estimates = pairwise_tdoa(channels, self.COORDS, FS)
        position, residual = localize_source(estimates, self.COORDS, plane_z=None)
        assert position is None or residual == float("inf")

    def test_channel_count_must_match_coords(self):
        with pytest.raises(ValueError):
            pairwise_tdoa(np.zeros((3, 100)), self.COORDS, FS)

    # -- geometry quality --------------------------------------------------

    def _fix(self, source):
        channels = _simulate_array(source, self.COORDS)
        estimates = pairwise_tdoa(channels, self.COORDS, FS)
        return localize_source(estimates, self.COORDS, plane_z=3.0)

    def test_the_residual_cannot_detect_a_degenerate_geometry(self):
        """Why GDOP exists: the residual is blind to conditioning.

        With four microphones and three unknowns the system is exactly
        determined, so it fits perfectly whatever the geometry. A source at the
        array circumcentre — an entirely ordinary place for a machine to sit —
        therefore produces a residual indistinguishable from, and here actually
        *smaller* than, a well-conditioned fix out near the microphones.
        """
        good = self._fix((3.5, 7.0, 3.0))
        centroid = tuple(self.COORDS.mean(axis=0))
        degenerate = self._fix(centroid)

        assert good.residual < 1e-6
        assert degenerate.residual < 1e-6
        # The residual gives the caller nothing to discriminate on.
        assert abs(good.residual - degenerate.residual) < 1e-6

    def test_gdop_flags_what_the_residual_misses(self):
        good = self._fix((3.5, 7.0, 3.0))
        centroid = self.COORDS.mean(axis=0)
        # Just off the exact centroid, where the hyperbolas are near-parallel
        # but the matrix is not exactly singular.
        near_centroid = self._fix(tuple(centroid + np.array([0.05, 0.05, 0.0])))

        assert good.reliable
        assert good.gdop < GDOP_UNRELIABLE
        assert not near_centroid.reliable
        assert near_centroid.gdop > good.gdop

    def test_gdop_is_finite_and_small_for_well_placed_sources(self):
        for source in [(1.0, 1.0, 3.0), (3.5, 7.0, 3.0), (4.0, 2.0, 3.0)]:
            fix = self._fix(source)
            assert np.isfinite(fix.gdop), source
            assert fix.gdop < GDOP_UNRELIABLE, (source, fix.gdop)
            assert fix.reliable, source

    def test_an_unlocalised_fix_is_never_reliable(self):
        estimates = [TDOAEstimate(i=0, j=1, tau=0.0, coherence=1.0, max_tau=0.02)]
        fix = localize_source(estimates, self.COORDS, plane_z=3.0)
        assert fix.position is None
        assert not fix.reliable
        assert fix.gdop == float("inf")

    def test_fix_still_unpacks_as_the_original_pair(self):
        """The two-value contract predates the geometry field and still holds."""
        position, residual = self._fix((3.5, 7.0, 3.0))
        assert position is not None
        assert residual < 1.0

    def test_rejects_non_2d_channels(self):
        with pytest.raises(ValueError):
            pairwise_tdoa(np.zeros(100), self.COORDS, FS)


class TestEdgeWeightModulation:
    EDGE_INDEX = np.array([[0, 1, 0, 2], [1, 0, 2, 0]])
    STATIC = np.array([1.0, 1.0, 0.5, 0.5])

    def test_coherent_pair_retains_weight(self):
        estimates = [TDOAEstimate(0, 1, 0.0, coherence=1.0, max_tau=0.02)]
        out = tdoa_edge_weights(self.EDGE_INDEX, estimates, self.STATIC)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(1.0)

    def test_incoherent_pair_is_attenuated_not_severed(self):
        estimates = [TDOAEstimate(0, 1, 0.0, coherence=0.0, max_tau=0.02)]
        out = tdoa_edge_weights(self.EDGE_INDEX, estimates, self.STATIC, floor=0.05)
        assert out[0] == pytest.approx(0.05)
        assert out[0] > 0.0, "a fully severed graph degenerates the GCN to an MLP"

    def test_symmetric_across_edge_direction(self):
        estimates = [TDOAEstimate(1, 0, 0.0, coherence=0.4, max_tau=0.02)]
        out = tdoa_edge_weights(self.EDGE_INDEX, estimates, self.STATIC)
        assert out[0] == pytest.approx(out[1])

    def test_unmeasured_edge_keeps_geometric_prior(self):
        estimates = [TDOAEstimate(0, 1, 0.0, coherence=0.1, max_tau=0.02)]
        out = tdoa_edge_weights(self.EDGE_INDEX, estimates, self.STATIC)
        # Edges 2 and 3 connect 0<->2, for which no estimate was supplied.
        assert out[2] == pytest.approx(0.5)
        assert out[3] == pytest.approx(0.5)

    def test_gamma_sharpens_attenuation(self):
        estimates = [TDOAEstimate(0, 1, 0.0, coherence=0.5, max_tau=0.02)]
        soft = tdoa_edge_weights(self.EDGE_INDEX, estimates, self.STATIC, gamma=1.0)
        sharp = tdoa_edge_weights(self.EDGE_INDEX, estimates, self.STATIC, gamma=4.0)
        assert sharp[0] < soft[0]

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            tdoa_edge_weights(self.EDGE_INDEX, [], np.array([1.0]))

    def test_bad_edge_index_rejected(self):
        with pytest.raises(ValueError):
            tdoa_edge_weights(np.zeros((3, 4)), [], self.STATIC)
