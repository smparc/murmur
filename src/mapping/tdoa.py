"""
Time-Difference-of-Arrival (TDOA) estimation via GCC-PHAT.

The graph the ST-GNN convolves over has, until now, been *static*: edges and
their weights derive purely from how far apart the microphones are bolted. That
encodes the room, but nothing about the sound currently in it. Two microphones
either side of a failing pump and two either side of a silent one carry
identical edge weights.

Multi-channel audio contains the missing signal. The delay between a source
reaching microphone *i* and microphone *j* localises that source onto a
hyperboloid; intersecting the hyperboloids from several pairs fixes it in space.
That delay is recoverable from the cross-correlation of the two channels.

Plain cross-correlation is dominated by whichever frequencies happen to be
loudest, which in a factory is the 50/60 Hz mains rumble shared by every
channel — the correlation peaks at zero lag regardless of where the machine is.
GCC-PHAT divides out the magnitude spectrum and keeps only phase, so every
frequency contributes equally to locating the peak. It is the standard estimator
for reverberant, broadband, poorly-conditioned environments, which describes a
factory floor exactly.

References
----------
Knapp & Carter (1976), "The Generalized Correlation Method for Estimation of
Time Delay", IEEE TASSP 24(4).
Chan & Ho (1994), "A Simple and Efficient Estimator for Hyperbolic Location",
IEEE Trans. Signal Processing 42(8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Speed of sound in dry air at 20 degC. Varies ~0.6 m/s per degC, which over a
# 5 m array is well under the sampling resolution, so a constant is fine.
SPEED_OF_SOUND = 343.0

_EPSILON = 1e-12


@dataclass(frozen=True)
class TDOAEstimate:
    """One microphone pair's delay estimate."""

    i: int
    j: int
    tau: float
    """Delay in seconds. Positive means the source reached ``j`` before ``i``."""
    coherence: float
    """Sharpness of the correlation peak in [0, 1]. See :func:`gcc_phat`."""
    max_tau: float
    """Physical bound imposed by the pair's separation, in seconds."""

    @property
    def saturated(self) -> bool:
        """
        True when the estimate sits on its physical bound.

        A delay pinned at the maximum usually means there was no common source
        to find and the search returned an edge of the window, not that the
        source is exactly on the array axis.
        """
        return abs(abs(self.tau) - self.max_tau) < _EPSILON * 10


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    interp: int = 8,
) -> tuple[float, float]:
    """
    Estimate the delay between two channels by phase-transform correlation.

    Parameters
    ----------
    sig, refsig:
        Single-channel waveforms. Need not be the same length.
    fs:
        Sample rate in Hz.
    max_tau:
        Largest plausible delay in seconds. Supply ``separation / SPEED_OF_SOUND``
        so the search cannot return a physically impossible lag — without it,
        uncorrelated noise routinely peaks at an arbitrary offset.
    interp:
        Frequency-domain zero-padding factor. The raw correlation resolves to
        one sample (62.5 us at 16 kHz, ~2 cm of path difference); interpolating
        recovers sub-sample resolution, which matters because a 5 m array spans
        only ~230 samples of delay in total.

    Returns
    -------
    ``(tau, coherence)``. ``tau`` is in seconds, positive when ``refsig`` leads.
    ``coherence`` is the height of the normalised correlation peak: 1.0 for a
    clean delayed copy, ~``1/sqrt(n)`` for uncorrelated noise. It is the natural
    confidence measure here, because PHAT weighting makes the peak of a
    perfectly correlated pair exactly unity regardless of signal level.
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    refsig = np.asarray(refsig, dtype=np.float64).ravel()
    if sig.size == 0 or refsig.size == 0:
        raise ValueError("gcc_phat needs two non-empty signals")
    if interp < 1:
        raise ValueError(f"interp must be >= 1, got {interp}")

    # Linear (not circular) correlation requires padding to the summed length.
    n = sig.size + refsig.size

    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    cross = SIG * np.conj(REF)

    # The phase transform: discard magnitude, keep only phase. This is what
    # stops a single dominant frequency from owning the correlation.
    magnitude = np.abs(cross)
    np.maximum(magnitude, _EPSILON, out=magnitude)
    cross /= magnitude

    padded = n * interp
    cc = np.fft.irfft(cross, n=padded)
    # irfft normalises by its output length, so zero-padding the spectrum shrinks
    # the peak by exactly `interp`. Undo that to keep coherence on a scale where
    # a perfect delayed copy reads 1.0.
    cc *= interp

    if max_tau is not None:
        if max_tau < 0:
            raise ValueError(f"max_tau must be >= 0, got {max_tau}")
        max_shift = min(int(fs * interp * max_tau), padded // 2)
    else:
        max_shift = padded // 2

    # Rotate negative lags to the front: [-max_shift .. +max_shift].
    if max_shift == 0:
        return 0.0, float(np.clip(abs(cc[0]), 0.0, 1.0))
    window = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])

    peak = int(np.argmax(np.abs(window)))
    tau = (peak - max_shift) / float(fs * interp)
    coherence = float(np.clip(abs(window[peak]), 0.0, 1.0))
    return tau, coherence


def max_delay_between(p_i: np.ndarray, p_j: np.ndarray, speed: float = SPEED_OF_SOUND) -> float:
    """Largest delay physically achievable between two positions, in seconds."""
    return float(np.linalg.norm(np.asarray(p_i, float) - np.asarray(p_j, float)) / speed)


def pairwise_tdoa(
    channels: np.ndarray,
    mic_coords: np.ndarray | list[tuple[float, float, float]],
    fs: int,
    pairs: list[tuple[int, int]] | None = None,
    interp: int = 8,
    speed: float = SPEED_OF_SOUND,
) -> list[TDOAEstimate]:
    """
    Estimate delays for every microphone pair (or a given subset).

    Parameters
    ----------
    channels:
        ``(num_mics, num_samples)`` time-aligned waveforms. Alignment is the
        caller's problem and it is a real one — see :func:`pairwise_tdoa` usage
        in the worker, where frames are matched by producer timestamp.
    mic_coords:
        ``(num_mics, 3)`` positions in metres, used only to bound each search.
    pairs:
        Defaults to all unordered pairs. Pass the graph's edge list to avoid
        computing delays for pairs the topology does not connect.
    """
    channels = np.asarray(channels, dtype=np.float64)
    if channels.ndim != 2:
        raise ValueError(f"channels must be (num_mics, num_samples), got {channels.shape}")

    coords = np.asarray(mic_coords, dtype=np.float64)
    num_mics = channels.shape[0]
    if coords.shape[0] != num_mics:
        raise ValueError(f"got {num_mics} channels but {coords.shape[0]} microphone positions")

    if pairs is None:
        pairs = [(i, j) for i in range(num_mics) for j in range(i + 1, num_mics)]

    estimates: list[TDOAEstimate] = []
    for i, j in pairs:
        bound = max_delay_between(coords[i], coords[j], speed)
        tau, coherence = gcc_phat(channels[i], channels[j], fs, max_tau=bound, interp=interp)
        estimates.append(TDOAEstimate(i=i, j=j, tau=tau, coherence=coherence, max_tau=bound))
    return estimates


def localize_source(
    estimates: list[TDOAEstimate],
    mic_coords: np.ndarray | list[tuple[float, float, float]],
    speed: float = SPEED_OF_SOUND,
    plane_z: float | None = None,
    min_coherence: float = 0.05,
) -> tuple[np.ndarray | None, float]:
    """
    Solve for the source position from a set of pairwise delays.

    Uses the linear (Chan-Ho) formulation. With ``r_i`` the range from the
    source to microphone *i* and ``d_i = c * tau_i0 = r_i - r_0``::

        2 (p_i - p_0) . s  +  2 d_i r_0  =  |p_i|^2 - |p_0|^2 - d_i^2

    which is linear in the unknowns ``[s, r_0]`` and solvable by least squares.

    Geometry caveat
    ---------------
    Three-dimensional position needs four independent equations, i.e. five
    microphones. The default four-microphone array is also *coplanar*, so its
    vertical component is unobservable in principle — a source above the plane
    and its mirror image below produce identical delays. ``plane_z`` therefore
    constrains the solution to a horizontal plane by default, reducing the
    problem to three unknowns that four microphones determine exactly. This is a
    real limitation of the array, not of the estimator; it is stated here rather
    than papered over with a plausible-looking but arbitrary z.

    Returns
    -------
    ``(position, residual)``. ``position`` is ``(3,)`` in metres, or ``None``
    when too few usable delays survive the coherence filter. ``residual`` is the
    RMS of the linear system, useful as a goodness-of-fit gate.
    """
    coords = np.asarray(mic_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"mic_coords must be (N, 3), got {coords.shape}")

    usable = [e for e in estimates if e.coherence >= min_coherence and not e.saturated]
    if not usable:
        return None, float("inf")

    # Re-reference every delay to a common anchor: the microphone appearing in
    # the most usable pairs, which is the best-observed node in this snapshot.
    counts = np.zeros(coords.shape[0], dtype=int)
    for e in usable:
        counts[e.i] += 1
        counts[e.j] += 1
    anchor = int(np.argmax(counts))

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    p0 = coords[anchor]
    solve_z = plane_z is None

    for e in usable:
        # tau is defined as (arrival at i) - (arrival at j); convert to a delay
        # relative to the anchor.
        if e.i == anchor:
            other, tau = e.j, -e.tau
        elif e.j == anchor:
            other, tau = e.i, e.tau
        else:
            continue

        d = speed * tau  # r_other - r_anchor
        p_i = coords[other]
        diff = p_i - p0

        if solve_z:
            row = np.array([2 * diff[0], 2 * diff[1], 2 * diff[2], 2 * d])
            const = float(p_i @ p_i - p0 @ p0 - d * d)
        else:
            # z is pinned; fold its contribution into the constant term.
            row = np.array([2 * diff[0], 2 * diff[1], 2 * d])
            const = float(p_i @ p_i - p0 @ p0 - d * d - 2 * diff[2] * plane_z)
        rows.append(row)
        rhs.append(const)

    position_dims = 3 if solve_z else 2
    unknowns = position_dims + 1  # + the anchor range r_0
    if len(rows) < unknowns:
        return None, float("inf")

    A = np.vstack(rows)
    b = np.asarray(rhs)

    # Rank is checked on the *position* columns only, not the whole system.
    # A source equidistant from every microphone — the centre of the array, and
    # a thoroughly ordinary place for a machine to sit — makes every delay zero,
    # which zeroes the r_0 column and drops the full matrix to rank-deficient.
    # The position is still perfectly determined there (it is the circumcentre);
    # only the range to the anchor becomes unidentifiable, and that is a
    # quantity this function does not return. Rejecting the whole solve on that
    # basis would blind the array precisely at its centre.
    if np.linalg.matrix_rank(A[:, :position_dims]) < position_dims:
        # Genuinely degenerate: a collinear array, or a coplanar one asked for
        # a 3-D fix.
        return None, float("inf")

    # lstsq returns the minimum-norm solution, which resolves the position block
    # correctly even when the r_0 column carries no information.
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    residual = float(np.sqrt(np.mean((A @ solution - b) ** 2)))
    position = solution[:3] if solve_z else np.array([solution[0], solution[1], float(plane_z)])
    return position, residual


def tdoa_edge_weights(
    edge_index: np.ndarray,
    estimates: list[TDOAEstimate],
    static_weight: np.ndarray,
    gamma: float = 1.0,
    floor: float = 0.05,
) -> np.ndarray:
    """
    Modulate static coupling weights by measured acoustic coherence.

    The static weight says two microphones *could* hear the same machine. The
    coherence says whether they currently *do*. Multiplying them gives the GCN
    a topology that tracks the acoustic scene instead of only the floorplan:
    edges across a pair that has fallen out of correlation stop propagating,
    and the graph effectively re-partitions itself around active sources.

    Parameters
    ----------
    edge_index:
        ``(2, E)`` directed edge list, as produced by ``build_acoustic_topology``.
    estimates:
        Pairwise delays. Order-independent; both directions of an edge receive
        the same coherence, since coherence is symmetric.
    static_weight:
        ``(E,)`` geometric weights to modulate.
    gamma:
        Sharpness. Higher values prune weakly-correlated edges more aggressively.
    floor:
        Minimum retained fraction. A graph whose edges all drop to zero has no
        propagation at all and the GCN degenerates to a per-node MLP, so edges
        are attenuated rather than severed.

    Returns
    -------
    ``(E,)`` effective weights.
    """
    static_weight = np.asarray(static_weight, dtype=np.float64)
    edge_index = np.asarray(edge_index)
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, E), got {edge_index.shape}")
    if edge_index.shape[1] != static_weight.shape[0]:
        raise ValueError(
            f"edge_index has {edge_index.shape[1]} edges but "
            f"{static_weight.shape[0]} weights were supplied"
        )
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")

    lookup: dict[tuple[int, int], float] = {}
    for e in estimates:
        lookup[(e.i, e.j)] = e.coherence
        lookup[(e.j, e.i)] = e.coherence

    gains = np.empty(edge_index.shape[1], dtype=np.float64)
    for k in range(edge_index.shape[1]):
        coherence = lookup.get((int(edge_index[0, k]), int(edge_index[1, k])))
        # An edge with no measurement keeps its geometric prior untouched;
        # absence of evidence is not evidence of decoupling.
        gains[k] = 1.0 if coherence is None else floor + (1.0 - floor) * coherence**gamma

    return static_weight * gains


def main() -> None:  # pragma: no cover - manual inspection helper
    from src.settings import settings

    rng = np.random.default_rng(0)
    coords = np.asarray(settings.MIC_COORDS, dtype=float)
    fs = settings.SAMPLE_RATE

    # Put a broadband source in the room and synthesise what each mic hears.
    source = np.array([3.5, 7.0, 3.0])
    n = settings.SAMPLES_PER_CHUNK
    signal = rng.standard_normal(n + 512)

    channels = []
    for p in coords:
        delay_samples = round(np.linalg.norm(source - p) / SPEED_OF_SOUND * fs)
        channels.append(signal[delay_samples : delay_samples + n])
    channels = np.stack(channels)

    estimates = pairwise_tdoa(channels, coords, fs)
    position, residual = localize_source(estimates, coords, plane_z=float(coords[:, 2].mean()))

    print(f"[*] True source:      {source}")
    print(f"[*] Estimated source: {position}")
    print(f"[*] Residual:         {residual:.4f}")
    for e in estimates:
        print(f"    mic {e.i}-{e.j}: tau={e.tau * 1e3:+.3f} ms  coherence={e.coherence:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
