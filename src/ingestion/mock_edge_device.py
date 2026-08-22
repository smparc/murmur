"""
Mock edge device that simulates real-time factory microphone streams.


Generates synthetic audio chunks with realistic degradation patterns:
    - Poisson-process anomaly injection (a mean-time-between-failure, not a
      flat per-tick coin flip that fires every few seconds)
    - Progressive degradation (gradual onset, not binary; gradual decay on
      recovery, not an instant reset)
    - No data leakage (is_anomalous_flag removed from payloads)
    - Multiple fault signatures, each tied to a plausible physical mechanism
      (bearing defect frequency, rotor imbalance at running speed, cavitation
      broadband spectrum) rather than an arbitrary hand-picked tone
"""

import logging
import time
from enum import Enum

import msgpack
import numpy as np
from confluent_kafka import Producer

from src.settings import settings

log = logging.getLogger(__name__)

# Nominal shaft rotation rate for the simulated motors: a 4-pole induction
# motor at 60 Hz mains turns near 1800 rpm, and slip under load settles a
# little below that — 1770 rpm is a textbook running speed. Fault signatures
# below are expressed relative to this rather than to arbitrary constants, so
# the bearing defect frequency and the imbalance modulation frequency sit in
# the same physically-consistent ratio a real spectrum analyzer would show.
_SHAFT_HZ = 1770.0 / 60.0  # ~29.5 Hz


class FaultType(Enum):
    NONE = "none"
    BEARING = "bearing"  # High-frequency squeal
    CAVITATION = "cavitation"  # Broadband noise burst
    IMBALANCE = "imbalance"  # Low-frequency modulation


def _delivery_report(err, msg):
    """Callback triggered on successful/failed message delivery."""
    if err is not None:
        log.error("Message delivery failed: %s", err)


def _pink_noise(n: int) -> np.ndarray:
    """
    Unit-variance 1/f-shaped noise of length ``n``.

    Real machine-room ambient noise is not flat white noise: energy is
    concentrated at low frequencies and decays with a long tail, which is why
    a factory floor sounds like a low rumble rather than static. Shaping a
    white-noise draw by 1/sqrt(f) in the frequency domain reproduces that
    without pulling in a filter-design dependency.

    Draws from the legacy ``np.random`` global state (not a local
    ``Generator``) so callers that seed ``np.random.seed(...)`` for
    reproducibility — see ``benchmarks/scenario.py`` — still get a
    deterministic result.
    """
    white = np.random.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if n > 1 else 1.0  # avoid a divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    std = pink.std()
    return pink / std if std > 1e-12 else pink


def poisson_tick_probability(dt: float, mtbf: float) -> float:
    """
    Per-tick probability of a memoryless (Poisson) arrival with mean ``mtbf``.

    A flat per-tick constant is not equivalent to a target MTBF: halving the
    chunk duration would silently halve the real-world fault rate too, unless
    the per-tick probability is derived from ``dt`` rather than picked once
    and left alone. ``1 - exp(-dt/mtbf)`` is the exact per-tick probability for
    a Poisson process observed in discrete steps of size ``dt``, so the
    long-run arrival rate stays ``1/mtbf`` regardless of chunk size.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if mtbf <= 0:
        raise ValueError(f"mtbf must be > 0, got {mtbf}")
    return 1.0 - np.exp(-dt / mtbf)


def _ambient_floor(t: np.ndarray, node_id: int) -> np.ndarray:
    """
    Structured factory ambient: mains hum plus shaped broadband noise.

    A single 60 Hz tone over white noise is spectrally unlike anything a
    microphone actually records near industrial equipment. Real floors carry
    the mains fundamental *and* its odd harmonics (motor slot noise,
    transformer buzz), on top of a 1/f-shaped broadband bed rather than a flat
    one. The per-node phase offset avoids every "microphone" emitting the
    identical waveform, which no two real sensors would.
    """
    phase = node_id * 0.7
    mains = (
        0.5 * np.sin(2 * np.pi * 60 * t + phase)
        + 0.15 * np.sin(2 * np.pi * 180 * t + phase)
        + 0.05 * np.sin(2 * np.pi * 300 * t + phase)
    )
    broadband = 0.1 * _pink_noise(t.size)
    return mains + broadband


def generate_mock_audio(
    node_id: int,
    fault: FaultType = FaultType.NONE,
    severity: float = 0.0,
) -> bytes:
    """
    Generate a raw audio waveform as float32 bytes.


    Args:
        node_id: Microphone identifier
        fault: Type of simulated machinery fault
        severity: Fault intensity [0.0 = none, 1.0 = severe]


    Returns raw bytes for efficient MessagePack transport.
    """
    t = np.linspace(0, settings.CHUNK_DURATION, settings.SAMPLES_PER_CHUNK, endpoint=False)

    base_noise = _ambient_floor(t, node_id)

    if fault == FaultType.BEARING and severity > 0:
        # A bearing defect does not ring continuously; it produces an impact
        # each time a rolling element crosses the defect, which excites a
        # high-frequency structural resonance that decays until the next
        # impact. The impact repetition rate — the "ball-pass frequency,
        # outer race" — is set by shaft speed and bearing geometry; ~3.5x
        # shaft speed is typical for a common deep-groove ball bearing, with
        # a little per-unit spread for bearing size/geometry.
        bpfo = 3.5 * _SHAFT_HZ + node_id * 0.4
        resonance = 2000 + node_id * 200

        # Sharpened half-sine lobes approximate the impact/ring-down train;
        # `pulse` is a periodic envelope, not a continuous carrier.
        pulse = np.clip(np.sin(2 * np.pi * bpfo * t), 0.0, None) ** 6
        pulse /= pulse.max() + 1e-12

        carrier = np.sin(2 * np.pi * resonance * t) + 0.4 * np.sin(2 * np.pi * resonance * 2 * t)
        base_noise += severity * 3.0 * pulse * carrier

    elif fault == FaultType.CAVITATION and severity > 0:
        # Cavitation is genuinely broadband — collapsing vapour bubbles excite
        # no particular resonance — but it is not flat either; energy skews
        # toward the mid/high band. Differencing pink noise is a cheap
        # highpass that produces that tilt without a filter-design dependency.
        shaped = np.diff(_pink_noise(t.size), prepend=0.0)
        shaped_std = shaped.std()
        if shaped_std > 1e-12:
            shaped /= shaped_std
        base_noise += severity * 0.5 * shaped

        # Individual bubble-collapse impacts on top of the broadband hiss.
        n_impulses = max(1, int(severity * 5))
        for _ in range(n_impulses):
            pos = np.random.randint(0, settings.SAMPLES_PER_CHUNK)
            width = min(50, settings.SAMPLES_PER_CHUNK - pos)
            base_noise[pos : pos + width] += severity * 1.5

    elif fault == FaultType.IMBALANCE and severity > 0:
        # The textbook rotating-imbalance signature is amplitude modulation at
        # 1x running speed — one heavy spot passing through its arc per
        # revolution — not an arbitrary low frequency. A real imbalance also
        # raises the 1x vibration tone directly, not only the modulation
        # depth of everything else, so a modest direct component rides
        # alongside the multiplicative envelope.
        mod_freq = _SHAFT_HZ + node_id * 0.3
        modulation = 1.0 + severity * 0.6 * np.sin(2 * np.pi * mod_freq * t)
        base_noise = base_noise * modulation + severity * 0.25 * np.sin(2 * np.pi * mod_freq * t)

    return base_noise.astype(np.float32).tobytes()


def run_edge_simulation(max_loops: int | None = None) -> int:
    """
    Stream continuous audio to Kafka from simulated microphone nodes.

    ``max_loops`` bounds the run so this is testable without a background
    thread. Returns the number of chunks streamed per node.
    """
    producer = Producer({"bootstrap.servers": settings.KAFKA_BROKER})
    num_nodes = settings.NUM_NODES

    log.info(
        "Mock Edge Device booting — %d nodes streaming to %s",
        num_nodes,
        settings.KAFKA_BROKER,
    )

    loop_count = 0

    # Stochastic degradation state per node
    node_degradation = dict.fromkeys(range(num_nodes), 0.0)
    node_fault_type = dict.fromkeys(range(num_nodes), FaultType.NONE)

    # Per-tick hazard of a *new* fault starting on an otherwise-healthy node,
    # derived from a mean-time-between-failure rather than picked to look
    # good on a demo. The previous flat 3%-per-tick constant implied an MTBF
    # of ~17 seconds, which starts a new episode on some node every few
    # seconds and reads as simulator noise rather than plant behaviour.
    anomaly_probability = poisson_tick_probability(
        settings.CHUNK_DURATION, settings.SIM_FAULT_MTBF_S
    )

    try:
        while max_loops is None or loop_count < max_loops:
            for node in range(num_nodes):
                # Stochastic anomaly injection (not deterministic)
                if node_degradation[node] == 0.0:
                    if np.random.random() < anomaly_probability:
                        # Start new degradation event
                        node_fault_type[node] = np.random.choice(
                            [FaultType.BEARING, FaultType.CAVITATION, FaultType.IMBALANCE]
                        )
                        node_degradation[node] = 0.1  # Start mild
                        log.debug(
                            "Node %d: %s degradation started", node, node_fault_type[node].value
                        )
                else:
                    # Progressive worsening (realistic ramp-up)
                    node_degradation[node] = min(
                        1.0, node_degradation[node] + np.random.uniform(0.01, 0.05)
                    )

                    # Chance of self-recovery for a still-mild issue. Real
                    # early-stage faults that resolve (a loose fitting
                    # re-seats, a transient load clears) fade out over several
                    # ticks; snapping straight to zero looks like a sensor
                    # dropout, not a mechanical condition clearing.
                    if (
                        node_degradation[node] < 0.3
                        and np.random.random() < settings.SIM_RECOVERY_PROBABILITY
                    ):
                        node_degradation[node] = max(
                            0.0, node_degradation[node] - np.random.uniform(0.05, 0.15)
                        )
                        if node_degradation[node] == 0.0:
                            node_fault_type[node] = FaultType.NONE

                audio_bytes = generate_mock_audio(
                    node, fault=node_fault_type[node], severity=node_degradation[node]
                )

                # Payload has NO anomaly label — the system must detect it
                payload = msgpack.packb(
                    {
                        "node_id": node,
                        "timestamp": time.time(),
                        "audio": audio_bytes,
                    },
                    use_bin_type=True,
                )

                producer.produce(
                    settings.RAW_TOPIC,
                    key=str(node).encode("utf-8"),
                    value=payload,
                    callback=_delivery_report,
                )

            producer.poll(0)
            loop_count += 1
            time.sleep(settings.CHUNK_DURATION)

            if loop_count % 10 == 0:
                log.info("Streamed chunk %d from %d nodes", loop_count, num_nodes)

    except KeyboardInterrupt:
        log.info("Stopping edge simulation.")
    finally:
        producer.flush(timeout=10)

    return loop_count


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_edge_simulation()


if __name__ == "__main__":  # pragma: no cover
    main()
