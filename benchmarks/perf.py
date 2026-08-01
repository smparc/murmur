"""
Performance benchmarks for the claims in the top-level README.

    python -m benchmarks.perf
    python -m benchmarks.perf --iterations 500 --json perf.json

The README asserts that MessagePack is "10-50x faster than JSON for spectrogram
payloads" and that GPU preprocessing "bypasses CPU bottlenecks". Both are
plausible and neither was measured. This module measures them on whatever
hardware it is run on and prints a table, so the numbers in the README can carry
a machine and a date rather than an adjective.

Every benchmark discards a warm-up round before timing. The first call through
any of these paths pays for lazy imports, filter-bank construction and, on CUDA,
context creation — timing that once and reporting it as steady-state throughput
would understate every path by a wide margin.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.settings import settings

log = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Latency distribution for one operation."""

    name: str
    samples_ms: list[float] = field(default_factory=list)
    payload_bytes: int = 0

    def _percentile(self, pct: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        index = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
        return ordered[index]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms) if self.samples_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def ops_per_second(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "name": self.name,
            "mean_ms": round(self.mean_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "ops_per_second": round(self.ops_per_second, 1),
            "payload_bytes": self.payload_bytes,
        }


def time_it(name: str, fn, iterations: int, warmup: int = 5) -> TimingResult:
    """Run ``fn`` repeatedly and collect a latency distribution."""
    for _ in range(warmup):
        fn()

    result = TimingResult(name=name)
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        result.samples_ms.append((time.perf_counter() - started) * 1000.0)
    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def benchmark_serialization(iterations: int = 200) -> dict[str, TimingResult]:
    """
    MessagePack vs JSON on a realistic spectrogram payload.

    The comparison is deliberately like-for-like on *content*, not on encoding
    strategy: JSON cannot carry binary, so a float array has to be sent as a list
    of decimal numbers. That asymmetry is the whole reason MessagePack wins here,
    and hiding it by base64-encoding the buffer for JSON would measure something
    nobody would actually deploy.
    """
    import msgpack

    rng = np.random.default_rng(0)
    spectrogram = rng.normal(0, 1, (settings.N_MELS, 64)).astype(np.float32)

    msgpack_payload = {
        "node_id": 3,
        "timestamp": time.time(),
        "shape": list(spectrogram.shape),
        "data": spectrogram.tobytes(),
    }
    json_payload = {
        "node_id": 3,
        "timestamp": time.time(),
        "shape": list(spectrogram.shape),
        "data": spectrogram.flatten().tolist(),
    }

    packed = msgpack.packb(msgpack_payload, use_bin_type=True)
    encoded = json.dumps(json_payload).encode("utf-8")

    results = {
        "msgpack_encode": time_it(
            "msgpack encode",
            lambda: msgpack.packb(msgpack_payload, use_bin_type=True),
            iterations,
        ),
        "msgpack_decode": time_it(
            "msgpack decode", lambda: msgpack.unpackb(packed, raw=False), iterations
        ),
        "json_encode": time_it(
            "json encode", lambda: json.dumps(json_payload).encode("utf-8"), iterations
        ),
        "json_decode": time_it(
            "json decode", lambda: json.loads(encoded.decode("utf-8")), iterations
        ),
    }
    results["msgpack_encode"].payload_bytes = len(packed)
    results["msgpack_decode"].payload_bytes = len(packed)
    results["json_encode"].payload_bytes = len(encoded)
    results["json_decode"].payload_bytes = len(encoded)
    return results


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def benchmark_preprocessing(iterations: int = 100) -> dict[str, TimingResult]:
    """Mel-spectrogram extraction on every available device."""
    from benchmarks.features import mel_transform, to_log_mel

    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.2, settings.SAMPLES_PER_CHUNK).astype(np.float32)

    results: dict[str, TimingResult] = {}

    cpu_transform = mel_transform("cpu")
    results["preprocess_cpu"] = time_it(
        "mel spectrogram (cpu)", lambda: to_log_mel(audio, cpu_transform, "cpu"), iterations
    )

    if torch.cuda.is_available():
        cuda_transform = mel_transform("cuda")
        tensor = torch.from_numpy(audio).cuda()

        def run_cuda():
            to_log_mel(tensor, cuda_transform, "cuda")
            # Without this the timer measures kernel *launch*, not execution.
            torch.cuda.synchronize()

        results["preprocess_cuda"] = time_it(
            "mel spectrogram (cuda)", run_cuda, iterations
        )

    return results


# ---------------------------------------------------------------------------
# Detection and localization
# ---------------------------------------------------------------------------


def benchmark_detection(iterations: int = 100) -> dict[str, TimingResult]:
    """Autoencoder scoring, and the per-frame scorer path around it."""
    from benchmarks.features import mel_transform, to_log_mel
    from src.detection.anomaly_detector import AnomalyScorer, SpectrogramAutoencoder

    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.2, settings.SAMPLES_PER_CHUNK).astype(np.float32)
    transform = mel_transform("cpu")
    spectrogram = to_log_mel(audio, transform, "cpu")

    autoencoder = SpectrogramAutoencoder(n_mels=settings.N_MELS).eval()
    batch = spectrogram.unsqueeze(0).unsqueeze(0)

    scorer = AnomalyScorer(autoencoder=autoencoder, num_nodes=4, warmup_frames=1)

    return {
        "autoencoder_forward": time_it(
            "autoencoder score", lambda: autoencoder.anomaly_score(batch), iterations
        ),
        "end_to_end_frame": time_it(
            "end-to-end frame (mel + score)",
            lambda: scorer.score(0, to_log_mel(audio, transform, "cpu")),
            iterations,
        ),
    }


def benchmark_localization(iterations: int = 50) -> dict[str, TimingResult]:
    """GCC-PHAT plus multilateration for a four-microphone array."""
    from src.localization import locate_source

    rng = np.random.default_rng(0)
    mics = settings.DEFAULT_MIC_COORDS
    signals = {
        i: rng.normal(0, 0.2, settings.SAMPLES_PER_CHUNK).astype(np.float64)
        for i in range(len(mics))
    }

    return {
        "localization": time_it(
            f"localization ({len(mics)} mics)",
            lambda: locate_source(signals, mics, settings.SAMPLE_RATE),
            iterations,
        )
    }


def benchmark_throughput_by_nodes(
    node_counts: tuple[int, ...] = (1, 2, 4, 8, 16), iterations: int = 20
) -> dict[str, TimingResult]:
    """How per-frame cost scales as microphones are added."""
    from benchmarks.features import mel_transform, to_log_mel
    from src.detection.anomaly_detector import AnomalyScorer, SpectrogramAutoencoder

    rng = np.random.default_rng(0)
    transform = mel_transform("cpu")
    autoencoder = SpectrogramAutoencoder(n_mels=settings.N_MELS).eval()

    results: dict[str, TimingResult] = {}
    for count in node_counts:
        audio = [
            rng.normal(0, 0.2, settings.SAMPLES_PER_CHUNK).astype(np.float32)
            for _ in range(count)
        ]
        scorer = AnomalyScorer(autoencoder=autoencoder, num_nodes=count, warmup_frames=1)

        def sweep(_audio=audio, _scorer=scorer):
            for node_id, chunk in enumerate(_audio):
                _scorer.score(node_id, to_log_mel(chunk, transform, "cpu"))

        results[f"nodes_{count}"] = time_it(f"{count:2d} nodes / frame", sweep, iterations)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _environment() -> dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_all(iterations: int = 200) -> dict:
    """Run every benchmark and return a JSON-serialisable report."""
    log.info("Environment: %s", _environment())

    serialization = benchmark_serialization(iterations)
    preprocessing = benchmark_preprocessing(max(20, iterations // 2))
    detection = benchmark_detection(max(20, iterations // 2))
    localization = benchmark_localization(max(10, iterations // 10))
    scaling = benchmark_throughput_by_nodes(iterations=max(10, iterations // 20))

    encode_speedup = (
        serialization["json_encode"].mean_ms / serialization["msgpack_encode"].mean_ms
        if serialization["msgpack_encode"].mean_ms > 0
        else 0.0
    )
    decode_speedup = (
        serialization["json_decode"].mean_ms / serialization["msgpack_decode"].mean_ms
        if serialization["msgpack_decode"].mean_ms > 0
        else 0.0
    )
    size_ratio = (
        serialization["json_encode"].payload_bytes
        / serialization["msgpack_encode"].payload_bytes
        if serialization["msgpack_encode"].payload_bytes
        else 0.0
    )

    report = {
        "environment": _environment(),
        "iterations": iterations,
        "serialization": {k: v.as_dict() for k, v in serialization.items()},
        "preprocessing": {k: v.as_dict() for k, v in preprocessing.items()},
        "detection": {k: v.as_dict() for k, v in detection.items()},
        "localization": {k: v.as_dict() for k, v in localization.items()},
        "scaling": {k: v.as_dict() for k, v in scaling.items()},
        "headline": {
            "msgpack_encode_speedup": round(encode_speedup, 1),
            "msgpack_decode_speedup": round(decode_speedup, 1),
            "payload_size_ratio": round(size_ratio, 1),
            "msgpack_bytes": serialization["msgpack_encode"].payload_bytes,
            "json_bytes": serialization["json_encode"].payload_bytes,
        },
    }
    return report


def format_report(report: dict) -> str:
    """Human-readable rendering of :func:`run_all`."""
    lines = ["", "Murmur performance benchmarks", "=" * 60]
    for key, value in report["environment"].items():
        lines.append(f"  {key:<16} {value}")

    for section, title in (
        ("serialization", "Serialization"),
        ("preprocessing", "Preprocessing"),
        ("detection", "Detection"),
        ("localization", "Localization"),
        ("scaling", "Scaling by node count"),
    ):
        rows = report[section]
        if not rows:
            continue
        lines += [
            "",
            title,
            "-" * len(title),
            f"{'operation':<34}{'mean':>11}{'p50':>11}{'p95':>11}{'ops/s':>12}",
        ]
        for row in rows.values():
            lines.append(
                f"{row['name']:<34}"
                f"{row['mean_ms']:>9.3f}ms"
                f"{row['p50_ms']:>9.3f}ms"
                f"{row['p95_ms']:>9.3f}ms"
                f"{row['ops_per_second']:>12,.0f}"
            )

    headline = report["headline"]
    lines += [
        "",
        "Headline",
        "-" * 8,
        f"  MessagePack encode is {headline['msgpack_encode_speedup']}x faster than JSON",
        f"  MessagePack decode is {headline['msgpack_decode_speedup']}x faster than JSON",
        f"  Payload is {headline['payload_size_ratio']}x smaller "
        f"({headline['msgpack_bytes']:,} vs {headline['json_bytes']:,} bytes)",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Murmur performance benchmarks")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_all(args.iterations)
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        log.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
