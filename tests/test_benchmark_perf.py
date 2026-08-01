"""
Tests for the performance benchmark harness.

Deliberately no assertions on wall-clock durations. A timing threshold that
passes on a workstation fails on a loaded CI runner, and the usual response —
loosening the bound until it stops flaking — leaves a test that asserts nothing.
What is checked here is that the harness measures the right things and reports
them in the right shape; the numbers themselves are for humans to read.
"""

import pytest

from benchmarks.perf import (
    TimingResult,
    benchmark_detection,
    benchmark_localization,
    benchmark_preprocessing,
    benchmark_serialization,
    benchmark_throughput_by_nodes,
    format_report,
    run_all,
    time_it,
)


class TestTimingResult:
    def test_percentiles_are_ordered(self):
        result = TimingResult("t", samples_ms=[float(i) for i in range(1, 101)])
        assert result.p50_ms <= result.p95_ms <= result.p99_ms

    def test_empty_result_does_not_divide_by_zero(self):
        result = TimingResult("t")
        assert result.mean_ms == 0.0
        assert result.p50_ms == 0.0
        assert result.ops_per_second == 0.0

    def test_ops_per_second_inverts_the_mean(self):
        result = TimingResult("t", samples_ms=[2.0, 2.0, 2.0])
        assert result.ops_per_second == pytest.approx(500.0)

    def test_serialises_every_field(self):
        result = TimingResult("t", samples_ms=[1.0, 2.0], payload_bytes=64)
        payload = result.as_dict()
        assert set(payload) == {
            "name",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "ops_per_second",
            "payload_bytes",
        }


class TestTimeIt:
    def test_collects_one_sample_per_iteration(self):
        calls = []
        result = time_it("noop", lambda: calls.append(1), iterations=7, warmup=3)
        assert len(result.samples_ms) == 7
        # Warm-up rounds run but are not timed.
        assert len(calls) == 10

    def test_samples_are_non_negative(self):
        result = time_it("noop", lambda: None, iterations=5, warmup=1)
        assert all(s >= 0 for s in result.samples_ms)


class TestSerializationBenchmark:
    def test_measures_both_codecs_in_both_directions(self):
        results = benchmark_serialization(iterations=5)
        assert set(results) == {
            "msgpack_encode",
            "msgpack_decode",
            "json_encode",
            "json_decode",
        }

    def test_records_payload_sizes(self):
        results = benchmark_serialization(iterations=5)
        assert results["msgpack_encode"].payload_bytes > 0
        assert results["json_encode"].payload_bytes > 0

    def test_msgpack_payload_is_smaller_than_json(self):
        # Structural, not a timing claim: JSON has to render every float as
        # decimal text, which cannot be smaller than the raw buffer.
        results = benchmark_serialization(iterations=5)
        assert (
            results["msgpack_encode"].payload_bytes < results["json_encode"].payload_bytes
        )


class TestOtherBenchmarks:
    def test_preprocessing_always_measures_cpu(self):
        results = benchmark_preprocessing(iterations=3)
        assert "preprocess_cpu" in results
        assert results["preprocess_cpu"].samples_ms

    def test_detection_measures_the_full_frame_path(self):
        results = benchmark_detection(iterations=3)
        assert set(results) == {"autoencoder_forward", "end_to_end_frame"}

    def test_localization_is_measured(self):
        results = benchmark_localization(iterations=2)
        assert results["localization"].samples_ms

    def test_scaling_covers_each_requested_node_count(self):
        results = benchmark_throughput_by_nodes(node_counts=(1, 2), iterations=2)
        assert set(results) == {"nodes_1", "nodes_2"}


class TestReport:
    def test_run_all_produces_every_section(self):
        report = run_all(iterations=2)
        for section in (
            "environment",
            "serialization",
            "preprocessing",
            "detection",
            "localization",
            "scaling",
            "headline",
        ):
            assert section in report

    def test_headline_reports_speedup_and_size_ratio(self):
        report = run_all(iterations=2)
        headline = report["headline"]
        assert headline["msgpack_encode_speedup"] > 0
        assert headline["payload_size_ratio"] > 1.0

    def test_report_renders_without_error(self):
        rendered = format_report(run_all(iterations=2))
        assert "Murmur performance benchmarks" in rendered
        assert "MessagePack encode is" in rendered

    def test_report_is_json_serialisable(self):
        import json

        json.dumps(run_all(iterations=2))
