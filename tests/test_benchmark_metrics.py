"""
Tests for the benchmark scoring metrics.

These matter more than most tests in the repository. Every claim Murmur makes
about its own accuracy is computed here, so a bug in this module does not cause
a crash — it silently produces a flattering number. The reference values below
were cross-checked against scikit-learn's ``roc_auc_score`` and
``average_precision_score`` on the same inputs.
"""

import pytest

from benchmarks.metrics import (
    average_precision,
    confusion_at_threshold,
    false_alarm_rate_per_hour,
    first_sustained_alarm,
    format_table,
    lead_time,
    partial_roc_auc,
    roc_auc,
    summarize,
)


class TestRocAuc:
    def test_perfect_separation(self):
        assert roc_auc([0.1, 0.2, 0.9, 0.95], [0, 0, 1, 1]) == pytest.approx(1.0)

    def test_perfectly_inverted(self):
        assert roc_auc([0.9, 0.95, 0.1, 0.2], [0, 0, 1, 1]) == pytest.approx(0.0)

    def test_all_scores_tied_is_chance(self):
        assert roc_auc([0.5] * 6, [0, 1, 0, 1, 0, 1]) == pytest.approx(0.5)

    def test_matches_sklearn_reference(self):
        # sklearn.metrics.roc_auc_score([0,0,1,1], [0.1,0.4,0.35,0.8]) == 0.75
        assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)

    def test_single_class_returns_chance_not_error(self):
        assert roc_auc([0.1, 0.2, 0.3], [0, 0, 0]) == pytest.approx(0.5)
        assert roc_auc([0.1, 0.2, 0.3], [1, 1, 1]) == pytest.approx(0.5)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            roc_auc([0.1, 0.2], [1])

    def test_partial_ties_use_average_ranks(self):
        # One positive and one negative share a score; the tie should land
        # exactly between "positive wins" (1.0) and "negative wins" (0.5).
        assert roc_auc([0.5, 0.5, 0.9, 0.1], [1, 0, 1, 0]) == pytest.approx(0.875)


class TestPartialRocAuc:
    def test_full_range_matches_roc_auc(self):
        scores = [0.1, 0.4, 0.35, 0.8, 0.62, 0.2]
        labels = [0, 0, 1, 1, 1, 0]
        assert partial_roc_auc(scores, labels, max_fpr=1.0) == pytest.approx(
            roc_auc(scores, labels)
        )

    def test_perfect_detector_scores_one_at_low_fpr(self):
        scores = [0.1, 0.2, 0.3, 0.9, 0.95, 0.99]
        labels = [0, 0, 0, 1, 1, 1]
        assert partial_roc_auc(scores, labels, max_fpr=0.1) == pytest.approx(1.0)

    def test_penalises_a_detector_that_only_wins_at_high_fpr(self):
        # Ranks every positive just below every negative for the first third of
        # the sweep: respectable full AUC, useless where it counts.
        scores = [0.9, 0.85, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
        labels = [0, 0, 1, 1, 1, 1, 0, 0]
        assert partial_roc_auc(scores, labels, max_fpr=0.1) < roc_auc(scores, labels)

    def test_invalid_max_fpr_rejected(self):
        with pytest.raises(ValueError):
            partial_roc_auc([0.1, 0.9], [0, 1], max_fpr=0.0)
        with pytest.raises(ValueError):
            partial_roc_auc([0.1, 0.9], [0, 1], max_fpr=1.5)


class TestAveragePrecision:
    def test_matches_sklearn_reference(self):
        # sklearn.metrics.average_precision_score([0,0,1,1], [0.1,0.4,0.35,0.8])
        assert average_precision([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(
            0.8333333, abs=1e-6
        )

    def test_perfect_ranking(self):
        assert average_precision([0.1, 0.2, 0.9, 0.95], [0, 0, 1, 1]) == pytest.approx(1.0)

    def test_no_positives_is_zero(self):
        assert average_precision([0.1, 0.5], [0, 0]) == pytest.approx(0.0)


class TestConfusion:
    def test_counts_and_derived_rates(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 0, 1, 0]
        m = confusion_at_threshold(scores, labels, threshold=0.5)
        assert (m.true_positives, m.false_positives) == (1, 1)
        assert (m.true_negatives, m.false_negatives) == (1, 1)
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)
        assert m.f1 == pytest.approx(0.5)
        assert m.specificity == pytest.approx(0.5)

    def test_empty_denominators_do_not_divide_by_zero(self):
        m = confusion_at_threshold([0.1, 0.2], [0, 0], threshold=0.5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0


class TestFalseAlarmRate:
    def test_counts_only_alarms_on_healthy_frames(self):
        # 4 healthy frames at 0.5 s each = 2 s of healthy runtime; 1 false alarm.
        predictions = [True, False, False, True, False]
        labels = [1, 0, 0, 0, 0]
        rate = false_alarm_rate_per_hour(predictions, labels, frame_interval_s=0.5)
        assert rate == pytest.approx(1 / (4 * 0.5 / 3600.0))

    def test_no_healthy_frames_is_zero(self):
        assert false_alarm_rate_per_hour([True], [1], 0.5) == 0.0

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            false_alarm_rate_per_hour([True], [0], 0.0)


class TestSustainedAlarm:
    def test_requires_a_run_of_consecutive_frames(self):
        predictions = [False, True, False, True, True, True, False]
        assert first_sustained_alarm(predictions, 0, 7, consecutive=3) == 3

    def test_isolated_blips_are_ignored(self):
        predictions = [True, False, True, False, True, False]
        assert first_sustained_alarm(predictions, 0, 6, consecutive=3) is None

    def test_search_is_bounded_by_the_window(self):
        predictions = [True, True, True, False, False]
        assert first_sustained_alarm(predictions, 3, 5, consecutive=3) is None

    def test_run_must_complete_inside_the_window(self):
        predictions = [False, False, True, True, True]
        # Window ends at 4, so only two of the three alarm frames are visible.
        assert first_sustained_alarm(predictions, 0, 4, consecutive=3) is None

    def test_rejects_zero_consecutive(self):
        with pytest.raises(ValueError):
            first_sustained_alarm([True], 0, 1, consecutive=0)


class TestLeadTime:
    def test_lead_and_delay_are_measured_from_the_sustained_alarm(self):
        # Onset 2, failure 10. Alarms sustain from index 5.
        predictions = [False] * 5 + [True] * 5
        result = lead_time(predictions, [(2, 10)], frame_interval_s=1.0, consecutive=3)
        assert result.detected == 1
        assert result.missed == 0
        assert result.lead_times_s == pytest.approx([5.0])
        assert result.detection_delays_s == pytest.approx([3.0])

    def test_event_with_no_alarm_is_missed(self):
        result = lead_time([False] * 10, [(2, 8)], frame_interval_s=1.0)
        assert result.missed == 1
        assert result.detected == 0
        assert result.detection_rate == 0.0
        assert result.mean_lead_time_s == 0.0

    def test_alarms_before_onset_are_not_credited(self):
        predictions = [True] * 4 + [False] * 8
        result = lead_time(predictions, [(6, 12)], frame_interval_s=1.0, consecutive=3)
        assert result.missed == 1

    def test_median_over_multiple_events(self):
        predictions = [False, True, True, True] * 3
        result = lead_time(
            predictions, [(0, 4), (4, 8), (8, 12)], frame_interval_s=2.0, consecutive=3
        )
        assert result.detected == 3
        assert result.median_lead_time_s == pytest.approx(6.0)

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            lead_time([True], [(0, 1)], frame_interval_s=0.0)


class TestSummarize:
    def test_produces_a_complete_report(self):
        scores = [0.1, 0.2, 0.8, 0.9, 0.85, 0.15]
        labels = [0, 0, 1, 1, 1, 0]
        predictions = [False, False, True, True, True, False]
        summary = summarize(scores, labels, predictions, [(2, 5)], 0.5, consecutive=3)

        for key in ("roc_auc", "pauc_10", "average_precision", "mean_lead_time_s"):
            assert key in summary
        assert summary["roc_auc"] == pytest.approx(1.0)
        assert summary["frames"] == 6
        assert summary["positive_rate"] == pytest.approx(0.5)

    def test_format_table_renders_counts_as_integers(self):
        summary = summarize([0.9, 0.1], [1, 0], [True, False], [], 0.5)
        rendered = format_table(summary, "T")
        assert "frames" in rendered
        # Counts must not print as 2.0000 — the table is read by humans.
        assert "2.0000" not in rendered
