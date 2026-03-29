"""Tests for RepComparator rep-over-rep progress logic."""

import pytest

from core.exevision.feedback.rep_comparator import RepComparator


class TestMetricImprovement:
    def test_improvement_percentage(self):
        comparator = RepComparator()

        pct = comparator.compute_improvement_percentage(prev_score=62, curr_score=68)

        assert abs(pct - 9.7) < 0.2

    def test_improvement_zero(self):
        comparator = RepComparator()

        pct = comparator.compute_improvement_percentage(prev_score=75, curr_score=75)

        assert pct == 0.0

    def test_regression_negative(self):
        comparator = RepComparator()

        pct = comparator.compute_improvement_percentage(prev_score=78, curr_score=70)

        assert pct < 0


class TestImprovementTier:
    def test_significant_improvement(self):
        comparator = RepComparator()
        tier = comparator.get_improvement_tier(improvement_pct=18.0)
        assert tier == "significant"

    def test_moderate_improvement(self):
        comparator = RepComparator()
        tier = comparator.get_improvement_tier(improvement_pct=10.5)
        assert tier == "moderate"

    def test_slight_improvement(self):
        comparator = RepComparator()
        tier = comparator.get_improvement_tier(improvement_pct=5.0)
        assert tier == "slight"

    def test_no_improvement(self):
        comparator = RepComparator()
        assert comparator.get_improvement_tier(improvement_pct=0.0) == "no_improvement"
        assert comparator.get_improvement_tier(improvement_pct=-5.0) == "no_improvement"


class TestRepComparison:
    def test_compare_reps_first_rep(self):
        comparator = RepComparator()
        curr_rep = {
            "rep_id": 1,
            "neural_score": 75,
            "sub_scores": {"depth": 72},
        }

        result = comparator.compare_reps(None, curr_rep)

        assert result is None

    def test_compare_reps_second_rep(self):
        comparator = RepComparator()
        prev_rep = {
            "rep_id": 1,
            "neural_score": 72,
            "sub_scores": {"depth": 65, "lean": 70},
        }
        curr_rep = {
            "rep_id": 2,
            "neural_score": 78,
            "sub_scores": {"depth": 72, "lean": 78},
        }

        result = comparator.compare_reps(prev_rep, curr_rep)

        assert result is not None
        assert result["overall_improvement_pct"] == pytest.approx((78 - 72) / 72 * 100, abs=0.5)
        assert result["metric_improvements"]["depth"] == pytest.approx((72 - 65) / 65 * 100, abs=0.5)
        assert result["metric_improvements"]["lean"] == pytest.approx((78 - 70) / 70 * 100, abs=0.5)
