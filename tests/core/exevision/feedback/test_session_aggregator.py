"""Tests for SessionAggregator session-level summary logic."""

import pytest

from core.exevision.feedback.session_aggregator import SessionAggregator


class TestTrajectoryDetection:
    def test_improving_trajectory(self):
        aggregator = SessionAggregator()
        trajectory = aggregator.detect_trajectory([72, 75, 80])
        assert trajectory == "improving"

    def test_declining_trajectory(self):
        aggregator = SessionAggregator()
        trajectory = aggregator.detect_trajectory([80, 75, 70])
        assert trajectory == "declining"

    def test_stable_trajectory(self):
        aggregator = SessionAggregator()
        trajectory = aggregator.detect_trajectory([75, 76, 77])
        assert trajectory == "stable"


class TestMostImprovedMetric:
    def test_most_improved_metric(self):
        aggregator = SessionAggregator()
        rep_metrics = [
            {"depth": 65, "lean": 70, "valgus": 80},
            {"depth": 68, "lean": 72, "valgus": 79},
            {"depth": 75, "lean": 75, "valgus": 80},
        ]

        most_improved = aggregator.find_most_improved_metric(rep_metrics)

        assert most_improved == "depth"


class TestPersistentIssue:
    def test_persistent_issue(self):
        aggregator = SessionAggregator()
        threshold = 75

        rep_metrics = [
            {"depth": 70, "lean": 75, "valgus": 80},
            {"depth": 72, "lean": 76, "valgus": 82},
            {"depth": 78, "lean": 74, "valgus": 83},
        ]

        persistent = aggregator.find_persistent_issue(rep_metrics, threshold)

        assert persistent == "depth"


class TestAverageScore:
    def test_average_score(self):
        aggregator = SessionAggregator()
        avg = aggregator.compute_average_score([72, 78, 85])
        assert avg == pytest.approx(78.33, abs=0.1)
