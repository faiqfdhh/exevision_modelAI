"""Session aggregation helpers for session-level coaching summary."""

from __future__ import annotations


class SessionAggregator:
    """Builds aggregate metrics and trajectory across all reps."""

    @staticmethod
    def detect_trajectory(rep_scores: list[float]) -> str:
        """Classify trajectory based on first-to-last score difference."""
        if len(rep_scores) < 2:
            return "stable"

        delta = rep_scores[-1] - rep_scores[0]
        if delta >= 5.0:
            return "improving"
        if delta <= -5.0:
            return "declining"
        return "stable"

    @staticmethod
    def compute_average_score(rep_scores: list[float]) -> float:
        """Compute mean score across reps."""
        if not rep_scores:
            return 0.0
        return sum(rep_scores) / len(rep_scores)

    @staticmethod
    def find_most_improved_metric(rep_metrics: list[dict[str, float]]) -> str:
        """Find metric with highest first-to-last positive delta."""
        if len(rep_metrics) < 2:
            return ""

        first = rep_metrics[0]
        last = rep_metrics[-1]
        deltas: dict[str, float] = {}

        for metric_key, first_value in first.items():
            if first_value is None or metric_key not in last or last[metric_key] is None:
                continue
            deltas[metric_key] = float(last[metric_key]) - float(first_value)

        if not deltas:
            return ""

        best_metric = max(deltas, key=deltas.get)
        if deltas[best_metric] <= 0.0:
            return ""
        return best_metric

    @staticmethod
    def find_persistent_issue(rep_metrics: list[dict[str, float]], threshold: float = 75.0) -> str:
        """Find metric that is below threshold in the most reps."""
        if not rep_metrics:
            return ""

        issue_counts: dict[str, int] = {}
        issue_severity_sum: dict[str, float] = {}

        for rep in rep_metrics:
            for metric_key, score in rep.items():
                if score is None:
                    continue
                if float(score) < threshold:
                    issue_counts[metric_key] = issue_counts.get(metric_key, 0) + 1
                    issue_severity_sum[metric_key] = issue_severity_sum.get(metric_key, 0.0) + (threshold - float(score))

        if not issue_counts:
            return ""

        return max(
            issue_counts,
            key=lambda metric: (issue_counts[metric], issue_severity_sum.get(metric, 0.0), metric),
        )
