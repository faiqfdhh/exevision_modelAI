"""Rep-to-rep comparison utilities for progress messaging."""

from __future__ import annotations

from typing import Any


class RepComparator:
    """Computes rep-over-rep deltas and improvement tiers."""

    @staticmethod
    def compute_improvement_percentage(prev_score: float, curr_score: float) -> float:
        """Compute percentage improvement from previous value to current value."""
        if prev_score == 0:
            return 0.0
        return ((curr_score - prev_score) / prev_score) * 100.0

    @staticmethod
    def get_improvement_tier(improvement_pct: float) -> str:
        """Map improvement percentage to tier labels."""
        if improvement_pct >= 15.0:
            return "significant"
        if improvement_pct >= 8.0:
            return "moderate"
        if improvement_pct > 0.0:
            return "slight"
        return "no_improvement"

    def compare_reps(self, prev_rep: dict[str, Any] | None, curr_rep: dict[str, Any]) -> dict[str, Any] | None:
        """Compare two reps and return aggregate and per-metric improvement metadata."""
        if prev_rep is None:
            return None

        prev_overall = float(prev_rep.get("neural_score", 0.0) or 0.0)
        curr_overall = float(curr_rep.get("neural_score", 0.0) or 0.0)
        overall_improvement_pct = self.compute_improvement_percentage(prev_overall, curr_overall)

        prev_scores = prev_rep.get("sub_scores", {}) or {}
        curr_scores = curr_rep.get("sub_scores", {}) or {}

        metric_improvements: dict[str, float] = {}
        metric_tiers: dict[str, str] = {}

        for metric_key, curr_value in curr_scores.items():
            if curr_value is None:
                continue
            prev_value = prev_scores.get(metric_key)
            if prev_value is None:
                continue
            pct = self.compute_improvement_percentage(float(prev_value), float(curr_value))
            metric_improvements[metric_key] = pct
            metric_tiers[metric_key] = self.get_improvement_tier(pct)

        return {
            "overall_improvement_pct": overall_improvement_pct,
            "overall_tier": self.get_improvement_tier(overall_improvement_pct),
            "metric_improvements": metric_improvements,
            "metric_tiers": metric_tiers,
            "prev_rep_id": prev_rep.get("rep_id"),
            "curr_rep_id": curr_rep.get("rep_id"),
        }
