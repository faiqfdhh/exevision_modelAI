"""Feedback template engine orchestration for rep and session coaching."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.exevision.feedback.rep_comparator import RepComparator
from core.exevision.feedback.session_aggregator import SessionAggregator
from core.exevision.feedback.template_renderer import TemplateRenderer


@dataclass
class RepFeedback:
    """Per-rep feedback payload."""

    rep_id: int
    score: float
    tier: str
    text: str
    wins: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    """Session-level aggregate feedback payload."""

    avg_score: float
    most_improved_metric: str
    persistent_issue: str
    aggregate_text: str
    coach_text: str
    trajectory: str


@dataclass
class FeedbackResult:
    """Complete feedback output for all reps plus session summary."""

    schema_version: str
    exercise: str
    reps: list[RepFeedback]
    session: SessionSummary


class QualityChecker:
    """Detect mismatch between overall score and sub-metric breakdown."""

    @staticmethod
    def detect_mismatch(overall_score: float, sub_scores: dict[str, float], threshold: float = 75.0) -> str:
        """Return mismatch type according to approved quality-check policy."""
        if not sub_scores:
            return "normal"

        valid_scores = [float(v) for v in sub_scores.values() if v is not None]
        if not valid_scores:
            return "normal"

        any_issue = any(v < threshold for v in valid_scores)
        if overall_score < threshold and not any_issue:
            return "low_overall_no_issues"
        if overall_score >= threshold and any_issue:
            return "high_overall_has_issues"
        return "normal"


class FeedbackEngine:
    """Main feedback orchestrator using config + templates + deterministic rendering."""

    def __init__(self, exercise_config_path: str, templates_path: str):
        self.exercise_config = self._load_json(exercise_config_path)
        self.templates = self._load_json(templates_path)
        self.exercise = self.exercise_config["exercise"]

        self._rep_comparator = RepComparator()
        self._session_aggregator = SessionAggregator()
        self._quality_checker = QualityChecker()
        self._renderer = TemplateRenderer()

    @staticmethod
    def _load_json(path: str) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def generate_feedback(self, rep_scores: list[dict[str, Any]], video_id: str = "unknown_video") -> FeedbackResult:
        """Generate per-rep and session-level feedback from score data."""
        rep_feedbacks: list[RepFeedback] = []
        previous_rep: dict[str, Any] | None = None

        threshold = float(self.exercise_config.get("improvement_threshold", 75.0))

        for rep_data in rep_scores:
            rep_id = int(rep_data.get("rep_id", len(rep_feedbacks) + 1))
            score = float(rep_data.get("neural_score") or 0.0)
            sub_scores = self._sanitize_scores(rep_data.get("sub_scores", {}))

            tier = self._get_tier(score)
            mismatch_type = self._quality_checker.detect_mismatch(score, sub_scores, threshold=threshold)
            comparison = self._rep_comparator.compare_reps(previous_rep, {**rep_data, "sub_scores": sub_scores})

            wins = self._get_wins(sub_scores, comparison, threshold)
            issue_scores = {k: v for k, v in sub_scores.items() if v < threshold}
            issues = list(issue_scores.keys())

            rep_text = self._build_rep_feedback(
                video_id=video_id,
                rep_id=rep_id,
                score=score,
                tier=tier,
                wins=wins,
                sub_scores=sub_scores,
                issue_scores=issue_scores,
                mismatch_type=mismatch_type,
                comparison=comparison,
                threshold=threshold,
            )

            rep_feedbacks.append(
                RepFeedback(
                    rep_id=rep_id,
                    score=score,
                    tier=tier,
                    text=rep_text,
                    wins=wins,
                    issues=issues,
                )
            )

            previous_rep = {
                "rep_id": rep_id,
                "neural_score": score,
                "sub_scores": sub_scores,
            }

        session_summary = self._build_session_summary(rep_scores, threshold)

        return FeedbackResult(
            schema_version=self.exercise_config.get("schema_version", "1.0"),
            exercise=self.exercise,
            reps=rep_feedbacks,
            session=session_summary,
        )

    def _sanitize_scores(self, sub_scores: dict[str, Any]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, value in (sub_scores or {}).items():
            if value is None:
                continue
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                continue
        return cleaned

    def _parse_bracket(self, bracket_range: str) -> tuple[float, float]:
        left, right = bracket_range.split("-", maxsplit=1)
        return float(left), float(right)

    def _get_tier(self, score: float) -> str:
        for bracket_range, bracket_info in self.exercise_config.get("score_brackets", {}).items():
            min_val, max_val = self._parse_bracket(bracket_range)
            if min_val <= score <= max_val:
                return str(bracket_info.get("tier", "unknown"))
        return "unknown"

    def _get_bracket_opener(self, tier: str) -> str:
        for _, bracket_info in self.exercise_config.get("score_brackets", {}).items():
            if bracket_info.get("tier") == tier:
                return str(bracket_info.get("opener", ""))
        return ""

    def _get_wins(self, sub_scores: dict[str, float], comparison: dict[str, Any] | None, threshold: float) -> list[str]:
        if comparison is None:
            return []

        wins: list[str] = []
        metric_tiers = comparison.get("metric_tiers", {})
        for metric_key, score in sub_scores.items():
            if score >= threshold and metric_tiers.get(metric_key) not in (None, "no_improvement"):
                wins.append(metric_key)
        return wins

    @staticmethod
    def _metric_phrase_tier(score: float) -> str:
        """Map an individual metric score to a phrase tier key."""
        if score >= 90:
            return "excellent"
        if score >= 85:
            return "strong"
        return "okay"

    def _build_win_texts(
        self,
        video_id: str,
        rep_id: int,
        wins: list[str],
        sub_scores: dict[str, float],
        comparison: dict[str, Any] | None,
    ) -> list[str]:
        if not wins or comparison is None:
            return []

        texts: list[str] = []
        metric_tiers = comparison.get("metric_tiers", {})
        improvement_catalog = self.templates.get("improvement_phrases", {})
        win_catalog = self.templates.get("win_phrases", {})

        for metric_key in wins:
            tier = metric_tiers.get(metric_key, "no_improvement")
            if tier == "no_improvement":
                continue

            improvement_phrases = improvement_catalog.get(tier, [])
            if not improvement_phrases:
                improvement_phrases = ["improving from last rep"]

            improvement_phrase = self._renderer.select_phrase(
                improvement_phrases,
                video_id,
                rep_id,
                f"{metric_key}_improvement",
            )

            phrase_tier = self._metric_phrase_tier(sub_scores.get(metric_key, 75.0))
            win_key = f"improving_metric_{phrase_tier}"
            win_templates = win_catalog.get(win_key, win_catalog.get("improving_metric_okay", []))
            if not win_templates:
                win_templates = ["Your [METRIC_LABEL] is improving — [IMPROVEMENT_PHRASE]!"]

            win_template = self._renderer.select_phrase(win_templates, video_id, rep_id, f"{metric_key}_win")
            text = self._renderer.fill_slots(
                win_template,
                {
                    "METRIC_LABEL": self._renderer.humanize_metric(metric_key),
                    "IMPROVEMENT_PHRASE": improvement_phrase,
                },
            )
            texts.append(text)

        return texts

    def _build_stable_texts(
        self,
        video_id: str,
        rep_id: int,
        sub_scores: dict[str, float],
        wins: list[str],
        issue_keys: list[str],
        threshold: float,
    ) -> list[str]:
        """Generate brief tier-appropriate mentions for metrics >= threshold that are not wins."""
        stable_catalog = self.templates.get("stable_phrases", {})
        if not stable_catalog:
            return []

        texts: list[str] = []
        issue_key_set = set(issue_keys)
        for metric_key, score in sub_scores.items():
            if score < threshold:
                continue
            if metric_key in wins:
                continue
            if metric_key in issue_key_set:
                continue

            phrase_tier = self._metric_phrase_tier(score)
            phrases = stable_catalog.get(phrase_tier, [])
            if not phrases:
                continue

            template = self._renderer.select_phrase(
                phrases,
                video_id,
                rep_id,
                f"{metric_key}_stable",
            )
            text = self._renderer.fill_slots(
                template,
                {"METRIC_LABEL": self._renderer.humanize_metric(metric_key)},
            )
            texts.append(text)

        return texts

    @staticmethod
    def _resolve_issue_tone_mode(overall_score: float, mismatch_type: str) -> str:
        """Resolve issue-language tone based on the rep score policy.

        Policy:
        - 80-100: soft
        - 70-79: strict
        - below 70: very strict

        mismatch_type is retained for compatibility and future fallback behavior.
        """
        if overall_score >= 80:
            return "soft"
        if overall_score >= 70:
            return "strict"
        return "very_strict"

    def _group_issue_cues(self, issue_scores: dict[str, float], tone_mode: str) -> list[str]:
        if not issue_scores:
            return []

        issue_groups = self.exercise_config.get("issue_groups", {})
        severity_band = float(self.exercise_config.get("severity_band", 5.0))

        entries: list[tuple[str, float]] = []
        consumed: set[str] = set()

        for _, group_info in issue_groups.items():
            metrics = group_info.get("metrics", [])
            low_metrics = [metric for metric in metrics if metric in issue_scores]
            if not low_metrics:
                continue

            severity = min(issue_scores[metric] for metric in low_metrics)
            if tone_mode == "soft":
                cue_tier = "needs_work"
            elif tone_mode == "very_strict":
                cue_tier = "focus_here"
            else:
                cue_tier = "focus_here" if severity < 60 else "needs_work"

            if len(low_metrics) > 1:
                combined = group_info.get("combined_cue", "")
                if isinstance(combined, dict):
                    cue = str(combined.get(cue_tier, combined.get("needs_work", ""))).strip()
                else:
                    cue = str(combined).strip()
            else:
                metric = low_metrics[0]
                single = group_info.get("single_cues", {}).get(metric, "")
                if isinstance(single, dict):
                    cue = str(single.get(cue_tier, single.get("needs_work", ""))).strip()
                else:
                    cue = str(single).strip()

            if cue:
                entries.append((cue, severity))
            consumed.update(low_metrics)

        for metric, score in issue_scores.items():
            if metric in consumed:
                continue
            label = self._renderer.humanize_metric(metric)
            if tone_mode == "soft":
                entries.append((f"You can improve {label} a bit more for cleaner reps.", score))
            elif tone_mode == "very_strict":
                entries.append((f"{label} needs immediate correction on your next reps.", score))
            elif score < 60:
                entries.append((f"Let's address {label} - this needs significant work.", score))
            else:
                entries.append((f"Work on {label} - there's room for improvement.", score))

        entries.sort(key=lambda item: item[1])

        # Keep severe issues together by score proximity (±severity_band) while
        # preserving deterministic order from the sorted entries.
        ordered_cues: list[str] = []
        band_anchor: float | None = None
        for cue, severity in entries:
            if band_anchor is None or abs(severity - band_anchor) > severity_band:
                band_anchor = severity
            ordered_cues.append(cue)

        if tone_mode == "soft":
            return [f"Something to keep in mind: {cue}" for cue in ordered_cues]
        if tone_mode == "very_strict":
            return [f"Priority fix: {cue}" for cue in ordered_cues]

        return ordered_cues

    def _build_rep_feedback(
        self,
        video_id: str,
        rep_id: int,
        score: float,
        tier: str,
        wins: list[str],
        sub_scores: dict[str, float],
        issue_scores: dict[str, float],
        mismatch_type: str,
        comparison: dict[str, Any] | None,
        threshold: float,
    ) -> str:
        parts: list[str] = []

        opener = self._get_bracket_opener(tier)
        if opener:
            parts.append(opener)

        if mismatch_type == "low_overall_no_issues":
            parts.append(
                "The system detected concerns with the overall movement quality it could not pinpoint "
                "to a specific metric. Consider reviewing the full rep video."
            )
            return " ".join(parts).strip()

        win_texts = self._build_win_texts(video_id, rep_id, wins, sub_scores, comparison)
        parts.extend(win_texts)

        stable_texts = self._build_stable_texts(
            video_id,
            rep_id,
            sub_scores,
            wins,
            list(issue_scores.keys()),
            threshold,
        )
        parts.extend(stable_texts)

        issue_tone_mode = self._resolve_issue_tone_mode(score, mismatch_type)
        issue_cues = self._group_issue_cues(issue_scores, tone_mode=issue_tone_mode)
        parts.extend(issue_cues)

        return " ".join(part for part in parts if part).strip()

    def _build_session_summary(self, rep_scores: list[dict[str, Any]], threshold: float) -> SessionSummary:
        overall_scores = [float(rep.get("neural_score") or 0.0) for rep in rep_scores]
        metric_scores = [self._sanitize_scores(rep.get("sub_scores", {})) for rep in rep_scores]

        avg_score = self._session_aggregator.compute_average_score(overall_scores)
        trajectory = self._session_aggregator.detect_trajectory(overall_scores)
        most_improved = self._session_aggregator.find_most_improved_metric(metric_scores)
        persistent_issue = self._session_aggregator.find_persistent_issue(metric_scores, threshold=threshold)

        session_templates = self.templates.get("session_summary", {})
        aggregate_template = str(
            session_templates.get(
                "metric_summary",
                "Most improved this session: [TOP_IMPROVEMENT]. Keep working on: [PERSISTENT_ISSUE].",
            )
        )
        aggregate_text = self._renderer.fill_slots(
            aggregate_template,
            {
                "TOP_IMPROVEMENT": self._renderer.humanize_metric(most_improved) if most_improved else "consistency",
                "PERSISTENT_ISSUE": self._renderer.humanize_metric(persistent_issue) if persistent_issue else "nothing specific",
            },
        )

        trajectory_openers = session_templates.get("trajectory_openers", {})
        opener_value = trajectory_openers.get(trajectory, "Session complete.")
        if isinstance(opener_value, list):
            opener_choices = [str(item) for item in opener_value if item]
            if opener_choices:
                trajectory_text = self._renderer.select_phrase(
                    opener_choices,
                    "session",
                    len(rep_scores),
                    f"trajectory_{trajectory}",
                )
            else:
                trajectory_text = "Session complete."
        else:
            trajectory_text = str(opener_value)

        coach_template = str(session_templates.get("coach_cue", "Next time, focus on [PERSISTENT_ISSUE_CUE]."))
        cue_text = self._resolve_issue_cue_text(persistent_issue)
        coach_suffix = self._renderer.fill_slots(coach_template, {"PERSISTENT_ISSUE_CUE": cue_text})
        coach_text = f"{trajectory_text} {coach_suffix}".strip()

        return SessionSummary(
            avg_score=avg_score,
            most_improved_metric=most_improved,
            persistent_issue=persistent_issue,
            aggregate_text=aggregate_text,
            coach_text=coach_text,
            trajectory=trajectory,
        )

    def _resolve_issue_cue_text(self, metric_key: str) -> str:
        if not metric_key:
            return "overall consistency"

        issue_groups = self.exercise_config.get("issue_groups", {})
        for _, group_info in issue_groups.items():
            if metric_key in group_info.get("metrics", []):
                cue = group_info.get("single_cues", {}).get(metric_key, "")
                if isinstance(cue, dict):
                    cue = cue.get("needs_work", "")
                if cue:
                    return str(cue).rstrip(".")
                return self._renderer.humanize_metric(metric_key)

        return self._renderer.humanize_metric(metric_key)
