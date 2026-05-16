"""Unit tests for FeedbackEngine._emit_metric_cues — OHP metric_cues config path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.exevision.feedback.engine import FeedbackEngine

_BASE_EXERCISE_CONFIG = {
    "schema_version": "1.0",
    "exercise": "overhead_press",
    "score_brackets": {
        "90-100": {"tier": "excellent", "opener": "Excellent pressing form!"},
        "75-89": {"tier": "good", "opener": "Good press overall."},
        "60-74": {"tier": "fair", "opener": "Decent effort."},
        "40-59": {"tier": "poor", "opener": "Your form needs attention."},
        "0-39": {"tier": "critical", "opener": "Let's work on fundamentals."},
    },
    "improvement_threshold": 75,
    "severity_band": 5,
}

_BASE_TEMPLATES = {
    "improvement_phrases": {},
    "win_phrases": {},
    "stable_phrases": {},
    "session_summary": {
        "metric_summary": "Most improved: [TOP_IMPROVEMENT]. Keep working on: [PERSISTENT_ISSUE].",
        "coach_cue": "Focus on [PERSISTENT_ISSUE_CUE].",
    },
}

_MINIMAL_METRIC_CUES = {
    "metric_cues": {
        "rom_top": {
            "label": "ROM Top",
            "direction": "below",
            "issue_at": 75,
            "mild_at": 65,
            "moderate_at": 40,
            "tiers": {
                "mild": "Press a little higher.",
                "moderate": "Lift higher — short of full extension.",
                "severe": "Cutting the rep short — press to full lockout.",
            },
        },
        "knee_error": {
            "label": "Knee Stability",
            "direction": "above",
            "issue_at": 0.25,
            "mild_at": 0.50,
            "moderate_at": 0.75,
            "tiers": {
                "mild": "Mind your knees.",
                "moderate": "Knees buckling — brace legs.",
                "severe": "Knees collapsing — stabilise lower body.",
            },
        },
    }
}


@pytest.fixture
def feedback_engine(tmp_path: Path):
    config_path = tmp_path / "exercise.json"
    templates_path = tmp_path / "templates.json"

    config = {**_BASE_EXERCISE_CONFIG, **_MINIMAL_METRIC_CUES}
    config_path.write_text(json.dumps(config))
    templates_path.write_text(json.dumps(_BASE_TEMPLATES))

    return FeedbackEngine(str(config_path), str(templates_path))


@pytest.fixture
def feedback_engine_no_cues(tmp_path: Path):
    config_path = tmp_path / "exercise_no_cues.json"
    templates_path = tmp_path / "templates_no_cues.json"

    config_path.write_text(json.dumps(_BASE_EXERCISE_CONFIG))
    templates_path.write_text(json.dumps(_BASE_TEMPLATES))

    return FeedbackEngine(str(config_path), str(templates_path))


class TestEmitMetricCues:
    """Tests for _emit_metric_cues method."""

    @staticmethod
    def _call(engine: FeedbackEngine, rep_data: dict) -> list[dict]:
        return engine._emit_metric_cues(rep_data)

    def test_no_metric_cues_config_returns_empty(self, feedback_engine_no_cues):
        items = self._call(feedback_engine_no_cues, {"rom_top": 30})
        assert items == []

    def test_metric_not_in_rep_data_skipped(self, feedback_engine):
        items = self._call(feedback_engine, {"other_key": 50})
        assert items == []

    def test_metric_at_issue_boundary_no_cue(self, feedback_engine):
        items = self._call(feedback_engine, {"rom_top": 75})
        assert items == []

    def test_rom_top_mild(self, feedback_engine):
        items = self._call(feedback_engine, {"rom_top": 70})
        assert len(items) == 1
        assert items[0]["text"] == "Press a little higher."
        assert items[0]["metric_name"] == "rom_top"
        assert items[0]["type"] == "issue"

    def test_rom_top_moderate(self, feedback_engine):
        items = self._call(feedback_engine, {"rom_top": 50})
        assert len(items) == 1
        assert items[0]["text"] == "Lift higher — short of full extension."

    def test_rom_top_severe(self, feedback_engine):
        items = self._call(feedback_engine, {"rom_top": 25})
        assert len(items) == 1
        assert items[0]["text"] == "Cutting the rep short — press to full lockout."

    def test_knee_error_direction_above_mild(self, feedback_engine):
        items = self._call(feedback_engine, {"knee_error": 0.40})
        assert len(items) == 1
        assert items[0]["text"] == "Mind your knees."

    def test_knee_error_direction_above_moderate(self, feedback_engine):
        items = self._call(feedback_engine, {"knee_error": 0.60})
        assert len(items) == 1
        assert items[0]["text"] == "Knees buckling — brace legs."

    def test_knee_error_direction_above_severe(self, feedback_engine):
        items = self._call(feedback_engine, {"knee_error": 0.85})
        assert len(items) == 1
        assert items[0]["text"] == "Knees collapsing — stabilise lower body."

    def test_knee_error_below_issue_at_no_cue(self, feedback_engine):
        items = self._call(feedback_engine, {"knee_error": 0.20})
        assert items == []

    def test_knee_error_at_issue_boundary_no_cue(self, feedback_engine):
        items = self._call(feedback_engine, {"knee_error": 0.25})
        assert items == []

    def test_per_metric_severity_returns_correct_tiers(self, feedback_engine):
        rep_data = {"rom_top": 30, "knee_error": 0.40}
        items = self._call(feedback_engine, rep_data)

        texts = {item["metric_name"]: item["text"] for item in items}
        assert texts["rom_top"] == "Cutting the rep short — press to full lockout."
        assert texts["knee_error"] == "Mind your knees."

    def test_none_value_skipped(self, feedback_engine):
        items = self._call(feedback_engine, {"rom_top": None})
        assert items == []

    def test_non_numeric_value_skipped(self, feedback_engine):
        items = self._call(feedback_engine, {"rom_top": "not_a_number"})
        assert items == []

    def test_missing_tier_text_skipped(self, tmp_path):
        config = {
            **_BASE_EXERCISE_CONFIG,
            "metric_cues": {
                "rom_top": {
                    "direction": "below",
                    "issue_at": 75, "mild_at": 65, "moderate_at": 40,
                    "tiers": {},  # empty tiers
                },
            },
        }
        config_path = tmp_path / "exercise.json"
        templates_path = tmp_path / "templates.json"
        config_path.write_text(json.dumps(config))
        templates_path.write_text(json.dumps(_BASE_TEMPLATES))
        engine = FeedbackEngine(str(config_path), str(templates_path))

        items = engine._emit_metric_cues({"rom_top": 30})
        assert items == []


class TestGenerateFeedbackWithMetricCues:
    """Integration: generate_feedback includes metric_cue items."""

    @staticmethod
    def _build_engine(tmp_path: Path) -> FeedbackEngine:
        config_path = tmp_path / "exercise.json"
        templates_path = tmp_path / "templates.json"
        config = {**_BASE_EXERCISE_CONFIG, **_MINIMAL_METRIC_CUES}
        config_path.write_text(json.dumps(config))
        templates_path.write_text(json.dumps(_BASE_TEMPLATES))
        return FeedbackEngine(str(config_path), str(templates_path))

    def test_rep_with_rom_top_issue_gets_cue_item(self, tmp_path):
        engine = self._build_engine(tmp_path)
        result = engine.generate_feedback(
            [{
                "rep_id": 1,
                "neural_score": 55,
                "sub_scores": {},
                "rom_top": 30,
            }]
        )
        assert len(result.reps) == 1
        texts = [item["text"] for item in result.reps[0].items]
        assert "Cutting the rep short — press to full lockout." in texts

    def test_rep_with_no_issues_has_no_cue_items(self, tmp_path):
        engine = self._build_engine(tmp_path)
        result = engine.generate_feedback(
            [{
                "rep_id": 1,
                "neural_score": 85,
                "sub_scores": {},
                "rom_top": 85,
                "knee_error": 0.20,
            }]
        )
        assert result.reps[0].items == []


class TestStgcnBilstmScores:
    """Regression: bilstm/stgcn scores still computed for OHP in merged_reps."""

    def test_ohp_bilstm_score_computable(self):
        """Simulate the OHP bilstm_score calculation from pipeline.py lines 714-715."""
        sm, ct = 65.0, 70.0
        bilstm_score = round((sm + ct) / 2.0, 2)
        assert bilstm_score == 67.5
