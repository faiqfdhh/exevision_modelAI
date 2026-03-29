"""Tests for FeedbackEngine quality checks and full orchestration."""

import json

from core.exevision.feedback.engine import FeedbackEngine, QualityChecker


class TestMismatchDetection:
    def test_no_mismatch_normal(self):
        checker = QualityChecker()

        mismatch = checker.detect_mismatch(
            overall_score=80,
            sub_scores={"depth": 75, "lean": 78, "valgus": 82},
            threshold=75,
        )

        assert mismatch == "normal"

    def test_mismatch_low_overall_no_issues(self):
        checker = QualityChecker()

        mismatch = checker.detect_mismatch(
            overall_score=65,
            sub_scores={"depth": 78, "lean": 80, "valgus": 82},
            threshold=75,
        )

        assert mismatch == "low_overall_no_issues"

    def test_mismatch_high_overall_has_issues(self):
        checker = QualityChecker()

        mismatch = checker.detect_mismatch(
            overall_score=78,
            sub_scores={"depth": 70, "lean": 72, "valgus": 80},
            threshold=75,
        )

        assert mismatch == "high_overall_has_issues"


class TestFeedbackEngineIntegration:
    def test_generate_feedback_full_flow(self, tmp_path):
        config_dir = tmp_path / "exercises"
        config_dir.mkdir()
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        exercise_config = {
            "schema_version": "1.0",
            "exercise": "squat",
            "score_brackets": {
                "90-100": {"tier": "excellent", "opener": "Excellent!"},
                "75-89": {"tier": "good", "opener": "Good."},
                "60-74": {"tier": "fair", "opener": "Fair."},
                "0-59": {"tier": "poor", "opener": "Poor."},
            },
            "improvement_threshold": 75,
            "severity_band": 5,
            "issue_groups": {
                "descent": {
                    "metrics": ["forward_lean", "hip_depth"],
                    "label": "descent",
                    "single_cues": {"forward_lean": "Upright.", "hip_depth": "Deeper."},
                    "combined_cue": "Work descent.",
                }
            },
            "metrics": {
                "forward_lean": {"good_threshold": 20, "bad_threshold": 45, "unit": "deg"},
                "hip_depth": {"good_threshold": 0.1, "bad_threshold": -0.1, "unit": "norm"},
            },
        }

        templates = {
            "schema_version": "1.0",
            "improvement_phrases": {"significant": ["much better"], "moderate": ["better"], "slight": ["slightly better"]},
            "win_phrases": {
                "excellent_metric": ["Great [METRIC_LABEL]!"],
                "improving_metric": ["Great [METRIC_LABEL] - [IMPROVEMENT_PHRASE]!"]
            },
            "issue_templates": {"single_issue": "[ISSUE_CUE]"},
            "session_summary": {
                "trajectory_openers": {"improving": "Good!", "stable": "Stable.", "declining": "Tough."},
                "metric_summary": "Best: [TOP_IMPROVEMENT]. Work: [PERSISTENT_ISSUE].",
                "coach_cue": "Next: [PERSISTENT_ISSUE_CUE].",
            },
        }

        (config_dir / "squat.json").write_text(json.dumps(exercise_config), encoding="utf-8")
        (template_dir / "feedback_templates.json").write_text(json.dumps(templates), encoding="utf-8")

        engine = FeedbackEngine(str(config_dir / "squat.json"), str(template_dir / "feedback_templates.json"))

        rep_scores = [
            {
                "rep_id": 1,
                "neural_score": 72,
                "metrics": {"forward_lean": 35, "hip_depth": 0.05, "smoothness": 70},
                "sub_scores": {"forward_lean": 60, "hip_depth": 70},
            },
            {
                "rep_id": 2,
                "neural_score": 78,
                "metrics": {"forward_lean": 30, "hip_depth": 0.08, "smoothness": 75},
                "sub_scores": {"forward_lean": 68, "hip_depth": 75},
            },
            {
                "rep_id": 3,
                "neural_score": 82,
                "metrics": {"forward_lean": 25, "hip_depth": 0.11, "smoothness": 80},
                "sub_scores": {"forward_lean": 78, "hip_depth": 80},
            },
        ]

        result = engine.generate_feedback(rep_scores, video_id="vid_001")

        assert result.schema_version == "1.0"
        assert result.exercise == "squat"
        assert len(result.reps) == 3
        assert result.session is not None

        assert result.reps[0].rep_id == 1
        assert result.reps[0].score == 72
        assert result.reps[0].tier == "fair"
        assert len(result.reps[0].text) > 0

        assert result.session.trajectory == "improving"
