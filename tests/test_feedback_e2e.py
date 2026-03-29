"""End-to-end feedback generation test using squat config and templates."""

import pytest

from core.exevision.feedback.engine import FeedbackEngine


def test_feedback_e2e_squat():
    config_root = __import__("pathlib").Path(__file__).parent.parent / "core" / "exevision" / "config"

    exercise_config = config_root / "exercises" / "squat.json"
    templates_config = config_root / "templates" / "feedback_templates.json"

    assert exercise_config.exists(), f"Exercise config not found: {exercise_config}"
    assert templates_config.exists(), f"Templates not found: {templates_config}"

    engine = FeedbackEngine(str(exercise_config), str(templates_config))

    rep_scores = [
        {
            "rep_id": 1,
            "neural_score": 72,
            "metrics": {
                "forward_lean": 35,
                "hip_depth": 0.05,
                "knee_valgus": 0.80,
                "knee_tracking": 0.85,
            },
            "sub_scores": {
                "forward_lean": 60,
                "hip_depth": 70,
                "knee_valgus": 75,
                "knee_tracking": 78,
                "smoothness": 70,
                "control": 68,
            },
        },
        {
            "rep_id": 2,
            "neural_score": 78,
            "metrics": {
                "forward_lean": 28,
                "hip_depth": 0.08,
                "knee_valgus": 0.85,
                "knee_tracking": 0.88,
            },
            "sub_scores": {
                "forward_lean": 68,
                "hip_depth": 75,
                "knee_valgus": 80,
                "knee_tracking": 82,
                "smoothness": 76,
                "control": 75,
            },
        },
        {
            "rep_id": 3,
            "neural_score": 85,
            "metrics": {
                "forward_lean": 22,
                "hip_depth": 0.11,
                "knee_valgus": 0.90,
                "knee_tracking": 0.91,
            },
            "sub_scores": {
                "forward_lean": 76,
                "hip_depth": 82,
                "knee_valgus": 85,
                "knee_tracking": 88,
                "smoothness": 82,
                "control": 84,
            },
        },
    ]

    result = engine.generate_feedback(rep_scores, video_id="vid_e2e")

    assert result.schema_version == "1.0"
    assert result.exercise == "squat"
    assert len(result.reps) == 3
    assert result.session is not None

    assert result.reps[0].rep_id == 1
    assert result.reps[0].score == 72
    assert result.reps[0].tier == "fair"
    assert len(result.reps[0].text) > 0
    assert "forward_lean" in result.reps[0].issues or "smoothness" in result.reps[0].issues

    assert result.reps[1].rep_id == 2
    assert result.reps[1].score == 78
    assert result.reps[1].tier == "good"

    assert result.reps[2].rep_id == 3
    assert result.reps[2].score == 85
    assert result.reps[2].tier == "good"

    assert result.session.trajectory == "improving"
    assert result.session.avg_score == pytest.approx(78.33, abs=0.1)
    assert len(result.session.coach_text) > 0
