# tests/feedback/test_llm_enhancer.py
"""Unit tests for LLMFeedbackEnhancer — all LLM calls mocked."""
import dataclasses
import json
from unittest.mock import MagicMock

import pytest

from core.exevision.feedback.engine import (
    FeedbackItem,
    FeedbackResult,
    RepFeedback,
    SessionSummary,
)
from core.exevision.feedback.llm_enhancer import LLMFeedbackEnhancer


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_rep(rep_id=1, score=55.0, tier="poor", text="Original template text."):
    return RepFeedback(
        rep_id=rep_id,
        score=score,
        tier=tier,
        text=text,
        items=[
            FeedbackItem(
                text="Fix lockout.",
                score=20,
                category="spatial",
                type="issue",
                metric_key="rom_top",
            )
        ],
    )


def _make_session_digest():
    return [
        {
            "rep_id": 1,
            "score": 55,
            "tier": "poor",
            "sub_scores": {"rom top": 20},
            "issues": [{"metric": "rom top", "score": 20, "cue": "Fix lockout."}],
            "wins": [],
        }
    ]


def _make_result(reps=None, exercise="overhead_press", session_digest=None):
    return FeedbackResult(
        schema_version="1.0",
        exercise=exercise,
        reps=[_make_rep()] if reps is None else reps,
        session=SessionSummary(
            avg_score=55.0,
            most_improved_metric="",
            persistent_issue="rom_top",
            aggregate_text="Keep working on rom_top.",
            coach_text="Next time focus on rom_top.",
            trajectory="stable",
            session_digest=_make_session_digest() if session_digest is None else session_digest,
        ),
    )


def _make_enhancer(llm_response: str) -> LLMFeedbackEnhancer:
    """Build enhancer with mocked rep chain — no real API calls."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = llm_response
    return LLMFeedbackEnhancer(api_key="fake-key", _chain=mock_chain)


def _make_session_enhancer(llm_response: str) -> LLMFeedbackEnhancer:
    """Build enhancer with mocked session chain — no real API calls."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = llm_response
    return LLMFeedbackEnhancer(api_key="fake-key", _session_chain=mock_chain)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_enhance_rep_returns_llm_output():
    """enhance_rep replaces text with LLM response."""
    enhancer = _make_enhancer("Drive to full lockout every rep.")
    rep = _make_rep()
    result = enhancer.enhance_rep(rep, "overhead_press")
    assert result == "Drive to full lockout every rep."


def test_enhance_rep_falls_back_on_exception():
    """enhance_rep returns original text when chain raises."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("API timeout")
    enhancer = LLMFeedbackEnhancer(api_key="fake-key", _chain=mock_chain)
    rep = _make_rep(text="Original template text.")
    result = enhancer.enhance_rep(rep, "overhead_press")
    assert result == "Original template text."


def test_enhance_rep_passes_correct_context():
    """enhance_rep passes score, tier, issues, and template_text to chain."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Enhanced."
    enhancer = LLMFeedbackEnhancer(api_key="fake-key", _chain=mock_chain)

    rep = _make_rep(score=45.0, tier="poor", text="Fix lockout.")
    enhancer.enhance_rep(rep, "overhead_press")

    # chain.invoke called with single positional dict — [0][0] = first positional arg
    call_kwargs = mock_chain.invoke.call_args[0][0]
    assert call_kwargs["exercise"] == "overhead_press"
    assert call_kwargs["score"] == 45
    assert call_kwargs["tier"] == "poor"
    assert "rom_top" in call_kwargs["issues"]
    assert call_kwargs["template_text"] == "Fix lockout."


def test_enhance_result_replaces_text_in_all_reps():
    """enhance_result returns new FeedbackResult with all rep texts replaced."""
    enhancer = _make_enhancer("Natural sentence.")
    result = _make_result(reps=[_make_rep(rep_id=1), _make_rep(rep_id=2)])
    enhanced = enhancer.enhance_result(result)

    assert len(enhanced.reps) == 2
    assert enhanced.reps[0].text == "Natural sentence."
    assert enhanced.reps[1].text == "Natural sentence."


def test_enhance_result_passes_exercise_to_enhance_rep():
    """enhance_result passes result.exercise to each enhance_rep call."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Enhanced."
    enhancer = LLMFeedbackEnhancer(api_key="fake-key", _chain=mock_chain)
    result = _make_result(reps=[_make_rep()], exercise="squat")
    enhancer.enhance_result(result)

    call_kwargs = mock_chain.invoke.call_args[0][0]
    assert call_kwargs["exercise"] == "squat"


def test_enhance_result_leaves_items_unchanged():
    """enhance_result does not modify items (structured UI bullets)."""
    enhancer = _make_enhancer("Natural sentence.")
    rep = _make_rep()
    original_items = rep.items.copy()
    result = _make_result(reps=[rep])
    enhanced = enhancer.enhance_result(result)

    assert enhanced.reps[0].items == original_items


def test_enhance_result_leaves_session_unchanged():
    """enhance_result does not modify session summary."""
    enhancer = _make_enhancer("Natural sentence.")
    result = _make_result()
    enhanced = enhancer.enhance_result(result)

    assert enhanced.session.coach_text == result.session.coach_text
    assert enhanced.session.trajectory == result.session.trajectory


def test_enhance_result_original_unchanged():
    """enhance_result returns a new object; original FeedbackResult is not mutated."""
    enhancer = _make_enhancer("Natural sentence.")
    result = _make_result()
    original_text = result.reps[0].text
    enhancer.enhance_result(result)

    assert result.reps[0].text == original_text


def test_enhance_result_handles_empty_reps():
    """enhance_result returns valid FeedbackResult when reps list is empty."""
    enhancer = _make_enhancer("Natural sentence.")
    result = _make_result(reps=[])
    enhanced = enhancer.enhance_result(result)

    assert enhanced.reps == []
    assert enhanced.exercise == result.exercise


def test_enhance_result_partial_failure_isolated():
    """If one rep's chain call fails, others still get enhanced."""
    mock_chain = MagicMock()
    # rep 1 succeeds, rep 2 fails, rep 3 succeeds
    mock_chain.invoke.side_effect = [
        "Enhanced rep 1.",
        RuntimeError("API timeout"),
        "Enhanced rep 3.",
    ]
    enhancer = LLMFeedbackEnhancer(api_key="fake-key", _chain=mock_chain)
    result = _make_result(reps=[
        _make_rep(rep_id=1, text="Original 1."),
        _make_rep(rep_id=2, text="Original 2."),
        _make_rep(rep_id=3, text="Original 3."),
    ])
    enhanced = enhancer.enhance_result(result)

    assert enhanced.reps[0].text == "Enhanced rep 1."
    assert enhanced.reps[1].text == "Original 2."   # fallback
    assert enhanced.reps[2].text == "Enhanced rep 3."


# ── enhance_session tests ───────────────────────────────────────────────────

def test_enhance_session_replaces_coach_text():
    """enhance_session replaces session.coach_text with LLM output."""
    enhancer = _make_session_enhancer("Lockout dropped from 80 to 55 across reps — drive to full extension.")
    result = _make_result()
    enhanced = enhancer.enhance_session(result)

    assert enhanced.session.coach_text == "Lockout dropped from 80 to 55 across reps — drive to full extension."


def test_enhance_session_falls_back_on_exception():
    """enhance_session returns original result when chain raises."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("API timeout")
    enhancer = LLMFeedbackEnhancer(api_key="fake-key", _session_chain=mock_chain)
    result = _make_result()

    enhanced = enhancer.enhance_session(result)

    assert enhanced.session.coach_text == result.session.coach_text


def test_enhance_session_passes_digest_to_chain():
    """enhance_session passes exercise, rep_count, avg_score, trajectory, and digest JSON."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Enhanced session notes."
    enhancer = LLMFeedbackEnhancer(api_key="fake-key", _session_chain=mock_chain)
    result = _make_result(exercise="overhead_press")

    enhancer.enhance_session(result)

    call_kwargs = mock_chain.invoke.call_args[0][0]
    assert call_kwargs["exercise"] == "overhead_press"
    assert call_kwargs["rep_count"] == 1
    assert call_kwargs["avg_score"] == 55
    assert call_kwargs["trajectory"] == "stable"
    assert json.loads(call_kwargs["session_digest"]) == _make_session_digest()


def test_enhance_session_leaves_reps_unchanged():
    """enhance_session does not modify per-rep feedback."""
    enhancer = _make_session_enhancer("Enhanced session notes.")
    result = _make_result()
    enhanced = enhancer.enhance_session(result)

    assert enhanced.reps == result.reps


def test_enhance_session_original_unchanged():
    """enhance_session returns a new object; original FeedbackResult is not mutated."""
    enhancer = _make_session_enhancer("Enhanced session notes.")
    result = _make_result()
    original_coach_text = result.session.coach_text

    enhancer.enhance_session(result)

    assert result.session.coach_text == original_coach_text
