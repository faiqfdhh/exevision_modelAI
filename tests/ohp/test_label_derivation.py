import pytest
from core.exevision.training.ohp.label_derivation import (
    compute_overlap_ratio,
    derive_rep_labels,
    RepLabels,
)


def test_overlap_no_errors():
    assert compute_overlap_ratio(0.0, 3.0, []) == 0.0


def test_overlap_full_coverage():
    ratio = compute_overlap_ratio(1.0, 4.0, [[0.0, 5.0]])
    assert ratio == pytest.approx(1.0)


def test_overlap_partial():
    # rep: 1.0–4.0 (3 sec), error: 2.0–3.0 (1 sec overlap)
    ratio = compute_overlap_ratio(1.0, 4.0, [[2.0, 3.0]])
    assert ratio == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_overlap_multiple_windows():
    # rep: 0–10s, errors: 1–2s (1s) and 5–7s (2s) = 3s / 10s = 0.3
    ratio = compute_overlap_ratio(0.0, 10.0, [[1.0, 2.0], [5.0, 7.0]])
    assert ratio == pytest.approx(0.3, rel=1e-4)


def test_overlap_clamped_at_one():
    # Two overlapping windows could naively exceed 1.0 — must be clamped
    ratio = compute_overlap_ratio(0.0, 2.0, [[0.0, 2.0], [0.5, 1.5]])
    assert ratio <= 1.0


def test_derive_rep_labels_no_errors():
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=3.0,
        elbow_windows=[],
        knee_windows=[],
        heuristic_score=80.0,
        seated=False,
    )
    assert isinstance(labels, RepLabels)
    assert labels.elbow_error_soft == pytest.approx(0.0)
    assert labels.knee_error_soft == pytest.approx(0.0)
    # error_score = 100, overall = 0.7*100 + 0.3*80 = 94
    assert labels.overall_score == pytest.approx(94.0, abs=0.1)


def test_derive_rep_labels_full_elbow_error():
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=3.0,
        elbow_windows=[[0.0, 3.0]],
        knee_windows=[],
        heuristic_score=50.0,
        seated=False,
    )
    # elbow_error_soft = 1.0, error_score = 100*(1-0.65) = 35
    # overall = 0.7*35 + 0.3*50 = 24.5 + 15 = 39.5
    assert labels.elbow_error_soft == pytest.approx(1.0)
    assert labels.overall_score == pytest.approx(39.5, abs=0.1)


def test_derive_rep_labels_seated_ignores_knee():
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=3.0,
        elbow_windows=[],
        knee_windows=[[0.0, 3.0]],  # full knee error
        heuristic_score=80.0,
        seated=True,
    )
    # For seated, knee_error_soft must always be 0.0
    assert labels.knee_error_soft == pytest.approx(0.0)
    # Without knee penalty: error_score = 100, overall = 0.7*100 + 0.3*80 = 94
    assert labels.overall_score == pytest.approx(94.0, abs=0.1)


def test_derive_rep_labels_score_clamped():
    # Pathological: huge errors shouldn't produce negative scores
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=1.0,
        elbow_windows=[[0.0, 1.0]],
        knee_windows=[[0.0, 1.0]],
        heuristic_score=0.0,
        seated=False,
    )
    assert labels.overall_score >= 0.0
    assert labels.overall_score <= 100.0
