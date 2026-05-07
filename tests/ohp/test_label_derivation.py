import pytest
from core.exevision.training.ohp.label_derivation import (
    compute_binary_overlap,
    derive_rep_labels,
    RepLabels,
)


def test_binary_overlap_no_errors():
    assert compute_binary_overlap(0.0, 3.0, []) == 0.0


def test_binary_overlap_full_coverage():
    assert compute_binary_overlap(1.0, 4.0, [[0.0, 5.0]]) == 1.0


def test_binary_overlap_partial():
    # Any overlap → 1.0 (binary, not fractional)
    assert compute_binary_overlap(1.0, 4.0, [[2.0, 3.0]]) == 1.0


def test_binary_overlap_no_intersection():
    # Error window entirely outside rep → 0.0
    assert compute_binary_overlap(0.0, 1.0, [[2.0, 3.0]]) == 0.0


def test_binary_overlap_touches_boundary():
    # Boundary touch (w_end == rep_start) is NOT overlap
    assert compute_binary_overlap(2.0, 4.0, [[0.0, 2.0]]) == 0.0


def test_derive_rep_labels_no_knee_error():
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=3.0,
        knee_windows=[],
        heuristic_score=80.0,
    )
    assert isinstance(labels, RepLabels)
    assert labels.knee_error == pytest.approx(0.0)
    # error_score = 100, overall = 0.7*100 + 0.3*80 = 94
    assert labels.overall_score == pytest.approx(94.0, abs=0.1)


def test_derive_rep_labels_with_knee_error():
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=3.0,
        knee_windows=[[0.0, 3.0]],
        heuristic_score=50.0,
    )
    # knee_error = 1.0, error_score = 100*(1-1.0) = 0
    # overall = 0.7*0 + 0.3*50 = 15
    assert labels.knee_error == pytest.approx(1.0)
    assert labels.overall_score == pytest.approx(15.0, abs=0.1)


def test_derive_rep_labels_score_clamped():
    # Score must stay in [0, 100]
    labels = derive_rep_labels(
        rep_start_sec=0.0,
        rep_end_sec=1.0,
        knee_windows=[[0.0, 1.0]],
        heuristic_score=0.0,
    )
    assert labels.overall_score >= 0.0
    assert labels.overall_score <= 100.0
