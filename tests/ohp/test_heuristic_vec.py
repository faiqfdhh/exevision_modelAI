import numpy as np
import pytest
from core.exevision.neural.ohp.heuristic_vec import (
    OHP_HEURISTIC_DIM,
    build_ohp_heuristic_vector,
)


def _make_rep(overall=70.0, metrics=None, flags=None):
    return {
        "heuristic_score": overall,
        "heuristic_metric_scores": metrics or {
            "grip_ratio": 80.0,
            "rom": 75.0,
            "lockout": 90.0,
            "elbow_flare": 85.0,
        },
        "flags": flags or {
            "incomplete_lockout": False,
            "elbow_flare": False,
            "forward_lean": False,
            "bar_drift": False,
            "wrist_deviation": False,
            "knee_instability": False,
        },
    }


def test_vector_length():
    vec = build_ohp_heuristic_vector(_make_rep(), "front")
    assert vec.shape == (OHP_HEURISTIC_DIM,)
    assert OHP_HEURISTIC_DIM == 16


def test_overall_score_normalised():
    vec = build_ohp_heuristic_vector(_make_rep(overall=80.0), "front")
    assert vec[0] == pytest.approx(0.8)


def test_metric_scores_normalised():
    vec = build_ohp_heuristic_vector(_make_rep(), "front")
    assert vec[1] == pytest.approx(0.80)  # grip_ratio
    assert vec[2] == pytest.approx(0.75)  # rom
    assert vec[3] == pytest.approx(0.90)  # lockout
    assert vec[4] == pytest.approx(0.85)  # elbow_flare


def test_flag_bit_set():
    rep = _make_rep(flags={"incomplete_lockout": True, "elbow_flare": False,
                            "forward_lean": False, "bar_drift": False,
                            "wrist_deviation": False, "knee_instability": False})
    vec = build_ohp_heuristic_vector(rep, "front")
    assert vec[5] == 1.0   # incomplete_lockout at index 5
    assert vec[6] == 0.0   # elbow_flare


def test_view_one_hot_front():
    vec = build_ohp_heuristic_vector(_make_rep(), "front")
    assert vec[11] == 1.0   # front
    assert sum(vec[11:16]) == pytest.approx(1.0)


def test_view_one_hot_side():
    vec = build_ohp_heuristic_vector(_make_rep(), "side")
    assert vec[13] == 1.0   # side
    assert vec[11] == 0.0


def test_unknown_view_all_zeros():
    vec = build_ohp_heuristic_vector(_make_rep(), "unknown")
    assert sum(vec[11:16]) == pytest.approx(0.0)


def test_missing_fields_default_to_zero():
    vec = build_ohp_heuristic_vector({}, "front")
    assert vec[0] == pytest.approx(0.0)
    assert vec.dtype == np.float32
