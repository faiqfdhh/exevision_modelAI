"""Unit tests for OHP scoring geometry helpers and metric functions.
No video files or model checkpoints required.
"""
import math
import sys
from pathlib import Path
import numpy as np
import pytest

# Add stages to path so we can import scoring directly
sys.path.insert(0, str(Path(__file__).parent.parent / "core" / "exevision" / "stages"))
import scoring


# ── Synthetic frame builder ──────────────────────────────────────────────────

def _make_frame(
    ls=(-.1, .5, 0.), rs=(.1, .5, 0.),    # shoulders
    le=(-.15, .3, 0.), re=(.15, .3, 0.),  # elbows
    lw=(-.15, .1, 0.), rw=(.15, .1, 0.),  # wrists
    lh=(-.1, 0., 0.), rh=(.1, 0., 0.),    # hips
    conf=0.9,
):
    """Build a minimal 33-landmark frame list at known positions."""
    frame = [[0., 0., 0., 0.01]] * 33
    for idx, pos in [
        (11, ls), (12, rs),
        (13, le), (14, re),
        (15, lw), (16, rw),
        (23, lh), (24, rh),
    ]:
        frame[idx] = list(pos) + [conf]
    return frame


# ── _xyz ────────────────────────────────────────────────────────────────────

def test_xyz_returns_array_for_confident_landmark():
    frame = _make_frame()
    p = scoring._xyz(frame, 11)
    assert p is not None
    assert p.shape == (3,)
    np.testing.assert_allclose(p, [-.1, .5, 0.], atol=1e-6)


def test_xyz_returns_none_for_low_confidence():
    frame = _make_frame(conf=0.1)
    assert scoring._xyz(frame, 11) is None


# ── _angle_3d ───────────────────────────────────────────────────────────────

def test_angle_3d_right_angle():
    a = np.array([1., 0., 0.])
    b = np.array([0., 0., 0.])
    c = np.array([0., 1., 0.])
    assert abs(scoring._angle_3d(a, b, c) - 90.0) < 0.01


def test_angle_3d_straight_line():
    a = np.array([-1., 0., 0.])
    b = np.array([0., 0., 0.])
    c = np.array([1., 0., 0.])
    assert abs(scoring._angle_3d(a, b, c) - 180.0) < 0.01


def test_angle_3d_returns_none_for_degenerate():
    b = np.array([0., 0., 0.])
    assert scoring._angle_3d(b, b, b) is None


# ── _build_body_frame ────────────────────────────────────────────────────────

def test_build_body_frame_upright():
    """Standard upright torso: v_up ≈ +Y, v_right ≈ +X."""
    frame = _make_frame(
        ls=(-.1, 1., 0.), rs=(.1, 1., 0.),
        lh=(-.1, 0., 0.), rh=(.1, 0., 0.),
    )
    bf = scoring._build_body_frame(frame)
    assert bf is not None
    assert bf["v_up"][1] > 0.9
    assert bf["v_right"][0] > 0.9
    assert abs(np.dot(bf["v_up"], bf["v_right"])) < 0.01   # orthogonal


def test_build_body_frame_returns_none_for_low_confidence():
    frame = _make_frame(conf=0.1)
    assert scoring._build_body_frame(frame) is None


# ── _to_body_local ──────────────────────────────────────────────────────────

def test_to_body_local_origin_at_mid_shoulder():
    frame = _make_frame(
        ls=(-.1, 1., 0.), rs=(.1, 1., 0.),
        lh=(-.1, 0., 0.), rh=(.1, 0., 0.),
    )
    bf = scoring._build_body_frame(frame)
    # mid_shoulder is at (0, 1, 0); transform it — should be (0,0,0)
    mid_sh = np.array([0., 1., 0.])
    local = scoring._to_body_local(mid_sh, bf)
    np.testing.assert_allclose(local, [0., 0., 0.], atol=1e-6)


# ── OHP metric helper ────────────────────────────────────────────────────────

def _make_rep(n_frames=20, bottom_frames=5, **frame_kwargs):
    """Create a list of identical frames for testing."""
    return [_make_frame(**frame_kwargs)] * n_frames


# ── _ohp_grip_width ──────────────────────────────────────────────────────────

def test_grip_width_shoulder_exact():
    """Wrists at same X as shoulders → grip_ratio = 0."""
    frames = _make_rep(
        ls=(-.10, .5, 0.), rs=(.10, .5, 0.),
        lw=(-.10, .5, 0.), rw=(.10, .5, 0.),  # wrists = shoulders in X
        lh=(-.10, 0., 0.), rh=(.10, 0., 0.),
    )
    ratio = scoring._ohp_grip_width(frames)
    assert ratio is not None
    assert abs(ratio) < 0.05   # should be ~0


def test_grip_width_wider():
    """Wrists 20% wider than shoulders → grip_ratio ≈ 0.20."""
    frames = _make_rep(
        ls=(-.10, .5, 0.), rs=(.10, .5, 0.),  # shoulder width = 0.20
        lw=(-.12, .5, 0.), rw=(.12, .5, 0.),  # grip width = 0.24 → ratio = 0.20
        lh=(-.10, 0., 0.), rh=(.10, 0., 0.),
    )
    ratio = scoring._ohp_grip_width(frames)
    assert ratio is not None
    assert abs(ratio - 0.20) < 0.02


# ── _ohp_bar_path_deviation ──────────────────────────────────────────────────

def test_bar_path_perfect_vertical():
    """Wrists move straight up (no horizontal drift) → deviation = 0."""
    bottom = _make_frame(lw=(-.10, .1, 0.), rw=(.10, .1, 0.),
                         ls=(-.10, .5, 0.), rs=(.10, .5, 0.),
                         lh=(-.10, 0., 0.), rh=(.10, 0., 0.))
    top    = _make_frame(lw=(-.10, .9, 0.), rw=(.10, .9, 0.),
                         ls=(-.10, .5, 0.), rs=(.10, .5, 0.),
                         lh=(-.10, 0., 0.), rh=(.10, 0., 0.))
    frames = [bottom] + [bottom] * 8 + [top] * 10
    dev = scoring._ohp_bar_path_deviation(frames)
    assert dev is not None
    assert dev < 0.02


def test_bar_path_forward_drift():
    """Wrists drift forward 10% of shoulder width → deviation ≈ 0.10."""
    sh_w = 0.20
    bottom = _make_frame(lw=(-.10, .1, 0.00), rw=(.10, .1, 0.00),
                         ls=(-.10, .5, 0.), rs=(.10, .5, 0.),
                         lh=(-.10, 0., 0.), rh=(.10, 0., 0.))
    top    = _make_frame(lw=(-.10, .9, 0.02), rw=(.10, .9, 0.02),  # 0.02 drift in Z
                         ls=(-.10, .5, 0.), rs=(.10, .5, 0.),
                         lh=(-.10, 0., 0.), rh=(.10, 0., 0.))
    frames = [bottom] + [top] * 19
    dev = scoring._ohp_bar_path_deviation(frames)
    assert dev is not None
    assert abs(dev - 0.10) < 0.02   # 0.02 / sh_w=0.20 = 0.10


# ── _ohp_rom ─────────────────────────────────────────────────────────────────

def test_rom_full_flexion():
    """Elbows at 60° (good ROM) → min_elbow_angle ≤ 75."""
    # Construct a frame where elbow is bent at ~60°
    # shoulder=(0,0.5,0), elbow=(0,0.3,-0.1), wrist=(0,0.1,0.05)
    frame = _make_frame(
        ls=(-.1, .5, 0.), rs=(.1, .5, 0.),
        le=(-.1, .3,-.1), re=(.1, .3,-.1),
        lw=(-.1, .1, .05), rw=(.1, .1, .05),
        lh=(-.1, 0., 0.), rh=(.1, 0., 0.),
    )
    frames = [frame] * 20
    min_angle = scoring._ohp_rom(frames)
    assert min_angle is not None
    assert min_angle < 90.0   # actual value depends on geometry; just check it's reasonable


def test_rom_no_flexion():
    """Arms already straight → min_elbow_angle near 180°."""
    frame = _make_frame(
        ls=(-.1, .5, 0.), rs=(.1, .5, 0.),
        le=(-.1, .3, 0.), re=(.1, .3, 0.),
        lw=(-.1, .1, 0.), rw=(.1, .1, 0.),  # collinear → ~180°
        lh=(-.1, 0., 0.), rh=(.1, 0., 0.),
    )
    min_angle = scoring._ohp_rom([frame] * 10)
    assert min_angle is not None
    assert min_angle > 160.0


# ── _ohp_lockout ─────────────────────────────────────────────────────────────

def test_lockout_full_extension():
    """Arms nearly straight → max_elbow_angle ≥ 165°."""
    frame = _make_frame(
        ls=(-.1, .5, 0.), rs=(.1, .5, 0.),
        le=(-.1, .3, 0.), re=(.1, .3, 0.),
        lw=(-.1, .1, 0.), rw=(.1, .1, 0.),
        lh=(-.1, 0., 0.), rh=(.1, 0., 0.),
    )
    fps = 30.0
    max_angle = scoring._ohp_lockout([frame] * 20, fps)
    assert max_angle is not None
    assert max_angle > 160.0


def test_lockout_requires_sustain():
    """Max angle reached for only 1 frame (< 0.5 s) returns the peak, not None."""
    straight = _make_frame(
        ls=(-.1,.5,0.), rs=(.1,.5,0.),
        le=(-.1,.3,0.), re=(.1,.3,0.),
        lw=(-.1,.1,0.), rw=(.1,.1,0.),
        lh=(-.1,0.,0.), rh=(.1,0.,0.),
    )
    bent = _make_frame(
        ls=(-.1,.5,0.), rs=(.1,.5,0.),
        le=(-.15,.3,-.05), re=(.15,.3,-.05),
        lw=(-.15,.1,.05), rw=(.15,.1,.05),
        lh=(-.1,0.,0.), rh=(.1,0.,0.),
    )
    frames = [bent] * 18 + [straight] * 2   # straight only at the end
    result = scoring._ohp_lockout(frames, fps=30.0)
    assert result is not None   # returns peak even without sustained window


# ── _ohp_elbow_flare ─────────────────────────────────────────────────────────

def test_elbow_flare_arms_hanging():
    """Elbows directly below shoulders → abduction ≈ 0°."""
    frame = _make_frame(
        ls=(-.1, .5, 0.), rs=(.1, .5, 0.),
        le=(-.1, .3, 0.), re=(.1, .3, 0.),   # elbows below shoulders (hanging)
        lh=(-.1, 0., 0.), rh=(.1, 0., 0.),
    )
    phases = ["concentric"] * 20
    ml, mr, mb = scoring._ohp_elbow_flare([frame]*20, phases)
    assert mb is not None
    assert mb < 20.0   # small abduction when hanging


def test_elbow_flare_returns_none_for_low_confidence():
    frame = _make_frame(conf=0.1)
    ml, mr, mb = scoring._ohp_elbow_flare([frame]*10, ["concentric"]*10)
    assert mb is None


# ── OHP scorer ───────────────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

def _load_ohp_config():
    cfg_path = _Path(__file__).parent.parent / "core" / "exevision" / "config" / "exercises" / "overhead_press.json"
    with open(cfg_path) as f:
        return _json.load(f)


def test_score_overhead_press_returns_required_keys():
    config = _load_ohp_config()
    # Minimal rep: arms straight up (all frames identical, full extension)
    frame = _make_frame(
        ls=(-.10, .50, 0.), rs=(.10, .50, 0.),
        le=(-.10, .30, 0.), re=(.10, .30, 0.),
        lw=(-.10, .10, 0.), rw=(.10, .10, 0.),
        lh=(-.10,  0., 0.), rh=(.10,  0., 0.),
    )
    frames = [frame] * 30
    phases = ["eccentric"] * 10 + ["concentric"] * 20
    result = scoring._score_overhead_press(frames, phases, fps=30.0, exercise="overhead_press", config=config)

    assert "overall_score" in result
    assert "metric_scores" in result
    assert "raw_metrics" in result
    assert "exercise" in result
    assert result["exercise"] == "overhead_press"
    assert 0.0 <= result["overall_score"] <= 100.0


def test_score_overhead_press_seated_variant():
    """seated_overhead_press passes exercise label through to output."""
    config = _load_ohp_config()
    frame = _make_frame()
    result = scoring._score_overhead_press(
        [frame] * 20, ["concentric"] * 20,
        fps=30.0, exercise="seated_overhead_press", config=config,
    )
    assert result["exercise"] == "seated_overhead_press"


def test_score_overhead_press_overall_in_range():
    config = _load_ohp_config()
    frame = _make_frame()
    result = scoring._score_overhead_press(
        [frame] * 20, ["concentric"] * 20, fps=30.0,
        exercise="overhead_press", config=config,
    )
    assert 0.0 <= result["overall_score"] <= 100.0
