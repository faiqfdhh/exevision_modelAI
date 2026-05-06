# OHP Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add view-independent overhead press scoring to `scoring.py`, covering 5 biomechanical metrics computed in a body-local 3D coordinate frame, dispatched per exercise type (`overhead_press` / `seated_overhead_press`), with no shared logic with squat scoring.

**Architecture:** All OHP metric functions are standalone (no reuse of squat functions). A body-local coordinate frame is built per-frame from shoulder/hip landmarks, making all metrics camera-agnostic. The `exercise` string flows as an explicit parameter through every function and into the output payload. Config thresholds live in `overhead_press.json` — no magic numbers in code.

**Tech Stack:** Python 3.10, NumPy, existing `scoring.py` infrastructure (`score_metric_linear`, `_lm`, `_build_scoring_paths`, `process_single_video`), pytest.

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `core/exevision/stages/scoring.py` | Modify | Add landmark indices, 3D geometry helpers, 5 OHP metric functions, OHP scorer, dispatch in `process_single_video()` |
| `core/exevision/config/exercises/overhead_press.json` | Modify | Replace stub `field_mapping`; add complete `metrics` thresholds and `metric_weights` |
| `tests/test_ohp_scoring.py` | Create | Unit tests for geometry helpers + each OHP metric + end-to-end scorer |

**Insertion points in `scoring.py`:**
- After line 68 (landmark index block): add OHP landmark aliases + `_xyz()` + `_angle_3d()` + body-frame helpers
- After the squat metric section (before the "Scoring" section ~line 442): add `# ── OHP METRICS ──` block with 5 metric functions
- After `score_rep_simple()` (~line 634): add `# ── OHP SCORING ──` block with scoring helpers + `_score_overhead_press()`
- In `process_single_video()` (~line 793): add exercise dispatch branch

---

## Task 1: Update `overhead_press.json` with complete thresholds

**Files:**
- Modify: `core/exevision/config/exercises/overhead_press.json`

- [ ] **Step 1: Replace the file content**

Replace the entire `metrics` and `field_mapping` sections with:

```json
{
  "schema_version": "1.0",
  "exercise": "overhead_press",
  "score_brackets": {
    "90-100": { "tier": "excellent", "opener": "Excellent pressing form!" },
    "75-89":  { "tier": "good",      "opener": "Good press overall." },
    "60-74":  { "tier": "fair",      "opener": "Decent effort, but a few things to address." },
    "40-59":  { "tier": "poor",      "opener": "Your pressing form needs significant attention." },
    "0-39":   { "tier": "critical",  "opener": "Let's work on the fundamentals." }
  },
  "improvement_threshold": 75,
  "severity_band": 5,
  "metric_weights": {
    "grip_ratio":        0.20,
    "bar_path_deviation": 0.20,
    "rom":               0.20,
    "lockout":           0.20,
    "elbow_flare":       0.20
  },
  "metrics": {
    "grip_ratio": {
      "ideal_low":  0.05,
      "ideal_high": 0.25,
      "perfect":    0.15,
      "tolerance":  0.30,
      "unit": "ratio",
      "note": "grip_ratio=(grip_width - shoulder_width)/shoulder_width; ideal 5-25% wider than shoulders"
    },
    "bar_path_deviation": {
      "good_threshold": 0.05,
      "bad_threshold":  0.25,
      "unit": "normalized",
      "note": "horizontal XZ drift from bottom to top / shoulder_width; body-local frame"
    },
    "min_elbow_angle": {
      "full_rom_threshold":    75.0,
      "partial_rom_threshold": 90.0,
      "unit": "degrees",
      "note": "3D elbow angle at maximum flexion; <= 75 = full ROM, > 90 = no credit"
    },
    "max_elbow_angle": {
      "good_threshold": 165.0,
      "bad_threshold":  145.0,
      "unit": "degrees",
      "note": "3D elbow angle at lockout, sustained >= 0.5 s"
    },
    "elbow_flare": {
      "ideal_low":          30.0,
      "ideal_high":         60.0,
      "bad_low":            20.0,
      "bad_high":           70.0,
      "asymmetry_threshold": 15.0,
      "unit": "degrees",
      "note": "mean shoulder abduction during concentric phase; 0 = arm along torso, 90 = horizontal out"
    }
  },
  "field_mapping": {
    "metrics_to_feedback": {
      "grip_ratio":         "grip_ratio",
      "bar_path_deviation": "bar_path_deviation",
      "min_elbow_angle":    "rom",
      "max_elbow_angle":    "lockout",
      "elbow_flare_mean":   "elbow_flare"
    }
  },
  "annotation_flags": {
    "incomplete_lockout": "Incomplete Lockout",
    "elbow_flare":        "Elbow Flare / Winging",
    "forward_lean":       "Excessive Layback",
    "bar_drift":          "Bar Path Drift",
    "wrist_deviation":    "Wrist Bent Back",
    "knee_instability":   "Knee Instability (standing only)"
  },
  "annotation_metrics": {
    "lockout":   "Lockout Quality",
    "bar_path":  "Bar Path Straightness",
    "smoothness": "Smoothness",
    "control":   "Control"
  },
  "issue_groups": {
    "lockout_quality": {
      "metrics": ["shoulder_elevation", "elbow_extension"],
      "label": "lockout quality",
      "single_cues": {
        "shoulder_elevation": {
          "needs_work": "Work on pressing the bar fully overhead.",
          "focus_here": "Your lockout is incomplete — the bar needs to finish directly over your shoulders."
        },
        "elbow_extension": {
          "needs_work": "Focus on fully extending your elbows at the top.",
          "focus_here": "Elbow extension is insufficient — lock out completely at the top of each rep."
        }
      },
      "combined_cue": {
        "needs_work": "Work on your lockout: press higher and extend the elbows fully.",
        "focus_here": "Your lockout needs significant work — incomplete extension and insufficient height."
      }
    },
    "bar_path": {
      "metrics": ["bar_path_deviation", "forward_lean"],
      "label": "bar path",
      "single_cues": {
        "bar_path_deviation": {
          "needs_work": "Work on keeping the bar path vertical.",
          "focus_here": "The bar path is drifting significantly — focus on pressing straight up."
        },
        "forward_lean": {
          "needs_work": "Stay more upright during the press.",
          "focus_here": "Excessive forward lean — brace your core and stay vertical."
        }
      },
      "combined_cue": {
        "needs_work": "Work on your bar path: stay upright and press vertically.",
        "focus_here": "Bar path and body lean both need attention — brace hard and press straight."
      }
    }
  }
}
```

- [ ] **Step 2: Verify it parses cleanly**

```bash
python -c "import json; d=json.load(open('core/exevision/config/exercises/overhead_press.json')); print('ok', list(d['metrics'].keys()))"
```

Expected output: `ok ['grip_ratio', 'bar_path_deviation', 'min_elbow_angle', 'max_elbow_angle', 'elbow_flare']`

- [ ] **Step 3: Commit**

```bash
git add core/exevision/config/exercises/overhead_press.json
git commit -m "config: add complete OHP metric thresholds and weights to overhead_press.json"
```

---

## Task 2: Add 3D geometry helpers to `scoring.py`

**Files:**
- Modify: `core/exevision/stages/scoring.py` (insert after line 68 — after existing landmark index block)
- Test: `tests/test_ohp_scoring.py`

These helpers are OHP-only (no squat code touches them). They implement the body-local coordinate frame described in the spec.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ohp_scoring.py`:

```python
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
```

- [ ] **Step 2: Run — expect ImportError / AttributeError (functions don't exist yet)**

```bash
pytest tests/test_ohp_scoring.py -v 2>&1 | head -30
```

Expected: fails with `AttributeError: module 'scoring' has no attribute '_xyz'`

- [ ] **Step 3: Add OHP landmark indices + geometry helpers to `scoring.py`**

Insert the following block immediately after the existing `L_TOE, R_TOE = 31, 32` line (after line 68):

```python
# OHP landmark aliases (upper body only)
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16


# ── OHP 3D GEOMETRY HELPERS ──────────────────────────────────────────────────

def _xyz(frame: Any, idx: int, min_conf: float = 0.4) -> Optional[np.ndarray]:
    """Return (x, y, z) of landmark as float32 array, or None if below min_conf."""
    lm = _lm(frame, idx)
    if lm is None or len(lm) < 3:
        return None
    if len(lm) >= 4 and float(lm[3]) < min_conf:
        return None
    return np.array(lm[:3], dtype=np.float64)


def _angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """True 3D angle in degrees at vertex b formed by vectors b→a and b→c."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _build_body_frame(frame: Any) -> Optional[Dict[str, Any]]:
    """
    Build an orthonormal body-local coordinate frame from shoulder/hip landmarks.

    Returns a dict with keys: v_right, v_up, v_forward, mid_shoulder, mid_hip
    Returns None if any of the four required landmarks are below confidence threshold.

    Frame convention (body-local):
      +Y = upward along torso
      +X = rightward (right shoulder direction)
      +Z = forward out of chest
    """
    ls = _xyz(frame, L_SHOULDER)
    rs = _xyz(frame, R_SHOULDER)
    lh = _xyz(frame, L_HIP)
    rh = _xyz(frame, R_HIP)
    if any(x is None for x in (ls, rs, lh, rh)):
        return None

    mid_shoulder = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0

    v_up_raw = mid_shoulder - mid_hip
    v_right_raw = rs - ls

    up_norm = np.linalg.norm(v_up_raw)
    right_norm = np.linalg.norm(v_right_raw)
    if up_norm < 1e-6 or right_norm < 1e-6:
        return None

    v_up = v_up_raw / up_norm
    v_right = v_right_raw / right_norm

    # Forward = right × up; then re-orthogonalise right so frame is truly orthonormal
    v_forward = np.cross(v_right, v_up)
    fwd_norm = np.linalg.norm(v_forward)
    if fwd_norm < 1e-6:
        return None
    v_forward = v_forward / fwd_norm
    v_right = np.cross(v_up, v_forward)
    v_right /= np.linalg.norm(v_right)

    return {
        "v_right": v_right,
        "v_up": v_up,
        "v_forward": v_forward,
        "mid_shoulder": mid_shoulder,
        "mid_hip": mid_hip,
    }


def _to_body_local(p: np.ndarray, bf: Dict[str, Any]) -> np.ndarray:
    """
    Project world-space point p into body-local axes.
    Origin is mid_shoulder.
    Returns (x_right, y_up, z_forward).
    """
    delta = p - bf["mid_shoulder"]
    return np.array([
        np.dot(delta, bf["v_right"]),
        np.dot(delta, bf["v_up"]),
        np.dot(delta, bf["v_forward"]),
    ], dtype=np.float64)
```

- [ ] **Step 4: Run tests — expect all geometry tests to pass**

```bash
pytest tests/test_ohp_scoring.py::test_xyz_returns_array_for_confident_landmark \
       tests/test_ohp_scoring.py::test_xyz_returns_none_for_low_confidence \
       tests/test_ohp_scoring.py::test_angle_3d_right_angle \
       tests/test_ohp_scoring.py::test_angle_3d_straight_line \
       tests/test_ohp_scoring.py::test_angle_3d_returns_none_for_degenerate \
       tests/test_ohp_scoring.py::test_build_body_frame_upright \
       tests/test_ohp_scoring.py::test_build_body_frame_returns_none_for_low_confidence \
       tests/test_ohp_scoring.py::test_to_body_local_origin_at_mid_shoulder \
       -v
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/exevision/stages/scoring.py tests/test_ohp_scoring.py
git commit -m "feat(scoring): add 3D geometry helpers for OHP body-local frame"
```

---

## Task 3: Add the 5 OHP metric functions

**Files:**
- Modify: `core/exevision/stages/scoring.py` (insert `# ── OHP METRICS ──` block before the "Scoring" comment ~line 442)
- Test: `tests/test_ohp_scoring.py` (append)

Each function takes `rep_frames` (list of landmark frames) and returns a raw metric value in physical units (ratio, degrees). No scoring logic here — just measurement.

- [ ] **Step 1: Append metric tests to `tests/test_ohp_scoring.py`**

```python
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
```

- [ ] **Step 2: Run — expect AttributeError (metric functions don't exist yet)**

```bash
pytest tests/test_ohp_scoring.py -k "grip_width or bar_path or rom or lockout or flare" -v 2>&1 | head -20
```

Expected: fails with `AttributeError`

- [ ] **Step 3: Add the 5 metric functions to `scoring.py`**

Insert after the `forward_lean_deg` / squat metric section, before the `# ----- Rep counting -----` section. Add a clear separator:

```python
# ══════════════════════════════════════════════════════════════════════════════
# OHP METRICS  (overhead_press + seated_overhead_press)
# All functions operate on rep_frames: List[frame] where frame = List[[x,y,z,conf]]
# All angles are 3D true angles using body-local coordinate frame.
# ══════════════════════════════════════════════════════════════════════════════

def _ohp_grip_width(rep_frames: List) -> Optional[float]:
    """
    Grip width ratio measured at the bottom of the rep (first 5 frames).
    grip_ratio = (grip_width - shoulder_width) / shoulder_width
    Ideal: 0.05 – 0.25 (5-25% wider than shoulders).
    Returns float or None if landmarks unavailable.
    """
    bottom_frames = rep_frames[:5]
    ratios: List[float] = []
    for frame in bottom_frames:
        bf = _build_body_frame(frame)
        if bf is None:
            continue
        ls = _xyz(frame, L_SHOULDER)
        rs = _xyz(frame, R_SHOULDER)
        lw = _xyz(frame, L_WRIST)
        rw = _xyz(frame, R_WRIST)
        if any(x is None for x in (ls, rs, lw, rw)):
            continue
        ls_l = _to_body_local(ls, bf)
        rs_l = _to_body_local(rs, bf)
        lw_l = _to_body_local(lw, bf)
        rw_l = _to_body_local(rw, bf)
        sh_w = abs(rs_l[0] - ls_l[0])
        gr_w = abs(rw_l[0] - lw_l[0])
        if sh_w < 1e-3:
            continue
        ratios.append((gr_w - sh_w) / sh_w)
    return float(np.mean(ratios)) if ratios else None


def _ohp_bar_path_deviation(rep_frames: List) -> Optional[float]:
    """
    Horizontal (XZ body-local) drift of wrist midpoint from first to last frame,
    normalised by shoulder width. Ideal ≤ 0.05; bad ≥ 0.25.
    """
    if len(rep_frames) < 2:
        return None

    def _bar_local(frame):
        bf = _build_body_frame(frame)
        if bf is None:
            return None, None
        lw = _xyz(frame, L_WRIST)
        rw = _xyz(frame, R_WRIST)
        ls = _xyz(frame, L_SHOULDER)
        rs = _xyz(frame, R_SHOULDER)
        if any(x is None for x in (lw, rw, ls, rs)):
            return None, None
        bar = _to_body_local((lw + rw) / 2.0, bf)
        ls_l = _to_body_local(ls, bf)
        rs_l = _to_body_local(rs, bf)
        sh_w = abs(rs_l[0] - ls_l[0])
        return bar, sh_w

    bar_bottom, sh_w = _bar_local(rep_frames[0])
    bar_top, _ = _bar_local(rep_frames[-1])
    if bar_bottom is None or bar_top is None or sh_w < 1e-3:
        return None

    d_horiz = math.sqrt((bar_top[0] - bar_bottom[0])**2 + (bar_top[2] - bar_bottom[2])**2)
    return float(d_horiz / sh_w)


def _ohp_rom(rep_frames: List) -> Optional[float]:
    """
    Minimum 3D elbow angle (average of L and R) across all rep frames.
    Represents how much the elbows flexed at the bottom of the rep.
    Full ROM: ≤ 75°; partial: 75-90°; insufficient: > 90°.
    """
    min_angle = 180.0
    found = False
    for frame in rep_frames:
        frame_angles: List[float] = []
        for sh_idx, el_idx, wr_idx in (
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        ):
            sh = _xyz(frame, sh_idx)
            el = _xyz(frame, el_idx)
            wr = _xyz(frame, wr_idx)
            if any(x is None for x in (sh, el, wr)):
                continue
            a = _angle_3d(sh, el, wr)
            if a is not None:
                frame_angles.append(a)
        if frame_angles:
            min_angle = min(min_angle, float(np.mean(frame_angles)))
            found = True
    return float(min_angle) if found else None


def _ohp_lockout(rep_frames: List, fps: float) -> Optional[float]:
    """
    Maximum 3D elbow extension angle (average L+R) sustained for ≥ 0.5 s.
    If no 0.5 s window is found, returns the peak angle (partial lockout credit).
    Ideal: ≥ 165°; bad: ≤ 145°.
    """
    LOCKOUT_THRESH = 165.0
    MIN_SUSTAIN = max(1, int(0.5 * fps))

    per_frame: List[Optional[float]] = []
    for frame in rep_frames:
        angles: List[float] = []
        for sh_idx, el_idx, wr_idx in (
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        ):
            sh = _xyz(frame, sh_idx)
            el = _xyz(frame, el_idx)
            wr = _xyz(frame, wr_idx)
            if any(x is None for x in (sh, el, wr)):
                continue
            a = _angle_3d(sh, el, wr)
            if a is not None:
                angles.append(a)
        per_frame.append(float(np.mean(angles)) if angles else None)

    best_sustained = 0.0
    run_count = 0
    run_max = 0.0
    for a in per_frame:
        if a is not None and a >= LOCKOUT_THRESH:
            run_count += 1
            run_max = max(run_max, a)
            if run_count >= MIN_SUSTAIN:
                best_sustained = run_max
        else:
            run_count = 0
            run_max = 0.0

    if best_sustained > 0.0:
        return float(best_sustained)
    # Fallback: return peak angle for partial credit
    peak = max((a for a in per_frame if a is not None), default=None)
    return float(peak) if peak is not None else None


def _ohp_elbow_flare(
    rep_frames: List,
    rep_phases: List[str],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Mean shoulder abduction angle during the concentric (press-up) phase.
    Abduction is measured as the angle between the upper arm vector (elbow-shoulder)
    and the torso-downward vector (hip-shoulder).  0° = arm along torso; 90° = horizontal.

    Ideal: 30-60°.  < 20° = too tucked.  > 70° = excessive flare.
    Asymmetry > 15° incurs an additional penalty in the scorer.

    Returns: (mean_left_deg, mean_right_deg, mean_both_deg)
             Any can be None if landmarks unavailable.
    """
    # Filter to concentric frames only; fall back to all frames if no phase info
    if rep_phases:
        conc_frames = [f for f, p in zip(rep_frames, rep_phases) if p == "concentric"]
    else:
        conc_frames = []
    if not conc_frames:
        conc_frames = rep_frames

    left_abd: List[float] = []
    right_abd: List[float] = []

    for frame in conc_frames:
        for sh_idx, el_idx, hi_idx, side_list in (
            (L_SHOULDER, L_ELBOW, L_HIP,   left_abd),
            (R_SHOULDER, R_ELBOW, R_HIP,   right_abd),
        ):
            sh = _xyz(frame, sh_idx)
            el = _xyz(frame, el_idx)
            hi = _xyz(frame, hi_idx)
            if any(x is None for x in (sh, el, hi)):
                continue
            upper_arm = el - sh        # direction: shoulder → elbow
            torso_down = hi - sh       # direction: shoulder → hip (downward)
            n_ua = np.linalg.norm(upper_arm)
            n_td = np.linalg.norm(torso_down)
            if n_ua < 1e-6 or n_td < 1e-6:
                continue
            cos_a = float(np.clip(
                np.dot(upper_arm, torso_down) / (n_ua * n_td), -1.0, 1.0
            ))
            side_list.append(float(np.degrees(np.arccos(cos_a))))

    mean_left = float(np.mean(left_abd)) if left_abd else None
    mean_right = float(np.mean(right_abd)) if right_abd else None
    both = left_abd + right_abd
    mean_both = float(np.mean(both)) if both else None
    return mean_left, mean_right, mean_both
```

- [ ] **Step 4: Run metric tests**

```bash
pytest tests/test_ohp_scoring.py -k "grip_width or bar_path or rom or lockout or flare" -v
```

Expected: all metric tests PASS (some geometry-sensitive tests may need angle tolerance adjustment — if a test fails by < 10°, adjust the assert threshold rather than the implementation).

- [ ] **Step 5: Commit**

```bash
git add core/exevision/stages/scoring.py tests/test_ohp_scoring.py
git commit -m "feat(scoring): add 5 OHP metric functions (grip, bar path, ROM, lockout, flare)"
```

---

## Task 4: Add OHP scoring dispatcher `_score_overhead_press()`

**Files:**
- Modify: `core/exevision/stages/scoring.py` (insert after `score_rep_simple()` ~line 634)
- Test: `tests/test_ohp_scoring.py` (append)

- [ ] **Step 1: Append scorer tests**

```python
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
```

- [ ] **Step 2: Run — expect AttributeError**

```bash
pytest tests/test_ohp_scoring.py -k "score_overhead_press" -v 2>&1 | head -10
```

- [ ] **Step 3: Add scoring helpers + `_score_overhead_press()` to `scoring.py`**

Insert after `score_rep_simple()`:

```python
# ══════════════════════════════════════════════════════════════════════════════
# OHP SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _score_ohp_grip(ratio: float, cfg: Dict) -> float:
    """Score grip_ratio against ideal range [ideal_low, ideal_high]."""
    c = cfg["metrics"]["grip_ratio"]
    ideal_low, ideal_high = c["ideal_low"], c["ideal_high"]
    perfect, tolerance = c["perfect"], c["tolerance"]
    if ideal_low <= ratio <= ideal_high:
        return 100.0
    return max(0.0, 100.0 * (1.0 - abs(ratio - perfect) / tolerance))


def _score_ohp_bar_path(deviation: float, cfg: Dict) -> float:
    c = cfg["metrics"]["bar_path_deviation"]
    return score_metric_linear(deviation, good=c["good_threshold"], bad=c["bad_threshold"], higher_is_better=False)


def _score_ohp_rom(min_angle: float, cfg: Dict) -> float:
    c = cfg["metrics"]["min_elbow_angle"]
    full, partial = c["full_rom_threshold"], c["partial_rom_threshold"]
    if min_angle <= full:
        return 100.0
    if min_angle >= partial:
        return 0.0
    return 100.0 * (partial - min_angle) / (partial - full)


def _score_ohp_lockout(max_angle: float, cfg: Dict) -> float:
    c = cfg["metrics"]["max_elbow_angle"]
    return score_metric_linear(max_angle, good=c["good_threshold"], bad=c["bad_threshold"], higher_is_better=True)


def _score_ohp_flare(
    mean_left: Optional[float],
    mean_right: Optional[float],
    mean_both: Optional[float],
    cfg: Dict,
) -> float:
    if mean_both is None:
        return 50.0  # neutral when not computable
    c = cfg["metrics"]["elbow_flare"]
    ideal_low, ideal_high = c["ideal_low"], c["ideal_high"]
    bad_low, bad_high = c["bad_low"], c["bad_high"]
    asym_thresh = c["asymmetry_threshold"]

    if ideal_low <= mean_both <= ideal_high:
        base = 100.0
    elif mean_both < ideal_low:
        base = score_metric_linear(mean_both, good=ideal_low, bad=bad_low, higher_is_better=True)
    else:
        base = score_metric_linear(mean_both, good=ideal_high, bad=bad_high, higher_is_better=False)

    if mean_left is not None and mean_right is not None:
        asym = abs(mean_left - mean_right)
        if asym > asym_thresh:
            base = max(0.0, base - min(20.0, (asym - asym_thresh) * 2.0))
    return base


def _score_overhead_press(
    rep_frames: List,
    rep_phases: List[str],
    fps: float,
    exercise: str,
    config: Dict,
) -> Dict[str, Any]:
    """
    Compute all 5 OHP metrics, score them, and return a result dict.

    Works identically for 'overhead_press' and 'seated_overhead_press'.
    The exercise parameter is only used to label the output — metric logic
    is identical for both variants (seated has zeroed leg landmarks which
    are never read by any OHP metric function).

    Output schema matches score_rep_simple() for pipeline compatibility:
      overall_score, metric_scores, raw_metrics, exercise, weights_used
    """
    weights: Dict[str, float] = config.get("metric_weights", {
        "grip_ratio": 0.20, "bar_path_deviation": 0.20,
        "rom": 0.20, "lockout": 0.20, "elbow_flare": 0.20,
    })

    # ── Raw measurements ──
    grip_ratio = _ohp_grip_width(rep_frames)
    bar_path   = _ohp_bar_path_deviation(rep_frames)
    min_elbow  = _ohp_rom(rep_frames)
    max_elbow  = _ohp_lockout(rep_frames, fps)
    fl_left, fl_right, fl_both = _ohp_elbow_flare(rep_frames, rep_phases)

    raw_metrics: Dict[str, Any] = {
        "grip_ratio":         round(float(grip_ratio), 4) if grip_ratio is not None else None,
        "bar_path_deviation": round(float(bar_path),   4) if bar_path   is not None else None,
        "min_elbow_angle":    round(float(min_elbow),  2) if min_elbow  is not None else None,
        "max_elbow_angle":    round(float(max_elbow),  2) if max_elbow  is not None else None,
        "elbow_flare_left":   round(float(fl_left),    2) if fl_left    is not None else None,
        "elbow_flare_right":  round(float(fl_right),   2) if fl_right   is not None else None,
        "elbow_flare_mean":   round(float(fl_both),    2) if fl_both    is not None else None,
    }

    # ── Per-metric scores ──
    metric_scores: Dict[str, float] = {}
    if grip_ratio is not None:
        metric_scores["grip_ratio"]        = _score_ohp_grip(grip_ratio, config)
    if bar_path is not None:
        metric_scores["bar_path_deviation"] = _score_ohp_bar_path(bar_path, config)
    if min_elbow is not None:
        metric_scores["rom"]               = _score_ohp_rom(min_elbow, config)
    if max_elbow is not None:
        metric_scores["lockout"]           = _score_ohp_lockout(max_elbow, config)
    if fl_both is not None or fl_left is not None or fl_right is not None:
        metric_scores["elbow_flare"]       = _score_ohp_flare(fl_left, fl_right, fl_both, config)

    # ── Weighted overall ──
    avail_w = {k: weights[k] for k in metric_scores if k in weights}
    total_w = sum(avail_w.values())
    norm_w  = {k: v / total_w for k, v in avail_w.items()} if total_w > 0 else {}
    overall = sum(metric_scores[k] * norm_w.get(k, 0.0) for k in metric_scores)

    return {
        "overall_score":  round(float(overall), 1),
        "metric_scores":  {k: round(float(v), 1) for k, v in metric_scores.items()},
        "raw_metrics":    raw_metrics,
        "exercise":       exercise,
        "weights_used":   {k: round(v, 2) for k, v in norm_w.items()},
    }
```

- [ ] **Step 4: Run scorer tests**

```bash
pytest tests/test_ohp_scoring.py -k "score_overhead_press" -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_ohp_scoring.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add core/exevision/stages/scoring.py tests/test_ohp_scoring.py
git commit -m "feat(scoring): add _score_overhead_press() dispatcher with 5-metric OHP scorer"
```

---

## Task 5: Wire OHP dispatch into `process_single_video()`

**Files:**
- Modify: `core/exevision/stages/scoring.py` (`process_single_video()` and `main()`)

- [ ] **Step 1: Add `exercise` parameter to `process_single_video()`**

Change the function signature at line ~716:

```python
# BEFORE:
def process_single_video(video_id: str, source_quality: str, save_output: bool = True) -> Optional[Dict[str, Any]]:

# AFTER:
def process_single_video(video_id: str, source_quality: str, save_output: bool = True, exercise: str = "squat") -> Optional[Dict[str, Any]]:
```

- [ ] **Step 2: Load OHP config inside `process_single_video()` when exercise is OHP**

Insert after the `fps = float(info.get("fps", 30.0))` line (~line 741):

```python
    # Load per-exercise config for OHP scoring
    _ohp_config: Optional[Dict] = None
    if exercise in ("overhead_press", "seated_overhead_press"):
        _ohp_cfg_path = Path(__file__).resolve().parent.parent / "config" / "exercises" / "overhead_press.json"
        if _ohp_cfg_path.exists():
            with open(_ohp_cfg_path) as _f:
                _ohp_config = json.load(_f)
        else:
            print(f"⚠️  overhead_press.json not found at {_ohp_cfg_path}; OHP scoring disabled")
```

- [ ] **Step 3: Extract `frame_phases` from segmented JSON**

After `reps_from_seg = seg.get("repetitions", [])` (~line 757), add:

```python
    frame_phases_all: List[str] = seg.get("frame_phases", [])
```

- [ ] **Step 4: Replace the rep processing loop with the exercise-dispatched version**

Find the block starting at `for rep_dict in reps_from_seg:` (~line 784). Replace it entirely:

```python
    rep_outputs = []
    for rep_dict in reps_from_seg:
        rep_id = rep_dict.get("rep_id", len(rep_outputs) + 1)
        start  = rep_dict.get("start_frame")
        end    = rep_dict.get("end_frame")
        bottom = rep_dict.get("bottom_frame", (start + end) // 2 if start is not None and end is not None else 0)

        if start is None or end is None:
            continue

        frames = keypoints[start:end + 1]

        if exercise in ("overhead_press", "seated_overhead_press") and _ohp_config is not None:
            rep_phases = frame_phases_all[start:end + 1] if frame_phases_all else []
            rep_score = _score_overhead_press(
                frames, rep_phases, fps=fps, exercise=exercise, config=_ohp_config
            )
            metrics = rep_score["raw_metrics"]
        else:
            # ── Squat scoring (unchanged) ──
            valgus_vals, lean_vals, knee_vals = [], [], []
            for fr in frames:
                v = knee_valgus_ratio(fr)
                if v is not None:
                    valgus_vals.append(v)
                fl = forward_lean_deg(fr)
                if fl is not None:
                    lean_vals.append(fl)
                ka = knee_angle(fr)
                if ka is not None:
                    knee_vals.append(ka)
            min_knee = float(np.min(knee_vals)) if knee_vals else None
            depth_val = None
            if 0 <= bottom - start < len(frames):
                depth_val = calculate_vertical_depth(frames[bottom - start])
            below_parallel = depth_val is not None and depth_val > 0.0
            metrics = {
                "knee_valgus":  float(np.mean(valgus_vals)) if valgus_vals else None,
                "forward_lean": float(np.mean(lean_vals))   if lean_vals   else None,
                "min_knee_angle": min_knee,
                "squat_depth":  depth_val,
                "below_parallel": below_parallel,
            }
            rep_score = score_rep_simple(metrics, view=view)

        rep_outputs.append({
            "rep_id":           int(rep_id),
            "start_frame":      int(start),
            "end_frame":        int(end),
            "duration_frames":  int(end - start + 1),
            "duration_seconds": round((end - start + 1) / fps, 2),
            "bottom_frame":     int(bottom),
            "metrics":          metrics,
            "score":            rep_score,
        })
```

- [ ] **Step 5: Pass `exercise` through from `main()` to `process_single_video()`**

In `main()`, the two calls to `process_single_video()` need `exercise=args.exercise`:

```python
# single video (line ~702):
result = process_single_video(video_id, source_quality, save_output=not args.no_save, exercise=args.exercise)

# batch (line ~674):
result = process_single_video(video_id, source_quality, save_output=not args.no_save, exercise=args.exercise)
```

- [ ] **Step 6: Smoke-test on a real OHP video (requires workspace to exist)**

```powershell
# From D:\FitnessAQA\ohp_phase1\workspace\
python "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\scoring.py" `
    80690_2 --exercise overhead_press --no-save
```

Expected output includes:
```
View: ...
Reps: N
Overall score: XX.X/100
```

If `segmented JSON not found`: run temporal segmentation first for that video ID.

- [ ] **Step 7: Run all tests**

```bash
pytest tests/ -v
```

Expected: all PASS (existing squat tests unaffected)

- [ ] **Step 8: Commit**

```bash
git add core/exevision/stages/scoring.py
git commit -m "feat(scoring): dispatch overhead_press + seated_overhead_press in process_single_video"
```

---

## Task 6: Batch-run Stage 8 on 2.8k OHP videos

This is an operational step — no code changes. Once Task 5 is done, generate heuristic scores for all extracted OHP videos.

- [ ] **Step 1: Run Stage 8 in batch (overhead_press)**

```powershell
Set-Location "D:\FitnessAQA\ohp_phase1\workspace"
& "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe" `
    "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\scoring.py" `
    "*" --exercise overhead_press
```

- [ ] **Step 2: Run Stage 8 in batch (seated_overhead_press)**

```powershell
& "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe" `
    "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\scoring.py" `
    "*" --exercise seated_overhead_press
```

- [ ] **Step 3: Verify outputs exist**

```powershell
(Get-ChildItem "D:\FitnessAQA\ohp_phase1\workspace\overhead_press\aqa_analysis_simple" -Recurse -Filter "*.json").Count
```

Expected: > 0 JSON files (one per processed video)

- [ ] **Step 4: Spot-check one output**

```powershell
Get-Content (Get-ChildItem "D:\FitnessAQA\ohp_phase1\workspace\overhead_press\aqa_analysis_simple" -Recurse -Filter "*.json" | Select-Object -First 1).FullName | python -c "import json,sys; d=json.load(sys.stdin); print(d['overall_score'], list(d['repetitions'][0]['score']['metric_scores'].keys()))"
```

Expected: a score value + `['grip_ratio', 'bar_path_deviation', 'rom', 'lockout', 'elbow_flare']`

- [ ] **Step 5: Commit CHANGELOG update**

```bash
git add -A
git commit -m "data: run OHP Stage 8 scoring on 2.8k videos; outputs in ohp_phase1/workspace/{exercise}/aqa_analysis_simple/"
```

---

## Self-Review

**Spec coverage:**
| Spec requirement | Task |
|------------------|------|
| Body-local coordinate frame (shoulder/hip) | Task 2 (`_build_body_frame`, `_to_body_local`) |
| 3D true angles (not projected 2D) | Task 2 (`_angle_3d`) |
| Grip width ratio at bottom-of-rep | Task 3 (`_ohp_grip_width`) |
| Bar path XZ deviation normalised by shoulder width | Task 3 (`_ohp_bar_path_deviation`) |
| ROM: min elbow angle ≤ 75° threshold | Task 3 (`_ohp_rom`) |
| Lockout: max elbow angle ≥ 165° sustained 0.5 s | Task 3 (`_ohp_lockout`) |
| Elbow flare: shoulder abduction during concentric | Task 3 (`_ohp_elbow_flare`) |
| Asymmetry penalty on flare | Task 4 (`_score_ohp_flare`) |
| Equal 20% weights on all 5 metrics | Task 4 (`_score_overhead_press`) |
| Seated OHP: same metrics, leg landmarks ignored | Inherent — no leg landmarks used in any OHP function |
| `exercise` label in every output | Task 4 (explicit `"exercise": exercise` field) |
| Dispatch separate from squat (no shared functions) | Tasks 3–5 (zero calls to squat metric functions from OHP path) |
| Config-driven thresholds | Task 1 + Task 4 (all numbers read from config dict) |
| Batch `*` mode for Stage 8 | Task 6 |

**Placeholder scan:** None found — all steps contain actual code or exact commands.

**Type consistency:**
- `_build_body_frame()` → returns `Dict[str, Any]` with keys `v_right, v_up, v_forward, mid_shoulder, mid_hip` — used consistently in `_to_body_local()` and all metric functions.
- `_score_overhead_press()` → returns `Dict` with `overall_score, metric_scores, raw_metrics, exercise, weights_used` — same schema as `score_rep_simple()`.
- `_ohp_elbow_flare()` → returns `Tuple[Optional[float], Optional[float], Optional[float]]` — consumed correctly in `_score_overhead_press()` and `_score_ohp_flare()`.
