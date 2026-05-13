# OHP Phase 3 — Multi-Task Manual Annotation & Fine-Tuning

**Created:** 2026-05-11 | **Branch:** `multiexercise` | **Status:** Approved — implement exactly as written

---

## BEFORE YOU START — READ THIS SECTION COMPLETELY

### What this plan does

Extends the OHP neural pipeline (Phase 2) with multi-task heads trained on ~250 manually annotated reps. Phase 2 predicts `quality + knee_error`. Phase 3 adds `smoothness`, `control`, `lockout`, `elbow_flare`, `grip_ratio`, `rom_top`, `rom_bottom`. Phase 3 trains purely on human annotations (not heuristic-derived labels). Phase 2 knee head is kept frozen — no re-annotation.

### Hard rules — violation stops the task

1. **DO NOT modify** `core/exevision/neural/nn_models.py` (squat BiLSTM/STGCN). Squat models are untouched.
2. **DO NOT modify** `core/exevision/training/finetune_models.py` (squat training). Untouched.
3. **DO NOT modify** any squat annotation saving logic in `apps/desktop-ui/app.py` lines 3768–3816.
4. All new OHP neural code goes under `core/exevision/neural/ohp/` only.
5. All new OHP training code goes under `core/exevision/training/ohp/` only.
6. `engine.py` gets NO exercise-specific branches — cues come from config JSON only.
7. If you are about to touch a file not listed in the Build Order, STOP and re-read this plan.

### Pre-flight — verify these before writing any code

Run these checks first:

```powershell
# 1. Phase 2 checkpoints exist
ls models\bilstm_ohp_phase2.pt, models\stgcn_ohp_phase2.pt, models\fusion_ohp_phase2.pt

# 2. Current OHP model has exactly quality_head + knee_error_head (no Phase 3 heads yet)
Select-String "def forward" core\exevision\neural\ohp\models.py

# 3. Current metric widget is hardcoded squat keys (not config-driven)
Select-String "metric_defs" apps\desktop-ui\app.py

# 4. Confirm inference.py bug: writes neural_scores not neural_analysis
Select-String "neural_scores" core\exevision\neural\ohp\inference.py
```

Expected: all four commands find exactly what's described. If any deviates, investigate before proceeding.

### Annotation schema convention

Phase 3 uses **existing rep-field keys** — no `manual_*` prefix invented:

| What you're annotating | Saved under | Type |
|---|---|---|
| Overall quality | `rep["human_score"]` | float 0-100 |
| Smoothness, control, elbow_flare, grip_ratio, rom_top, rom_bottom | `rep["human_metric_scores"]["<key>"]` | float 0-100 or `None` |
| Lockout (binary) | `rep["human_flags"]["lockout"]` | bool |
| Confidence | `rep["annotator_confidence"]` | int 1-5 |
| Notes | `rep["annotation_notes"]` | str |

`None` in `human_metric_scores` → `float('nan')` in PyTorch tensor → masked out of loss (NaN-safe masked MSE). Only `grip_ratio` is ever `None` (side view).

Phase 3 annotations write to `training_dataset/ohp_phase3_annotations/videos/{video_id}.json` — NOT the squat dir `training_dataset/annotations/videos/`.

### Existing bug (fix in Step 19a before any Phase 3 inference)

`core/exevision/neural/ohp/inference.py` line 117 writes to `{exercise}/neural_scores/...`.
`apps/api/pipeline.py` line 630 reads from `{exercise}/neural_analysis/...`.
Result: OHP Phase 2 neural outputs are silently unreachable by the API today. Fix in Step 19a.

---

## Build Order

Steps 1–4 must complete before Step 6. Steps 2, 3, 5, 8, 9, 12 are parallelisable.
Manual steps 13–15 block training (16–18). Step 19a must precede 19b.

| Step | File | Type |
|------|------|------|
| 1 | `core/exevision/config/exercises/overhead_press.json` | EDIT |
| 2 | `core/exevision/utils/skeleton_overlay.py` | NEW |
| 3 | `core/exevision/neural/registry.py` | NEW |
| 4 | `core/exevision/neural/ohp/models.py` | EDIT |
| 5 | `core/exevision/analysis/select_ohp_phase3_samples.py` | NEW |
| 6 | `apps/desktop-ui/annotation_overhead_press.py` | NEW |
| 7 | `apps/desktop-ui/app.py` | EDIT (minimal — dispatcher only) |
| 8 | `core/exevision/training/ohp/data_phase3.py` | NEW |
| 9 | `core/exevision/training/ohp/losses.py` | NEW |
| 10 | `core/exevision/training/ohp/finetune_phase3.py` | NEW |
| 11 | `core/exevision/training/ohp/evaluate_phase3.py` | NEW |
| 12 | `core/exevision/training/ohp/tta.py` | NEW |
| 13–15 | Manual steps (sample selection, annotation, ICC check) | MANUAL |
| 16–18 | Training scripts | SCRIPT RUN |
| 19a | `core/exevision/neural/ohp/inference.py` | BUG FIX |
| 19b | `core/exevision/stages/neural_fusion_inference.py` | EDIT |
| 20 | `core/exevision/feedback/engine.py` | EDIT |
| 21 | `CLAUDE.md` + `CHANGELOG.md` | DOCS |

---

## Step 1 — Config: overhead_press.json

**File:** `core/exevision/config/exercises/overhead_press.json`  
**Pre-condition:** File exists. Current keys include `annotation_flags`, `annotation_metrics`, `issue_groups`.  
**Action:** Add three new top-level keys to the JSON object. Do NOT modify any existing keys.

Add at end of the JSON object (before closing `}`):

```json
  "annotation_metrics_phase3": {
    "smoothness":  "Smoothness",
    "control":     "Control",
    "elbow_flare": "Elbow Flare (high=clean)",
    "grip_ratio":  "Grip Width",
    "rom_top":     "ROM Top Extension",
    "rom_bottom":  "ROM Bottom Depth"
  },
  "annotation_binary_phase3": {
    "lockout": "Full Lockout Achieved"
  },
  "annotation_grip_null_views": ["side"],
  "error_cues_phase3": {
    "lockout":     "Press fully overhead — finish each rep with elbows locked.",
    "elbow_flare": "Elbows flared wide — keep them under the bar.",
    "smoothness":  "Press tempo was jerky — drive smoothly through the movement.",
    "control":     "Bar wandered — control the path top to bottom.",
    "grip_ratio":  "Grip width is off — aim for just outside shoulder width.",
    "rom_top":     "Incomplete top extension — press to full lockout overhead.",
    "rom_bottom":  "Incomplete bottom depth — lower the bar to chin/upper chest level.",
    "knee_error":  "Knees buckled during the press — brace and stabilise your stance."
  }
```

**Verification:** `python -c "import json; d=json.load(open('core/exevision/config/exercises/overhead_press.json')); assert 'annotation_metrics_phase3' in d and 'error_cues_phase3' in d; print('OK')`

---

## Step 2 — New: skeleton_overlay.py

**File:** `core/exevision/utils/skeleton_overlay.py`  
**Pre-condition:** `core/exevision/utils/` directory exists.  
**Purpose:** Exercise-agnostic skeleton draw utility. Reads cached keypoints from features JSON — NO MediaPipe re-run.

```python
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# OHP active joint indices (matches nn_utils.OHP_ACTIVE_JOINTS ordering)
OHP_CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),             # shoulders
    (11, 23), (12, 24),  # torso sides
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]


def draw_skeleton(
    frame: np.ndarray,
    keypoints_xy: List[Tuple[float, float]],  # (x_norm, y_norm) per landmark, len=33
    confidences: Optional[List[float]] = None,
    connections: Optional[List[Tuple[int, int]]] = None,
    conf_threshold: float = 0.4,
    point_radius: int = 4,
    line_thickness: int = 2,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    line_color: Tuple[int, int, int] = (0, 200, 255),
    low_conf_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Draw skeleton on frame from cached normalised keypoints.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        keypoints_xy: List of (x_norm, y_norm) in [0, 1] for each of 33 MediaPipe landmarks.
        confidences: Visibility scores [0, 1] per landmark. If None, all drawn.
        connections: List of (idx_a, idx_b) pairs. Defaults to OHP_CONNECTIONS.
        conf_threshold: Landmarks below this visibility are drawn in low_conf_color.

    Returns:
        Annotated frame (copy, original unchanged).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    conns = connections if connections is not None else OHP_CONNECTIONS
    confs = confidences if confidences is not None else [1.0] * len(keypoints_xy)

    # Draw connections
    for a, b in conns:
        if a >= len(keypoints_xy) or b >= len(keypoints_xy):
            continue
        xa, ya = keypoints_xy[a]
        xb, yb = keypoints_xy[b]
        ca = confs[a] if a < len(confs) else 1.0
        cb = confs[b] if b < len(confs) else 1.0
        if ca < conf_threshold or cb < conf_threshold:
            color = low_conf_color
        else:
            color = line_color
        pt_a = (int(xa * w), int(ya * h))
        pt_b = (int(xb * w), int(yb * h))
        cv2.line(out, pt_a, pt_b, color, line_thickness)

    # Draw points
    for i, (x, y) in enumerate(keypoints_xy):
        c = confs[i] if i < len(confs) else 1.0
        color = point_color if c >= conf_threshold else low_conf_color
        cv2.circle(out, (int(x * w), int(y * h)), point_radius, color, -1)

    return out


def extract_keypoints_from_frame(frame_data: dict) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Extract (keypoints_xy, confidences) from a single frame dict in features JSON.

    Features JSON frame format: {"landmarks": [{"x": ..., "y": ..., "visibility": ...}, ...]}
    Returns normalised (x, y) pairs and visibility scores for all 33 landmarks.
    """
    landmarks = frame_data.get("landmarks") or []
    xy = [(float(lm.get("x", 0)), float(lm.get("y", 0))) for lm in landmarks]
    conf = [float(lm.get("visibility", 0)) for lm in landmarks]
    return xy, conf
```

**Verification:** `python -c "from core.exevision.utils.skeleton_overlay import draw_skeleton, extract_keypoints_from_frame; print('OK')"`

---

## Step 3 — New: registry.py

**File:** `core/exevision/neural/registry.py`  
**Pre-condition:** `core/exevision/neural/ohp/models.py` exists with `OHPBiLSTMScorer` and `OHPSTGCNScorer`. Step 4 extends these classes — registry.py can be written now but only fully functional after Step 4.

```python
from __future__ import annotations

from typing import Any, Dict, Type


def _lazy_squat():
    from core.exevision.neural.nn_models import BiLSTMScorer, STGCNScorer
    return {"bilstm": BiLSTMScorer, "stgcn": STGCNScorer}


def _lazy_ohp():
    from core.exevision.neural.ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
    return {"bilstm": OHPBiLSTMScorer, "stgcn": OHPSTGCNScorer}


# Registry maps exercise name → {"bilstm": class, "stgcn": class}
# To add a new exercise: add one entry here + create neural/<exercise>/models.py
_REGISTRY_FACTORIES = {
    "squat":                   _lazy_squat,
    "overhead_press":          _lazy_ohp,
    "seated_overhead_press":   _lazy_ohp,  # same model; knee_error_prob suppressed at inference
}


def get_model_classes(exercise: str) -> Dict[str, Type[Any]]:
    """Return {"bilstm": Class, "stgcn": Class} for the given exercise.

    Raises KeyError for unknown exercises.
    """
    factory = _REGISTRY_FACTORIES.get(exercise)
    if factory is None:
        raise KeyError(f"No neural registry entry for exercise '{exercise}'. "
                       f"Known: {list(_REGISTRY_FACTORIES)}")
    return factory()
```

**Verification:** `python -c "from core.exevision.neural.registry import get_model_classes; print(get_model_classes('overhead_press')); print('OK')"`

---

## Step 4 — Edit: models.py (new heads + load_phase2_for_phase3)

**File:** `core/exevision/neural/ohp/models.py`  
**Pre-condition:** Read the file first. Current `OHPBiLSTMScorer.__init__` ends around line 74 with `self.knee_error_head`. Current `OHPSTGCNScorer.__init__` ends around line 145 with `self.knee_error_head`.  
**DO NOT touch:** `_score_head()`, `_error_head()`, `load_pretrained()` — these stay exactly as-is.

### 4a — Add heads to OHPBiLSTMScorer.__init__

Find this block in `OHPBiLSTMScorer.__init__` (currently the last two lines before `def encode`):
```python
        embed_dim = hidden_dim * 2
        self.quality_head = _score_head(embed_dim)
        self.knee_error_head = _error_head(embed_dim)
```

Replace with:
```python
        embed_dim = hidden_dim * 2
        self.quality_head = _score_head(embed_dim)
        self.knee_error_head = _error_head(embed_dim)
        # Phase 3 temporal heads
        self.smoothness_head = _score_head(embed_dim)
        self.control_head = _score_head(embed_dim)
```

### 4b — Update OHPBiLSTMScorer.forward

Find:
```python
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(x)
        return {
            "embedding": emb,
            "quality": self.quality_head(emb).squeeze(-1) * 100.0,
            "knee_error": self.knee_error_head(emb).squeeze(-1),
        }
```

Replace with:
```python
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(x)
        return {
            "embedding":  emb,
            "quality":    self.quality_head(emb).squeeze(-1) * 100.0,
            "knee_error": self.knee_error_head(emb).squeeze(-1),
            "smoothness": self.smoothness_head(emb).squeeze(-1) * 100.0,
            "control":    self.control_head(emb).squeeze(-1) * 100.0,
        }
```

### 4c — Add heads to OHPSTGCNScorer.__init__

Find in `OHPSTGCNScorer.__init__` (the last block before `def encode`):
```python
        embed_dim = 256
        self.quality_head = _score_head(embed_dim + self._VIEW_DIM)
        self.knee_error_head = _error_head(embed_dim)
```

Replace with:
```python
        embed_dim = 256
        self.quality_head = _score_head(embed_dim + self._VIEW_DIM)
        self.knee_error_head = _error_head(embed_dim)
        # Phase 3 spatial heads (embed_dim only — no view concat needed)
        self.lockout_head    = _error_head(embed_dim)
        self.elbow_flare_head = _score_head(embed_dim)
        self.grip_ratio_head  = _score_head(embed_dim)
        self.rom_top_head     = _score_head(embed_dim)
        self.rom_bottom_head  = _score_head(embed_dim)
```

### 4d — Update OHPSTGCNScorer.forward

Find:
```python
        return {
            "embedding": emb,
            "quality": self.quality_head(spatial_in).squeeze(-1) * 100.0,
            "knee_error": self.knee_error_head(emb).squeeze(-1),
        }
```

Replace with:
```python
        return {
            "embedding":   emb,
            "quality":     self.quality_head(spatial_in).squeeze(-1) * 100.0,
            "knee_error":  self.knee_error_head(emb).squeeze(-1),
            "lockout":     self.lockout_head(emb).squeeze(-1),
            "elbow_flare": self.elbow_flare_head(emb).squeeze(-1) * 100.0,
            "grip_ratio":  self.grip_ratio_head(emb).squeeze(-1) * 100.0,
            "rom_top":     self.rom_top_head(emb).squeeze(-1) * 100.0,
            "rom_bottom":  self.rom_bottom_head(emb).squeeze(-1) * 100.0,
        }
```

### 4e — Add load_phase2_for_phase3() to BOTH model classes

Add this method to `OHPBiLSTMScorer` (after `load_pretrained`, before class ends):

```python
    def load_phase2_for_phase3(self, path: str) -> None:
        """Transfer Phase 2 encoder + knee_error_head; re-init quality_head; freeze knee.

        Encoder weights (lstm1, lstm2, temporal_attention) transfer if shapes match.
        quality_head excluded — Phase 3 trains it from scratch on manual annotations.
        knee_error_head transfers and is frozen (Phase 2 labels, not re-annotated).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = self.state_dict()
        transfer = {
            k: v for k, v in state.items()
            if not k.startswith("quality_head")
            and k in model_state
            and v.shape == model_state[k].shape
        }
        self.load_state_dict(transfer, strict=False)
        for p in self.knee_error_head.parameters():
            p.requires_grad = False
```

Add the identical method to `OHPSTGCNScorer` (same body — the key filter logic applies equally).

**Verification:**
```python
python -c "
import torch
from core.exevision.neural.ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from core.exevision.neural.nn_utils import build_adjacency_matrix_ohp
import numpy as np

bilstm = OHPBiLSTMScorer()
A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32)
stgcn = OHPSTGCNScorer(A)

x_b = torch.zeros(2, 64, 8)
out_b = bilstm(x_b)
assert 'smoothness' in out_b and 'control' in out_b, 'BiLSTM missing Phase 3 heads'

x_s = torch.zeros(2, 3, 64, 10)
out_s = stgcn(x_s)
assert 'lockout' in out_s and 'elbow_flare' in out_s and 'grip_ratio' in out_s, 'STGCN missing Phase 3 heads'

print('All Phase 3 heads present. OK')
"
```

---

## Step 5 — New: select_ohp_phase3_samples.py

**File:** `core/exevision/analysis/select_ohp_phase3_samples.py`  
**Purpose:** Reads Phase 2 neural scores + heuristic scores from workspace to stratify ~250 reps for annotation.

**CLI contract:**
```
python select_ohp_phase3_samples.py \
    --neural-dir D:\FitnessAQA\ohp_phase2\workspace\overhead_press\neural_analysis \
    --aqa-dir    D:\FitnessAQA\ohp_phase2\workspace\overhead_press\aqa_analysis_simple \
    --output     phase3_target_reps.json \
    --n          250
```

**Output JSON format:**
```json
[
  {"video_id": "80830_5", "rep_id": "rep_001", "stratum": "uncertainty", "heuristic_score": 55.2},
  ...
]
```

**Stratification logic** (implement exactly):
```python
STRATA = [
    ("calibration", lambda r: r["heuristic"] < 40 or r["heuristic"] > 85, 25),
    ("uncertainty", lambda r: 40 <= r["heuristic"] <= 70, 125),
    ("error",       lambda r: r.get("knee_error_prob", 0) > 0.5 or r.get("knee_error_prob", 1) < 0.2, 63),
    ("view_balance", None, 38),   # fill remainder capped per-view at 30% of final pool
]
```

Annotation order: calibration → uncertainty → error → view_balance.

**Verification:** Script runs and produces valid JSON with keys `video_id`, `rep_id`, `stratum`.

---

## Step 6 — New: annotation_overhead_press.py

**File:** `apps/desktop-ui/annotation_overhead_press.py`  
**Purpose:** OHP Phase 3 annotation module. Completely separate from squat AnnotationToolUI. `app.py` imports and delegates to it.

**Key constants at top of file:**
```python
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
CONFIG_EXERCISES_DIR = _REPO / "core" / "exevision" / "config" / "exercises"

# Default paths — overridable via env vars
PHASE3_RAW_VIDEOS_DIR = Path(
    os.environ.get("EXEVISION_PHASE3_VIDEOS_DIR",
                   r"D:\FitnessAQA\ohp_phase3\videos")
)
PHASE3_FEATURES_DIR = Path(
    os.environ.get("EXEVISION_PHASE3_FEATURES_DIR",
                   r"D:\FitnessAQA\ohp_phase2\workspace\overhead_press\extracted_features_clean\raw_unfiltered")
)
OHP_PHASE3_ANNOTATIONS_DIR = _REPO / "training_dataset" / "ohp_phase3_annotations" / "videos"
```

**Config loading (call once at init):**
```python
def _load_ohp_config() -> dict:
    path = CONFIG_EXERCISES_DIR / "overhead_press.json"
    import json
    return json.loads(path.read_text(encoding="utf-8"))
```

**Widget layout contract** (implement in `OHPPhase3AnnotatorWindow` class):

```
Two-pane top: [raw video frame] | [skeleton overlay frame]
Rep phase strip (text label showing current phase)

Quality:      [ttk.Scale 0-100, default=50]   → human_score

─── Temporal (BiLSTM) ──
Smoothness:   [ttk.Scale 0-100, default=50]   → human_metric_scores["smoothness"]
Control:      [ttk.Scale 0-100, default=50]   → human_metric_scores["control"]

─── Spatial (ST-GCN) ───
Full Lockout: [Radiobutton Yes(1)/No(0)]       → human_flags["lockout"]
Elbow Flare:  [ttk.Scale 0-100, default=50]   → human_metric_scores["elbow_flare"]
Grip Width:   [ttk.Scale 0-100, default=50]   → human_metric_scores["grip_ratio"]
              (disabled + auto-null when view in config["annotation_grip_null_views"])
ROM Top:      [ttk.Scale 0-100, default=50]   → human_metric_scores["rom_top"]
ROM Bottom:   [ttk.Scale 0-100, default=50]   → human_metric_scores["rom_bottom"]

─── Meta ────────────────
Confidence:   [ttk.Combobox 1-5, default=3]   → annotator_confidence
Notes:        [ttk.Entry]                      → annotation_notes

[Submit] → save → reveal heuristic Δ → advance to next rep
```

**Dynamic slider build from config** (follow `_rebuild_annotation_flags()` pattern from `app.py:2311-2337`):
```python
def _build_metric_sliders(self, parent, config: dict) -> dict[str, tk.DoubleVar]:
    """Read annotation_metrics_phase3 from config, build sliders. Returns {key: DoubleVar}."""
    metric_vars = {}
    for key, label in config.get("annotation_metrics_phase3", {}).items():
        var = tk.DoubleVar(value=50.0)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Scale(row, from_=0, to=100, variable=var, orient=tk.HORIZONTAL).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        metric_vars[key] = var
    return metric_vars
```

**Save method contract:**
```python
def _on_submit(self) -> None:
    """Save current annotation to OHP_PHASE3_ANNOTATIONS_DIR, reveal heuristic."""
    rep = self._current_rep   # dict from segmentation JSON
    view = self._current_view  # str e.g. "front"

    rep["human_score"] = self._quality_var.get()
    rep["human_metric_scores"] = {
        k: (None if (k == "grip_ratio" and view in self._cfg.get("annotation_grip_null_views", []))
            else self._metric_vars[k].get())
        for k in self._metric_vars
    }
    rep["human_flags"] = {"lockout": bool(self._lockout_var.get())}
    rep["human_flag_severities"] = {}
    rep["annotator_confidence"] = int(self._confidence_var.get())
    rep["annotation_notes"] = self._notes_var.get().strip()

    self._save_annotation()   # writes to OHP_PHASE3_ANNOTATIONS_DIR / f"{video_id}.json"
    self._reveal_heuristic()
    self._advance_rep()
```

**Skeleton frame rendering** (use `skeleton_overlay.draw_skeleton`):
```python
from core.exevision.utils.skeleton_overlay import draw_skeleton, extract_keypoints_from_frame

def _render_skeleton_frame(self, frame: np.ndarray, frame_data: dict) -> np.ndarray:
    xy, conf = extract_keypoints_from_frame(frame_data)
    return draw_skeleton(frame, xy, conf)
```

**Fallback:** if features JSON not found, show warning label, disable Submit, allow skip.

**Verification:** Module imports without error: `python -c "import apps.desktop_ui.annotation_overhead_press; print('OK')"`

---

## Step 7 — Edit: app.py (dispatcher only, minimal change)

**File:** `apps/desktop-ui/app.py`  
**Scope:** Add OHP Phase 3 dispatcher. Touch NOTHING else. No squat annotation code changes.

**Find** in `AnnotationToolUI` the method or entry point that starts the annotation flow (search for `_annotation_exercise_var`). The exercise selector exists at app.py ~line 2140 and controls which exercise is being annotated.

**Add** an OHP Phase 3 launch button in the annotation tab (near the exercise selector row), wired as:

```python
def _launch_ohp_phase3_annotator(self) -> None:
    """Open OHP Phase 3 annotator window (separate from squat annotation flow)."""
    from annotation_overhead_press import OHPPhase3AnnotatorWindow
    win = tk.Toplevel(self.master)
    OHPPhase3AnnotatorWindow(win)
```

Add a button in the annotation tab UI near the exercise selector:
```python
ttk.Button(
    annotation_tab_frame,
    text="OHP Phase 3 Annotator",
    command=self._launch_ohp_phase3_annotator,
).pack(...)
```

**Constraint check:** After editing, verify squat annotation still works by searching for `_annotation_exercise_var` — must still exist and be unchanged.

**Verification:** `python apps/desktop-ui/app.py` starts without import error. OHP Phase 3 button appears.

---

## Step 8 — New: data_phase3.py

**File:** `core/exevision/training/ohp/data_phase3.py`  
**Purpose:** Dataset loader for Phase 3 manual annotations. Mirrors `data.py` structure but reads from `ohp_phase3_annotations/`.

**Class contract:**

```python
class OHPPhase3Dataset(torch.utils.data.Dataset):
    """Reads OHP Phase 3 manual annotations.

    Each item returns dict with keys:
      bilstm_input   : float32 (FIXED_SEQ_LEN, NUM_OHP_BILSTM_CHANNELS)
      stgcn_input    : float32 (STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_OHP_ACTIVE_JOINTS)
      heuristic_vec  : float32 (16,)
      view_vec       : float32 (5,)
      quality        : float32 scalar  [0-100]
      smoothness     : float32 scalar  [0-100 or nan]
      control        : float32 scalar  [0-100 or nan]
      lockout        : float32 scalar  [0.0 or 1.0]
      elbow_flare    : float32 scalar  [0-100 or nan]
      grip_ratio     : float32 scalar  [0-100 or nan]  (nan for side view)
      rom_top        : float32 scalar  [0-100 or nan]
      rom_bottom     : float32 scalar  [0-100 or nan]
      view           : str
    """
```

**Label extraction from rep dict:**
```python
def _extract_labels(rep: dict) -> dict:
    hms = rep.get("human_metric_scores") or {}
    flags = rep.get("human_flags") or {}

    def _to_tensor(val) -> torch.Tensor:
        return torch.tensor(float(val) if val is not None else float("nan"), dtype=torch.float32)

    return {
        "quality":     torch.tensor(float(rep.get("human_score", 50.0)), dtype=torch.float32),
        "smoothness":  _to_tensor(hms.get("smoothness")),
        "control":     _to_tensor(hms.get("control")),
        "lockout":     torch.tensor(1.0 if flags.get("lockout") else 0.0, dtype=torch.float32),
        "elbow_flare": _to_tensor(hms.get("elbow_flare")),
        "grip_ratio":  _to_tensor(hms.get("grip_ratio")),
        "rom_top":     _to_tensor(hms.get("rom_top")),
        "rom_bottom":  _to_tensor(hms.get("rom_bottom")),
    }
```

**Annotation path:** Read JSONs from `training_dataset/ohp_phase3_annotations/videos/*.json`.  
**Feature path:** Use `feat_path` and `seg_path` stored in annotation JSON (same structure as Phase 2 `data.py`).  
**Split:** Annotations include `fitnessaqa_split` field (train/val/test) set by sample selection script.

**build_phase3_dataloaders()** function — mirrors `build_dataloaders()` from `data.py`:
```python
def build_phase3_dataloaders(annotation_dir: Path, batch_size: int = 16, num_workers: int = 0) -> dict:
    ...
```

**Verification:** `python -c "from core.exevision.training.ohp.data_phase3 import OHPPhase3Dataset; print('OK')"`

---

## Step 9 — New: losses.py

**File:** `core/exevision/training/ohp/losses.py`

```python
from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss ignoring NaN targets. Returns 0.0 if all targets are NaN."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred[mask], target[mask])


def weighted_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    neg_weight: float = 4.0,
) -> torch.Tensor:
    """BCE with higher weight on negative class (majority for lockout).

    neg_weight = n_full_lockout / n_incomplete. Default 4.0 until computed from pool.
    Label smoothing: 0 → 0.1, 1 → 0.9.
    """
    smoothed = target * 0.9 + (1.0 - target) * 0.1
    w = torch.where(target > 0.5, torch.ones_like(target), torch.full_like(target, neg_weight))
    bce = F.binary_cross_entropy(pred, smoothed, reduction="none")
    return (bce * w).mean()


def compute_phase3_loss(
    bilstm_out: dict,
    stgcn_out: dict,
    fusion_score: torch.Tensor,
    batch: dict,
    lockout_neg_weight: float = 4.0,
) -> torch.Tensor:
    """Combined Phase 3 loss. All continuous targets normalised to [0,1] for MSE."""
    t_qual   = batch["quality"] / 100.0
    t_smooth = batch["smoothness"] / 100.0
    t_ctrl   = batch["control"] / 100.0
    t_lock   = batch["lockout"]
    t_flare  = batch["elbow_flare"] / 100.0
    t_grip   = batch["grip_ratio"] / 100.0
    t_rtop   = batch["rom_top"] / 100.0
    t_rbot   = batch["rom_bottom"] / 100.0

    # Quality: fusion (primary) + both component models
    L_quality = (
        F.mse_loss(fusion_score / 100.0, t_qual)
        + 0.3 * masked_mse(bilstm_out["quality"] / 100.0, t_qual)
        + 0.3 * masked_mse(stgcn_out["quality"] / 100.0, t_qual)
    )

    L_smooth  = masked_mse(bilstm_out["smoothness"] / 100.0, t_smooth)
    L_ctrl    = masked_mse(bilstm_out["control"] / 100.0, t_ctrl)
    L_lock    = weighted_bce(stgcn_out["lockout"], t_lock, lockout_neg_weight)
    L_flare   = masked_mse(stgcn_out["elbow_flare"] / 100.0, t_flare)
    L_grip    = masked_mse(stgcn_out["grip_ratio"] / 100.0, t_grip)
    L_rtop    = masked_mse(stgcn_out["rom_top"] / 100.0, t_rtop)
    L_rbot    = masked_mse(stgcn_out["rom_bottom"] / 100.0, t_rbot)

    return (
        1.0 * L_quality
        + 0.5 * L_smooth
        + 0.5 * L_ctrl
        + 0.7 * L_lock
        + 0.5 * L_flare
        + 0.5 * L_grip
        + 0.5 * L_rtop
        + 0.5 * L_rbot
    )
```

**Verification:** `python -c "from core.exevision.training.ohp.losses import masked_mse, weighted_bce, compute_phase3_loss; print('OK')"`

---

## Step 10 — New: finetune_phase3.py

**File:** `core/exevision/training/ohp/finetune_phase3.py`

**CLI interface:**
```
python finetune_phase3.py \
    --annotation-dir training_dataset/ohp_phase3_annotations \
    --pretrain-bilstm models/bilstm_ohp_phase2.pt \
    --pretrain-stgcn  models/stgcn_ohp_phase2.pt \
    --output-dir      models/ \
    [--final]   # retrain on full 225 × 5 seeds instead of CV
```

**Progressive unfreezing — implement exactly:**

```python
STAGE1_EPOCHS = 20      # freeze encoder, train heads only
STAGE2_EPOCHS = 60      # unfreeze encoder at low LR
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS   # = 80

STAGE1_HEAD_LR     = 5e-4
STAGE2_HEAD_LR     = 3e-4
STAGE2_ENCODER_LR_LAST  = 5e-5   # last LSTM/block layers
STAGE2_ENCODER_LR_EARLY = 1e-5   # early LSTM/block layers


def _make_optimizer_stage1(bilstm, stgcn, fusion):
    """Stage 1: encoder frozen. Only heads + fusion trained."""
    # Collect only head + fusion params (encoder params excluded from optimizer)
    params = (
        list(bilstm.quality_head.parameters()) +
        list(bilstm.smoothness_head.parameters()) +
        list(bilstm.control_head.parameters()) +
        list(stgcn.quality_head.parameters()) +
        list(stgcn.lockout_head.parameters()) +
        list(stgcn.elbow_flare_head.parameters()) +
        list(stgcn.grip_ratio_head.parameters()) +
        list(stgcn.rom_top_head.parameters()) +
        list(stgcn.rom_bottom_head.parameters()) +
        list(fusion.parameters())
    )
    # knee_error_head NOT in optimizer (frozen, requires_grad=False)
    return torch.optim.Adam(params, lr=STAGE1_HEAD_LR, weight_decay=1e-3)


def _make_optimizer_stage2(bilstm, stgcn, fusion):
    """Stage 2: encoder unfrozen at low LR. Heads at higher LR. Knee frozen."""
    param_groups = [
        {"params": list(bilstm.lstm1.parameters()) + list(stgcn.block1.parameters()) +
                   list(stgcn.block2.parameters()),
         "lr": STAGE2_ENCODER_LR_EARLY},
        {"params": list(bilstm.lstm2.parameters()) + list(bilstm.temporal_attention.parameters()) +
                   list(stgcn.block3.parameters()) + list(stgcn.block4.parameters()) +
                   list(stgcn.block5.parameters()),
         "lr": STAGE2_ENCODER_LR_LAST},
        {"params": (
            list(bilstm.quality_head.parameters()) +
            list(bilstm.smoothness_head.parameters()) +
            list(bilstm.control_head.parameters()) +
            list(stgcn.quality_head.parameters()) +
            list(stgcn.lockout_head.parameters()) +
            list(stgcn.elbow_flare_head.parameters()) +
            list(stgcn.grip_ratio_head.parameters()) +
            list(stgcn.rom_top_head.parameters()) +
            list(stgcn.rom_bottom_head.parameters()) +
            list(fusion.parameters())
         ), "lr": STAGE2_HEAD_LR},
    ]
    return torch.optim.Adam(param_groups, weight_decay=1e-3)
```

**SGDR schedule (Stage 2 only):** `CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)`  
Note: `T_mult=1` means all cycles are 20 epochs → cycle 1: ep1-20, cycle 2: ep21-40, cycle 3: ep41-60 (Stage 2 ep 41-60 of Stage 2 = global ep 61-80 range, SWA starts at stage2_epoch 51).  
Actually implement with `T_0=20, T_mult=1` so cycles are fixed-length. SWA starts at Stage 2 epoch 51 (global epoch 71).

**SWA:**
```python
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model_bilstm = AveragedModel(bilstm)
swa_model_stgcn  = AveragedModel(stgcn)
SWA_START_STAGE2_EPOCH = 51   # Stage 2 epoch 51 = global epoch 71

# In training loop, Stage 2 only:
if stage2_epoch >= SWA_START_STAGE2_EPOCH:
    swa_model_bilstm.update_parameters(bilstm)
    swa_model_stgcn.update_parameters(stgcn)
```

**5-fold CV** (outer loop):
```python
from sklearn.model_selection import StratifiedKFold

# stratify_key = view + "_" + str(int(lockout_label))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Final retrain** (`--final` flag): 5 seeds {42, 7, 19, 3, 99}, train on full 225 reps (no val set), save as `bilstm_ohp_phase3_seed{seed}.pt` and `stgcn_ohp_phase3_seed{seed}.pt`.

**Checkpoint naming:**
- CV: `bilstm_ohp_phase3_fold{k}.pt`, `stgcn_ohp_phase3_fold{k}.pt`
- Final: `bilstm_ohp_phase3_seed{seed}.pt`, `stgcn_ohp_phase3_seed{seed}.pt`, `fusion_ohp_phase3_seed{seed}.pt`

---

## Step 11 — New: evaluate_phase3.py

**File:** `core/exevision/training/ohp/evaluate_phase3.py`

**CLI:** `python evaluate_phase3.py --annotation-dir ... --model-dir models/ --output results/ohp_phase3_eval.json`

**Metrics to compute and report:**

```python
ACCEPTANCE_THRESHOLDS = {
    "quality_mae":     12.0,
    "lockout_auc":     0.75,
    "smoothness_mae":  18.0,
    "control_mae":     18.0,
    "elbow_flare_mae": 15.0,
    "grip_ratio_mae":  12.0,
    "rom_top_mae":     12.0,
    "rom_bottom_mae":  12.0,
}
```

Report format:
```json
{
  "metrics": {"quality_mae": 9.2, "lockout_auc": 0.81, ...},
  "thresholds_met": true,
  "per_metric_pass": {"quality_mae": true, "lockout_auc": true, ...},
  "test_set_size": 25,
  "ensemble_seeds": [42, 7, 19, 3, 99]
}
```

**Gate:** print `PASS` or `FAIL` summary. Exit code 0 = pass, 1 = fail.

---

## Step 12 — New: tta.py

**File:** `core/exevision/training/ohp/tta.py`

**4 TTA variants per rep:**
1. Original
2. Horizontal flip (swap left/right joint indices in ST-GCN input)
3. +1 frame temporal jitter
4. -1 frame temporal jitter

```python
OHP_LR_SWAP_INDICES = {
    11: 12, 12: 11,  # shoulders
    13: 14, 14: 13,  # elbows
    15: 16, 16: 15,  # wrists
    23: 24, 24: 23,  # hips
    25: 26, 26: 25,  # knees
    27: 28, 28: 27,  # ankles
}

def apply_tta(bilstm_t, stgcn_t) -> list[tuple]:
    """Return list of (bilstm_variant, stgcn_variant) for 4 TTA versions."""
    ...
```

Average predictions across 4 variants before returning final rep scores.

---

## Steps 13–15 — Manual Steps (human-only, not code)

**Step 13:** Run `python core/exevision/analysis/select_ohp_phase3_samples.py` with Phase 2 workspace dirs. Output: `phase3_target_reps.json`.

**Step 14:** Annotate ~250 reps using the OHP Phase 3 annotator UI (Step 6). Estimated 12–20 hours. Follow annotation order: calibration anchors → uncertainty → error → view balance.

**Step 15:** Re-annotate 20 reps after 1-week gap. Compute ICC. Require ICC ≥ 0.7 before proceeding to training. Also verify: ≥ 30 lockout=0 reps, each view ≥ 20 reps.

**Sanity check script to run after annotation:**
```python
# Quick sanity — run before training
from pathlib import Path
import json

annots = list(Path("training_dataset/ohp_phase3_annotations/videos").glob("*.json"))
reps = [r for a in annots for r in json.loads(a.read_text()).get("reps", [])]
print(f"Total annotated reps: {len(reps)}")
print(f"Lockout=0: {sum(1 for r in reps if not r.get('human_flags', {}).get('lockout', True))}")
print(f"Lockout=1: {sum(1 for r in reps if r.get('human_flags', {}).get('lockout', False))}")
```

---

## Steps 16–18 — Training (run after annotation complete)

```powershell
# Step 16: 5-fold CV
python core\exevision\training\ohp\finetune_phase3.py `
    --annotation-dir training_dataset\ohp_phase3_annotations `
    --pretrain-bilstm models\bilstm_ohp_phase2.pt `
    --pretrain-stgcn  models\stgcn_ohp_phase2.pt `
    --output-dir      models\

# Step 17: Final 5-seed retrain
python core\exevision\training\ohp\finetune_phase3.py `
    --annotation-dir training_dataset\ohp_phase3_annotations `
    --pretrain-bilstm models\bilstm_ohp_phase2.pt `
    --pretrain-stgcn  models\stgcn_ohp_phase2.pt `
    --output-dir      models\ `
    --final

# Step 18: Evaluate on 25-rep held-out test set
python core\exevision\training\ohp\evaluate_phase3.py `
    --annotation-dir training_dataset\ohp_phase3_annotations `
    --model-dir      models\ `
    --output         results\ohp_phase3_eval.json
```

Expected output from Step 18: `PASS` with all 8 thresholds met. If `FAIL`: check which metrics failed, adjust loss weights in `losses.py` and retrain.

---

## Step 19a — Bug Fix: inference.py output path

**File:** `core/exevision/neural/ohp/inference.py`  
**Line:** 117  
**Bug:** writes to `neural_scores/` but `pipeline.py:630` reads from `neural_analysis/`.

Find:
```python
    output_path = workspace / exercise / "neural_scores" / tier / video_id / f"{video_id}_neural.json"
```

Replace with:
```python
    output_path = workspace / exercise / "neural_analysis" / tier / video_id / f"{video_id}_neural.json"
```

**Verification:** `Select-String "neural_scores" core\exevision\neural\ohp\inference.py` — must return NO matches after fix.

---

## Step 19b — Edit: neural_fusion_inference.py (Phase 3 routing)

**File:** `core/exevision/stages/neural_fusion_inference.py`  
**Pre-condition:** Step 19a complete. Step 10 complete (Phase 3 checkpoints exist after training).

Find (around line 539):
```python
    if args.exercise in ("overhead_press", "seated_overhead_press"):
        ...
        from inference import run_ohp_inference
        run_ohp_inference(args)
        return 0
```

Replace with a router that:
1. Checks for Phase 3 seed checkpoints: `models_dir.glob("bilstm_ohp_phase3_seed*.pt")`
2. If found: calls `run_ohp_phase3_ensemble(args)` (new function in `inference.py`)
3. If not found: falls back to existing `run_ohp_inference(args)` (Phase 2)

**Phase 3 ensemble function** (add to `core/exevision/neural/ohp/inference.py`):

```python
def run_ohp_phase3_ensemble(args) -> None:
    """Run Phase 3 5-seed ensemble inference. Writes to neural_analysis/."""
    workspace = Path(args.workspace_root)
    exercise = args.exercise
    video_id = args.video_id
    model_dir = Path(getattr(args, "model_dir", "models"))
    tier = getattr(args, "quality", "raw_unfiltered")

    # Suppress knee for seated
    suppress_knee = (exercise == "seated_overhead_press")

    seed_paths = sorted(model_dir.glob("bilstm_ohp_phase3_seed*.pt"))
    if not seed_paths:
        print(json.dumps({"error": "No Phase 3 seed checkpoints found", "neural_available": False}))
        return

    # [load models, run inference per seed, average predictions]
    # ... (implement loop over seeds, average all output heads)

    rep_results = []
    for rep in ...:
        avg = {key: mean across seeds}
        entry = {
            "rep_id":          rep.get("rep_id"),
            "neural_available": True,
            "neural_score":    avg["quality"],
            "lockout_prob":    avg["lockout"],
            "smoothness":      avg["smoothness"],
            "control":         avg["control"],
            "elbow_flare":     avg["elbow_flare"],
            "grip_ratio":      avg.get("grip_ratio"),   # None for side view
            "rom_top":         avg["rom_top"],
            "rom_bottom":      avg["rom_bottom"],
            "ensemble_std":    std_across_seeds["quality"],
        }
        if not suppress_knee:
            entry["knee_error_prob"] = avg["knee_error"]
        rep_results.append(entry)

    output_path = workspace / exercise / "neural_analysis" / tier / video_id / f"{video_id}_neural.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"exercise": exercise, "video_id": video_id, "reps": rep_results}, indent=2))
    print(json.dumps({"status": "ok", "reps_scored": len(rep_results), "output": str(output_path)}))
```

---

## Step 20 — Edit: engine.py (feedback cue wiring)

**File:** `core/exevision/feedback/engine.py`  
**Constraint:** No exercise-specific branches. Cue rules are config-driven from `error_cues_phase3` in `overhead_press.json` (added in Step 1).

Find where the engine currently reads neural outputs and emits cues. Add Phase 3 cue logic:

```python
# Threshold rules for Phase 3 OHP cues (read from config, not hardcoded here)
PHASE3_CUE_THRESHOLDS = {
    "lockout_prob":    ("below", 0.5),   # lockout_prob < 0.5 → emit cue
    "elbow_flare":     ("below", 50.0),
    "smoothness":      ("below", 50.0),
    "control":         ("below", 50.0),
    "grip_ratio":      ("below", 40.0),
    "rom_top":         ("below", 50.0),
    "rom_bottom":      ("below", 50.0),
    "knee_error_prob": ("above", 0.5),
}
```

Engine reads `error_cues_phase3` from exercise config and emits the cue text when threshold triggers. No hardcoded cue strings in `engine.py`.

---

## Step 21 — Docs: CLAUDE.md + CHANGELOG.md

Update `CLAUDE.md`:
- **Models table:** Add Phase 3 entries (`bilstm_ohp_phase3_seed*.pt` × 5, `stgcn_ohp_phase3_seed*.pt` × 5, `fusion_ohp_phase3_seed*.pt` × 5)
- **Active Issue #20:** Mark as RESOLVED after Step 19a
- **Active Issue #4:** Update — `knee_error_prob` now wired via Phase 3 feedback engine
- **Stage 9 routing description:** Update to reflect Phase 3 ensemble → Phase 2 fallback logic

Update `CHANGELOG.md` with Session 9 entry summarising Phase 3 implementation.

---

## Final Verification Checklist

Run all of these before declaring Phase 3 complete:

```powershell
# 1. No neural_scores path anywhere in OHP inference
Select-String "neural_scores" core\exevision\neural\ohp\inference.py   # must be empty

# 2. Phase 3 model heads present
python -c "
from core.exevision.neural.ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
import torch
b = OHPBiLSTMScorer()
s = OHPSTGCNScorer(torch.eye(10))
assert hasattr(b, 'smoothness_head') and hasattr(b, 'control_head')
assert hasattr(s, 'lockout_head') and hasattr(s, 'rom_top_head')
print('Model heads: OK')
"

# 3. Config keys present
python -c "
import json
d = json.load(open('core/exevision/config/exercises/overhead_press.json'))
for k in ['annotation_metrics_phase3','annotation_binary_phase3','annotation_grip_null_views','error_cues_phase3']:
    assert k in d, f'Missing: {k}'
print('Config keys: OK')
"

# 4. Registry works for all exercises
python -c "
from core.exevision.neural.registry import get_model_classes
for ex in ['squat','overhead_press','seated_overhead_press']:
    get_model_classes(ex)
print('Registry: OK')
"

# 5. Squat inference unchanged (sanity)
Select-String "bilstm_finetuned" apps\api\pipeline.py   # must still exist

# 6. Phase 3 eval passes thresholds (after training)
python core\exevision\training\ohp\evaluate_phase3.py --annotation-dir training_dataset\ohp_phase3_annotations --model-dir models\ --output results\ohp_phase3_eval.json
# Must print PASS
```

---

## Modularity Contract

All Phase 3 additions satisfy the exercise modularity contract:

- `core/exevision/neural/nn_models.py` — untouched (squat models)
- `core/exevision/training/finetune_models.py` — untouched (squat training)
- `engine.py` — config-driven cues only, no new exercise branches
- New exercise in future = 1 line in `registry.py` + new `neural/<exercise>/` + new `training/<exercise>/`
