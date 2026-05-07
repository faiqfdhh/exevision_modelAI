# OHP Phase 2 — Multi-Task Fine-Tuning with FitnessAQA

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune pre-trained BiLSTM and ST-GCN encoders on 2,260 FitnessAQA labeled OHP videos using a multi-task objective — simultaneously predict overall quality score and soft elbow/knee error probabilities — then wire the error heads into inference-time feedback.

**Architecture:** New exercise-isolated modules under `core/exevision/neural/ohp/` and `core/exevision/training/ohp/` contain all OHP-specific neural code. Nothing in the squat path (`nn_models.py`, `finetune_models.py`, `evaluate_model.py`) is modified. FitnessAQA's error-window timestamps are converted to soft per-rep labels by a pure-function `label_derivation` module, then consumed by `OHPRepDataset`. Two models are trained: one for `overhead_press` (with knee error head) and one for `seated_overhead_press` (elbow error head only). The fusion layer reuses `HeuristicGuidedFusion(heuristic_dim=16)` unchanged.

**Tech Stack:** Python 3.10, PyTorch (CPU), pytest, existing `nn_utils.py` + `pretrain_bilstm.py` + `pretrain_stgcn.py` encoder classes (imported, not modified).

**Envisioned end product:** After this plan executes you will have:
- `models/bilstm_ohp_phase2.pt` and `models/stgcn_ohp_phase2.pt` — standing OHP phase 2 checkpoints
- `models/bilstm_seated_ohp_phase2.pt` and `models/stgcn_seated_ohp_phase2.pt` — seated OHP phase 2 checkpoints
- Inference stage that returns `elbow_error_prob` and `knee_error_prob` alongside the overall score
- Feedback engine that maps those probabilities + heuristic metric scores to specific form cues

Phase 3 (manual annotation fine-tuning and fusion training) is a separate plan.

---

## Pre-conditions (must be true before Task 1)

- Pre-trained weights exist: `models/bilstm_ohp_pretrained.pt`, `models/bilstm_seated_ohp_pretrained.pt`, `models/stgcn_ohp_pretrained_encoder.pt`, `models/stgcn_seated_ohp_pretrained_encoder.pt`
- FitnessAQA labeled dataset is at `D:\FitnessAQA\Overhead Press\Labeled_Dataset-OHP\Labeled_Dataset\` with sub-paths:
  - `videos\videos\*.mp4` (2,367 files)
  - `Labels\error_elbows.json`, `Labels\error_knees.json`
  - `Splits\train_keys.json`, `Splits\val_keys.json`, `Splits\test_keys.json`
- Branch `multiexercise` is checked out

---

## File Map

```
CREATE
  core/exevision/neural/ohp/__init__.py
  core/exevision/neural/ohp/models.py           OHPBiLSTMScorer, OHPSTGCNScorer
  core/exevision/neural/ohp/heuristic_vec.py    build_ohp_heuristic_vector (16-dim)
  core/exevision/neural/ohp/fusion.py           build_ohp_fusion() factory
  core/exevision/neural/ohp/inference.py        run_ohp_inference() for dispatch
  core/exevision/training/ohp/__init__.py
  core/exevision/training/ohp/label_derivation.py  pure overlap math
  core/exevision/training/ohp/prepare_dataset.py   FitnessAQA → annotation JSONs
  core/exevision/training/ohp/data.py              OHPRepDataset
  core/exevision/training/ohp/finetune.py          multi-task training entry point
  core/exevision/training/ohp/evaluate.py          evaluation script
  tests/ohp/__init__.py
  tests/ohp/test_label_derivation.py
  tests/ohp/test_heuristic_vec.py
  tests/ohp/test_models_smoke.py
  tests/ohp/test_prepare_dataset.py

MODIFY (minimal, squat path untouched)
  core/exevision/stages/neural_fusion_inference.py   add OHP exercise dispatch (~10 lines)
```

---

## Modularity Rules (enforce throughout)

1. **Never import from `nn_models.py` into any OHP module except `fusion.py`** (which reuses `HeuristicGuidedFusion` as-is).
2. **Never modify `nn_models.py`, `finetune_models.py`, or `evaluate_model.py`.**
3. Every OHP file has its own sys.path setup if it needs to import from `neural/`.
4. Each file does one thing. If a file grows past ~250 lines, split it.
5. No magic numbers in code — all thresholds in named constants at the top of the file.

---

## Task 1: Scaffold directories and empty `__init__.py` files

**Files:**
- Create: `core/exevision/neural/ohp/__init__.py`
- Create: `core/exevision/training/ohp/__init__.py`
- Create: `tests/ohp/__init__.py`

- [ ] **Step 1: Create directories and empty init files**

```powershell
New-Item -ItemType Directory -Force "core\exevision\neural\ohp"
New-Item -ItemType Directory -Force "core\exevision\training\ohp"
New-Item -ItemType Directory -Force "tests\ohp"
"" | Out-File -Encoding utf8 "core\exevision\neural\ohp\__init__.py"
"" | Out-File -Encoding utf8 "core\exevision\training\ohp\__init__.py"
"" | Out-File -Encoding utf8 "tests\ohp\__init__.py"
```

- [ ] **Step 2: Verify**

```powershell
Test-Path "core\exevision\neural\ohp\__init__.py"   # True
Test-Path "core\exevision\training\ohp\__init__.py" # True
Test-Path "tests\ohp\__init__.py"                   # True
```

- [ ] **Step 3: Commit**

```bash
git add core/exevision/neural/ohp/ core/exevision/training/ohp/ tests/ohp/
git commit -m "chore: scaffold ohp neural and training module directories"
```

---

## Task 2: `label_derivation.py` — pure overlap math

**Files:**
- Create: `core/exevision/training/ohp/label_derivation.py`
- Create: `tests/ohp/test_label_derivation.py`

### What this module does

Given a rep window in seconds and FitnessAQA error windows (list of `[start, end]` second pairs), compute:
- `elbow_error_soft`: fraction of rep duration covered by elbow error windows, clamped [0, 1]
- `knee_error_soft`: same for knee errors (always 0.0 for seated OHP)
- `overall_score`: blended from error score and heuristic score

Score formula:
```
error_score   = 100 × (1 − 0.65 × elbow_error_soft − 0.35 × knee_error_soft)
overall_score = clamp(0.7 × error_score + 0.3 × heuristic_score, 0, 100)
```

- [ ] **Step 1: Write the failing tests first**

Create `tests/ohp/test_label_derivation.py`:

```python
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
```

- [ ] **Step 2: Run — expect ImportError / AttributeError**

```bash
pytest tests/ohp/test_label_derivation.py -v
```

Expected: `ImportError: No module named 'core.exevision.training.ohp.label_derivation'`

- [ ] **Step 3: Implement `label_derivation.py`**

Create `core/exevision/training/ohp/label_derivation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# Weights for score formula — change here to affect all derived labels
_ELBOW_PENALTY_WEIGHT = 0.65
_KNEE_PENALTY_WEIGHT = 0.35
_FITNESSAQA_BLEND = 0.70   # weight for error_score in final blend
_HEURISTIC_BLEND = 0.30    # weight for heuristic_score in final blend


@dataclass(frozen=True)
class RepLabels:
    overall_score: float      # 0–100
    elbow_error_soft: float   # 0.0–1.0
    knee_error_soft: float    # 0.0–1.0 (always 0.0 for seated)


def compute_overlap_ratio(
    rep_start_sec: float,
    rep_end_sec: float,
    error_windows: List[List[float]],
) -> float:
    """Return fraction of rep duration covered by error_windows, clamped to [0, 1].

    Overlapping windows are unioned before dividing, so they never double-count.
    """
    rep_dur = rep_end_sec - rep_start_sec
    if rep_dur <= 0.0 or not error_windows:
        return 0.0

    # Collect overlapping seconds as a sorted list of (start, end) pairs clipped to rep
    clipped: List[Tuple[float, float]] = []
    for window in error_windows:
        w_start, w_end = float(window[0]), float(window[1])
        overlap_start = max(w_start, rep_start_sec)
        overlap_end = min(w_end, rep_end_sec)
        if overlap_end > overlap_start:
            clipped.append((overlap_start, overlap_end))

    if not clipped:
        return 0.0

    # Union overlapping intervals
    clipped.sort()
    merged: List[Tuple[float, float]] = [clipped[0]]
    for start, end in clipped[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_overlap = sum(end - start for start, end in merged)
    return min(total_overlap / rep_dur, 1.0)


def derive_rep_labels(
    rep_start_sec: float,
    rep_end_sec: float,
    elbow_windows: List[List[float]],
    knee_windows: List[List[float]],
    heuristic_score: float,
    seated: bool,
) -> RepLabels:
    """Derive soft training labels for one OHP rep from FitnessAQA error windows.

    For seated OHP, knee_error_soft is forced to 0.0 regardless of knee_windows
    because leg landmarks are zeroed in the seated variant.
    """
    elbow_soft = compute_overlap_ratio(rep_start_sec, rep_end_sec, elbow_windows)
    knee_soft = 0.0 if seated else compute_overlap_ratio(rep_start_sec, rep_end_sec, knee_windows)

    error_score = 100.0 * (
        1.0 - _ELBOW_PENALTY_WEIGHT * elbow_soft - _KNEE_PENALTY_WEIGHT * knee_soft
    )
    overall = _FITNESSAQA_BLEND * error_score + _HEURISTIC_BLEND * heuristic_score
    overall = max(0.0, min(100.0, overall))

    return RepLabels(
        overall_score=round(overall, 4),
        elbow_error_soft=round(elbow_soft, 6),
        knee_error_soft=round(knee_soft, 6),
    )
```

- [ ] **Step 4: Run — expect all pass**

```bash
pytest tests/ohp/test_label_derivation.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add core/exevision/training/ohp/label_derivation.py tests/ohp/test_label_derivation.py
git commit -m "feat(ohp): add label_derivation module with overlap math and RepLabels"
```

---

## Task 3: `heuristic_vec.py` — OHP 16-dim feature vector

**Files:**
- Create: `core/exevision/neural/ohp/heuristic_vec.py`
- Create: `tests/ohp/test_heuristic_vec.py`

### Vector layout (16 dims)

| Index | Content |
|-------|---------|
| 0 | overall heuristic score / 100 |
| 1–4 | per-metric scores / 100 (grip_ratio, rom, lockout, elbow_flare) |
| 5–10 | flag bits (incomplete_lockout, elbow_flare, forward_lean, bar_drift, wrist_deviation, knee_instability) |
| 11–15 | view one-hot (front, back, side, front_side, back_side) |

- [ ] **Step 1: Write failing tests**

Create `tests/ohp/test_heuristic_vec.py`:

```python
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
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/ohp/test_heuristic_vec.py -v
```

- [ ] **Step 3: Implement `heuristic_vec.py`**

Create `core/exevision/neural/ohp/heuristic_vec.py`:

```python
from __future__ import annotations

from typing import Optional

import numpy as np

OHP_HEURISTIC_DIM = 16

_OHP_METRIC_ORDER = ["grip_ratio", "rom", "lockout", "elbow_flare"]

_OHP_FLAG_ORDER = [
    "incomplete_lockout",
    "elbow_flare",
    "forward_lean",
    "bar_drift",
    "wrist_deviation",
    "knee_instability",
]

_VIEW_ORDER = ["front", "back", "side", "front_side", "back_side"]


def _safe(value: object) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
        return 0.0 if not (v == v) else v  # NaN check without math import
    except (TypeError, ValueError):
        return 0.0


def build_ohp_heuristic_vector(rep_data: dict, view: Optional[str]) -> np.ndarray:
    """Build a 16-dim float32 feature vector for one OHP rep.

    Layout:
      [0]    overall heuristic score normalised to [0, 1]
      [1–4]  per-metric scores (grip_ratio, rom, lockout, elbow_flare) normalised
      [5–10] 6 flag bits
      [11–15] view one-hot (front, back, side, front_side, back_side)
    """
    vec = np.zeros(OHP_HEURISTIC_DIM, dtype=np.float32)

    vec[0] = _safe(rep_data.get("heuristic_score")) / 100.0

    hms = rep_data.get("heuristic_metric_scores") or {}
    for i, metric in enumerate(_OHP_METRIC_ORDER):
        vec[1 + i] = _safe(hms.get(metric)) / 100.0

    flags = rep_data.get("flags") or {}
    for i, flag in enumerate(_OHP_FLAG_ORDER):
        vec[5 + i] = 1.0 if bool(flags.get(flag, False)) else 0.0

    view_lower = (view or "").lower().strip()
    for i, v in enumerate(_VIEW_ORDER):
        vec[11 + i] = 1.0 if view_lower == v else 0.0

    return vec
```

- [ ] **Step 4: Run — expect all pass**

```bash
pytest tests/ohp/test_heuristic_vec.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add core/exevision/neural/ohp/heuristic_vec.py tests/ohp/test_heuristic_vec.py
git commit -m "feat(ohp): add 16-dim OHP heuristic vector builder"
```

---

## Task 4: `models.py` — OHPBiLSTMScorer and OHPSTGCNScorer

**Files:**
- Create: `core/exevision/neural/ohp/models.py`
- Create: `tests/ohp/test_models_smoke.py`

### Architecture

Both models mirror the squat encoders exactly (same layer names for weight loading) but have OHP-specific heads.

**OHPBiLSTMScorer heads:**
- `quality_head`: Linear(256→64) → ReLU → Dropout(0.2) → Linear(64→1) → Sigmoid → ×100
- `elbow_error_head`: Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
- `knee_error_head`: same as elbow (only created when `include_knee_head=True`)

**OHPSTGCNScorer heads:**
- `quality_head`: Linear(256+5→64) → ReLU → Dropout(0.2) → Linear(64→1) → Sigmoid → ×100
- `elbow_error_head`: Linear(256→64) → ReLU → Linear(64→1) → Sigmoid
- `knee_error_head`: same (optional)

- [ ] **Step 1: Write smoke tests first**

Create `tests/ohp/test_models_smoke.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure neural dirs are on path
_NEURAL = Path(__file__).resolve().parents[2] / "core" / "exevision" / "neural"
_TRAIN = Path(__file__).resolve().parents[2] / "core" / "exevision" / "training"
for _p in [str(_NEURAL), str(_NEURAL / "ohp"), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import NUM_BILSTM_CHANNELS, FIXED_SEQ_LEN, build_adjacency_matrix
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer


BATCH = 4


def test_bilstm_output_keys():
    model = OHPBiLSTMScorer(include_knee_head=True)
    x = torch.zeros(BATCH, FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
    out = model(x)
    assert "embedding" in out
    assert "quality" in out
    assert "elbow_error" in out
    assert "knee_error" in out


def test_bilstm_output_shapes():
    model = OHPBiLSTMScorer(include_knee_head=True)
    x = torch.zeros(BATCH, FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
    out = model(x)
    assert out["quality"].shape == (BATCH,)
    assert out["elbow_error"].shape == (BATCH,)
    assert out["knee_error"].shape == (BATCH,)


def test_bilstm_seated_no_knee_head():
    model = OHPBiLSTMScorer(include_knee_head=False)
    x = torch.zeros(BATCH, FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
    out = model(x)
    assert "knee_error" not in out


def test_bilstm_quality_range():
    model = OHPBiLSTMScorer()
    x = torch.randn(BATCH, FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
    out = model(x)
    assert out["quality"].min() >= 0.0
    assert out["quality"].max() <= 100.0


def test_bilstm_error_probs_range():
    model = OHPBiLSTMScorer()
    x = torch.randn(BATCH, FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
    out = model(x)
    assert out["elbow_error"].min() >= 0.0
    assert out["elbow_error"].max() <= 1.0


def test_stgcn_output_keys():
    A = torch.tensor(build_adjacency_matrix())
    model = OHPSTGCNScorer(A, include_knee_head=True)
    # ST-GCN input: (B, C, T, J)
    from nn_utils import STGCN_CHANNELS, NUM_ACTIVE_JOINTS
    x = torch.zeros(BATCH, STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_ACTIVE_JOINTS)
    out = model(x)
    assert "embedding" in out
    assert "quality" in out
    assert "elbow_error" in out
    assert "knee_error" in out


def test_stgcn_seated_no_knee_head():
    A = torch.tensor(build_adjacency_matrix())
    model = OHPSTGCNScorer(A, include_knee_head=False)
    from nn_utils import STGCN_CHANNELS, NUM_ACTIVE_JOINTS
    x = torch.zeros(BATCH, STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_ACTIVE_JOINTS)
    out = model(x)
    assert "knee_error" not in out


def test_stgcn_quality_range():
    A = torch.tensor(build_adjacency_matrix())
    model = OHPSTGCNScorer(A)
    from nn_utils import STGCN_CHANNELS, NUM_ACTIVE_JOINTS
    x = torch.randn(BATCH, STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_ACTIVE_JOINTS)
    out = model(x)
    assert out["quality"].min() >= 0.0
    assert out["quality"].max() <= 100.0
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/ohp/test_models_smoke.py -v
```

- [ ] **Step 3: Implement `models.py`**

Create `core/exevision/neural/ohp/models.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Resolve shared neural utilities without modifying sys.path permanently
_NEURAL_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_ROOT = Path(__file__).resolve().parents[3] / "training"
for _p in [str(_NEURAL_ROOT), str(_TRAIN_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import NUM_BILSTM_CHANNELS
from pretrain_bilstm import TemporalAttention   # encoder building block — not modified
from pretrain_stgcn import STGCNBlock           # encoder building block — not modified


def _score_head(in_dim: int) -> nn.Sequential:
    """Linear head that maps embeddings to a quality score in [0, 100]."""
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def _error_head(in_dim: int) -> nn.Sequential:
    """Linear head that maps embeddings to an error probability in [0, 1]."""
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class OHPBiLSTMScorer(nn.Module):
    """Temporal scorer for OHP with optional knee error head.

    Encoder layer names (lstm1, lstm2, temporal_attention) match the pretrain
    checkpoint exactly so that load_pretrained() can transfer weights without
    key remapping.

    Args:
        input_dim: Number of BiLSTM signal channels (default: NUM_BILSTM_CHANNELS = 4).
        hidden_dim: LSTM hidden size (default: 128, must match pretrained).
        dropout: Dropout rate (default: 0.3, must match pretrained).
        include_knee_head: Set False for seated OHP — removes the knee error head.
    """

    def __init__(
        self,
        input_dim: int = NUM_BILSTM_CHANNELS,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        include_knee_head: bool = True,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.temporal_attention = TemporalAttention(hidden_dim * 2)

        embed_dim = hidden_dim * 2
        self.quality_head = _score_head(embed_dim)
        self.elbow_error_head = _error_head(embed_dim)
        self.knee_error_head = _error_head(embed_dim) if include_knee_head else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        return self.temporal_attention(out)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(x)
        result: Dict[str, torch.Tensor] = {
            "embedding": emb,
            "quality": self.quality_head(emb).squeeze(-1) * 100.0,
            "elbow_error": self.elbow_error_head(emb).squeeze(-1),
        }
        if self.knee_error_head is not None:
            result["knee_error"] = self.knee_error_head(emb).squeeze(-1)
        return result

    def load_pretrained(self, path: str) -> Tuple[int, int]:
        """Load encoder weights from a pretrained checkpoint.

        Ignores reconstruction_head keys so this works with both the full
        pretrain checkpoint and encoder-only variants.
        Returns (n_missing, n_unexpected).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        encoder_keys = {
            k: v for k, v in state.items()
            if not k.startswith("reconstruction_head")
        }
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        return len(missing), len(unexpected)


class OHPSTGCNScorer(nn.Module):
    """Spatial scorer for OHP with optional knee error head.

    Encoder block names (block1–block5) match the pretrain checkpoint exactly.

    Args:
        A: Normalised adjacency matrix, shape (11, 11), float32 numpy array or tensor.
        dropout: Dropout rate (default: 0.2, must match pretrained).
        include_knee_head: Set False for seated OHP.
    """

    _VIEW_DIM = 5   # size of view one-hot appended to embedding before quality head

    def __init__(
        self,
        A,
        dropout: float = 0.2,
        include_knee_head: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32)

        from nn_utils import STGCN_CHANNELS
        self.block1 = STGCNBlock(STGCN_CHANNELS, 64,  A, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64,  64,  A, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64,  128, A, stride=2, dropout=dropout)
        self.block4 = STGCNBlock(128, 128, A, stride=1, dropout=dropout)
        self.block5 = STGCNBlock(128, 256, A, stride=2, dropout=dropout)

        embed_dim = 256
        self.quality_head = _score_head(embed_dim + self._VIEW_DIM)
        self.elbow_error_head = _error_head(embed_dim)
        self.knee_error_head = _error_head(embed_dim) if include_knee_head else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.mean(dim=(2, 3))   # global average over time and joints → (B, 256)

    def forward(
        self,
        x: torch.Tensor,
        view_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emb = self.encode(x)
        if view_vec is None:
            view_vec = torch.zeros(emb.shape[0], self._VIEW_DIM, device=emb.device)
        spatial_in = torch.cat([emb, view_vec], dim=-1)

        result: Dict[str, torch.Tensor] = {
            "embedding": emb,
            "quality": self.quality_head(spatial_in).squeeze(-1) * 100.0,
            "elbow_error": self.elbow_error_head(emb).squeeze(-1),
        }
        if self.knee_error_head is not None:
            result["knee_error"] = self.knee_error_head(emb).squeeze(-1)
        return result

    def load_pretrained(self, path: str) -> Tuple[int, int]:
        """Load encoder weights from pretrained checkpoint (encoder-only .pt preferred).

        Accepts both full pretrain checkpoint and encoder-only file.
        Returns (n_missing, n_unexpected).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        encoder_keys = {
            k: v for k, v in state.items()
            if k.startswith(("block1", "block2", "block3", "block4", "block5"))
        }
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        return len(missing), len(unexpected)
```

- [ ] **Step 4: Run — expect all pass**

```bash
pytest tests/ohp/test_models_smoke.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add core/exevision/neural/ohp/models.py tests/ohp/test_models_smoke.py
git commit -m "feat(ohp): add OHPBiLSTMScorer and OHPSTGCNScorer with multi-task heads"
```

---

## Task 5: `fusion.py` — OHP fusion factory

**Files:**
- Create: `core/exevision/neural/ohp/fusion.py`

No dedicated test needed — `HeuristicGuidedFusion` is already tested in the squat suite; this file is a one-function factory.

- [ ] **Step 1: Implement `fusion.py`**

Create `core/exevision/neural/ohp/fusion.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

_NEURAL_ROOT = Path(__file__).resolve().parents[1]
if str(_NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEURAL_ROOT))

from nn_models import HeuristicGuidedFusion   # reused unchanged — heuristic_dim is a param

OHP_HEURISTIC_DIM = 16   # must match build_ohp_heuristic_vector output length


def build_ohp_fusion(neural_dim: int = 256, fusion_dim: int = 64) -> HeuristicGuidedFusion:
    """Return a HeuristicGuidedFusion configured for OHP's 16-dim heuristic vector."""
    return HeuristicGuidedFusion(
        heuristic_dim=OHP_HEURISTIC_DIM,
        neural_dim=neural_dim,
        fusion_dim=fusion_dim,
    )
```

- [ ] **Step 2: Smoke-check import**

```bash
python -c "from core.exevision.neural.ohp.fusion import build_ohp_fusion; m = build_ohp_fusion(); print('ok', m)"
```

Expected: prints `ok HeuristicGuidedFusion(...)`

- [ ] **Step 3: Commit**

```bash
git add core/exevision/neural/ohp/fusion.py
git commit -m "feat(ohp): add build_ohp_fusion factory wrapping HeuristicGuidedFusion(heuristic_dim=16)"
```

---

## Task 6: `prepare_dataset.py` — FitnessAQA → annotation JSONs

**Files:**
- Create: `core/exevision/training/ohp/prepare_dataset.py`
- Create: `tests/ohp/test_prepare_dataset.py`

### What this script does

For each video_id in all three splits:
1. Load `error_elbows.json[video_id]` and `error_knees.json[video_id]` (empty list if key absent).
2. Find the features JSON at `{workspace}/overhead_press/extracted_features_clean/raw_unfiltered/{video_id}.json` — **skip video if missing** (Stage 2.5 not yet run).
3. Find the segmented JSON at `{workspace}/overhead_press/segmented_reps/raw_unfiltered/{video_id}_segmented.json` — **fallback to whole-video as 1 rep** if missing.
4. Find the scoring JSON at `{workspace}/overhead_press/aqa_analysis_simple/raw_unfiltered/{video_id}/{video_id}_aqa_simple.json` — **default heuristic_score=50.0** if missing.
5. For each rep: derive `RepLabels` via `label_derivation.derive_rep_labels()`.
6. Write two annotation JSONs: `{output_dir}/{video_id}.json` (OHP) and `{output_dir}/{video_id}_seated.json` (seated OHP — identical except `exercise` field and `features_json` path).

### Annotation JSON schema

```json
{
  "video_id": "72676_1",
  "exercise": "overhead_press",
  "pipeline_run": "ohp_phase2",
  "pipeline_outputs": {
    "features_json": "D:\\FitnessAQA\\ohp_phase2\\workspace\\overhead_press\\extracted_features_clean\\raw_unfiltered\\72676_1.json",
    "segmented_json": "D:\\FitnessAQA\\ohp_phase2\\workspace\\overhead_press\\segmented_reps\\raw_unfiltered\\72676_1_segmented.json",
    "scoring_json":   "D:\\FitnessAQA\\ohp_phase2\\workspace\\overhead_press\\aqa_analysis_simple\\raw_unfiltered\\72676_1\\72676_1_aqa_simple.json"
  },
  "view": "front",
  "fps": 30.0,
  "calibration": {"body_scale": 0.22, "standing_hip_height": 0.57},
  "total_reps": 1,
  "annotation_source": "fitnessaqa_derived",
  "fitnessaqa_split": "train",
  "annotated_at": "2026-05-07T12:00:00",
  "reps": [
    {
      "rep_id": 1,
      "start_frame": 0,
      "end_frame": 89,
      "start_sec": 0.0,
      "end_sec": 2.97,
      "human_score": 75.2,
      "heuristic_score": 70.0,
      "heuristic_metric_scores": {"grip_ratio": 80.0, "rom": 75.0, "lockout": 90.0, "elbow_flare": 85.0},
      "flags": {"incomplete_lockout": false, "elbow_flare": false, "forward_lean": false, "bar_drift": false, "wrist_deviation": false, "knee_instability": false},
      "elbow_error_soft": 0.352,
      "knee_error_soft": 0.0,
      "annotation_source": "fitnessaqa_derived"
    }
  ]
}
```

The seated JSON is identical but `exercise = "seated_overhead_press"`, `features_json` path uses `seated_overhead_press/` instead of `overhead_press/`, and `knee_error_soft` is always `0.0`.

- [ ] **Step 1: Write integration test with fixtures**

Create `tests/ohp/test_prepare_dataset.py`:

```python
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure module paths resolve
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "core" / "exevision" / "training" / "ohp"))
sys.path.insert(0, str(_REPO / "core" / "exevision" / "training"))


@pytest.fixture
def fake_workspace(tmp_path):
    """Build a minimal fake ohp_phase2 workspace for one video 'test_001'."""
    vid = "test_001"
    fps = 30.0
    total_frames = 90

    # features JSON (mimics Stage 2.5 output structure)
    feat_dir = tmp_path / "overhead_press" / "extracted_features_clean" / "raw_unfiltered"
    feat_dir.mkdir(parents=True)
    feat_path = feat_dir / f"{vid}.json"
    feat_path.write_text(json.dumps({
        "info": {"fps": fps, "view": "front", "calibration": {"body_scale": 0.22, "standing_hip_height": 0.57}},
        "keypoints_img": [[[0.5, 0.5, 0.0, 1.0]] * 33] * total_frames,
    }))

    # segmented JSON (1 rep covering full video)
    seg_dir = tmp_path / "overhead_press" / "segmented_reps" / "raw_unfiltered"
    seg_dir.mkdir(parents=True)
    seg_path = seg_dir / f"{vid}_segmented.json"
    seg_path.write_text(json.dumps({
        "info": {"fps": fps},
        "signals": {
            "normalized_hip_displacement": [0.5] * total_frames,
            "window_velocity": [0.0] * total_frames,
            "knee_angles": [160.0] * total_frames,
            "landmark_confidence": [0.99] * total_frames,
        },
        "repetitions": [{"rep_id": 1, "start_frame": 0, "end_frame": total_frames - 1}],
    }))

    # scoring JSON
    score_dir = tmp_path / "overhead_press" / "aqa_analysis_simple" / "raw_unfiltered" / vid
    score_dir.mkdir(parents=True)
    score_path = score_dir / f"{vid}_aqa_simple.json"
    score_path.write_text(json.dumps({
        "reps": [{
            "rep_id": 1,
            "overall_score": 70.0,
            "metric_scores": {"grip_ratio": 80.0, "rom": 75.0, "lockout": 90.0, "elbow_flare": 85.0},
            "flags": {"incomplete_lockout": False, "elbow_flare": False, "forward_lean": False,
                      "bar_drift": False, "wrist_deviation": False, "knee_instability": False},
        }],
    }))

    # seated features (just needs to exist with the right path)
    seated_dir = tmp_path / "seated_overhead_press" / "extracted_features_clean" / "raw_unfiltered"
    seated_dir.mkdir(parents=True)
    (seated_dir / f"{vid}.json").write_text(feat_path.read_text())

    return tmp_path, vid, fps, total_frames


@pytest.fixture
def fake_labels_dir(tmp_path):
    labels = tmp_path / "Labels"
    labels.mkdir()
    (labels / "error_elbows.json").write_text(json.dumps({
        "test_001": [[0.5, 1.5]],   # 1 sec overlap in a 3-sec rep → 1/3
    }))
    (labels / "error_knees.json").write_text(json.dumps({
        "test_001": [],
    }))
    splits = tmp_path / "Splits"
    splits.mkdir()
    (splits / "train_keys.json").write_text(json.dumps(["test_001"]))
    (splits / "val_keys.json").write_text(json.dumps([]))
    (splits / "test_keys.json").write_text(json.dumps([]))
    return tmp_path


def test_prepare_writes_both_variants(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, fps, total_frames = fake_workspace
    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    ohp_path = out_dir / f"{vid}.json"
    seated_path = out_dir / f"{vid}_seated.json"
    assert ohp_path.exists(), "OHP annotation not written"
    assert seated_path.exists(), "Seated OHP annotation not written"


def test_ohp_annotation_schema(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, _, _ = fake_workspace
    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    data = json.loads((out_dir / f"{vid}.json").read_text())
    assert data["exercise"] == "overhead_press"
    assert data["fitnessaqa_split"] == "train"
    assert data["annotation_source"] == "fitnessaqa_derived"
    assert len(data["reps"]) == 1

    rep = data["reps"][0]
    assert "human_score" in rep
    assert "elbow_error_soft" in rep
    assert "knee_error_soft" in rep
    assert 0.0 <= rep["human_score"] <= 100.0
    assert 0.0 <= rep["elbow_error_soft"] <= 1.0


def test_seated_always_zero_knee(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, _, _ = fake_workspace
    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    # Give seated a knee error — should still be 0.0
    knee_path = fake_labels_dir / "Labels" / "error_knees.json"
    knee_path.write_text(json.dumps({"test_001": [[0.0, 3.0]]}))

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    seated = json.loads((out_dir / f"{vid}_seated.json").read_text())
    for rep in seated["reps"]:
        assert rep["knee_error_soft"] == 0.0


def test_missing_segmented_fallback(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, fps, total_frames = fake_workspace
    # Remove segmented JSON to trigger fallback
    (ws_root / "overhead_press" / "segmented_reps" / "raw_unfiltered" / f"{vid}_segmented.json").unlink()

    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    data = json.loads((out_dir / f"{vid}.json").read_text())
    assert len(data["reps"]) == 1
    rep = data["reps"][0]
    assert rep["start_frame"] == 0
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/ohp/test_prepare_dataset.py -v
```

- [ ] **Step 3: Implement `prepare_dataset.py`**

Create `core/exevision/training/ohp/prepare_dataset.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_OHP_TRAIN = Path(__file__).resolve().parent
if str(_OHP_TRAIN) not in sys.path:
    sys.path.insert(0, str(_OHP_TRAIN))

from label_derivation import RepLabels, derive_rep_labels

_DEFAULT_HEURISTIC_SCORE = 50.0
_QUALITY_TIER = "raw_unfiltered"


# ---------------------------------------------------------------------------
# Workspace path helpers
# ---------------------------------------------------------------------------

def _features_path(workspace: Path, exercise: str, video_id: str) -> Path:
    return workspace / exercise / "extracted_features_clean" / _QUALITY_TIER / f"{video_id}.json"


def _segmented_path(workspace: Path, video_id: str) -> Path:
    return workspace / "overhead_press" / "segmented_reps" / _QUALITY_TIER / f"{video_id}_segmented.json"


def _scoring_path(workspace: Path, video_id: str) -> Path:
    return workspace / "overhead_press" / "aqa_analysis_simple" / _QUALITY_TIER / video_id / f"{video_id}_aqa_simple.json"


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_label_windows(path: Path) -> Dict[str, List[List[float]]]:
    data = _load_json(path) or {}
    return {k: v for k, v in data.items()}


def _load_splits(splits_dir: Path) -> Dict[str, str]:
    """Return {video_id: split_name} for all splits."""
    result: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        p = splits_dir / f"{split_name}_keys.json"
        if p.exists():
            for vid in (json.loads(p.read_text()) or []):
                result[str(vid)] = split_name
    return result


# ---------------------------------------------------------------------------
# Rep boundary extraction
# ---------------------------------------------------------------------------

def _extract_reps_from_segmented(seg_data: dict, fps: float) -> List[dict]:
    reps = seg_data.get("repetitions", []) or []
    out = []
    for rep in reps:
        sf = int(rep.get("start_frame", 0))
        ef = int(rep.get("end_frame", sf))
        out.append({"start_frame": sf, "end_frame": ef,
                    "start_sec": sf / fps, "end_sec": ef / fps})
    return out


def _whole_video_rep(feat_data: dict) -> List[dict]:
    kp = feat_data.get("keypoints_img", [])
    total_frames = len(kp) if kp else 1
    fps = float((feat_data.get("info") or {}).get("fps", 30.0))
    return [{"start_frame": 0, "end_frame": total_frames - 1,
             "start_sec": 0.0, "end_sec": (total_frames - 1) / fps}]


# ---------------------------------------------------------------------------
# Per-rep heuristic score extraction
# ---------------------------------------------------------------------------

def _rep_heuristic(scoring_data: Optional[dict], rep_id: int) -> tuple:
    """Return (overall_score, metric_scores_dict, flags_dict) for a rep."""
    _default_flags = {
        "incomplete_lockout": False, "elbow_flare": False, "forward_lean": False,
        "bar_drift": False, "wrist_deviation": False, "knee_instability": False,
    }
    _default_metrics = {"grip_ratio": 0.0, "rom": 0.0, "lockout": 0.0, "elbow_flare": 0.0}
    if scoring_data is None:
        return _DEFAULT_HEURISTIC_SCORE, _default_metrics, _default_flags
    for r in (scoring_data.get("reps") or []):
        if r.get("rep_id") == rep_id:
            return (
                float(r.get("overall_score", _DEFAULT_HEURISTIC_SCORE)),
                r.get("metric_scores") or _default_metrics,
                r.get("flags") or _default_flags,
            )
    return _DEFAULT_HEURISTIC_SCORE, _default_metrics, _default_flags


# ---------------------------------------------------------------------------
# Annotation JSON builder
# ---------------------------------------------------------------------------

def _build_annotation(
    video_id: str,
    exercise: str,
    workspace: Path,
    feat_data: dict,
    reps_boundaries: List[dict],
    scoring_data: Optional[dict],
    elbow_windows: List[List[float]],
    knee_windows: List[List[float]],
    split: str,
) -> dict:
    info = feat_data.get("info") or {}
    fps = float(info.get("fps", 30.0))
    view = str(info.get("view", "unknown"))
    calibration = info.get("calibration") or {}
    seated = exercise == "seated_overhead_press"

    reps_out = []
    for i, bounds in enumerate(reps_boundaries):
        rep_id = i + 1
        h_score, h_metrics, h_flags = _rep_heuristic(scoring_data, rep_id)
        labels: RepLabels = derive_rep_labels(
            rep_start_sec=bounds["start_sec"],
            rep_end_sec=bounds["end_sec"],
            elbow_windows=elbow_windows,
            knee_windows=knee_windows,
            heuristic_score=h_score,
            seated=seated,
        )
        reps_out.append({
            "rep_id": rep_id,
            "start_frame": bounds["start_frame"],
            "end_frame": bounds["end_frame"],
            "start_sec": bounds["start_sec"],
            "end_sec": bounds["end_sec"],
            "human_score": labels.overall_score,
            "heuristic_score": h_score,
            "heuristic_metric_scores": h_metrics,
            "flags": h_flags,
            "elbow_error_soft": labels.elbow_error_soft,
            "knee_error_soft": labels.knee_error_soft,
            "annotation_source": "fitnessaqa_derived",
        })

    return {
        "video_id": video_id,
        "exercise": exercise,
        "pipeline_run": "ohp_phase2",
        "pipeline_outputs": {
            "features_json": str(_features_path(workspace, exercise, video_id)),
            "segmented_json": str(_segmented_path(workspace, video_id)),
            "scoring_json": str(_scoring_path(workspace, video_id)),
        },
        "view": view,
        "fps": fps,
        "calibration": calibration,
        "total_reps": len(reps_out),
        "annotation_source": "fitnessaqa_derived",
        "fitnessaqa_split": split,
        "annotated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "reps": reps_out,
    }


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_preparation(
    workspace: Path,
    labels_dir: Path,
    splits_dir: Path,
    output_dir: Path,
) -> None:
    workspace = Path(workspace)
    labels_dir = Path(labels_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elbow_labels = _load_label_windows(labels_dir / "error_elbows.json")
    knee_labels = _load_label_windows(labels_dir / "error_knees.json")
    splits = _load_splits(splits_dir)

    written = skipped = 0
    for video_id, split in sorted(splits.items()):
        feat_path = _features_path(workspace, "overhead_press", video_id)
        feat_data = _load_json(feat_path)
        if feat_data is None:
            print(f"  SKIP {video_id}: features JSON not found at {feat_path}")
            skipped += 1
            continue

        fps = float((feat_data.get("info") or {}).get("fps", 30.0))
        seg_data = _load_json(_segmented_path(workspace, video_id))
        reps = (
            _extract_reps_from_segmented(seg_data, fps)
            if seg_data and seg_data.get("repetitions")
            else _whole_video_rep(feat_data)
        )
        if not reps:
            reps = _whole_video_rep(feat_data)

        scoring_data = _load_json(_scoring_path(workspace, video_id))
        elbow_windows = elbow_labels.get(video_id, [])
        knee_windows = knee_labels.get(video_id, [])

        for exercise in ("overhead_press", "seated_overhead_press"):
            suffix = "_seated" if exercise == "seated_overhead_press" else ""
            seated_feat = _features_path(workspace, exercise, video_id)
            if not seated_feat.exists() and exercise == "seated_overhead_press":
                continue   # seated features not generated — skip silently
            anno = _build_annotation(
                video_id, exercise, workspace, feat_data, reps,
                scoring_data, elbow_windows, knee_windows, split,
            )
            out_path = output_dir / f"{video_id}{suffix}.json"
            out_path.write_text(json.dumps(anno, indent=2), encoding="utf-8")

        written += 1

    print(f"\nDone. Written: {written} videos ({written * 2} JSON files). Skipped: {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FitnessAQA labels to OHP annotation JSONs")
    parser.add_argument("--workspace", required=True, help="Path to ohp_phase2/workspace")
    parser.add_argument("--labels-dir", required=True, help="Path to Labeled_Dataset/Labels")
    parser.add_argument("--splits-dir", required=True, help="Path to Labeled_Dataset/Splits")
    parser.add_argument("--output-dir", required=True, help="Where to write annotation JSONs")
    args = parser.parse_args()
    run_preparation(
        workspace=Path(args.workspace),
        labels_dir=Path(args.labels_dir),
        splits_dir=Path(args.splits_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/ohp/test_prepare_dataset.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add core/exevision/training/ohp/prepare_dataset.py tests/ohp/test_prepare_dataset.py
git commit -m "feat(ohp): add prepare_dataset — FitnessAQA error windows → annotation JSONs"
```

---

## Task 7: `data.py` — OHPRepDataset

**Files:**
- Create: `core/exevision/training/ohp/data.py`

This dataset class reads annotation JSONs produced by `prepare_dataset.py`, loads the corresponding pose features, and returns tensors ready for model input.

- [ ] **Step 1: Implement `data.py`**

Create `core/exevision/training/ohp/data.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO   = Path(__file__).resolve().parents[4]   # ohp/ → training/ → exevision/ → core/ → repo root
_NEURAL = _REPO / "core" / "exevision" / "neural"
_TRAIN  = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_NEURAL / "ohp"), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import (
    FIXED_SEQ_LEN,
    NUM_ACTIVE_JOINTS,
    STGCN_CHANNELS,
    _extract_rep_matrix,
    _extract_stgcn_rep,
    _load_json,
    build_adjacency_matrix,
    pad_or_truncate,
)
from ohp.heuristic_vec import build_ohp_heuristic_vector


class OHPRepDataset(Dataset):
    """Dataset of OHP reps sourced from FitnessAQA-derived annotation JSONs.

    Each item returns a dict with keys:
      bilstm_input   : float32 tensor (FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
      stgcn_input    : float32 tensor (STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_ACTIVE_JOINTS)
      heuristic_vec  : float32 tensor (16,)
      view_vec       : float32 tensor (5,)
      overall_score  : float32 scalar tensor
      elbow_error    : float32 scalar tensor  [0, 1]
      knee_error     : float32 scalar tensor  [0, 1]  (0.0 for seated)
    """

    def __init__(self, annotation_paths: List[Path], split: Optional[str] = None) -> None:
        """
        Args:
            annotation_paths: List of annotation JSON file paths to index.
            split: If provided, only include reps where annotation["fitnessaqa_split"] == split.
        """
        self._records: List[dict] = []
        for path in annotation_paths:
            try:
                anno = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            if split is not None and anno.get("fitnessaqa_split") != split:
                continue
            feat_path = Path(anno.get("pipeline_outputs", {}).get("features_json", ""))
            seg_path = Path(anno.get("pipeline_outputs", {}).get("segmented_json", ""))
            if not feat_path.exists():
                continue
            view = anno.get("view", "unknown")
            fps = float(anno.get("fps", 30.0))
            for rep in (anno.get("reps") or []):
                self._records.append({
                    "feat_path": feat_path,
                    "seg_path": seg_path,
                    "rep": rep,
                    "view": view,
                    "fps": fps,
                    "calibration": anno.get("calibration") or {},
                })

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self._records[idx]
        rep = record["rep"]
        feat_data = _load_json(record["feat_path"]) or {}
        seg_data = _load_json(record["seg_path"]) if record["seg_path"].exists() else {}

        bilstm_raw = _extract_rep_matrix(seg_data, rep)
        if bilstm_raw is None:
            bilstm_raw = np.zeros((1, 4), dtype=np.float32)
        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN))

        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep)
        if stgcn_raw is None:
            stgcn_raw = np.zeros((1, NUM_ACTIVE_JOINTS, STGCN_CHANNELS), dtype=np.float32)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)  # (T, J, C)
        # Reorder to (C, T, J) for ST-GCN conv
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype(np.float32, copy=False)
        )

        hvec = build_ohp_heuristic_vector(rep, record["view"])
        view_vec = hvec[11:16].copy()   # 5-dim view one-hot

        return {
            "bilstm_input": bilstm_t,
            "stgcn_input": stgcn_t,
            "heuristic_vec": torch.from_numpy(hvec),
            "view_vec": torch.from_numpy(view_vec),
            "overall_score": torch.tensor(float(rep.get("human_score", 50.0)), dtype=torch.float32),
            "elbow_error": torch.tensor(float(rep.get("elbow_error_soft", 0.0)), dtype=torch.float32),
            "knee_error": torch.tensor(float(rep.get("knee_error_soft", 0.0)), dtype=torch.float32),
        }


def build_dataloaders(
    annotation_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Return train/val/test DataLoaders from all annotation JSONs in annotation_dir."""
    all_paths = sorted(Path(annotation_dir).glob("*.json"))
    loaders = {}
    for split in ("train", "val", "test"):
        ds = OHPRepDataset(all_paths, split=split)
        if len(ds) == 0:
            continue
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
    return loaders
```

- [ ] **Step 2: Smoke-check import**

```bash
python -c "from core.exevision.training.ohp.data import OHPRepDataset, build_dataloaders; print('ok')"
```

- [ ] **Step 3: Commit**

```bash
git add core/exevision/training/ohp/data.py
git commit -m "feat(ohp): add OHPRepDataset and build_dataloaders"
```

---

## Task 8: `finetune.py` — multi-task training entry point

**Files:**
- Create: `core/exevision/training/ohp/finetune.py`

This is the Phase 2 training script. Run it twice — once for `overhead_press`, once for `seated_overhead_press`.

- [ ] **Step 1: Implement `finetune.py`**

Create `core/exevision/training/ohp/finetune.py`:

```python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[4]
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from data import OHPRepDataset, build_dataloaders

# Loss weights — tune these if Phase 2 MAE is above 15
_LAMBDA_COMPONENT_QUALITY = 0.3   # weight for per-model quality loss vs fusion
_LAMBDA_ELBOW = 0.3               # weight for elbow BCE
_LAMBDA_KNEE = 0.2                # weight for knee BCE (skipped for seated)

SEED = 42
EPOCHS = 60
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
PATIENCE = 10   # early stopping patience (val loss non-improvement)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_loss(
    bilstm_out: dict,
    stgcn_out: dict,
    fusion_score: torch.Tensor,
    batch: dict,
    include_knee: bool,
) -> torch.Tensor:
    target = batch["overall_score"] / 100.0   # normalize to [0, 1]
    target_elbow = batch["elbow_error"]
    target_knee = batch["knee_error"]

    mse_fusion = F.mse_loss(fusion_score / 100.0, target)
    mse_bilstm = F.mse_loss(bilstm_out["quality"] / 100.0, target)
    mse_stgcn = F.mse_loss(stgcn_out["quality"] / 100.0, target)

    bce_bilstm_elbow = F.binary_cross_entropy(bilstm_out["elbow_error"], target_elbow)
    bce_stgcn_elbow = F.binary_cross_entropy(stgcn_out["elbow_error"], target_elbow)

    loss = mse_fusion
    loss += _LAMBDA_COMPONENT_QUALITY * (mse_bilstm + mse_stgcn)
    loss += _LAMBDA_ELBOW * (bce_bilstm_elbow + bce_stgcn_elbow)

    if include_knee and "knee_error" in bilstm_out:
        bce_bilstm_knee = F.binary_cross_entropy(bilstm_out["knee_error"], target_knee)
        bce_stgcn_knee = F.binary_cross_entropy(stgcn_out["knee_error"], target_knee)
        loss += _LAMBDA_KNEE * (bce_bilstm_knee + bce_stgcn_knee)

    return loss


def _run_epoch(
    bilstm: OHPBiLSTMScorer,
    stgcn: OHPSTGCNScorer,
    fusion,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    include_knee: bool,
    train: bool,
) -> float:
    bilstm.train(train)
    stgcn.train(train)
    fusion.train(train)
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            bilstm_out = bilstm(batch["bilstm_input"])
            stgcn_out = stgcn(batch["stgcn_input"], batch["view_vec"])
            fusion_score, _ = fusion(
                batch["heuristic_vec"],
                stgcn_out["embedding"],
                bilstm_out["embedding"],
            )
            loss = _compute_loss(bilstm_out, stgcn_out, fusion_score, batch, include_knee)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train(
    annotation_dir: Path,
    pretrain_bilstm: Path,
    pretrain_stgcn: Path,
    output_dir: Path,
    exercise: str,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    include_knee = exercise != "seated_overhead_press"
    output_suffix = "ohp_phase2" if exercise == "overhead_press" else "seated_ohp_phase2"

    loaders = build_dataloaders(annotation_dir, batch_size=batch_size)
    if "train" not in loaders:
        raise RuntimeError(f"No training data found in {annotation_dir}")

    A = torch.tensor(build_adjacency_matrix(), dtype=torch.float32).to(device)

    bilstm = OHPBiLSTMScorer(include_knee_head=include_knee).to(device)
    stgcn = OHPSTGCNScorer(A, include_knee_head=include_knee).to(device)
    fusion = build_ohp_fusion().to(device)

    m_bilstm, _ = bilstm.load_pretrained(str(pretrain_bilstm))
    m_stgcn, _ = stgcn.load_pretrained(str(pretrain_stgcn))
    print(f"BiLSTM missing keys after pretrain load: {m_bilstm}")
    print(f"ST-GCN missing keys after pretrain load: {m_stgcn}")

    optimizer = torch.optim.Adam(
        list(bilstm.parameters()) + list(stgcn.parameters()) + list(fusion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(bilstm, stgcn, fusion, loaders["train"], optimizer, device, include_knee, train=True)
        val_loss = float("inf")
        if "val" in loaders:
            val_loss = _run_epoch(bilstm, stgcn, fusion, loaders["val"], optimizer, device, include_knee, train=False)
            scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bilstm.state_dict(), output_dir / f"bilstm_{output_suffix}.pt")
            torch.save(stgcn.state_dict(), output_dir / f"stgcn_{output_suffix}.pt")
            torch.save(fusion.state_dict(), output_dir / f"fusion_{output_suffix}.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
                break

    print(f"Best val loss: {best_val:.4f}. Checkpoints saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 OHP multi-task fine-tuning")
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--pretrain-bilstm", required=True)
    parser.add_argument("--pretrain-stgcn", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exercise", default="overhead_press",
                        choices=["overhead_press", "seated_overhead_press"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(
        annotation_dir=Path(args.annotation_dir),
        pretrain_bilstm=Path(args.pretrain_bilstm),
        pretrain_stgcn=Path(args.pretrain_stgcn),
        output_dir=Path(args.output_dir),
        exercise=args.exercise,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import is clean**

```bash
python -c "import sys; sys.path.insert(0,'core/exevision/training/ohp'); from finetune import train; print('ok')"
```

- [ ] **Step 3: Commit**

```bash
git add core/exevision/training/ohp/finetune.py
git commit -m "feat(ohp): add Phase 2 multi-task fine-tuning script"
```

---

## Task 9: `evaluate.py` — OHP evaluation

**Files:**
- Create: `core/exevision/training/ohp/evaluate.py`

Loads phase 2 checkpoints, runs on test split, reports MAE (quality) and AUC (error heads).

- [ ] **Step 1: Implement `evaluate.py`**

Create `core/exevision/training/ohp/evaluate.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[4]
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from data import build_dataloaders


def _auc_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn dependency."""
    sorted_idx = np.argsort(-probs)
    labels_sorted = labels[sorted_idx]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for lbl in labels_sorted:
        if lbl >= 0.5:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    # Trapezoidal AUC
    return float(np.trapz(tpr, fpr))


def evaluate(
    annotation_dir: Path,
    checkpoint_dir: Path,
    exercise: str,
    batch_size: int = 32,
) -> dict:
    include_knee = exercise != "seated_overhead_press"
    suffix = "ohp_phase2" if exercise == "overhead_press" else "seated_ohp_phase2"
    device = torch.device("cpu")

    loaders = build_dataloaders(annotation_dir, batch_size=batch_size)
    if "test" not in loaders:
        raise RuntimeError("No test split found")

    A = torch.tensor(build_adjacency_matrix())
    bilstm = OHPBiLSTMScorer(include_knee_head=include_knee).to(device)
    stgcn = OHPSTGCNScorer(A, include_knee_head=include_knee).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm.load_state_dict(torch.load(checkpoint_dir / f"bilstm_{suffix}.pt", map_location="cpu"))
    stgcn.load_state_dict(torch.load(checkpoint_dir / f"stgcn_{suffix}.pt", map_location="cpu"))
    fusion.load_state_dict(torch.load(checkpoint_dir / f"fusion_{suffix}.pt", map_location="cpu"))

    bilstm.eval(); stgcn.eval(); fusion.eval()

    preds, targets, elbow_probs, elbow_true, knee_probs, knee_true = [], [], [], [], [], []

    with torch.no_grad():
        for batch in loaders["test"]:
            bilstm_out = bilstm(batch["bilstm_input"])
            stgcn_out = stgcn(batch["stgcn_input"], batch["view_vec"])
            score, _ = fusion(batch["heuristic_vec"], stgcn_out["embedding"], bilstm_out["embedding"])
            preds.extend(score.tolist())
            targets.extend(batch["overall_score"].tolist())
            elbow_probs.extend(bilstm_out["elbow_error"].tolist())
            elbow_true.extend(batch["elbow_error"].tolist())
            if include_knee and "knee_error" in bilstm_out:
                knee_probs.extend(bilstm_out["knee_error"].tolist())
                knee_true.extend(batch["knee_error"].tolist())

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    elbow_auc = _auc_binary(np.array(elbow_probs), np.array(elbow_true))
    knee_auc = _auc_binary(np.array(knee_probs), np.array(knee_true)) if knee_probs else float("nan")

    results = {
        "exercise": exercise,
        "n_test_reps": len(preds),
        "mae_overall": round(mae, 3),
        "elbow_error_auc": round(elbow_auc, 3),
        "knee_error_auc": round(knee_auc, 3) if not np.isnan(knee_auc) else "n/a",
        "acceptance": {
            "mae_pass": mae < 15.0,
            "elbow_auc_pass": elbow_auc > 0.65 if not np.isnan(elbow_auc) else False,
        },
    }
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--exercise", default="overhead_press",
                        choices=["overhead_press", "seated_overhead_press"])
    args = parser.parse_args()
    evaluate(Path(args.annotation_dir), Path(args.checkpoint_dir), args.exercise)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add core/exevision/training/ohp/evaluate.py
git commit -m "feat(ohp): add OHP Phase 2 evaluation script with MAE + AUC metrics"
```

---

## Task 10: `neural_fusion_inference.py` — OHP dispatch

**Files:**
- Modify: `core/exevision/stages/neural_fusion_inference.py` (add ~15 lines, squat path untouched)
- Create: `core/exevision/neural/ohp/inference.py`

The inference stage currently runs squat only. Add an OHP branch that dispatches to a new `run_ohp_inference()` function. The squat path is not touched.

- [ ] **Step 1: Create `core/exevision/neural/ohp/inference.py`**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch

_NEURAL = Path(__file__).resolve().parents[1]
_OHP = Path(__file__).resolve().parent
_TRAIN_OHP = Path(__file__).resolve().parents[3] / "training" / "ohp"
for _p in [str(_NEURAL), str(_OHP), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix, _extract_stgcn_rep, _extract_rep_matrix, pad_or_truncate, FIXED_SEQ_LEN
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from ohp.heuristic_vec import build_ohp_heuristic_vector
from nn_models import HeuristicGuidedFusion   # fusion base class


def _load_checkpoints(model_dir: Path, exercise: str):
    """Load BiLSTM, ST-GCN, and fusion checkpoints for the given exercise."""
    suffix = "overhead_press" if exercise == "overhead_press" else "seated_overhead_press"
    # Prefer phase2 checkpoints; fall back to phase-labelled names
    def _ckpt(name):
        for candidate in [
            model_dir / f"{name}_{suffix}.pt",
            model_dir / f"{name}_ohp_phase2.pt",
            model_dir / f"{name}_seated_ohp_phase2.pt",
        ]:
            if candidate.exists():
                return candidate
        return None

    bilstm_path = _ckpt("bilstm")
    stgcn_path = _ckpt("stgcn")
    fusion_path = _ckpt("fusion")
    return bilstm_path, stgcn_path, fusion_path


def run_ohp_inference(args) -> None:
    """Entry point called from neural_fusion_inference.py for OHP exercises."""
    workspace = Path(args.workspace_root)
    exercise = args.exercise
    video_id = args.video_id
    model_dir = Path(args.model_dir) if hasattr(args, "model_dir") else Path("models")

    include_knee = exercise != "seated_overhead_press"
    device = torch.device("cpu")
    A = torch.tensor(build_adjacency_matrix())

    bilstm = OHPBiLSTMScorer(include_knee_head=include_knee).to(device)
    stgcn = OHPSTGCNScorer(A, include_knee_head=include_knee).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm_path, stgcn_path, fusion_path = _load_checkpoints(model_dir, exercise)
    if bilstm_path:
        bilstm.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
    if stgcn_path:
        stgcn.load_state_dict(torch.load(stgcn_path, map_location="cpu"))
    if fusion_path:
        fusion.load_state_dict(torch.load(fusion_path, map_location="cpu"))

    bilstm.eval(); stgcn.eval(); fusion.eval()

    # Load stage outputs
    tier = getattr(args, "quality", "raw_unfiltered")
    feat_path = workspace / exercise / "extracted_features_clean" / tier / f"{video_id}.json"
    seg_path = workspace / exercise / "segmented_reps" / tier / f"{video_id}_segmented.json"
    score_path = workspace / exercise / "aqa_analysis_simple" / tier / video_id / f"{video_id}_aqa_simple.json"

    if not feat_path.exists() or not seg_path.exists():
        print(json.dumps({"error": f"Stage outputs not found for {video_id}", "neural_available": False}))
        return

    feat_data = json.loads(feat_path.read_text())
    seg_data = json.loads(seg_path.read_text())
    score_data = json.loads(score_path.read_text()) if score_path.exists() else {}
    view = str((feat_data.get("info") or {}).get("view", "unknown"))

    rep_results = []
    for rep in (seg_data.get("repetitions") or []):
        bilstm_raw = _extract_rep_matrix(seg_data, rep)
        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep)
        if bilstm_raw is None or stgcn_raw is None:
            continue

        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN)).unsqueeze(0)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)
        import numpy as np
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype("float32")
        ).unsqueeze(0)

        # Get heuristic score for this rep from scoring JSON
        rep_score_data = next(
            (r for r in (score_data.get("reps") or []) if r.get("rep_id") == rep.get("rep_id")), {}
        )
        hvec = torch.from_numpy(build_ohp_heuristic_vector(rep_score_data, view)).unsqueeze(0)
        view_vec = hvec[:, 11:16]

        with torch.no_grad():
            bilstm_out = bilstm(bilstm_t)
            stgcn_out = stgcn(stgcn_t, view_vec)
            final_score, residual = fusion(hvec, stgcn_out["embedding"], bilstm_out["embedding"])

        rep_results.append({
            "rep_id": rep.get("rep_id"),
            "neural_score": round(float(final_score.item()), 2),
            "elbow_error_prob": round(float(bilstm_out["elbow_error"].item()), 4),
            "knee_error_prob": round(float(bilstm_out["knee_error"].item()), 4) if include_knee and "knee_error" in bilstm_out else None,
            "neural_available": True,
        })

    output_path = workspace / exercise / "neural_scores" / tier / video_id / f"{video_id}_neural.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"exercise": exercise, "video_id": video_id, "reps": rep_results}, indent=2))
    print(json.dumps({"status": "ok", "reps_scored": len(rep_results), "output": str(output_path)}))
```

- [ ] **Step 2: Add OHP dispatch to `neural_fusion_inference.py`**

Find the `main()` function in `core/exevision/stages/neural_fusion_inference.py`. It currently has squat-specific logic. Add an exercise check at the top of main, before any squat code, so the squat path is completely bypassed for OHP:

```python
# At the top of main(), BEFORE any existing squat logic:
if args.exercise in ("overhead_press", "seated_overhead_press"):
    import sys as _sys
    from pathlib import Path as _Path
    _ohp = str(_Path(__file__).resolve().parents[1] / "neural" / "ohp")
    if _ohp not in _sys.path:
        _sys.path.insert(0, _ohp)
    from inference import run_ohp_inference
    run_ohp_inference(args)
    return
# ... existing squat code unchanged below ...
```

- [ ] **Step 3: Run existing squat tests to confirm squat path is untouched**

```bash
pytest tests/ -k "not ohp" -v --tb=short
```

Expected: all squat tests pass.

- [ ] **Step 4: Commit**

```bash
git add core/exevision/neural/ohp/inference.py core/exevision/stages/neural_fusion_inference.py
git commit -m "feat(ohp): wire OHP inference dispatch into neural_fusion_inference stage"
```

---

## Task 11: Batch Stage Execution

Before running `prepare_dataset.py`, Stage 2.5 must be run on the 2,260 labeled videos. Run Stage 5 and Stage 8 for `overhead_press` only (seated variant reuses OHP segmentation paths).

**Pre-flight check — verify workspace doesn't already have outputs:**

```powershell
$ws = "D:\FitnessAQA\ohp_phase2\workspace"
(Get-ChildItem "$ws\overhead_press\extracted_features_clean\raw_unfiltered\*.json" -ErrorAction SilentlyContinue).Count
# If 0: stages haven't run yet — proceed
# If > 0: stages partially or fully run — skip Stage 2.5 for already-processed videos
```

**Stage 2.5 — Feature extraction**

```powershell
$py  = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe"
$s25 = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\extract_selected_features.py"
$env:EXEVISION_MODEL_PATH      = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\models\pose_landmarker_heavy.task"
$env:EXEVISION_FACE_MODEL_PATH = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\models\blaze_face_short_range.tflite"

Set-Location "D:\FitnessAQA\ohp_phase2\workspace"

& $py $s25 unfiltered `
  --exercise overhead_press `
  --video-dir "D:\FitnessAQA\Overhead Press\Labeled_Dataset-OHP\Labeled_Dataset\videos\videos" `
  --no-viz `
  --no-report
# Processes all labeled videos. Re-running is safe — skipped videos registry prevents re-processing.
# Seated features are auto-generated alongside OHP features by Stage 2.5.
```

**Stage 5 — Temporal segmentation (OHP only)**

```powershell
$s5 = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\temporal_segmentation.py"
$featDir = "D:\FitnessAQA\ohp_phase2\workspace\overhead_press\extracted_features_clean\raw_unfiltered"

Get-ChildItem $featDir -Filter "*.json" | ForEach-Object {
    $vid = $_.BaseName
    & $py $s5 `
      --video-id $vid `
      --exercise overhead_press `
      --no-viz
}
# Run from workspace root so relative paths resolve correctly.
```

**Stage 8 — Scoring (OHP only)**

```powershell
$s8 = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\scoring.py"

Get-ChildItem $featDir -Filter "*.json" | ForEach-Object {
    $vid = $_.BaseName
    & $py $s8 $vid `
      --exercise overhead_press
}
```

**Verify outputs:**

```powershell
$featCount = (Get-ChildItem "D:\FitnessAQA\ohp_phase2\workspace\overhead_press\extracted_features_clean\raw_unfiltered\*.json").Count
$segCount  = (Get-ChildItem "D:\FitnessAQA\ohp_phase2\workspace\overhead_press\segmented_reps\raw_unfiltered\*_segmented.json").Count
Write-Host "Features: $featCount  Segmented: $segCount"
# Expect features ≈ 2000–2260 (some may fail quality gating)
# Expect segmented ≈ features (some may have 0 reps)
```

---

## Task 12: Run `prepare_dataset.py`

Requires: Task 11 complete.

- [ ] **Step 1: Run preparation**

```powershell
$py = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe"
$script = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\training\ohp\prepare_dataset.py"

& $py $script `
  --workspace "D:\FitnessAQA\ohp_phase2\workspace" `
  --labels-dir "D:\FitnessAQA\Overhead Press\Labeled_Dataset-OHP\Labeled_Dataset\Labels" `
  --splits-dir "D:\FitnessAQA\Overhead Press\Labeled_Dataset-OHP\Labeled_Dataset\Splits" `
  --output-dir "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\training_dataset\annotations\videos"
```

- [ ] **Step 2: Verify output count**

```powershell
$annoDir = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\training_dataset\annotations\videos"
$ohp    = (Get-ChildItem $annoDir -Filter "[0-9]*_[0-9]*.json" | Where-Object { $_.Name -notmatch "_seated" }).Count
$seated = (Get-ChildItem $annoDir -Filter "*_seated.json").Count
Write-Host "OHP: $ohp  Seated: $seated"
# Expect both ≈ 1500–2260 (limited by how many videos passed Stage 2.5 quality gate)
```

- [ ] **Step 3: Sanity-check one file**

```powershell
$sample = Get-ChildItem $annoDir -Filter "[0-9]*_[0-9]*.json" | Select-Object -First 1
Get-Content $sample.FullName | ConvertFrom-Json | Select-Object video_id, exercise, fitnessaqa_split, total_reps
# Expect: exercise=overhead_press, fitnessaqa_split=train/val/test, total_reps=1
```

---

## Task 13: Run Phase 2 fine-tuning and evaluate

Requires: Task 12 complete.

- [ ] **Step 1: Train OHP (standing)**

```powershell
$py     = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe"
$script = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\training\ohp\finetune.py"
$models = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\models"
$annoDir = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\training_dataset\annotations\videos"

& $py $script `
  --annotation-dir $annoDir `
  --pretrain-bilstm "$models\bilstm_ohp_pretrained.pt" `
  --pretrain-stgcn "$models\stgcn_ohp_pretrained_encoder.pt" `
  --output-dir $models `
  --exercise overhead_press `
  --epochs 60
```

Expected output: epoch-by-epoch losses, final message "Checkpoints saved to models/"

- [ ] **Step 2: Train seated OHP**

```powershell
& $py $script `
  --annotation-dir $annoDir `
  --pretrain-bilstm "$models\bilstm_seated_ohp_pretrained.pt" `
  --pretrain-stgcn "$models\stgcn_seated_ohp_pretrained_encoder.pt" `
  --output-dir $models `
  --exercise seated_overhead_press `
  --epochs 60
```

- [ ] **Step 3: Verify checkpoint files exist**

```powershell
Test-Path "$models\bilstm_ohp_phase2.pt"         # True
Test-Path "$models\stgcn_ohp_phase2.pt"          # True
Test-Path "$models\fusion_ohp_phase2.pt"         # True
Test-Path "$models\bilstm_seated_ohp_phase2.pt"  # True
```

- [ ] **Step 4: Evaluate OHP**

```powershell
$eval = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\training\ohp\evaluate.py"
& $py $eval `
  --annotation-dir $annoDir `
  --checkpoint-dir $models `
  --exercise overhead_press
```

Expected output:
```json
{
  "mae_overall": <target: < 15.0>,
  "elbow_error_auc": <target: > 0.65>,
  "acceptance": {"mae_pass": true, "elbow_auc_pass": true}
}
```

If `mae_pass` is false: check rep segmentation quality — bad Stage 5 outputs propagate noisy labels. Re-run evaluation with `--exercise seated_overhead_press` for the seated variant.

- [ ] **Step 5: Commit final checkpoints note to CHANGELOG**

Update `CHANGELOG.md` with:
- Date
- Phase 2 completed: exercise names, checkpoint file names, test MAE and AUC values observed

```bash
git add CHANGELOG.md
git commit -m "docs: record OHP Phase 2 fine-tuning results"
```

---

## Acceptance Criteria

| Metric | Target | What to do if failing |
|---|---|---|
| Test MAE (quality score) | < 15.0 | Debug Stage 5 rep segmentation; re-run prepare_dataset; retrain |
| Elbow AUC | > 0.65 | Check FitnessAQA label coverage; try increasing `_LAMBDA_ELBOW`; retrain |
| Knee AUC (standing only) | > 0.60 | As above with `_LAMBDA_KNEE` |
| All squat tests pass | 100% | Do NOT modify squat code to fix — check dispatch logic |
| `inference.py` returns `elbow_error_prob` in output | required | Verify `run_ohp_inference` writes the neural JSON |

---

## What this plan deliberately does NOT cover

- Phase 3 (manual annotation fine-tuning and fusion calibration) — separate plan
- Feedback engine integration of `elbow_error_prob` / `knee_error_prob` — part of Phase 3 plan
- Desktop UI neural toggle for OHP — deferred until Phase 3 models are ready
- `seated_overhead_press` seated-variant Stage 5 run — seated features reuse OHP segmentation paths, so a separate Stage 5 run is not required
