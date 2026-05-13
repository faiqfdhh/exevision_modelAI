# OHP Neural View Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a per-frame MLP on spatial landmark signals from 165 annotated OHP videos and integrate it into Stage 4 (`classify_views.py`) as a drop-in replacement for the heuristic classifier, with confidence-gated fallback.

**Architecture:** Feature extraction reuses the same 12 signals the heuristic already computes (shoulder_width, hip_width, nose_z, arm visibility, face_detected, etc.) from `keypoints_img` frames. Each frame is one training sample; inference uses majority-vote across all video frames. Stage 4 tries neural first; if confidence ≥ threshold → use it, else fall back to heuristic.

**Tech Stack:** PyTorch (MLP), NumPy, scikit-learn (StratifiedKFold, class_weight), existing `classify_views.py` heuristic untouched as fallback.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `core/exevision/neural/ohp/view_classifier.py` | Model class, feature extraction, predict_video, load helper |
| Create | `core/exevision/training/ohp/train_view_classifier.py` | Training script: load annotations, extract features, 5-fold CV, final save |
| Modify | `core/exevision/stages/classify_views.py` | Add `--neural` / `--confidence-threshold` flags; try neural before heuristic |
| Create | `tests/test_view_classifier.py` | Unit tests for feature extraction + model predict |

---

## Task 1: Write failing tests

**Files:**
- Create: `tests/test_view_classifier.py`

- [ ] **Step 1: Create test file**

```python
# tests/test_view_classifier.pycd D:\FitnessAQA\ohp_phase3\personal_videos
python C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\classify_views.py `
  --exercise overhead_press --neural --video-id 11882_1
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "exevision" / "neural" / "ohp"))

from view_classifier import (
    extract_frame_features,
    ViewClassifierMLP,
    predict_video,
    VIEW_LABELS,
    N_FEATURES,
)


def _make_frame(shoulder_width=0.3, hip_width=0.25, nose_z=-0.05, face_vis=0.9,
                l_arm_vis=0.8, r_arm_vis=0.8):
    """Build a minimal fake frame: 33 landmarks of [x, y, z, vis]."""
    frame = [[0.5, 0.5, 0.0, 0.5]] * 33
    # Shoulders
    frame[11] = [0.5 - shoulder_width / 2, 0.4, 0.0, 0.9]
    frame[12] = [0.5 + shoulder_width / 2, 0.4, 0.0, 0.9]
    # Hips
    frame[23] = [0.5 - hip_width / 2, 0.7, 0.0, 0.9]
    frame[24] = [0.5 + hip_width / 2, 0.7, 0.0, 0.9]
    # Nose
    frame[0]  = [0.5, 0.2, nose_z, face_vis]
    # Eyes
    frame[2]  = [0.48, 0.18, 0.0, face_vis]
    frame[5]  = [0.52, 0.18, 0.0, face_vis]
    # Elbows / wrists
    frame[13] = [0.3, 0.5, 0.0, l_arm_vis]
    frame[15] = [0.25, 0.6, 0.0, l_arm_vis]
    frame[14] = [0.7, 0.5, 0.0, r_arm_vis]
    frame[16] = [0.75, 0.6, 0.0, r_arm_vis]
    return frame


def test_extract_frame_features_shape():
    frame = _make_frame()
    feat = extract_frame_features(frame, face_detected=False)
    assert feat is not None
    assert feat.shape == (N_FEATURES,)
    assert feat.dtype == np.float32


def test_extract_frame_features_returns_none_for_short_frame():
    assert extract_frame_features([0.0] * 10, face_detected=False) is None


def test_extract_frame_features_values():
    frame = _make_frame(shoulder_width=0.3, l_arm_vis=0.9, r_arm_vis=0.2)
    feat = extract_frame_features(frame, face_detected=True)
    shoulder_width = feat[0]
    arm_asym = feat[5]
    face_det = feat[11]
    assert abs(shoulder_width - 0.3) < 1e-4
    assert arm_asym > 0.3        # asymmetric arms
    assert face_det == 1.0


def test_model_forward():
    model = ViewClassifierMLP()
    x = torch.randn(8, N_FEATURES)
    logits = model(x)
    assert logits.shape == (8, len(VIEW_LABELS))


def test_predict_video_returns_valid_label():
    model = ViewClassifierMLP()
    frames = [_make_frame() for _ in range(20)]
    face_list = [False] * 20
    label, conf = predict_video(model, frames, face_list)
    assert label in VIEW_LABELS
    assert 0.0 <= conf <= 1.0


def test_predict_video_empty_frames():
    model = ViewClassifierMLP()
    label, conf = predict_video(model, [], [])
    assert label == "unknown"
    assert conf == 0.0
```

- [ ] **Step 2: Run tests to confirm they all fail**

```
pytest tests/test_view_classifier.py -v
```
Expected: `ImportError: cannot import name 'extract_frame_features'` — module doesn't exist yet.

---

## Task 2: Implement model + feature extraction

**Files:**
- Create: `core/exevision/neural/ohp/view_classifier.py`

- [ ] **Step 1: Create the file**

```python
# core/exevision/neural/ohp/view_classifier.py
"""Standalone neural view classifier for OHP.

Predicts camera angle (front / back / side / front_side / back_side) from
per-frame MediaPipe landmark signals. Trained on manually annotated view labels.

Usage:
    from view_classifier import load_view_classifier, predict_video
    model = load_view_classifier("models/view_classifier_ohp.pt")
    label, confidence = predict_video(model, keypoints_img, face_detected_list)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── Label encoding ─────────────────────────────────────────────────────────────
VIEW_LABELS = ["front", "back", "side", "front_side", "back_side"]
VIEW_TO_IDX = {v: i for i, v in enumerate(VIEW_LABELS)}
N_FEATURES  = 12   # per-frame feature vector dimension

# Joint indices (MediaPipe BlazePose)
_NOSE       = 0
_L_EYE      = 2
_R_EYE      = 5
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16
_L_HIP      = 23
_R_HIP      = 24


def extract_frame_features(frame: list, face_detected: bool = False) -> Optional[np.ndarray]:
    """Extract 12 spatial signals from one frame's landmark list.

    Returns float32 array of shape (N_FEATURES,), or None if frame is unusable.

    Features:
      0  shoulder_width      — abs(L_SHOULDER.x - R_SHOULDER.x)
      1  hip_width           — abs(L_HIP.x - R_HIP.x)
      2  nose_z_rel_hip      — NOSE.z - mean(L_HIP.z, R_HIP.z)
      3  l_arm_vis           — mean(L_ELBOW.vis, L_WRIST.vis)
      4  r_arm_vis           — mean(R_ELBOW.vis, R_WRIST.vis)
      5  arm_asym            — abs(l_arm_vis - r_arm_vis)
      6  nose_vis            — NOSE.vis
      7  l_eye_vis           — L_EYE.vis
      8  r_eye_vis           — R_EYE.vis
      9  l_shoulder_vis      — L_SHOULDER.vis
      10 r_shoulder_vis      — R_SHOULDER.vis
      11 face_detected       — 1.0 if BlazeFace fired, else 0.0
    """
    if not frame or len(frame) < 25:
        return None
    has_vis = len(frame[0]) > 3

    def _vis(idx: int) -> float:
        return float(frame[idx][3]) if has_vis else 0.5

    shoulder_width = abs(frame[_L_SHOULDER][0] - frame[_R_SHOULDER][0])
    hip_width      = abs(frame[_L_HIP][0]      - frame[_R_HIP][0])
    nose_z_rel_hip = (
        frame[_NOSE][2] - (frame[_L_HIP][2] + frame[_R_HIP][2]) / 2.0
    )
    l_arm_vis  = (_vis(_L_ELBOW) + _vis(_L_WRIST)) / 2.0
    r_arm_vis  = (_vis(_R_ELBOW) + _vis(_R_WRIST)) / 2.0
    arm_asym   = abs(l_arm_vis - r_arm_vis)

    return np.array([
        shoulder_width,
        hip_width,
        nose_z_rel_hip,
        l_arm_vis,
        r_arm_vis,
        arm_asym,
        _vis(_NOSE),
        _vis(_L_EYE),
        _vis(_R_EYE),
        _vis(_L_SHOULDER),
        _vis(_R_SHOULDER),
        float(face_detected),
    ], dtype=np.float32)


# ── Model ──────────────────────────────────────────────────────────────────────

class ViewClassifierMLP(nn.Module):
    """2-layer MLP: (N_FEATURES → 64 → 32 → 5 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(VIEW_LABELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # logits (B, 5)


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_video(
    model: ViewClassifierMLP,
    keypoints_img: list,
    face_detected_list: list,
    device: str = "cpu",
) -> tuple[str, float]:
    """Classify camera view for a full video using per-frame majority vote.

    Args:
        model:             Loaded ViewClassifierMLP in eval mode.
        keypoints_img:     List of frames; each frame is a list of [x,y,z,vis] landmarks.
        face_detected_list: Per-frame BlazeFace detection flags.
        device:            Torch device string.

    Returns:
        (view_label, confidence) — confidence = fraction of frames that voted for winner.
        Returns ("unknown", 0.0) if no usable frames.
    """
    model.eval()
    frame_feats: list[np.ndarray] = []
    for i, frame in enumerate(keypoints_img or []):
        fd   = face_detected_list[i] if i < len(face_detected_list) else False
        feat = extract_frame_features(frame, fd)
        if feat is not None:
            frame_feats.append(feat)

    if not frame_feats:
        return "unknown", 0.0

    X = torch.from_numpy(np.stack(frame_feats)).to(device)
    with torch.no_grad():
        logits = model(X)
        preds  = logits.argmax(dim=1).cpu().numpy()

    counts     = np.bincount(preds, minlength=len(VIEW_LABELS))
    winner     = int(counts.argmax())
    confidence = float(counts[winner] / counts.sum())
    return VIEW_LABELS[winner], confidence


def load_view_classifier(model_path: str, device: str = "cpu") -> ViewClassifierMLP:
    """Load a saved ViewClassifierMLP checkpoint."""
    model = ViewClassifierMLP().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
```

- [ ] **Step 2: Run the tests — expect pass**

```
pytest tests/test_view_classifier.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 3: Commit**

```
git add core/exevision/neural/ohp/view_classifier.py tests/test_view_classifier.py
git commit -m "feat(ohp): add ViewClassifierMLP with per-frame feature extraction"
```

---

## Task 3: Training script

**Files:**
- Create: `core/exevision/training/ohp/train_view_classifier.py`

- [ ] **Step 1: Create the training script**

```python
# core/exevision/training/ohp/train_view_classifier.py
"""Train the OHP neural view classifier.

Loads 165 annotated reps, extracts per-frame spatial features from ALL frames
in each video (not just rep segments), trains a ViewClassifierMLP with
class-weighted cross-entropy, runs 5-fold stratified CV, then retrains on
the full dataset and saves models/view_classifier_ohp.pt.

CLI:
    python core/exevision/training/ohp/train_view_classifier.py \\
        --annotation-dir training_dataset/ohp_phase3_annotations/videos \\
        --features-dir   training_dataset/ohp_phase3_annotations/extracted_features \\
        --output         models/view_classifier_ohp.pt \\
        [--epochs 150] [--lr 1e-3] [--batch-size 256]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

_REPO = Path(__file__).resolve().parents[4]
_OHP_NEURAL = _REPO / "core" / "exevision" / "neural" / "ohp"
if str(_OHP_NEURAL) not in sys.path:
    sys.path.insert(0, str(_OHP_NEURAL))

from view_classifier import (
    ViewClassifierMLP,
    VIEW_LABELS,
    VIEW_TO_IDX,
    extract_frame_features,
)


def _load_dataset(
    annotation_dir: Path,
    features_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, video_ids) from annotation JSONs + feature JSONs.

    X: float32 (N_frames_total, 12)
    y: int64   (N_frames_total,)   — class index
    video_ids: list[str] len=N_frames_total (for stratified split at video level)
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[int]        = []
    vid_rows: list[str]      = []

    for anno_path in sorted(annotation_dir.glob("*.json")):
        try:
            anno = json.loads(anno_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_id = anno.get("video_id", anno_path.stem)

        # Ground truth: use annotated_view from first annotated rep (human-verified)
        view_label = None
        for rep in anno.get("reps", []):
            v = rep.get("annotated_view") or rep.get("view") or anno.get("view")
            if v and v in VIEW_TO_IDX:
                view_label = v
                break
        if view_label is None:
            # Fallback to top-level view (heuristic — less reliable)
            view_label = anno.get("view", "")
        if view_label not in VIEW_TO_IDX:
            continue   # unknown / unmapped label — skip

        class_idx = VIEW_TO_IDX[view_label]

        # Load feature JSON — all frames (not limited to rep segments)
        feat_path = features_dir / f"{video_id}.json"
        if not feat_path.exists():
            continue
        try:
            feat_data = json.loads(feat_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        keypoints_img    = feat_data.get("keypoints_img", [])
        face_detected_list = feat_data.get("face_detected", [])

        n_added = 0
        for i, frame in enumerate(keypoints_img):
            fd   = face_detected_list[i] if i < len(face_detected_list) else False
            feat = extract_frame_features(frame, fd)
            if feat is not None:
                X_rows.append(feat)
                y_rows.append(class_idx)
                vid_rows.append(video_id)
                n_added += 1

        if n_added == 0:
            print(f"  WARNING: 0 usable frames for {video_id}")

    if not X_rows:
        raise RuntimeError("No training samples extracted — check annotation/feature paths.")

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64)
    print(f"Dataset: {X.shape[0]} frame samples from {len(set(vid_rows))} videos")
    for i, lbl in enumerate(VIEW_LABELS):
        count = int((y == i).sum())
        print(f"  {lbl:12s}: {count} frames")
    return X, y, vid_rows


def _build_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.arange(len(VIEW_LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    label: str = "",
) -> tuple[ViewClassifierMLP, float]:
    """Train one model and return (model, val_accuracy)."""
    model     = ViewClassifierMLP().to(device)
    class_w   = _build_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)

    n = len(X_t)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            idx   = perm[start:start + batch_size]
            logits = model(X_t[idx])
            loss   = criterion(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        if ep % 30 == 0 or ep == epochs:
            print(f"  [{label}] ep {ep:3d}/{epochs} loss={ep_loss / max(1, n // batch_size):.4f}")

    # Evaluate on val set
    model.eval()
    with torch.no_grad():
        X_v  = torch.from_numpy(X_val).to(device)
        preds = model(X_v).argmax(dim=1).cpu().numpy()
    acc = float((preds == y_val).mean())
    return model, acc


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    vid_rows: list[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> None:
    """5-fold stratified CV at video level — stratify by class label."""
    # One row per video for splitting
    vids  = sorted(set(vid_rows))
    vid_y = np.array([y[vid_rows.index(v)] for v in vids])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(vids, vid_y)):
        tr_vids  = {vids[i] for i in tr_idx}
        val_vids = {vids[i] for i in val_idx}
        mask_tr  = np.array([v in tr_vids  for v in vid_rows])
        mask_val = np.array([v in val_vids for v in vid_rows])
        _, acc = _train(
            X[mask_tr], y[mask_tr], X[mask_val], y[mask_val],
            epochs, lr, batch_size, device, label=f"fold{fold}",
        )
        print(f"  Fold {fold} val accuracy: {acc:.3f}")

        # Per-class accuracy
        X_v  = torch.from_numpy(X[mask_val]).to(device)
        with torch.no_grad():
            preds = ViewClassifierMLP().to(device)  # fresh model for display only
        # Use the trained model returned from _train
        accs.append(acc)

    print(f"\n5-fold CV mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


def run_final(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    output: Path,
) -> None:
    """Retrain on all data, save model."""
    # Use a 10% hold-out just for final accuracy reporting
    from sklearn.model_selection import train_test_split
    vids = sorted(set(
        [str(i) for i in range(len(X))]  # frame-level; use stratified frame split
    ))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    model, acc = _train(X_tr, y_tr, X_val, y_val, epochs, lr, batch_size, device, label="final")
    print(f"\nFinal model held-out accuracy: {acc:.3f}")

    # Per-class breakdown
    X_v  = torch.from_numpy(X_val).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_v).argmax(dim=1).cpu().numpy()
    print("Per-class accuracy on held-out set:")
    for i, lbl in enumerate(VIEW_LABELS):
        mask = y_val == i
        if mask.sum() > 0:
            cls_acc = float((preds[mask] == i).mean())
            print(f"  {lbl:12s}: {cls_acc:.3f}  (n={mask.sum()})")

    # Retrain on ALL data for final checkpoint
    model_full, _ = _train(X, y, X[:1], y[:1], epochs, lr, batch_size, device, label="full")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_full.state_dict(), output)
    print(f"\nSaved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OHP view classifier MLP")
    parser.add_argument("--annotation-dir", type=Path,
                        default=Path("training_dataset/ohp_phase3_annotations/videos"))
    parser.add_argument("--features-dir",   type=Path,
                        default=Path("training_dataset/ohp_phase3_annotations/extracted_features"))
    parser.add_argument("--output",  type=Path, default=Path("models/view_classifier_ohp.pt"))
    parser.add_argument("--epochs",  type=int,   default=150)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cv-only",  action="store_true",
                        help="Run 5-fold CV only, skip final retrain")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, vid_rows = _load_dataset(args.annotation_dir, args.features_dir)

    run_cv(X, y, vid_rows, args.epochs, args.lr, args.batch_size, device)

    if not args.cv_only:
        run_final(X, y, args.epochs, args.lr, args.batch_size, device, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run CV to see baseline accuracy**

```powershell
$py = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe"
& $py core\exevision\training\ohp\train_view_classifier.py --cv-only
```

Expected output: prints dataset stats, per-fold accuracy. Goal: mean CV accuracy >0.70. If <0.60, the feature set or label quality needs investigation before proceeding.

- [ ] **Step 3: Train final model**

```powershell
& $py core\exevision\training\ohp\train_view_classifier.py
```

Expected: saves `models/view_classifier_ohp.pt`, prints per-class held-out accuracy.

- [ ] **Step 4: Commit**

```
git add core/exevision/training/ohp/train_view_classifier.py models/view_classifier_ohp.pt
git commit -m "feat(ohp): train view classifier MLP, save checkpoint"
```

---

## Task 4: Stage 4 integration

**Files:**
- Modify: `core/exevision/stages/classify_views.py`

The integration adds two things to `classify_views.py`:
1. A `_try_neural_classify` helper that loads the model and classifies one video
2. Two new CLI flags: `--neural` and `--confidence-threshold`

- [ ] **Step 1: Add neural helper near top of classify_views.py (after imports)**

Add after line 8 (after the `from collections import ...` import block):

```python
# ── Optional neural view classifier ────────────────────────────────────────────
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OHP_NEURAL = os.path.join(_REPO_ROOT, "core", "exevision", "neural", "ohp")
if _OHP_NEURAL not in _sys.path:
    _sys.path.insert(0, _OHP_NEURAL)

_neural_model_cache = {}   # path → loaded model


def _try_neural_classify(
    keypoints_img: list,
    face_detected_list: list,
    model_path: str,
    confidence_threshold: float = 0.7,
) -> tuple[str | None, float]:
    """Run neural view classifier. Returns (label, confidence) or (None, 0.0) on any failure."""
    if not os.path.exists(model_path):
        return None, 0.0
    try:
        import torch
        from view_classifier import load_view_classifier, predict_video

        if model_path not in _neural_model_cache:
            _neural_model_cache[model_path] = load_view_classifier(model_path)
        model = _neural_model_cache[model_path]

        label, conf = predict_video(model, keypoints_img, face_detected_list)
        if label == "unknown" or conf < confidence_threshold:
            return None, conf
        return label, conf
    except Exception:
        return None, 0.0
```

- [ ] **Step 2: Update `process_video_classification` to accept and use neural args**

Find the existing `process_video_classification` function (around line 288) and replace it:

```python
def process_video_classification(
    json_path: str,
    neural_model_path: str = "",
    confidence_threshold: float = 0.7,
) -> tuple:
    """Read JSON, classify view, update JSON, return result."""
    video_id = os.path.splitext(os.path.basename(json_path))[0]
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        keypoints_img      = data.get("keypoints_img", [])
        face_detected_list = data.get("face_detected", [])

        # Try neural classifier first if requested
        view = None
        neural_used = False
        if neural_model_path:
            neural_label, conf = _try_neural_classify(
                keypoints_img, face_detected_list,
                neural_model_path, confidence_threshold,
            )
            if neural_label is not None:
                view = neural_label
                neural_used = True

        # Fall back to heuristic
        if view is None:
            view = get_view_label(keypoints_img, face_detected_list)

        if "info" not in data:
            data["info"] = {}
        data["info"]["view"] = view
        data["info"]["view_reliable"] = neural_used or (view != "unknown")

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return video_id, "Success", view

    except Exception as e:
        return video_id, "Error", str(e)
```

- [ ] **Step 3: Update `run_classification` to pass neural args through**

Find `run_classification` (around line 321). Change its signature and the call to `process_video_classification`:

```python
def run_classification(
    quality_filter=None,
    video_id_filter=None,
    exercise="squat",
    neural_model_path: str = "",
    confidence_threshold: float = 0.7,
):
```

And update the call inside the loop from:
```python
video_id, status, view = process_video_classification(json_path)
```
to:
```python
video_id, status, view = process_video_classification(
    json_path, neural_model_path, confidence_threshold)
```

- [ ] **Step 4: Add CLI flags to `argparse` block at bottom of file**

Find the `argparse` block (around line 405). Add two arguments:

```python
parser.add_argument(
    "--neural", action="store_true",
    help="Use neural view classifier (falls back to heuristic if model absent or low confidence)",
)
parser.add_argument(
    "--neural-model", default="models/view_classifier_ohp.pt",
    help="Path to view_classifier_ohp.pt (default: models/view_classifier_ohp.pt)",
)
parser.add_argument(
    "--confidence-threshold", type=float, default=0.7,
    help="Neural confidence below this → use heuristic fallback (default: 0.7)",
)
```

And update the `run_classification` call at the bottom to pass the new args:

```python
run_classification(
    quality_filter=args.quality,
    video_id_filter=args.video_id,
    exercise=args.exercise,
    neural_model_path=args.neural_model if args.neural else "",
    confidence_threshold=args.confidence_threshold,
)
```

- [ ] **Step 5: Smoke-test Stage 4 with neural flag on one video**

```powershell
$py = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe"
$ws = "D:\FitnessAQA\ohp_phase3\personal_videos"
& $py core\exevision\stages\classify_views.py `
    --exercise overhead_press `
    --neural `
    --video-id "video_2026-05-12_13-56-51 (2)"
```

Expected: prints classified view + no Python errors.

- [ ] **Step 6: Commit**

```
git add core/exevision/stages/classify_views.py
git commit -m "feat(stage4): add --neural flag for confidence-gated neural view classification"
```

---

## Task 5: Accuracy gate — decide replace vs supplement

After training (Task 3) and initial classification test (Task 4):

- [ ] **Step 1: Check CV accuracy output from Task 3 Step 2**

Decision table:

| CV mean accuracy | Action |
|-----------------|--------|
| ≥ 0.85          | Lower `--confidence-threshold` to 0.5 in `run_classification` default; neural is primary |
| 0.70–0.84       | Keep default threshold 0.7; neural fires only when confident |
| < 0.70          | Investigate per-class accuracy — likely `side` or `front` starved; collect more annotations for those views before deploying |

- [ ] **Step 2: Update CLAUDE.md with neural view classifier info**

Add to Active Components section under Stage 4 entry:
```
- **Neural view classifier (opt-in):** `core/exevision/neural/ohp/view_classifier.py`
  - Checkpoint: `models/view_classifier_ohp.pt`
  - Activate via `--neural` flag on Stage 4
  - Trained on 165 annotated OHP videos, per-frame spatial signals
  - Falls back to heuristic if confidence < threshold (default 0.7)
```

- [ ] **Step 3: Commit**

```
git add CLAUDE.md
git commit -m "docs: document neural view classifier in CLAUDE.md"
```

---

## Running order

```
Task 1 → Task 2 → Task 3 (CV run) → Task 4 → Task 5
```

Task 3 CV results inform Task 5 threshold decision. Tasks 1–2 have no external dependencies.
