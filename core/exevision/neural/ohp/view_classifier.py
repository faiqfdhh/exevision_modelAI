"""Neural view classifier for overhead press (OHP).

Predicts camera view labels from per-frame MediaPipe landmark signals.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Label encoding
VIEW_LABELS = ["front", "back", "side", "front_side", "back_side"]
VIEW_TO_IDX = {v: i for i, v in enumerate(VIEW_LABELS)}
N_FEATURES = 12

# BlazePose landmark indices
_NOSE = 0
_L_EYE = 2
_R_EYE = 5
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW = 13
_R_ELBOW = 14
_L_WRIST = 15
_R_WRIST = 16
_L_HIP = 23
_R_HIP = 24


def _vis(frame: list, idx: int, has_vis: bool) -> float:
    if has_vis:
        return float(frame[idx][3])
    return 0.0


def extract_frame_features(frame: list, face_detected: bool = False) -> Optional[np.ndarray]:
    """Extract per-frame signals used by the OHP view classifier.

    Returns float32 array of shape (N_FEATURES,), or None if frame is unusable.
    """
    if not frame or len(frame) < (_R_HIP + 1):
        return None
    has_vis = len(frame[0]) > 3

    shoulder_width = abs(frame[_L_SHOULDER][0] - frame[_R_SHOULDER][0])
    hip_width = abs(frame[_L_HIP][0] - frame[_R_HIP][0])
    nose_z_rel_hip = frame[_NOSE][2] - (frame[_L_HIP][2] + frame[_R_HIP][2]) / 2.0

    l_arm_vis = (_vis(frame, _L_ELBOW, has_vis) + _vis(frame, _L_WRIST, has_vis)) / 2.0
    r_arm_vis = (_vis(frame, _R_ELBOW, has_vis) + _vis(frame, _R_WRIST, has_vis)) / 2.0
    arm_asym = abs(l_arm_vis - r_arm_vis)

    return np.array(
        [
            shoulder_width,
            hip_width,
            nose_z_rel_hip,
            l_arm_vis,
            r_arm_vis,
            arm_asym,
            _vis(frame, _NOSE, has_vis),
            _vis(frame, _L_EYE, has_vis),
            _vis(frame, _R_EYE, has_vis),
            _vis(frame, _L_SHOULDER, has_vis),
            _vis(frame, _R_SHOULDER, has_vis),
            float(face_detected),
        ],
        dtype=np.float32,
    )


class ViewClassifierMLP(nn.Module):
    """Simple MLP for view classification."""

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
        return self.net(x)


def predict_video(
    model: ViewClassifierMLP,
    keypoints_img: list,
    face_detected_list: list | None = None,
    device: str = "cpu",
) -> tuple[str, float]:
    """Predict view label and confidence using per-frame majority vote."""
    face_detected_list = face_detected_list or []
    frame_feats: list[np.ndarray] = []
    for idx, frame in enumerate(keypoints_img or []):
        face_detected = face_detected_list[idx] if idx < len(face_detected_list) else False
        feat = extract_frame_features(frame, face_detected)
        if feat is not None:
            frame_feats.append(feat)

    if not frame_feats:
        return "unknown", 0.0

    model = model.to(device)
    model.eval()
    X = torch.from_numpy(np.stack(frame_feats)).to(device)
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()

    counts = np.bincount(preds, minlength=len(VIEW_LABELS))
    winner = int(counts.argmax())
    confidence = float(counts[winner] / counts.sum())
    return VIEW_LABELS[winner], confidence


def load_view_classifier(model_path: str, device: str = "cpu") -> ViewClassifierMLP:
    """Load a saved ViewClassifierMLP checkpoint."""
    model = ViewClassifierMLP().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
