import numpy as np
import torch

from core.exevision.neural.ohp.view_classifier import (
    extract_frame_features,
    ViewClassifierMLP,
    predict_video,
    VIEW_LABELS,
    N_FEATURES,
)


def _make_frame(
    shoulder_width=0.3,
    hip_width=0.25,
    nose_z=-0.05,
    face_vis=0.9,
    l_arm_vis=0.8,
    r_arm_vis=0.8,
):
    """Build a minimal fake frame: 33 landmarks of [x, y, z, vis]."""
    frame = [[0.5, 0.5, 0.0, 0.5]] * 33
    # Shoulders
    frame[11] = [0.5 - shoulder_width / 2, 0.4, 0.0, 0.9]
    frame[12] = [0.5 + shoulder_width / 2, 0.4, 0.0, 0.9]
    # Hips
    frame[23] = [0.5 - hip_width / 2, 0.7, 0.0, 0.9]
    frame[24] = [0.5 + hip_width / 2, 0.7, 0.0, 0.9]
    # Nose
    frame[0] = [0.5, 0.2, nose_z, face_vis]
    # Eyes
    frame[2] = [0.48, 0.18, 0.0, face_vis]
    frame[5] = [0.52, 0.18, 0.0, face_vis]
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
    assert arm_asym > 0.3
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
