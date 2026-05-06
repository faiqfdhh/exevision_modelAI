from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# MediaPipe indices used by the project (11 active joints after foot consolidation)
ACTIVE_JOINTS = [0, 1, 2, 11, 12, 23, 24, 25, 26, 27, 28]
NUM_ACTIVE_JOINTS = 11

# Mapping from MediaPipe index to local index (0-10)
MP_TO_LOCAL = {mp_idx: local_idx for local_idx, mp_idx in enumerate(ACTIVE_JOINTS)}

# Bone edges using LOCAL indices (not MediaPipe indices)
BONE_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 4),
    (3, 5), (4, 6),
    (5, 6),
    (5, 7), (6, 8),
    (7, 9), (8, 10),
]

# Biomechanical symmetry edges (domain knowledge for squat analysis)
SYMMETRY_EDGES = [
    (7, 8),
    (9, 10),
]

ALL_EDGES = BONE_EDGES + SYMMETRY_EDGES

# BiLSTM signal channels
BILSTM_SIGNAL_KEYS = [
    "normalized_hip_displacement",
    "window_velocity",
    "knee_angles",
    "landmark_confidence",
]
NUM_BILSTM_CHANNELS = len(BILSTM_SIGNAL_KEYS)

# Fixed sequence length for padding/truncation
FIXED_SEQ_LEN = 128

# ST-GCN input channels per joint: x, y, z, visibility, vx, vy, vz
STGCN_CHANNELS = 7


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_float_array(values: Iterable) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        return array
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array.astype(np.float32, copy=False)


def _safe_slice(array: np.ndarray, start: int, end: int) -> np.ndarray:
    if array.size == 0:
        return array
    start = max(0, int(start))
    end = min(int(end), len(array) - 1)
    if end < start:
        return array[:0]
    return array[start : end + 1]


@lru_cache(maxsize=16)
def _discover_feature_index(features_dir_str: str) -> Dict[str, Path]:
    features_dir = _as_path(features_dir_str)
    index: Dict[str, Path] = {}
    if not features_dir.exists():
        return index
    for json_path in sorted(features_dir.rglob("*.json")):
        if json_path.name.endswith("_segmented.json"):
            continue
        index.setdefault(json_path.stem, json_path)
    return index


def _discover_segmented_files(segmented_dir: str | Path) -> List[Path]:
    segmented_root = _as_path(segmented_dir)
    if not segmented_root.exists():
        return []
    return sorted(path for path in segmented_root.rglob("*_segmented.json") if path.is_file())


def _extract_rep_matrix(seg_data: dict, rep: dict) -> Optional[np.ndarray]:
    signals = seg_data.get("signals", {})
    arrays = []
    for key in BILSTM_SIGNAL_KEYS:
        values = signals.get(key, [])
        arrays.append(_to_float_array(values))

    if not arrays or any(array.size == 0 for array in arrays):
        return None

    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", -1))
    sliced = [_safe_slice(array, start, end) for array in arrays]
    if any(array.size == 0 for array in sliced):
        return None

    min_len = min(len(array) for array in sliced)
    if min_len <= 0:
        return None
    stacked = np.stack([array[:min_len] for array in sliced], axis=-1).astype(np.float32, copy=False)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    # Per-channel z-score normalization: removes unit/scale disparity between channels
    # (e.g. knee_angles in degrees [88-175] vs hip_displacement in [0-0.05])
    for ch in range(stacked.shape[-1]):
        col = stacked[:, ch]
        std = col.std()
        stacked[:, ch] = (col - col.mean()) / std if std > 1e-8 else col - col.mean()
    return stacked


def load_bilstm_reps(features_dir, segmented_dir, max_videos=None):
    """
    Load all reps as temporal signal matrices for BiLSTM pre-training.

    Returns: list of numpy arrays, each shape (T_rep, 4) where 4 = number of signal channels.
    Skips videos with 0 reps.
    """
    del features_dir
    reps: List[np.ndarray] = []
    processed_videos = 0
    for segmented_path in _discover_segmented_files(segmented_dir):
        if max_videos is not None and processed_videos >= max_videos:
            break
        seg_data = _load_json(segmented_path)
        processed_videos += 1
        repetitions = seg_data.get("repetitions", []) or []
        if not repetitions:
            continue
        for rep in repetitions:
            rep_matrix = _extract_rep_matrix(seg_data, rep)
            if rep_matrix is None:
                continue
            reps.append(rep_matrix)
    return reps


def _extract_active_joints(frame: np.ndarray) -> np.ndarray:
    joints = np.zeros((NUM_ACTIVE_JOINTS, frame.shape[-1]), dtype=np.float32)
    for mp_idx, local_idx in MP_TO_LOCAL.items():
        if mp_idx < frame.shape[0]:
            joints[local_idx] = frame[mp_idx]
    return joints


def _normalize_stgcn_sequence(sequence: np.ndarray, body_scale: float) -> np.ndarray:
    normalized = sequence.astype(np.float32, copy=True)
    if body_scale is None or not np.isfinite(body_scale) or body_scale <= 0:
        body_scale = 1.0
    normalized[:, :, :3] /= float(body_scale)
    for frame_idx in range(normalized.shape[0]):
        hips = normalized[frame_idx, [5, 6], :3]
        if hips.shape[0] < 2:
            continue
        hip_mid = hips.mean(axis=0)
        normalized[frame_idx, :, :3] -= hip_mid
    return normalized


def _compute_velocity(sequence: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(sequence[:, :, :3], dtype=np.float32)
    if sequence.shape[0] <= 1:
        return velocity
    diffs = (sequence[1:, :, :3] - sequence[:-1, :, :3]) * float(fps)
    velocity[1:] = diffs
    return velocity


def _extract_stgcn_rep(seg_data: dict, feature_data: dict, rep: dict) -> Optional[np.ndarray]:
    keypoints_img = feature_data.get("keypoints_img", []) or []
    if not keypoints_img:
        return None

    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", -1))
    if end < start:
        return None
    end = min(end, len(keypoints_img) - 1)
    start = max(0, start)
    if start >= len(keypoints_img) or end < start:
        return None

    rep_frames = keypoints_img[start : end + 1]
    if not rep_frames:
        return None

    frame_arrays = []
    for frame in rep_frames:
        frame_array = np.asarray(frame, dtype=np.float32)
        if frame_array.ndim != 2 or frame_array.shape[1] < 4:
            return None
        joints = _extract_active_joints(frame_array)
        frame_arrays.append(joints)

    sequence = np.stack(frame_arrays, axis=0).astype(np.float32, copy=False)
    sequence[:, :, :4] = np.nan_to_num(sequence[:, :, :4], nan=0.0, posinf=0.0, neginf=0.0)

    body_scale = seg_data.get("info", {}).get("calibration", {}).get("body_scale", 1.0)
    fps = float(feature_data.get("info", {}).get("fps", seg_data.get("info", {}).get("fps", 30.0)))

    sequence = _normalize_stgcn_sequence(sequence, body_scale)
    velocity = _compute_velocity(sequence, fps)

    visibility = sequence[:, :, 3:4]
    normalized = np.concatenate([sequence[:, :, :3], visibility, velocity], axis=-1).astype(np.float32, copy=False)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _build_feature_for_video(features_index: Dict[str, Path], video_id: str) -> Optional[dict]:
    feature_path = features_index.get(video_id)
    if feature_path is None or not feature_path.exists():
        return None
    return _load_json(feature_path)


def load_stgcn_reps(features_dir, segmented_dir, max_videos=None):
    """
    Load all reps as skeleton sequences for ST-GCN pre-training.

    Returns: list of numpy arrays, each shape (T_rep, 11, 7) where:
        - 11 = active joints
        - 7 = x, y, z, visibility, vx, vy, vz
    """
    grouped = load_stgcn_reps_by_video(features_dir, segmented_dir, max_videos=max_videos)
    reps: List[np.ndarray] = []
    for rep_list in grouped.values():
        reps.extend(rep_list)
    return reps


def load_stgcn_reps_by_video(features_dir, segmented_dir, max_videos=None):
    """
    Same as load_stgcn_reps but returns dict: video_id -> list of rep arrays.

    Needed for contrastive learning (positive pairs = reps from same video).

    Returns: dict[str, list[np.ndarray]]  where each array is (T_rep, 11, 7)
    """
    features_index = _discover_feature_index(str(_as_path(features_dir)))
    grouped: Dict[str, List[np.ndarray]] = {}
    processed_videos = 0

    for segmented_path in _discover_segmented_files(segmented_dir):
        if max_videos is not None and processed_videos >= max_videos:
            break
        seg_data = _load_json(segmented_path)
        processed_videos += 1
        repetitions = seg_data.get("repetitions", []) or []
        if not repetitions:
            continue

        video_id = segmented_path.stem.replace("_segmented", "")
        feature_data = _build_feature_for_video(features_index, video_id)
        if feature_data is None:
            continue

        rep_list = grouped.setdefault(video_id, [])
        for rep in repetitions:
            rep_sequence = _extract_stgcn_rep(seg_data, feature_data, rep)
            if rep_sequence is None:
                continue
            rep_list.append(rep_sequence)
        if not rep_list:
            grouped.pop(video_id, None)
    return grouped


def pad_or_truncate(seq, target_len=FIXED_SEQ_LEN):
    """
    Pad (with zeros) or truncate a sequence to fixed length.
    Input: (T, ...) numpy array
    Output: (target_len, ...) numpy array
    """
    array = np.asarray(seq, dtype=np.float32)
    if array.shape[0] >= target_len:
        return array[:target_len].astype(np.float32, copy=False)
    pad_shape = (target_len - array.shape[0],) + array.shape[1:]
    padding = np.zeros(pad_shape, dtype=np.float32)
    return np.concatenate([array, padding], axis=0).astype(np.float32, copy=False)


def build_adjacency_matrix():
    """
    Build the normalized adjacency matrix for the 11-joint skeleton graph.

    Returns: numpy array of shape (11, 11), float32

    Steps:
        1. Create binary adjacency A from ALL_EDGES (undirected: add both directions)
        2. Add self-loops: A = A + I
        3. Degree-normalize: D^{-1/2} @ A @ D^{-1/2}
    """
    adjacency = np.zeros((NUM_ACTIVE_JOINTS, NUM_ACTIVE_JOINTS), dtype=np.float32)
    for src, dst in ALL_EDGES:
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    adjacency += np.eye(NUM_ACTIVE_JOINTS, dtype=np.float32)
    degrees = adjacency.sum(axis=1)
    inv_sqrt_deg = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inv_sqrt_deg[nonzero] = np.power(degrees[nonzero], -0.5)
    normalized = inv_sqrt_deg[:, None] * adjacency * inv_sqrt_deg[None, :]
    return normalized.astype(np.float32, copy=False)
