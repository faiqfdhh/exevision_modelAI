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
