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
