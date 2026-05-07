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
