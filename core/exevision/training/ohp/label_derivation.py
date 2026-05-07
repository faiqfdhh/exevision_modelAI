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
    elbow_error_soft: float   # 0.0 (no error) or 1.0 (error detected) — binary label
    knee_error_soft: float    # 0.0 (no error) or 1.0 (error detected) — binary label (always 0.0 for seated)


def compute_overlap_ratio(
    rep_start_sec: float,
    rep_end_sec: float,
    error_windows: List[List[float]],
) -> float:
    """Return binary error label: 1.0 if ANY part of rep overlaps with error_windows, else 0.0.

    FitnessAQA paper uses binary classification with class weighting, not soft labels.
    """
    rep_dur = rep_end_sec - rep_start_sec
    if rep_dur <= 0.0 or not error_windows:
        return 0.0

    # Check if any error window overlaps with the rep window
    for window in error_windows:
        w_start, w_end = float(window[0]), float(window[1])
        if w_end > rep_start_sec and w_start < rep_end_sec:
            return 1.0  # Any overlap → error detected

    return 0.0  # No overlap → no error


def derive_rep_labels(
    rep_start_sec: float,
    rep_end_sec: float,
    elbow_windows: List[List[float]],
    knee_windows: List[List[float]],
    heuristic_score: float,
    seated: bool,
) -> RepLabels:
    """Derive binary training labels for one OHP rep from FitnessAQA error windows.

    Binary classification: 1.0 if error detected (any overlap), 0.0 if no error.
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
