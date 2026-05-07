from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Knee-only formula (elbow labels in FitnessAQA dataset are too subtle/strict to be useful)
_KNEE_PENALTY_WEIGHT = 1.0     # full penalty when knee error detected
_FITNESSAQA_BLEND = 0.70       # weight for error_score in final blend
_HEURISTIC_BLEND = 0.30        # weight for heuristic_score in final blend


@dataclass(frozen=True)
class RepLabels:
    overall_score: float       # 0–100
    knee_error: float          # 0.0 (no error) or 1.0 (error detected) — binary label


def compute_binary_overlap(
    rep_start_sec: float,
    rep_end_sec: float,
    error_windows: List[List[float]],
) -> float:
    """Return 1.0 if ANY part of rep overlaps with error_windows, else 0.0.

    Per FitnessAQA paper: binary classification with class weighting,
    not soft labels.
    """
    rep_dur = rep_end_sec - rep_start_sec
    if rep_dur <= 0.0 or not error_windows:
        return 0.0

    for window in error_windows:
        w_start, w_end = float(window[0]), float(window[1])
        if w_end > rep_start_sec and w_start < rep_end_sec:
            return 1.0

    return 0.0


def derive_rep_labels(
    rep_start_sec: float,
    rep_end_sec: float,
    knee_windows: List[List[float]],
    heuristic_score: float,
) -> RepLabels:
    """Derive binary knee error label and overall quality score for one standing OHP rep.

    Standing OHP only — seated variant is excluded from FitnessAQA fine-tuning
    because leg landmarks are zeroed and there's no useful FitnessAQA signal.
    """
    knee = compute_binary_overlap(rep_start_sec, rep_end_sec, knee_windows)

    error_score = 100.0 * (1.0 - _KNEE_PENALTY_WEIGHT * knee)
    overall = _FITNESSAQA_BLEND * error_score + _HEURISTIC_BLEND * heuristic_score
    overall = max(0.0, min(100.0, overall))

    return RepLabels(
        overall_score=round(overall, 4),
        knee_error=round(knee, 6),
    )
