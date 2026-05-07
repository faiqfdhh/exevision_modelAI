from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Quality target uses heuristic score directly — decoupled from error labels.
# Deriving quality from binary error labels creates bimodal targets that break
# quality regression (MAE spikes to 25+). Error head is trained separately.


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
    """Derive labels for one standing OHP rep.

    Quality target = heuristic_score directly (neural model refines heuristic).
    Knee error = binary 1.0/0.0 from FitnessAQA error windows.

    Decoupling quality from error labels avoids bimodal targets that break
    quality regression (MAE spikes when binary error → quality formula is used).
    """
    knee = compute_binary_overlap(rep_start_sec, rep_end_sec, knee_windows)
    overall = max(0.0, min(100.0, float(heuristic_score)))

    return RepLabels(
        overall_score=round(overall, 4),
        knee_error=round(knee, 6),
    )
