from __future__ import annotations

from typing import Any, Callable, Dict, Type

import numpy as np


def _lazy_squat():
    from nn_models import BiLSTMScorer, STGCNScorer
    return {"bilstm": BiLSTMScorer, "stgcn": STGCNScorer}


def _lazy_ohp():
    from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
    return {"bilstm": OHPBiLSTMScorer, "stgcn": OHPSTGCNScorer}


# Registry maps exercise name → {"bilstm": class, "stgcn": class}
# To add a new exercise: add one entry here + create neural/<exercise>/models.py
_REGISTRY_FACTORIES = {
    "squat":                   _lazy_squat,
    "overhead_press":          _lazy_ohp,
    "seated_overhead_press":   _lazy_ohp,  # same model; knee_error_prob suppressed at inference
}


def get_model_classes(exercise: str) -> Dict[str, Type[Any]]:
    """Return {"bilstm": Class, "stgcn": Class} for the given exercise.

    Raises KeyError for unknown exercises.
    """
    factory = _REGISTRY_FACTORIES.get(exercise)
    if factory is None:
        raise KeyError(f"No neural registry entry for exercise '{exercise}'. "
                       f"Known: {list(_REGISTRY_FACTORIES)}")
    return factory()


# ── Exercise inference handlers ──────────────────────────────────────────────
# Each handler is a dict of callables/constants that fully configure neural
# inference for one exercise.  Adding a new exercise requires one entry here
# (plus model classes in the registry above and optional post-processing in
# neural_fusion_inference.py).

ExerciseHandler = Dict[str, Any]


def _squat_gbm_features(heuristic_vec, result: dict) -> np.ndarray:
    """14-feature GBM vector for squat: 5 heads×100, heuristic×100, view one-hot."""
    return np.array([[
        float(heuristic_vec[0]) * 100.0,
        result.get("smoothness", 0.0) * 100.0,
        result.get("control", 0.0) * 100.0,
        result.get("depth", 0.0) * 100.0,
        result.get("forward_lean", 0.0) * 100.0,
        result.get("knee_tracking", 0.0) * 100.0,
        float(heuristic_vec[1]) * 100.0,
        float(heuristic_vec[2]) * 100.0,
        float(heuristic_vec[3]) * 100.0,
        float(heuristic_vec[10]),
        float(heuristic_vec[11]),
        float(heuristic_vec[12]),
        float(heuristic_vec[13]),
        float(heuristic_vec[14]),
    ]], dtype=np.float64)


def _ohp_gbm_features(heuristic_vec, result: dict) -> np.ndarray:
    """17-feature GBM vector for OHP — moved from neural_fusion_inference.py hardcode."""
    return np.array([[
        float(heuristic_vec[0]) * 100.0,
        result.get("smoothness", 0.0), result.get("control", 0.0),
        result.get("lockout", 0.0), result.get("elbow_flare", 0.0),
        result.get("grip_ratio", 0.0), result.get("rom_top", 0.0), result.get("rom_bottom", 0.0),
        float(heuristic_vec[1]) * 100.0, float(heuristic_vec[2]) * 100.0,
        float(heuristic_vec[3]) * 100.0, float(heuristic_vec[4]) * 100.0,
        float(heuristic_vec[11]), float(heuristic_vec[12]), float(heuristic_vec[13]),
        float(heuristic_vec[14]), float(heuristic_vec[15]),
    ]], dtype=np.float64)


def _handler_squat() -> ExerciseHandler:
    from nn_models import HeuristicGuidedFusion, build_heuristic_vector
    from nn_utils import build_adjacency_matrix
    return {
        "exercise": "squat",
        "adjacency_fn": build_adjacency_matrix,
        "fusion_builder": lambda: HeuristicGuidedFusion(),
        "heuristic_fn": build_heuristic_vector,
        "heuristic_dim": 15,
        "view_vec_slice": (10, 15),
        "ckpt_dir": "models/runtime_neural_squat",
        "bilstm_ckpt_name": "bilstm_finetuned.pt",
        "stgcn_ckpt_name": "stgcn_finetuned.pt",
        "fusion_ckpt_name": "fusion_layer.pt",
        "suppress_knee": False,
        "grip_ratio_side_exclude": False,
        "gbm_feature_fn": _squat_gbm_features,
    }


def _handler_ohp() -> ExerciseHandler:
    from ohp.fusion import build_ohp_fusion
    from ohp.heuristic_vec import build_ohp_heuristic_vector
    from nn_utils import build_adjacency_matrix_ohp
    return {
        "exercise": "overhead_press",
        "adjacency_fn": build_adjacency_matrix_ohp,
        "fusion_builder": build_ohp_fusion,
        "heuristic_fn": build_ohp_heuristic_vector,
        "heuristic_dim": 16,
        "view_vec_slice": (11, 16),
        "ckpt_dir": "models/runtime_neural_ohp",
        "bilstm_ckpt_name": "bilstm_ohp_finetuned.pt",
        "stgcn_ckpt_name": "stgcn_ohp_finetuned.pt",
        "fusion_ckpt_name": "fusion_ohp_finetuned.pt",
        "suppress_knee": False,
        "grip_ratio_side_exclude": True,
        # 5-seed ensemble (Session 2026-06-08): auto-loads *_seed*.pt triples from
        # ckpt_dir if present, else falls back to the single *_finetuned.pt model.
        # All seeds vote on every head; seed7's fusion is dropped from the QUALITY
        # average only (its fusion converged poorly — best val at epoch 1).
        # Test (23 reps): lockout_auc 0.742→0.780 (gate cleared), quality_pearson
        # 0.546→0.550, all spatial MAEs improved; quality_mae 8.77→9.01 (noise).
        #
        # ── v2 fusion experiment (Session 2026-06-08, quality_mae focus) ──
        # TRIED & REJECTED: retrained fusion only (head_dim=7 head-scalar input +
        # bucket-weighted MSE, 3x on quality<60) → quality_mae 9.01→14.60,
        # quality_pearson 0.550→0.006 (ranking collapsed). Bucket weighting on
        # ~12 low-score training reps caused overfitting/distortion, not a fix.
        # fusion_ohp_v2_seed*.pt checkpoints kept on disk for analysis but NOT
        # wired in — handler stays on v1 fusion_ohp_finetuned*.pt (locked-in,
        # all 8 gates pass). See CHANGELOG.md for full diagnosis + next ideas
        # (GBM meta-learner / lighter bucket weight / more low-score data).
        "ensemble": True,
        "fusion_exclude_seeds": ["_seed7"],
        # Phase C (Session 2026-06-08): LightGBM quality meta-learner trained on
        # heuristic anchor + per-metric heuristic scores + view one-hot + the
        # ensemble's OWN predicted heads (excluding quality). Trees ignore the bad
        # heuristic anchor (heuristic_pearson=-0.11) and fix the 40-60 bucket blind
        # spot directly from data instead of loss-shaping. WINS vs locked-in
        # ensemble: quality_mae 9.01→7.30, quality_pearson 0.550→0.717, 40-60 bucket
        # MAE 27.07→16.32 (alpha sweep on val picked alpha=0 — pure GBM). Replaces
        # neural quality_score outright when present; absent file → pure neural
        # (same fallback pattern as "ensemble"). See train_quality_gbm.py.
        "quality_gbm_name": "quality_gbm.pkl",
        "quality_gbm_meta_name": "quality_gbm_meta.json",
        "gbm_feature_fn": _ohp_gbm_features,
    }


_EXERCISE_HANDLER_FACTORIES: Dict[str, Callable[[], ExerciseHandler]] = {
    "squat": _handler_squat,
    "overhead_press": _handler_ohp,
    "seated_overhead_press": _handler_ohp,
}


def get_exercise_handler(exercise: str) -> ExerciseHandler:
    """Return handler dict with all exercise-specific config for neural inference.

    To add a new exercise:
      1. Add model classes via ``_REGISTRY_FACTORIES`` (above).
      2. Add a ``_handler_<name>()`` factory here.
      3. Add it to ``_EXERCISE_HANDLER_FACTORIES``.
      4. Optionally add post-processing in ``neural_fusion_inference.py``.
    """
    factory = _EXERCISE_HANDLER_FACTORIES.get(exercise)
    if factory is None:
        raise KeyError(
            f"No neural handler for exercise '{exercise}'. "
            f"Known: {list(_EXERCISE_HANDLER_FACTORIES)}"
        )
    handler = factory()
    handler["exercise"] = exercise
    if exercise == "seated_overhead_press":
        handler["suppress_knee"] = True
    return handler
