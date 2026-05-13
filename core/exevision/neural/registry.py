from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


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
        "ckpt_dir": "models",
        "bilstm_ckpt_name": "bilstm_finetuned.pt",
        "stgcn_ckpt_name": "stgcn_finetuned.pt",
        "fusion_ckpt_name": "fusion_layer.pt",
        "suppress_knee": False,
        "grip_ratio_side_exclude": False,
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
