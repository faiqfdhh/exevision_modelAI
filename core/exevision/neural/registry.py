from __future__ import annotations

from typing import Any, Dict, Type


def _lazy_squat():
    from core.exevision.neural.nn_models import BiLSTMScorer, STGCNScorer
    return {"bilstm": BiLSTMScorer, "stgcn": STGCNScorer}


def _lazy_ohp():
    from core.exevision.neural.ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
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
