from __future__ import annotations

import sys
from pathlib import Path

_NEURAL_ROOT = Path(__file__).resolve().parents[1]
if str(_NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEURAL_ROOT))

from nn_models import HeuristicGuidedFusion   # reused unchanged — heuristic_dim is a param

OHP_HEURISTIC_DIM = 16   # must match build_ohp_heuristic_vector output length


def build_ohp_fusion(neural_dim: int = 256, fusion_dim: int = 64) -> HeuristicGuidedFusion:
    """Return a HeuristicGuidedFusion configured for OHP's 16-dim heuristic vector."""
    return HeuristicGuidedFusion(
        heuristic_dim=OHP_HEURISTIC_DIM,
        neural_dim=neural_dim,
        fusion_dim=fusion_dim,
    )
