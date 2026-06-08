from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

_NEURAL_ROOT = Path(__file__).resolve().parents[1]
if str(_NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEURAL_ROOT))

from nn_models import HeuristicGuidedFusion   # reused unchanged — heuristic_dim is a param

OHP_HEURISTIC_DIM = 16   # must match build_ohp_heuristic_vector output length

# Fixed order of neural head scalars fed into the fusion's optional head branch
# (v2 fusion, head_dim=7). MUST match build_ohp_head_vector exactly — order drift
# silently corrupts a trained fusion's inputs.
HEAD_SCALAR_ORDER = [
    "smoothness", "control", "lockout", "elbow_flare", "grip_ratio", "rom_top", "rom_bottom",
]
OHP_HEAD_DIM = len(HEAD_SCALAR_ORDER)   # 7


def build_ohp_fusion(neural_dim: int = 256, fusion_dim: int = 64, head_dim: int = 0) -> HeuristicGuidedFusion:
    """Return a HeuristicGuidedFusion configured for OHP's 16-dim heuristic vector.

    head_dim=0 (default) builds the original v1 architecture (loads
    fusion_ohp_finetuned*.pt). head_dim=OHP_HEAD_DIM builds the v2 architecture
    that also consumes neural head scalars (loads fusion_ohp_v2*.pt) — see
    build_ohp_head_vector.
    """
    return HeuristicGuidedFusion(
        heuristic_dim=OHP_HEURISTIC_DIM,
        neural_dim=neural_dim,
        fusion_dim=fusion_dim,
        head_dim=head_dim,
    )


def build_ohp_head_vector(b_out: Dict[str, torch.Tensor], s_out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Assemble the detached 7-dim head-scalar vector for v2 fusion input.

    Order: [smoothness, control, lockout, elbow_flare, grip_ratio, rom_top, rom_bottom],
    each scaled to roughly [0, 1] (lockout is already a probability; the rest are
    /100 score-head outputs). Detached — encoders/heads stay frozen during fusion
    training, this branch must not backprop into them.
    """
    parts = [
        b_out["smoothness"] / 100.0,
        b_out["control"] / 100.0,
        s_out["lockout"],
        s_out["elbow_flare"] / 100.0,
        s_out["grip_ratio"] / 100.0,
        s_out["rom_top"] / 100.0,
        s_out["rom_bottom"] / 100.0,
    ]
    return torch.stack(parts, dim=-1).detach()
