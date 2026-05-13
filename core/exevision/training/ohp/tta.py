"""Test-Time Augmentation for OHP Phase 3 ensemble inference.

4 variants per rep:
  1. Original
  2. Horizontal flip (swap left/right joints using LOCAL positional indices in OHP_ACTIVE_JOINTS)
  3. +1 frame temporal jitter
  4. -1 frame temporal jitter

IMPORTANT — index mapping:
  OHP_ACTIVE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
  Local position:      [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9]

  stgcn_t shape: (C, T, J) where J=10 (local indices 0-9, NOT raw MediaPipe indices).
  The LR swap must use local (positional) indices — not raw MediaPipe landmark indices.
  Using raw MediaPipe indices (11-26) would silently skip all swaps (all ≥ J=10).
"""
from __future__ import annotations

import torch

# MediaPipe landmark index → raw LR swap partner (for reference / documentation only)
OHP_LR_SWAP_INDICES_MEDIAPIPE = {
    11: 12, 12: 11,  # shoulders
    13: 14, 14: 13,  # elbows
    15: 16, 16: 15,  # wrists
    23: 24, 24: 23,  # hips
    25: 26, 26: 25,  # knees
    27: 28, 28: 27,  # ankles  (not in OHP active joints — kept for completeness)
}

# OHP_ACTIVE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
# Local positions    =  0   1   2   3   4   5   6   7   8   9
# LR swap: 0↔1 (shoulders), 2↔3 (elbows), 4↔5 (wrists), 6↔7 (hips), 8↔9 (knees)
OHP_LR_SWAP_LOCAL: list[tuple[int, int]] = [
    (0, 1),   # shoulders: local pos 0 (mp11) ↔ local pos 1 (mp12)
    (2, 3),   # elbows:    local pos 2 (mp13) ↔ local pos 3 (mp14)
    (4, 5),   # wrists:    local pos 4 (mp15) ↔ local pos 5 (mp16)
    (6, 7),   # hips:      local pos 6 (mp23) ↔ local pos 7 (mp24)
    (8, 9),   # knees:     local pos 8 (mp25) ↔ local pos 9 (mp26)
]


def _horizontal_flip_stgcn(stgcn_t: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints in a (C, T, J) ST-GCN tensor.

    Uses OHP_LR_SWAP_LOCAL (positional indices 0-9 within the J dimension).
    """
    flipped = stgcn_t.clone()
    for l_pos, r_pos in OHP_LR_SWAP_LOCAL:
        # stgcn_t shape: (B, C, T, J) or (C, T, J) — swap along last dim
        flipped[..., l_pos] = stgcn_t[..., r_pos]
        flipped[..., r_pos] = stgcn_t[..., l_pos]
    return flipped


def apply_tta(bilstm_t: torch.Tensor, stgcn_t: torch.Tensor) -> list[tuple]:
    """Return list of (bilstm_variant, stgcn_variant) for 4 TTA versions.

    Args:
        bilstm_t: BiLSTM input tensor, shape (B, T, C) or (T, C).
        stgcn_t:  ST-GCN input tensor, shape (B, C, T, J) or (C, T, J).
                  J = 10 (OHP active joints, local indices 0-9).

    Returns:
        List of 4 (bilstm_variant, stgcn_variant) tuples:
          [0] Original
          [1] Horizontal flip (left/right joint swap)
          [2] +1 frame temporal jitter (torch.roll +1 on time axis)
          [3] -1 frame temporal jitter (torch.roll -1 on time axis)
    """
    variants: list[tuple] = []

    # 1. Original
    variants.append((bilstm_t, stgcn_t))

    # 2. Horizontal flip — swap LOCAL positional indices, NOT MediaPipe indices
    stgcn_flip = _horizontal_flip_stgcn(stgcn_t)
    variants.append((bilstm_t, stgcn_flip))

    # 3. +1 frame temporal jitter
    # bilstm: time axis is dim 1 (B, T, C) or dim 0 (T, C)
    # stgcn:  time axis is dim 2 (B, C, T, J) or dim 1 (C, T, J) → always second-to-last
    bilstm_plus1 = torch.roll(bilstm_t, shifts=1, dims=-2)
    stgcn_plus1  = torch.roll(stgcn_t,  shifts=1, dims=-2)
    variants.append((bilstm_plus1, stgcn_plus1))

    # 4. -1 frame temporal jitter
    bilstm_minus1 = torch.roll(bilstm_t, shifts=-1, dims=-2)
    stgcn_minus1  = torch.roll(stgcn_t,  shifts=-1, dims=-2)
    variants.append((bilstm_minus1, stgcn_minus1))

    return variants
