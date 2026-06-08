from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss ignoring NaN targets. Returns 0.0 if all targets are NaN."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred[mask], target[mask])


def bucket_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    low_threshold: float = 0.6,
    low_weight: float = 3.0,
) -> torch.Tensor:
    """MSE that up-weights low-score samples (target < low_threshold, both in [0,1]).

    Combats bad-rep blindness: the 40-60 quality bucket is rare in training data,
    so plain MSE barely penalises misses there. low_threshold=0.6 / low_weight=3.0
    means reps scored <60 contribute 3x to the loss.
    """
    mask = ~torch.isnan(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    p, t = pred[mask], target[mask]
    w = torch.where(t < low_threshold, torch.full_like(t, low_weight), torch.ones_like(t))
    return (w * (p - t) ** 2).mean()


def weighted_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    neg_weight: float = 4.0,
) -> torch.Tensor:
    """BCE with higher weight on negative class (majority for lockout).

    neg_weight = n_full_lockout / n_incomplete. Default 4.0 until computed from pool.
    Label smoothing: 0 → 0.1, 1 → 0.9.
    """
    smoothed = target * 0.9 + (1.0 - target) * 0.1
    w = torch.where(target > 0.5, torch.ones_like(target), torch.full_like(target, neg_weight))
    bce = F.binary_cross_entropy(pred, smoothed, reduction="none")
    return (bce * w).mean()


def compute_phase3_loss(
    bilstm_out: dict,
    stgcn_out: dict,
    fusion_score: torch.Tensor,
    batch: dict,
    lockout_neg_weight: float = 4.0,
) -> torch.Tensor:
    """Combined Phase 3 loss. All continuous targets normalised to [0,1] for MSE."""
    t_qual   = batch["quality"] / 100.0
    t_smooth = batch["smoothness"] / 100.0
    t_ctrl   = batch["control"] / 100.0
    t_lock   = batch["lockout"]
    t_flare  = batch["elbow_flare"] / 100.0
    t_grip   = batch["grip_ratio"] / 100.0
    t_rtop   = batch["rom_top"] / 100.0
    t_rbot   = batch["rom_bottom"] / 100.0

    # Quality: fusion (primary) + both component models
    L_quality = (
        F.mse_loss(fusion_score / 100.0, t_qual)
        + 0.3 * masked_mse(bilstm_out["quality"] / 100.0, t_qual)
        + 0.3 * masked_mse(stgcn_out["quality"] / 100.0, t_qual)
    )

    L_smooth  = masked_mse(bilstm_out["smoothness"] / 100.0, t_smooth)
    L_ctrl    = masked_mse(bilstm_out["control"] / 100.0, t_ctrl)
    L_lock    = weighted_bce(stgcn_out["lockout"], t_lock, lockout_neg_weight)
    L_flare   = masked_mse(stgcn_out["elbow_flare"] / 100.0, t_flare)
    L_grip    = masked_mse(stgcn_out["grip_ratio"] / 100.0, t_grip)
    L_rtop    = masked_mse(stgcn_out["rom_top"] / 100.0, t_rtop)
    L_rbot    = masked_mse(stgcn_out["rom_bottom"] / 100.0, t_rbot)

    return (
        1.0 * L_quality
        + 0.5 * L_smooth
        + 0.5 * L_ctrl
        + 0.7 * L_lock
        + 0.5 * L_flare
        + 0.5 * L_grip
        + 0.5 * L_rtop
        + 0.5 * L_rbot
    )
