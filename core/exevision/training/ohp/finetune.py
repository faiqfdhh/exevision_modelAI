from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[4]
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from data import OHPRepDataset, build_dataloaders

# Loss weights — per FitnessAQA paper methodology
# Uses binary error labels with class weights for imbalance handling
_LAMBDA_COMPONENT_QUALITY = 0.3   # weight for per-model quality loss vs fusion
_LAMBDA_ELBOW = 0.3               # weight for elbow BCE
_LAMBDA_KNEE = 0.2                # weight for knee BCE (skipped for seated)

# Class weights: (n_negative / n_positive) — computed dynamically from data
# For now, estimated from label analysis; computed per-batch in training loop
_ELBOW_CLASS_WEIGHT = 3.58        # (2218 neg / 620 pos) from 2838 total reps
_KNEE_CLASS_WEIGHT = 5.10         # (2373 neg / 465 pos) from 2838 total reps

SEED = 42
EPOCHS = 60
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
PATIENCE = 10   # early stopping patience (val loss non-improvement)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_loss(
    bilstm_out: dict,
    stgcn_out: dict,
    fusion_score: torch.Tensor,
    batch: dict,
    include_knee: bool,
) -> torch.Tensor:
    target = batch["overall_score"] / 100.0   # normalize to [0, 1]
    target_elbow = batch["elbow_error"]
    target_knee = batch["knee_error"]

    mse_fusion = F.mse_loss(fusion_score / 100.0, target)
    mse_bilstm = F.mse_loss(bilstm_out["quality"] / 100.0, target)
    mse_stgcn = F.mse_loss(stgcn_out["quality"] / 100.0, target)

    # Binary classification with class weights (per FitnessAQA paper methodology)
    # Compute per-sample weights: higher weight for positive (minority) class
    bce_bilstm_elbow = F.binary_cross_entropy(bilstm_out["elbow_error"], target_elbow, reduction="none")
    weight_elbow = torch.where(target_elbow > 0.5, _ELBOW_CLASS_WEIGHT, 1.0)
    bce_bilstm_elbow = (bce_bilstm_elbow * weight_elbow).mean()

    bce_stgcn_elbow = F.binary_cross_entropy(stgcn_out["elbow_error"], target_elbow, reduction="none")
    bce_stgcn_elbow = (bce_stgcn_elbow * weight_elbow).mean()

    loss = mse_fusion
    loss += _LAMBDA_COMPONENT_QUALITY * (mse_bilstm + mse_stgcn)
    loss += _LAMBDA_ELBOW * (bce_bilstm_elbow + bce_stgcn_elbow)

    if include_knee and "knee_error" in bilstm_out:
        bce_bilstm_knee = F.binary_cross_entropy(bilstm_out["knee_error"], target_knee, reduction="none")
        weight_knee = torch.where(target_knee > 0.5, _KNEE_CLASS_WEIGHT, 1.0)
        bce_bilstm_knee = (bce_bilstm_knee * weight_knee).mean()

        bce_stgcn_knee = F.binary_cross_entropy(stgcn_out["knee_error"], target_knee, reduction="none")
        bce_stgcn_knee = (bce_stgcn_knee * weight_knee).mean()

        loss += _LAMBDA_KNEE * (bce_bilstm_knee + bce_stgcn_knee)

    return loss


def _run_epoch(
    bilstm: OHPBiLSTMScorer,
    stgcn: OHPSTGCNScorer,
    fusion,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    include_knee: bool,
    train: bool,
) -> float:
    bilstm.train(train)
    stgcn.train(train)
    fusion.train(train)
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            bilstm_out = bilstm(batch["bilstm_input"])
            stgcn_out = stgcn(batch["stgcn_input"], batch["view_vec"])
            fusion_score, _ = fusion(
                batch["heuristic_vec"],
                stgcn_out["embedding"],
                bilstm_out["embedding"],
            )
            loss = _compute_loss(bilstm_out, stgcn_out, fusion_score, batch, include_knee)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train(
    annotation_dir: Path,
    pretrain_bilstm: Path,
    pretrain_stgcn: Path,
    output_dir: Path,
    exercise: str,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    include_knee = exercise != "seated_overhead_press"
    output_suffix = "ohp_phase2" if exercise == "overhead_press" else "seated_ohp_phase2"

    loaders = build_dataloaders(annotation_dir, batch_size=batch_size, exercise=exercise)
    if "train" not in loaders:
        raise RuntimeError(f"No training data found in {annotation_dir}")

    A = torch.tensor(build_adjacency_matrix(), dtype=torch.float32).to(device)

    bilstm = OHPBiLSTMScorer(include_knee_head=include_knee).to(device)
    stgcn = OHPSTGCNScorer(A, include_knee_head=include_knee).to(device)
    fusion = build_ohp_fusion().to(device)

    m_bilstm, _ = bilstm.load_pretrained(str(pretrain_bilstm))
    m_stgcn, _ = stgcn.load_pretrained(str(pretrain_stgcn))
    print(f"BiLSTM missing keys after pretrain load: {m_bilstm}")
    print(f"ST-GCN missing keys after pretrain load: {m_stgcn}")

    optimizer = torch.optim.Adam(
        list(bilstm.parameters()) + list(stgcn.parameters()) + list(fusion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(bilstm, stgcn, fusion, loaders["train"], optimizer, device, include_knee, train=True)
        val_loss = float("inf")
        if "val" in loaders:
            val_loss = _run_epoch(bilstm, stgcn, fusion, loaders["val"], optimizer, device, include_knee, train=False)
            scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bilstm.state_dict(), output_dir / f"bilstm_{output_suffix}.pt")
            torch.save(stgcn.state_dict(), output_dir / f"stgcn_{output_suffix}.pt")
            torch.save(fusion.state_dict(), output_dir / f"fusion_{output_suffix}.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
                break

    print(f"Best val loss: {best_val:.4f}. Checkpoints saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 OHP multi-task fine-tuning")
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--pretrain-bilstm", required=True)
    parser.add_argument("--pretrain-stgcn", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exercise", default="overhead_press",
                        choices=["overhead_press", "seated_overhead_press"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(
        annotation_dir=Path(args.annotation_dir),
        pretrain_bilstm=Path(args.pretrain_bilstm),
        pretrain_stgcn=Path(args.pretrain_stgcn),
        output_dir=Path(args.output_dir),
        exercise=args.exercise,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
