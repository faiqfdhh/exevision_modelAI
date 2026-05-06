from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


NEURAL_DIR = Path(__file__).resolve().parents[2] / "neural"
if str(NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(NEURAL_DIR))

from nn_utils import (
    FIXED_SEQ_LEN,
    NUM_ACTIVE_JOINTS,
    STGCN_CHANNELS,
    build_adjacency_matrix,
    load_stgcn_reps,
    pad_or_truncate,
)


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "models/stgcn_seated_ohp_pretrained.pt"
DEFAULT_FEATURES_DIR = r"D:\FitnessAQA\ohp_phase1\workspace\seated_overhead_press\extracted_features_clean\raw_unfiltered"
DEFAULT_SEGMENTED_DIR = r"D:\FitnessAQA\ohp_phase1\workspace\seated_overhead_press\segmented_reps\raw_unfiltered"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _resolve_data_dir(preferred_path: str, fallback_relative: str) -> str:
    preferred = Path(preferred_path)
    if preferred.exists():
        return str(preferred)
    fallback = Path(__file__).resolve().parents[2] / fallback_relative
    return str(fallback)


class SpatialGraphConv(nn.Module):
    def __init__(self, c_in, c_out, A):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        return x


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        batch, channels, frames, joints = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch * joints, channels, frames)
        x = self.conv(x)
        out_frames = x.shape[-1]
        x = x.view(batch, joints, channels, out_frames).permute(0, 2, 3, 1).contiguous()
        return x


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A, kernel_size=9, stride=1, dropout=0.2):
        super().__init__()
        self.spatial = SpatialGraphConv(c_in, c_out, A)
        self.temporal = TemporalConv(c_out, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.bn(x)
        x = self.relu(x + res)
        x = self.dropout(x)
        return x


class STGCNPretrainer(nn.Module):
    def __init__(self, A, dropout=0.2):
        super().__init__()
        self.block1 = STGCNBlock(STGCN_CHANNELS, 64, A, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64, 64, A, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64, 128, A, stride=2, dropout=dropout)
        self.block4 = STGCNBlock(128, 128, A, stride=1, dropout=dropout)
        self.block5 = STGCNBlock(128, 256, A, stride=2, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, STGCN_CHANNELS * FIXED_SEQ_LEN * NUM_ACTIVE_JOINTS),
        )

    def encode(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decoder(embedding)
        reconstruction = reconstruction.view(-1, STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_ACTIVE_JOINTS)
        return reconstruction


def _convert_rep_to_stgcn_tensor(rep_array: np.ndarray) -> torch.Tensor:
    padded = pad_or_truncate(rep_array, FIXED_SEQ_LEN)
    tensor = torch.from_numpy(padded).float()
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


class MaskedRepDataset(Dataset):
    def __init__(self, reps):
        self.reps = [pad_or_truncate(rep, FIXED_SEQ_LEN) for rep in reps]

    def __len__(self):
        return len(self.reps)

    def __getitem__(self, idx):
        rep = torch.from_numpy(self.reps[idx]).float()
        rep = rep.permute(2, 0, 1).contiguous()
        return rep


def apply_joint_masking(batch, mask_ratio=0.25):
    """
    batch: (B, C, T, J) — mask random valid (frame, joint) positions via Bernoulli sampling.
    Returns: masked_input, targets, mask  — mask is (B, T, J) bool, True where zeroed.
    Fully vectorized: no Python loops over batch or positions.
    """
    targets = batch.clone()
    valid = batch.abs().sum(dim=1) > 0  # (B, T, J)
    rand = torch.rand(batch.shape[0], batch.shape[2], batch.shape[3], device=batch.device)
    mask = (rand < mask_ratio) & valid
    masked_batch = batch.masked_fill(mask.unsqueeze(1).expand_as(batch), 0.0)
    return masked_batch, targets, mask


def build_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    start_time = time.time()

    features_dir = _resolve_data_dir(args.features_dir, "training/seated_overhead_press")
    segmented_dir = _resolve_data_dir(args.segmented_dir, "training/seated_overhead_press/segmented_reps")

    print(f"[1/7] Loading ST-GCN reps from {segmented_dir}")
    reps = load_stgcn_reps(features_dir, segmented_dir, max_videos=args.max_videos)
    if args.max_reps is not None:
        reps = reps[: args.max_reps]

    print(f"✓ Loaded {len(reps)} reps")

    if not reps:
        raise RuntimeError("No ST-GCN reps were loaded. Check the segmented directory path.")

    adjacency = build_adjacency_matrix()
    dataset = MaskedRepDataset(reps)
    loader = build_dataloader(dataset, args.batch_size)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = STGCNPretrainer(adjacency, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            batch = batch.to(device)
            masked_input, targets, mask = apply_joint_masking(batch, mask_ratio=args.mask_ratio)

            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(masked_input)
            mask_expanded = mask.unsqueeze(1).expand_as(reconstructed)
            if mask_expanded.any().item():
                loss = F.mse_loss(reconstructed[mask_expanded], targets[mask_expanded])
            else:
                loss = F.mse_loss(reconstructed, targets)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = epoch_loss / max(1, batches)
        scheduler.step(epoch_loss)

        improved = epoch_loss < best_loss
        if improved:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            encoder_state = {k: v for k, v in model.state_dict().items() if not k.startswith("decoder")}
            torch.save(encoder_state, save_path.with_name(f"{save_path.stem}_encoder.pt"))
        else:
            no_improve += 1

        marker = " ✓" if improved else f" (no improve {no_improve}/{args.early_stop_patience})"
        print(f"Epoch {epoch:03d}/{args.epochs} | Loss: {epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}{marker}")

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"✓ Best ST-GCN weights saved to {save_path}")
    print(f"Summary: reps={len(reps)} | best_loss={best_loss:.6f} | time={elapsed:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="ST-GCN masked reconstruction pretraining for seated overhead press")
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--segmented-dir", type=str, default=DEFAULT_SEGMENTED_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"])
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-reps", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Stop if no improvement for N epochs (0 = disabled)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 = disabled)")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()