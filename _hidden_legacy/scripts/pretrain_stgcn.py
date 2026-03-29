from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
SAVE_PATH = "models/stgcn_pretrained.pt"


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
    fallback = Path(__file__).resolve().parents[1] / fallback_relative
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
    batch: (batch, 7, 128, 11)
    Randomly mask joint positions across frames.

    Returns: masked_input, targets, mask
    mask: (batch, 128, 11) boolean -- True where masked
    """
    targets = batch.clone()
    mask = torch.zeros(batch.shape[0], batch.shape[2], batch.shape[3], dtype=torch.bool, device=batch.device)

    for i in range(batch.shape[0]):
        valid = (batch[i].abs().sum(dim=0) > 0)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue

        num_mask = max(1, int(valid_count * mask_ratio))
        valid_indices = valid.nonzero(as_tuple=False)
        if valid_indices.numel() == 0:
            continue

        perm = torch.randperm(len(valid_indices), device=batch.device)[:num_mask]
        selected = valid_indices[perm]

        for frame_idx, joint_idx in selected:
            frame_idx = int(frame_idx.item())
            joint_idx = int(joint_idx.item())
            mask[i, frame_idx, joint_idx] = True
            batch[i, :, frame_idx, joint_idx] = 0.0

    return batch, targets, mask


def build_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    start_time = time.time()

    features_dir = _resolve_data_dir(args.features_dir, "squat/extracted_features_clean")
    segmented_dir = _resolve_data_dir(args.segmented_dir, "squat/segmented_reps")

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

    model = STGCNPretrainer(adjacency, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
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
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = epoch_loss / max(1, batches)
        scheduler.step(epoch_loss)
        best_loss = min(best_loss, epoch_loss)
        print(f"Epoch {epoch:03d}/{args.epochs} | Loss: {epoch_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    encoder_state = {key: value for key, value in model.state_dict().items() if not key.startswith("decoder")}
    torch.save(encoder_state, save_path.with_name(f"{save_path.stem}_encoder.pt"))

    elapsed = time.time() - start_time
    print(f"✓ Saved ST-GCN pretrained weights to {save_path}")
    print(f"Summary: reps={len(reps)} | final_loss={best_loss:.6f} | time={elapsed:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="ST-GCN triplet contrastive pretraining for squats")
    parser.add_argument("--features-dir", type=str, default=r"D:\squat\unlabeled_features\raw_unfiltered")
    parser.add_argument("--segmented-dir", type=str, default=r"D:\squat\unlabeled_features\raw_unfiltered\segmented_reps")
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
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()