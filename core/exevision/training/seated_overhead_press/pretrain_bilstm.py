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
    NUM_BILSTM_CHANNELS,
    load_bilstm_reps,
    pad_or_truncate,
)


BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 80
MASK_RATIO = 0.25
MIN_MASK_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "models/bilstm_seated_ohp_pretrained.pt"
DEFAULT_FEATURES_DIR = r"D:\FitnessAQA\ohp_phase1\workspace\seated_overhead_press"
DEFAULT_SEGMENTED_DIR = r"D:\FitnessAQA\ohp_phase1\workspace\seated_overhead_press\segmented_reps"


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


class RepSignalDataset(Dataset):
    def __init__(self, reps):
        self.reps = [pad_or_truncate(rep, FIXED_SEQ_LEN).astype(np.float32, copy=False) for rep in reps]

    def __len__(self):
        return len(self.reps)

    def __getitem__(self, index):
        sample = torch.from_numpy(self.reps[index]).float()
        return sample


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 64)
        self.score = nn.Linear(64, 1)

    def forward(self, lstm_output):
        scores = self.score(torch.tanh(self.projection(lstm_output))).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(lstm_output * weights, dim=1)


class BiLSTMPretrainer(nn.Module):
    def __init__(self, input_dim=NUM_BILSTM_CHANNELS, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.reconstruction_head = nn.Linear(hidden_dim * 2, input_dim)
        self.temporal_attention = TemporalAttention(hidden_dim * 2)

    def encode(self, x):
        output, _ = self.lstm1(x)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        output = self.dropout2(output)
        return self.temporal_attention(output)

    def forward(self, x):
        output, _ = self.lstm1(x)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        output = self.dropout2(output)
        return self.reconstruction_head(output)


def apply_masking(batch, mask_ratio=MASK_RATIO, min_mask_len=MIN_MASK_LEN):
    batch = batch.clone()
    targets = batch.clone()
    mask = torch.zeros(batch.shape[:2], dtype=torch.bool, device=batch.device)

    for sample_index in range(batch.shape[0]):
        valid_rows = torch.any(batch[sample_index].abs() > 0, dim=-1)
        valid_length = int(valid_rows.sum().item())
        if valid_length <= 0:
            valid_length = batch.shape[1]

        mask_len = max(min_mask_len, int(valid_length * mask_ratio))
        mask_len = min(mask_len, valid_length)
        if mask_len <= 0:
            continue

        max_start = max(0, valid_length - mask_len)
        start = random.randint(0, max_start) if max_start > 0 else 0
        end = start + mask_len
        batch[sample_index, start:end] = 0.0
        mask[sample_index, start:end] = True

    return batch, targets, mask


def build_dataloader(reps, batch_size):
    dataset = RepSignalDataset(reps)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    start_time = time.time()

    features_dir = _resolve_data_dir(args.features_dir, "training/seated_overhead_press")
    segmented_dir = _resolve_data_dir(args.segmented_dir, "training/seated_overhead_press/segmented_reps")

    print(f"[1/7] Loading BiLSTM reps from {segmented_dir}")
    reps = load_bilstm_reps(features_dir, segmented_dir, max_videos=args.max_videos)
    if args.max_reps is not None:
        reps = reps[: args.max_reps]
    print(f"✓ Loaded {len(reps)} reps")

    if not reps:
        raise RuntimeError("No BiLSTM reps were loaded. Check the segmented directory path.")

    loader = build_dataloader(reps, args.batch_size)
    model = BiLSTMPretrainer().to(device)
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
            masked_input, targets, mask = apply_masking(batch, mask_ratio=args.mask_ratio, min_mask_len=args.min_mask_len)

            optimizer.zero_grad(set_to_none=True)
            output = model(masked_input)
            if mask.any().item():
                loss = F.mse_loss(output[mask], targets[mask])
            else:
                loss = F.mse_loss(output, targets)
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

    elapsed = time.time() - start_time
    print(f"✓ Saved BiLSTM pretrained weights to {save_path}")
    print(f"Summary: reps={len(reps)} | final_loss={best_loss:.6f} | time={elapsed:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM masked reconstruction pretraining for seated overhead press")
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--segmented-dir", type=str, default=DEFAULT_SEGMENTED_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    parser.add_argument("--min-mask-len", type=int, default=MIN_MASK_LEN)
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cuda", "cpu"])
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-reps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()