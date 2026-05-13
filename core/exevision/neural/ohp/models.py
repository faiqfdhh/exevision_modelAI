from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Resolve shared neural utilities without modifying sys.path permanently
_NEURAL_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_ROOT = Path(__file__).resolve().parents[2] / "training"
for _p in [str(_NEURAL_ROOT), str(_TRAIN_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import NUM_OHP_BILSTM_CHANNELS, NUM_OHP_ACTIVE_JOINTS
from pretrain_bilstm import TemporalAttention   # encoder building block — not modified
from pretrain_stgcn import STGCNBlock           # encoder building block — not modified


def _score_head(in_dim: int) -> nn.Sequential:
    """Linear head that maps embeddings to a quality score in [0, 100]."""
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def _error_head(in_dim: int) -> nn.Sequential:
    """Linear head that maps embeddings to an error probability in [0, 1]."""
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class OHPBiLSTMScorer(nn.Module):
    """Temporal scorer for standing OHP with knee error head.

    Encoder layer names (lstm1, lstm2, temporal_attention) match the pretrain
    checkpoint exactly so that load_pretrained() can transfer weights without
    key remapping.

    Phase 2 fine-tunes standing OHP only — seated OHP uses pre-trained
    encoder + heuristic-only inference (no Phase 2 model).

    Args:
        input_dim: Number of BiLSTM signal channels (default: NUM_BILSTM_CHANNELS = 4).
        hidden_dim: LSTM hidden size (default: 128, must match pretrained).
        dropout: Dropout rate (default: 0.3, must match pretrained).
    """

    def __init__(
        self,
        input_dim: int = NUM_OHP_BILSTM_CHANNELS,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.temporal_attention = TemporalAttention(hidden_dim * 2)

        embed_dim = hidden_dim * 2
        self.quality_head = _score_head(embed_dim)
        self.knee_error_head = _error_head(embed_dim)
        # Phase 3 temporal heads
        self.smoothness_head = _score_head(embed_dim)
        self.control_head = _score_head(embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        return self.temporal_attention(out)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(x)
        return {
            "embedding":  emb,
            "quality":    self.quality_head(emb).squeeze(-1) * 100.0,
            "knee_error": self.knee_error_head(emb).squeeze(-1),
            "smoothness": self.smoothness_head(emb).squeeze(-1) * 100.0,
            "control":    self.control_head(emb).squeeze(-1) * 100.0,
        }

    def load_pretrained(self, path: str) -> Tuple[int, int]:
        """Load encoder weights from a pretrained checkpoint.

        Ignores reconstruction_head keys and skips any key whose tensor shape
        doesn't match the current model (e.g. lstm1 input weights when channel
        count changed from 4 → 8). Returns (n_missing, n_unexpected).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = self.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if not k.startswith("reconstruction_head")
            and k in model_state
            and v.shape == model_state[k].shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return len(missing), len(unexpected)

    def load_phase2_for_phase3(self, path: str) -> None:
        """Transfer Phase 2 encoder + knee_error_head; re-init quality_head; freeze knee.

        Encoder weights (lstm1, lstm2, temporal_attention) transfer if shapes match.
        quality_head excluded — Phase 3 trains it from scratch on manual annotations.
        knee_error_head transfers and is frozen (Phase 2 labels, not re-annotated).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = self.state_dict()
        transfer = {
            k: v for k, v in state.items()
            if not k.startswith("quality_head")
            and k in model_state
            and v.shape == model_state[k].shape
        }
        self.load_state_dict(transfer, strict=False)
        for p in self.knee_error_head.parameters():
            p.requires_grad = False


class OHPSTGCNScorer(nn.Module):
    """Spatial scorer for standing OHP with knee error head.

    Encoder block names (block1–block5) match the pretrain checkpoint exactly.

    Phase 2 fine-tunes standing OHP only — seated OHP uses pre-trained
    encoder + heuristic-only inference (no Phase 2 model).

    Args:
        A: Normalised adjacency matrix, shape (11, 11), float32 numpy array or tensor.
        dropout: Dropout rate (default: 0.2, must match pretrained).
    """

    _VIEW_DIM = 5   # size of view one-hot appended to embedding before quality head

    def __init__(
        self,
        A,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32)

        from nn_utils import STGCN_CHANNELS
        # A must be (NUM_OHP_ACTIVE_JOINTS, NUM_OHP_ACTIVE_JOINTS) — built via build_adjacency_matrix_ohp()
        self.block1 = STGCNBlock(STGCN_CHANNELS, 64,  A, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64,  64,  A, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64,  128, A, stride=2, dropout=dropout)
        self.block4 = STGCNBlock(128, 128, A, stride=1, dropout=dropout)
        self.block5 = STGCNBlock(128, 256, A, stride=2, dropout=dropout)

        embed_dim = 256
        self.quality_head = _score_head(embed_dim + self._VIEW_DIM)
        self.knee_error_head = _error_head(embed_dim)
        # Phase 3 spatial heads (embed_dim only — no view concat needed)
        self.lockout_head    = _error_head(embed_dim)
        self.elbow_flare_head = _score_head(embed_dim)
        self.grip_ratio_head  = _score_head(embed_dim)
        self.rom_top_head     = _score_head(embed_dim)
        self.rom_bottom_head  = _score_head(embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.mean(dim=(2, 3))   # global average over time and joints → (B, 256)

    def forward(
        self,
        x: torch.Tensor,
        view_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emb = self.encode(x)
        if view_vec is None:
            view_vec = torch.zeros(emb.shape[0], self._VIEW_DIM, device=emb.device)
        spatial_in = torch.cat([emb, view_vec], dim=-1)

        return {
            "embedding":   emb,
            "quality":     self.quality_head(spatial_in).squeeze(-1) * 100.0,
            "knee_error":  self.knee_error_head(emb).squeeze(-1),
            "lockout":     self.lockout_head(emb).squeeze(-1),
            "elbow_flare": self.elbow_flare_head(emb).squeeze(-1) * 100.0,
            "grip_ratio":  self.grip_ratio_head(emb).squeeze(-1) * 100.0,
            "rom_top":     self.rom_top_head(emb).squeeze(-1) * 100.0,
            "rom_bottom":  self.rom_bottom_head(emb).squeeze(-1) * 100.0,
        }

    def load_pretrained(self, path: str) -> Tuple[int, int]:
        """Load encoder weights from pretrained checkpoint (encoder-only .pt preferred).

        Accepts both full pretrain checkpoint and encoder-only file. Skips keys
        whose tensor shape doesn't match the current model (e.g. spatial.A when
        joint count changed from 11 → 10). Returns (n_missing, n_unexpected).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = self.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k.startswith(("block1", "block2", "block3", "block4", "block5"))
            and k in model_state
            and v.shape == model_state[k].shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return len(missing), len(unexpected)

    def load_phase2_for_phase3(self, path: str) -> None:
        """Transfer Phase 2 encoder + knee_error_head; re-init quality_head; freeze knee.

        Encoder weights (lstm1, lstm2, temporal_attention) transfer if shapes match.
        quality_head excluded — Phase 3 trains it from scratch on manual annotations.
        knee_error_head transfers and is frozen (Phase 2 labels, not re-annotated).
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = self.state_dict()
        transfer = {
            k: v for k, v in state.items()
            if not k.startswith("quality_head")
            and k in model_state
            and v.shape == model_state[k].shape
        }
        self.load_state_dict(transfer, strict=False)
        for p in self.knee_error_head.parameters():
            p.requires_grad = False
