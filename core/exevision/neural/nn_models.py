from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nn_utils import NUM_BILSTM_CHANNELS
from pretrain_bilstm import TemporalAttention
from pretrain_stgcn import STGCNBlock


VIEW_ORDER = ["front", "back", "side", "front_side", "back_side"]
FLAG_ORDER = [
    "insufficient_squat_depth",
    "knee_valgus",
    "lumbar_flexion",
    "heel_rise",
    "asymmetric_descent",
    "forward_lean",
]


class BiLSTMScorer(nn.Module):
    """Fine-tuned BiLSTM for temporal quality scoring."""

    def __init__(self, input_dim: int = NUM_BILSTM_CHANNELS, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.temporal_attention = TemporalAttention(hidden_dim * 2)

        self.smoothness_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm1(x)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        output = self.dropout2(output)
        return self.temporal_attention(output)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encode(x)
        smoothness = self.smoothness_head(embedding).squeeze(-1)
        control = self.control_head(embedding).squeeze(-1)
        return {
            "embedding": embedding,
            "smoothness": smoothness,
            "control": control,
        }

    def load_pretrained(self, path: str) -> Tuple[int, int]:
        pretrained = torch.load(path, map_location="cpu")
        if isinstance(pretrained, dict) and "state_dict" in pretrained:
            pretrained = pretrained["state_dict"]
        encoder_keys = {
            k: v
            for k, v in pretrained.items()
            if not k.startswith("reconstruction_head")
        }
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        return len(missing), len(unexpected)


class STGCNScorer(nn.Module):
    """Fine-tuned ST-GCN for spatial quality scoring."""

    def __init__(self, A: np.ndarray, dropout: float = 0.2):
        super().__init__()
        self.block1 = STGCNBlock(7, 64, A, stride=1, dropout=dropout)
        self.block2 = STGCNBlock(64, 64, A, stride=1, dropout=dropout)
        self.block3 = STGCNBlock(64, 128, A, stride=2, dropout=dropout)
        self.block4 = STGCNBlock(128, 128, A, stride=1, dropout=dropout)
        self.block5 = STGCNBlock(128, 256, A, stride=2, dropout=dropout)

        self.spatial_head = nn.Sequential(
            nn.Linear(256 + 5, 64),  # +5 for view one-hot (dims 10-14 of heuristic_vec)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid(),  # Force outputs to [0, 1]; will be scaled to [0, 100] at inference
        )

        self.auxiliary_heuristic_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor, view_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedding = self.encode(x)
        # view_vec: (B, 5) one-hot from heuristic_vec[:, 10:15]. Zeros = unknown view.
        if view_vec is None:
            view_vec = torch.zeros(embedding.shape[0], 5, device=embedding.device)
        spatial_input = torch.cat([embedding, view_vec], dim=-1)
        spatial_scores = self.spatial_head(spatial_input)
        aux_metrics = self.auxiliary_heuristic_head(embedding)
        return {
            "embedding": embedding,
            "depth": spatial_scores[:, 0],
            "forward_lean": spatial_scores[:, 1],
            "knee_tracking": spatial_scores[:, 2],
            "aux_min_knee_angle": aux_metrics[:, 0],
            "aux_forward_lean_deg": aux_metrics[:, 1],
            "aux_knee_valgus": aux_metrics[:, 2],
            "aux_squat_depth": aux_metrics[:, 3],
        }

    def load_pretrained(self, path: str) -> Tuple[int, int]:
        pretrained = torch.load(path, map_location="cpu")
        if isinstance(pretrained, dict) and "state_dict" in pretrained:
            pretrained = pretrained["state_dict"]
        encoder_keys = {
            k: v
            for k, v in pretrained.items()
            if k.startswith("block1")
            or k.startswith("block2")
            or k.startswith("block3")
            or k.startswith("block4")
            or k.startswith("block5")
        }
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        return len(missing), len(unexpected)


class HeuristicGuidedFusion(nn.Module):
    """
    Heuristic-anchored fusion where neural branches propose a bounded residual correction.

    Architecture:
      - Gated spatial and temporal branches modulate ST-GCN / BiLSTM embeddings
      - residual_head learns (human_score - heuristic_score) directly
      - tanh × 40 bounds output to ±40 points (covers any realistic heuristic error)
      - final = clamp(heuristic + residual, 0, 100)

    Confidence head removed: with ~120 training samples the correction × confidence
    decomposition is overparameterised. A single bounded residual is more stable.
    """

    def __init__(self, heuristic_dim: int = 15, neural_dim: int = 256, fusion_dim: int = 64):
        super().__init__()
        self.stgcn_compress = nn.Linear(neural_dim, fusion_dim)
        self.bilstm_compress = nn.Linear(neural_dim, fusion_dim)

        self.heuristic_proj = nn.Sequential(
            nn.Linear(heuristic_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_gate = nn.Sequential(
            nn.Linear(heuristic_dim + fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.temporal_gate = nn.Sequential(
            nn.Linear(heuristic_dim + fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        # Residual head: tanh bounds output to [-1, 1]; scaled by 40 outside → ±40 pts max.
        # Dropout reduced from 0.3 to 0.1: aggressive dropout destabilises learning
        # on small datasets (batch=16, n_train≈121).
        self.residual_head = nn.Sequential(
            nn.Linear(fusion_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        heuristic_vec: torch.Tensor,
        stgcn_embedding: torch.Tensor,
        bilstm_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            final_score: clamped prediction in [0, 100]
            residual:    raw residual in [-40, +40] (before clamping)
        """
        stgcn_comp = self.stgcn_compress(stgcn_embedding)
        bilstm_comp = self.bilstm_compress(bilstm_embedding)
        h_proj = self.heuristic_proj(heuristic_vec)

        sg = self.spatial_gate(torch.cat([heuristic_vec, stgcn_comp], dim=-1))
        tg = self.temporal_gate(torch.cat([heuristic_vec, bilstm_comp], dim=-1))

        gated_spatial = sg * stgcn_comp
        gated_temporal = tg * bilstm_comp
        fused = torch.cat([h_proj, gated_spatial, gated_temporal], dim=-1)

        # tanh output is in [-1, 1]; ×40 gives ±40 point correction range
        residual = self.residual_head(fused) * 40.0

        heuristic_score = heuristic_vec[:, 0:1] * 100.0
        final_score = torch.clamp(heuristic_score + residual, 0.0, 100.0)

        return final_score.squeeze(-1), residual.squeeze(-1)


def _safe_score(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    return float(value)


def build_heuristic_vector(rep_data: dict, view: str) -> np.ndarray:
    vec = np.zeros(15, dtype=np.float32)

    vec[0] = _safe_score(rep_data.get("heuristic_score", 0.0)) / 100.0

    hms = rep_data.get("heuristic_metric_scores", {}) or {}
    vec[1] = _safe_score(hms.get("forward_lean", 0.0)) / 100.0
    vec[2] = _safe_score(hms.get("depth", 0.0)) / 100.0
    vec[3] = _safe_score(hms.get("squat_depth", 0.0)) / 100.0

    flags = rep_data.get("flags", {}) or {}
    for i, flag_name in enumerate(FLAG_ORDER):
        vec[4 + i] = 1.0 if bool(flags.get(flag_name, False)) else 0.0

    view_lower = (view or "unknown").lower()
    for i, v in enumerate(VIEW_ORDER):
        vec[10 + i] = 1.0 if view_lower == v else 0.0

    return vec


def apply_safety_clamps(predicted_score: float, flags: dict, flag_severities: dict) -> float:
    score = float(predicted_score)

    if bool(flags.get("knee_valgus", False)) and int(flag_severities.get("knee_valgus", 0)) >= 2:
        score = min(score, 60.0)

    if bool(flags.get("forward_lean", False)) and int(flag_severities.get("forward_lean", 0)) >= 2:
        score = min(score, 65.0)

    if bool(flags.get("insufficient_squat_depth", False)) and int(
        flag_severities.get("insufficient_squat_depth", 0)
    ) >= 3:
        score = min(score, 50.0)

    return max(0.0, min(100.0, score))
