from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO   = Path(__file__).resolve().parents[4]   # ohp/ → training/ → exevision/ → core/ → repo root
_NEURAL = _REPO / "core" / "exevision" / "neural"
_TRAIN  = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_NEURAL / "ohp"), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import (
    FIXED_SEQ_LEN,
    NUM_ACTIVE_JOINTS,
    STGCN_CHANNELS,
    _extract_rep_matrix,
    _extract_stgcn_rep,
    _load_json,
    build_adjacency_matrix,
    pad_or_truncate,
)
from ohp.heuristic_vec import build_ohp_heuristic_vector


class OHPRepDataset(Dataset):
    """Dataset of OHP reps sourced from FitnessAQA-derived annotation JSONs.

    Each item returns a dict with keys:
      bilstm_input   : float32 tensor (FIXED_SEQ_LEN, NUM_BILSTM_CHANNELS)
      stgcn_input    : float32 tensor (STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_ACTIVE_JOINTS)
      heuristic_vec  : float32 tensor (16,)
      view_vec       : float32 tensor (5,)
      overall_score  : float32 scalar tensor
      elbow_error    : float32 scalar tensor  [0, 1]
      knee_error     : float32 scalar tensor  [0, 1]  (0.0 for seated)
    """

    def __init__(self, annotation_paths: List[Path], split: Optional[str] = None) -> None:
        """
        Args:
            annotation_paths: List of annotation JSON file paths to index.
            split: If provided, only include reps where annotation["fitnessaqa_split"] == split.
        """
        self._records: List[dict] = []
        for path in annotation_paths:
            try:
                anno = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            if split is not None and anno.get("fitnessaqa_split") != split:
                continue
            feat_path = Path(anno.get("pipeline_outputs", {}).get("features_json", ""))
            seg_path = Path(anno.get("pipeline_outputs", {}).get("segmented_json", ""))
            if not feat_path.exists():
                continue
            view = anno.get("view", "unknown")
            fps = float(anno.get("fps", 30.0))
            for rep in (anno.get("reps") or []):
                self._records.append({
                    "feat_path": feat_path,
                    "seg_path": seg_path,
                    "rep": rep,
                    "view": view,
                    "fps": fps,
                    "calibration": anno.get("calibration") or {},
                })

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self._records[idx]
        rep = record["rep"]
        feat_data = _load_json(record["feat_path"]) or {}
        seg_data = _load_json(record["seg_path"]) if record["seg_path"].exists() else {}

        bilstm_raw = _extract_rep_matrix(seg_data, rep)
        if bilstm_raw is None:
            bilstm_raw = np.zeros((1, 4), dtype=np.float32)
        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN))

        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep)
        if stgcn_raw is None:
            stgcn_raw = np.zeros((1, NUM_ACTIVE_JOINTS, STGCN_CHANNELS), dtype=np.float32)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)  # (T, J, C)
        # Reorder to (C, T, J) for ST-GCN conv
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype(np.float32, copy=False)
        )

        hvec = build_ohp_heuristic_vector(rep, record["view"])
        view_vec = hvec[11:16].copy()   # 5-dim view one-hot

        return {
            "bilstm_input": bilstm_t,
            "stgcn_input": stgcn_t,
            "heuristic_vec": torch.from_numpy(hvec),
            "view_vec": torch.from_numpy(view_vec),
            "overall_score": torch.tensor(float(rep.get("human_score", 50.0)), dtype=torch.float32),
            "elbow_error": torch.tensor(float(rep.get("elbow_error_soft", 0.0)), dtype=torch.float32),
            "knee_error": torch.tensor(float(rep.get("knee_error_soft", 0.0)), dtype=torch.float32),
        }


def build_dataloaders(
    annotation_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Return train/val/test DataLoaders from all annotation JSONs in annotation_dir."""
    all_paths = sorted(Path(annotation_dir).glob("*.json"))
    loaders = {}
    for split in ("train", "val", "test"):
        ds = OHPRepDataset(all_paths, split=split)
        if len(ds) == 0:
            continue
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
    return loaders
