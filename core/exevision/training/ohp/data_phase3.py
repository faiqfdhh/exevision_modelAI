from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO   = Path(__file__).resolve().parents[4]   # ohp/ → training/ → exevision/ → core/ → repo root
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_ROOT = Path(__file__).resolve().parents[1]  # core/exevision/training  (parent of ohp/)
_TRAIN  = Path(__file__).resolve().parent          # core/exevision/training/ohp
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_ROOT), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import (
    FIXED_SEQ_LEN,
    NUM_OHP_ACTIVE_JOINTS,
    STGCN_CHANNELS,
    NUM_OHP_BILSTM_CHANNELS,
    _extract_rep_matrix,
    _extract_stgcn_rep,
    _load_json,
    pad_or_truncate,
)
from heuristic_vec import build_ohp_heuristic_vector


class OHPPhase3Dataset(Dataset):
    """Reads OHP Phase 3 manual annotations.

    Each item returns dict with keys:
      bilstm_input   : float32 (FIXED_SEQ_LEN, NUM_OHP_BILSTM_CHANNELS)
      stgcn_input    : float32 (STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_OHP_ACTIVE_JOINTS)
      heuristic_vec  : float32 (16,)
      view_vec       : float32 (5,)
      quality        : float32 scalar  [0-100]
      smoothness     : float32 scalar  [0-100 or nan]
      control        : float32 scalar  [0-100 or nan]
      lockout        : float32 scalar  [0.0 or 1.0]
      elbow_flare    : float32 scalar  [0-100 or nan]
      grip_ratio     : float32 scalar  [0-100 or nan]  (nan for side view)
      rom_top        : float32 scalar  [0-100 or nan]
      rom_bottom     : float32 scalar  [0-100 or nan]
      view           : str
    """

    # Default consolidated dirs (relative to repo root) — populated by gather step
    _DEFAULT_FEAT_DIR = _REPO / "training_dataset" / "ohp_phase3_annotations" / "extracted_features"
    _DEFAULT_SEG_DIR  = _REPO / "training_dataset" / "ohp_phase3_annotations" / "segmented_reps"

    def __init__(
        self,
        annotation_paths: List[Path],
        split: Optional[str] = None,
        feat_dir: Optional[Path] = None,
        seg_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            annotation_paths: List of annotation JSON file paths to index.
            split: If provided, only include reps where annotation["fitnessaqa_split"] == split.
            feat_dir: Directory containing {video_id}.json feature files.
                      Falls back to pipeline_outputs.features_json in annotation if absent.
                      Default: training_dataset/ohp_phase3_annotations/extracted_features/
            seg_dir:  Directory containing {video_id}_segmented.json files.
                      Default: training_dataset/ohp_phase3_annotations/segmented_reps/
        """
        feat_dir = Path(feat_dir) if feat_dir else self._DEFAULT_FEAT_DIR
        seg_dir  = Path(seg_dir)  if seg_dir  else self._DEFAULT_SEG_DIR

        self._records: list[dict] = []
        for path in annotation_paths:
            try:
                anno = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            if split is not None and anno.get("fitnessaqa_split") != split:
                continue

            video_id = anno.get("video_id", Path(path).stem)

            # Resolve feature path: consolidated dir first, then pipeline_outputs fallback
            feat_path = feat_dir / f"{video_id}.json"
            if not feat_path.exists():
                _fallback = Path(anno.get("pipeline_outputs", {}).get("features_json", ""))
                if _fallback.exists():
                    feat_path = _fallback
                else:
                    continue  # no feature file — skip

            # Segmented path: optional
            seg_path = seg_dir / f"{video_id}_segmented.json"
            if not seg_path.exists():
                _fallback = Path(anno.get("pipeline_outputs", {}).get("segmented_json", ""))
                seg_path = _fallback if _fallback.exists() else Path("")

            view = anno.get("view", "unknown")
            fps  = float(anno.get("fps", 30.0))
            for rep in (anno.get("reps") or []):
                if rep.get("human_score") is None:
                    continue
                self._records.append({
                    "feat_path": feat_path,
                    "seg_path":  seg_path,
                    "rep":       rep,
                    "view":      view,
                    "fps":       fps,
                    "calibration": anno.get("calibration") or {},
                })

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]
        rep = record["rep"]
        feat_data = _load_json(record["feat_path"]) or {}
        seg_data  = _load_json(record["seg_path"]) if record["seg_path"].exists() else {}

        # --- BiLSTM input ---
        bilstm_raw = _extract_rep_matrix(seg_data, rep, exercise="overhead_press")
        if bilstm_raw is None:
            bilstm_raw = np.zeros((1, NUM_OHP_BILSTM_CHANNELS), dtype=np.float32)
        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN))

        # --- ST-GCN input ---
        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep, exercise="overhead_press")
        if stgcn_raw is None:
            stgcn_raw = np.zeros((1, NUM_OHP_ACTIVE_JOINTS, STGCN_CHANNELS), dtype=np.float32)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)  # (T, J, C)
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype(np.float32, copy=False)
        )  # (C, T, J)

        # --- Heuristic vector & view ---
        hvec = build_ohp_heuristic_vector(rep, record["view"])
        view_vec = hvec[11:16].copy()   # 5-dim view one-hot

        # --- Labels ---
        labels = self._extract_labels(rep)

        return {
            "bilstm_input": bilstm_t,
            "stgcn_input":  stgcn_t,
            "heuristic_vec": torch.from_numpy(hvec),
            "view_vec":      torch.from_numpy(view_vec),
            "view":          record["view"],
            **labels,
        }

    def _extract_labels(self, rep: dict) -> dict:
        hms   = rep.get("human_metric_scores") or {}
        flags = rep.get("human_flags") or {}

        def _to_tensor(val) -> torch.Tensor:
            return torch.tensor(
                float(val) if val is not None else float("nan"),
                dtype=torch.float32,
            )

        return {
            "quality":     torch.tensor(float(rep.get("human_score", 50.0)), dtype=torch.float32),
            "smoothness":  _to_tensor(hms.get("smoothness")),
            "control":     _to_tensor(hms.get("control")),
            "lockout":     torch.tensor(1.0 if flags.get("lockout") else 0.0, dtype=torch.float32),
            "elbow_flare": _to_tensor(hms.get("elbow_flare")),
            "grip_ratio":  _to_tensor(hms.get("grip_ratio")),
            "rom_top":     _to_tensor(hms.get("rom_top")),
            "rom_bottom":  _to_tensor(hms.get("rom_bottom")),
        }


def build_phase3_dataloaders(
    annotation_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict:
    """Return train/val/test DataLoaders from all Phase 3 OHP annotation JSONs."""
    all_paths = sorted(Path(annotation_dir).glob("*.json"))
    loaders = {}
    for split in ("train", "val", "test"):
        ds = OHPPhase3Dataset(all_paths, split=split)
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
