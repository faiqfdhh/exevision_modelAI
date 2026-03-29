from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nn_models import BiLSTMScorer, HeuristicGuidedFusion, STGCNScorer, build_heuristic_vector
from nn_utils import _extract_stgcn_rep, build_adjacency_matrix, pad_or_truncate


BUCKET_EDGES = [20.0, 40.0, 60.0, 80.0, 100.0]
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_bucket(score: float) -> int:
    if score < 20.0:
        return 0
    if score < 40.0:
        return 1
    if score < 60.0:
        return 2
    if score < 80.0:
        return 3
    return 4


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> Optional[torch.Tensor]:
    mask = mask.float()
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
        target = target.unsqueeze(-1)
        mask = mask.unsqueeze(-1)

    valid_per_sample = mask.sum(dim=1)
    valid_rows = valid_per_sample > 0
    if not valid_rows.any().item():
        return None

    sq = (pred - target) ** 2
    mse_per_sample = (sq * mask).sum(dim=1) / valid_per_sample.clamp(min=1.0)
    mse_per_sample = mse_per_sample[valid_rows]
    w = sample_weight[valid_rows]
    w = w * (w.shape[0] / w.sum().clamp(min=1e-8))
    return (mse_per_sample * w).mean()


def resolve_feature_path(root: Path, annotation_data: dict, video_id: str) -> Optional[Path]:
    candidates: List[Path] = []
    raw = (annotation_data.get("pipeline_outputs", {}) or {}).get("features_json")
    if isinstance(raw, str) and raw.strip():
        raw = raw.strip()
        candidates.append(Path(raw))
        candidates.append(Path(raw.replace("\\", "/")))
        candidates.append(root / raw)
        candidates.append(root / raw.replace("\\", "/"))

    for c in candidates:
        if c.exists():
            return c

    fallback_roots = [
        root / "squat" / "extracted_features_clean" / "excellent",
        root / "squat" / "extracted_features_clean" / "good",
        root / "squat" / "extracted_features_clean" / "fair",
        root / "squat" / "extracted_features_clean" / "raw_unfiltered",
    ]
    for base in fallback_roots:
        p = base / f"{video_id}.json"
        if p.exists():
            return p
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_video_score(annotation_data: dict) -> Optional[float]:
    reps = annotation_data.get("reps", []) or []
    vals = [safe_float((r or {}).get("human_score")) for r in reps]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def stratified_video_split(video_scores: Dict[str, float], seed: int = SEED) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[str]] = {i: [] for i in range(5)}
    for vid, score in video_scores.items():
        buckets[score_bucket(score)].append(vid)

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for _, vids in buckets.items():
        if not vids:
            continue
        vids = vids.copy()
        rng.shuffle(vids)
        n = len(vids)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))

        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        train.extend(vids[:n_train])
        val.extend(vids[n_train:n_train + n_val])
        test.extend(vids[n_train + n_val:n_train + n_val + n_test])

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


@dataclass
class RepRecord:
    video_id: str
    rep_id: int
    view: str
    confidence: float
    human_score: float
    bucket: int
    weight: float
    bilstm_seq: np.ndarray
    stgcn_seq: np.ndarray
    heuristic_vec: np.ndarray
    temporal_target: np.ndarray
    temporal_mask: np.ndarray
    spatial_target: np.ndarray
    spatial_mask: np.ndarray
    aux_target: np.ndarray
    aux_mask: np.ndarray
    heuristic_score: float
    heuristic_flags: Dict[str, bool]
    human_flags: Dict[str, bool]
    flag_severities: Dict[str, int]


class MultiModalRepDataset(Dataset):
    def __init__(self, records: Sequence[RepRecord], training: bool = False):
        self.records = list(records)
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def _augment_bilstm(self, seq: np.ndarray) -> np.ndarray:
        x = seq.copy()
        x += np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)

        shift = int(np.random.randint(-2, 3))
        if shift > 0:
            x = np.concatenate([np.repeat(x[:1], shift, axis=0), x[:-shift]], axis=0)
        elif shift < 0:
            x = np.concatenate([x[-shift:], np.repeat(x[-1:], -shift, axis=0)], axis=0)

        if np.random.rand() < 0.10:
            channel = int(np.random.randint(0, x.shape[1]))
            x[:, channel] = 0.0

        # Time warping: resample at ±15% speed, then re-pad/truncate to original length.
        # Simulates squats performed at different tempos; teaches the BiLSTM that tempo
        # should not change the quality score.
        if np.random.rand() < 0.50:
            T = x.shape[0]
            warp = float(np.random.uniform(0.85, 1.15))
            new_len = max(10, int(round(T * warp)))
            old_idx = np.linspace(0, T - 1, new_len)
            x = np.stack(
                [np.interp(old_idx, np.arange(T), x[:, c]) for c in range(x.shape[1])],
                axis=-1,
            ).astype(np.float32)
            if new_len >= T:
                x = x[:T]
            else:
                x = np.concatenate([x, np.repeat(x[-1:], T - new_len, axis=0)], axis=0)

        return x

    def _augment_stgcn(self, seq: np.ndarray) -> np.ndarray:
        x = seq.copy()

        theta = np.deg2rad(np.random.uniform(-15.0, 15.0))
        c = np.cos(theta)
        s = np.sin(theta)
        xr = x[:, :, 0].copy()
        zr = x[:, :, 2].copy()
        x[:, :, 0] = c * xr + s * zr
        x[:, :, 2] = -s * xr + c * zr

        scale = float(np.random.uniform(0.9, 1.1))
        x[:, :, :3] *= scale

        x[:, :, :3] += np.random.normal(0.0, 0.01, size=x[:, :, :3].shape).astype(np.float32)

        drop_mask = np.random.rand(x.shape[0], x.shape[1]) < 0.05
        x[drop_mask] = 0.0
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]

        bilstm_seq = r.bilstm_seq.copy()
        stgcn_seq = r.stgcn_seq.copy()
        if self.training:
            bilstm_seq = self._augment_bilstm(bilstm_seq)
            stgcn_seq = self._augment_stgcn(stgcn_seq)

        return {
            "bilstm": torch.from_numpy(bilstm_seq).float(),
            "stgcn": torch.from_numpy(stgcn_seq).float().permute(2, 0, 1).contiguous(),
            "heuristic": torch.from_numpy(r.heuristic_vec).float(),
            "weight": torch.tensor(r.weight, dtype=torch.float32),
            "human_score": torch.tensor(r.human_score, dtype=torch.float32),
            "temporal_target": torch.from_numpy(r.temporal_target).float(),
            "temporal_mask": torch.from_numpy(r.temporal_mask).float(),
            "spatial_target": torch.from_numpy(r.spatial_target).float(),
            "spatial_mask": torch.from_numpy(r.spatial_mask).float(),
            "aux_target": torch.from_numpy(r.aux_target).float(),
            "aux_mask": torch.from_numpy(r.aux_mask).float(),
        }


def build_records(root: Path, index_path: Path) -> Tuple[List[RepRecord], Dict[str, float]]:
    index_data = load_json(index_path)
    index_videos = (index_data.get("videos", {}) or {})
    video_ids = sorted(index_videos.keys())

    all_records: List[RepRecord] = []
    video_scores: Dict[str, float] = {}

    for video_id in video_ids:
        anno_path = root / "dataset" / "annotations" / "videos" / f"{video_id}.json"
        if not anno_path.exists():
            continue

        annotation_data = load_json(anno_path)
        view = (annotation_data.get("view") or "unknown").lower()
        feature_path = resolve_feature_path(root, annotation_data, video_id)
        if feature_path is None:
            continue

        feature_data = load_json(feature_path)
        calibration = (annotation_data.get("calibration", {}) or {})
        seg_stub = {
            "info": {
                "calibration": calibration,
                "fps": annotation_data.get("fps", 30.0),
            }
        }

        reps = annotation_data.get("reps", []) or []
        sample_scores: List[float] = []
        for rep in reps:
            human_score = safe_float((rep or {}).get("human_score"))
            if human_score is None:
                continue

            signals = (rep.get("signals", {}) or {})
            channels = []
            for k in [
                "normalized_hip_displacement",
                "window_velocity",
                "knee_angles",
                "landmark_confidence",
            ]:
                arr = np.asarray(signals.get(k, []), dtype=np.float32)
                if arr.size == 0:
                    arr = np.zeros((1,), dtype=np.float32)
                channels.append(arr)

            min_len = min(len(x) for x in channels)
            if min_len <= 0:
                continue
            bilstm = np.stack([x[:min_len] for x in channels], axis=-1)
            bilstm = pad_or_truncate(bilstm).astype(np.float32)

            stgcn = _extract_stgcn_rep(seg_stub, feature_data, rep)
            if stgcn is None:
                continue
            stgcn = pad_or_truncate(stgcn).astype(np.float32)

            hm = (rep.get("human_metric_scores", {}) or {})
            smoothness = safe_float(hm.get("smoothness"))
            control = safe_float(hm.get("control_at_bottom"))
            depth = safe_float(hm.get("depth"))
            lean = safe_float(hm.get("forward_lean"))
            knee_tracking = safe_float(hm.get("knee_tracking"))

            temporal_target = np.array([
                0.0 if smoothness is None else smoothness / 100.0,
                0.0 if control is None else control / 100.0,
            ], dtype=np.float32)
            temporal_mask = np.array([
                0.0 if smoothness is None else 1.0,
                0.0 if control is None else 1.0,
            ], dtype=np.float32)

            spatial_target = np.array([
                0.0 if depth is None else depth / 100.0,
                0.0 if lean is None else lean / 100.0,
                0.0 if knee_tracking is None else knee_tracking / 100.0,
            ], dtype=np.float32)
            spatial_mask = np.array([
                0.0 if depth is None else 1.0,
                0.0 if lean is None else 1.0,
                0.0 if knee_tracking is None else 1.0,
            ], dtype=np.float32)

            heur = (rep.get("heuristic_metrics", {}) or {})
            aux_raw = {
                "min_knee_angle": safe_float(heur.get("min_knee_angle")),
                "forward_lean": safe_float(heur.get("forward_lean")),
                "knee_valgus": safe_float(heur.get("knee_valgus")),
                "squat_depth": safe_float(heur.get("squat_depth")),
            }
            aux_target = np.array([
                0.0 if aux_raw["min_knee_angle"] is None else aux_raw["min_knee_angle"] / 180.0,
                0.0 if aux_raw["forward_lean"] is None else abs(aux_raw["forward_lean"]) / 90.0,
                0.0 if aux_raw["knee_valgus"] is None else aux_raw["knee_valgus"],
                0.0
                if aux_raw["squat_depth"] is None
                else (np.clip(aux_raw["squat_depth"], -1.0, 1.0) + 1.0) / 2.0,
            ], dtype=np.float32)
            aux_mask = np.array([
                0.0 if aux_raw["min_knee_angle"] is None else 1.0,
                0.0 if aux_raw["forward_lean"] is None else 1.0,
                0.0 if aux_raw["knee_valgus"] is None else 1.0,
                0.0 if aux_raw["squat_depth"] is None else 1.0,
            ], dtype=np.float32)

            confidence = float(max(1.0, min(5.0, safe_float(rep.get("annotator_confidence")) or 3.0)))
            rec = RepRecord(
                video_id=video_id,
                rep_id=int(rep.get("rep_id", 0)),
                view=view,
                confidence=confidence,
                human_score=float(human_score),
                bucket=score_bucket(human_score),
                weight=1.0,
                bilstm_seq=bilstm,
                stgcn_seq=stgcn,
                heuristic_vec=build_heuristic_vector(rep, view),
                temporal_target=temporal_target,
                temporal_mask=temporal_mask,
                spatial_target=spatial_target,
                spatial_mask=spatial_mask,
                aux_target=aux_target,
                aux_mask=aux_mask,
                heuristic_score=float(safe_float(rep.get("heuristic_score")) or 0.0),
                heuristic_flags={
                    k: bool(v)
                    for k, v in ((rep.get("flags", {}) or {}).items())
                },
                human_flags={
                    k: bool(v)
                    for k, v in ((rep.get("human_flags", {}) or {}).items())
                },
                flag_severities={
                    k: int(v)
                    for k, v in ((rep.get("flag_severities", {}) or {}).items())
                    if safe_float(v) is not None
                },
            )
            all_records.append(rec)
            sample_scores.append(float(human_score))

        if sample_scores:
            video_scores[video_id] = float(np.mean(sample_scores))

    return all_records, video_scores


def assign_training_weights(train_records: List[RepRecord]) -> Dict[int, float]:
    bucket_counts: Dict[int, int] = {}
    for r in train_records:
        bucket_counts[r.bucket] = bucket_counts.get(r.bucket, 0) + 1

    if not bucket_counts:
        return {}

    median_count = float(np.median(list(bucket_counts.values())))
    bucket_weight: Dict[int, float] = {}
    for b, c in bucket_counts.items():
        w = median_count / max(1.0, float(c))
        bucket_weight[b] = float(min(5.0, w))

    for r in train_records:
        conf_w = r.confidence / 5.0
        inv_w = bucket_weight.get(r.bucket, 1.0)
        r.weight = float(conf_w * inv_w)

    return bucket_weight


def split_records(records: Sequence[RepRecord], split: Dict[str, List[str]]) -> Tuple[List[RepRecord], List[RepRecord], List[RepRecord]]:
    train_ids = set(split.get("train", []))
    val_ids = set(split.get("val", []))
    test_ids = set(split.get("test", []))

    train = [r for r in records if r.video_id in train_ids]
    val = [r for r in records if r.video_id in val_ids]
    test = [r for r in records if r.video_id in test_ids]
    return train, val, test


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate_bilstm(model: BiLSTMScorer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch["bilstm"])
            loss = masked_weighted_mse(
                torch.stack([out["smoothness"], out["control"]], dim=1),
                batch["temporal_target"],
                batch["temporal_mask"],
                batch["weight"],
            )
            if loss is not None:
                losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_stgcn(model: STGCNScorer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch["stgcn"], batch["heuristic"][:, 10:15])
            spatial_pred = torch.stack([out["depth"], out["forward_lean"], out["knee_tracking"]], dim=1)
            aux_pred = torch.stack(
                [
                    out["aux_min_knee_angle"],
                    out["aux_forward_lean_deg"],
                    out["aux_knee_valgus"],
                    out["aux_squat_depth"],
                ],
                dim=1,
            )
            l_spatial = masked_weighted_mse(
                spatial_pred,
                batch["spatial_target"],
                batch["spatial_mask"],
                batch["weight"],
            )
            l_aux = masked_weighted_mse(
                aux_pred,
                batch["aux_target"],
                batch["aux_mask"],
                batch["weight"],
            )
            if l_spatial is None and l_aux is None:
                continue
            if l_spatial is None:
                loss = 0.3 * l_aux
            elif l_aux is None:
                loss = l_spatial
            else:
                loss = l_spatial + 0.3 * l_aux
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_fusion(
    bilstm: BiLSTMScorer,
    stgcn: STGCNScorer,
    fusion: HeuristicGuidedFusion,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Returns (val_mse, mean_abs_residual, residual_std) for Phase 3 monitoring."""
    bilstm.eval()
    stgcn.eval()
    fusion.eval()

    losses: List[float] = []
    residuals: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            b_out = bilstm(batch["bilstm"])
            s_out = stgcn(batch["stgcn"], batch["heuristic"][:, 10:15])
            pred, residual = fusion(batch["heuristic"], s_out["embedding"], b_out["embedding"])

            weights = batch["weight"]
            weights = weights * (weights.shape[0] / weights.sum().clamp(min=1e-8))
            mse = ((pred - batch["human_score"]) ** 2 * weights).mean()
            losses.append(float(mse.item()))

            residuals.extend(residual.detach().cpu().numpy().tolist())

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    mean_abs = float(np.mean(np.abs(residuals))) if residuals else 0.0
    std_res = float(np.std(residuals)) if residuals else 0.0
    return mean_loss, mean_abs, std_res


def train_phase_bilstm(
    model: BiLSTMScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch["bilstm"])
            loss = masked_weighted_mse(
                torch.stack([out["smoothness"], out["control"]], dim=1),
                batch["temporal_target"],
                batch["temporal_mask"],
                batch["weight"],
            )
            if loss is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss = evaluate_bilstm(model, val_loader, device)
        scheduler.step(val_loss)
        train_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"[Phase1] epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best:
            best = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)


def train_phase_stgcn(
    model: STGCNScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch["stgcn"], batch["heuristic"][:, 10:15])

            spatial_pred = torch.stack([out["depth"], out["forward_lean"], out["knee_tracking"]], dim=1)
            aux_pred = torch.stack(
                [
                    out["aux_min_knee_angle"],
                    out["aux_forward_lean_deg"],
                    out["aux_knee_valgus"],
                    out["aux_squat_depth"],
                ],
                dim=1,
            )

            l_spatial = masked_weighted_mse(
                spatial_pred,
                batch["spatial_target"],
                batch["spatial_mask"],
                batch["weight"],
            )
            l_aux = masked_weighted_mse(aux_pred, batch["aux_target"], batch["aux_mask"], batch["weight"])

            if l_spatial is None and l_aux is None:
                continue
            if l_spatial is None:
                loss = 0.3 * l_aux
            elif l_aux is None:
                loss = l_spatial
            else:
                loss = l_spatial + 0.3 * l_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss = evaluate_stgcn(model, val_loader, device)
        scheduler.step(val_loss)
        train_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"[Phase2] epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best:
            best = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)


def train_phase_fusion(
    bilstm: BiLSTMScorer,
    stgcn: STGCNScorer,
    fusion: HeuristicGuidedFusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Path,
) -> None:
    """
    Phase 3: Fusion layer training with encoder fine-tuning via differential LR.

    Key design decisions:
    - Encoders unfrozen at lr×0.05 (slow adaptation — they already have useful
      representations from Phases 1-2 but need to re-orient toward the human-heuristic
      gap rather than sub-metric values)
    - Fusion layer at full lr with weight_decay=1e-4 (L2 regularisation instead of
      the old L1 correction penalty which caused collapse to a near-zero constant offset)
    - No explicit correction regularisation: tanh×40 in residual_head already bounds
      corrections to ±40 pts; weight_decay + gradient clipping provide stability
    - Early stopping patience = 25 (model needs more time to escape the trivial
      zero-residual minimum when encoders are also adapting)
    """
    # Unfreeze both encoders — use differential learning rates so pretrained weights
    # adapt slowly while the fusion layer learns faster
    for p in bilstm.parameters():
        p.requires_grad = True
    for p in stgcn.parameters():
        p.requires_grad = True

    encoder_lr = lr * 0.05
    optimizer = torch.optim.Adam([
        {"params": bilstm.parameters(), "lr": encoder_lr},
        {"params": stgcn.parameters(), "lr": encoder_lr},
        {"params": fusion.parameters(), "lr": lr, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5)

    best = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 25

    for epoch in range(1, epochs + 1):
        bilstm.train()
        stgcn.train()
        fusion.train()
        losses: List[float] = []
        for batch in train_loader:
            batch = to_device(batch, device)

            b_out = bilstm(batch["bilstm"])
            s_out = stgcn(batch["stgcn"], batch["heuristic"][:, 10:15])
            pred, residual = fusion(batch["heuristic"], s_out["embedding"], b_out["embedding"])

            weights = batch["weight"]
            weights = weights * (weights.shape[0] / weights.sum().clamp(min=1e-8))

            # Pure weighted MSE — no correction regularisation.
            # Regularisation comes from: weight_decay (L2), Dropout(0.1), gradient clipping.
            loss = ((pred - batch["human_score"]) ** 2 * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(bilstm.parameters()) + list(stgcn.parameters()) + list(fusion.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss, mean_abs_res, std_res = evaluate_fusion(
            bilstm,
            stgcn,
            fusion,
            val_loader,
            device,
        )
        scheduler.step(val_loss)
        train_loss = float(np.mean(losses)) if losses else float("inf")

        print(
            f"[Phase3] epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} "
            f"|res|={mean_abs_res:.3f} res_std={std_res:.3f} "
            f"enc_lr={optimizer.param_groups[0]['lr']:.2e} fus_lr={optimizer.param_groups[2]['lr']:.2e}"
        )

        if val_loss < best:
            best = val_loss
            epochs_no_improve = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fusion.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"[Phase3] Early stopping: no improvement for {early_stop_patience} epochs. Stopping at epoch {epoch}.")
            break


def train_optional_joint(
    bilstm: BiLSTMScorer,
    stgcn: STGCNScorer,
    fusion: HeuristicGuidedFusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    save_path: Path,
) -> None:
    for p in bilstm.parameters():
        p.requires_grad = True
    for p in stgcn.parameters():
        p.requires_grad = True

    params = list(bilstm.parameters()) + list(stgcn.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best = float("inf")
    worsening = 0
    early_stop_patience = 8  # was 3 — gives the joint phase breathing room to find a minimum
    for epoch in range(1, epochs + 1):
        bilstm.train()
        stgcn.train()
        fusion.train()
        losses: List[float] = []

        for batch in train_loader:
            batch = to_device(batch, device)

            b_out = bilstm(batch["bilstm"])
            s_out = stgcn(batch["stgcn"], batch["heuristic"][:, 10:15])
            pred, corr = fusion(batch["heuristic"], s_out["embedding"], b_out["embedding"])

            l_fusion = ((pred - batch["human_score"]) ** 2).mean()

            l_b = masked_weighted_mse(
                torch.stack([b_out["smoothness"], b_out["control"]], dim=1),
                batch["temporal_target"],
                batch["temporal_mask"],
                batch["weight"],
            )
            if l_b is None:
                l_b = torch.tensor(0.0, device=device)

            spatial_pred = torch.stack([s_out["depth"], s_out["forward_lean"], s_out["knee_tracking"]], dim=1)
            aux_pred = torch.stack(
                [
                    s_out["aux_min_knee_angle"],
                    s_out["aux_forward_lean_deg"],
                    s_out["aux_knee_valgus"],
                    s_out["aux_squat_depth"],
                ],
                dim=1,
            )
            l_s = masked_weighted_mse(spatial_pred, batch["spatial_target"], batch["spatial_mask"], batch["weight"])
            l_a = masked_weighted_mse(aux_pred, batch["aux_target"], batch["aux_mask"], batch["weight"])
            if l_s is None:
                l_s = torch.tensor(0.0, device=device)
            if l_a is None:
                l_a = torch.tensor(0.0, device=device)
            l_stgcn = l_s + 0.3 * l_a

            loss = l_fusion + 0.2 * l_b + 0.2 * l_stgcn

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss, mean_abs_res, std_res = evaluate_fusion(bilstm, stgcn, fusion, val_loader, device)
        scheduler.step(val_loss)
        train_loss = float(np.mean(losses)) if losses else float("inf")
        print(
            f"[Phase4] epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} "
            f"|res|={mean_abs_res:.3f} res_std={std_res:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best:
            best = val_loss
            worsening = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "bilstm": bilstm.state_dict(),
                    "stgcn": stgcn.state_dict(),
                    "fusion": fusion.state_dict(),
                },
                save_path,
            )
        else:
            worsening += 1
            if worsening >= early_stop_patience:
                print(f"[Phase4] Early stopping: no improvement for {early_stop_patience} epochs. Stopping at epoch {epoch}.")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExeVision Step 2 supervised fine-tuning")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--index-json", type=str, default="dataset/annotations/index.json")
    parser.add_argument("--splits-json", type=str, default="dataset/splits.json")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--bilstm-pretrained", type=str, default="models/bilstm_pretrained.pt")
    parser.add_argument("--stgcn-pretrained", type=str, default="models/stgcn_pretrained.pt")
    parser.add_argument("--bilstm-out", type=str, default="models/bilstm_finetuned.pt")
    parser.add_argument("--stgcn-out", type=str, default="models/stgcn_finetuned.pt")
    parser.add_argument("--fusion-out", type=str, default="models/fusion_layer.pt")
    parser.add_argument("--joint-out", type=str, default="models/joint_finetuned.pt")

    parser.add_argument("--epochs-bilstm", type=int, default=20)
    parser.add_argument("--epochs-stgcn", type=int, default=20)
    parser.add_argument("--epochs-fusion", type=int, default=60)
    parser.add_argument("--epochs-joint", type=int, default=15)

    parser.add_argument("--lr-bilstm", type=float, default=5e-4)
    parser.add_argument("--lr-stgcn", type=float, default=5e-4)
    parser.add_argument("--lr-fusion", type=float, default=5e-4)

    parser.add_argument("--run-phase4", action="store_true", help="Enable optional joint end-to-end fine-tuning")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch per phase for sanity checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = Path(args.root).resolve()
    index_path = (root / args.index_json).resolve()
    splits_path = (root / args.splits_json).resolve()
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    records, video_scores = build_records(root, index_path)
    if not records:
        raise RuntimeError("No indexed annotated reps could be loaded.")

    splits = stratified_video_split(video_scores, seed=args.seed)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    train_records, val_records, test_records = split_records(records, splits)
    assign_training_weights(train_records)
    for r in val_records:
        r.weight = 1.0
    for r in test_records:
        r.weight = 1.0

    print(
        f"Loaded reps={len(records)} train={len(train_records)} val={len(val_records)} "
        f"test={len(test_records)} videos={len(video_scores)}"
    )

    train_ds = MultiModalRepDataset(train_records, training=True)
    val_ds = MultiModalRepDataset(val_records, training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    adjacency = build_adjacency_matrix()
    bilstm = BiLSTMScorer().to(device)
    stgcn = STGCNScorer(adjacency).to(device)
    fusion = HeuristicGuidedFusion().to(device)

    bilstm_pretrained = (root / args.bilstm_pretrained).resolve()
    stgcn_pretrained = (root / args.stgcn_pretrained).resolve()
    bilstm.load_pretrained(str(bilstm_pretrained))
    stgcn.load_pretrained(str(stgcn_pretrained))

    if args.dry_run:
        batch = next(iter(train_loader))
        batch = to_device(batch, device)
        with torch.no_grad():
            bo = bilstm(batch["bilstm"])
            so = stgcn(batch["stgcn"], batch["heuristic"][:, 10:15])
            pred, corr = fusion(batch["heuristic"], so["embedding"], bo["embedding"])
        print(
            "Dry-run shapes:",
            bo["embedding"].shape,
            so["embedding"].shape,
            pred.shape,
            corr.shape,
        )

    train_phase_bilstm(
        bilstm,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_bilstm,
        lr=args.lr_bilstm,
        save_path=(root / args.bilstm_out).resolve(),
    )

    bilstm.load_state_dict(torch.load((root / args.bilstm_out).resolve(), map_location=device))

    train_phase_stgcn(
        stgcn,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_stgcn,
        lr=args.lr_stgcn,
        save_path=(root / args.stgcn_out).resolve(),
    )

    stgcn.load_state_dict(torch.load((root / args.stgcn_out).resolve(), map_location=device))

    train_phase_fusion(
        bilstm,
        stgcn,
        fusion,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs_fusion,
        lr=args.lr_fusion,
        save_path=(root / args.fusion_out).resolve(),
    )

    if args.run_phase4:
        train_optional_joint(
            bilstm,
            stgcn,
            fusion,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs_joint,
            save_path=(root / args.joint_out).resolve(),
        )

    required_files = [
        (root / args.bilstm_out).resolve(),
        (root / args.stgcn_out).resolve(),
        (root / args.fusion_out).resolve(),
        splits_path,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")

    print("Training complete. Required artifacts saved.")


if __name__ == "__main__":
    main()
