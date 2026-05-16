"""Evaluate fusion models vs heuristic baseline on all annotations.

Compares fusion predictions against heuristic baseline for both squat and OHP,
reporting summary metrics and improvements.

CLI:
    python core/exevision/evaluation/evaluate1.py \
            --output results/fusion_vs_heuristic_eval.json \
            [--device cpu|cuda] [--squat-only|--ohp-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]  # evaluation/ → exevision/ → core/ → repo
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = _REPO / "core" / "exevision" / "training" / "ohp"
_TRAIN_SQUAT = _REPO / "core" / "exevision" / "training" / "squat"

for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP), str(_TRAIN_SQUAT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_models import (
    BiLSTMScorer,
    HeuristicGuidedFusion,
    STGCNScorer,
    apply_safety_clamps,
    build_heuristic_vector,
)
from nn_utils import (
    FIXED_SEQ_LEN,
    NUM_OHP_ACTIVE_JOINTS,
    NUM_OHP_BILSTM_CHANNELS,
    STGCN_CHANNELS,
    _extract_rep_matrix,
    _extract_stgcn_rep,
    _load_json,
    build_adjacency_matrix,
    build_adjacency_matrix_ohp,
    pad_or_truncate,
)
from models import OHPBiLSTMScorer, OHPSTGCNScorer
from fusion import build_ohp_fusion
from heuristic_vec import build_ohp_heuristic_vector


BUCKET_EDGES = [20.0, 40.0, 60.0, 80.0, 100.0]


def _bucket(score: float) -> str:
    """Assign score to a quality bucket."""
    for i, edge in enumerate(BUCKET_EDGES):
        if score < edge:
            lo = 0.0 if i == 0 else BUCKET_EDGES[i - 1]
            return f"{int(lo)}-{int(edge)}"
    return f"{int(BUCKET_EDGES[-2])}-{int(BUCKET_EDGES[-1])}"


def _nanmae(pred: list[float], target: list[float]) -> float:
    """Mean Absolute Error, ignoring NaN in target."""
    p = np.array(pred, dtype=np.float32)
    t = np.array(target, dtype=np.float32)
    mask = ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p[mask] - t[mask])))


def _rmse(pred: list[float], target: list[float]) -> float:
    """Root Mean Squared Error, ignoring NaN in target."""
    p = np.array(pred, dtype=np.float32)
    t = np.array(target, dtype=np.float32)
    mask = ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2)))


def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation coefficient, ignoring NaN in both."""
    x = np.array(a, dtype=np.float32)
    y = np.array(b, dtype=np.float32)
    if len(x) < 2:
        return float("nan")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _improvement_pct(baseline_mae: float, model_mae: float) -> float:
    """Percentage improvement of model over baseline (lower MAE is better)."""
    if np.isnan(baseline_mae) or np.isnan(model_mae) or baseline_mae <= 0:
        return float("nan")
    return float(100.0 * (baseline_mae - model_mae) / baseline_mae)


def load_squat_models(device: torch.device) -> dict:
    ckpt_dir = _REPO / "models" / "runtime_neural_squat"
    A = torch.tensor(build_adjacency_matrix(), dtype=torch.float32).to(device)

    bilstm = BiLSTMScorer().to(device)
    stgcn = STGCNScorer(A).to(device)
    fusion = HeuristicGuidedFusion(heuristic_dim=15).to(device)

    bilstm.load_state_dict(torch.load(ckpt_dir / "bilstm_finetuned.pt", map_location=device))
    stgcn.load_state_dict(torch.load(ckpt_dir / "stgcn_finetuned.pt", map_location=device))
    fusion.load_state_dict(torch.load(ckpt_dir / "fusion_layer.pt", map_location=device))

    for m in (bilstm, stgcn, fusion):
        m.eval()
    return {"bilstm": bilstm, "stgcn": stgcn, "fusion": fusion}


class OHPEvaluationDataset:
    """Load OHP annotations with features and segmented reps."""

    _DEFAULT_FEAT_DIR = _REPO / "training_dataset" / "ohp_phase3_annotations" / "extracted_features"
    _DEFAULT_SEG_DIR = _REPO / "training_dataset" / "ohp_phase3_annotations" / "segmented_reps"

    def __init__(
        self,
        annotation_dir: Path,
        feat_dir: Optional[Path] = None,
        seg_dir: Optional[Path] = None,
    ) -> None:
        """Load all annotations and index feature/segmented files."""
        self.annotation_dir = Path(annotation_dir)
        self.feat_dir = Path(feat_dir) if feat_dir else self._DEFAULT_FEAT_DIR
        self.seg_dir = Path(seg_dir) if seg_dir else self._DEFAULT_SEG_DIR
        self._records: List[Dict] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load all annotation JSONs from videos/ directory."""
        videos_dir = self.annotation_dir / "videos"
        if not videos_dir.exists():
            print(f"Warning: Videos directory not found: {videos_dir}")
            return

        for anno_path in sorted(videos_dir.glob("*.json")):
            try:
                anno = _load_json(anno_path)
            except Exception as e:
                print(f"Warning: Failed to load {anno_path}: {e}")
                continue

            video_id = anno.get("video_id", anno_path.stem)
            view = anno.get("view", "unknown")

            # Resolve feature path
            feat_path = self.feat_dir / f"{video_id}.json"
            if not feat_path.exists():
                fallback = Path(anno.get("pipeline_outputs", {}).get("features_json", ""))
                if fallback.exists():
                    feat_path = fallback
                else:
                    continue  # Skip videos without features

            # Resolve segmented path (optional)
            seg_path = self.seg_dir / f"{video_id}_segmented.json"
            if not seg_path.exists():
                fallback = Path(anno.get("pipeline_outputs", {}).get("segmented_json", ""))
                seg_path = fallback if fallback.exists() else Path("")

            fps = float(anno.get("fps", 30.0))

            # Index each rep
            for rep in anno.get("reps", []):
                if rep.get("human_score") is None:
                    continue
                self._records.append({
                    "video_id": video_id,
                    "feat_path": feat_path,
                    "seg_path": seg_path,
                    "rep": rep,
                    "view": view,
                    "fps": fps,
                })

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        """Return tensors and labels for one rep."""
        record = self._records[idx]
        rep = record["rep"]

        try:
            feat_data = _load_json(record["feat_path"]) or {}
        except Exception:
            feat_data = {}

        try:
            seg_data = _load_json(record["seg_path"]) if record["seg_path"].exists() else {}
        except Exception:
            seg_data = {}

        # --- BiLSTM input ---
        bilstm_raw = _extract_rep_matrix(seg_data, rep, exercise="overhead_press")
        if bilstm_raw is None:
            bilstm_raw = np.zeros((1, NUM_OHP_BILSTM_CHANNELS), dtype=np.float32)
        bilstm_padded = pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN)
        bilstm_t = torch.from_numpy(bilstm_padded)

        # --- ST-GCN input ---
        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep, exercise="overhead_press")
        if stgcn_raw is None:
            stgcn_raw = np.zeros((1, NUM_OHP_ACTIVE_JOINTS, STGCN_CHANNELS), dtype=np.float32)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)
        stgcn_t = torch.from_numpy(np.transpose(stgcn_padded, (2, 0, 1)).astype(np.float32))

        # --- Heuristic vector ---
        hvec = build_ohp_heuristic_vector(rep, record["view"])

        # --- Human scores ---
        human_score = float(rep.get("human_score", 50.0))
        heuristic_score = float(rep.get("heuristic_score", 50.0))

        return {
            "bilstm_input": bilstm_t,
            "stgcn_input": stgcn_t,
            "heuristic_vec": torch.from_numpy(hvec),
            "human_score": human_score,
            "heuristic_score": heuristic_score,
            "view": record["view"],
            "video_id": record["video_id"],
        }


def resolve_squat_feature_path(root: Path, annotation: dict, video_id: str) -> Optional[Path]:
    pipeline_outputs = annotation.get("pipeline_outputs", {}) or {}
    raw = pipeline_outputs.get("features_json", "")
    if not raw:
        return None

    candidates = [Path(raw)]
    if not raw.startswith(str(root)):
        candidates.append(root / raw)
    if raw.startswith("pipeline_ui_runs/"):
        candidates.append(root / "_hidden_legacy" / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for subdir in ["excellent", "good", "fair", "raw_unfiltered"]:
        fallback = root / "squat" / "extracted_features_clean" / subdir / f"{video_id}.json"
        if fallback.exists():
            return fallback

    return None


def evaluate_squat(models: dict, device: torch.device) -> tuple[list[float], list[float], list[float], list[dict]]:
    """Run fusion inference on all annotated squat reps."""
    anno_dir = _REPO / "training_dataset" / "squat_annotations" / "videos"
    human_scores: list[float] = []
    heuristic_scores: list[float] = []
    fusion_scores: list[float] = []
    per_rep: list[dict] = []

    anno_files = sorted(anno_dir.glob("*.json"))
    print(f"  Found {len(anno_files)} squat annotation files")

    with torch.no_grad():
        for anno_file in anno_files:
            anno = _load_json(anno_file)
            video_id = anno.get("video_id", anno_file.stem)
            view = (anno.get("view") or "unknown").lower()
            fps = float(anno.get("fps", 30.0))
            calibration = (anno.get("calibration", {}) or {})

            feature_path = resolve_squat_feature_path(_REPO, anno, video_id)
            if feature_path is None:
                continue

            feature_data = _load_json(feature_path)
            seg_stub = {"info": {"calibration": calibration, "fps": fps}}

            for rep in (anno.get("reps", []) or []):
                human_score = rep.get("human_score")
                if human_score is None:
                    continue

                signals = (rep.get("signals", {}) or {})
                channels = []
                for key in ["normalized_hip_displacement", "window_velocity", "knee_angles", "landmark_confidence"]:
                    arr = np.asarray(signals.get(key, []), dtype=np.float32)
                    if arr.size == 0:
                        arr = np.zeros((1,), dtype=np.float32)
                    channels.append(arr)
                min_len = min(len(x) for x in channels)
                if min_len <= 0:
                    continue
                bilstm_raw = np.stack([x[:min_len] for x in channels], axis=-1)
                bilstm = pad_or_truncate(bilstm_raw)

                stgcn_raw = _extract_stgcn_rep(seg_stub, feature_data, rep, exercise="squat")
                if stgcn_raw is None:
                    continue
                stgcn = pad_or_truncate(stgcn_raw)

                heur_vec = build_heuristic_vector(rep, view)

                bilstm_t = torch.from_numpy(bilstm).float().unsqueeze(0).to(device)
                stgcn_t = torch.from_numpy(stgcn).float().permute(2, 0, 1).unsqueeze(0).to(device)
                heur_t = torch.from_numpy(heur_vec).float().unsqueeze(0).to(device)
                view_vec = heur_t[:, 10:15]

                bilstm_out = models["bilstm"](bilstm_t)
                stgcn_out = models["stgcn"](stgcn_t, view_vec)
                pred, residual = models["fusion"](heur_t, stgcn_out["embedding"], bilstm_out["embedding"])

                raw_pred = float(pred.detach().cpu().item())
                raw_residual = float(residual.detach().cpu().item())

                clamped = apply_safety_clamps(
                    raw_pred,
                    rep.get("flags", {}) or {},
                    rep.get("flag_severities", {}) or {},
                )

                heur_score = float(rep.get("heuristic_score", 0.0))

                human_scores.append(float(human_score))
                heuristic_scores.append(heur_score)
                fusion_scores.append(float(clamped))

                per_rep.append({
                    "exercise": "squat",
                    "video_id": video_id,
                    "rep_id": rep.get("rep_id", 0),
                    "view": view,
                    "human_score": float(human_score),
                    "heuristic_score": heur_score,
                    "fusion_score": float(clamped),
                    "fusion_raw": raw_pred,
                    "residual": raw_residual,
                })

    return human_scores, heuristic_scores, fusion_scores, per_rep


def evaluate_fusion_vs_heuristic(
    annotation_dir: Path,
    model_dir: Path = None,
    device: str = "cpu",
    output_path: Path = None,
) -> Dict:
    """Evaluate fusion model vs heuristic baseline."""
    if model_dir is None:
        model_dir = _REPO / "models" / "runtime_neural_ohp"
    else:
        model_dir = Path(model_dir)

    if output_path is not None:
        output_path = Path(output_path)

    device = torch.device(device)
    print(f"Evaluating on device: {device}")

    # Load dataset
    print(f"\nLoading annotations from {annotation_dir}...")
    dataset = OHPEvaluationDataset(annotation_dir)
    print(f"Loaded {len(dataset)} reps with human scores")

    if len(dataset) == 0:
        raise RuntimeError("No reps with human scores found.")

    # Load models
    print(f"\nLoading models from {model_dir}...")
    bilstm_path = model_dir / "bilstm_ohp_finetuned.pt"
    stgcn_path = model_dir / "stgcn_ohp_finetuned.pt"
    fusion_path = model_dir / "fusion_ohp_finetuned.pt"

    for p in [bilstm_path, stgcn_path, fusion_path]:
        if not p.exists():
            raise RuntimeError(f"Checkpoint not found: {p}")

    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
    stgcn.load_state_dict(torch.load(stgcn_path, map_location="cpu"))
    fusion.load_state_dict(torch.load(fusion_path, map_location="cpu"))

    bilstm.eval()
    stgcn.eval()
    fusion.eval()
    print("Models loaded successfully")

    # Run inference
    print("\nRunning inference...")
    fusion_scores = []
    heuristic_scores = []
    human_scores = []
    views = []

    with torch.no_grad():
        for i in range(len(dataset)):
            if (i + 1) % max(1, len(dataset) // 10) == 0:
                print(f"  {i + 1}/{len(dataset)}")

            batch = dataset[i]
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            bilstm_input = bd["bilstm_input"].unsqueeze(0).to(device)
            stgcn_input = bd["stgcn_input"].unsqueeze(0).to(device)
            heuristic_vec = bd["heuristic_vec"].unsqueeze(0).to(device)
            view_vec = heuristic_vec[:, 11:16]

            b_out = bilstm(bilstm_input)
            s_out = stgcn(stgcn_input, view_vec)
            fscore, _ = fusion(heuristic_vec, s_out["embedding"], b_out["embedding"])

            fusion_scores.append(float(fscore.item()))
            heuristic_scores.append(bd["heuristic_score"])
            human_scores.append(bd["human_score"])
            views.append(bd["view"])

    print(f"Inference complete. Processed {len(fusion_scores)} reps")

    # Compute metrics
    print("\nComputing metrics...")
    fusion_mae = _nanmae(fusion_scores, human_scores)
    fusion_rmse = _rmse(fusion_scores, human_scores)
    fusion_pearson = _pearson(fusion_scores, human_scores)

    heuristic_mae = _nanmae(heuristic_scores, human_scores)
    heuristic_rmse = _rmse(heuristic_scores, human_scores)
    heuristic_pearson = _pearson(heuristic_scores, human_scores)

    mae_improvement_pct = _improvement_pct(heuristic_mae, fusion_mae)
    rmse_improvement_pct = _improvement_pct(heuristic_rmse, fusion_rmse)

    # Per-bucket stratification
    print("\nStratifying by quality bucket...")
    bucket_data: Dict[str, Dict] = {}
    for fusion_pred, heur_pred, human_true in zip(fusion_scores, heuristic_scores, human_scores):
        bucket = _bucket(human_true)
        if bucket not in bucket_data:
            bucket_data[bucket] = {
                "fusion_preds": [],
                "heuristic_preds": [],
                "human_trues": [],
            }
        bucket_data[bucket]["fusion_preds"].append(fusion_pred)
        bucket_data[bucket]["heuristic_preds"].append(heur_pred)
        bucket_data[bucket]["human_trues"].append(human_true)

    per_bucket_results = {}
    for bucket in sorted(bucket_data.keys()):
        data = bucket_data[bucket]
        fusion_mae_b = _nanmae(data["fusion_preds"], data["human_trues"])
        heur_mae_b = _nanmae(data["heuristic_preds"], data["human_trues"])
        improvement_b = _improvement_pct(heur_mae_b, fusion_mae_b)

        per_bucket_results[bucket] = {
            "count": len(data["fusion_preds"]),
            "fusion_mae": round(fusion_mae_b, 4) if not np.isnan(fusion_mae_b) else None,
            "heuristic_mae": round(heur_mae_b, 4) if not np.isnan(heur_mae_b) else None,
            "improvement_pct": round(improvement_b, 2) if not np.isnan(improvement_b) else None,
        }

    # Per-view stratification
    print("Stratifying by view...")
    view_data: Dict[str, Dict] = {}
    for fusion_pred, heur_pred, human_true, view in zip(
        fusion_scores, heuristic_scores, human_scores, views
    ):
        if view not in view_data:
            view_data[view] = {
                "fusion_preds": [],
                "heuristic_preds": [],
                "human_trues": [],
            }
        view_data[view]["fusion_preds"].append(fusion_pred)
        view_data[view]["heuristic_preds"].append(heur_pred)
        view_data[view]["human_trues"].append(human_true)

    per_view_results = {}
    for view in sorted(view_data.keys()):
        data = view_data[view]
        fusion_mae_v = _nanmae(data["fusion_preds"], data["human_trues"])
        heur_mae_v = _nanmae(data["heuristic_preds"], data["human_trues"])
        improvement_v = _improvement_pct(heur_mae_v, fusion_mae_v)

        per_view_results[view] = {
            "count": len(data["fusion_preds"]),
            "fusion_mae": round(fusion_mae_v, 4) if not np.isnan(fusion_mae_v) else None,
            "heuristic_mae": round(heur_mae_v, 4) if not np.isnan(heur_mae_v) else None,
            "improvement_pct": round(improvement_v, 2) if not np.isnan(improvement_v) else None,
        }

    # Build report
    def _fmt(v: float) -> float | None:
        return round(v, 4) if not np.isnan(v) else None

    report = {
        "metadata": {
            "dataset_size": len(dataset),
            "device": str(device),
        },
        "summary": {
            "fusion_mae": _fmt(fusion_mae),
            "fusion_rmse": _fmt(fusion_rmse),
            "fusion_pearson": _fmt(fusion_pearson),
            "heuristic_mae": _fmt(heuristic_mae),
            "heuristic_rmse": _fmt(heuristic_rmse),
            "heuristic_pearson": _fmt(heuristic_pearson),
            "mae_improvement_pct": round(mae_improvement_pct, 2) if not np.isnan(mae_improvement_pct) else None,
            "rmse_improvement_pct": round(rmse_improvement_pct, 2) if not np.isnan(rmse_improvement_pct) else None,
        },
        "per_bucket": per_bucket_results,
        "per_view": per_view_results,
    }

    # Save report
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FUSION vs HEURISTIC EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {len(dataset)} reps from OHP Phase 3 annotations")
    print(f"\nOverall Metrics:")
    print(f"  Fusion MAE:        {fusion_mae:.2f}  (Pearson: {fusion_pearson:.3f})")
    print(f"  Heuristic MAE:     {heuristic_mae:.2f}  (Pearson: {heuristic_pearson:.3f})")
    print(f"\nImprovement:")
    if not np.isnan(mae_improvement_pct) and mae_improvement_pct > 0:
        print(f"  ✓ Fusion beats heuristic by {mae_improvement_pct:.1f}% on MAE")
    else:
        print(f"  ✗ Heuristic is better ({mae_improvement_pct:.1f}% worse for fusion)")

    print(f"\nPer-Bucket Results:")
    for bucket, metrics in per_bucket_results.items():
        improvement = metrics["improvement_pct"]
        symbol = "✓" if improvement is not None and improvement > 0 else "✗"
        print(
            f"  {symbol} {bucket:8} (n={metrics['count']:3}): "
            f"Fusion={metrics['fusion_mae']}, "
            f"Heur={metrics['heuristic_mae']}, "
            f"Improve={improvement}%"
        )

    print(f"\nPer-View Results:")
    for view, metrics in per_view_results.items():
        improvement = metrics["improvement_pct"]
        symbol = "✓" if improvement is not None and improvement > 0 else "✗"
        print(
            f"  {symbol} {view:12} (n={metrics['count']:3}): "
            f"Fusion={metrics['fusion_mae']}, "
            f"Heur={metrics['heuristic_mae']}, "
            f"Improve={improvement}%"
        )

    print("\n" + "=" * 70)
    return report


def _basic_metrics(human: list[float], heuristic: list[float], fusion: list[float]) -> dict:
    return {
        "n_reps": len(human),
        "heuristic_mae": _nanmae(heuristic, human),
        "heuristic_rmse": _rmse(heuristic, human),
        "heuristic_pearson": _pearson(heuristic, human),
        "fusion_mae": _nanmae(fusion, human),
        "fusion_rmse": _rmse(fusion, human),
        "fusion_pearson": _pearson(fusion, human),
        "mae_improvement": _nanmae(heuristic, human) - _nanmae(fusion, human),
        "rmse_improvement": _rmse(heuristic, human) - _rmse(fusion, human),
        "pearson_improvement": _pearson(fusion, human) - _pearson(heuristic, human),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fusion models vs heuristic baseline"
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Path to OHP Phase 3 annotations (default: training_dataset/ohp_phase3_annotations)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model checkpoints (default: models/runtime_neural_ohp)",
    )
    parser.add_argument(
        "--squat-only",
        action="store_true",
        help="Evaluate squat only",
    )
    parser.add_argument(
        "--ohp-only",
        action="store_true",
        help="Evaluate OHP only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON (default: results/ohp_fusion_vs_heuristic_eval.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()

    if args.squat_only and args.ohp_only:
        print("Choose only one of --squat-only or --ohp-only.")
        return

    annotation_dir = (
        args.annotation_dir
        if args.annotation_dir
        else _REPO / "training_dataset" / "ohp_phase3_annotations"
    )

    results: dict[str, dict] = {}
    per_rep: list[dict] = []
    device = torch.device(args.device)

    if not args.ohp_only:
        print("\n[1/2] Evaluating SQUAT fusion vs heuristic...")
        squat_models = load_squat_models(device)
        human, heuristic, fusion, squat_per_rep = evaluate_squat(squat_models, device)
        print(f"  Loaded {len(human)} squat reps with valid features")
        if human:
            squat_metrics = _basic_metrics(human, heuristic, fusion)
            results["squat"] = squat_metrics
            per_rep.extend(squat_per_rep)
            print("\nSQUAT SUMMARY")
            print(f"  MAE:     heuristic={squat_metrics['heuristic_mae']:.2f}  fusion={squat_metrics['fusion_mae']:.2f}")
            print(f"  Pearson: heuristic={squat_metrics['heuristic_pearson']:.3f}  fusion={squat_metrics['fusion_pearson']:.3f}")

    if not args.squat_only:
        print("\n[2/2] Evaluating OHP fusion vs heuristic...")
        ohp_report = evaluate_fusion_vs_heuristic(
            annotation_dir=annotation_dir,
            model_dir=args.model_dir,
            device=args.device,
            output_path=None,
        )
        results["ohp"] = ohp_report

    output_path = _REPO / (args.output or "results/fusion_vs_heuristic_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": results,
        "per_rep": per_rep,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()