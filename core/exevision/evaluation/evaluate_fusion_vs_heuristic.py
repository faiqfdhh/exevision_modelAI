"""Evaluate fusion models vs heuristic baseline on all annotated data.

Compares HeuristicGuidedFusion predictions against pure heuristic scores
using human annotator scores as ground truth. Evaluates both squat and OHP.

Usage:
    python evaluate_fusion_vs_heuristic.py
    python evaluate_fusion_vs_heuristic.py --cpu --output results/my_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]  # evaluation/ → exevision/ → core/ → repo/
NEURAL = REPO / "core" / "exevision" / "neural"
OHP_NEURAL = NEURAL / "ohp"
TRAIN = REPO / "core" / "exevision" / "training"
TRAIN_SQUAT = TRAIN / "squat"
TRAIN_OHP = TRAIN / "ohp"

for _p in [str(NEURAL), str(OHP_NEURAL), str(TRAIN_SQUAT), str(TRAIN_OHP)]:
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
    _extract_rep_matrix,
    _extract_stgcn_rep,
    _load_json,
    build_adjacency_matrix,
    build_adjacency_matrix_ohp,
    pad_or_truncate,
)
from ohp.fusion import build_ohp_fusion
from ohp.heuristic_vec import build_ohp_heuristic_vector
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fusion models vs heuristic baseline")
    p.add_argument("--root", type=str, default=str(REPO),
                   help="Repository root directory")
    p.add_argument("--output", type=str, default="results/fusion_vs_heuristic_eval.json",
                   help="Output JSON path")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p.add_argument("--squat-only", action="store_true", help="Evaluate squat only")
    p.add_argument("--ohp-only", action="store_true", help="Evaluate OHP only")
    return p.parse_args()


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def mae(x: List[float], y: List[float]) -> float:
    if not x:
        return 0.0
    return float(np.mean(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))))


def rmse(x: List[float], y: List[float]) -> float:
    if not x:
        return 0.0
    return float(np.sqrt(np.mean((np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))


def load_squat_models(device: torch.device) -> dict:
    ckpt_dir = REPO / "models" / "runtime_neural_squat"
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


def load_ohp_models(device: torch.device) -> dict:
    ckpt_dir = REPO / "models" / "runtime_neural_ohp"
    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)

    bilstm = OHPBiLSTMScorer().to(device)
    stgcn = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm.load_state_dict(torch.load(ckpt_dir / "bilstm_ohp_finetuned.pt", map_location=device))
    stgcn.load_state_dict(torch.load(ckpt_dir / "stgcn_ohp_finetuned.pt", map_location=device))
    fusion.load_state_dict(torch.load(ckpt_dir / "fusion_ohp_finetuned.pt", map_location=device))

    for m in (bilstm, stgcn, fusion):
        m.eval()
    return {"bilstm": bilstm, "stgcn": stgcn, "fusion": fusion}


def resolve_squat_feature_path(root: Path, annotation: dict, video_id: str) -> Optional[Path]:
    po = annotation.get("pipeline_outputs", {}) or {}
    raw = po.get("features_json", "")
    if not raw:
        return None

    candidates = [Path(raw)]
    if not raw.startswith(str(root)):
        candidates.append(root / raw)
    if raw.startswith("pipeline_ui_runs/"):
        candidates.append(root / "_hidden_legacy" / raw)

    for c in candidates:
        if c.exists():
            return c

    for subdir in ["excellent", "good", "fair", "raw_unfiltered"]:
        fallback = root / "squat" / "extracted_features_clean" / subdir / f"{video_id}.json"
        if fallback.exists():
            return fallback

    return None


def evaluate_squat(models: dict, device: torch.device) -> Tuple[List[float], List[float], List[float], List[dict]]:
    """Run fusion inference on all annotated squat reps."""
    anno_dir = REPO / "training_dataset" / "squat_annotations" / "videos"
    human_scores: List[float] = []
    heuristic_scores: List[float] = []
    fusion_scores: List[float] = []
    per_rep: List[dict] = []

    anno_files = sorted(anno_dir.glob("*.json"))
    print(f"  Found {len(anno_files)} squat annotation files")

    with torch.no_grad():
        for anno_file in anno_files:
            anno = _load_json(anno_file)
            video_id = anno.get("video_id", anno_file.stem)
            view = (anno.get("view") or "unknown").lower()
            fps = float(anno.get("fps", 30.0))
            calibration = (anno.get("calibration", {}) or {})

            feature_path = resolve_squat_feature_path(REPO, anno, video_id)
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
                for k in ["normalized_hip_displacement", "window_velocity",
                          "knee_angles", "landmark_confidence"]:
                    arr = np.asarray(signals.get(k, []), dtype=np.float32)
                    if arr.size == 0:
                        arr = np.zeros((1,), dtype=np.float32)
                    channels.append(arr)
                min_len = min(len(x) for x in channels)
                if min_len <= 0:
                    continue
                bilstm_raw = np.stack([x[:min_len] for x in channels], axis=-1)
                bilstm = pad_or_truncate(bilstm_raw)

                stgcn_raw = _extract_stgcn_rep(seg_stub, feature_data, rep)
                if stgcn_raw is None:
                    continue
                stgcn = pad_or_truncate(stgcn_raw)

                heur_vec = build_heuristic_vector(rep, view)

                bilstm_t = torch.from_numpy(bilstm).float().unsqueeze(0).to(device)
                stgcn_t = torch.from_numpy(stgcn).float().permute(2, 0, 1).unsqueeze(0).to(device)
                heur_t = torch.from_numpy(heur_vec).float().unsqueeze(0).to(device)
                view_vec = heur_t[:, 10:15]

                bo = models["bilstm"](bilstm_t)
                so = models["stgcn"](stgcn_t, view_vec)
                pred, residual = models["fusion"](heur_t, so["embedding"], bo["embedding"])

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


def evaluate_ohp(models: dict, device: torch.device) -> Tuple[List[float], List[float], List[float], List[dict]]:
    """Run fusion inference on all annotated OHP reps."""
    anno_dir = REPO / "training_dataset" / "ohp_phase3_annotations" / "videos"
    feat_dir = REPO / "training_dataset" / "ohp_phase3_annotations" / "extracted_features"
    seg_dir = REPO / "training_dataset" / "ohp_phase3_annotations" / "segmented_reps"

    human_scores: List[float] = []
    heuristic_scores: List[float] = []
    fusion_scores: List[float] = []
    per_rep: List[dict] = []

    anno_files = sorted(anno_dir.glob("*.json"))
    print(f"  Found {len(anno_files)} OHP annotation files")

    with torch.no_grad():
        for anno_file in anno_files:
            anno = _load_json(anno_file)
            video_id = anno.get("video_id", anno_file.stem)
            view = (anno.get("view") or "unknown").lower()

            feat_path = feat_dir / f"{video_id}.json"
            if not feat_path.exists():
                continue
            seg_path = seg_dir / f"{video_id}_segmented.json"

            feat_data = _load_json(feat_path)
            seg_data = _load_json(seg_path) if seg_path.exists() else {}

            for rep in (anno.get("reps", []) or []):
                human_score = rep.get("human_score")
                if human_score is None:
                    continue

                bilstm_raw = _extract_rep_matrix(seg_data, rep, exercise="overhead_press")
                if bilstm_raw is None:
                    bilstm_raw = np.zeros((1, 8), dtype=np.float32)
                bilstm = pad_or_truncate(bilstm_raw)

                stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep, exercise="overhead_press")
                if stgcn_raw is None:
                    stgcn_raw = np.zeros((1, 10, 7), dtype=np.float32)
                stgcn = pad_or_truncate(stgcn_raw)

                heur_vec = build_ohp_heuristic_vector(rep, view)

                bilstm_t = torch.from_numpy(bilstm).float().unsqueeze(0).to(device)
                stgcn_t = torch.from_numpy(stgcn).float().permute(2, 0, 1).unsqueeze(0).to(device)
                heur_t = torch.from_numpy(heur_vec).float().unsqueeze(0).to(device)
                view_vec = heur_t[:, 11:16]

                bo = models["bilstm"](bilstm_t)
                so = models["stgcn"](stgcn_t, view_vec)
                pred, residual = models["fusion"](heur_t, so["embedding"], bo["embedding"])

                raw_pred = float(pred.detach().cpu().item())
                raw_residual = float(residual.detach().cpu().item())
                clamped = max(0.0, min(100.0, raw_pred))

                heur_score = float(rep.get("heuristic_score", 0.0))

                human_scores.append(float(human_score))
                heuristic_scores.append(heur_score)
                fusion_scores.append(float(clamped))

                per_rep.append({
                    "exercise": "overhead_press",
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


def compute_metrics(name: str, human: List[float], heuristic: List[float],
                    fusion: List[float]) -> dict:
    return {
        "name": name,
        "n_reps": len(human),
        "heuristic_mae": mae(heuristic, human),
        "heuristic_pearson": pearson(heuristic, human),
        "heuristic_rmse": rmse(heuristic, human),
        "fusion_mae": mae(fusion, human),
        "fusion_pearson": pearson(fusion, human),
        "fusion_rmse": rmse(fusion, human),
        "mae_improvement": mae(heuristic, human) - mae(fusion, human),
        "pearson_improvement": pearson(fusion, human) - pearson(heuristic, human),
        "rmse_improvement": rmse(heuristic, human) - rmse(fusion, human),
    }


def print_report(all_metrics: List[dict]) -> None:
    print()
    print("=" * 72)
    print("  FUSION MODEL vs HEURISTIC BASELINE — EVALUATION REPORT")
    print("=" * 72)

    for m in all_metrics:
        print(f"\n── {m['name']} ({m['n_reps']} reps) ──")
        print(f"  {'':24s} {'Heuristic':>12s} {'Fusion':>12s} {'Improvement':>14s}")
        print(f"  {'MAE':24s} {m['heuristic_mae']:12.2f} {m['fusion_mae']:12.2f} {m['mae_improvement']:+13.2f}")
        print(f"  {'Pearson r':24s} {m['heuristic_pearson']:12.3f} {m['fusion_pearson']:12.3f} {m['pearson_improvement']:+13.3f}")
        print(f"  {'RMSE':24s} {m['heuristic_rmse']:12.2f} {m['fusion_rmse']:12.2f} {m['rmse_improvement']:+13.2f}")

    if len(all_metrics) > 1:
        all_human = []
        all_heuristic = []
        all_fusion = []
        for m in all_metrics:
            all_human.extend(m.get("_human", []))
            all_heuristic.extend(m.get("_heuristic", []))
            all_fusion.extend(m.get("_fusion", []))
        combined = compute_metrics("COMBINED", all_human, all_heuristic, all_fusion)
        print(f"\n── {combined['name']} ({combined['n_reps']} reps) ──")
        print(f"  {'MAE':24s} {combined['heuristic_mae']:12.2f} {combined['fusion_mae']:12.2f} {combined['mae_improvement']:+13.2f}")
        print(f"  {'Pearson r':24s} {combined['heuristic_pearson']:12.3f} {combined['fusion_pearson']:12.3f} {combined['pearson_improvement']:+13.3f}")
        print(f"  {'RMSE':24s} {combined['heuristic_rmse']:12.2f} {combined['fusion_rmse']:12.2f} {combined['rmse_improvement']:+13.2f}")

    print("\n" + "=" * 72)
    print("  Positive improvement = fusion beats heuristic.")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    all_metrics: List[dict] = []
    all_per_rep: List[dict] = []

    if not args.ohp_only:
        print("\n[1/2] Evaluating SQUAT fusion vs heuristic...")
        squat_models = load_squat_models(device)
        human, heuristic, fusion, per_rep = evaluate_squat(squat_models, device)
        print(f"  Loaded {len(human)} squat reps with valid features")
        if human:
            m = compute_metrics("SQUAT", human, heuristic, fusion)
            m["_human"] = human
            m["_heuristic"] = heuristic
            m["_fusion"] = fusion
            all_metrics.append(m)
            all_per_rep.extend(per_rep)

    if not args.squat_only:
        print("\n[2/2] Evaluating OHP fusion vs heuristic...")
        ohp_models = load_ohp_models(device)
        human, heuristic, fusion, per_rep = evaluate_ohp(ohp_models, device)
        print(f"  Loaded {len(human)} OHP reps with valid features")
        if human:
            m = compute_metrics("OVERHEAD PRESS", human, heuristic, fusion)
            m["_human"] = human
            m["_heuristic"] = heuristic
            m["_fusion"] = fusion
            all_metrics.append(m)
            all_per_rep.extend(per_rep)

    if not all_per_rep:
        print("No valid reps found. Check that annotation files and features exist.")
        return

    print_report(all_metrics)

    output_path = REPO / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": [{k: v for k, v in m.items() if not k.startswith("_")} for m in all_metrics],
        "per_rep": all_per_rep,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
