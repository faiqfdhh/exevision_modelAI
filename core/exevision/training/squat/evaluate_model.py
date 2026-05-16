from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from finetune_models import (
    MultiModalRepDataset,
    build_records,
    get_device,
    score_bucket,
    set_seed,
    split_records,
    stratified_video_split,
    to_device,
)
from nn_models import BiLSTMScorer, HeuristicGuidedFusion, STGCNScorer, apply_safety_clamps
from nn_utils import build_adjacency_matrix


FLAG_NAMES = [
    "insufficient_squat_depth",
    "knee_valgus",
    "lumbar_flexion",
    "heel_rise",
    "asymmetric_descent",
    "forward_lean",
]

METRIC_NAMES = ["smoothness", "control", "depth", "forward_lean", "knee_tracking"]


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
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(xv - yv)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ExeVision Step 2 models")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--index-json", type=str, default="dataset/annotations/index.json")
    parser.add_argument("--splits-json", type=str, default="dataset/splits.json")

    parser.add_argument("--bilstm-ckpt", type=str, default="models/runtime_neural_squat/bilstm_finetuned.pt")
    parser.add_argument("--stgcn-ckpt", type=str, default="models/runtime_neural_squat/stgcn_finetuned.pt")
    parser.add_argument("--fusion-ckpt", type=str, default="models/runtime_neural_squat/fusion_layer.pt")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/evaluation_report.json")
    parser.add_argument(
        "--n-splits", type=int, default=1,
        help="Repeat evaluation with N different random stratified splits and report mean±std. "
             "Default 1 = single split (original seed, writes JSON report as usual)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = Path(args.root).resolve()
    index_path = (root / args.index_json).resolve()
    splits_path = (root / args.splits_json).resolve()
    device = get_device(force_cpu=args.cpu)

    if not splits_path.exists():
        raise RuntimeError("Missing dataset splits. Run finetune_models first.")

    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    records, _ = build_records(root, index_path)
    train_records, _, test_records = split_records(records, splits)
    if not test_records:
        raise RuntimeError("Test split is empty.")

    test_ds = MultiModalRepDataset(test_records, training=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    adjacency = build_adjacency_matrix()
    bilstm = BiLSTMScorer().to(device)
    stgcn = STGCNScorer(adjacency).to(device)
    fusion = HeuristicGuidedFusion().to(device)

    bilstm.load_state_dict(torch.load((root / args.bilstm_ckpt).resolve(), map_location=device))
    stgcn.load_state_dict(torch.load((root / args.stgcn_ckpt).resolve(), map_location=device))
    fusion.load_state_dict(torch.load((root / args.fusion_ckpt).resolve(), map_location=device))

    bilstm.eval()
    stgcn.eval()
    fusion.eval()

    human_scores: List[float] = []
    heuristic_scores: List[float] = []
    pred_pre_clamp: List[float] = []
    pred_post_clamp: List[float] = []
    corrections: List[float] = []

    per_metric_pairs: Dict[str, List[List[float]]] = {k: [] for k in METRIC_NAMES}
    per_rep_rows: List[Dict[str, float]] = []

    idx_offset = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_size = batch["human_score"].shape[0]
            batch = to_device(batch, device)

            bo = bilstm(batch["bilstm"])
            so = stgcn(batch["stgcn"], batch["heuristic"][:, 10:15])
            pred, corr = fusion(batch["heuristic"], so["embedding"], bo["embedding"])

            pred_np = pred.detach().cpu().numpy()
            corr_np = corr.detach().cpu().numpy()
            smooth_np = (bo["smoothness"].detach().cpu().numpy() * 100.0)
            control_np = (bo["control"].detach().cpu().numpy() * 100.0)
            depth_np = (so["depth"].detach().cpu().numpy() * 100.0)
            lean_np = (so["forward_lean"].detach().cpu().numpy() * 100.0)
            knee_np = (so["knee_tracking"].detach().cpu().numpy() * 100.0)

            temporal_target = (batch["temporal_target"].detach().cpu().numpy() * 100.0)
            temporal_mask = batch["temporal_mask"].detach().cpu().numpy()
            spatial_target = (batch["spatial_target"].detach().cpu().numpy() * 100.0)
            spatial_mask = batch["spatial_mask"].detach().cpu().numpy()

            for i in range(batch_size):
                rec = test_records[idx_offset + i]
                clamped = apply_safety_clamps(
                    float(pred_np[i]),
                    rec.heuristic_flags,
                    rec.flag_severities,
                )

                human_scores.append(float(rec.human_score))
                heuristic_scores.append(float(rec.heuristic_score))
                pred_pre_clamp.append(float(pred_np[i]))
                pred_post_clamp.append(float(clamped))
                corrections.append(float(corr_np[i]))

                if temporal_mask[i, 0] > 0:
                    per_metric_pairs["smoothness"].append([float(smooth_np[i]), float(temporal_target[i, 0])])
                if temporal_mask[i, 1] > 0:
                    per_metric_pairs["control"].append([float(control_np[i]), float(temporal_target[i, 1])])
                if spatial_mask[i, 0] > 0:
                    per_metric_pairs["depth"].append([float(depth_np[i]), float(spatial_target[i, 0])])
                if spatial_mask[i, 1] > 0:
                    per_metric_pairs["forward_lean"].append([float(lean_np[i]), float(spatial_target[i, 1])])
                if spatial_mask[i, 2] > 0:
                    per_metric_pairs["knee_tracking"].append([float(knee_np[i]), float(spatial_target[i, 2])])

                per_rep_rows.append(
                    {
                        "video_id": rec.video_id,
                        "rep_id": rec.rep_id,
                        "human_score": float(rec.human_score),
                        "heuristic_score": float(rec.heuristic_score),
                        "predicted_score": float(clamped),
                        "pre_clamp_score": float(pred_np[i]),
                        "residual": float(corr_np[i]),
                    }
                )

            idx_offset += batch_size

    train_x = np.asarray([r.heuristic_vec for r in train_records], dtype=np.float32)
    train_y = np.asarray([r.human_score for r in train_records], dtype=np.float32)
    test_x = np.asarray([r.heuristic_vec for r in test_records], dtype=np.float32)
    linreg = LinearRegression()
    linreg.fit(train_x, train_y)
    linreg_pred = linreg.predict(test_x).tolist()

    post_corr = pearson(pred_post_clamp, human_scores)
    post_mae = mae(pred_post_clamp, human_scores)
    pre_corr = pearson(pred_pre_clamp, human_scores)
    pre_mae = mae(pred_pre_clamp, human_scores)

    heuristic_corr = pearson(heuristic_scores, human_scores)
    heuristic_mae = mae(heuristic_scores, human_scores)

    linear_corr = pearson(linreg_pred, human_scores)
    linear_mae = mae(linreg_pred, human_scores)

    metric_mae = {}
    for k, pairs in per_metric_pairs.items():
        if not pairs:
            metric_mae[k] = None
            continue
        pred_k = [p[0] for p in pairs]
        true_k = [p[1] for p in pairs]
        metric_mae[k] = mae(pred_k, true_k)

    flag_agreement = {}
    for flag in FLAG_NAMES:
        total = 0
        match = 0
        for r in test_records:
            if flag in r.human_flags:
                total += 1
                if bool(r.heuristic_flags.get(flag, False)) == bool(r.human_flags.get(flag, False)):
                    match += 1
        flag_agreement[flag] = None if total == 0 else (100.0 * match / total)

    bucket_mae = {}
    for b in range(5):
        y_true = []
        y_pred = []
        for i, r in enumerate(test_records):
            if score_bucket(r.human_score) == b:
                y_true.append(float(r.human_score))
                y_pred.append(float(pred_post_clamp[i]))
        label = ["0-20", "20-40", "40-60", "60-80", "80-100"][b]
        bucket_mae[label] = None if not y_true else mae(y_pred, y_true)

    failure_cases = []
    for i, r in enumerate(test_records):
        err = abs(float(pred_post_clamp[i]) - float(r.human_score))
        if err > 20.0:
            failure_cases.append(
                {
                    "video_id": r.video_id,
                    "rep_id": r.rep_id,
                    "human_score": float(r.human_score),
                    "predicted_score": float(pred_post_clamp[i]),
                    "residual": float(corrections[i]),
                    "abs_error": float(err),
                }
            )

    # Diagnostic: Clamp impact analysis
    clamp_hits = sum(1 for i in range(len(test_records)) if pred_pre_clamp[i] != pred_post_clamp[i])
    clamp_deltas = [abs(pred_post_clamp[i] - pred_pre_clamp[i]) for i in range(len(test_records)) if pred_pre_clamp[i] != pred_post_clamp[i]]
    mean_clamp_delta = float(np.mean(clamp_deltas)) if clamp_deltas else 0.0

    # Diagnostic: Residual analysis (human - heuristic gap)
    residuals_true = [human_scores[i] - heuristic_scores[i] for i in range(len(human_scores))]
    residuals_pred_pre = [pred_pre_clamp[i] - heuristic_scores[i] for i in range(len(heuristic_scores))]
    residual_mae_pre = mae(residuals_pred_pre, residuals_true)

    # Diagnostic: Bucket coverage
    bucket_counts = {}
    for r in test_records:
        b = score_bucket(r.human_score)
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    bucket_coverage = {
        "0-20": bucket_counts.get(0, 0),
        "20-40": bucket_counts.get(1, 0),
        "40-60": bucket_counts.get(2, 0),
        "60-80": bucket_counts.get(3, 0),
        "80-100": bucket_counts.get(4, 0),
    }

    report = {
        "counts": {
            "train_reps": len(train_records),
            "test_reps": len(test_records),
        },
        "primary_metrics_post_clamp": {
            "pearson": post_corr,
            "mae": post_mae,
        },
        "diagnostic_metrics_pre_clamp": {
            "pearson": pre_corr,
            "mae": pre_mae,
        },
        "heuristic_baseline": {
            "pearson": heuristic_corr,
            "mae": heuristic_mae,
        },
        "linear_baseline": {
            "pearson": linear_corr,
            "mae": linear_mae,
        },
        "improvement": {
            "fusion_vs_heuristic_corr_delta": post_corr - heuristic_corr,
            "fusion_vs_linear_corr_delta": post_corr - linear_corr,
        },
        "per_metric_mae": metric_mae,
        "per_flag_agreement_percent": flag_agreement,
        "score_bucket_mae": bucket_mae,
        "clamp_diagnostics": {
            "clamp_hit_count": int(clamp_hits),
            "mean_clamp_delta": float(mean_clamp_delta),
        },
        "residual_mae_diagnostics": {
            "residual_mae_pre_clamp": float(residual_mae_pre),
            "note": "MAE between (pred - heuristic) and (human - heuristic); measures learned residual quality independent of baseline"
        },
        "bucket_coverage": bucket_coverage,
        "failure_cases": failure_cases,
        "residual_statistics": {
            "mean": float(np.mean(corrections)) if corrections else 0.0,
            "std": float(np.std(corrections)) if corrections else 0.0,
            "min": float(np.min(corrections)) if corrections else 0.0,
            "max": float(np.max(corrections)) if corrections else 0.0,
            "note": "std > 2.0 indicates meaningful per-rep corrections (not collapsed); std < 0.5 indicates heuristic echo",
        },
        "per_rep_table": per_rep_rows,
    }

    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Evaluation Summary ===")
    print(f"Test reps: {len(test_records)}")
    print(f"Fusion (post-clamp): corr={post_corr:.4f}, mae={post_mae:.3f}")
    print(f"Fusion (pre-clamp): corr={pre_corr:.4f}, mae={pre_mae:.3f}")
    print(f"Heuristic baseline: corr={heuristic_corr:.4f}, mae={heuristic_mae:.3f}")
    print(f"Linear baseline: corr={linear_corr:.4f}, mae={linear_mae:.3f}")
    print(f"Delta vs heuristic: {post_corr - heuristic_corr:.4f}")
    print(f"Delta vs linear: {post_corr - linear_corr:.4f}")
    print(f"Failure cases (>20 abs error): {len(failure_cases)}")
    print(f"Saved report: {output_path}")

    # Repeated-split confidence interval (only when --n-splits > 1)
    if args.n_splits > 1:
        print(f"\n=== Confidence Interval over {args.n_splits} Random Splits ===")
        all_records_full, video_scores_full = build_records(root, index_path)
        multi_post_corr: List[float] = [post_corr]
        multi_post_mae: List[float] = [post_mae]
        multi_pre_corr: List[float] = [pre_corr]
        multi_pre_mae: List[float] = [pre_mae]

        for split_i in range(1, args.n_splits):
            seed_i = args.seed + split_i * 17
            splits_i = stratified_video_split(video_scores_full, seed=seed_i)
            _, _, test_recs_i = split_records(all_records_full, splits_i)
            train_recs_i, _, _ = split_records(all_records_full, splits_i)
            if not test_recs_i:
                continue

            ds_i = MultiModalRepDataset(test_recs_i, training=False)
            loader_i = DataLoader(ds_i, batch_size=args.batch_size, shuffle=False, drop_last=False)

            hs_i: List[float] = []
            pp_i: List[float] = []
            pre_i: List[float] = []
            offset_i = 0
            with torch.no_grad():
                for batch_i in loader_i:
                    bsz = batch_i["human_score"].shape[0]
                    batch_i = to_device(batch_i, device)
                    bo_i = bilstm(batch_i["bilstm"])
                    so_i = stgcn(batch_i["stgcn"], batch_i["heuristic"][:, 10:15])
                    pred_i, _ = fusion(batch_i["heuristic"], so_i["embedding"], bo_i["embedding"])
                    pred_np_i = pred_i.detach().cpu().numpy()
                    for j in range(bsz):
                        r = test_recs_i[offset_i + j]
                        clamped_i = apply_safety_clamps(float(pred_np_i[j]), r.heuristic_flags, r.flag_severities)
                        hs_i.append(float(r.human_score))
                        pp_i.append(float(clamped_i))
                        pre_i.append(float(pred_np_i[j]))
                    offset_i += bsz

            multi_post_corr.append(pearson(pp_i, hs_i))
            multi_post_mae.append(mae(pp_i, hs_i))
            multi_pre_corr.append(pearson(pre_i, hs_i))
            multi_pre_mae.append(mae(pre_i, hs_i))

        print(f"Post-clamp Pearson: {np.mean(multi_post_corr):.4f} ± {np.std(multi_post_corr):.4f}  (splits: {[round(v,4) for v in multi_post_corr]})")
        print(f"Post-clamp MAE:     {np.mean(multi_post_mae):.3f} ± {np.std(multi_post_mae):.3f}  (splits: {[round(v,3) for v in multi_post_mae]})")
        print(f"Pre-clamp Pearson:  {np.mean(multi_pre_corr):.4f} ± {np.std(multi_pre_corr):.4f}")
        print(f"Pre-clamp MAE:      {np.mean(multi_pre_mae):.3f} ± {np.std(multi_pre_mae):.3f}")


if __name__ == "__main__":
    main()
