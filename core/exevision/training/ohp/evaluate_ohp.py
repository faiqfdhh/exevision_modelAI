"""OHP evaluation — squat methodology.

Loads test split, runs single-model inference, reports comprehensive metrics.

CLI:
  python evaluate_ohp.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --model-dir      models/ \\
      --output         results/ohp_eval.json \\
      [--all-data]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO        = Path(__file__).resolve().parents[4]
_NEURAL      = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL  = _NEURAL / "ohp"
_TRAIN_OHP   = Path(__file__).resolve().parent
_TRAIN_OHP_PRETRAIN = _REPO / "core" / "exevision" / "training" / "overhead_press"
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP_PRETRAIN), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp
from models import OHPBiLSTMScorer, OHPSTGCNScorer
from fusion import build_ohp_fusion
from data_phase3 import OHPPhase3Dataset

BUCKET_EDGES = [20.0, 40.0, 60.0, 80.0, 100.0]

ACCEPTANCE = {
    "quality_mae":     12.0,
    "lockout_auc":     0.75,
    "smoothness_mae":  18.0,
    "control_mae":     18.0,
    "elbow_flare_mae": 15.0,
    "grip_ratio_mae":  12.0,
    "rom_top_mae":     12.0,
    "rom_bottom_mae":  12.0,
}


def _nanmae(pred: list[float], target: list[float]) -> float:
    p = np.array(pred, dtype=np.float32)
    t = np.array(target, dtype=np.float32)
    mask = ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p[mask] - t[mask])))


def _pearson(a: list[float], b: list[float]) -> float:
    x = np.array(a, dtype=np.float32)
    y = np.array(b, dtype=np.float32)
    if len(x) < 2:
        return float("nan")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _auc(probs: list[float], labels: list[float]) -> float:
    p = np.array(probs, dtype=np.float32)
    l = np.array(labels, dtype=np.float32)
    n_pos, n_neg = int(l.sum()), int((1 - l).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sorted_idx = np.argsort(-p)
    l_s = l[sorted_idx]
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    for lbl in l_s:
        if lbl >= 0.5:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return float(np.trapz(tprs, fprs))


def _bucket(score: float) -> str:
    for i, edge in enumerate(BUCKET_EDGES):
        if score < edge:
            lo = 0.0 if i == 0 else BUCKET_EDGES[i - 1]
            return f"{int(lo)}-{int(edge)}"
    return f"{int(BUCKET_EDGES[-2])}-{int(BUCKET_EDGES[-1])}"


def evaluate(
    annotation_dir: Path,
    model_dir: Path,
    output_path: Path,
    all_data: bool = False,
) -> dict:
    device = torch.device("cpu")

    split = None if all_data else "test"
    all_paths = sorted((annotation_dir / "videos").glob("*.json"))
    ds = OHPPhase3Dataset(all_paths, split=split)
    if len(ds) == 0:
        raise RuntimeError(
            f"No {'annotated' if all_data else 'test-split'} reps found. "
            + ("" if all_data else "Run stamp_phase3_splits.py first, or use --all-data.")
        )
    print(f"Evaluating on {len(ds)} reps ({'all data' if all_data else 'test split'})")

    # Load models
    bilstm_path = model_dir / "bilstm_ohp_finetuned.pt"
    stgcn_path  = model_dir / "stgcn_ohp_finetuned.pt"
    fusion_path = model_dir / "fusion_ohp_finetuned.pt"
    for p in [bilstm_path, stgcn_path, fusion_path]:
        if not p.exists():
            raise RuntimeError(f"Checkpoint not found: {p}. Run finetune_ohp.py first.")

    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn  = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)
    bilstm.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
    stgcn.load_state_dict(torch.load(stgcn_path, map_location="cpu"))
    fusion.load_state_dict(torch.load(fusion_path, map_location="cpu"))
    bilstm.eval(); stgcn.eval(); fusion.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    acc: dict[str, list] = {k: [] for k in [
        "quality_pred", "quality_true", "heuristic",
        "smoothness_pred", "smoothness_true",
        "control_pred", "control_true",
        "lockout_pred", "lockout_true",
        "elbow_flare_pred", "elbow_flare_true",
        "grip_ratio_pred", "grip_ratio_true",
        "rom_top_pred", "rom_top_true",
        "rom_bottom_pred", "rom_bottom_true",
    ]}

    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            b_out = bilstm(bd["bilstm_input"])
            s_out = stgcn(bd["stgcn_input"], bd["view_vec"])
            fscore, _ = fusion(bd["heuristic_vec"], s_out["embedding"], b_out["embedding"])

            # Heuristic score is the first element of heuristic_vec × 100
            h_score = float(bd["heuristic_vec"][0, 0].item() * 100)

            acc["quality_pred"].append(float(fscore.item()))
            acc["quality_true"].append(float(bd["quality"].item()))
            acc["heuristic"].append(h_score)
            acc["smoothness_pred"].append(float(b_out["smoothness"].item()))
            acc["smoothness_true"].append(float(bd["smoothness"].item()))
            acc["control_pred"].append(float(b_out["control"].item()))
            acc["control_true"].append(float(bd["control"].item()))
            acc["lockout_pred"].append(float(s_out["lockout"].item()))
            acc["lockout_true"].append(float(bd["lockout"].item()))
            acc["elbow_flare_pred"].append(float(s_out["elbow_flare"].item()))
            acc["elbow_flare_true"].append(float(bd["elbow_flare"].item()))
            acc["grip_ratio_pred"].append(float(s_out["grip_ratio"].item()))
            acc["grip_ratio_true"].append(float(bd["grip_ratio"].item()))
            acc["rom_top_pred"].append(float(s_out["rom_top"].item()))
            acc["rom_top_true"].append(float(bd["rom_top"].item()))
            acc["rom_bottom_pred"].append(float(s_out["rom_bottom"].item()))
            acc["rom_bottom_true"].append(float(bd["rom_bottom"].item()))

    # ── Primary metrics ────────────────────────────────────────────────────────
    quality_mae      = _nanmae(acc["quality_pred"], acc["quality_true"])
    quality_pearson  = _pearson(acc["quality_pred"], acc["quality_true"])
    heuristic_mae    = _nanmae(acc["heuristic"],    acc["quality_true"])
    heuristic_pearson = _pearson(acc["heuristic"],  acc["quality_true"])

    # ── Residual diagnostics ───────────────────────────────────────────────────
    residuals = [p - h for p, h in zip(acc["quality_pred"], acc["heuristic"])]
    residual_std  = float(np.std(residuals)) if residuals else float("nan")
    residual_mean = float(np.mean(residuals)) if residuals else float("nan")

    # ── Secondary metrics ──────────────────────────────────────────────────────
    smoothness_mae  = _nanmae(acc["smoothness_pred"],  acc["smoothness_true"])
    control_mae     = _nanmae(acc["control_pred"],     acc["control_true"])
    lockout_auc     = _auc(acc["lockout_pred"],        acc["lockout_true"])
    elbow_flare_mae = _nanmae(acc["elbow_flare_pred"], acc["elbow_flare_true"])
    grip_ratio_mae  = _nanmae(acc["grip_ratio_pred"],  acc["grip_ratio_true"])
    rom_top_mae     = _nanmae(acc["rom_top_pred"],     acc["rom_top_true"])
    rom_bottom_mae  = _nanmae(acc["rom_bottom_pred"],  acc["rom_bottom_true"])

    # ── Per-bucket quality MAE ─────────────────────────────────────────────────
    bucket_errors: dict[str, list[float]] = {}
    for pred, true in zip(acc["quality_pred"], acc["quality_true"]):
        b = _bucket(true)
        bucket_errors.setdefault(b, []).append(abs(pred - true))
    per_bucket_mae = {b: round(float(np.mean(errs)), 4) for b, errs in sorted(bucket_errors.items())}

    # ── Failure cases ──────────────────────────────────────────────────────────
    failure_cases = [
        {"pred": round(p, 2), "true": round(t, 2), "error": round(abs(p - t), 2)}
        for p, t in zip(acc["quality_pred"], acc["quality_true"])
        if abs(p - t) > 20
    ]

    # ── Acceptance thresholds ─────────────────────────────────────────────────
    metrics_for_gate = {
        "quality_mae":     quality_mae,
        "lockout_auc":     lockout_auc,
        "smoothness_mae":  smoothness_mae,
        "control_mae":     control_mae,
        "elbow_flare_mae": elbow_flare_mae,
        "grip_ratio_mae":  grip_ratio_mae,
        "rom_top_mae":     rom_top_mae,
        "rom_bottom_mae":  rom_bottom_mae,
    }
    per_metric_pass = {}
    for key, threshold in ACCEPTANCE.items():
        val = metrics_for_gate.get(key, float("nan"))
        if np.isnan(val):
            per_metric_pass[key] = False
        elif key.endswith("_auc"):
            per_metric_pass[key] = val >= threshold
        else:
            per_metric_pass[key] = val <= threshold

    thresholds_met = all(per_metric_pass.values())

    def _fmt(v: float) -> float | None:
        return round(v, 4) if not np.isnan(v) else None

    report = {
        "test_set_size": len(ds),
        "split": "all" if all_data else "test",
        "metrics": {
            "quality_mae":       _fmt(quality_mae),
            "quality_pearson":   _fmt(quality_pearson),
            "heuristic_mae":     _fmt(heuristic_mae),
            "heuristic_pearson": _fmt(heuristic_pearson),
            "residual_std":      _fmt(residual_std),
            "residual_mean":     _fmt(residual_mean),
            "smoothness_mae":    _fmt(smoothness_mae),
            "control_mae":       _fmt(control_mae),
            "lockout_auc":       _fmt(lockout_auc),
            "elbow_flare_mae":   _fmt(elbow_flare_mae),
            "grip_ratio_mae":    _fmt(grip_ratio_mae),
            "rom_top_mae":       _fmt(rom_top_mae),
            "rom_bottom_mae":    _fmt(rom_bottom_mae),
        },
        "per_bucket_mae":   per_bucket_mae,
        "failure_cases":    failure_cases[:20],  # cap output
        "thresholds_met":   thresholds_met,
        "per_metric_pass":  per_metric_pass,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n--- Summary ---")
    print(f"quality_mae:       {quality_mae:.2f}  (heuristic baseline: {heuristic_mae:.2f})")
    print(f"quality_pearson:   {quality_pearson:.3f}  (heuristic baseline: {heuristic_pearson:.3f})")
    print(f"residual_std:      {residual_std:.3f}  "
          f"({'model adapts beyond heuristic' if residual_std > 2.0 else 'WARNING: model echoes heuristic' if residual_std < 0.5 else 'moderate adaptation'})")
    print(f"lockout_auc:       {lockout_auc:.3f}")
    if failure_cases:
        print(f"failure cases:     {len(failure_cases)} reps with |error| > 20")

    if thresholds_met:
        print("\nPASS — all acceptance thresholds met.")
    else:
        failed = [k for k, v in per_metric_pass.items() if not v]
        print(f"\nFAIL — thresholds not met: {', '.join(failed)}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OHP evaluation — squat methodology")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--model-dir",      type=Path, required=True)
    parser.add_argument("--output",         type=Path, required=True)
    parser.add_argument("--all-data", action="store_true",
                        help="Evaluate on all reps regardless of split")
    args = parser.parse_args()

    report = evaluate(
        annotation_dir=args.annotation_dir,
        model_dir=args.model_dir,
        output_path=args.output,
        all_data=args.all_data,
    )
    raise SystemExit(0 if report["thresholds_met"] else 1)


if __name__ == "__main__":
    main()
