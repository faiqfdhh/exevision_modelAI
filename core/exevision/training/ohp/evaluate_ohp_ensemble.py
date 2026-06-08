"""OHP ensemble evaluation — averages N seed checkpoints at the prediction level.

Mirrors evaluate_ohp.py exactly (same metric functions, same acceptance gates) but
loads every {bilstm,stgcn,fusion}_ohp_finetuned_seed*.pt triple, runs each per rep,
and averages the scalar outputs BEFORE computing metrics. Averaging the lockout
probability is what reorders borderline reps and lifts lockout_auc.

CLI:
  python evaluate_ohp_ensemble.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --model-dir      models/runtime_neural_ohp \\
      --output         results/ohp_ensemble_eval.json \\
      [--all-data]
"""
from __future__ import annotations

import argparse
import json
import re
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
# Reuse the single-model eval's metric functions verbatim — identical comparison.
from evaluate_ohp import _nanmae, _pearson, _auc, _bucket, ACCEPTANCE


def _discover_seeds(model_dir: Path) -> list[str]:
    """Return seed suffixes (e.g. ['_seed42', '_seed7']) with a complete triple."""
    seeds = []
    for b in sorted(model_dir.glob("bilstm_ohp_finetuned_seed*.pt")):
        m = re.search(r"(_seed\w+)\.pt$", b.name)
        if not m:
            continue
        sfx = m.group(1)
        if (model_dir / f"stgcn_ohp_finetuned{sfx}.pt").exists() and \
           (model_dir / f"fusion_ohp_finetuned{sfx}.pt").exists():
            seeds.append(sfx)
    return seeds


def _load_member(model_dir: Path, sfx: str, A: torch.Tensor, device: torch.device):
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn  = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)
    bilstm.load_state_dict(torch.load(model_dir / f"bilstm_ohp_finetuned{sfx}.pt", map_location="cpu"))
    stgcn.load_state_dict(torch.load(model_dir / f"stgcn_ohp_finetuned{sfx}.pt", map_location="cpu"))
    fusion.load_state_dict(torch.load(model_dir / f"fusion_ohp_finetuned{sfx}.pt", map_location="cpu"))
    bilstm.eval(); stgcn.eval(); fusion.eval()
    return bilstm, stgcn, fusion


def evaluate_ensemble(
    annotation_dir: Path,
    model_dir: Path,
    output_path: Path,
    all_data: bool = False,
    exclude_fusion: set[str] | None = None,
) -> dict:
    exclude_fusion = exclude_fusion or set()
    device = torch.device("cpu")

    split = None if all_data else "test"
    all_paths = sorted((annotation_dir / "videos").glob("*.json"))
    ds = OHPPhase3Dataset(all_paths, split=split)
    if len(ds) == 0:
        raise RuntimeError("No reps found. Run stamp_phase3_splits.py first, or use --all-data.")

    seeds = _discover_seeds(model_dir)
    if not seeds:
        raise RuntimeError(f"No *_seed*.pt triples found in {model_dir}. Run finetune_ohp.py with --suffix.")
    print(f"Evaluating ENSEMBLE of {len(seeds)} seeds {seeds} on {len(ds)} reps "
          f"({'all data' if all_data else 'test split'})")

    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    members = [(sfx, *_load_member(model_dir, sfx, A, device)) for sfx in seeds]
    fusion_seeds = [s for s in seeds if s not in exclude_fusion]
    if exclude_fusion:
        print(f"Excluding fusion (quality) from seeds: {sorted(exclude_fusion)} "
              f"-> quality averaged over {len(fusion_seeds)} seeds {fusion_seeds}")

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

            # Run every member; collect scalar outputs to average.
            preds = {k: [] for k in (
                "quality", "smoothness", "control", "lockout",
                "elbow_flare", "grip_ratio", "rom_top", "rom_bottom")}
            for sfx, bilstm, stgcn, fusion in members:
                b_out = bilstm(bd["bilstm_input"])
                s_out = stgcn(bd["stgcn_input"], bd["view_vec"])
                fscore, _ = fusion(bd["heuristic_vec"], s_out["embedding"], b_out["embedding"])
                if sfx not in exclude_fusion:        # quality = fusion only; honor exclusion
                    preds["quality"].append(float(fscore.item()))
                preds["smoothness"].append(float(b_out["smoothness"].item()))
                preds["control"].append(float(b_out["control"].item()))
                preds["lockout"].append(float(s_out["lockout"].item()))
                preds["elbow_flare"].append(float(s_out["elbow_flare"].item()))
                preds["grip_ratio"].append(float(s_out["grip_ratio"].item()))
                preds["rom_top"].append(float(s_out["rom_top"].item()))
                preds["rom_bottom"].append(float(s_out["rom_bottom"].item()))

            avg = {k: float(np.mean(v)) for k, v in preds.items()}
            h_score = float(bd["heuristic_vec"][0, 0].item() * 100)

            acc["quality_pred"].append(avg["quality"])
            acc["quality_true"].append(float(bd["quality"].item()))
            acc["heuristic"].append(h_score)
            acc["smoothness_pred"].append(avg["smoothness"])
            acc["smoothness_true"].append(float(bd["smoothness"].item()))
            acc["control_pred"].append(avg["control"])
            acc["control_true"].append(float(bd["control"].item()))
            acc["lockout_pred"].append(avg["lockout"])
            acc["lockout_true"].append(float(bd["lockout"].item()))
            acc["elbow_flare_pred"].append(avg["elbow_flare"])
            acc["elbow_flare_true"].append(float(bd["elbow_flare"].item()))
            acc["grip_ratio_pred"].append(avg["grip_ratio"])
            acc["grip_ratio_true"].append(float(bd["grip_ratio"].item()))
            acc["rom_top_pred"].append(avg["rom_top"])
            acc["rom_top_true"].append(float(bd["rom_top"].item()))
            acc["rom_bottom_pred"].append(avg["rom_bottom"])
            acc["rom_bottom_true"].append(float(bd["rom_bottom"].item()))

    quality_mae       = _nanmae(acc["quality_pred"], acc["quality_true"])
    quality_pearson   = _pearson(acc["quality_pred"], acc["quality_true"])
    heuristic_mae     = _nanmae(acc["heuristic"],    acc["quality_true"])
    heuristic_pearson = _pearson(acc["heuristic"],   acc["quality_true"])

    residuals = [p - h for p, h in zip(acc["quality_pred"], acc["heuristic"])]
    residual_std  = float(np.std(residuals)) if residuals else float("nan")
    residual_mean = float(np.mean(residuals)) if residuals else float("nan")

    smoothness_mae  = _nanmae(acc["smoothness_pred"],  acc["smoothness_true"])
    control_mae     = _nanmae(acc["control_pred"],     acc["control_true"])
    lockout_auc     = _auc(acc["lockout_pred"],        acc["lockout_true"])
    elbow_flare_mae = _nanmae(acc["elbow_flare_pred"], acc["elbow_flare_true"])
    grip_ratio_mae  = _nanmae(acc["grip_ratio_pred"],  acc["grip_ratio_true"])
    rom_top_mae     = _nanmae(acc["rom_top_pred"],     acc["rom_top_true"])
    rom_bottom_mae  = _nanmae(acc["rom_bottom_pred"],  acc["rom_bottom_true"])

    bucket_errors: dict[str, list[float]] = {}
    for pred, true in zip(acc["quality_pred"], acc["quality_true"]):
        bucket_errors.setdefault(_bucket(true), []).append(abs(pred - true))
    per_bucket_mae = {b: round(float(np.mean(e)), 4) for b, e in sorted(bucket_errors.items())}

    metrics_for_gate = {
        "quality_mae": quality_mae, "lockout_auc": lockout_auc,
        "smoothness_mae": smoothness_mae, "control_mae": control_mae,
        "elbow_flare_mae": elbow_flare_mae, "grip_ratio_mae": grip_ratio_mae,
        "rom_top_mae": rom_top_mae, "rom_bottom_mae": rom_bottom_mae,
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

    def _fmt(v: float):
        return round(v, 4) if not np.isnan(v) else None

    report = {
        "test_set_size": len(ds),
        "split": "all" if all_data else "test",
        "ensemble_seeds": seeds,
        "metrics": {
            "quality_mae": _fmt(quality_mae), "quality_pearson": _fmt(quality_pearson),
            "heuristic_mae": _fmt(heuristic_mae), "heuristic_pearson": _fmt(heuristic_pearson),
            "residual_std": _fmt(residual_std), "residual_mean": _fmt(residual_mean),
            "smoothness_mae": _fmt(smoothness_mae), "control_mae": _fmt(control_mae),
            "lockout_auc": _fmt(lockout_auc), "elbow_flare_mae": _fmt(elbow_flare_mae),
            "grip_ratio_mae": _fmt(grip_ratio_mae), "rom_top_mae": _fmt(rom_top_mae),
            "rom_bottom_mae": _fmt(rom_bottom_mae),
        },
        "per_bucket_mae": per_bucket_mae,
        "thresholds_met": thresholds_met,
        "per_metric_pass": per_metric_pass,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    print("\n--- Ensemble Summary ---")
    print(f"seeds:             {seeds}")
    print(f"quality_mae:       {quality_mae:.2f}  (heuristic baseline: {heuristic_mae:.2f})")
    print(f"quality_pearson:   {quality_pearson:.3f}")
    print(f"lockout_auc:       {lockout_auc:.3f}  (gate 0.75)")
    if thresholds_met:
        print("\nPASS — all acceptance thresholds met.")
    else:
        failed = [k for k, v in per_metric_pass.items() if not v]
        print(f"\nFAIL — thresholds not met: {', '.join(failed)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OHP ensemble evaluation")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--model-dir",      type=Path, required=True)
    parser.add_argument("--output",         type=Path, required=True)
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--exclude-fusion", type=str, default="",
                        help="Comma-separated seed suffixes to drop from the QUALITY average "
                             "only (e.g. _seed7). Other heads still use all seeds.")
    args = parser.parse_args()
    exclude = {s.strip() for s in args.exclude_fusion.split(",") if s.strip()}
    report = evaluate_ensemble(args.annotation_dir, args.model_dir, args.output,
                               args.all_data, exclude_fusion=exclude)
    raise SystemExit(0 if report["thresholds_met"] else 1)


if __name__ == "__main__":
    main()
