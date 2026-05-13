"""Phase 3 OHP evaluation — 5-seed ensemble on the held-out test split.

CLI:
  python evaluate_phase3.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --model-dir      models/ \\
      --output         results/ohp_phase3_eval.json

Exit code 0 = PASS (all thresholds met), 1 = FAIL.

Metrics reported (and acceptance thresholds):
  quality_mae     < 12.0  (ensemble avg vs human_score, 0-100 scale)
  lockout_auc     > 0.75  (ROC-AUC for lockout binary head)
  smoothness_mae  < 18.0
  control_mae     < 18.0
  elbow_flare_mae < 15.0
  grip_ratio_mae  < 12.0  (NaN-masked — side view reps excluded)
  rom_top_mae     < 12.0
  rom_bottom_mae  < 12.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO      = Path(__file__).resolve().parents[4]
_NEURAL    = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_ROOT = Path(__file__).resolve().parents[1]  # core/exevision/training
_TRAIN_OHP  = Path(__file__).resolve().parent      # core/exevision/training/ohp
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_ROOT), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp
from models import OHPBiLSTMScorer, OHPSTGCNScorer
from fusion import build_ohp_fusion
from data_phase3 import OHPPhase3Dataset
from tta import apply_tta

ACCEPTANCE_THRESHOLDS = {
    "quality_mae":     12.0,
    "lockout_auc":     0.75,
    "smoothness_mae":  18.0,
    "control_mae":     18.0,
    "elbow_flare_mae": 15.0,
    "grip_ratio_mae":  12.0,
    "rom_top_mae":     12.0,
    "rom_bottom_mae":  12.0,
}

FINAL_SEEDS = [42, 7, 19, 3, 99]


def _nanmae(pred: list[float], target: list[float]) -> float:
    """MAE ignoring NaN targets (for grip_ratio side-view masking etc.)."""
    p = np.array(pred, dtype=np.float32)
    t = np.array(target, dtype=np.float32)
    mask = ~np.isnan(t)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p[mask] - t[mask])))


def _auc_binary(probs: list[float], labels: list[float]) -> float:
    """ROC-AUC without sklearn. Returns nan if only one class present."""
    p = np.array(probs, dtype=np.float32)
    l = np.array(labels, dtype=np.float32)
    n_pos = int(l.sum())
    n_neg = len(l) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sorted_idx = np.argsort(-p)
    l_sorted = l[sorted_idx]
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    for lbl in l_sorted:
        if lbl >= 0.5:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return float(np.trapz(tprs, fprs))


def _load_seed_models(model_dir: Path, device: torch.device):
    """Load all available Phase 3 seed models. Returns list of (bilstm, stgcn, fusion)."""
    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    models = []
    for seed in FINAL_SEEDS:
        bp = model_dir / f"bilstm_ohp_phase3_seed{seed}.pt"
        sp = model_dir / f"stgcn_ohp_phase3_seed{seed}.pt"
        fp = model_dir / f"fusion_ohp_phase3_seed{seed}.pt"
        if not (bp.exists() and sp.exists() and fp.exists()):
            print(f"  [WARN] seed {seed} checkpoints missing — skipping")
            continue
        bilstm = OHPBiLSTMScorer().to(device)
        stgcn  = OHPSTGCNScorer(A).to(device)
        fusion = build_ohp_fusion().to(device)
        bilstm.load_state_dict(torch.load(bp, map_location="cpu"))
        stgcn.load_state_dict(torch.load(sp, map_location="cpu"))
        fusion.load_state_dict(torch.load(fp, map_location="cpu"))
        bilstm.eval(); stgcn.eval(); fusion.eval()
        models.append((bilstm, stgcn, fusion))
    return models


def evaluate(
    annotation_dir: Path,
    model_dir: Path,
    output_path: Path,
    all_data: bool = False,
) -> dict:
    device = torch.device("cpu")

    # Load test split (or all data when --all-data flag used)
    all_paths = sorted(Path(annotation_dir).glob("*.json"))
    split_arg = None if all_data else "test"
    test_ds = OHPPhase3Dataset(all_paths, split=split_arg)
    if len(test_ds) == 0:
        raise RuntimeError(
            f"No {'annotated' if all_data else 'test-split'} reps found in {annotation_dir}. "
            + ("" if all_data else "Ensure annotation JSONs have fitnessaqa_split='test', or use --all-data.")
        )
    print(f"Test set size: {len(test_ds)} reps")

    # Load seed models
    models = _load_seed_models(model_dir, device)
    if not models:
        raise RuntimeError(
            f"No Phase 3 seed checkpoints found in {model_dir}. "
            "Run finetune_phase3.py --final first."
        )
    print(f"Loaded {len(models)} seed model(s)")

    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Accumulators
    acc: dict[str, list[float]] = {
        "quality_pred": [], "quality_true": [],
        "smoothness_pred": [], "smoothness_true": [],
        "control_pred": [], "control_true": [],
        "lockout_pred": [], "lockout_true": [],
        "elbow_flare_pred": [], "elbow_flare_true": [],
        "grip_ratio_pred": [], "grip_ratio_true": [],
        "rom_top_pred": [], "rom_top_true": [],
        "rom_bottom_pred": [], "rom_bottom_true": [],
    }

    with torch.no_grad():
        for batch in loader:
            bilstm_in = batch["bilstm_input"].to(device)
            stgcn_in  = batch["stgcn_input"].to(device)
            hvec      = batch["heuristic_vec"].to(device)
            view_vec  = batch["view_vec"].to(device)

            # Ensemble: average over seeds × TTA variants
            preds: dict[str, list[float]] = {
                k: [] for k in ["quality", "smoothness", "control",
                                 "lockout", "elbow_flare", "grip_ratio",
                                 "rom_top", "rom_bottom"]
            }
            for bilstm, stgcn, fusion in models:
                for v_b, v_s in apply_tta(bilstm_in, stgcn_in):
                    b_out = bilstm(v_b)
                    s_out = stgcn(v_s, view_vec)
                    fscore, _ = fusion(hvec, s_out["embedding"], b_out["embedding"])
                    preds["quality"].append(float(fscore.item()))
                    preds["smoothness"].append(float(b_out["smoothness"].item()))
                    preds["control"].append(float(b_out["control"].item()))
                    preds["lockout"].append(float(s_out["lockout"].item()))
                    preds["elbow_flare"].append(float(s_out["elbow_flare"].item()))
                    preds["grip_ratio"].append(float(s_out["grip_ratio"].item()))
                    preds["rom_top"].append(float(s_out["rom_top"].item()))
                    preds["rom_bottom"].append(float(s_out["rom_bottom"].item()))

            avg = {k: sum(v) / len(v) for k, v in preds.items()}

            acc["quality_pred"].append(avg["quality"])
            acc["quality_true"].append(float(batch["quality"].item()))
            acc["smoothness_pred"].append(avg["smoothness"])
            acc["smoothness_true"].append(float(batch["smoothness"].item()))
            acc["control_pred"].append(avg["control"])
            acc["control_true"].append(float(batch["control"].item()))
            acc["lockout_pred"].append(avg["lockout"])
            acc["lockout_true"].append(float(batch["lockout"].item()))
            acc["elbow_flare_pred"].append(avg["elbow_flare"])
            acc["elbow_flare_true"].append(float(batch["elbow_flare"].item()))
            acc["grip_ratio_pred"].append(avg["grip_ratio"])
            acc["grip_ratio_true"].append(float(batch["grip_ratio"].item()))  # nan for side view
            acc["rom_top_pred"].append(avg["rom_top"])
            acc["rom_top_true"].append(float(batch["rom_top"].item()))
            acc["rom_bottom_pred"].append(avg["rom_bottom"])
            acc["rom_bottom_true"].append(float(batch["rom_bottom"].item()))

    # Compute metrics
    metrics = {
        "quality_mae":     _nanmae(acc["quality_pred"],     acc["quality_true"]),
        "lockout_auc":     _auc_binary(acc["lockout_pred"], acc["lockout_true"]),
        "smoothness_mae":  _nanmae(acc["smoothness_pred"],  acc["smoothness_true"]),
        "control_mae":     _nanmae(acc["control_pred"],     acc["control_true"]),
        "elbow_flare_mae": _nanmae(acc["elbow_flare_pred"], acc["elbow_flare_true"]),
        "grip_ratio_mae":  _nanmae(acc["grip_ratio_pred"],  acc["grip_ratio_true"]),
        "rom_top_mae":     _nanmae(acc["rom_top_pred"],     acc["rom_top_true"]),
        "rom_bottom_mae":  _nanmae(acc["rom_bottom_pred"],  acc["rom_bottom_true"]),
    }

    # Gate against acceptance thresholds
    per_metric_pass = {}
    for key, threshold in ACCEPTANCE_THRESHOLDS.items():
        val = metrics.get(key, float("nan"))
        if np.isnan(val):
            per_metric_pass[key] = False   # NaN = not enough data
        elif key.endswith("_auc"):
            per_metric_pass[key] = val >= threshold
        else:
            per_metric_pass[key] = val <= threshold

    thresholds_met = all(per_metric_pass.values())

    report = {
        "metrics":          {k: round(v, 4) if not np.isnan(v) else None
                             for k, v in metrics.items()},
        "thresholds_met":   thresholds_met,
        "per_metric_pass":  per_metric_pass,
        "test_set_size":    len(test_ds),
        "ensemble_seeds":   FINAL_SEEDS,
        "seeds_loaded":     len(models),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if thresholds_met:
        print("\nPASS — all acceptance thresholds met.")
        return report
    else:
        failed = [k for k, v in per_metric_pass.items() if not v]
        print(f"\nFAIL — thresholds not met for: {', '.join(failed)}")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 OHP evaluation — 5-seed ensemble on test split")
    parser.add_argument("--annotation-dir", type=Path, required=True,
                        help="Path to training_dataset/ohp_phase3_annotations/videos")
    parser.add_argument("--model-dir",      type=Path, required=True,
                        help="Directory containing bilstm/stgcn/fusion_ohp_phase3_seed*.pt files")
    parser.add_argument("--output",         type=Path, required=True,
                        help="Output JSON path (e.g. results/ohp_phase3_eval.json)")
    parser.add_argument("--all-data", action="store_true",
                        help="Evaluate on all annotated reps (no test-split filter)")
    args = parser.parse_args()

    report = evaluate(
        annotation_dir=Path(args.annotation_dir),
        model_dir=Path(args.model_dir),
        output_path=Path(args.output),
        all_data=args.all_data,
    )
    raise SystemExit(0 if report["thresholds_met"] else 1)


if __name__ == "__main__":
    main()
