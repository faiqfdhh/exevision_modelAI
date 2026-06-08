"""Phase C — LightGBM quality meta-learner, blended with the frozen v1 OHP ensemble.

Trains a small GBM on per-rep tabular features (heuristic anchor + per-metric
heuristic scores + view one-hot + the 5-seed ensemble's OWN predicted heads,
excluding quality) to regress human_score directly. Trees can ignore the bad
heuristic anchor (heuristic_pearson = -0.11) and handle the ~12-sample 40-60
minority better than reweighted deep-net MSE (see A+B rejection in CHANGELOG).

Final score = alpha * neural_ensemble_quality + (1 - alpha) * gbm_pred, with
alpha swept on the val split (never test) to minimise val MAE.

Uses the SAME frozen v1 ensemble as production (fusion_ohp_finetuned*.pt,
quality averaged excluding _seed7 — mirrors evaluate_ohp_ensemble.py @ HEAD).

CLI:
  python train_quality_gbm.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --model-dir      models/runtime_neural_ohp \\
      --output-dir     results \\
      [--save]   # writes quality_gbm.pkl + quality_gbm_meta.json into --model-dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from lightgbm import LGBMRegressor

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
from evaluate_ohp import _nanmae, _pearson, _bucket

EXCLUDE_FUSION = {"_seed7"}

FEATURE_NAMES = [
    "heuristic_score",
    "ens_smoothness", "ens_control", "ens_lockout", "ens_elbow_flare",
    "ens_grip_ratio", "ens_rom_top", "ens_rom_bottom",
    "h_grip_ratio", "h_rom", "h_lockout", "h_elbow_flare",
    "view_front", "view_back", "view_side", "view_front_side", "view_back_side",
]


def _discover_seeds(model_dir: Path) -> list[str]:
    """v1 triples: {bilstm,stgcn,fusion}_ohp_finetuned_seed*.pt (production ensemble)."""
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


def _collect_split(ds, members, exclude_fusion, device):
    """Run the frozen ensemble over a split; return (X, y, neural_quality, view_list)."""
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    X, y, neural_q, views = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preds = {k: [] for k in (
                "quality", "smoothness", "control", "lockout",
                "elbow_flare", "grip_ratio", "rom_top", "rom_bottom")}
            for sfx, bilstm, stgcn, fusion in members:
                b_out = bilstm(bd["bilstm_input"])
                s_out = stgcn(bd["stgcn_input"], bd["view_vec"])
                fscore, _ = fusion(bd["heuristic_vec"], s_out["embedding"], b_out["embedding"])
                if sfx not in exclude_fusion:
                    preds["quality"].append(float(fscore.item()))
                preds["smoothness"].append(float(b_out["smoothness"].item()))
                preds["control"].append(float(b_out["control"].item()))
                preds["lockout"].append(float(s_out["lockout"].item()))
                preds["elbow_flare"].append(float(s_out["elbow_flare"].item()))
                preds["grip_ratio"].append(float(s_out["grip_ratio"].item()))
                preds["rom_top"].append(float(s_out["rom_top"].item()))
                preds["rom_bottom"].append(float(s_out["rom_bottom"].item()))
            avg = {k: float(np.mean(v)) for k, v in preds.items()}

            hvec = bd["heuristic_vec"][0].cpu().numpy()
            feat = [
                float(hvec[0]) * 100.0,
                avg["smoothness"], avg["control"], avg["lockout"], avg["elbow_flare"],
                avg["grip_ratio"], avg["rom_top"], avg["rom_bottom"],
                float(hvec[1]) * 100.0, float(hvec[2]) * 100.0,
                float(hvec[3]) * 100.0, float(hvec[4]) * 100.0,
                float(hvec[11]), float(hvec[12]), float(hvec[13]), float(hvec[14]), float(hvec[15]),
            ]
            X.append(feat)
            y.append(float(bd["quality"].item()))
            neural_q.append(avg["quality"])
            views.append(bd["view"][0])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), \
        np.array(neural_q, dtype=np.float64), views


def _sweep_alpha(neural_pred: np.ndarray, gbm_pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return (best_alpha, best_val_mae) minimising MAE of alpha*neural + (1-alpha)*gbm."""
    best_alpha, best_mae = 1.0, _nanmae(list(neural_pred), list(target))
    print("  alpha curve (val MAE):")
    for alpha in np.arange(0.0, 1.0001, 0.1):
        blend = alpha * neural_pred + (1 - alpha) * gbm_pred
        mae = _nanmae(list(blend), list(target))
        print(f"    alpha={alpha:.1f}  val_mae={mae:.2f}")
        if mae < best_mae:
            best_alpha, best_mae = float(alpha), mae
    return best_alpha, best_mae


def _report(name: str, pred: np.ndarray, target: np.ndarray) -> dict:
    mae = _nanmae(list(pred), list(target))
    pearson = _pearson(list(pred), list(target))
    buckets: dict[str, list[float]] = {}
    for p, t in zip(pred, target):
        buckets.setdefault(_bucket(t), []).append(abs(p - t))
    per_bucket = {b: round(float(np.mean(e)), 4) for b, e in sorted(buckets.items())}
    print(f"  {name:14s}  mae={mae:.2f}  pearson={pearson:.3f}  per_bucket={per_bucket}")
    return {"mae": round(mae, 4), "pearson": round(pearson, 4), "per_bucket_mae": per_bucket}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OHP quality GBM meta-learner (Phase C)")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--model-dir",      type=Path, required=True)
    parser.add_argument("--output-dir",     type=Path, default=Path("results"))
    parser.add_argument("--save", action="store_true",
                        help="Write quality_gbm.pkl + quality_gbm_meta.json into --model-dir")
    args = parser.parse_args()

    device = torch.device("cpu")
    all_paths = sorted((args.annotation_dir / "videos").glob("*.json"))
    train_ds = OHPPhase3Dataset(all_paths, split="train")
    val_ds   = OHPPhase3Dataset(all_paths, split="val")
    test_ds  = OHPPhase3Dataset(all_paths, split="test")
    print(f"splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    seeds = _discover_seeds(args.model_dir)
    if not seeds:
        raise RuntimeError(f"No v1 seed triples found in {args.model_dir}")
    print(f"frozen ensemble seeds: {seeds}  (fusion excluded: {sorted(EXCLUDE_FUSION)})")
    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    members = [(sfx, *_load_member(args.model_dir, sfx, A, device)) for sfx in seeds]

    print("Running frozen ensemble over splits to build GBM features...")
    X_train, y_train, nq_train, _ = _collect_split(train_ds, members, EXCLUDE_FUSION, device)
    X_val,   y_val,   nq_val,   _ = _collect_split(val_ds,   members, EXCLUDE_FUSION, device)
    X_test,  y_test,  nq_test,  _ = _collect_split(test_ds,  members, EXCLUDE_FUSION, device)

    gbm = LGBMRegressor(
        n_estimators=200, learning_rate=0.03,
        num_leaves=7, max_depth=3, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    gbm.fit(X_train, y_train)

    gbm_val  = gbm.predict(X_val)
    gbm_test = gbm.predict(X_test)

    alpha, val_mae_at_alpha = _sweep_alpha(nq_val, gbm_val, y_val)
    print(f"\nalpha sweep (val, n={len(y_val)}): best alpha={alpha:.2f}  val_mae={val_mae_at_alpha:.2f}")

    blend_test = alpha * nq_test + (1 - alpha) * gbm_test

    print(f"\n--- Test split (n={len(y_test)}) ---")
    rep_neural = _report("neural-only",  nq_test,    y_test)
    rep_gbm    = _report("gbm-only",     gbm_test,   y_test)
    rep_blend  = _report(f"blend a={alpha:.2f}", blend_test, y_test)

    baseline_path = _REPO / "results" / "ohp_ensemble_eval_no7.json"
    baseline_mae = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_mae = baseline.get("metrics", {}).get("quality_mae")
        print(f"\nbaseline (locked-in v1 ensemble, {baseline_path.name}): quality_mae={baseline_mae}")

    wins = baseline_mae is not None and rep_blend["mae"] < baseline_mae
    print(f"\n{'WINS' if wins else 'DOES NOT WIN'} vs baseline "
          f"({rep_blend['mae']:.2f} vs {baseline_mae})")

    report = {
        "ensemble_seeds": seeds,
        "fusion_excluded": sorted(EXCLUDE_FUSION),
        "feature_names": FEATURE_NAMES,
        "alpha": alpha,
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "test": {"neural_only": rep_neural, "gbm_only": rep_gbm, "blend": rep_blend},
        "baseline_quality_mae": baseline_mae,
        "wins_vs_baseline": wins,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "ohp_quality_gbm_eval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote eval report -> {out_path}")

    if args.save:
        if not wins:
            print("Refusing --save: GBM blend does not beat baseline. Not writing model files.")
        else:
            model_path = args.model_dir / "quality_gbm.pkl"
            meta_path  = args.model_dir / "quality_gbm_meta.json"
            joblib.dump(gbm, model_path)
            meta_path.write_text(json.dumps({
                "alpha": alpha,
                "feature_names": FEATURE_NAMES,
                "ensemble_seeds": seeds,
                "fusion_excluded": sorted(EXCLUDE_FUSION),
                "test_quality_mae": rep_blend["mae"],
                "test_quality_pearson": rep_blend["pearson"],
            }, indent=2), encoding="utf-8")
            print(f"Saved {model_path.name} + {meta_path.name} -> {args.model_dir}")


if __name__ == "__main__":
    main()
