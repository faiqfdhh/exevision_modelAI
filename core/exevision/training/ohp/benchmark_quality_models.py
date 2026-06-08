"""Benchmark alternative OHP quality meta-learners against the deployed LightGBM.

Reuses the EXACT feature pipeline from train_quality_gbm.py (frozen 5-seed ensemble
-> 17 tabular features -> human_score target; zero leakage, identical to production).

Protocol (honest small-data evaluation, 119/23/23 train/val/test rows):
  1. CV on train+val combined (142 rows) via RepeatedKFold(5x5) -- the real signal.
  2. One-shot refit-on-(train+val) -> predict test -- the final guardrail, NOT the selector.
A candidate is only worth deploying if it wins CV AND beats the locked-in test MAE 7.30.

CLI:
  python benchmark_quality_models.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --model-dir      models/runtime_neural_ohp \\
      --output-dir     results \\
      [--save]   # overwrite quality_gbm.pkl + quality_gbm_meta.json IF the winner clears both bars
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

_REPO        = Path(__file__).resolve().parents[4]
_NEURAL      = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL  = _NEURAL / "ohp"
_TRAIN_OHP   = Path(__file__).resolve().parent
_TRAIN_OHP_PRETRAIN = _REPO / "core" / "exevision" / "training" / "overhead_press"
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP_PRETRAIN), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp
from data_phase3 import OHPPhase3Dataset
from evaluate_ohp import _nanmae, _pearson, _bucket
from train_quality_gbm import (
    _discover_seeds, _load_member, _collect_split, _sweep_alpha, _report,
    FEATURE_NAMES, EXCLUDE_FUSION,
)

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

CV = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
GBM_TEST_MAE_BAR = 7.30   # locked-in deployed LightGBM (results/ohp_quality_gbm_eval.json)


def _build_candidates() -> dict:
    cands = {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0, random_state=42))]),
        "elasticnet": Pipeline([("scale", StandardScaler()),
                                ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=5000))]),
        "svr_rbf": Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, epsilon=2.0))]),
        "gaussian_process": Pipeline([("scale", StandardScaler()), ("model", GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
            alpha=1e-6, normalize_y=True, random_state=42))]),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=4, random_state=42),
        "xgboost": XGBRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=3, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0),
        "lightgbm": LGBMRegressor(
            n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=0.5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
    }
    if _HAS_CATBOOST:
        cands["catboost"] = CatBoostRegressor(
            iterations=300, depth=3, learning_rate=0.03, l2_leaf_reg=5.0,
            random_state=42, verbose=False)

    cands["stack_ridge_lgbm"] = StackingRegressor(
        estimators=[
            ("ridge", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0, random_state=42))])),
            ("lgbm", LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3,
                                   min_child_samples=10, reg_alpha=0.1, reg_lambda=0.5,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
        ],
        final_estimator=Ridge(alpha=1.0, random_state=42), cv=5,
    )
    return cands


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OHP quality meta-learner candidates")
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--model-dir",      type=Path, required=True)
    parser.add_argument("--output-dir",     type=Path, default=Path("results"))
    parser.add_argument("--save", action="store_true")
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
    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    members = [(sfx, *_load_member(args.model_dir, sfx, A, device)) for sfx in seeds]

    print("Running frozen ensemble over splits to build features (identical to train_quality_gbm.py)...")
    X_train, y_train, nq_train, _ = _collect_split(train_ds, members, EXCLUDE_FUSION, device)
    X_val,   y_val,   nq_val,   _ = _collect_split(val_ds,   members, EXCLUDE_FUSION, device)
    X_test,  y_test,  nq_test,  _ = _collect_split(test_ds,  members, EXCLUDE_FUSION, device)

    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)
    print(f"CV pool: {len(y_cv)} rows (train+val) | test (one-shot guardrail): {len(y_test)} rows\n")

    candidates = _build_candidates()
    if not _HAS_CATBOOST:
        print("NOTE: catboost not installed -- skipping that candidate.\n")

    results = []
    for name, est in candidates.items():
        cv_scores = cross_val_score(est, X_cv, y_cv, cv=CV, scoring="neg_mean_absolute_error", n_jobs=1)
        cv_mae_mean, cv_mae_std = float(-cv_scores.mean()), float(cv_scores.std())

        est.fit(X_cv, y_cv)
        test_pred = est.predict(X_test)
        test_mae = _nanmae(list(test_pred), list(y_test))
        test_pearson = _pearson(list(test_pred), list(y_test))
        buckets: dict[str, list[float]] = {}
        for p, t in zip(test_pred, y_test):
            buckets.setdefault(_bucket(t), []).append(abs(p - t))
        per_bucket = {b: round(float(np.mean(e)), 4) for b, e in sorted(buckets.items())}

        print(f"{name:18s}  cv_mae={cv_mae_mean:6.2f} (+/-{cv_mae_std:.2f})   "
              f"test_mae={test_mae:6.2f}  test_pearson={test_pearson:.3f}  bucket={per_bucket}")
        results.append({
            "name": name, "cv_mae_mean": round(cv_mae_mean, 4), "cv_mae_std": round(cv_mae_std, 4),
            "test_mae": round(test_mae, 4), "test_pearson": round(test_pearson, 4),
            "per_bucket_mae": per_bucket,
        })

    results.sort(key=lambda r: r["cv_mae_mean"])
    winner = results[0]
    gbm_row = next((r for r in results if r["name"] == "lightgbm"), None)

    print(f"\n--- Ranked by CV MAE (train+val, {len(y_cv)} rows, RepeatedKFold 5x5) ---")
    for r in results:
        tag = "  <-- CV winner" if r is winner else ("  (deployed baseline)" if r["name"] == "lightgbm" else "")
        print(f"  {r['name']:18s} cv_mae={r['cv_mae_mean']:.2f} (+/-{r['cv_mae_std']:.2f})  "
              f"test_mae={r['test_mae']:.2f}{tag}")

    clears_cv = gbm_row is not None and winner["cv_mae_mean"] < gbm_row["cv_mae_mean"]
    clears_test = winner["test_mae"] < GBM_TEST_MAE_BAR
    wins = winner["name"] != "lightgbm" and clears_cv and clears_test

    print(f"\nCV winner: {winner['name']}  (cv_mae={winner['cv_mae_mean']:.2f}, test_mae={winner['test_mae']:.2f})")
    print(f"Beats deployed LightGBM on CV: {clears_cv}   Beats locked-in test bar {GBM_TEST_MAE_BAR}: {clears_test}")
    print(f"{'SWAP RECOMMENDED' if wins else 'KEEP DEPLOYED LIGHTGBM'} "
          f"({'wins both CV and one-shot test' if wins else 'does not clear both bars -- GBM stays'})")

    report = {
        "cv_protocol": "RepeatedKFold(n_splits=5, n_repeats=5) on train+val (142 rows)",
        "deployed_baseline": "lightgbm (quality_gbm.pkl, alpha=0)",
        "deployed_test_mae_bar": GBM_TEST_MAE_BAR,
        "ranked_by_cv_mae": results,
        "cv_winner": winner["name"],
        "swap_recommended": wins,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "ohp_quality_model_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark report -> {out_path}")

    if args.save:
        if not wins:
            print("Refusing --save: no candidate clears both the CV and one-shot test bars. quality_gbm.pkl untouched.")
        else:
            winner_est = candidates[winner["name"]]
            winner_est.fit(X_cv, y_cv)
            winner_test_pred = winner_est.predict(X_test)
            alpha, _ = _sweep_alpha(nq_val, winner_est.predict(X_val), y_val)
            blended_test = alpha * nq_test + (1 - alpha) * winner_test_pred
            blended_rep = _report(f"{winner['name']} blend a={alpha:.2f}", blended_test, y_test)

            model_path = args.model_dir / "quality_gbm.pkl"
            meta_path  = args.model_dir / "quality_gbm_meta.json"
            joblib.dump(winner_est, model_path)
            meta_path.write_text(json.dumps({
                "model_type": winner["name"],
                "alpha": alpha,
                "feature_names": FEATURE_NAMES,
                "ensemble_seeds": seeds,
                "fusion_excluded": sorted(EXCLUDE_FUSION),
                "test_quality_mae": blended_rep["mae"],
                "test_quality_pearson": blended_rep["pearson"],
                "cv_mae_mean": winner["cv_mae_mean"],
                "cv_mae_std": winner["cv_mae_std"],
            }, indent=2), encoding="utf-8")
            print(f"Saved {model_path.name} ({winner['name']}) + {meta_path.name} -> {args.model_dir}")


if __name__ == "__main__":
    main()
