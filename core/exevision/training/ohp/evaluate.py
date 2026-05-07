from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[4]
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from data import build_dataloaders


def _auc_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn dependency."""
    sorted_idx = np.argsort(-probs)
    labels_sorted = labels[sorted_idx]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for lbl in labels_sorted:
        if lbl >= 0.5:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    return float(np.trapz(tpr, fpr))


def evaluate(
    annotation_dir: Path,
    checkpoint_dir: Path,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cpu")

    loaders = build_dataloaders(annotation_dir, batch_size=batch_size)
    if "test" not in loaders:
        raise RuntimeError("No test split found")

    A = torch.tensor(build_adjacency_matrix_ohp())
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm.load_state_dict(torch.load(checkpoint_dir / "bilstm_ohp_phase2.pt", map_location="cpu"))
    stgcn.load_state_dict(torch.load(checkpoint_dir / "stgcn_ohp_phase2.pt", map_location="cpu"))
    fusion.load_state_dict(torch.load(checkpoint_dir / "fusion_ohp_phase2.pt", map_location="cpu"))

    bilstm.eval(); stgcn.eval(); fusion.eval()

    preds, targets, knee_probs, knee_true = [], [], [], []

    with torch.no_grad():
        for batch in loaders["test"]:
            bilstm_out = bilstm(batch["bilstm_input"])
            stgcn_out = stgcn(batch["stgcn_input"], batch["view_vec"])
            score, _ = fusion(batch["heuristic_vec"], stgcn_out["embedding"], bilstm_out["embedding"])
            preds.extend(score.tolist())
            targets.extend(batch["overall_score"].tolist())
            knee_probs.extend(bilstm_out["knee_error"].tolist())
            knee_true.extend(batch["knee_error"].tolist())

    preds_arr = np.array(preds)
    targets_arr = np.array(targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    knee_auc = _auc_binary(np.array(knee_probs), np.array(knee_true))

    results = {
        "exercise": "overhead_press",
        "n_test_reps": len(preds),
        "mae_overall": round(mae, 3),
        "knee_error_auc": round(knee_auc, 3) if not np.isnan(knee_auc) else "n/a",
        "acceptance": {
            "mae_pass": mae < 15.0,
            "knee_auc_pass": knee_auc > 0.65 if not np.isnan(knee_auc) else False,
        },
    }
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 standing OHP model (knee-only)")
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    evaluate(Path(args.annotation_dir), Path(args.checkpoint_dir))


if __name__ == "__main__":
    main()
