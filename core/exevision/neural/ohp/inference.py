from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch

_NEURAL = Path(__file__).resolve().parents[1]
_OHP = Path(__file__).resolve().parent
_TRAIN_OHP = Path(__file__).resolve().parents[2] / "training" / "ohp"
for _p in [str(_NEURAL), str(_OHP), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix, _extract_stgcn_rep, _extract_rep_matrix, pad_or_truncate, FIXED_SEQ_LEN
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from ohp.heuristic_vec import build_ohp_heuristic_vector
from nn_models import HeuristicGuidedFusion   # fusion base class


def _load_checkpoints(model_dir: Path, exercise: str):
    """Load BiLSTM, ST-GCN, and fusion checkpoints for the given exercise."""
    suffix = "overhead_press" if exercise == "overhead_press" else "seated_overhead_press"
    # Prefer phase2 checkpoints; fall back to phase-labelled names
    def _ckpt(name):
        for candidate in [
            model_dir / f"{name}_{suffix}.pt",
            model_dir / f"{name}_ohp_phase2.pt",
            model_dir / f"{name}_seated_ohp_phase2.pt",
        ]:
            if candidate.exists():
                return candidate
        return None

    bilstm_path = _ckpt("bilstm")
    stgcn_path = _ckpt("stgcn")
    fusion_path = _ckpt("fusion")
    return bilstm_path, stgcn_path, fusion_path


def run_ohp_inference(args) -> None:
    """Entry point called from neural_fusion_inference.py for OHP exercises."""
    workspace = Path(args.workspace_root)
    exercise = args.exercise
    video_id = args.video_id
    model_dir = Path(args.model_dir) if hasattr(args, "model_dir") else Path("models")

    include_knee = exercise != "seated_overhead_press"
    device = torch.device("cpu")
    A = torch.tensor(build_adjacency_matrix())

    bilstm = OHPBiLSTMScorer(include_knee_head=include_knee).to(device)
    stgcn = OHPSTGCNScorer(A, include_knee_head=include_knee).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm_path, stgcn_path, fusion_path = _load_checkpoints(model_dir, exercise)
    if bilstm_path:
        bilstm.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
    if stgcn_path:
        stgcn.load_state_dict(torch.load(stgcn_path, map_location="cpu"))
    if fusion_path:
        fusion.load_state_dict(torch.load(fusion_path, map_location="cpu"))

    bilstm.eval(); stgcn.eval(); fusion.eval()

    # Load stage outputs
    tier = getattr(args, "quality", "raw_unfiltered")
    feat_path = workspace / exercise / "extracted_features_clean" / tier / f"{video_id}.json"
    seg_path = workspace / exercise / "segmented_reps" / tier / f"{video_id}_segmented.json"
    score_path = workspace / exercise / "aqa_analysis_simple" / tier / video_id / f"{video_id}_aqa_simple.json"

    if not feat_path.exists() or not seg_path.exists():
        print(json.dumps({"error": f"Stage outputs not found for {video_id}", "neural_available": False}))
        return

    feat_data = json.loads(feat_path.read_text())
    seg_data = json.loads(seg_path.read_text())
    score_data = json.loads(score_path.read_text()) if score_path.exists() else {}
    view = str((feat_data.get("info") or {}).get("view", "unknown"))

    rep_results = []
    for rep in (seg_data.get("repetitions") or []):
        bilstm_raw = _extract_rep_matrix(seg_data, rep)
        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep)
        if bilstm_raw is None or stgcn_raw is None:
            continue

        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN)).unsqueeze(0)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)
        import numpy as np
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype("float32")
        ).unsqueeze(0)

        # Get heuristic score for this rep from scoring JSON
        rep_score_data = next(
            (r for r in (score_data.get("reps") or []) if r.get("rep_id") == rep.get("rep_id")), {}
        )
        hvec = torch.from_numpy(build_ohp_heuristic_vector(rep_score_data, view)).unsqueeze(0)
        view_vec = hvec[:, 11:16]

        with torch.no_grad():
            bilstm_out = bilstm(bilstm_t)
            stgcn_out = stgcn(stgcn_t, view_vec)
            final_score, residual = fusion(hvec, stgcn_out["embedding"], bilstm_out["embedding"])

        rep_results.append({
            "rep_id": rep.get("rep_id"),
            "neural_score": round(float(final_score.item()), 2),
            "elbow_error_prob": round(float(bilstm_out["elbow_error"].item()), 4),
            "knee_error_prob": round(float(bilstm_out["knee_error"].item()), 4) if include_knee and "knee_error" in bilstm_out else None,
            "neural_available": True,
        })

    output_path = workspace / exercise / "neural_scores" / tier / video_id / f"{video_id}_neural.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"exercise": exercise, "video_id": video_id, "reps": rep_results}, indent=2))
    print(json.dumps({"status": "ok", "reps_scored": len(rep_results), "output": str(output_path)}))
