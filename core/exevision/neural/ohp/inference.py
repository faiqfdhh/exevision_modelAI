from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_NEURAL    = Path(__file__).resolve().parents[1]
_OHP       = Path(__file__).resolve().parent
_REPO      = Path(__file__).resolve().parents[4]   # ohp/ → neural/ → exevision/ → core/ → repo root
_TRAIN_OHP = _REPO / "core" / "exevision" / "training" / "ohp"
for _p in [str(_NEURAL), str(_OHP), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp, _extract_stgcn_rep, _extract_rep_matrix, pad_or_truncate, FIXED_SEQ_LEN
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from ohp.heuristic_vec import build_ohp_heuristic_vector


def _load_checkpoints(model_dir: Path):
    """Load standing OHP Phase 2 checkpoints (BiLSTM, ST-GCN, fusion)."""
    bilstm = model_dir / "bilstm_ohp_phase2.pt"
    stgcn = model_dir / "stgcn_ohp_phase2.pt"
    fusion = model_dir / "fusion_ohp_phase2.pt"
    return (
        bilstm if bilstm.exists() else None,
        stgcn if stgcn.exists() else None,
        fusion if fusion.exists() else None,
    )


def run_ohp_inference(args) -> None:
    """Entry point called from neural_fusion_inference.py for OHP exercises.

    Standing overhead_press only — seated_overhead_press uses heuristic-only
    inference (no Phase 2 model, leg landmarks are zeroed).
    """
    workspace = Path(args.workspace_root)
    exercise = args.exercise
    video_id = args.video_id
    model_dir = Path(args.model_dir) if hasattr(args, "model_dir") else _REPO / "models"

    if exercise == "seated_overhead_press":
        print(json.dumps({
            "status": "skipped",
            "reason": "seated_overhead_press uses heuristic-only inference (no Phase 2 model)",
            "neural_available": False,
        }))
        return

    device = torch.device("cpu")
    A = torch.tensor(build_adjacency_matrix_ohp())

    bilstm = OHPBiLSTMScorer().to(device)
    stgcn = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)

    bilstm_path, stgcn_path, fusion_path = _load_checkpoints(model_dir)
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
        bilstm_raw = _extract_rep_matrix(seg_data, rep, exercise=exercise)
        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep, exercise=exercise)
        if bilstm_raw is None or stgcn_raw is None:
            continue

        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN)).unsqueeze(0)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)
        import numpy as np
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype("float32")
        ).unsqueeze(0)

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
            "knee_error_prob": round(float(bilstm_out["knee_error"].item()), 4),
            "neural_available": True,
        })

    output_path = workspace / exercise / "neural_analysis" / tier / video_id / f"{video_id}_neural.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"exercise": exercise, "video_id": video_id, "reps": rep_results}, indent=2))
    print(json.dumps({"status": "ok", "reps_scored": len(rep_results), "output": str(output_path)}))

def run_ohp_phase3_ensemble(args) -> None:
    """Run Phase 3 5-seed ensemble inference. Writes to neural_analysis/."""
    workspace = Path(args.workspace_root)
    exercise = args.exercise
    video_id = args.video_id
    model_dir = Path(getattr(args, "model_dir", None) or (_REPO / "models"))
    tier = getattr(args, "quality", "raw_unfiltered")

    suppress_knee = (exercise == "seated_overhead_press")

    seed_paths = sorted(model_dir.glob("bilstm_ohp_phase3_seed*.pt"))
    if not seed_paths:
        print(json.dumps({"error": "No Phase 3 seed checkpoints found", "neural_available": False}))
        return

    device = torch.device("cpu")
    A = torch.tensor(build_adjacency_matrix_ohp())
    
    models = []
    for seed_path in seed_paths:
        seed = seed_path.stem.replace("bilstm_ohp_phase3_seed", "")
        bilstm = OHPBiLSTMScorer().to(device)
        stgcn = OHPSTGCNScorer(A).to(device)
        fusion = build_ohp_fusion().to(device)
        try:
            bilstm.load_state_dict(torch.load(seed_path, map_location="cpu"))
            stgcn.load_state_dict(torch.load(model_dir / f"stgcn_ohp_phase3_seed{seed}.pt", map_location="cpu"))
            fusion.load_state_dict(torch.load(model_dir / f"fusion_ohp_phase3_seed{seed}.pt", map_location="cpu"))
        except Exception:
            pass
        bilstm.eval(); stgcn.eval(); fusion.eval()
        models.append((bilstm, stgcn, fusion))
        
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
    import numpy as np
    from tta import apply_tta

    for rep in (seg_data.get("repetitions") or []):
        bilstm_raw = _extract_rep_matrix(seg_data, rep, exercise=exercise)
        stgcn_raw = _extract_stgcn_rep(seg_data, feat_data, rep, exercise=exercise)
        if bilstm_raw is None or stgcn_raw is None:
            continue

        bilstm_t = torch.from_numpy(pad_or_truncate(bilstm_raw, FIXED_SEQ_LEN)).unsqueeze(0)
        stgcn_padded = pad_or_truncate(stgcn_raw, FIXED_SEQ_LEN)
        stgcn_t = torch.from_numpy(
            np.transpose(stgcn_padded, (2, 0, 1)).astype("float32")
        ).unsqueeze(0)

        rep_score_data = next(
            (r for r in (score_data.get("reps") or []) if r.get("rep_id") == rep.get("rep_id")), {}
        )
        hvec = torch.from_numpy(build_ohp_heuristic_vector(rep_score_data, view)).unsqueeze(0)
        view_vec = hvec[:, 11:16]

        preds = {"quality": [], "knee_error": [], "lockout": [], "smoothness": [], "control": [], "elbow_flare": [], "grip_ratio": [], "rom_top": [], "rom_bottom": []}
        
        for bilstm, stgcn, fusion in models:
            with torch.no_grad():
                variants = apply_tta(bilstm_t, stgcn_t)
                for v_b, v_s in variants:
                    b_out = bilstm(v_b)
                    s_out = stgcn(v_s, view_vec)
                    final_score, _ = fusion(hvec, s_out["embedding"], b_out["embedding"])
                    
                    preds["quality"].append(float(final_score.item()))
                    preds["knee_error"].append(float(b_out["knee_error"].item()))
                    preds["smoothness"].append(float(b_out["smoothness"].item()))
                    preds["control"].append(float(b_out["control"].item()))
                    preds["lockout"].append(float(s_out["lockout"].item()))
                    preds["elbow_flare"].append(float(s_out["elbow_flare"].item()))
                    preds["grip_ratio"].append(float(s_out["grip_ratio"].item()))
                    preds["rom_top"].append(float(s_out["rom_top"].item()))
                    preds["rom_bottom"].append(float(s_out["rom_bottom"].item()))
                    
        avg = {k: sum(v)/len(v) for k, v in preds.items()}
        std = float(np.std(preds["quality"]))

        entry = {
            "rep_id":          rep.get("rep_id"),
            "neural_available": True,
            "neural_score":    round(avg["quality"], 2),
            "lockout_prob":    round(avg["lockout"], 4),
            "smoothness":      round(avg["smoothness"], 2),
            "control":         round(avg["control"], 2),
            "elbow_flare":     round(avg["elbow_flare"], 2),
            "grip_ratio":      None if view in ("side", "unknown") else round(avg["grip_ratio"], 2),
            "rom_top":         round(avg["rom_top"], 2),
            "rom_bottom":      round(avg["rom_bottom"], 2),
            "ensemble_std":    round(std, 2),
        }
        if not suppress_knee:
            entry["knee_error_prob"] = round(avg["knee_error"], 4)
        rep_results.append(entry)

    output_path = workspace / exercise / "neural_analysis" / tier / video_id / f"{video_id}_neural.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"exercise": exercise, "video_id": video_id, "reps": rep_results}, indent=2))
    print(json.dumps({"status": "ok", "reps_scored": len(rep_results), "output": str(output_path)}))

