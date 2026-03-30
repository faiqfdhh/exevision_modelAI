"""
Neural Fusion Inference Script - Stage 9.

Reads Stage 2.5-8 pipeline outputs from workspace and runs trained neural models
for per-video per-rep quality predictions. Outputs to squat/neural_analysis/.

Usage:
  python 9_neural_fusion_inference.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure module directories are on sys.path when this stage runs from an isolated
# pipeline workspace (cwd != project root).
_NEURAL_DIR = Path(__file__).resolve().parents[1] / "neural"
_TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(_NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(_NEURAL_DIR))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from nn_models import BiLSTMScorer, HeuristicGuidedFusion, STGCNScorer, apply_safety_clamps, build_heuristic_vector
from nn_utils import _extract_stgcn_rep, build_adjacency_matrix, pad_or_truncate


# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# parents: stages/ -> exevision/ -> core/ -> exevision_modelAI/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _safe_torch_load(path: Path, device: torch.device):
    """Load checkpoints with forward-compatible torch.load options."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_state_dict_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip common wrappers/prefixes from checkpoint keys."""
    if not state:
        return state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    out = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def _load_stgcn_with_compat(stgcn: STGCNScorer, ckpt_path: Path, device: torch.device) -> None:
    """Load ST-GCN checkpoint and adapt known head-shape drift (256->261 input)."""
    state = _safe_torch_load(ckpt_path, device)
    state = _normalize_state_dict_keys(state)

    key = "spatial_head.0.weight"
    if key in state:
        src_w = state[key]
        tgt_w = stgcn.state_dict()[key]
        if src_w.ndim == 2 and tgt_w.ndim == 2 and src_w.shape[0] == tgt_w.shape[0] and src_w.shape[1] != tgt_w.shape[1]:
            if src_w.shape[1] == 256 and tgt_w.shape[1] == 261:
                pad = torch.zeros((src_w.shape[0], 5), dtype=src_w.dtype, device=src_w.device)
                state[key] = torch.cat([src_w, pad], dim=1)
                logger.info("Adapted ST-GCN spatial_head input weights from 256 to 261 (+5 zero-initialized view channels)")

    try:
        stgcn.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        model_state = stgcn.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape") and tuple(v.shape) == tuple(model_state[k].shape)
        }
        missing, unexpected = stgcn.load_state_dict(filtered, strict=False)
        logger.warning(
            "ST-GCN strict load failed (%s). Loaded compatible subset instead. Missing=%d Unexpected=%d",
            exc,
            len(missing),
            len(unexpected),
        )


def _load_model_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load generic model state dict with normalized keys."""
    state = _safe_torch_load(ckpt_path, device)
    state = _normalize_state_dict_keys(state)
    model.load_state_dict(state, strict=True)


def get_device(force_cpu: bool = False) -> torch.device:
    """Get torch device."""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_bilstm_rep(seg_data: dict, rep: dict) -> Optional[np.ndarray]:
    """Extract BiLSTM sequence (128, 4) for one rep from segmentation data."""
    signals = seg_data.get("signals", {})
    arrays = []
    for key in ["normalized_hip_displacement", "window_velocity", "knee_angles", "landmark_confidence"]:
        values = signals.get(key, [])
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((1,), dtype=np.float32)
        arrays.append(arr)

    if not arrays or any(arr.size == 0 for arr in arrays):
        return None

    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", -1))

    def safe_slice(arr, s, e):
        if arr.size == 0:
            return arr
        s = max(0, int(s))
        e = min(int(e), len(arr) - 1)
        if e < s:
            return arr[:0]
        return arr[s : e + 1]

    sliced = [safe_slice(arr, start, end) for arr in arrays]
    if any(arr.size == 0 for arr in sliced):
        return None

    min_len = min(len(arr) for arr in sliced)
    if min_len <= 0:
        return None

    stacked = np.stack([arr[:min_len] for arr in sliced], axis=-1).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def infer_rep(
    rep_data: dict,
    feature_data: dict,
    seg_data: dict,
    view: str,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """
    Infer one rep.

    Returns dict with neural_score, residual, and sub-metrics, or None on failure.
    """
    try:
        # Extract sequences
        bilstm_seq = extract_bilstm_rep(seg_data, rep_data)
        if bilstm_seq is None:
            return None
        bilstm_seq = pad_or_truncate(bilstm_seq).astype(np.float32)

        stgcn_seq = _extract_stgcn_rep(seg_data, feature_data, rep_data)
        if stgcn_seq is None:
            return None
        stgcn_seq = pad_or_truncate(stgcn_seq).astype(np.float32)

        # Build heuristic vector
        heuristic_vec = build_heuristic_vector(rep_data, view)

        # Convert to batches of 1
        bilstm_t = torch.from_numpy(bilstm_seq).unsqueeze(0).float().to(device)
        stgcn_t = torch.from_numpy(stgcn_seq).permute(2, 0, 1).unsqueeze(0).float().to(device)
        heur_t = torch.from_numpy(heuristic_vec).unsqueeze(0).float().to(device)

        # Inference
        with torch.no_grad():
            bo = models["bilstm"](bilstm_t)
            so = models["stgcn"](stgcn_t, heur_t[:, 10:15])
            pred, residual = models["fusion"](heur_t, so["embedding"], bo["embedding"])

            # Extract scalars
            neural_score_pre = float(pred[0].cpu().numpy())
            residual_val = float(residual[0].cpu().numpy())
            smoothness = max(0.0, min(100.0, float(bo["smoothness"][0].cpu().numpy()) * 100.0))
            control = max(0.0, min(100.0, float(bo["control"][0].cpu().numpy()) * 100.0))
            # ST-GCN spatial head outputs are unbounded; clamp to [0, 100] after scaling
            depth = max(0.0, min(100.0, float(so["depth"][0].cpu().numpy()) * 100.0))
            forward_lean = max(0.0, min(100.0, float(so["forward_lean"][0].cpu().numpy()) * 100.0))
            knee_tracking = max(0.0, min(100.0, float(so["knee_tracking"][0].cpu().numpy()) * 100.0))

        # Apply safety clamps
        flags = rep_data.get("heuristic_flags", {}) or {}
        flag_severities = rep_data.get("flag_severities", {}) or {}
        neural_score_clamped = apply_safety_clamps(neural_score_pre, flags, flag_severities)

        # Determine which clamps were applied
        clamp_reasons = []
        if bool(flags.get("knee_valgus", False)) and int(flag_severities.get("knee_valgus", 0)) >= 2:
            if neural_score_clamped < neural_score_pre:
                clamp_reasons.append("knee_valgus_severity>=2")
        if bool(flags.get("forward_lean", False)) and int(flag_severities.get("forward_lean", 0)) >= 2:
            if neural_score_clamped < neural_score_pre:
                clamp_reasons.append("forward_lean_severity>=2")
        if bool(flags.get("insufficient_squat_depth", False)) and int(
            flag_severities.get("insufficient_squat_depth", 0)
        ) >= 3:
            if neural_score_clamped < neural_score_pre:
                clamp_reasons.append("insufficient_squat_depth_severity>=3")

        return {
            "neural_score": float(neural_score_clamped),
            "neural_score_pre_clamp": float(neural_score_pre),
            "residual": float(residual_val),
            "smoothness": float(smoothness),
            "control": float(control),
            "depth": float(depth),
            "forward_lean": float(forward_lean),
            "knee_tracking": float(knee_tracking),
            "safety_clamps_applied": clamp_reasons,
        }
    except Exception as e:
        logger.warning(f"Inference failed for rep: {str(e)}")
        return None


def process_video(
    video_id: str,
    workspace_root: Path,
    quality_tier: str,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """
    Process one video: read segmented reps, infer all, return output dict.

    Returns dict with "video_id", "quality", "reps", or None on critical failure.
    """
    # Discover feature and segmentation JSONs
    feature_json_path = workspace_root / "squat" / "extracted_features_clean" / quality_tier / f"{video_id}.json"
    seg_json_path = workspace_root / "squat" / "segmented_reps" / quality_tier / f"{video_id}_segmented.json"

    if not feature_json_path.exists():
        logger.warning(f"Missing feature JSON for {video_id}: {feature_json_path}")
        return None
    if not seg_json_path.exists():
        seg_root = workspace_root / "squat" / "segmented_reps"
        fallback = sorted(seg_root.rglob(f"{video_id}_segmented.json")) if seg_root.exists() else []
        if fallback:
            seg_json_path = fallback[0]
            logger.warning(
                "Segmented JSON not found in expected tier '%s'; using fallback: %s",
                quality_tier,
                seg_json_path,
            )
        else:
            logger.warning(f"Missing segmented JSON for {video_id}: {seg_json_path}")
            return None

    try:
        with feature_json_path.open("r", encoding="utf-8") as f:
            feature_data = json.load(f)
        with seg_json_path.open("r", encoding="utf-8") as f:
            seg_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSONs for {video_id}: {str(e)}")
        return None

    # Get view from feature data or default
    view = feature_data.get("info", {}).get("view", "unknown")

    # Load Stage 8 AQA JSON to get heuristic scores per rep.
    # The segmented reps JSON (Stage 5) does not contain heuristic_score — that is
    # written to a separate file by Stage 8.  Without this merge, build_heuristic_vector
    # defaults heuristic_score to 0, making the fusion anchor 0 instead of the real
    # heuristic value, which collapses the final score entirely.
    # Search the ENTIRE aqa_analysis_simple/ tree, not just {quality_tier}/.
    # Stage 8 writes to aqa_analysis_simple/{source_quality}/{score_tier}/ where
    # source_quality can differ from quality_tier (e.g., "unknown" vs "raw_unfiltered").
    aqa_base = workspace_root / "squat" / "aqa_analysis_simple"
    aqa_by_rep_id: dict = {}
    if aqa_base.exists():
        aqa_matches = sorted(aqa_base.rglob(f"{video_id}_aqa_simple.json"))
        if aqa_matches:
            try:
                with aqa_matches[0].open("r", encoding="utf-8") as f:
                    aqa_data = json.load(f)
                for aqa_rep in aqa_data.get("repetitions", []):
                    rid = aqa_rep.get("rep_id")
                    if rid is not None:
                        aqa_by_rep_id[rid] = aqa_rep
                logger.info(
                    "Loaded AQA heuristic data for %d reps from %s",
                    len(aqa_by_rep_id),
                    aqa_matches[0].relative_to(aqa_base),
                )
            except Exception as e:
                logger.warning(f"Could not load AQA JSON for {video_id}: {e}")
    if not aqa_by_rep_id:
        logger.warning(
            f"No AQA JSON found for {video_id} under {aqa_base} (searched recursively). "
            "heuristic_score will default to 0 — neural scores will be unreliable."
        )

    # Process each rep
    reps_output = []
    repetitions = seg_data.get("repetitions", []) or []
    for rep_idx, rep in enumerate(repetitions):
        if rep is None:
            continue

        # Merge heuristic output from Stage 8 into the seg rep dict so that
        # build_heuristic_vector receives the correct anchor and metric scores.
        aqa_rep = aqa_by_rep_id.get(rep.get("rep_id"))
        if aqa_rep:
            rep = {
                **rep,
                "heuristic_score": aqa_rep.get("score", {}).get("overall_score", 0.0),
                "heuristic_metric_scores": aqa_rep.get("score", {}).get("metric_scores", {}),
                # flags/flag_severities are not available at inference time (no annotation);
                # defaulting to {} means safety clamps will not trigger unless the heuristic
                # stage itself produces flag data.
                "heuristic_flags": rep.get("heuristic_flags", {}),
                "flag_severities": rep.get("flag_severities", {}),
            }
        else:
            logger.debug(f"No AQA rep data for rep_id={rep.get('rep_id')} in {video_id}")

        neural_output = infer_rep(rep, feature_data, seg_data, view, models, device)
        if neural_output is None:
            logger.debug(f"Skipped rep {rep_idx + 1} in {video_id}")
            continue

        rep_result = {
            "rep_id": int(rep.get("rep_id", rep_idx + 1)),
            "start_frame": int(rep.get("start_frame", 0)),
            "end_frame": int(rep.get("end_frame", 0)),
            **neural_output,
        }
        reps_output.append(rep_result)

    if not reps_output:
        logger.warning(f"No successful reps for {video_id}")
        return None

    return {
        "video_id": video_id,
        "quality": quality_tier,
        "view": view,
        "reps": reps_output,
    }


def discover_videos(workspace_root: Path, quality_tier_filter: Optional[str] = None) -> List[tuple[str, str]]:
    """
    Discover all processed videos across quality tiers.
    
    If quality_tier_filter is set (e.g. 'raw_unfiltered'), only scan that subdirectory.
    Otherwise scan all subdirectories.

    Returns list of (video_id, quality_tier) tuples.
    """
    videos = []
    features_dir = workspace_root / "squat" / "extracted_features_clean"
    if not features_dir.exists():
        return videos

    if quality_tier_filter:
        # Only scan the specified quality tier
        quality_dir = features_dir / quality_tier_filter
        if quality_dir.is_dir():
            for feature_json in quality_dir.glob("*.json"):
                video_id = feature_json.stem
                videos.append((video_id, quality_tier_filter))
    else:
        # Scan all quality tiers
        for quality_dir in features_dir.iterdir():
            if not quality_dir.is_dir():
                continue
            quality_tier = quality_dir.name
            for feature_json in quality_dir.glob("*.json"):
                video_id = feature_json.stem
                videos.append((video_id, quality_tier))

    return videos


def save_outputs(
    video_results: Dict[str, Dict[str, Any]],
    workspace_root: Path,
) -> None:
    """Save per-video neural outputs and aggregate scoreboard."""
    neural_dir = workspace_root / "squat" / "neural_analysis"
    neural_dir.mkdir(parents=True, exist_ok=True)

    scoreboard = {
        "total_videos": len(video_results),
        "videos": {},
    }

    for video_id, v_result in video_results.items():
        if v_result is None:
            continue

        quality = v_result["quality"]
        quality_dir = neural_dir / quality
        quality_dir.mkdir(parents=True, exist_ok=True)

        output_json = quality_dir / f"{video_id}_neural.json"
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(v_result, f, indent=2)

        # Aggregate rep stats for scoreboard
        if v_result.get("reps"):
            scores = [r["neural_score"] for r in v_result["reps"]]
            scoreboard["videos"][video_id] = {
                "quality": quality,
                "rep_count": len(v_result["reps"]),
                "mean_neural_score": float(np.mean(scores)),
                "min_neural_score": float(np.min(scores)),
                "max_neural_score": float(np.max(scores)),
            }

    scoreboard_json = neural_dir / "neural_scoreboard.json"
    with scoreboard_json.open("w", encoding="utf-8") as f:
        json.dump(scoreboard, f, indent=2)

    logger.info(f"Saved {len(video_results)} neural analysis results")
    logger.info(f"Scoreboard: {scoreboard_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural Fusion Inference - Stage 9")
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=".",
        help="Path to pipeline workspace (default: current dir)",
    )
    parser.add_argument(
        "--bilstm-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "models" / "bilstm_finetuned.pt"),
        help="Path to BiLSTM checkpoint",
    )
    parser.add_argument(
        "--stgcn-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "models" / "stgcn_finetuned.pt"),
        help="Path to ST-GCN checkpoint",
    )
    parser.add_argument(
        "--fusion-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "models" / "fusion_layer.pt"),
        help="Path to fusion checkpoint",
    )
    parser.add_argument("--video-id", type=str, default="", help="Process only one video id")
    parser.add_argument(
        "--quality-tier",
        type=str,
        default=None,
        help="Restrict neural inference to one quality tier (e.g. raw_unfiltered). "
             "When omitted, all tiers are processed.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    device = get_device(force_cpu=args.cpu)

    logger.info("=== Neural Fusion Inference ===")
    logger.info(f"Workspace: {workspace_root}")
    logger.info(f"Device: {device}")

    # Check workspace structure
    if not (workspace_root / "squat" / "extracted_features_clean").exists():
        logger.error("Missing extracted_features_clean in workspace")
        return 1

    # Discover videos
    videos = discover_videos(workspace_root, quality_tier_filter=args.quality_tier)
    if args.video_id:
        videos = [(video_id, q) for video_id, q in videos if video_id == args.video_id]
        logger.info(f"Video filter enabled: {args.video_id}")
    if args.quality_tier:
        logger.info(f"Quality tier filter enabled: {args.quality_tier}")
    logger.info(f"Discovered {len(videos)} videos across quality tiers")
    if not videos:
        logger.warning("No videos found in extracted_features_clean. This usually means Stage 2.5 (extract_selected_features) failed or was not run.")
        return 1

    # Load models
    logger.info("Loading models...")
    try:
        bilstm_ckpt = Path(args.bilstm_ckpt).resolve()
        stgcn_ckpt = Path(args.stgcn_ckpt).resolve()
        fusion_ckpt = Path(args.fusion_ckpt).resolve()

        if not bilstm_ckpt.exists():
            logger.error(f"BiLSTM checkpoint not found: {bilstm_ckpt}")
            return 1
        if not stgcn_ckpt.exists():
            logger.error(f"ST-GCN checkpoint not found: {stgcn_ckpt}")
            return 1
        if not fusion_ckpt.exists():
            logger.error(f"Fusion checkpoint not found: {fusion_ckpt}")
            return 1

        adjacency = build_adjacency_matrix()
        bilstm = BiLSTMScorer().to(device)
        stgcn = STGCNScorer(adjacency).to(device)
        fusion = HeuristicGuidedFusion().to(device)

        _load_model_state(bilstm, bilstm_ckpt, device)
        _load_stgcn_with_compat(stgcn, stgcn_ckpt, device)
        _load_model_state(fusion, fusion_ckpt, device)

        bilstm.eval()
        stgcn.eval()
        fusion.eval()

        models = {"bilstm": bilstm, "stgcn": stgcn, "fusion": fusion}
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return 1

    # Process all videos
    logger.info("Starting inference...")
    video_results = {}
    successful = 0
    failed = 0

    for video_id, quality_tier in videos:
        try:
            result = process_video(video_id, workspace_root, quality_tier, models, device)
            if result is not None:
                video_results[video_id] = result
                successful += 1
                logger.info(f"✓ {video_id} ({quality_tier}): {len(result['reps'])} reps inferred")
            else:
                failed += 1
                logger.warning(f"✗ {video_id} ({quality_tier}): inference returned None")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {video_id} ({quality_tier}): {str(e)}")

    logger.info(f"Inference complete: {successful} successful, {failed} failed")

    # Save outputs
    if video_results:
        save_outputs(video_results, workspace_root)

    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
