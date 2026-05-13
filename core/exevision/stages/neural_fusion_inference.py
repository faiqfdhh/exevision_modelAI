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
_TRAINING_BASE = Path(__file__).resolve().parents[1] / "training"
if str(_NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(_NEURAL_DIR))
# Don't add _TRAINING_BASE to sys.path here — there is a training/ohp/ package
# that shadows neural/ohp/.  Exercise-specific subdirectories (e.g. training/squat/)
# are added in _load_models() where the exercise is known.

from nn_utils import _extract_rep_matrix, _extract_stgcn_rep, pad_or_truncate, FIXED_SEQ_LEN
from registry import get_exercise_handler, get_model_classes


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


def _infer_rep(
    rep_data: dict,
    feature_data: dict,
    seg_data: dict,
    view: str,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    handler: dict,
) -> Optional[Dict[str, Any]]:
    """Generic per-rep neural inference. Returns dict with neural_score + model outputs."""
    try:
        exercise = handler["exercise"]
        bilstm_seq = _extract_rep_matrix(seg_data, rep_data, exercise=exercise)
        if bilstm_seq is None:
            return None
        bilstm_seq = pad_or_truncate(bilstm_seq).astype(np.float32)

        stgcn_seq = _extract_stgcn_rep(seg_data, feature_data, rep_data, exercise=exercise)
        if stgcn_seq is None:
            return None
        stgcn_seq = pad_or_truncate(stgcn_seq).astype(np.float32)

        heuristic_vec = handler["heuristic_fn"](rep_data, view)
        bilstm_t = torch.from_numpy(bilstm_seq).unsqueeze(0).float().to(device)
        stgcn_t = torch.from_numpy(stgcn_seq).permute(2, 0, 1).unsqueeze(0).float().to(device)
        heur_t = torch.from_numpy(heuristic_vec).unsqueeze(0).float().to(device)

        start, end = handler["view_vec_slice"]
        view_vec = heur_t[:, start:end]

        with torch.no_grad():
            b_out = models["bilstm"](bilstm_t)
            s_out = models["stgcn"](stgcn_t, view_vec)
            pred, residual = models["fusion"](heur_t, s_out["embedding"], b_out["embedding"])

        result = {
            "neural_score": round(float(pred.item()), 2),
            "residual": round(float(residual.item()), 2),
            "neural_available": True,
        }

        for name, val in {**b_out, **s_out}.items():
            if name == "embedding":
                continue
            is_prob = any(p in name for p in ("prob", "error"))
            result[name] = round(float(val.item()), 4 if is_prob else 2)

        if handler.get("grip_ratio_side_exclude", False) and view in ("side", "unknown"):
            result["grip_ratio"] = None

        if handler.get("suppress_knee", False):
            for k in list(result.keys()):
                if "knee" in k.lower():
                    result.pop(k, None)

        post_process = handler.get("post_process")
        if post_process:
            result = post_process(result, rep_data, view, heuristic_vec)

        return result
    except Exception as e:
        logger.warning(f"Inference failed for rep: {str(e)}")
        return None


def _squat_post_process(
    result: dict, rep_data: dict, view: str, heuristic_vec: np.ndarray,
) -> dict:
    """Squat-specific safety clamps, depth reliability gating, sub-score ceiling."""
    from nn_models import _safe_score, apply_safety_clamps

    residual_val = result.get("residual", 0.0)
    heuristic_raw = float(heuristic_vec[0]) * 100.0
    depth = max(0.0, min(100.0, result.get("depth", 0.0)))

    hms = rep_data.get("heuristic_metric_scores") or {}
    heuristic_depth = _safe_score(hms.get("depth", 0.0))
    depth_unreliable = abs(depth - heuristic_depth) > 30.0
    residual_dampening = 0.6 if depth_unreliable else 1.0
    dampened_residual = residual_val * residual_dampening
    neural_score_pre = heuristic_raw + dampened_residual

    flags = rep_data.get("heuristic_flags") or {}
    flag_severities = rep_data.get("flag_severities") or {}
    neural_score_clamped = apply_safety_clamps(neural_score_pre, flags, flag_severities)

    clamp_reasons = []
    if depth_unreliable:
        clamp_reasons.append(
            f"st_gcn_depth_unreliable(heuristic={heuristic_depth:.0f},st_gcn={depth:.0f},dampening=0.6)"
        )
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

    all_subscores = [
        result.get("forward_lean", 0.0),
        result.get("knee_tracking", 0.0),
        result.get("smoothness", 0.0),
        result.get("control", 0.0),
    ]
    if not depth_unreliable:
        all_subscores.append(depth)
    subscore_worst = min(all_subscores) if all_subscores else 100.0
    subscore_ceiling = 100.0 if subscore_worst >= 100.0 else min(99.0, subscore_worst * 0.5 + 50.0)
    if neural_score_clamped > subscore_ceiling:
        neural_score_clamped = subscore_ceiling
        clamp_reasons.append(f"subscore_ceiling(worst={subscore_worst:.1f},ceiling={subscore_ceiling:.1f})")

    result.update({
        "neural_score": float(neural_score_clamped),
        "neural_score_pre_clamp": float(neural_score_pre),
        "residual_dampening": float(residual_dampening),
        "residual_dampened": float(dampened_residual),
        "safety_clamps_applied": clamp_reasons,
    })
    return result


def aggregate_scores(heuristic: float, neural: float) -> float:
    """Blend heuristic + neural into a single final score.

    Only if both are 100 can the final be 100.  Otherwise blend toward the
    lower score so a single inflated pipeline cannot overrule the other.
    """
    if heuristic >= 100.0 and neural >= 100.0:
        return 100.0
    lower = min(heuristic, neural)
    higher = max(heuristic, neural)
    return round(lower + (higher - lower) * 0.3, 1)


def _load_models(handler: dict, device: torch.device, bilstm_ckpt: Path, stgcn_ckpt: Path, fusion_ckpt: Path) -> Dict[str, torch.nn.Module]:
    """Load exercise-specific neural models from checkpoints."""
    model_classes = get_model_classes(handler["exercise"])
    adjacency = handler["adjacency_fn"]()
    A = torch.tensor(adjacency, dtype=torch.float32).to(device)

    bilstm = model_classes["bilstm"]().to(device)
    stgcn = model_classes["stgcn"](A).to(device)
    fusion = handler["fusion_builder"]().to(device)

    _load_model_state(bilstm, bilstm_ckpt, device)
    _load_stgcn_with_compat(stgcn, stgcn_ckpt, device)
    _load_model_state(fusion, fusion_ckpt, device)

    bilstm.eval()
    stgcn.eval()
    fusion.eval()

    return {"bilstm": bilstm, "stgcn": stgcn, "fusion": fusion}


def process_video(
    video_id: str,
    workspace_root: Path,
    quality_tier: str,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    handler: dict,
) -> Optional[Dict[str, Any]]:
    """
    Process one video: read segmented reps, infer all, return output dict.
    Returns dict with "video_id", "quality", "reps", or None on critical failure.
    """
    exercise = handler["exercise"]
    feature_json_path = workspace_root / exercise / "extracted_features_clean" / quality_tier / f"{video_id}.json"
    seg_json_path = workspace_root / exercise / "segmented_reps" / quality_tier / f"{video_id}_segmented.json"

    if not feature_json_path.exists():
        logger.warning(f"Missing feature JSON for {video_id}: {feature_json_path}")
        return None
    if not seg_json_path.exists():
        seg_root = workspace_root / exercise / "segmented_reps"
        fallback = sorted(seg_root.rglob(f"{video_id}_segmented.json")) if seg_root.exists() else []
        if fallback:
            seg_json_path = fallback[0]
            logger.warning("Segmented JSON not found in expected tier '%s'; using fallback: %s", quality_tier, seg_json_path)
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

    view = feature_data.get("info", {}).get("view", "unknown")

    aqa_base = workspace_root / exercise / "aqa_analysis_simple"
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
                logger.info("Loaded AQA heuristic data for %d reps from %s", len(aqa_by_rep_id), aqa_matches[0].relative_to(aqa_base))
            except Exception as e:
                logger.warning(f"Could not load AQA JSON for {video_id}: {e}")
    if not aqa_by_rep_id:
        logger.warning(f"No AQA JSON found for {video_id} under {aqa_base} (searched recursively). heuristic_score will default to 0.")

    reps_output = []
    repetitions = seg_data.get("repetitions", []) or []
    for rep_idx, rep in enumerate(repetitions):
        if rep is None:
            continue

        aqa_rep = aqa_by_rep_id.get(rep.get("rep_id"))
        if aqa_rep:
            rep = {
                **rep,
                "heuristic_score": aqa_rep.get("score", {}).get("overall_score", 0.0),
                "heuristic_metric_scores": aqa_rep.get("score", {}).get("metric_scores", {}),
                "heuristic_flags": rep.get("heuristic_flags", {}),
                "flag_severities": rep.get("flag_severities", {}),
            }
        else:
            logger.debug(f"No AQA rep data for rep_id={rep.get('rep_id')} in {video_id}")

        neural_output = _infer_rep(rep, feature_data, seg_data, view, models, device, handler)
        if neural_output is None:
            logger.debug(f"Skipped rep {rep_idx + 1} in {video_id}")
            continue

        rep_result = {
            "rep_id": int(rep.get("rep_id", rep_idx + 1)),
            "start_frame": int(rep.get("start_frame", 0)),
            "end_frame": int(rep.get("end_frame", 0)),
            **neural_output,
        }

        h_score = rep.get("heuristic_score", 0.0)
        n_score = neural_output.get("neural_score")
        if n_score is not None:
            rep_result["aggregated_score"] = aggregate_scores(float(h_score), float(n_score))
        reps_output.append(rep_result)

    if not reps_output:
        logger.warning(f"No successful reps for {video_id}")
        return None

    return {"video_id": video_id, "quality": quality_tier, "view": view, "reps": reps_output}


def discover_videos(workspace_root: Path, quality_tier_filter: Optional[str] = None, exercise: str = "squat") -> List[tuple[str, str]]:
    """
    Discover all processed videos across quality tiers.
    
    If quality_tier_filter is set (e.g. 'raw_unfiltered'), only scan that subdirectory.
    Otherwise scan all subdirectories.

    Returns list of (video_id, quality_tier) tuples.
    """
    videos = []
    features_dir = workspace_root / exercise / "extracted_features_clean"
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
    exercise: str = "squat",
) -> None:
    """Save per-video neural outputs and aggregate scoreboard."""
    neural_dir = workspace_root / exercise / "neural_analysis"
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
        default=None,
        help="Path to BiLSTM checkpoint (default: exercise-specific)",
    )
    parser.add_argument(
        "--stgcn-ckpt",
        type=str,
        default=None,
        help="Path to ST-GCN checkpoint (default: exercise-specific)",
    )
    parser.add_argument(
        "--fusion-ckpt",
        type=str,
        default=None,
        help="Path to fusion checkpoint (default: exercise-specific)",
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
    parser.add_argument("--exercise", default="squat", help="Exercise type (default: squat)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exercise = args.exercise

    # Add exercise-specific training dir to sys.path so model files can import
    # their pretrain_bilstm/pretrain_stgcn modules (these live in exercise subdirs).
    exercise_train_dir = _TRAINING_BASE / exercise
    if exercise_train_dir.exists() and str(exercise_train_dir) not in sys.path:
        sys.path.insert(0, str(exercise_train_dir))

    try:
        handler = get_exercise_handler(exercise)
    except KeyError:
        logger.error(f"No neural inference config for exercise '{exercise}'")
        return 1

    # Resolve checkpoint paths — explicit CLI args override handler defaults
    ckpt_dir = PROJECT_ROOT / handler["ckpt_dir"]
    bilstm_ckpt = Path(args.bilstm_ckpt or str(ckpt_dir / handler["bilstm_ckpt_name"])).resolve()
    stgcn_ckpt = Path(args.stgcn_ckpt or str(ckpt_dir / handler["stgcn_ckpt_name"])).resolve()
    fusion_ckpt = Path(args.fusion_ckpt or str(ckpt_dir / handler["fusion_ckpt_name"])).resolve()

    # Attach squat-specific post-processing
    if exercise == "squat":
        handler["post_process"] = _squat_post_process

    workspace_root = Path(args.workspace_root).resolve()
    device = get_device(force_cpu=args.cpu)

    logger.info("=== Neural Fusion Inference ===")
    logger.info(f"Workspace: {workspace_root}")
    logger.info(f"Exercise: {exercise}")
    logger.info(f"Device: {device}")

    if not (workspace_root / exercise / "extracted_features_clean").exists():
        logger.error(f"Missing extracted_features_clean in {exercise} workspace")
        return 1

    videos = discover_videos(workspace_root, quality_tier_filter=args.quality_tier, exercise=exercise)
    if args.video_id:
        videos = [(video_id, q) for video_id, q in videos if video_id == args.video_id]
        logger.info(f"Video filter enabled: {args.video_id}")
    if args.quality_tier:
        logger.info(f"Quality tier filter enabled: {args.quality_tier}")
    logger.info(f"Discovered {len(videos)} videos across quality tiers")
    if not videos:
        logger.warning("No videos found in extracted_features_clean. This usually means Stage 2.5 (extract_selected_features) failed or was not run.")
        return 1

    logger.info("Loading models...")
    try:
        if not bilstm_ckpt.exists():
            logger.error(f"BiLSTM checkpoint not found: {bilstm_ckpt}")
            return 1
        if not stgcn_ckpt.exists():
            logger.error(f"ST-GCN checkpoint not found: {stgcn_ckpt}")
            return 1
        if not fusion_ckpt.exists():
            logger.error(f"Fusion checkpoint not found: {fusion_ckpt}")
            return 1
        models = _load_models(handler, device, bilstm_ckpt, stgcn_ckpt, fusion_ckpt)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return 1

    logger.info("Starting inference...")
    video_results = {}
    successful = 0
    failed = 0

    for video_id, quality_tier in videos:
        try:
            result = process_video(video_id, workspace_root, quality_tier, models, device, handler)
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

    if video_results:
        save_outputs(video_results, workspace_root, exercise=exercise)

    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
