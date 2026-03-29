"""
ExeVision AI — Pipeline Runner

Mirrors the stage execution logic from apps/desktop-ui/app.py without Tkinter.
Runs the five pipeline stages as subprocesses inside an isolated workspace.
Does NOT modify any scoring, segmentation, or feature-extraction logic.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from core.exevision.feedback.engine import FeedbackEngine

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
# apps/api/ → apps/ → exevision_modelAI/
_API_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = _API_DIR.parents[1]
STAGES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "stages"
RUNS_ROOT = WORKSPACE_ROOT / "pipeline_ui_runs"
SHARED_MODEL_PATH = WORKSPACE_ROOT / "models" / "pose_landmarker_heavy.task"
SHARED_FACE_MODEL_PATH = WORKSPACE_ROOT / "models" / "blaze_face_short_range.tflite"
BILSTM_CKPT = WORKSPACE_ROOT / "models" / "bilstm_finetuned.pt"
STGCN_CKPT = WORKSPACE_ROOT / "models" / "stgcn_finetuned.pt"
FUSION_CKPT = WORKSPACE_ROOT / "models" / "fusion_layer.pt"
FEEDBACK_EXERCISE_CONFIG = WORKSPACE_ROOT / "core" / "exevision" / "config" / "exercises" / "squat.json"
FEEDBACK_TEMPLATES_CONFIG = WORKSPACE_ROOT / "core" / "exevision" / "config" / "templates" / "feedback_templates.json"


# ── Stage definitions ──────────────────────────────────────────────────────────
@dataclass
class StageSpec:
    key: str
    script: Path
    output_dirs: tuple[str, ...] = field(default_factory=tuple)


STAGE_SPECS: dict[str, StageSpec] = {
    "extract_selected_features": StageSpec(
        key="extract_selected_features",
        script=STAGES_DIR / "extract_selected_features.py",
        output_dirs=("squat/extracted_features_clean", "squat/visualized_poses_clean", "squat/analysis_reports"),
    ),
    "classify_views": StageSpec(
        key="classify_views",
        script=STAGES_DIR / "classify_views.py",
        output_dirs=("squat/extracted_features_clean",),
    ),
    "temporal_segmentation": StageSpec(
        key="temporal_segmentation",
        script=STAGES_DIR / "temporal_segmentation.py",
        output_dirs=("squat/segmented_reps", "squat/visualized_segmentation"),
    ),
    "scoring": StageSpec(
        key="scoring",
        script=STAGES_DIR / "scoring.py",
        output_dirs=("squat/aqa_analysis_simple",),
    ),
    "neural_fusion": StageSpec(
        key="neural_fusion",
        script=STAGES_DIR / "neural_fusion_inference.py",
        output_dirs=("squat/neural_analysis",),
    ),
}

DEFAULT_STAGES = [
    "extract_selected_features",
    "classify_views",
    "temporal_segmentation",
    "scoring",
    "neural_fusion",
]


# ── Workspace helpers ──────────────────────────────────────────────────────────
def _prepare_workspace(workspace_root: Path, video_path: Path) -> None:
    """Create workspace directory tree and copy the input video into it."""
    videos_dir = workspace_root / "squat" / "dataset_videos_all"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / "squat" / "aqa_analysis_simple").mkdir(parents=True, exist_ok=True)
    dest = videos_dir / video_path.name
    shutil.copy2(video_path, dest)


def _build_stage_cmd(key: str, script: Path, video_id: str, mode: str) -> list[str]:
    """Build the subprocess command for a stage, mirroring app.py arg construction.

    API runs skip all visualization outputs (--no-viz, --no-report) because the
    annotated MP4s and PNG plots are only consumed by the desktop UI — they account
    for ~60% of per-run disk usage and are never served by the API.
    """
    base = [sys.executable, str(script)]
    if key == "extract_selected_features":
        return base + [mode, "--video-id", video_id, "--no-viz", "--no-report"]
    elif key == "temporal_segmentation":
        return base + ["--video-id", video_id, "--no-viz"]
    elif key == "classify_views":
        return base + ["--video-id", video_id]
    elif key == "scoring":
        return base + [video_id]
    elif key == "neural_fusion":
        return base + [
            "--video-id", video_id,
            "--bilstm-ckpt", str(BILSTM_CKPT),
            "--stgcn-ckpt", str(STGCN_CKPT),
            "--fusion-ckpt", str(FUSION_CKPT),
        ]
    return base


def _run_stage(
    key: str,
    script: Path,
    video_id: str,
    workspace_root: Path,
    logs_root: Path,
    mode: str,
) -> str:
    """Run one pipeline stage; returns captured stdout+stderr."""
    cmd = _build_stage_cmd(key, script, video_id, mode)
    env = os.environ.copy()
    env["EXEVISION_MODEL_PATH"] = str(SHARED_MODEL_PATH)
    env["EXEVISION_FACE_MODEL_PATH"] = str(SHARED_FACE_MODEL_PATH)

    log_file = logs_root / f"{key}.log"
    result = subprocess.run(
        cmd,
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = result.stdout + result.stderr
    log_file.write_text(combined, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage '{key}' failed (exit {result.returncode}).\n"
            f"Log: {log_file}\n"
            f"{combined[-2000:]}"
        )
    return combined


def _validate_stage_output(key: str, workspace_root: Path, video_id: str) -> None:
    """Ensure each stage produced the expected artifact for the requested video."""
    expected_patterns = {
        "extract_selected_features": f"squat/extracted_features_clean/**/{video_id}.json",
        "classify_views": f"squat/extracted_features_clean/**/{video_id}.json",
        "temporal_segmentation": f"squat/segmented_reps/**/{video_id}_segmented.json",
        "scoring": f"squat/aqa_analysis_simple/**/{video_id}_aqa_simple.json",
        "neural_fusion": f"squat/neural_analysis/**/{video_id}_neural.json",
    }

    pattern = expected_patterns.get(key)
    if not pattern:
        return

    matches = sorted(workspace_root.glob(pattern))
    if not matches:
        raise RuntimeError(
            f"Stage '{key}' completed but expected output was not found for video '{video_id}'. "
            f"Missing pattern: {workspace_root / pattern}"
        )


# ── Workspace cleanup helpers ──────────────────────────────────────────────────
def _delete_input_video(workspace_root: Path, filename: str) -> None:
    """Remove the input video copy from the workspace after extraction."""
    video_copy = workspace_root / "squat" / "dataset_videos_all" / filename
    if video_copy.exists():
        video_copy.unlink()


def _cleanup_workspace(workspace_root: Path) -> None:
    """
    Remove heavy intermediate artifacts from the workspace after results are collected.

    What is removed:
    - squat/visualized_poses_clean/   (annotated pose MP4s — only useful in desktop UI)
    - squat/visualized_segmentation/  (phase overlay MP4s — only useful in desktop UI)
    - squat/analysis_reports/         (PNG plots — only useful in desktop UI)
    - squat/dataset_videos_all/       (input video copy — deleted earlier; belt-and-suspenders)
    - squat/extracted_features_clean/ (large landmark JSONs — no longer needed after scoring)

    What is kept (in run_root/logs/):
    - Per-stage log files (~21 KB) — useful for debugging failures

    What is kept (in run_root/workspace/squat/):
    - aqa_analysis_simple/  (AQA JSON — source of truth for re-collecting results)
    - neural_analysis/      (neural JSON — same)
    """
    subdirs_to_remove = [
        "squat/visualized_poses_clean",
        "squat/visualized_segmentation",
        "squat/analysis_reports",
        "squat/dataset_videos_all",
        "squat/extracted_features_clean",
        "squat/segmented_reps",
    ]
    for rel in subdirs_to_remove:
        target = workspace_root / rel
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


# ── Result collection ──────────────────────────────────────────────────────────
def _find_json(base: Path, pattern: str) -> Path | None:
    """Glob for a JSON file anywhere under base."""
    matches = sorted(base.rglob(pattern))
    return matches[0] if matches else None


def collect_results(workspace_root: Path, video_id: str) -> dict[str, Any]:
    """
    Locate Stage 8 (heuristic) and Stage 9 (neural) output JSONs and merge them
    into a single normalized result dict for the web app.
    """
    import json
    
    print(f"[DIAGNOSTIC] collect_results START: video_id={video_id}, workspace={workspace_root}", flush=True)

    aqa_base = workspace_root / "squat" / "aqa_analysis_simple"
    neural_base = workspace_root / "squat" / "neural_analysis"

    aqa_file = _find_json(aqa_base, f"{video_id}_aqa_simple.json")
    neural_file = _find_json(neural_base, f"{video_id}_neural.json")

    if aqa_file is None:
        raise FileNotFoundError(
            f"AQA output not found for '{video_id}' under {aqa_base}"
        )

    aqa = json.loads(aqa_file.read_text(encoding="utf-8"))
    neural = json.loads(neural_file.read_text(encoding="utf-8")) if neural_file else None

    # Build rep-level merged list
    aqa_reps: list[dict] = aqa.get("repetitions", [])
    neural_reps: list[dict] = (neural or {}).get("reps", [])
    neural_by_id = {r["rep_id"]: r for r in neural_reps}

    merged_reps = []
    for r in aqa_reps:
        rid = r.get("rep_id")
        nr = neural_by_id.get(rid, {})
        h_score = r.get("score", {}).get("overall_score")
        n_score = nr.get("neural_score")

        # Defensive correction: the fusion model bounds |residual| <= 40 (tanh × 40).
        # If |neural - heuristic| > 40, the heuristic anchor was wrong during inference
        # (e.g., defaulted to 0). Re-anchor by clamping the deviation to [-40, +40].
        corrected_score = None
        anchor_correction = False
        if h_score is not None and n_score is not None:
            deviation = n_score - h_score
            if abs(deviation) > 40.0:
                clamped_dev = max(-40.0, min(40.0, deviation))
                corrected_score = round(max(0.0, min(100.0, h_score + clamped_dev)), 2)
                anchor_correction = True

        # Per-judge aggregate scores: BiLSTM (temporal) and ST-GCN (spatial).
        # These average the individual head outputs so the web app can show a
        # three-judge breakdown: BiLSTM vs ST-GCN vs Heuristic.
        bilstm_score = None
        stgcn_score = None
        if nr:
            sm = nr.get("smoothness")
            ct = nr.get("control")
            if sm is not None and ct is not None:
                bilstm_score = round((sm + ct) / 2.0, 2)
            dp = nr.get("depth")
            fl = nr.get("forward_lean")
            kt = nr.get("knee_tracking")
            spatial_vals = [v for v in (dp, fl, kt) if v is not None]
            if spatial_vals:
                stgcn_score = round(sum(spatial_vals) / len(spatial_vals), 2)

        merged_reps.append({
            "rep_id": rid,
            "start_frame": r.get("start_frame"),
            "end_frame": r.get("end_frame"),
            "duration_seconds": r.get("duration_seconds"),
            "heuristic_score": h_score,
            "neural_score": corrected_score if anchor_correction else n_score,
            "neural_score_raw": n_score if anchor_correction else None,
            "neural_score_pre_clamp": nr.get("neural_score_pre_clamp"),
            "residual": nr.get("residual"),
            "anchor_correction_applied": anchor_correction,
            "bilstm_score": bilstm_score,
            "stgcn_score": stgcn_score,
            "metrics": r.get("metrics", {}),
            "metric_scores": r.get("score", {}).get("metric_scores", {}),
            "sub_scores": {
                "smoothness": nr.get("smoothness"),
                "control": nr.get("control"),
                "depth": nr.get("depth"),
                "forward_lean": nr.get("forward_lean"),
                "knee_tracking": nr.get("knee_tracking"),
            } if nr else None,
            "safety_clamps": nr.get("safety_clamps_applied", []),
        })

    neural_scores = [r["neural_score"] for r in merged_reps if r["neural_score"] is not None]
    bilstm_scores = [r["bilstm_score"] for r in merged_reps if r["bilstm_score"] is not None]
    stgcn_scores = [r["stgcn_score"] for r in merged_reps if r["stgcn_score"] is not None]
    any_corrections = any(r.get("anchor_correction_applied", False) for r in merged_reps)

    feedback_payload = None
    
    # === Diagnostic logging (forcing output for GCR visibility) ===
    exercise_config_exists = FEEDBACK_EXERCISE_CONFIG.exists()
    templates_config_exists = FEEDBACK_TEMPLATES_CONFIG.exists()
    merged_reps_count = len(merged_reps)
    
    print(f"[DIAGNOSTIC] Feedback prereqs: exercise_config={exercise_config_exists} ({FEEDBACK_EXERCISE_CONFIG}), templates={templates_config_exists} ({FEEDBACK_TEMPLATES_CONFIG}), merged_reps={merged_reps_count}", flush=True)
    logger.info(
        "[pipeline] Feedback config check — exercise_config exists: %s, templates exists: %s, merged_reps: %d",
        exercise_config_exists,
        templates_config_exists,
        merged_reps_count,
    )
    
    if exercise_config_exists and templates_config_exists and merged_reps:
        print(f"[DIAGNOSTIC] All prereqs met. Initializing FeedbackEngine...", flush=True)
        try:
            print(f"[DIAGNOSTIC] Loading exercise config from: {FEEDBACK_EXERCISE_CONFIG}", flush=True)
            print(f"[DIAGNOSTIC] Loading templates from: {FEEDBACK_TEMPLATES_CONFIG}", flush=True)
            feedback_engine = FeedbackEngine(
                exercise_config_path=str(FEEDBACK_EXERCISE_CONFIG),
                templates_path=str(FEEDBACK_TEMPLATES_CONFIG),
            )
            print(f"[DIAGNOSTIC] FeedbackEngine initialized successfully", flush=True)
            feedback_input: list[dict[str, Any]] = []
            for rep in merged_reps:
                metric_scores = rep.get("metric_scores") or {}
                sub_scores = rep.get("sub_scores") or {}
                metrics = rep.get("metrics") or {}

                normalized_sub_scores = {
                    "forward_lean": sub_scores.get("forward_lean") if sub_scores.get("forward_lean") is not None else metric_scores.get("forward_lean"),
                    "hip_depth": sub_scores.get("depth") if sub_scores.get("depth") is not None else metric_scores.get("depth"),
                    "knee_tracking": sub_scores.get("knee_tracking") if sub_scores.get("knee_tracking") is not None else metric_scores.get("knee_tracking"),
                    "knee_valgus": metric_scores.get("knee_valgus"),
                    "smoothness": sub_scores.get("smoothness"),
                    "control": sub_scores.get("control"),
                }

                feedback_input.append(
                    {
                        "rep_id": rep.get("rep_id"),
                        "neural_score": rep.get("neural_score") if rep.get("neural_score") is not None else rep.get("heuristic_score"),
                        "metrics": {
                            "forward_lean": metrics.get("forward_lean_deg") if metrics.get("forward_lean_deg") is not None else metrics.get("forward_lean"),
                            "hip_depth": metrics.get("squat_depth") if metrics.get("squat_depth") is not None else metrics.get("hip_depth"),
                            "knee_valgus": metrics.get("knee_valgus_ratio") if metrics.get("knee_valgus_ratio") is not None else metrics.get("knee_valgus"),
                            "knee_tracking": metrics.get("knee_tracking"),
                        },
                        "sub_scores": normalized_sub_scores,
                    }
                )

            print(f"[DIAGNOSTIC] Built feedback_input for {len(feedback_input)} reps. Calling generate_feedback...", flush=True)
            feedback_result = feedback_engine.generate_feedback(feedback_input, video_id=video_id)
            print(f"[DIAGNOSTIC] generate_feedback returned successfully with {len(feedback_result.reps)} reps", flush=True)
            feedback_payload = {
                "schema_version": feedback_result.schema_version,
                "exercise": feedback_result.exercise,
                "reps": [
                    {
                        "rep_id": item.rep_id,
                        "score": item.score,
                        "tier": item.tier,
                        "text": item.text,
                        "wins": item.wins,
                        "issues": item.issues,
                    }
                    for item in feedback_result.reps
                ],
                "session": {
                    "avg_score": feedback_result.session.avg_score,
                    "trajectory": feedback_result.session.trajectory,
                    "most_improved_metric": feedback_result.session.most_improved_metric,
                    "persistent_issue": feedback_result.session.persistent_issue,
                    "aggregate_text": feedback_result.session.aggregate_text,
                    "coach_text": feedback_result.session.coach_text,
                },
            }
            print(f"[DIAGNOSTIC] Feedback payload built successfully: {len(feedback_payload['reps'])} reps, trajectory={feedback_payload['session']['trajectory']}", flush=True)
            logger.info(
                "[pipeline] Feedback generated successfully: %d rep(s), session trajectory=%s",
                len(feedback_payload["reps"]),
                feedback_payload["session"]["trajectory"],
            )
        except Exception as exc:
            import traceback
            tb_str = traceback.format_exc()
            print(f"[DIAGNOSTIC] FeedbackEngine EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
            print(f"[DIAGNOSTIC] Traceback:\n{tb_str}", flush=True)
            logger.error(
                "[pipeline] Feedback generation FAILED for video_id=%s: %s\n%s",
                video_id,
                exc,
                tb_str,
            )
            feedback_payload = {
                "schema_version": "1.0",
                "exercise": "squat",
                "error": f"Feedback generation skipped: {exc}",
            }
    else:
        # Log why we skip feedback generation
        if not exercise_config_exists:
            print(f"[DIAGNOSTIC] Skipping feedback: exercise_config not found at {FEEDBACK_EXERCISE_CONFIG}", flush=True)
        if not templates_config_exists:
            print(f"[DIAGNOSTIC] Skipping feedback: templates_config not found at {FEEDBACK_TEMPLATES_CONFIG}", flush=True)
        if not merged_reps:
            print(f"[DIAGNOSTIC] Skipping feedback: no merged_reps to process (count={merged_reps_count})", flush=True)

    result = {
        "video_id": video_id,
        "view": aqa.get("view"),
        "quality": aqa.get("source_quality"),
        "rep_count": len(merged_reps),
        "overall_heuristic_score": aqa.get("overall_score"),
        "overall_neural_score": round(sum(neural_scores) / len(neural_scores), 2) if neural_scores else None,
        "overall_bilstm_score": round(sum(bilstm_scores) / len(bilstm_scores), 2) if bilstm_scores else None,
        "overall_stgcn_score": round(sum(stgcn_scores) / len(stgcn_scores), 2) if stgcn_scores else None,
        "neural_available": neural is not None,
        "any_anchor_corrections": any_corrections,
        "reps": merged_reps,
        "feedback": feedback_payload,
    }
    
    has_feedback = feedback_payload is not None and "error" not in (feedback_payload or {})
    feedback_reps = len(feedback_payload.get("reps", [])) if has_feedback else 0
    print(f"[DIAGNOSTIC] collect_results END: rep_count={result['rep_count']}, has_feedback={has_feedback}, feedback_reps={feedback_reps}, neural_available={result['neural_available']}", flush=True)
    
    return result


# ── Top-level runner ───────────────────────────────────────────────────────────
async def download_video(url: str, dest_dir: Path) -> Path:
    """Download a video from a URL (e.g. Supabase signed URL) into dest_dir."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        r = await client.get(url)
        r.raise_for_status()

    # Derive filename from Content-Disposition or URL path
    cd = r.headers.get("content-disposition", "")
    if "filename=" in cd:
        filename = cd.split("filename=")[-1].strip('"')
    else:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "video.mp4"

    dest = dest_dir / filename
    dest.write_bytes(r.content)
    return dest


def run_pipeline_sync(
    job_id: str,
    video_path: Path,
    stages: list[str] | None = None,
    mode: str = "filtered",
) -> dict[str, Any]:
    """
    Run the full pipeline synchronously for a single video.
    Called from a background thread/process by the API server.
    Returns the merged result dict.
    """
    print(f"[DIAGNOSTIC] run_pipeline_sync START: job_id={job_id}, video={video_path.name}, stages={stages or 'default'}, mode={mode}", flush=True)
    stages = stages or DEFAULT_STAGES
    run_root = RUNS_ROOT / job_id
    workspace_root = run_root / "workspace"
    logs_root = run_root / "logs"

    workspace_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    _prepare_workspace(workspace_root, video_path)
    video_id = video_path.stem

    for key in stages:
        spec = STAGE_SPECS.get(key)
        if spec is None:
            raise ValueError(f"Unknown stage: '{key}'")
        if not spec.script.exists():
            raise FileNotFoundError(f"Stage script not found: {spec.script}")

        # Neural fusion is optional: if it fails, we still return feedback based on heuristic scores.
        # All other stages are mandatory: failure propagates.
        try:
            _run_stage(key, spec.script, video_id, workspace_root, logs_root, mode)
            _validate_stage_output(key, workspace_root, video_id)
        except RuntimeError as exc:
            if key == "neural_fusion":
                # Neural fusion failure is non-fatal. Log and continue.
                # Feedback will be generated from heuristic scores instead.
                logger.warning(f"Neural fusion stage failed (non-fatal): {exc}")
            else:
                raise

        # After extraction completes, remove the input video copy — it is no longer
        # needed (stage 2.5 has already produced the feature JSON) and accounts for
        # ~15% of workspace disk usage.
        if key == "extract_selected_features":
            _delete_input_video(workspace_root, video_path.name)

    result = collect_results(workspace_root, video_id)

    # Note: if neural_fusion was requested but failed, result.neural_available will be False
    # and all reps will have neural_score=None. This is NOT an error condition — feedback
    # is still generated (it's based on heuristic scores), and the frontend can detect
    # neural unavailability via result.neural_available flag. We do NOT raise here because:
    # 1. Feedback generation must always succeed, even if neural failed
    # 2. The frontend has fallback UI for when neural is unavailable
    # 3. Raising would prevent valid feedback from being returned to the user

    # Tear down the workspace after results are safely collected.  The meaningful
    # output (2 KB of JSON) is returned in-memory; the workspace is ~5–7 MB of
    # intermediate files that are not needed after this point.
    # Logs are retained for debugging; only the heavy intermediate files are removed.
    _cleanup_workspace(workspace_root)

    print(f"[DIAGNOSTIC] run_pipeline_sync END: job_id={job_id}, rep_count={result['rep_count']}, has_feedback={result['feedback'] is not None}, neural_available={result['neural_available']}", flush=True)
    return result
