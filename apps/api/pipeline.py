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


def _build_phase_timeline(
    seg_rep: dict,
    rep_idx: int,
    all_seg_reps: list[dict],
    fps: float,
    frame_count: int,
) -> dict[str, Any] | None:
    """
    Build a phase timeline for one rep, including idle phases inferred from gaps.

    Timeline origin is the start of the idle-before window (or rep.start_frame for
    the first rep when the video begins immediately with movement). All times are in
    seconds relative to that origin — so the chart x-axis always starts at 0.

    Returns None if inputs are missing or fps is invalid.
    """
    if not seg_rep or fps <= 0:
        return None

    rep_start = int(seg_rep.get("start_frame", 0))
    rep_end = int(seg_rep.get("end_frame", 0))

    # Idle-before: from previous rep's end+1 (or frame 0 for the first rep) to rep start-1
    prev_end = int(all_seg_reps[rep_idx - 1].get("end_frame", 0)) + 1 if rep_idx > 0 else 0
    idle_before_start = prev_end
    idle_before_end = rep_start - 1

    # Idle-after: from rep end+1 to next rep's start-1 (or video end for the last rep)
    next_start = int(all_seg_reps[rep_idx + 1].get("start_frame", frame_count)) if rep_idx < len(all_seg_reps) - 1 else frame_count
    idle_after_start = rep_end + 1
    idle_after_end = next_start - 1

    # t=0 is the beginning of the idle-before window
    origin = idle_before_start

    def to_s(frame: int) -> float:
        return round((frame - origin) / fps, 3)

    phases: list[dict] = []
    eccentric_s = isometric_s = concentric_s = 0.0

    idle_before_frames = max(0, idle_before_end - idle_before_start + 1)
    if idle_before_frames > 0:
        phases.append({
            "phase": "idle",
            "start_s": to_s(idle_before_start),
            "end_s": to_s(idle_before_end + 1),
            "duration_s": round(idle_before_frames / fps, 3),
        })

    for phase in seg_rep.get("phases", []):
        ptype = phase.get("phase_type", "unknown")
        pstart = int(phase.get("start_frame", rep_start))
        pend = int(phase.get("end_frame", rep_end))
        dur = round(phase.get("duration_seconds", (pend - pstart + 1) / fps), 3)
        phases.append({
            "phase": ptype,
            "start_s": to_s(pstart),
            "end_s": to_s(pend + 1),
            "duration_s": dur,
        })
        if ptype == "eccentric":
            eccentric_s = dur
        elif ptype == "isometric":
            isometric_s = dur
        elif ptype == "concentric":
            concentric_s = dur

    idle_after_frames = max(0, idle_after_end - idle_after_start + 1)
    if idle_after_frames > 0:
        phases.append({
            "phase": "idle",
            "start_s": to_s(idle_after_start),
            "end_s": to_s(idle_after_end + 1),
            "duration_s": round(idle_after_frames / fps, 3),
        })

    return {
        "fps": fps,
        "phases": phases,
        "summary": {
            "eccentric_s": eccentric_s,
            "isometric_s": isometric_s,
            "concentric_s": concentric_s,
            "idle_before_s": round(idle_before_frames / fps, 3),
            "idle_after_s": round(idle_after_frames / fps, 3),
            "tempo_ratio": round(eccentric_s / concentric_s, 2) if concentric_s > 0 else None,
        },
    }


def _build_kinematic_data(
    seg_rep: dict,
    all_seg_reps: list[dict],
    rep_idx: int,
    fps: float,
    frame_count: int,
    hip_displacement: list[float],
) -> list[dict[str, float]] | None:
    """
    Build ROM time-series for one rep using hip vertical displacement.

    Returns list of {time, rom} dicts, sampled at phase transitions + uniform spacing.
    ROM = (1 - normalized_hip_displacement) * 100, so:
    - 0% = deepest squat (max displacement)
    - 100% = standing (zero displacement)

    If hip_displacement is missing or rep data incomplete, return None.
    """
    if not seg_rep or not hip_displacement:
        return None

    rep_start = int(seg_rep.get("start_frame", 0))
    rep_end = int(seg_rep.get("end_frame", 0))
    phases = seg_rep.get("phases", [])

    if rep_end <= rep_start or rep_end >= len(hip_displacement):
        return None

    # Collect all sample frame indices
    sample_frames = set()

    # Add phase transition frames
    sample_frames.add(rep_start)  # Idle start
    for phase in phases:
        sample_frames.add(int(phase.get("start_frame", rep_start)))
        sample_frames.add(int(phase.get("end_frame", rep_end)))
    sample_frames.add(rep_end)  # Last frame

    # Add uniform spacing: target 10–20 samples per rep
    rep_frames = rep_end - rep_start + 1
    target_count = 15  # Aim for 15 samples
    interval = max(1, rep_frames // target_count)
    for f in range(rep_start, rep_end + 1, interval):
        sample_frames.add(f)

    # Remove out-of-bounds frames
    sample_frames = sorted(f for f in sample_frames if rep_start <= f <= rep_end)

    # Build kinematic data
    kinematic_data = []
    for frame in sample_frames:
        disp = hip_displacement[frame]
        rom = round((1.0 - disp) * 100.0, 1)  # 1.0 disp → 0% ROM, 0.0 disp → 100% ROM
        rom = max(0.0, min(100.0, rom))  # Clamp to [0, 100]
        time = round((frame - rep_start) / fps, 3)
        kinematic_data.append({"time": time, "rom": rom})

    return sorted(kinematic_data, key=lambda x: x["time"])


def _tier_for_score(score: float) -> str:
    """Map numeric score to feedback tier labels expected by the frontend."""
    if score >= 85.0:
        return "excellent"
    if score >= 75.0:
        return "strong"
    if score >= 60.0:
        return "okay"
    if score >= 40.0:
        return "needs_work"
    return "focus_here"

def _build_feedback_fallback(merged_reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a schema-compatible fallback when template/config files are unavailable."""
    rep_payload: list[dict[str, Any]] = []
    rep_scores: list[float] = []

    for rep in merged_reps:
        score_val = rep.get("neural_score")
        if score_val is None:
            score_val = rep.get("heuristic_score")
        score = float(score_val if score_val is not None else 0.0)
        rep_scores.append(score)
        rep_payload.append(
            {
                "rep_id": rep.get("rep_id"),
                "score": round(score, 2),
                "tier": _tier_for_score(score),
                "text": "Coaching baseline: detailed narrative template is unavailable in this deployment, but scoring data is valid.",
                "wins": [],
                "issues": [],
            }
        )

    avg_score = round(sum(rep_scores) / len(rep_scores), 2) if rep_scores else 0.0
    return {
        "schema_version": "1.0",
        "exercise": "squat",
        "reps": rep_payload,
        "session": {
            "avg_score": avg_score,
            "trajectory": "stable",
            "most_improved_metric": None,
            "persistent_issue": None,
            "aggregate_text": "Detailed coaching template unavailable in this deployment.",
            "coach_text": "Scoring completed successfully. Narrative templates are temporarily unavailable.",
        },
    }


def collect_results(workspace_root: Path, video_id: str) -> dict[str, Any]:
    """
    Locate Stage 8 (heuristic), Stage 9 (neural), and Stage 5 (segmentation) output
    JSONs and merge them into a single normalized result dict for the web app.
    """
    import json

    print(f"[DIAGNOSTIC] collect_results START: video_id={video_id}, workspace={workspace_root}", flush=True)

    aqa_base = workspace_root / "squat" / "aqa_analysis_simple"
    neural_base = workspace_root / "squat" / "neural_analysis"
    seg_base = workspace_root / "squat" / "segmented_reps"

    aqa_file = _find_json(aqa_base, f"{video_id}_aqa_simple.json")
    neural_file = _find_json(neural_base, f"{video_id}_neural.json")
    seg_file = _find_json(seg_base, f"{video_id}_segmented.json")

    if aqa_file is None:
        raise FileNotFoundError(
            f"AQA output not found for '{video_id}' under {aqa_base}"
        )

    aqa = json.loads(aqa_file.read_text(encoding="utf-8"))
    neural = json.loads(neural_file.read_text(encoding="utf-8")) if neural_file else None
    seg = json.loads(seg_file.read_text(encoding="utf-8")) if seg_file else None

    # Build rep-level merged list
    aqa_reps: list[dict] = aqa.get("repetitions", [])
    neural_reps: list[dict] = (neural or {}).get("reps", [])
    neural_by_id = {r["rep_id"]: r for r in neural_reps}

    # Phase timeline data (from Stage 5 segmentation)
    seg_reps: list[dict] = (seg or {}).get("repetitions", [])
    seg_by_id = {r["rep_id"]: r for r in seg_reps}
    seg_fps = (seg or {}).get("info", {}).get("fps", 30.0)
    seg_frame_count = (seg or {}).get("info", {}).get("frame_count", 0)

    # Extract per-frame hip displacement from segmented JSON
    seg_hip_displacement = []
    if seg:
        seg_signals = seg.get("signals", {})
        seg_hip_displacement = seg_signals.get("normalized_hip_displacement", [])

    merged_reps = []
    for rep_idx, r in enumerate(aqa_reps):
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

        # Phase timeline from Stage 5 segmentation (optional)
        seg_rep = seg_by_id.get(rid)
        phase_timeline = None
        if seg_rep:
            phase_timeline = _build_phase_timeline(
                seg_rep,
                rep_idx,
                seg_reps,
                fps=seg_fps,
                frame_count=seg_frame_count,
            )

        # Kinematic data (ROM time-series) from Stage 5 segmentation
        kinematic_data = None
        if seg_rep and seg_hip_displacement:
            kinematic_data = _build_kinematic_data(
                seg_rep,
                seg_reps,
                rep_idx,
                fps=seg_fps,
                frame_count=seg_frame_count,
                hip_displacement=seg_hip_displacement,
            )

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
            "phase_timeline": phase_timeline,
            "kinematic_data": kinematic_data,
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
    elif merged_reps:
        logger.warning(
            "[pipeline] Feedback config missing; returning schema-compatible fallback payload. "
            "exercise_config=%s templates_config=%s",
            exercise_config_exists,
            templates_config_exists,
        )
        feedback_payload = _build_feedback_fallback(merged_reps)
        print(
            f"[DIAGNOSTIC] Fallback feedback generated: {len(feedback_payload.get('reps', []))} reps "
            f"(exercise_config={exercise_config_exists}, templates_config={templates_config_exists})",
            flush=True,
        )
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
