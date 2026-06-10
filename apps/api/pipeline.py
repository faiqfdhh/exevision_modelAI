"""
ExeVision AI — Pipeline Runner

Mirrors the stage execution logic from apps/desktop-ui/app.py without Tkinter.
Runs the five pipeline stages as subprocesses inside an isolated workspace.
Does NOT modify any scoring, segmentation, or feature-extraction logic.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from core.exevision.feedback.engine import FeedbackEngine

logger = logging.getLogger(__name__)

# Module-level cache for LLMFeedbackEnhancer to avoid rebuilding on every request
_llm_feedback_enhancer_cache: "LLMFeedbackEnhancer | None" = None

_DIAGONAL_ALIASES = {"front_side", "back_side", "front-side", "back-side"}
_STRAIGHT_ALIASES = {"front", "back"}

# Neural sub-score relaxation: applies a power-curve to BiLSTM outputs
# to reduce strictness.  display = 100 * pow(score/100, exponent)
# where exponent < 1 inflates low-to-mid scores more than high ones.
# Configurable per-instance by changing the default.
_DEFAULT_SMOOTH_EXPONENT = 0.65


def _smooth_relax(sub_scores: dict, exponent: float = _DEFAULT_SMOOTH_EXPONENT) -> dict:
    for k in ("smoothness", "control"):
        v = sub_scores.get(k)
        if v is not None:
            relaxed = 100.0 * pow(max(0.001, v / 100.0), exponent)
            sub_scores[k] = round(max(0.0, min(100.0, relaxed)), 1)
    return sub_scores


def _display_view(raw_view) -> str | None:
    """User-facing view label. Collapses front_side/back_side → 'diagonal',
    and front/back → 'straight'. Backend keeps raw labels; API consumers
    see unified display labels."""
    if raw_view is None:
        return None
    v = str(raw_view).lower().strip()
    if not v:
        return raw_view
    if v in _DIAGONAL_ALIASES:
        return "diagonal"
    if v in _STRAIGHT_ALIASES:
        return "straight"
    return v

# ── Paths ──────────────────────────────────────────────────────────────────────
# apps/api/ → apps/ → exevision_modelAI/
_API_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = _API_DIR.parents[1]
STAGES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "stages"
RUNS_ROOT = WORKSPACE_ROOT / "pipeline_ui_runs"
SHARED_MODEL_PATH = WORKSPACE_ROOT / "models" / "runtime_pose_and_face" / "pose_landmarker_heavy.task"
SHARED_FACE_MODEL_PATH = WORKSPACE_ROOT / "models" / "runtime_pose_and_face" / "blaze_face_short_range.tflite"


def _get_model_path(model_name: str, exercise: str) -> Path:
    """Construct exercise-specific model path, with fallback chain."""
    if exercise in ("overhead_press", "seated_overhead_press"):
        ohp_finetuned = WORKSPACE_ROOT / "models" / "runtime_neural_ohp" / f"{model_name}_ohp_finetuned.pt"
        if ohp_finetuned.exists():
            return ohp_finetuned
        ohp_specific = WORKSPACE_ROOT / "models" / f"{model_name}_{exercise}.pt"
        if ohp_specific.exists():
            return ohp_specific
    specific = WORKSPACE_ROOT / "models" / f"{model_name}_{exercise}.pt"
    if specific.exists():
        return specific
    if model_name in ["bilstm", "stgcn", "fusion"]:
        squat_ckpt = WORKSPACE_ROOT / "models" / "runtime_neural_squat" / f"{model_name}_finetuned.pt"
        if squat_ckpt.exists():
            if exercise not in ("squat",):
                logger.warning(
                    "[pipeline] No %s checkpoint for exercise='%s'; falling back to squat weights. "
                    "Neural fusion will likely fail and degrade to heuristic-only.",
                    model_name, exercise,
                )
            return squat_ckpt
        fusion_layer = WORKSPACE_ROOT / "models" / "runtime_neural_squat" / "fusion_layer.pt"
        if model_name == "fusion" and fusion_layer.exists():
            if exercise not in ("squat",):
                logger.warning(
                    "[pipeline] No fusion checkpoint for exercise='%s'; falling back to squat fusion_layer.pt. "
                    "Neural fusion will likely fail and degrade to heuristic-only.",
                    exercise,
                )
            return fusion_layer
        if exercise not in ("squat",):
            logger.warning(
                "[pipeline] No %s checkpoint for exercise='%s' at expected paths; "
                "falling back to flat models/ dir. Neural fusion will likely fail.",
                model_name, exercise,
            )
        return WORKSPACE_ROOT / "models" / f"{model_name}_finetuned.pt"
    return WORKSPACE_ROOT / "models" / f"{model_name}.pt"
BILSTM_CKPT = _get_model_path("bilstm", "squat")
STGCN_CKPT = _get_model_path("stgcn", "squat")
FUSION_CKPT = _get_model_path("fusion", "squat")

FEEDBACK_EXERCISE_CONFIG = WORKSPACE_ROOT / "core" / "exevision" / "config" / "exercises" / "squat.json"
FEEDBACK_TEMPLATES_CONFIG = WORKSPACE_ROOT / "core" / "exevision" / "config" / "templates" / "feedback_templates.json"
EXERCISES_CONFIG_DIR = WORKSPACE_ROOT / "core" / "exevision" / "config" / "exercises"


def _resolve_exercise_config(exercise: str) -> Path:
    """Resolve the exercise config JSON path, raising if not found.

    Maps exercise aliases (e.g. seated_overhead_press) to their base config.
    """
    _EXERCISE_CONFIG_ALIASES: dict[str, str] = {
        "seated_overhead_press": "overhead_press",
    }
    config_name = _EXERCISE_CONFIG_ALIASES.get(exercise, exercise)
    path = EXERCISES_CONFIG_DIR / f"{config_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Exercise config not found: {path}")
    return path


# ── Stage definitions ──────────────────────────────────────────────────────────
@dataclass
class StageSpec:
    key: str
    script: Path
    output_dirs: tuple[str, ...] = field(default_factory=tuple)


def _build_stage_specs(exercise: str = "squat") -> dict[str, StageSpec]:
    """Build stage specs with exercise-specific paths."""
    return {
        "extract_selected_features": StageSpec(
            key="extract_selected_features",
            script=STAGES_DIR / "extract_selected_features.py",
            output_dirs=(f"{exercise}/extracted_features_clean", f"{exercise}/visualized_poses_clean", f"{exercise}/analysis_reports"),
        ),
        "classify_views": StageSpec(
            key="classify_views",
            script=STAGES_DIR / "classify_views.py",
            output_dirs=(f"{exercise}/extracted_features_clean",),
        ),
        "temporal_segmentation": StageSpec(
            key="temporal_segmentation",
            script=STAGES_DIR / "temporal_segmentation.py",
            output_dirs=(f"{exercise}/segmented_reps", f"{exercise}/visualized_segmentation"),
        ),
        "scoring": StageSpec(
            key="scoring",
            script=STAGES_DIR / "scoring.py",
            output_dirs=(f"{exercise}/aqa_analysis_simple",),
        ),
        "neural_fusion": StageSpec(
            key="neural_fusion",
            script=STAGES_DIR / "neural_fusion_inference.py",
            output_dirs=(f"{exercise}/neural_analysis",),
        ),
    }


# Default to squat for backward compatibility
STAGE_SPECS = _build_stage_specs("squat")

DEFAULT_STAGES = [
    "extract_selected_features",
    "classify_views",
    "temporal_segmentation",
    "scoring",
    "neural_fusion",
]


# ── Workspace helpers ──────────────────────────────────────────────────────────
def _prepare_workspace(workspace_root: Path, video_path: Path, exercise: str = "squat") -> None:
    """Create workspace directory tree and copy the input video into it."""
    videos_dir = workspace_root / exercise / "dataset_videos_all"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / exercise / "aqa_analysis_simple").mkdir(parents=True, exist_ok=True)
    dest = videos_dir / video_path.name
    shutil.copy2(video_path, dest)


def _build_stage_cmd(key: str, script: Path, video_id: str, mode: str, generate_viz: bool = True, exercise: str = "squat", rep_boundaries_path: str | None = None, legacy_fsm: bool = False) -> list[str]:
    """Build the subprocess command for a stage, mirroring app.py arg construction.

    API runs skip all visualization outputs (--no-report) by default.
    Visualized outputs (--no-viz) are generated only if generate_viz is True.
    """
    base = [sys.executable, str(script)]
    exercise_args = ["--exercise", exercise]
    if key == "extract_selected_features":
        cmd = base + [mode, "--video-id", video_id, "--no-report"] + exercise_args
        if not generate_viz:
            cmd.append("--no-viz")
        return cmd
    elif key == "temporal_segmentation":
        cmd = base + ["--video-id", video_id] + exercise_args
        if rep_boundaries_path:
            cmd.extend(["--rep-boundaries", rep_boundaries_path])
        if not generate_viz:
            cmd.append("--no-viz")
        if legacy_fsm:
            cmd.append("--legacy-fsm")
        return cmd
    elif key == "classify_views":
        return base + ["--video-id", video_id] + exercise_args
    elif key == "scoring":
        return base + [video_id] + exercise_args
    elif key == "neural_fusion":
        bilstm_path = _get_model_path("bilstm", exercise)
        stgcn_path = _get_model_path("stgcn", exercise)
        fusion_path = _get_model_path("fusion", exercise)
        return base + [
            "--video-id", video_id,
            "--bilstm-ckpt", str(bilstm_path),
            "--stgcn-ckpt", str(stgcn_path),
            "--fusion-ckpt", str(fusion_path),
            "--quality-tier", "raw_unfiltered",
        ] + exercise_args
    return base


def _run_stage(
    key: str,
    script: Path,
    video_id: str,
    workspace_root: Path,
    logs_root: Path,
    mode: str,
    generate_viz: bool = True,
    exercise: str = "squat",
    rep_boundaries_path: str | None = None,
    legacy_fsm: bool = False,
) -> str:
    """Run one pipeline stage; returns captured stdout+stderr."""
    cmd = _build_stage_cmd(key, script, video_id, mode, generate_viz, exercise, rep_boundaries_path, legacy_fsm=legacy_fsm)
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


def _validate_stage_output(key: str, workspace_root: Path, video_id: str, exercise: str = "squat") -> None:
    """Ensure each stage produced the expected artifact for the requested video."""
    expected_patterns = {
        "extract_selected_features": f"{exercise}/extracted_features_clean/**/{video_id}.json",
        "classify_views": f"{exercise}/extracted_features_clean/**/{video_id}.json",
        "temporal_segmentation": f"{exercise}/segmented_reps/**/{video_id}_segmented.json",
        "scoring": f"{exercise}/aqa_analysis_simple/**/{video_id}_aqa_simple.json",
        "neural_fusion": f"{exercise}/neural_analysis/**/{video_id}_neural.json",
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

def _delete_input_video(workspace_root: Path, filename: str, exercise: str = "squat") -> None:
    """Remove the input video copy from the workspace after extraction."""
    video_copy = workspace_root / exercise / "dataset_videos_all" / filename
    if video_copy.exists():
        video_copy.unlink()


def _cleanup_workspace(workspace_root: Path, generate_viz: bool = True, exercise: str = "squat") -> None:
    """
    Remove heavy intermediate artifacts from the workspace after results are collected.

    Args:
        workspace_root: Root directory of the workspace
        generate_viz: If True, visualization directories are KEPT (served to frontend).
                      If False, they are removed as they are unneeded.
        exercise: Exercise type; used to construct path prefixes.

    What is removed:
    - {exercise}/visualized_segmentation/  (phase overlay MP4s — only removed if generate_viz=False)
    - {exercise}/analysis_reports/         (PNG plots — only useful in desktop UI)
    - {exercise}/extracted_features_clean/ (large landmark JSONs — no longer needed after scoring)
    - {exercise}/segmented_reps/           (intermediate rep JSON — no longer needed)

    What is kept (in run_root/logs/):
    - Per-stage log files (~21 KB) — useful for debugging failures

    What is kept (in run_root/workspace/{exercise}/):
    - aqa_analysis_simple/  (AQA JSON — source of truth for re-collecting results)
    - neural_analysis/      (neural JSON — same)
    - visualized_poses_clean/    (annotated pose MP4s — kept if generate_viz=True)
    - visualized_segmentation/   (phase overlay MP4s — kept if generate_viz=True)
    """
    subdirs_to_remove = [
        f"{exercise}/analysis_reports",
        f"{exercise}/extracted_features_clean",
        f"{exercise}/segmented_reps",
    ]
    # Only remove visualization directories if visualization was not requested
    if not generate_viz:
        subdirs_to_remove.extend([
            f"{exercise}/visualized_segmentation",
            f"{exercise}/visualized_poses_clean",
        ])
    
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
        return "good"
    if score >= 60.0:
        return "fair"
    if score >= 40.0:
        return "poor"
    return "critical"


def coerce_old_feedback_format(old_feedback: dict[str, Any]) -> dict[str, Any]:
    """Convert old {wins, issues} format to new {items} format for backward compatibility."""
    items: list[dict[str, Any]] = []

    # Process old wins array
    for win in old_feedback.get("wins", []):
        if isinstance(win, dict):
            text = win.get("text", "")
            score = win.get("score", 80)
        else:
            text = str(win)
            score = 80

        items.append(
            {
                "text": text,
                "score": int(max(0, min(100, score))),
                "category": "geometric",
                "type": "win",
            }
        )

    # Process old issues array
    for issue in old_feedback.get("issues", []):
        if isinstance(issue, dict):
            text = issue.get("text", "")
            score = issue.get("score") or issue.get("metric_value", 50)
        else:
            text = str(issue)
            score = 50

        items.append(
            {
                "text": text,
                "score": int(max(0, min(100, score))),
                "category": "geometric",
                "type": "issue",
            }
        )

    # Sort by score (ascending — most severe first)
    items.sort(key=lambda x: x["score"])

    # Build new format
    return {
        "schema_version": "2.0",
        "reps": [
            {
                "rep_id": old_feedback.get("rep_id"),
                "score": old_feedback.get("score", 50),
                "tier": old_feedback.get("tier", "fair"),
                "text": old_feedback.get("text", ""),
                "items": items,
            }
        ],
        "session": old_feedback.get("session", {}),
    }


def _build_feedback_fallback(merged_reps: list[dict[str, Any]], exercise: str = "squat") -> dict[str, Any]:
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
                "items": [],
            }
        )

    avg_score = round(sum(rep_scores) / len(rep_scores), 2) if rep_scores else 0.0
    return {
        "schema_version": "2.0",
        "exercise": exercise,
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


def _upload_visualization_to_supabase(
    viz_path: Path,
    job_id: str,
) -> str | None:
    """
    Upload annotated video to Supabase Storage and return relative bucket path.

    Frontend generates signed URLs via /api/inference/visualization/[jobId].
    Backend only handles upload and returns the relative path.

    Args:
        viz_path: Path to the annotated MP4 video file
        job_id: Job/inference ID (used in Supabase path)

    Returns:
        Relative path "inference-results/{job_id}/with_landmarks.mp4" if successful, or None.
    """
    if not viz_path.exists():
        logger.warning(f"Visualization file not found: {viz_path}")
        return None

    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

    if not supabase_url or not supabase_key:
        logger.error(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required. "
            "See apps/api/.env.example for setup instructions."
        )
        return None

    try:
        from supabase import create_client, Client

        supabase: Client = create_client(supabase_url, supabase_key)
        bucket = "inference-results"
        file_path = f"{job_id}/with_landmarks.mp4"

        logger.info(f"Uploading visualization to Supabase: {bucket}/{file_path}")

        with open(viz_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                path=file_path,
                file=f,
                file_options={
                    "content-type": "video/mp4",
                    "x-upsert": "true",
                },
            )

        # Return relative path for frontend to generate signed URL
        relative_path = f"{bucket}/{file_path}"
        logger.info(f"Visualization uploaded successfully: {relative_path}")
        return relative_path

    except Exception as e:
        logger.error(f"Failed to upload visualization to Supabase: {e}", exc_info=True)
        return None


def collect_results(workspace_root: Path, video_id: str, exercise: str = "squat") -> dict[str, Any]:
    """
    Locate Stage 8 (heuristic), Stage 9 (neural), and Stage 5 (segmentation) output
    JSONs and merge them into a single normalized result dict for the web app.
    """
    import json

    print(f"[DIAGNOSTIC] collect_results START: video_id={video_id}, workspace={workspace_root}, exercise={exercise}", flush=True)

    aqa_base = workspace_root / exercise / "aqa_analysis_simple"
    neural_base = workspace_root / exercise / "neural_analysis"
    seg_base = workspace_root / exercise / "segmented_reps"

    aqa_file = _find_json(aqa_base, f"{video_id}_aqa_simple.json")
    neural_file = _find_json(neural_base, f"{video_id}_neural.json")

    # Prefer the segmented JSON from the same quality tier as the neural file.
    # _find_json sorts alphabetically and "excellent" (0 reps) precedes "raw_unfiltered"
    # (1+ reps), so naively grabbing the first match returns an empty-rep file.
    candidate: Path | None = None
    if neural_file is not None:
        quality_tier = neural_file.parent.name  # e.g. "raw_unfiltered"
        candidate = seg_base / quality_tier / f"{video_id}_segmented.json"
        seg_file = candidate if candidate.exists() else _find_json(seg_base, f"{video_id}_segmented.json")
    else:
        seg_file = _find_json(seg_base, f"{video_id}_segmented.json")

    print(f"[DIAGNOSTIC] neural_file={neural_file}, neural_file.parent.name={neural_file.parent.name if neural_file else 'N/A'}", flush=True)
    print(f"[DIAGNOSTIC] candidate={candidate if neural_file else 'N/A'}, exists={candidate.exists() if neural_file else 'N/A'}", flush=True)
    print(f"[DIAGNOSTIC] seg_file={seg_file}, exists={seg_file.exists() if seg_file else 'NONE'}", flush=True)

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
                bilstm_score = round(max(0.0, min(100.0, (sm + ct) / 2.0 * 100.0)), 2)
            # Prefer the model's own pre-computed aggregate (exercise-agnostic,
            # correct weighting, immune to scale mismatches between heads).
            agg = nr.get("aggregated_score")
            if agg is not None:
                stgcn_score = round(float(agg), 2)
            else:
                # Fallback: average 0-100 spatial sub-scores manually.
                # Sub-scores from model heads are 0-1; multiply by 100 for display.
                dp = nr.get("depth")
                fl = nr.get("forward_lean")
                kt = nr.get("knee_tracking")
                spatial_vals = [v * 100.0 for v in (dp, fl, kt) if v is not None]
                if not spatial_vals:
                    ef = nr.get("elbow_flare")
                    gr = nr.get("grip_ratio")
                    rt = nr.get("rom_top")
                    rb = nr.get("rom_bottom")
                    spatial_vals = [v * 100.0 for v in (ef, gr, rt, rb) if v is not None]
                if spatial_vals:
                    stgcn_score = round(max(0.0, min(100.0, sum(spatial_vals) / len(spatial_vals))), 2)

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
                "forward_lean": nr.get("forward_lean"),
                "knee_tracking": nr.get("knee_tracking"),
                "lockout": nr.get("lockout"),
                "elbow_flare": nr.get("elbow_flare"),
                "grip_ratio": nr.get("grip_ratio"),
                "rom_top": nr.get("rom_top"),
                "rom_bottom": nr.get("rom_bottom"),
                "knee_error": nr.get("knee_error"),
            } if nr else None,
            "safety_clamps": nr.get("safety_clamps_applied", []),
            "phase_timeline": phase_timeline,
            "kinematic_data": kinematic_data,
        })

    feedback_payload = None
    
    # === Diagnostic logging (forcing output for GCR visibility) ===
    exercise_config_exists = _resolve_exercise_config(exercise).exists() if EXERCISES_CONFIG_DIR.exists() else False
    templates_config_exists = FEEDBACK_TEMPLATES_CONFIG.exists()
    merged_reps_count = len(merged_reps)
    
    print(f"[DIAGNOSTIC] Feedback prereqs: exercise_config={exercise_config_exists} ({_resolve_exercise_config(exercise) if EXERCISES_CONFIG_DIR.exists() else 'N/A'}), templates={templates_config_exists} ({FEEDBACK_TEMPLATES_CONFIG}), merged_reps={merged_reps_count}", flush=True)
    logger.info(
        "[pipeline] Feedback config check — exercise_config exists: %s, templates exists: %s, merged_reps: %d",
        exercise_config_exists,
        templates_config_exists,
        merged_reps_count,
    )
    
    # Convert sub_scores from 0-1 model-head range to 0-100 display range.
    # Error-probability heads (lockout, knee_error): sigmoid 0-1, higher=worse → (1-val)*100.
    # Linear quality heads (smoothness, control, depth, forward_lean, knee_tracking, …):
    # 0-1 quality, higher=better → val*100, clamped to [0, 100].
    _ERROR_KEYS = ("lockout", "knee_error")
    _LINEAR_KEYS = (
        "smoothness", "control", "forward_lean",
        "knee_tracking", "elbow_flare", "grip_ratio",
        "rom_top", "rom_bottom",
    )
    for _rep in merged_reps:
        _sub = _rep.get("sub_scores")
        if _sub is None:
            continue
        for _k in _ERROR_KEYS:
            _v = _sub.get(_k)
            if _v is not None:
                _sub[_k] = round((1 - _v) * 100, 1)
        for _k in _LINEAR_KEYS:
            _v = _sub.get(_k)
            if _v is not None:
                _sub[_k] = round(max(0.0, min(100.0, _v * 100.0)), 1)
        _smooth_relax(_sub)

    # Merge ST-GCN sub_scores with heuristic metric_scores (squat only).
    # Heuristic gets 60% weight, ST-GCN gets 40%.
    for _rep in merged_reps:
        _sub = _rep.get("sub_scores")
        _ms = _rep.get("metric_scores")
        if _sub is None or _ms is None:
            continue
        # forward_lean: heuristic forward_lean vs ST-GCN forward_lean
        h_fl = _ms.get("forward_lean")
        n_fl = _sub.get("forward_lean")
        if h_fl is not None and n_fl is not None:
            _sub["forward_lean"] = round(h_fl * 0.6 + n_fl * 0.4, 1)
        # knee_tracking: heuristic knee_valgus vs ST-GCN knee_tracking
        h_kv = _ms.get("knee_valgus")
        n_kt = _sub.get("knee_tracking")
        if h_kv is not None and n_kt is not None:
            _sub["knee_tracking"] = round(h_kv * 0.6 + n_kt * 0.4, 1)

    # Recompute per-rep bilstm_score from relaxed sub-scores
    for _rep in merged_reps:
        _sub = _rep.get("sub_scores")
        if _sub is None:
            continue
        sm = _sub.get("smoothness")
        ct = _sub.get("control")
        if sm is not None and ct is not None:
            _rep["bilstm_score"] = round(max(0.0, min(100.0, (sm + ct) / 2.0)), 2)

    for _rep in merged_reps:
        h = _rep.get("heuristic_score")
        s = _rep.get("stgcn_score")
        b = _rep.get("bilstm_score")
        if h is not None and s is not None and b is not None:
            _rep["neural_score"] = round((b + s + h) / 3.0, 2)
            _rep["neural_score_raw"] = None

    # ── Generate feedback (moved here so avg_score uses final neural_score values) ──
    if exercise_config_exists and templates_config_exists and merged_reps:
        print(f"[DIAGNOSTIC] All prereqs met. Initializing FeedbackEngine...", flush=True)
        try:
            exercise_config_path = _resolve_exercise_config(exercise)
            print(f"[DIAGNOSTIC] Loading exercise config from: {exercise_config_path}", flush=True)
            print(f"[DIAGNOSTIC] Loading templates from: {FEEDBACK_TEMPLATES_CONFIG}", flush=True)
            feedback_engine = FeedbackEngine(
                exercise_config_path=str(exercise_config_path),
                templates_path=str(FEEDBACK_TEMPLATES_CONFIG),
            )
            print(f"[DIAGNOSTIC] FeedbackEngine initialized successfully", flush=True)
            feedback_input: list[dict[str, Any]] = []

            for rep in merged_reps:
                sub_scores = rep.get("sub_scores") or {}

                normalized_sub_scores: dict[str, float | None] = {
                    "forward_lean": sub_scores.get("forward_lean"),
                    "knee_tracking": sub_scores.get("knee_tracking"),
                    "knee_valgus": None,
                    "smoothness": sub_scores.get("smoothness"),
                    "control": sub_scores.get("control"),
                    "lockout": sub_scores.get("lockout"),
                    "elbow_flare": sub_scores.get("elbow_flare"),
                    "grip_ratio": sub_scores.get("grip_ratio"),
                    "rom_top": sub_scores.get("rom_top"),
                    "rom_bottom": sub_scores.get("rom_bottom"),
                    "knee_error": sub_scores.get("knee_error"),
                }

                metric_scores = rep.get("metric_scores") or {}
                if normalized_sub_scores.get("forward_lean") is None:
                    normalized_sub_scores["forward_lean"] = metric_scores.get("forward_lean_deg")
                if normalized_sub_scores.get("knee_tracking") is None:
                    normalized_sub_scores["knee_tracking"] = metric_scores.get("knee_tracking_ratio")
                if normalized_sub_scores.get("knee_valgus") is None:
                    normalized_sub_scores["knee_valgus"] = metric_scores.get("knee_valgus")

                n_score = rep.get("neural_score") if rep.get("neural_score") is not None else rep.get("heuristic_score")

                feedback_input.append({
                    "rep_id": rep.get("rep_id"),
                    "neural_score": n_score,
                    "lockout": sub_scores.get("lockout"),
                    "elbow_flare": sub_scores.get("elbow_flare"),
                    "grip_ratio": sub_scores.get("grip_ratio"),
                    "rom_top": sub_scores.get("rom_top"),
                    "rom_bottom": sub_scores.get("rom_bottom"),
                    "smoothness": sub_scores.get("smoothness"),
                    "control": sub_scores.get("control"),
                    "knee_error": sub_scores.get("knee_error"),
                    "sub_scores": normalized_sub_scores,
                })

            print(f"[DIAGNOSTIC] Built feedback_input for {len(feedback_input)} reps. Calling generate_feedback...", flush=True)
            feedback_result = feedback_engine.generate_feedback(feedback_input, video_id=video_id)
            _deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            if _deepseek_key:
                try:
                    from core.exevision.feedback.llm_enhancer import LLMFeedbackEnhancer
                    global _llm_feedback_enhancer_cache
                    if _llm_feedback_enhancer_cache is None:
                        _llm_feedback_enhancer_cache = LLMFeedbackEnhancer(api_key=_deepseek_key)
                    feedback_result = _llm_feedback_enhancer_cache.enhance_result(feedback_result)
                    print("[DIAGNOSTIC] LLM feedback enhancement applied", flush=True)
                except Exception as _llm_exc:
                    logger.warning("LLM feedback enhancement failed, returning template-based feedback: %s", _llm_exc)
                try:
                    if _llm_feedback_enhancer_cache is not None:
                        feedback_result = _llm_feedback_enhancer_cache.enhance_session(feedback_result)
                        print("[DIAGNOSTIC] LLM session enhancement applied", flush=True)
                except Exception as _llm_session_exc:
                    logger.warning("LLM session enhancement failed, returning template-based coach_text: %s", _llm_session_exc)
            else:
                print("[DIAGNOSTIC] DEEPSEEK_API_KEY not set — using template-based feedback (no LLM enhancement)", flush=True)
            print(f"[DIAGNOSTIC] generate_feedback returned successfully with {len(feedback_result.reps)} reps", flush=True)
            feedback_payload = {
                "schema_version": "2.0",
                "exercise": feedback_result.exercise,
                "reps": [
                    {
                        "rep_id": item.rep_id,
                        "score": item.score,
                        "tier": item.tier,
                        "text": item.text,
                        "items": item.items,
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
        feedback_payload = _build_feedback_fallback(merged_reps, exercise)
        print(
            f"[DIAGNOSTIC] Fallback feedback generated: {len(feedback_payload.get('reps', []))} reps "
            f"(exercise_config={exercise_config_exists}, templates_config={templates_config_exists})",
            flush=True,
        )
    else:
        if not exercise_config_exists:
            print(f"[DIAGNOSTIC] Skipping feedback: exercise_config not found at {FEEDBACK_EXERCISE_CONFIG}", flush=True)
        if not templates_config_exists:
            print(f"[DIAGNOSTIC] Skipping feedback: templates_config not found at {FEEDBACK_TEMPLATES_CONFIG}", flush=True)
        if not merged_reps:
            print(f"[DIAGNOSTIC] Skipping feedback: no merged_reps to process (count={merged_reps_count})", flush=True)

    neural_scores = [r["neural_score"] for r in merged_reps if r["neural_score"] is not None]
    bilstm_scores = [r["bilstm_score"] for r in merged_reps if r["bilstm_score"] is not None]
    stgcn_scores = [r["stgcn_score"] for r in merged_reps if r["stgcn_score"] is not None]
    any_corrections = any(r.get("anchor_correction_applied", False) for r in merged_reps)

    base_url = os.environ.get("API_PUBLIC_URL", "http://localhost:8000").rstrip("/")
    job_id = workspace_root.parent.name

    logger.info(f"collect_results: workspace_root={workspace_root}, video_id={video_id}, exercise={exercise}")
    logger.info(f"workspace_root exists: {workspace_root.exists()}")
    logger.info(f"workspace_root type: {type(workspace_root)}")

    # Check if these directories exist
    dataset_dir = workspace_root / exercise / "dataset_videos_all"
    logger.info(f"dataset_dir={dataset_dir}, exists={dataset_dir.exists()}")

    viz_dir = workspace_root / exercise / "visualized_poses_clean"
    logger.info(f"viz_dir={viz_dir}, exists={viz_dir.exists()}")

    # Safe lookups
    videos_dict = {}

    try:
        if dataset_dir.exists():
            raw_matches = sorted(dataset_dir.glob(f"{video_id}.*"))
            logger.info(f"raw_matches: {raw_matches}")
            if raw_matches:
                videos_dict["raw"] = f"{base_url}/results/{job_id}/workspace/{raw_matches[0].relative_to(workspace_root).as_posix()}"
    except Exception as e:
        logger.error(f"Error finding raw video: {e}")

    try:
        if viz_dir.exists():
            # Prefer raw_unfiltered annotated video; fall back to any match
            _unfiltered_viz = viz_dir / "raw_unfiltered" / f"{video_id}_annotated.mp4"
            if _unfiltered_viz.exists():
                viz_matches = [_unfiltered_viz]
            else:
                viz_matches = sorted(viz_dir.rglob(f"{video_id}_annotated.mp4"))

            logger.info(f"viz_matches: {viz_matches}")
            logger.info(f"viz_dir exists: {viz_dir}, files: {[p.name for p in sorted(viz_dir.iterdir())]}")
            if viz_matches:
                viz_file = viz_matches[0]
                # Upload to Supabase (required for both local and production)
                viz_path = _upload_visualization_to_supabase(viz_file, job_id)
                if viz_path:
                    videos_dict["with_landmarks"] = viz_path
                    logger.info(f"Using uploaded visualization: {viz_path}")
                else:
                    logger.warning("Visualization upload failed; will not include in result")
    except Exception as e:
        logger.error(f"Error finding/uploading annotated video: {e}")

    logger.info(f"videos_dict: {videos_dict}")

    result = {
        "video_id": video_id,
        "view": _display_view(aqa.get("view")),
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
        "videos": videos_dict,
        "visualization_available": bool(videos_dict.get("with_landmarks")),
        "visualization_url": videos_dict.get("with_landmarks"),
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
    generate_viz: bool = True,
    exercise: str = "squat",
    rep_boundaries: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline synchronously for a single video.
    Called from a background thread/process by the API server.
    Returns the merged result dict.

    When rep_boundaries is provided (list of {start_sec, end_sec} from browser
    real-time rep detection), the temporal segmentation stage is given a
    --rep-boundaries path to override FSM-based rep grouping.
    """
    print(f"[DIAGNOSTIC] run_pipeline_sync START: job_id={job_id}, video={video_path.name}, stages={stages or 'default'}, mode={mode}, exercise={exercise}", flush=True)
    stages = stages or DEFAULT_STAGES
    run_root = RUNS_ROOT / job_id
    workspace_root = run_root / "workspace"
    logs_root = run_root / "logs"

    workspace_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    _prepare_workspace(workspace_root, video_path, exercise)
    video_id = video_path.stem

    # Write browser-captured rep boundaries when provided (Task 5: real-time boundaries bridge)
    rep_boundaries_path: str | None = None
    if rep_boundaries:
        boundaries_file = workspace_root / "rep_boundaries.json"
        boundaries_file.write_text(
            json.dumps({"rep_boundaries": rep_boundaries}, ensure_ascii=False), encoding="utf-8"
        )
        rep_boundaries_path = str(boundaries_file)
        logger.info("[pipeline] Wrote %d browser-captured rep boundaries to %s", len(rep_boundaries), boundaries_file)
    
    # Build exercise-specific stage specs
    stage_specs = _build_stage_specs(exercise)

    for key in stages:
        spec = stage_specs.get(key)
        if spec is None:
            raise ValueError(f"Unknown stage: '{key}'")
        if not spec.script.exists():
            raise FileNotFoundError(f"Stage script not found: {spec.script}")

        # Special handling for dual extraction: respect mode parameter
        if key == "extract_selected_features":
            if mode == "unfiltered":
                try:
                    _run_stage(key, spec.script, video_id, workspace_root, logs_root, "unfiltered", generate_viz, exercise)
                    logger.info("Stage extract_selected_features (unfiltered) completed.")
                except RuntimeError as e:
                    logger.warning(f"Unfiltered extraction failed: {e}")
                    return {
                        "video_id": video_id,
                        "exercise": exercise,
                        "extraction_failed": True,
                        "extraction_failure_reason": "unfiltered_failed",
                    }
            else:
                # Default: run filtered first (non-fatal), then unfiltered as safety net
                try:
                    _run_stage(key, spec.script, video_id, workspace_root, logs_root, "filtered", generate_viz, exercise)
                    logger.info("Stage extract_selected_features (filtered) completed.")
                except RuntimeError as e:
                    logger.warning(f"Filtered extraction failed (likely Poor quality): {e}")
                try:
                    _run_stage(key, spec.script, video_id, workspace_root, logs_root, "unfiltered", generate_viz, exercise)
                    logger.info("Stage extract_selected_features (unfiltered) completed.")
                except RuntimeError as e:
                    logger.warning(f"Unfiltered extraction also failed: {e}")
                    return {
                        "video_id": video_id,
                        "exercise": exercise,
                        "extraction_failed": True,
                        "extraction_failure_reason": "both_filtered_and_unfiltered_failed",
                    }
            # Validate that at least one run produced the expected artifact
            try:
                _validate_stage_output(key, workspace_root, video_id, exercise)
            except RuntimeError as exc:
                logger.warning(f"Feature extraction produced no output (non-fatal): {exc}")
                return {
                    "video_id": video_id,
                    "exercise": exercise,
                    "extraction_failed": True,
                    "extraction_failure_reason": "no_poses_detected",
                }
        else:
            # Neural fusion is optional: if it fails, we still return feedback based on heuristic scores.
            # All other stages are mandatory: failure propagates.
            try:
                _run_stage(key, spec.script, video_id, workspace_root, logs_root, mode, generate_viz, exercise, rep_boundaries_path)
                _validate_stage_output(key, workspace_root, video_id, exercise)
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
        # if key == "extract_selected_features":
        #     _delete_input_video(workspace_root, video_path.name, exercise)

    result = collect_results(workspace_root, video_id, exercise)

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
    # Visualization directories are retained if generate_viz=True so frontend can serve them.
    _cleanup_workspace(workspace_root, generate_viz=generate_viz, exercise=exercise)

    print(f"[DIAGNOSTIC] run_pipeline_sync END: job_id={job_id}, rep_count={result['rep_count']}, has_feedback={result['feedback'] is not None}, neural_available={result['neural_available']}", flush=True)
    return result
