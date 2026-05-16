#!/usr/bin/env python3
"""
ExeVision batch pipeline runner — stages 2.5 → 4 → 5 → 8 (scoring).

Runs the full heuristic pipeline for every video in a directory, or a
single video specified by --video-id.  Stage 9 (neural fusion) is opt-in
via --include-neural (single-video only).

Workspace layout created under --output-dir:
  {output_dir}/
    {exercise}/extracted_features_clean/...
    {exercise}/segmented_reps/...
    {exercise}/aqa_analysis_simple/...
    _logs/
      extract.log  classify.log  segment.log  score.log  [neural.log]

Usage examples:
  # Batch — all videos in a folder
  python core/exevision/stages/run_pipeline.py \\
      --video-dir D:/FitnessAQA/ohp/videos \\
      --exercise overhead_press --mode unfiltered --no-viz

  # Single video
  python core/exevision/stages/run_pipeline.py \\
      --video-dir D:/FitnessAQA/ohp/videos \\
      --video-id 80863_1 --exercise overhead_press

  # Resume from existing workspace (skip extraction)
  python core/exevision/stages/run_pipeline.py \\
      --video-dir D:/FitnessAQA/ohp/videos \\
      --output-dir batch_runs/20260513_120000 \\
      --exercise overhead_press --skip-extract
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# core/exevision/stages/ → core/exevision/ → core/ → repo root
_REPO = Path(__file__).resolve().parents[3]
_STAGES = _REPO / "core" / "exevision" / "stages"
_MODELS = _REPO / "models"
_RUNS = _REPO / "batch_runs"


def _make_env() -> dict:
    env = os.environ.copy()
    env["EXEVISION_MODEL_PATH"] = str(_MODELS / "runtime_pose_and_face" / "pose_landmarker_heavy.task")
    env["EXEVISION_FACE_MODEL_PATH"] = str(_MODELS / "runtime_pose_and_face" / "blaze_face_short_range.tflite")
    return env


# ── Stage runner ───────────────────────────────────────────────────────────────

def _run_stage(label: str, cmd: list[str], workspace: Path, log_path: Path, dry_run: bool) -> bool:
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 64}")

    if dry_run:
        print("  [dry-run — skipped]")
        return True

    t0 = time.time()
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(workspace),
        env=_make_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0
    combined = proc.stdout + proc.stderr

    # Write full log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")

    # Print tail to console (last 3000 chars to avoid flooding)
    tail = combined[-3000:] if len(combined) > 3000 else combined
    if tail.strip():
        print(tail)

    ok = proc.returncode == 0
    marker = "✓" if ok else "✗"
    try:
        display_log = log_path.relative_to(_REPO)
    except ValueError:
        display_log = log_path
    print(f"  {marker}  finished in {elapsed:.1f}s — log: {display_log}")
    return ok


# ── Model path helpers ─────────────────────────────────────────────────────────

def _model(name: str, exercise: str) -> Path:
    """Return exercise-specific model path, falling back to generic squat names."""
    specific = _MODELS / f"{name}_{exercise}.pt"
    if specific.exists():
        return specific
    generic_map = {
        "bilstm": "runtime_neural_squat/bilstm_finetuned.pt",
        "stgcn": "runtime_neural_squat/stgcn_finetuned.pt",
        "fusion": "runtime_neural_squat/fusion_layer.pt",
    }
    return _MODELS / generic_map.get(name, f"{name}.pt")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ExeVision pipeline (stages 2.5 → 4 → 5 → 8) on a video directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    inp = parser.add_argument_group("Input")
    inp.add_argument(
        "--video-dir", required=True,
        help="Directory containing input video files (searched recursively by Stage 2.5).",
    )
    inp.add_argument(
        "--video-id",
        help="Process only this one video ID (e.g. 80863_1). Omit to process all videos.",
    )

    # ── Pipeline config ────────────────────────────────────────────────────────
    cfg = parser.add_argument_group("Pipeline config")
    cfg.add_argument(
        "--exercise", default="squat",
        help="Exercise type: squat | overhead_press | seated_overhead_press (default: squat).",
    )
    cfg.add_argument(
        "--mode", default="filtered", choices=["filtered", "unfiltered"],
        help="Extraction mode passed to Stage 2.5 (default: filtered).",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output-dir",
        help=(
            "Workspace root directory.  All stage outputs land here.  "
            "Defaults to batch_runs/YYYYMMDD_HHMMSS/ inside the repo."
        ),
    )

    # ── Stage flags ────────────────────────────────────────────────────────────
    flags = parser.add_argument_group("Stage flags")
    flags.add_argument("--no-viz", action="store_true", help="Skip visualization video generation.")
    flags.add_argument("--no-report", action="store_true", help="Skip extraction analysis reports.")
    flags.add_argument("--include-poor", action="store_true", help="Save poor-quality extractions instead of skipping.")
    flags.add_argument("--max-videos", type=int, help="Cap number of videos extracted (Stage 2.5).")
    flags.add_argument("--workers", type=int, help="Extraction worker count (default: auto).")
    flags.add_argument(
        "--include-neural", action="store_true",
        help="Also run Stage 9 neural fusion after scoring (single --video-id only).",
    )

    # ── Skip flags ─────────────────────────────────────────────────────────────
    skip = parser.add_argument_group("Skip stages (for re-runs)")
    skip.add_argument("--skip-extract", action="store_true", help="Skip Stage 2.5 (re-use existing extracted features).")
    skip.add_argument("--skip-classify", action="store_true", help="Skip Stage 4 view classification.")
    skip.add_argument("--skip-segment", action="store_true", help="Skip Stage 5 temporal segmentation.")
    skip.add_argument("--skip-score", action="store_true", help="Skip Stage 8 scoring.")

    # ── Misc ───────────────────────────────────────────────────────────────────
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")

    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    if not video_dir.exists():
        sys.exit(f"ERROR: --video-dir does not exist: {video_dir}")

    # Resolve workspace
    if args.output_dir:
        workspace = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace = _RUNS / ts
    workspace.mkdir(parents=True, exist_ok=True)
    logs_dir = workspace / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ex = args.exercise
    vid = args.video_id

    print(f"\n{'=' * 64}")
    print(f"  ExeVision Pipeline Runner")
    print(f"  Exercise  : {ex}")
    print(f"  Mode      : {args.mode}")
    print(f"  Video dir : {video_dir}")
    if vid:
        print(f"  Video ID  : {vid}")
    else:
        print(f"  Videos    : all in directory")
    print(f"  Workspace : {workspace}")
    print(f"  Dry run   : {'yes' if args.dry_run else 'no'}")
    print(f"{'=' * 64}")

    passed: list[str] = []
    failed: list[str] = []

    def record(label: str, ok: bool) -> None:
        (passed if ok else failed).append(label)

    # ── Stage 2.5 — Pose Extraction ────────────────────────────────────────────
    if not args.skip_extract:
        cmd = [
            sys.executable,
            _STAGES / "extract_selected_features.py",
            args.mode,
            "--exercise", ex,
            "--video-dir", str(video_dir),
        ]
        if vid:
            cmd += ["--video-id", vid]
        if args.no_viz:
            cmd.append("--no-viz")
        if args.no_report:
            cmd.append("--no-report")
        if args.max_videos:
            cmd += ["--max-videos", str(args.max_videos)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.include_poor:
            cmd.append("--include-poor")

        ok = _run_stage(
            "Stage 2.5 — Pose Extraction",
            cmd, workspace, logs_dir / "extract.log", args.dry_run,
        )
        record("2.5_extract", ok)
        if not ok:
            print("\n⚠  Extraction failed — aborting pipeline.\n")
            sys.exit(1)

    # ── Stage 4 — View Classification ─────────────────────────────────────────
    if not args.skip_classify:
        cmd = [sys.executable, _STAGES / "classify_views.py", "--exercise", ex]
        if vid:
            cmd += ["--video-id", vid]

        ok = _run_stage(
            "Stage 4 — View Classification",
            cmd, workspace, logs_dir / "classify.log", args.dry_run,
        )
        record("4_classify", ok)

    # ── Stage 5 — Temporal Segmentation ───────────────────────────────────────
    if not args.skip_segment:
        cmd = [sys.executable, _STAGES / "temporal_segmentation.py", "--exercise", ex]
        if vid:
            cmd += ["--video-id", vid]
        if args.no_viz:
            cmd.append("--no-viz")

        ok = _run_stage(
            "Stage 5 — Temporal Segmentation",
            cmd, workspace, logs_dir / "segment.log", args.dry_run,
        )
        record("5_segment", ok)

    # ── Stage 8 — Scoring ─────────────────────────────────────────────────────
    if not args.skip_score:
        score_target = vid if vid else "*"
        cmd = [sys.executable, _STAGES / "scoring.py", score_target, "--exercise", ex]

        ok = _run_stage(
            "Stage 8 — Scoring",
            cmd, workspace, logs_dir / "score.log", args.dry_run,
        )
        record("8_score", ok)

    # ── Stage 9 — Neural Fusion (opt-in) ──────────────────────────────────────
    if args.include_neural:
        if not vid:
            print("\n⚠  --include-neural requires --video-id (batch neural not supported). Skipping.")
        else:
            cmd = [
                sys.executable, _STAGES / "neural_fusion_inference.py",
                "--video-id", vid,
                "--exercise", ex,
                "--bilstm-ckpt", str(_model("bilstm", ex)),
                "--stgcn-ckpt", str(_model("stgcn", ex)),
                "--fusion-ckpt", str(_model("fusion", ex)),
                "--quality-tier", "raw_unfiltered",
            ]
            ok = _run_stage(
                "Stage 9 — Neural Fusion",
                cmd, workspace, logs_dir / "neural.log", args.dry_run,
            )
            record("9_neural", ok)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  Pipeline complete")
    print(f"  ✓ Passed : {', '.join(passed) or '—'}")
    if failed:
        print(f"  ✗ Failed : {', '.join(failed)}")
    print(f"  Workspace: {workspace}")
    print(f"  Logs     : {logs_dir}")
    print(f"{'=' * 64}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
