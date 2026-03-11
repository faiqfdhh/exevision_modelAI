import os
import re
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    script_path: Path
    args: tuple[str, ...]
    output_paths: tuple[str, ...]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "pipeline_ui_runs"
SHARED_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_heavy.task"


STAGES: tuple[Stage, ...] = (
    Stage(
        key="extract_selected_features",
        label="2.5 Extract Selected Features",
        script_path=PROJECT_ROOT / "scripts" / "2.5_extract_selected_features.py",
        args=(),
        output_paths=(
            "squat/extracted_features_clean",
            "squat/visualized_poses_clean",
            "squat/analysis_reports",
        ),
    ),
    Stage(
        key="classify_views",
        label="4 Classify Views",
        script_path=PROJECT_ROOT / "scripts" / "4_classify_views.py",
        args=(),
        output_paths=("squat/extracted_features_clean",),
    ),
    Stage(
        key="temporal_segmentation",
        label="5 Temporal Segmentation",
        script_path=PROJECT_ROOT / "scripts" / "5_temporal_segmentation.py",
        args=(),
        output_paths=("squat/segmented_reps", "squat/visualized_segmentation"),
    ),
    Stage(
        key="scoring",
        label="8 Scoring",
        script_path=PROJECT_ROOT / "scripts" / "8_scoring.py",
        args=("*",),
        output_paths=("squat/aqa_analysis_simple",),
    ),
    Stage(
        key="analyze_results",
        label="9 Analyze Results",
        script_path=PROJECT_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py",
        args=(),
        output_paths=("squat/aqa_analysis_simple/analysis_visualizations",),
    ),
)


def ordered_stages(stage_keys: list[str]) -> list[Stage]:
    selected = set(stage_keys)
    return [stage for stage in STAGES if stage.key in selected]


def get_view_thresholds(view: str) -> dict[str, dict[str, float | bool]]:
    view_lower = str(view).lower()

    default_thresholds = {
        "knee_valgus": {"good": 0.95, "bad": 0.75, "higher_is_better": True},
        "forward_lean": {"good": 20.0, "bad": 45.0, "higher_is_better": False},
        "depth": {"good": 95.0, "bad": 125.0, "higher_is_better": False},
        "squat_depth": {"good": 0.1, "bad": -0.1, "higher_is_better": True},
    }

    if "side" in view_lower and "front" not in view_lower and "back" not in view_lower:
        return {
            "knee_valgus": {"good": 0.95, "bad": 0.70, "higher_is_better": True},
            "forward_lean": {"good": 30.0, "bad": 60.0, "higher_is_better": False},
            "depth": {"good": 70.0, "bad": 120.0, "higher_is_better": False},
            "squat_depth": {"good": 0.15, "bad": -0.05, "higher_is_better": True},
        }

    if view_lower in ["front", "back"]:
        return {
            "knee_valgus": {"good": 0.97, "bad": 0.80, "higher_is_better": True},
            "forward_lean": {"good": 25.0, "bad": 50.0, "higher_is_better": False},
            "depth": {"good": 100.0, "bad": 130.0, "higher_is_better": False},
            "squat_depth": {"good": 0.08, "bad": -0.08, "higher_is_better": True},
        }

    if "front_side" in view_lower or "front-side" in view_lower:
        return {
            "knee_valgus": {"good": 0.95, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 18.0, "bad": 40.0, "higher_is_better": False},
            "depth": {"good": 95.0, "bad": 122.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.05, "higher_is_better": True},
        }

    if "back_side" in view_lower or "back-side" in view_lower:
        return {
            "knee_valgus": {"good": 1.2, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 35.0, "bad": 50.0, "higher_is_better": False},
            "depth": {"good": 95.0, "bad": 122.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.05, "higher_is_better": True},
        }

    return default_thresholds


def score_severity(metric_score: float) -> str:
    if metric_score >= 90:
        return "Good"
    if metric_score >= 75:
        return "Slight"
    if metric_score >= 50:
        return "Moderate"
    if metric_score >= 25:
        return "Major"
    return "Severe"


class PipelineRunnerUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ExeVision Pipeline Runner")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        self.input_mode_var = tk.StringVar(value="video")
        self.mode_var = tk.StringVar(value="full")
        self.processing_mode_var = tk.StringVar(value="filtered")
        self.dataset_var = tk.StringVar(value=str(PROJECT_ROOT / "squat" / "dataset_videos_all"))
        self.video_var = tk.StringVar(value="")
        self.run_name_var = tk.StringVar(value=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
        self.single_stage_var = tk.StringVar(value=STAGES[0].key)
        self.output_video_var = tk.StringVar(value="")
        self.preview_status_var = tk.StringVar(value="Run pipeline to preview annotated output.")
        self.overall_progress_var = tk.DoubleVar(value=0.0)
        self.stage_progress_var = tk.DoubleVar(value=0.0)
        self.view_type_var = tk.StringVar(value="")
        
        # Segmentation Data
        self.current_frame_phases: list[str] | None = None
        self.current_reps: int = 0
        self.current_view: str = ""
        self.current_analysis_data: dict | None = None
        self.current_analysis_summary_data: dict | None = None
        self.current_score_data: dict | None = None
        self.current_run_root: Path | None = None

        self.stage_checks: dict[str, tk.BooleanVar] = {
            stage.key: tk.BooleanVar(value=True) for stage in STAGES
        }

        self.preview_video_map: dict[str, tuple[Path, Path | None]] = {}
        self.preview_overlay_cap: cv2.VideoCapture | None = None
        self.preview_original_cap: cv2.VideoCapture | None = None
        self.preview_thread: threading.Thread | None = None
        self.preview_stop_event = threading.Event()
        self.preview_is_playing = False
        self.preview_fps = 30.0
        self._preview_overlay_photo = None
        self._preview_original_photo = None

        self.current_process: subprocess.Popen | None = None
        self.stop_requested = False
        self.pipeline_running = False

        self._build_layout()
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        # Top bar with window controls
        title_bar = ttk.Frame(self.root)
        title_bar.pack(fill=tk.X, padx=8, pady=(8, 4))
        title_bar.columnconfigure(0, weight=1)
        
        ttk.Label(title_bar, text="ExeVision Pipeline Runner", font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT)
        
        button_frame_top = ttk.Frame(title_bar)
        button_frame_top.pack(side=tk.RIGHT)
        ttk.Button(button_frame_top, text="−", width=2, command=self.root.iconify).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(button_frame_top, text="✕", width=2, command=self._on_close).pack(side=tk.LEFT)
        
        # Main container with two columns
        main_container = ttk.Frame(self.root, padding=8)
        main_container.pack(fill=tk.BOTH, expand=True)
        main_container.columnconfigure(0, weight=0, minsize=320)  # Left column: settings (fixed width)
        main_container.columnconfigure(1, weight=1)  # Right column: preview (expandable)
        main_container.rowconfigure(0, weight=1)

        # LEFT COLUMN: Settings and Controls
        left_panel = ttk.Frame(main_container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_panel.columnconfigure(0, weight=1)

        # Run Name
        ttk.Label(left_panel, text="Run Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(left_panel, textvariable=self.run_name_var, width=30).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Dataset Input Folder
        ttk.Label(left_panel, text="Dataset Input").grid(row=1, column=0, sticky="w", pady=(8, 0))
        dataset_frame = ttk.Frame(left_panel)
        dataset_frame.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(8, 0))
        dataset_frame.columnconfigure(0, weight=1)
        self.dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_var, width=20)
        self.dataset_entry.grid(row=0, column=0, sticky="ew")
        self.dataset_browse_button = ttk.Button(dataset_frame, text="B", width=2, command=self._pick_dataset)
        self.dataset_browse_button.grid(row=0, column=1, padx=(4, 0))

        # Input Source
        source_frame = ttk.LabelFrame(left_panel, text="Input Source", padding=8)
        source_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        source_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            source_frame,
            text="Dataset",
            variable=self.input_mode_var,
            value="dataset",
            command=self._sync_input_controls,
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            source_frame,
            text="Single Video",
            variable=self.input_mode_var,
            value="video",
            command=self._sync_input_controls,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        video_frame = ttk.Frame(source_frame)
        video_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(4, 0))
        video_frame.columnconfigure(0, weight=1)
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_var, width=16)
        self.video_entry.grid(row=0, column=0, sticky="ew")
        self.video_browse_button = ttk.Button(video_frame, text="B", width=2, command=self._pick_video)
        self.video_browse_button.grid(row=0, column=1, padx=(4, 0))

        # Pipeline Mode
        mode_frame = ttk.LabelFrame(left_panel, text="Pipeline Mode", padding=8)
        mode_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        ttk.Radiobutton(mode_frame, text="Full", variable=self.mode_var, value="full", command=self._sync_mode_controls).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Individual", variable=self.mode_var, value="single", command=self._sync_mode_controls).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Radiobutton(mode_frame, text="Custom", variable=self.mode_var, value="custom", command=self._sync_mode_controls).grid(row=2, column=0, sticky="w", pady=(4, 0))

        self.single_stage_combo = ttk.Combobox(
            mode_frame,
            state="readonly",
            values=[stage.key for stage in STAGES],
            width=20,
        )
        self.single_stage_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        self.single_stage_combo.current(0)

        custom_frame = ttk.Frame(mode_frame)
        custom_frame.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        self.custom_checks: list[ttk.Checkbutton] = []
        for idx, stage in enumerate(STAGES):
            check = ttk.Checkbutton(custom_frame, text=stage.key, variable=self.stage_checks[stage.key])
            check.grid(row=idx, column=0, sticky="w", pady=(2, 2))
            self.custom_checks.append(check)

        # Processing Mode (for extraction stage)
        processing_frame = ttk.LabelFrame(left_panel, text="Extraction Mode", padding=8)
        processing_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Radiobutton(processing_frame, text="Filtered (default)", variable=self.processing_mode_var, value="filtered").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(processing_frame, text="Unfiltered (raw)", variable=self.processing_mode_var, value="unfiltered").grid(row=1, column=0, sticky="w", pady=(4, 0))

        # Start/Stop Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.start_button = ttk.Button(button_frame, text="▶ Start", command=self._start, width=14)
        self.start_button.pack(side=tk.TOP, padx=(0, 0), fill=tk.X)
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", command=self._stop_pipeline_safe, state="disabled", width=14)
        self.stop_button.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # Progress Bars
        progress_frame = ttk.LabelFrame(left_panel, text="Progress", padding=6)
        progress_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.overall_progress = ttk.Progressbar(progress_frame, variable=self.overall_progress_var, maximum=100, mode="determinate")
        self.overall_progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, text="Pipeline").grid(row=0, column=1, sticky="w", padx=(6, 0))

        self.stage_progress = ttk.Progressbar(progress_frame, variable=self.stage_progress_var, maximum=100, mode="determinate")
        self.stage_progress.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(progress_frame, text="Stage").grid(row=1, column=1, sticky="w", padx=(6, 0))

        left_panel.columnconfigure(1, weight=1)
        left_panel.rowconfigure(6, weight=1)

        # RIGHT COLUMN: Logs and Preview
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=0)  # Logs
        right_panel.rowconfigure(1, weight=1)  # Preview

        # Logs
        log_label = ttk.Label(right_panel, text="Process Log", font=("TkDefaultFont", 10, "bold"))
        log_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

        log_frame = ttk.Frame(right_panel)
        log_frame.grid(row=0, column=0, sticky="ew")
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=8, width=50)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Preview
        preview_frame = ttk.LabelFrame(right_panel, text="Annotated Video Preview", padding=8)
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(2, weight=1)

        preview_controls = ttk.Frame(preview_frame)
        preview_controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        preview_controls.columnconfigure(0, weight=1)

        self.output_video_combo = ttk.Combobox(
            preview_controls,
            textvariable=self.output_video_var,
            state="readonly",
            width=40,
        )
        self.output_video_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(preview_controls, text="Load", command=self._load_selected_preview, width=8).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(preview_controls, text="Play", command=self._toggle_preview_playback, width=8).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(preview_controls, text="Stop", command=self._stop_preview, width=8).grid(row=0, column=3, padx=(4, 0))
        ttk.Button(preview_controls, text="Player", command=self._open_in_default_player, width=8).grid(row=0, column=4, padx=(4, 0))
        ttk.Button(preview_controls, text="Report", command=self._show_report, width=8).grid(row=0, column=5, padx=(4, 0))

        self.original_title_label = ttk.Label(preview_frame, text="Original")
        self.original_title_label.grid(row=1, column=0, sticky="w")
        self.overlay_title_label = ttk.Label(preview_frame, text="Annotated")
        self.overlay_title_label.grid(row=1, column=1, sticky="w")

        self.original_image_label = ttk.Label(preview_frame, anchor="center", text="No original preview loaded.")
        self.original_image_label.grid(row=2, column=0, sticky="nsew", pady=(6, 4), padx=(0, 6))

        self.overlay_image_label = ttk.Label(preview_frame, anchor="center", text="No annotated preview loaded.")
        self.overlay_image_label.grid(row=2, column=1, sticky="nsew", pady=(6, 4), padx=(6, 0))

        ttk.Label(preview_frame, textvariable=self.preview_status_var).grid(row=3, column=0, sticky="w")
        ttk.Label(preview_frame, textvariable=self.view_type_var, font=("TkDefaultFont", 10, "bold")).grid(row=3, column=1, sticky="e")

        score_frame = ttk.LabelFrame(preview_frame, text="Scoring Results", padding=8)
        score_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        score_frame.columnconfigure(0, weight=1)
        score_frame.rowconfigure(0, weight=1)

        self.score_text = tk.Text(score_frame, wrap=tk.WORD, height=12, width=60)
        self.score_text.grid(row=0, column=0, sticky="nsew")
        self.score_text.insert(tk.END, "Run the pipeline to see squat scoring results here.")
        self.score_text.configure(state="disabled")

        score_scrollbar = ttk.Scrollbar(score_frame, orient=tk.VERTICAL, command=self.score_text.yview)
        score_scrollbar.grid(row=0, column=1, sticky="ns")
        self.score_text.config(yscrollcommand=score_scrollbar.set)

        self._sync_input_controls()
        self._sync_mode_controls()

    def _pick_dataset(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(PROJECT_ROOT))
        if selected:
            self.dataset_var.set(selected)

    def _pick_video(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(PROJECT_ROOT),
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.flv"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.video_var.set(selected)

    def _sync_input_controls(self) -> None:
        use_dataset = self.input_mode_var.get() == "dataset"
        dataset_state = "normal" if use_dataset else "disabled"
        video_state = "normal" if not use_dataset else "disabled"

        self.dataset_entry.configure(state=dataset_state)
        self.dataset_browse_button.configure(state=dataset_state)
        self.video_entry.configure(state=video_state)
        self.video_browse_button.configure(state=video_state)

    def _sync_mode_controls(self) -> None:
        mode = self.mode_var.get()
        self.single_stage_combo.configure(state="readonly" if mode == "single" else "disabled")

        state = "normal" if mode == "custom" else "disabled"
        for check in self.custom_checks:
            check.configure(state=state)

    def _log(self, text: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self._log, text)
            return
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _resolve_stage_selection(self) -> list[Stage]:
        mode = self.mode_var.get()

        if mode == "full":
            return list(STAGES)

        if mode == "single":
            selected_text = self.single_stage_combo.get()
            selected_key = selected_text.split(":", 1)[0].strip()
            return ordered_stages([selected_key])

        selected_keys = [k for k, v in self.stage_checks.items() if v.get()]
        return ordered_stages(selected_keys)

    def _start(self) -> None:
        stages = self._resolve_stage_selection()
        input_mode = self.input_mode_var.get()
        dataset_path: Path | None = None
        video_path: Path | None = None

        if not stages:
            messagebox.showerror("No stages selected", "Select at least one stage to run.")
            return

        if input_mode == "dataset":
            dataset_path = Path(self.dataset_var.get()).resolve()
            if not dataset_path.exists() or not dataset_path.is_dir():
                messagebox.showerror("Invalid dataset", f"Dataset folder not found:\n{dataset_path}")
                return
        else:
            video_path = Path(self.video_var.get()).resolve()
            valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".flv"}
            if not video_path.exists() or not video_path.is_file():
                messagebox.showerror("Invalid video", f"Video file not found:\n{video_path}")
                return
            if video_path.suffix.lower() not in valid_extensions:
                messagebox.showerror("Invalid video", "Select a supported video file (.mp4, .mov, .avi, .mkv, .flv).")
                return

        run_name = self.run_name_var.get().strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_root = RUNS_ROOT / run_name
        self.current_run_root = run_root

        self.stop_requested = False
        self.pipeline_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_overall_progress(0)
        self._set_stage_progress(0)
        self._log(f"▶ Starting run: {run_name}")
        self._set_overall_progress(0)
        self._set_stage_progress(0)
        self._log(f"▶ Starting run: {run_name}")
        self._set_preview_outputs([])
        self._clear_score_display("Run in progress. Scoring results will appear here when available.")
        
        # STOP and RELEASE the preview before we do anything else
        # This ensures the video file is not locked by cv2.VideoCapture
        self._stop_preview(release_only=True)

        if video_path is not None:
            self._log(f"Input video: {video_path}")
        else:
            self._log(f"Input dataset: {dataset_path}")

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(run_root, stages, dataset_path, video_path),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_thread(
        self,
        run_root: Path,
        stages: list[Stage],
        dataset_path: Path | None,
        video_path: Path | None,
    ) -> None:
        try:
            workspace_root = run_root / "workspace"
            stage_outputs_root = run_root / "stage_outputs"
            logs_root = run_root / "logs"

            for path in (workspace_root, stage_outputs_root, logs_root):
                if path.exists() or path.is_symlink():
                    if path.is_symlink():
                        path.unlink()
                    else:
                        shutil.rmtree(path)

            self._prepare_workspace(workspace_root, dataset_path=dataset_path, video_path=video_path)
            stage_outputs_root.mkdir(parents=True, exist_ok=True)
            logs_root.mkdir(parents=True, exist_ok=True)

            self._log(f"Workspace: {workspace_root}")
            self._log("Running stages in order:\n  - " + "\n  - ".join(stage.label for stage in stages))

            for index, stage in enumerate(stages, start=1):
                if self.stop_requested:
                    self._log("\n⚠️  Pipeline stopped by user.")
                    break

                self.root.after(0, self._set_stage_progress, 0)
                self._log(f"\n[{index}/{len(stages)}] Running: {stage.label}")
                self._run_stage(stage, workspace_root, logs_root, video_path=video_path)

                stage_folder = stage_outputs_root / f"{index:02d}_{stage.key}"
                self._snapshot_stage_outputs(stage, workspace_root, stage_folder)
                self._log(f"Saved stage snapshot: {stage_folder}")
                self.root.after(0, self._set_stage_progress, 100)
                self.root.after(0, self._set_overall_progress, index * 100 / max(len(stages), 1))

            selected_stem = video_path.stem if video_path is not None else None
            annotated_videos = self._find_annotated_videos(run_root, selected_stem)
            self.root.after(0, lambda: self._set_preview_outputs(annotated_videos))
            if selected_stem:
                score_path = self._find_score_json(run_root, selected_stem)
                if score_path is not None:
                    self.root.after(0, lambda p=score_path: self._load_score_data(p))

            if self.stop_requested:
                self._log(f"\nPartial results saved under: {run_root}")
            else:
                self._log("\n✅ Pipeline run complete.")
                self._log(f"All outputs saved under: {run_root}")
                self.root.after(0, lambda: messagebox.showinfo("Completed", f"Run completed successfully.\n\n{run_root}"))

        except Exception as exc:
            self._log(f"\n❌ Run failed: {exc}")
            self.root.after(0, lambda: messagebox.showerror("Pipeline failed", str(exc)))
        finally:
            self.pipeline_running = False
            if self.stop_requested:
                self.root.after(0, self._set_stage_progress, 0)
            self.root.after(0, self._update_button_states)

    def _prepare_workspace(
        self,
        workspace_root: Path,
        dataset_path: Path | None,
        video_path: Path | None,
    ) -> None:
        squat_dir = workspace_root / "squat"

        (workspace_root / "squat").mkdir(parents=True, exist_ok=True)

        dataset_target = squat_dir / "dataset_videos_all"
        if dataset_target.exists() or dataset_target.is_symlink():
            try:
                dataset_target.unlink()
            except Exception:
                try:
                    os.rmdir(dataset_target)
                except Exception:
                    shutil.rmtree(dataset_target)

        self._log("Preparing run workspace...")
        if video_path is not None:
            dataset_target.mkdir(parents=True, exist_ok=True)
            video_target = dataset_target / video_path.name
            shutil.copy2(video_path, video_target)
            self._log(f"Copied single video: {video_path} -> {video_target}")
        elif dataset_path is not None:
            self._safe_link_or_copy(dataset_path, dataset_target)
        else:
            raise RuntimeError("No valid input source provided.")

        if not SHARED_MODEL_PATH.exists():
            raise RuntimeError(f"Shared model file not found: {SHARED_MODEL_PATH}")
        self._log(f"Using shared pose model: {SHARED_MODEL_PATH}")

        # Ensure local copy of analyze_results.py exists inside workspace
        analyze_src = PROJECT_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py"
        analyze_dst = workspace_root / "squat" / "aqa_analysis_simple" / "analyze_results.py"
        analyze_dst.parent.mkdir(parents=True, exist_ok=True)
        if analyze_src.exists():
            shutil.copy2(analyze_src, analyze_dst)

    def _safe_link_or_copy(self, src: Path, dst: Path) -> None:
        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
            self._log(f"Linked: {src} -> {dst}")
            return
        except Exception:
            pass

        if os.name == 'nt' and src.is_dir():
            try:
                subprocess.run(
                    ["cmd.exe", "/c", "mklink", "/J", str(dst), str(src)],
                    check=True,
                    capture_output=True
                )
                self._log(f"Junctioned: {src} -> {dst}")
                return
            except Exception:
                pass

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        self._log(f"Copied: {src} -> {dst}")

    def _run_stage(
        self,
        stage: Stage,
        workspace_root: Path,
        logs_root: Path,
        video_path: Path | None = None,
    ) -> None:
        script_to_run = stage.script_path
        if stage.key == "analyze_results":
            script_to_run = workspace_root / "squat" / "aqa_analysis_simple" / "analyze_results.py"

        if not script_to_run.exists():
            raise RuntimeError(f"Script not found for stage '{stage.label}': {script_to_run}")

        # Add processing mode for extraction stage
        stage_args = list(stage.args)
        if stage.key == "extract_selected_features":
            processing_mode = self.processing_mode_var.get()
            stage_args = [processing_mode] + stage_args
        elif stage.key == "scoring" and video_path is not None:
            stage_args = [video_path.stem]

        cmd = [sys.executable, str(script_to_run), *stage_args]
        log_file = logs_root / f"{stage.key}.log"
        env = os.environ.copy()
        env["EXEVISION_MODEL_PATH"] = str(SHARED_MODEL_PATH)

        with open(log_file, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd,
                cwd=str(workspace_root),
            env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            self.current_process = process

            if process.stdout is None:
                raise RuntimeError("Failed to capture process output")

            try:
                for line in process.stdout:
                    if self.stop_requested:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        self.current_process = None
                        raise RuntimeError("Pipeline stopped by user.")

                    line = line.rstrip("\n")
                    self._handle_stage_output_line(line)
                    f.write(line + "\n")

                return_code = process.wait()
            finally:
                self.current_process = None

        if return_code != 0:
            raise RuntimeError(
                f"Stage failed: {stage.label} (exit code {return_code}). Check log: {log_file}"
            )

    def _handle_stage_output_line(self, line: str) -> None:
        percent_match = re.search(r"(\d{1,3})%", line)
        if percent_match:
            value = min(100, max(0, int(percent_match.group(1))))
            self.root.after(0, self._set_stage_progress, value)
            return

        lowered = line.lower()
        important_tokens = (
            "error",
            "failed",
            "warning",
            "skipped",
            "completed",
            "success",
            "unreliable video",
            "processing summary",
            "run complete",
            "view rejected",
            "classified as",
            "unreliable view",
            "overall score",
            "score:",
            "summary:",
        )
        if any(token in lowered for token in important_tokens):
            self._log(line)

    def _snapshot_stage_outputs(self, stage: Stage, workspace_root: Path, stage_folder: Path) -> None:
        if stage_folder.exists():
            shutil.rmtree(stage_folder)
        stage_folder.mkdir(parents=True, exist_ok=True)

        copied_any = False
        for rel_output in stage.output_paths:
            src = workspace_root / rel_output
            if not src.exists():
                continue

            dst = stage_folder / rel_output
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            copied_any = True

        manifest = stage_folder / "stage_manifest.txt"
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(f"stage_key={stage.key}\n")
            f.write(f"stage_label={stage.label}\n")
            f.write("expected_outputs=\n")
            for rel in stage.output_paths:
                f.write(f"  - {rel}\n")
            f.write(f"outputs_found={copied_any}\n")

    def _find_annotated_videos(self, run_root: Path, selected_stem: str | None) -> list[Path]:
        search_roots = [
            run_root / "workspace" / "squat" / "visualized_poses_clean",
            run_root / "workspace" / "squat" / "visualized_segmentation",
            run_root / "stage_outputs",
        ]

        found: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            found.extend(sorted(root.rglob("*_annotated.mp4")))
            found.extend(sorted(root.rglob("*_segmented.mp4")))
            found.extend(sorted(root.rglob("*_phases.mp4")))

        if selected_stem:
            # Match annotated, segmented, or phases video
            filtered = [
                p for p in found 
                if p.stem == f"{selected_stem}_annotated" or p.stem == f"{selected_stem}_segmented" or p.stem == f"{selected_stem}_phases"
            ]
            if filtered:
                return filtered

        return found

    def _set_preview_outputs(self, video_paths: list[Path]) -> None:
        self.preview_video_map = {}
        for path in video_paths:
            original = self._find_original_video_for_overlay(path)
            label = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
            self.preview_video_map[label] = (path, original)

        values = list(self.preview_video_map.keys())
        # Sort to prioritize _phases, then _segmented, then _annotated
        def sort_key(name):
            if "_phases" in name: return 0
            if "_segmented" in name: return 1
            return 2
        values.sort(key=sort_key)
        run_root = Path(RUNS_ROOT) / self.run_name_var.get().strip()
        dataset_dir = run_root / "workspace" / "squat" / "dataset_videos_all"
        all_stems = set()
        if dataset_dir.exists():
            for p in dataset_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".flv"}:
                    all_stems.add(p.stem)

        for stem in all_stems:
            found = False
            for val in values:
                if val.startswith(stem):
                    found = True
                    break
            if not found:
                values.append(f"{stem} [Missing Output]")

        self.output_video_combo["values"] = values

        if values:
            self.output_video_var.set(values[0])
            self.preview_status_var.set(f"Found {len(values)} input video(s).")
            self._load_selected_preview()
        else:
            self.output_video_var.set("")
            self.original_image_label.configure(image="", text="No original preview loaded.")
            self.overlay_image_label.configure(image="", text="No annotated preview loaded.")
            self._preview_original_photo = None
            self._preview_overlay_photo = None
            self.preview_status_var.set("No video output generated. (Perhaps the view was 'unknown' or skipped)")

    def _find_score_json(self, run_root: Path, video_id: str) -> Path | None:
        search_roots = [
            run_root / "workspace" / "squat" / "aqa_analysis_simple",
            run_root / "stage_outputs",
        ]

        for root in search_roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(f"{video_id}_aqa_simple.json"))
            if matches:
                return matches[0]
        return None

    def _find_analysis_summary_json(self, run_root: Path, video_id: str) -> Path | None:
        search_roots = [
            run_root / "workspace" / "squat" / "aqa_analysis_simple" / "analysis_visualizations",
            run_root / "stage_outputs",
        ]

        for root in search_roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(f"{video_id}_analysis_summary.json"))
            if matches:
                return matches[0]
        return None

    def _find_run_root_for_path(self, path: Path) -> Path | None:
        for parent in path.parents:
            if parent.parent.name == "pipeline_ui_runs":
                return parent
        return None

    def _set_score_display(self, text: str) -> None:
        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", tk.END)
        self.score_text.insert(tk.END, text)
        self.score_text.configure(state="disabled")

    def _clear_score_display(self, text: str = "No scoring results available.") -> None:
        self.current_analysis_summary_data = None
        self.current_score_data = None
        self._set_score_display(text)

    def _metric_specs(self) -> dict[str, dict[str, str]]:
        return {
            "knee_valgus": {
                "label": "Knee tracking",
                "source_key": "knee_valgus",
                "unit": "ratio",
                "evaluation": "direct",
            },
            "forward_lean": {
                "label": "Forward lean",
                "source_key": "forward_lean",
                "unit": "deg",
                "evaluation": "absolute",
            },
            "depth": {
                "label": "Depth by knee angle",
                "source_key": "min_knee_angle",
                "unit": "deg",
                "evaluation": "direct",
            },
            "squat_depth": {
                "label": "Bottom depth",
                "source_key": "squat_depth",
                "unit": "normalized",
                "evaluation": "direct",
            },
        }

    def _format_rule_value(self, value: float, unit: str) -> str:
        if unit == "deg":
            return f"{value:.2f} deg"
        return f"{value:.3f}"

    def _format_threshold_rule(self, threshold: dict[str, float | bool], unit: str) -> str:
        good = self._format_rule_value(float(threshold["good"]), unit)
        bad = self._format_rule_value(float(threshold["bad"]), unit)
        if bool(threshold["higher_is_better"]):
            return f"good if >= {good}; bad if <= {bad}"
        return f"good if <= {good}; bad if >= {bad}"

    def _build_metric_diagnostic(self, metric_name: str, rep: dict, view: str) -> dict | None:
        specs = self._metric_specs()[metric_name]
        metrics = rep.get("metrics", {})
        score = rep.get("score", {})
        metric_scores = score.get("metric_scores", {})
        weights = score.get("weights_used", {})
        raw_value = metrics.get(specs["source_key"])
        metric_score = metric_scores.get(metric_name)

        if raw_value is None or metric_score is None:
            return None

        threshold = get_view_thresholds(view)[metric_name]
        evaluated_value = abs(raw_value) if specs["evaluation"] == "absolute" else raw_value
        violated = (
            evaluated_value < float(threshold["good"])
            if bool(threshold["higher_is_better"])
            else evaluated_value > float(threshold["good"])
        )
        evaluation_text = (
            f"abs({float(raw_value):.2f}) = {evaluated_value:.2f} deg"
            if specs["evaluation"] == "absolute"
            else self._format_rule_value(float(evaluated_value), specs["unit"])
        )
        return {
            "label": specs["label"],
            "metric_key": metric_name,
            "metric_score": float(metric_score),
            "severity": score_severity(float(metric_score)),
            "weight": float(weights.get(metric_name, 0.0)),
            "violated": violated,
            "evaluation_text": evaluation_text,
            "rule_text": self._format_threshold_rule(threshold, specs["unit"]),
            "direction_text": "higher is better" if bool(threshold["higher_is_better"]) else "lower is better",
        }

    def _build_rep_diagnostics(self, rep: dict, view: str) -> list[dict]:
        diagnostics = []
        for metric_name in ("knee_valgus", "forward_lean", "depth", "squat_depth"):
            detail = self._build_metric_diagnostic(metric_name, rep, view)
            if detail is not None:
                diagnostics.append(detail)
        diagnostics.sort(key=lambda item: item["metric_score"])
        return diagnostics

    def _format_rep_analysis(self, rep: dict, view: str, analysis_summary: dict | None = None) -> list[str]:
        rep_id = rep.get("rep_id", "?")
        if analysis_summary:
            for analyzed_rep in analysis_summary.get("repetitions", []):
                if analyzed_rep.get("rep_id") == rep_id:
                    diagnostics = analyzed_rep.get("diagnostics", [])
                    headline = analyzed_rep.get("headline")
                    break
            else:
                diagnostics = []
                headline = None
        else:
            diagnostics = self._build_rep_diagnostics(rep, view)
            causes = [detail for detail in diagnostics if detail["violated"]]
            headline = "All scored metrics were within the target range."
            if causes:
                top = causes[0]
                headline = (
                    f"Main issue: {top['label']} ({top['severity'].lower()}) with value "
                    f"{top['evaluation_text']} and metric score {top['metric_score']:.1f}/100."
                )

        lines = []
        if headline:
            lines.append(f"  Summary: {headline}")

        if not diagnostics:
            lines.append("  No metric-level analysis was available.")
            return lines

        violations = [detail for detail in diagnostics if detail.get("violated")]
        focus = violations if violations else diagnostics
        lines.append("  Why the score moved:")
        for detail in focus:
            lines.append(
                f"    - {detail['label']}: {detail['severity']} | "
                f"Detected {detail['evaluation_text']} | "
                f"Rule used: {detail['direction_text']}, {detail['rule_text']} | "
                f"Metric score: {detail['metric_score']:.1f}/100 | Weight: {detail['weight']:.2f}"
            )
        return lines

    def _format_score_report(self, data: dict, analysis_summary: dict | None = None) -> str:
        lines = []
        lines.append(f"Video ID: {data.get('video_id', 'Unknown')}")
        lines.append(f"Overall Score: {data.get('overall_score', 0):.1f}/100")
        lines.append(f"View: {str(data.get('view', 'Unknown')).replace('_', ' ').title()}")
        lines.append(f"Repetitions: {data.get('rep_count', 0)}")
        lines.append(f"Source Quality: {str(data.get('source_quality', 'Unknown')).title()}")
        if data.get("message"):
            lines.append(f"Status: {data['message']}")
        lines.append("-" * 40)

        repetitions = data.get("repetitions", [])
        if not repetitions:
            lines.append("No scored repetitions found.")
            return "\n".join(lines)

        if analysis_summary and analysis_summary.get("summary_lines"):
            lines.append("Score Explanation:")
            for item in analysis_summary["summary_lines"]:
                lines.append(f"  {item}")
            lines.append("-" * 40)

        for rep in repetitions:
            rep_score = rep.get("score", {})
            metrics = rep.get("metrics", {})
            metric_scores = rep_score.get("metric_scores", {})

            lines.append(f"Rep {rep.get('rep_id', '?')}: {rep_score.get('overall_score', 0):.1f}/100")
            lines.append(
                f"  Frames: {rep.get('start_frame', '?')} -> {rep.get('end_frame', '?')} | "
                f"Duration: {rep.get('duration_seconds', 0):.2f}s"
            )
            lines.append(
                f"  Depth: {self._format_optional_metric(metrics.get('squat_depth'))} | "
                f"Knee Angle: {self._format_optional_metric(metrics.get('min_knee_angle'))}"
            )
            lines.append(
                f"  Knee Valgus: {self._format_optional_metric(metrics.get('knee_valgus'))} | "
                f"Forward Lean: {self._format_optional_metric(metrics.get('forward_lean'))}"
            )
            if metric_scores:
                score_parts = [
                    f"{name.replace('_', ' ').title()}: {value:.1f}"
                    for name, value in metric_scores.items()
                ]
                lines.append("  Metric Scores: " + ", ".join(score_parts))
            lines.extend(self._format_rep_analysis(rep, str(data.get("view", "unknown")), analysis_summary))
            lines.append("")

        return "\n".join(lines).rstrip()

    def _format_optional_metric(self, value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}"
        return "N/A"

    def _load_score_data(self, score_path: Path | None) -> None:
        if score_path is None or not score_path.exists():
            self._clear_score_display()
            return

        try:
            with open(score_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.current_analysis_summary_data = None
            self.current_score_data = None
            self._set_score_display(f"Failed to load scoring results.\n\n{exc}")
            return

        analysis_summary = None
        run_root = self._find_run_root_for_path(score_path)
        if run_root is not None:
            summary_path = self._find_analysis_summary_json(run_root, str(data.get("video_id", "")))
            if summary_path is not None and summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        analysis_summary = json.load(f)
                except Exception:
                    analysis_summary = None

        self.current_analysis_summary_data = analysis_summary
        self.current_score_data = data
        self._set_score_display(self._format_score_report(data, analysis_summary))

    def _find_original_video_for_overlay(self, overlay_path: Path) -> Path | None:
        try:
            stem_base = overlay_path.stem.replace("_annotated", "").replace("_segmented", "").replace("_phases", "")
            run_root = self._find_run_root_for_path(overlay_path)

            if run_root is None:
                return None

            dataset_root = run_root / "workspace" / "squat" / "dataset_videos_all"
            if not dataset_root.exists():
                return None

            for path in dataset_root.rglob("*"):
                if path.is_file() and path.stem == stem_base:
                    return path
        except Exception:
            return None
        return None

    def _load_selected_preview(self) -> None:
        selection = self.output_video_var.get().strip()
        if not selection:
            self.preview_status_var.set("Select an output video first.")
            return

        # Handle missing outputs
        if "[Missing Output]" in selection:
            stem = selection.replace(" [Missing Output]", "")
            run_root = None
            for parent in self.dataset_var.get() and [Path(RUNS_ROOT) / self.run_name_var.get()]:
                if parent.exists():
                    run_root = parent
                    break
            
            error_msg = "Pipeline failed before generating output video."
            
            # Try to find corresponding JSON to extract error
            if run_root:
                 search_dirs = [
                     run_root / "stage_outputs" / "03_temporal_segmentation",
                     run_root / "workspace" / "squat" / "segmented_reps"
                 ]
                 for path in search_dirs:
                     if not path.exists():
                         continue
                     for json_file in path.rglob(f"{stem}_segmented.json"):
                         try:
                             with open(json_file, 'r') as f:
                                 data = json.load(f)
                                 if "error" in data:
                                     error_msg = f"Failed. Error: {data['error']}"
                                 elif "view" in data.get("info", {}):
                                     view = data["info"]["view"]
                                     if "unknown" in view.lower():
                                         error_msg = f"Failed. Video skipped due to 'unknown' view."
                         except Exception:
                             pass

            self.preview_status_var.set(error_msg)
            self._stop_preview(release_only=True)
            self.original_image_label.configure(image="", text="No original preview loaded.")
            self.overlay_image_label.configure(image="", text=error_msg)
            self._preview_original_photo = None
            self._preview_overlay_photo = None
            
            # Try to update view metadata anyway if possible
            dummy_path = run_root / "workspace" / "squat" / "segmented_reps" / "excellent" / f"{stem}_segmented.json" if run_root else Path(f"{stem}_segmented.json")
            self._update_metadata_display(dummy_path)
            return

        selected_pair = self.preview_video_map.get(selection)
        if selected_pair is None:
            self.preview_status_var.set("Selected output video is missing.")
            return

        selected_overlay_path, selected_original_path = selected_pair
        if not selected_overlay_path.exists():
            self.preview_status_var.set("Selected output video is missing.")
            return

        self._stop_preview(release_only=True)

        self.preview_overlay_cap = cv2.VideoCapture(str(selected_overlay_path))
        if not self.preview_overlay_cap.isOpened():
            self.preview_overlay_cap = None
            self.preview_status_var.set("Failed to open selected video.")
            return

        if selected_original_path is not None and selected_original_path.exists():
            self.preview_original_cap = cv2.VideoCapture(str(selected_original_path))
            if not self.preview_original_cap.isOpened():
                self.preview_original_cap = None
        else:
            self.preview_original_cap = None

        fps = self.preview_overlay_cap.get(cv2.CAP_PROP_FPS)
        self.preview_fps = fps if fps and fps > 0 else 30.0

        ok_overlay, overlay_frame = self.preview_overlay_cap.read()
        original_frame = None
        if self.preview_original_cap is not None:
            ok_original, original_frame = self.preview_original_cap.read()
            if not ok_original:
                original_frame = None

        if ok_overlay:
            self._render_frame(overlay_frame, original_frame)
            self.preview_overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.preview_original_cap is not None:
                self.preview_original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if selected_original_path is not None:
                self.preview_status_var.set(
                    f"Loaded side-by-side preview: {selected_original_path.name} + {selected_overlay_path.name}"
                )
            else:
                self.preview_status_var.set(f"Loaded overlay preview: {selected_overlay_path.name} (original not found)")
        else:
            self.preview_status_var.set("Could not read frames from selected video.")

        # Update view/reps classification label
        self._update_metadata_display(selected_overlay_path)

    def _update_metadata_display(self, overlay_path: Path) -> None:
        self.view_type_var.set("")
        self.current_frame_phases = None
        self.current_reps = 0
        self.current_view = ""
        self.current_analysis_data = None
        self.current_analysis_summary_data = None
        self.current_score_data = None
        
        try:
            # 1. Identify if it's segmentation or classification
            is_segmentation = "_segmented" in overlay_path.stem or "_phases" in overlay_path.stem
            
            quality_folder = overlay_path.parent.name
            stem_clean = overlay_path.stem.replace("_annotated", "").replace("_segmented", "").replace("_phases", "")
            
            # 2. Construct potential JSON path
            # Search upwards for the 'squat' directory to anchor our path
            squat_root = None
            for parent in overlay_path.parents:
                if parent.name == "squat":
                    squat_root = parent
                    break
            
            if squat_root is None:
                return

            run_root = self._find_run_root_for_path(overlay_path)
            if run_root is not None:
                self._load_score_data(self._find_score_json(run_root, stem_clean))
            else:
                self._clear_score_display()

            if is_segmentation:
                # .../squat/segmented_reps/<quality>/<vid>_segmented.json
                json_path = squat_root / "segmented_reps" / quality_folder / f"{stem_clean}_segmented.json"
                
                if json_path.exists():
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        self.current_analysis_data = data
                        self.current_reps = data.get("info", {}).get("total_reps", 0)
                        self.current_view = data.get("info", {}).get("view", "Unknown").replace("_", " ").title()
                        self.current_frame_phases = data.get("frame_phases", [])
                        print(f"DEBUG: Loaded {len(self.current_frame_phases)} phases for video.")
                        
                        self.view_type_var.set(f"View: {self.current_view} | Reps: {self.current_reps}")
                else:
                    self.view_type_var.set("Reps: ?")

            else:
                # .../squat/extracted_features_clean/<quality>/<vid>.json
                json_path = squat_root / "extracted_features_clean" / quality_folder / f"{stem_clean}.json"

                if json_path.exists():
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        view = data.get("info", {}).get("view", "Unknown")
                        self.view_type_var.set(f"View: {view.replace('_', ' ').title()}")
                else:
                    self.view_type_var.set("View: Not found")

            if self.current_score_data and not self.current_analysis_data:
                score_view = str(self.current_score_data.get("view", "Unknown")).replace("_", " ").title()
                score_reps = self.current_score_data.get("rep_count", 0)
                score_value = self.current_score_data.get("overall_score", 0)
                self.view_type_var.set(f"View: {score_view} | Reps: {score_reps} | Score: {score_value:.1f}")

        except Exception as e:
            print(f"Error reading view classification: {e}")
            self.view_type_var.set("View: Error")
            self._clear_score_display("Failed to load scoring results.")

    def _toggle_preview_playback(self) -> None:
        if self.preview_overlay_cap is None:
            self._load_selected_preview()
            if self.preview_overlay_cap is None:
                return

        if self.preview_is_playing:
            self.preview_is_playing = False
            self.preview_status_var.set("Playback paused.")
            return

        self.preview_is_playing = True
        self.preview_stop_event.clear()
        self.preview_status_var.set("Playback running...")

        self.preview_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.preview_thread.start()

    def _playback_loop(self) -> None:
        while self.preview_is_playing and not self.preview_stop_event.is_set():
            if self.preview_overlay_cap is None:
                return

            ok_overlay, overlay_frame = self.preview_overlay_cap.read()
            if not ok_overlay:
                self.preview_overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_overlay, overlay_frame = self.preview_overlay_cap.read()
                if not ok_overlay:
                    return

            original_frame = None
            if self.preview_original_cap is not None:
                ok_original, original_frame = self.preview_original_cap.read()
                if not ok_original:
                    self.preview_original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_original, original_frame = self.preview_original_cap.read()
                    if not ok_original:
                        original_frame = None

            frame_idx = int(self.preview_overlay_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.root.after(0, self._render_frame, overlay_frame, original_frame, frame_idx)
            time.sleep(max(1.0 / self.preview_fps, 0.01))

    def _frame_to_photo(self, frame, max_width: int, max_height: int):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=image)

    def _render_frame(self, overlay_frame, original_frame=None, frame_idx: int = 0) -> None:
        if Image is None or ImageTk is None:
            self.original_image_label.configure(
                image="",
                text="Install Pillow to enable in-app video preview (pip install pillow).",
            )
            self.overlay_image_label.configure(
                image="",
                text="Install Pillow to enable in-app video preview (pip install pillow).",
            )
            return

        if overlay_frame is not None:
            overlay_photo = self._frame_to_photo(overlay_frame, max_width=520, max_height=280)
            self._preview_overlay_photo = overlay_photo
            self.overlay_image_label.configure(image=overlay_photo, text="")

        if original_frame is not None:
            original_photo = self._frame_to_photo(original_frame, max_width=520, max_height=280)
            self._preview_original_photo = original_photo
            self.original_image_label.configure(image=original_photo, text="")
        else:
            self.original_image_label.configure(image="", text="Original video not found for this overlay.")

        # Update real-time phase display if available
        if self.current_frame_phases:
            if 0 <= frame_idx < len(self.current_frame_phases):
                phase = self.current_frame_phases[frame_idx]
                text = f"View: {self.current_view} | Reps: {self.current_reps} | Phase: {phase.upper()}"
                self.view_type_var.set(text)
                # print(f"DEBUG: Frame {frame_idx}, Phase {phase}, Text: {text}") 
            else:
                # print(f"DEBUG: Frame {frame_idx} out of range (0-{len(self.current_frame_phases)})")
                pass
        else:
             # print("DEBUG: No current_frame_phases found")
             pass

    def _set_overall_progress(self, value: float) -> None:
        self.overall_progress_var.set(float(value))

    def _set_stage_progress(self, value: float) -> None:
        self.stage_progress_var.set(float(value))

    def _stop_preview(self, release_only: bool = False) -> None:
        self.preview_is_playing = False
        self.preview_stop_event.set()

        if self.preview_overlay_cap is not None:
            if release_only:
                self.preview_overlay_cap.release()
                self.preview_overlay_cap = None
                if self.preview_original_cap is not None:
                    self.preview_original_cap.release()
                    self.preview_original_cap = None
            else:
                self.preview_overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_overlay, overlay_frame = self.preview_overlay_cap.read()
                original_frame = None
                if self.preview_original_cap is not None:
                    self.preview_original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_original, original_frame = self.preview_original_cap.read()
                    if not ok_original:
                        original_frame = None
                if ok_overlay:
                    self._render_frame(overlay_frame, original_frame)
                    self.preview_overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self.preview_original_cap is not None:
                        self.preview_original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Wait for playback thread to actually stop to avoid race conditions
        if self.preview_thread is not None and self.preview_thread.is_alive():
            try:
                self.preview_thread.join(timeout=0.2)
            except Exception:
                pass
            self.preview_thread = None

        if not release_only:
            self.preview_status_var.set("Playback stopped.")

    def _stop_pipeline_safe(self) -> None:
        self._log("\n⏸️  Stopping pipeline...")
        self.stop_requested = True
        if self.current_process is not None:
            try:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._log("Force killing process...")
                    self.current_process.kill()
                    self.current_process.wait(timeout=1)
            except Exception as e:
                self._log(f"Warning: Failed to stop process: {e}")
        self.stop_button.configure(state="disabled")

    def _update_button_states(self) -> None:
        if self.pipeline_running:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _open_in_default_player(self) -> None:
        selection = self.output_video_var.get().strip()
        if not selection:
            self.preview_status_var.set("Select an output video first.")
            return

        selected_pair = self.preview_video_map.get(selection)
        if selected_pair is None:
            self.preview_status_var.set("Selected output video is missing.")
            return

        selected_path, _ = selected_pair
        if not selected_path.exists():
            self.preview_status_var.set("Selected output video is missing.")
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(selected_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(selected_path)])
            else:
                subprocess.Popen(["xdg-open", str(selected_path)])
            self.preview_status_var.set(f"Opened in system player: {selected_path.name}")
        except Exception as exc:
            self.preview_status_var.set("Failed to open system video player.")
            messagebox.showerror("Open Player Failed", str(exc))

    def _on_close(self) -> None:
        if self.pipeline_running:
            if messagebox.askyesno("Confirm", "Pipeline is running. Stop and close?"):
                self.stop_requested = True
                if self.current_process is not None:
                    try:
                        self.current_process.kill()
                    except Exception:
                        pass
        self._stop_preview(release_only=True)
        self.root.destroy()

    def _show_report(self) -> None:
        if self.current_score_data:
            report_text = self._format_score_report(self.current_score_data, self.current_analysis_summary_data)
        elif self.current_analysis_data:
            data = self.current_analysis_data
            info = data.get("info", {})
            reps = data.get("repetitions", [])

            lines = []
            lines.append(f"Video ID: {data.get('video_id', 'Unknown')}")
            lines.append(f"Quality: {info.get('quality_rating', 'Unknown').upper()}")
            lines.append(f"View: {info.get('view', 'Unknown').title()}")
            lines.append("-" * 40)
            lines.append(f"Total Reps: {info.get('total_reps', 0)}")
            lines.append("-" * 40)

            for rep in reps:
                lines.append(f"Rep {rep['rep_id']}:")
                lines.append(f"  Start: {rep['start_frame']} -> End: {rep['end_frame']}")
                lines.append(f"  Depth: {rep['squat_depth_normalized']:.3f} (Angle: {rep['squat_depth_angle']:.1f}°)")
                
                lines.append("  Phases:")
                for p in rep.get("phases", []):
                    duration = p.get("duration_seconds", 0)
                    reason = p.get("transition_reason", "")
                    reason_str = f" [{reason}]" if reason else ""
                    lines.append(f"    - {p['phase_type'].upper()}: {duration:.2f}s{reason_str}")
                lines.append("")

            report_text = "\n".join(lines)
        else:
            messagebox.showinfo("Report", "No analysis data available for this video.")
            return

        # Show in new window
        top = tk.Toplevel(self.root)
        top.title("Analysis Report")
        top.geometry("500x600")

        text_widget = tk.Text(top, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, report_text)
        text_widget.configure(state="disabled")  # Read-only


def main() -> None:
    root = tk.Tk()
    app = PipelineRunnerUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
