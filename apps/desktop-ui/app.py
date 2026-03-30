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

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
# Canonical stage scripts location (post-migration)
STAGES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "stages"
# Legacy root: still used for dataset_videos_all default and analyze_results script
RUNTIME_ROOT = WORKSPACE_ROOT / "_hidden_legacy"
RUNS_ROOT = WORKSPACE_ROOT / "pipeline_ui_runs"
LEGACY_RUNS_ROOT = RUNTIME_ROOT / "pipeline_ui_runs"
SHARED_MODEL_PATH = WORKSPACE_ROOT / "models" / "pose_landmarker_heavy.task"
SHARED_FACE_MODEL_PATH = WORKSPACE_ROOT / "models" / "blaze_face_short_range.tflite"


STAGES: tuple[Stage, ...] = (
    Stage(
        key="extract_selected_features",
        label="2.5 Extract Selected Features",
        script_path=STAGES_DIR / "extract_selected_features.py",
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
        script_path=STAGES_DIR / "classify_views.py",
        args=(),
        output_paths=("squat/extracted_features_clean",),
    ),
    Stage(
        key="temporal_segmentation",
        label="5 Temporal Segmentation",
        script_path=STAGES_DIR / "temporal_segmentation.py",
        args=(),
        output_paths=("squat/segmented_reps", "squat/visualized_segmentation"),
    ),
    Stage(
        key="scoring",
        label="8 Scoring",
        script_path=STAGES_DIR / "scoring.py",
        args=("*",),
        output_paths=("squat/aqa_analysis_simple",),
    ),
    Stage(
        key="analyze_results",
        label="9 Analyze Results",
        script_path=RUNTIME_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py",
        args=(),
        output_paths=("squat/aqa_analysis_simple/analysis_visualizations",),
    ),
    Stage(
        key="neural_fusion",
        label="9 Neural Fusion Scoring",
        script_path=STAGES_DIR / "neural_fusion_inference.py",
        args=(),
        output_paths=("squat/neural_analysis",),
    ),
)


def ordered_stages(stage_keys: list[str]) -> list[Stage]:
    selected = set(stage_keys)
    return [stage for stage in STAGES if stage.key in selected]


def get_view_thresholds(view: str) -> dict[str, dict[str, float | bool]]:
    view_lower = str(view).lower()

    default_thresholds = {
        "knee_valgus": {"good": 0.95, "bad": 0.75, "higher_is_better": True},
        "forward_lean": {"good": 25.0, "bad": 50.0, "higher_is_better": False},
        "depth": {"good": 75.0, "bad": 110.0, "higher_is_better": False},
        "squat_depth": {"good": 0.1, "bad": -0.1, "higher_is_better": True},
    }

    if "side" in view_lower and "front" not in view_lower and "back" not in view_lower:
        return {
            "knee_valgus": {"good": 0.95, "bad": 0.70, "higher_is_better": True},
            "forward_lean": {"good": 35.0, "bad": 60.0, "higher_is_better": False},
            "depth": {"good": 50.0, "bad": 100.0, "higher_is_better": False},
            "squat_depth": {"good": 0.15, "bad": -0.05, "higher_is_better": True},
        }

    if view_lower in ["front", "back"]:
        return {
            "knee_valgus": {"good": 0.97, "bad": 0.80, "higher_is_better": True},
            "forward_lean": {"good": 30.0, "bad": 55.0, "higher_is_better": False},
            "depth": {"good": 80.0, "bad": 120.0, "higher_is_better": False},
            "squat_depth": {"good": 0.08, "bad": -0.08, "higher_is_better": True},
        }

    if "front_side" in view_lower or "front-side" in view_lower:
        return {
            "knee_valgus": {"good": 0.95, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 22.0, "bad": 45.0, "higher_is_better": False},
            "depth": {"good": 75.0, "bad": 112.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.05, "higher_is_better": True},
        }

    if "back_side" in view_lower or "back-side" in view_lower:
        return {
            "knee_valgus": {"good": 1.2, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 35.0, "bad": 55.0, "higher_is_better": False},
            "depth": {"good": 75.0, "bad": 115.0, "higher_is_better": False},
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
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.dataset_var = tk.StringVar(value=str(RUNTIME_ROOT / "squat" / "dataset_videos_all"))
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
        self.current_neural_data: dict | None = None
        self.current_run_root: Path | None = None

        self.stage_checks: dict[str, tk.BooleanVar] = {
            stage.key: tk.BooleanVar(value=True) for stage in STAGES
        }

        self.preview_video_map: dict[str, tuple[Path, Path | None]] = {}
        self.preview_overlay_cap: cv2.VideoCapture | None = None
        self.preview_original_cap: cv2.VideoCapture | None = None
        self._preview_cap_lock = threading.Lock()
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
        ttk.Radiobutton(processing_frame, text="Dual (Filtered + Neural on Raw)", variable=self.processing_mode_var, value="dual").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            processing_frame,
            text="Use GPU when available",
            variable=self.use_gpu_var,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        # Neural Pipeline Section
        neural_frame = ttk.LabelFrame(left_panel, text="Neural Pipeline (Optional)", padding=8)
        neural_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        neural_frame.columnconfigure(0, weight=1)
        ttk.Label(neural_frame, text="Run full pipeline with neural fusion scoring:", wraplength=250, justify=tk.LEFT).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.neural_button = ttk.Button(neural_frame, text="▶ Full Neural Pipeline", command=self._start_neural_pipeline, width=24)
        self.neural_button.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Start/Stop Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.start_button = ttk.Button(button_frame, text="▶ Start", command=self._start, width=14)
        self.start_button.pack(side=tk.TOP, padx=(0, 0), fill=tk.X)
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", command=self._stop_pipeline_safe, state="disabled", width=14)
        self.stop_button.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # Progress Bars
        progress_frame = ttk.LabelFrame(left_panel, text="Progress", padding=6)
        progress_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
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

    def _start_neural_pipeline(self) -> None:
        """Start full pipeline with neural fusion scoring (stages 2.5 → 4 → 5 → 8 → neural_fusion)."""
        # Run full pipeline ending with neural fusion stage
        neural_stages = ordered_stages([
            "extract_selected_features",
            "classify_views",
            "temporal_segmentation",
            "scoring",
            "neural_fusion",
        ])
        
        input_mode = self.input_mode_var.get()
        dataset_path: Path | None = None
        video_path: Path | None = None

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

        run_name = f"neural_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_root = RUNS_ROOT / run_name
        self.current_run_root = run_root

        self.stop_requested = False
        self.pipeline_running = True
        self.start_button.configure(state="disabled")
        self.neural_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_overall_progress(0)
        self._set_stage_progress(0)
        self._log(f"▶ Starting neural pipeline run: {run_name}")
        self._set_preview_outputs([])
        self._clear_score_display("Run in progress. Neural scoring results will appear here when available.")
        
        self._stop_preview(release_only=True)

        if video_path is not None:
            self._log(f"Input video: {video_path}")
        else:
            self._log(f"Input dataset: {dataset_path}")

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(run_root, neural_stages, dataset_path, video_path),
            daemon=True,
        )
        thread.start()

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
                try:
                    self._run_stage(stage, workspace_root, logs_root, video_path=video_path)
                except Exception as exc:
                    if stage.key == "neural_fusion":
                        self._log(f"⚠️ Neural stage failed, continuing with heuristic results: {exc}")
                        self.root.after(0, self._set_stage_progress, 100)
                        self.root.after(0, self._set_overall_progress, index * 100 / max(len(stages), 1))
                        continue
                    raise

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

        if not SHARED_FACE_MODEL_PATH.exists():
            raise RuntimeError(f"Shared face model file not found: {SHARED_FACE_MODEL_PATH}")
        self._log(f"Using shared face model: {SHARED_FACE_MODEL_PATH}")

        # Ensure local copy of analyze_results.py exists inside workspace
        analyze_src = RUNTIME_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py"
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

        # Setup environment early (needed for subprocess calls)
        env = os.environ.copy()
        env["EXEVISION_MODEL_PATH"] = str(SHARED_MODEL_PATH)
        env["EXEVISION_FACE_MODEL_PATH"] = str(SHARED_FACE_MODEL_PATH)

        # Add processing mode for extraction stage
        stage_args = list(stage.args)
        if stage.key == "extract_selected_features":
            processing_mode = self.processing_mode_var.get()
            
            # Handle dual mode: run filtered first (with failure handling), then unfiltered
            if processing_mode == "dual":
                # Run filtered first (may fail for Poor quality videos)
                try:
                    filtered_args = ["filtered"] + stage_args
                    if self.use_gpu_var.get():
                        filtered_args.append("--gpu")
                    if video_path is not None:
                        filtered_args.extend(["--video-id", video_path.stem])
                    filtered_cmd = [sys.executable, str(script_to_run), *filtered_args]
                    
                    # Run filtered extraction silently
                    self._log("[Dual Mode] Running filtered extraction (may skip for Poor quality)...")
                    result_filtered = subprocess.run(
                        filtered_cmd,
                        cwd=str(workspace_root),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result_filtered.returncode == 0:
                        self._log("[Dual Mode] Filtered extraction succeeded.")
                    else:
                        self._log("[Dual Mode] Filtered extraction skipped (likely Poor quality - continuing with unfiltered).")
                except Exception as e:
                    self._log(f"[Dual Mode] Filtered extraction failed: {e}. Continuing with unfiltered.")
                
                # Always run unfiltered (unconditional)
                unfiltered_args = ["unfiltered"] + stage_args
                if self.use_gpu_var.get():
                    unfiltered_args.append("--gpu")
                if video_path is not None:
                    unfiltered_args.extend(["--video-id", video_path.stem])
                stage_args = unfiltered_args
            else:
                # Single mode (either filtered or unfiltered)
                stage_args = [processing_mode] + stage_args
                if self.use_gpu_var.get():
                    stage_args.append("--gpu")
                if video_path is not None:
                    stage_args.extend(["--video-id", video_path.stem])
        elif stage.key == "classify_views" and video_path is not None:
            stage_args = ["--video-id", video_path.stem]
        elif stage.key == "temporal_segmentation" and video_path is not None:
            stage_args = ["--video-id", video_path.stem]
        elif stage.key == "scoring" and video_path is not None:
            stage_args = [video_path.stem]
        elif stage.key == "neural_fusion" and video_path is not None:
            stage_args = ["--video-id", video_path.stem]
            # If in dual mode, restrict neural to raw_unfiltered
            processing_mode = self.processing_mode_var.get()
            if processing_mode == "dual":
                stage_args.extend(["--quality-tier", "raw_unfiltered"])

        cmd = [sys.executable, str(script_to_run), *stage_args]
        log_file = logs_root / f"{stage.key}.log"

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
            found.extend(sorted(root.rglob("*_segmented.avi")))
            found.extend(sorted(root.rglob("*_phases.avi")))

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

    def _find_neural_json(self, run_root: Path, video_id: str) -> Path | None:
        """Find neural fusion scoring JSON for a video."""
        search_roots = [
            run_root / "workspace" / "squat" / "neural_analysis",
            run_root / "stage_outputs",
        ]

        for root in search_roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(f"{video_id}_neural.json"))
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

    def _normalize_diagnostic_detail(self, detail: dict) -> dict:
        normalized = dict(detail)
        if "rule_text" not in normalized:
            normalized["rule_text"] = normalized.get("threshold_text", "thresholds unavailable")
        if "direction_text" not in normalized:
            higher_is_better = normalized.get("higher_is_better")
            if higher_is_better is True:
                normalized["direction_text"] = "higher is better"
            elif higher_is_better is False:
                normalized["direction_text"] = "lower is better"
            else:
                normalized["direction_text"] = "rule direction unavailable"
        if "weight" not in normalized:
            normalized["weight"] = 0.0
        if "severity" not in normalized:
            normalized["severity"] = score_severity(float(normalized.get("metric_score", 0.0)))
        return normalized

    def _format_rep_analysis(self, rep: dict, view: str, analysis_summary: dict | None = None) -> list[str]:
        rep_id = rep.get("rep_id", "?")
        if analysis_summary:
            for analyzed_rep in analysis_summary.get("repetitions", []):
                if analyzed_rep.get("rep_id") == rep_id:
                    diagnostics = [self._normalize_diagnostic_detail(item) for item in analyzed_rep.get("diagnostics", [])]
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

    def _format_score_report(self, data: dict, analysis_summary: dict | None = None, neural_data: dict | None = None) -> str:
        lines = []
        lines.append(f"Video ID: {data.get('video_id', 'Unknown')}")
        lines.append(f"Overall Score: {data.get('overall_score', 0):.1f}/100")
        lines.append(f"View: {str(data.get('view', 'Unknown')).replace('_', ' ').title()}")
        lines.append(f"Repetitions: {data.get('rep_count', 0)}")
        lines.append(f"Source Quality: {str(data.get('source_quality', 'Unknown')).title()}")
        if data.get("message"):
            lines.append(f"Status: {data['message']}")
        if neural_data and neural_data.get("message"):
            lines.append(f"Neural Status: {neural_data['message']}")
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

        # Create neural rep lookup (by rep_id)
        neural_reps_by_id = {}
        if neural_data and neural_data.get("reps"):
            for nr in neural_data["reps"]:
                neural_reps_by_id[nr.get("rep_id")] = nr

        for rep in repetitions:
            rep_score = rep.get("score", {})
            metrics = rep.get("metrics", {})
            metric_scores = rep_score.get("metric_scores", {})
            rep_id = rep.get("rep_id", "?")

            lines.append(f"Rep {rep_id}: {rep_score.get('overall_score', 0):.1f}/100")
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
            
            # Add neural metrics if available
            neural_rep = neural_reps_by_id.get(rep_id)
            if neural_rep:
                neural_score = neural_rep.get("neural_score", None)
                if neural_score is not None:
                    lines.append(f"  [NEURAL] Score: {neural_score:.1f}/100 | Pre-clamp: {neural_rep.get('neural_score_pre_clamp', neural_score):.1f}")
                    
                    # Sub-metrics
                    sub_metrics = []
                    if neural_rep.get("smoothness") is not None:
                        sub_metrics.append(f"Smoothness: {neural_rep['smoothness']:.1f}")
                    if neural_rep.get("control") is not None:
                        sub_metrics.append(f"Control: {neural_rep['control']:.1f}")
                    if neural_rep.get("depth") is not None:
                        sub_metrics.append(f"Depth: {neural_rep['depth']:.1f}")
                    if neural_rep.get("forward_lean") is not None:
                        sub_metrics.append(f"Forward Lean: {neural_rep['forward_lean']:.1f}")
                    if neural_rep.get("knee_tracking") is not None:
                        sub_metrics.append(f"Knee Tracking: {neural_rep['knee_tracking']:.1f}")
                    
                    if sub_metrics:
                        lines.append(f"  [NEURAL] Metrics: {' | '.join(sub_metrics)}")
                    
                    # Safety clamps applied
                    clamps = neural_rep.get("safety_clamps_applied", [])
                    if clamps:
                        lines.append(f"  [NEURAL] Safety Clamps: {', '.join(clamps)}")
            
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
        neural_data = None
        run_root = self._find_run_root_for_path(score_path)
        if run_root is not None:
            summary_path = self._find_analysis_summary_json(run_root, str(data.get("video_id", "")))
            if summary_path is not None and summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        analysis_summary = json.load(f)
                except Exception:
                    analysis_summary = None
            
            # Load neural metrics if available
            neural_path = self._find_neural_json(run_root, str(data.get("video_id", "")))
            if neural_path is not None and neural_path.exists():
                try:
                    with open(neural_path, "r", encoding="utf-8") as f:
                        neural_data = json.load(f)
                except Exception:
                    neural_data = None

        self.current_analysis_summary_data = analysis_summary
        self.current_score_data = data
        self.current_neural_data = neural_data
        try:
            report_text = self._format_score_report(data, analysis_summary, neural_data)
        except Exception as exc:
            self.current_analysis_summary_data = analysis_summary
            self.current_score_data = data
            self.current_neural_data = neural_data
            self._set_score_display(
                "Failed to format scoring results.\n\n"
                f"{exc}\n\n"
                "The scoring JSON loaded successfully, but the UI report formatter hit an unexpected data shape."
            )
            return

        self._set_score_display(report_text)

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
            with self._preview_cap_lock:
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

        # Wait for playback thread to stop before touching capture handles.
        if self.preview_thread is not None and self.preview_thread.is_alive():
            try:
                self.preview_thread.join(timeout=1.0)
            except Exception:
                pass
        self.preview_thread = None

        with self._preview_cap_lock:
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
            self.neural_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.neural_button.configure(state="normal")
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


# ---------------------------------------------------------------------------
# Annotation Tool UI
# ---------------------------------------------------------------------------

ANNOTATIONS_DIR = (
    WORKSPACE_ROOT / "training_dataset" / "annotations"
    if (WORKSPACE_ROOT / "training_dataset" / "annotations").exists()
    else WORKSPACE_ROOT / "dataset" / "annotations"
)
ANNOTATIONS_VIDEOS_DIR = ANNOTATIONS_DIR / "videos"
ANNOTATIONS_INDEX = ANNOTATIONS_DIR / "index.json"
VIDEO_SCAN_BATCH_SIZE = 500


class AnnotationToolUI:
    """
    Rep annotation tool for ExeVision neural training dataset.

    Workflow:
      1. User picks a folder of raw squat videos
      2. Tool shows a browsable video list — user clicks one to annotate
      3. Tool auto-detects existing pipeline output in pipeline_ui_runs/
         - Found: loads segmentation + scoring results
         - Not found: runs Stages 2.5→4→5→8 automatically
      4. Presents each rep for human scoring (blind — heuristic revealed after)
      5. Annotations saved per-video to dataset/annotations/videos/{video_id}.json

    Keyboard-driven: type score + Enter, F1-F5 toggle flags.
    """

    def __init__(self, parent: ttk.Frame):
        self.parent = parent
        self.root = parent.winfo_toplevel()

        # State ----------------------------------------------------------
        self.videos_folder: Path | None = None
        self._last_scanned_folder: Path | None = None
        self.video_files: list[Path] = []
        self.loaded_video_count: int = 0
        self._scan_in_progress: bool = False
        self._processed_vids_cache: set[str] = set()
        self.video_scores: dict[str, float] = {}
        self._initial_load_target: int = VIDEO_SCAN_BATCH_SIZE
        self.current_video_path: Path | None = None
        self.current_video_id: str = ""
        self.current_annotation: dict | None = None  # loaded per-video annotation
        self.current_rep_idx: int = 0                 # which rep we're annotating
        self.current_video_idx: int | None = None     # index of current selected video
        self.pipeline_run_used: str = ""              # which run folder was used
        self.annotation_extraction_mode_var = tk.StringVar(value="Filtered")
        self._annotation_extraction_mode_map = {
            "Filtered": "filtered",
            "Unfiltered (raw)": "unfiltered",
            "Dual (Filtered + Neural on Raw)": "dual",
        }

        # Video playback
        self.cap_raw: cv2.VideoCapture | None = None
        self.cap_vis: cv2.VideoCapture | None = None
        self._annotation_cap_lock = threading.Lock()
        self.playing: bool = False
        self.play_stop_event = threading.Event()
        self.play_thread: threading.Thread | None = None
        self.rep_start: int = 0
        self.rep_end: int = 0
        self.fps: float = 30.0
        self._photo_raw = None
        self._photo_vis = None

        # Pipeline subprocess
        self.pipeline_process: subprocess.Popen | None = None
        self.pipeline_running = False
        self.stage_timeouts_sec = {
            "extract_selected_features": 180,
            "classify_views": 120,
            "temporal_segmentation": 300,
            "scoring": 120,
        }

        self._build_ui()

    # ================================================================== UI
    def _build_ui(self) -> None:
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)

        # Use PanedWindow so the user can resize left vs. right by dragging
        paned = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL, sashwidth=5, sashrelief="raised")
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- LEFT PANEL: folder + video list + scoring ----
        left_outer = ttk.Frame(paned, padding=0)
        paned.add(left_outer, width=320, minsize=280)

        left = ttk.Frame(left_outer, padding=8)
        left.pack(fill=tk.BOTH, expand=True)
        left.columnconfigure(0, weight=1)
        # Weight will be set later for the video list row (row 3)

        # -- Folder picker
        folder_frame = ttk.LabelFrame(left, text="Video Folder", padding=8)
        folder_frame.grid(row=0, column=0, sticky="ew")
        folder_frame.columnconfigure(0, weight=1)

        self.folder_var = tk.StringVar(value="")
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        folder_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_frame, text="Browse…", width=8,
                   command=self._pick_folder).grid(row=0, column=1, padx=(4, 0))

        # -- Status
        self.status_var = tk.StringVar(value="Pick a folder of squat videos to begin.")
        ttk.Label(left, textvariable=self.status_var,
                  wraplength=290, font=("TkDefaultFont", 9)).grid(
            row=1, column=0, sticky="w", pady=(4, 4))

        # -- Video list: collapsible section
        list_header = ttk.Frame(left)
        list_header.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        list_header.columnconfigure(1, weight=1)

        self._list_expanded = True

        def _toggle_list():
            if self._list_expanded:
                self._list_container.grid_remove()
                list_btn_frame.grid_remove()
                toggle_btn.configure(text="\u25b6 Videos (collapsed)")
                left.rowconfigure(3, weight=0)
                # Show bottom elements when collapsed
                score_frame.grid()
                flags_frame.grid()
                metrics_frame.grid()
                conf_frame.grid()
                notes_frame.grid()
                self.feedback_var_label.grid()
                self.progress_var_label.grid()
                btn_frame.grid()
                # Remove weight from left to prevent empty space at bottom
                left.rowconfigure(14, weight=0)
            else:
                self._list_container.grid()
                list_btn_frame.grid()
                toggle_btn.configure(text="\u25bc Videos")
                left.rowconfigure(3, weight=1)
                # Hide bottom elements when expanded
                score_frame.grid_remove()
                flags_frame.grid_remove()
                metrics_frame.grid_remove()
                conf_frame.grid_remove()
                notes_frame.grid_remove()
                self.feedback_var_label.grid_remove()
                self.progress_var_label.grid_remove()
                btn_frame.grid_remove()
                # Add weight to bottom to allow listbox to expand
                left.rowconfigure(14, weight=1)
            self._list_expanded = not self._list_expanded

        # Use a standard button style to avoid deformities
        toggle_btn = ttk.Button(list_header, text="▼ Videos", command=_toggle_list, width=15)
        toggle_btn.grid(row=0, column=0, sticky="w")

        self._list_container = ttk.Frame(left)
        self._list_container.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        left.rowconfigure(3, weight=1)  # list expands
        left.rowconfigure(2, weight=0)  # header row stays fixed
        self._list_container.columnconfigure(0, weight=1)
        self._list_container.rowconfigure(0, weight=1)

        self.video_listbox = tk.Listbox(self._list_container, selectmode=tk.EXTENDED,
                                         font=("TkDefaultFont", 9), height=10) # Reduced default height, relies on weight=1 to expand
        self.video_listbox.grid(row=0, column=0, sticky="nsew")
        self.video_listbox.bind("<Double-1>", self._on_video_double_click)

        list_scroll = ttk.Scrollbar(self._list_container, orient=tk.VERTICAL,
                                     command=self.video_listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.video_listbox.config(yscrollcommand=list_scroll.set)

        list_btn_frame = ttk.Frame(left)
        list_btn_frame.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        list_btn_frame.columnconfigure(0, weight=1)
        list_btn_frame.columnconfigure(1, weight=1)
        list_btn_frame.columnconfigure(2, weight=1)
        list_btn_frame.columnconfigure(3, weight=1)
        ttk.Button(list_btn_frame, text="Annotate Selected (Dbl-Click)", command=self._annotate_selected).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(list_btn_frame, text="Process Selected", command=lambda: self._process_selected_videos(force_reprocess=False)).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(list_btn_frame, text="Reprocess Selected (Overwrite)", command=self._reprocess_selected_videos).grid(row=0, column=2, sticky="ew", padx=(2, 0))
        self.load_more_btn = ttk.Button(list_btn_frame, text="Load More", command=self._load_more_videos)
        self.load_more_btn.grid(row=0, column=3, sticky="ew", padx=(2, 0))

        mode_row = ttk.Frame(list_btn_frame)
        mode_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(mode_row, text="Extraction Mode:").pack(side=tk.LEFT)
        self.annotation_mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.annotation_extraction_mode_var,
            state="readonly",
            width=18,
            values=("Filtered", "Unfiltered (raw)", "Dual (Filtered + Neural on Raw)"),
        )
        self.annotation_mode_combo.pack(side=tk.LEFT, padx=(6, 0))
        
        ttk.Button(mode_row, text="Sort by Heuristic", 
                   command=self._sort_list_by_score).pack(side=tk.RIGHT)

        # -- Loading Bar
        self.loading_bar = ttk.Progressbar(left, orient=tk.HORIZONTAL, mode='indeterminate')
        self.loading_bar.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.loading_bar.grid_remove() # Hidden by default

        # -- View label and Scoring panel
        score_frame = ttk.LabelFrame(left, text="Score This Rep (0-100)", padding=8)
        score_frame.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        score_frame.columnconfigure(1, weight=1)

        ttk.Label(score_frame, text="Score:",
                  font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.score_entry = ttk.Entry(score_frame, width=8,
                                      font=("TkDefaultFont", 14))
        self.score_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.score_entry.bind("<Return>", self._submit_score)

        btn_row = ttk.Frame(score_frame)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Submit (Enter)",
                   command=self._submit_score).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Skip",
                   command=self._skip_rep).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="← Prev Rep",
                   command=self._prev_rep).pack(side=tk.LEFT)

        view_row = ttk.Frame(score_frame)
        view_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(view_row, text="View Label:").pack(side=tk.LEFT, padx=(0, 6))
        self.view_var = tk.StringVar(value="")
        self.view_combo = ttk.Combobox(view_row, textvariable=self.view_var, state="readonly", width=15)
        self.view_combo["values"] = ("side", "front", "back", "front_side", "back_side", "unknown")
        self.view_combo.pack(side=tk.LEFT)

        # -- Errors (replaces old Flags)
        flags_frame = ttk.LabelFrame(left, text="Form Errors", padding=8)
        flags_frame.grid(row=8, column=0, sticky="ew", pady=(8, 0))

        self.flag_vars: dict[str, tk.BooleanVar] = {}
        self.flag_severity_vars: dict[str, tk.DoubleVar] = {}
        flag_defs = [
            ("insufficient_squat_depth",      "Insufficient Squat Depth"),
            ("knee_valgus",                   "Knee Valgus"),
            ("lumbar_flexion",                "Lumbar Flexion"),
            ("heel_rise",                     "Heel Rise"),
            ("asymmetric_descent",            "Asymmetric Descent"),
            ("forward_lean",                  "Forward Lean"),
        ]
        for i, (key, label) in enumerate(flag_defs):
            var = tk.BooleanVar(value=False)
            self.flag_vars[key] = var
            sev_var = tk.DoubleVar(value=0)
            self.flag_severity_vars[key] = sev_var
            
            row = ttk.Frame(flags_frame)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.columnconfigure(0, weight=1)
            
            cb = ttk.Checkbutton(row, text=label, variable=var, 
                                 command=lambda k=key: self._on_checkbox_toggled(k))
            cb.grid(row=0, column=0, sticky="w")
            
            slider = ttk.Scale(row, from_=0, to=5, variable=sev_var, orient=tk.HORIZONTAL, length=80,
                               command=lambda v, k=key: self._on_scale_changed(v, k))
            slider.grid(row=0, column=1, padx=(10, 0))
            
            val_lbl = ttk.Label(row, textvariable=sev_var, width=3)
            val_lbl.grid(row=0, column=2, padx=(4, 0))

        # Keybinds F1-F7 mapping to checkboxes (optional but helpful)
        flag_keys_list = list(self.flag_vars.keys())
        for idx in range(min(7, len(flag_keys_list))):
            fkey = f"<F{idx + 1}>"
            flag_key = flag_keys_list[idx]
            self.root.bind(fkey, lambda e, k=flag_key: self._toggle_flag(k))

        # -- Neural Metrics
        metrics_frame = ttk.LabelFrame(left, text="Target Metrics (0-100, blank = null)", padding=8)
        metrics_frame.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        
        self.metric_vars: dict[str, tk.StringVar] = {}
        metric_defs = [
            ("depth", "Depth:"),
            ("knee_tracking", "Knee Track:"),
            ("forward_lean", "Fwd Lean:"),
            ("smoothness", "Smoothness:"),
            ("control_at_bottom", "Ctrl at Btm:"),
        ]
        for i, (key, label) in enumerate(metric_defs):
            row = ttk.Frame(metrics_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            self.metric_vars[key] = var
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT)

        # -- Annotator Confidence
        conf_frame = ttk.Frame(left)
        conf_frame.grid(row=10, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(conf_frame, text="Confidence (1-5):").pack(side=tk.LEFT, padx=(0, 6))
        self.confidence_var = tk.StringVar(value="4")
        self.conf_combo = ttk.Combobox(conf_frame, textvariable=self.confidence_var, state="readonly", width=4)
        self.conf_combo["values"] = ("1", "2", "3", "4", "5")
        self.conf_combo.pack(side=tk.LEFT)

        # -- Notes
        notes_frame = ttk.Frame(left)
        notes_frame.grid(row=11, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(notes_frame, text="Notes:").pack(side=tk.LEFT, anchor="n")
        self.notes_var = tk.StringVar(value="")
        ttk.Entry(notes_frame, textvariable=self.notes_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # -- Feedback
        self.feedback_var = tk.StringVar(value="")
        self.feedback_var_label = ttk.Label(left, textvariable=self.feedback_var,
                  font=("TkDefaultFont", 10), foreground="gray",
                  wraplength=285, padding=(0, 6))
        self.feedback_var_label.grid(row=12, column=0, sticky="ew")

        # -- Progress for current video
        self.progress_var = tk.StringVar(value="")
        self.progress_var_label = ttk.Label(left, textvariable=self.progress_var,
                  font=("TkDefaultFont", 10, "bold"),
                  padding=(0, 2))
        self.progress_var_label.grid(row=13, column=0, sticky="ew") # bumped row to 13 because both were on 12 before

        # -- Action Buttons
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=14, column=0, sticky="ew", pady=(4, 0)) # bumped row to 14
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        btn_frame.columnconfigure(3, weight=1)
        
        ttk.Button(btn_frame, text="✅ Save",
                   command=self._manual_save).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn_frame, text="⏮ Prev",
                   command=self._prev_video).grid(row=0, column=1, sticky="ew", padx=(2, 2))
        ttk.Button(btn_frame, text="⏭ Next",
                   command=self._next_video).grid(row=0, column=2, sticky="ew", padx=(2, 2))
        ttk.Button(btn_frame, text="📊 Analyze All",
                   command=self._run_analysis).grid(row=0, column=3, sticky="ew", padx=(2, 0))

        # Initially start expanded (which means bottom is hidden according to new logic)
        self._list_expanded = True
        score_frame.grid_remove()
        flags_frame.grid_remove()
        metrics_frame.grid_remove()
        conf_frame.grid_remove()
        notes_frame.grid_remove()
        self.feedback_var_label.grid_remove()
        self.progress_var_label.grid_remove()
        btn_frame.grid_remove()

        # ---- RIGHT PANEL: dual side-by-side video + info ----
        right = ttk.Frame(paned, padding=8)
        paned.add(right)
        right.columnconfigure(0, weight=1, uniform="equal")
        right.columnconfigure(1, weight=1, uniform="equal")
        right.rowconfigure(1, weight=1)

        # Rep info bar (spans both columns)
        info_frame = ttk.Frame(right)
        info_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.rep_info_var = tk.StringVar(value="Select a video from the list.")
        ttk.Label(info_frame, textvariable=self.rep_info_var,
                  font=("TkDefaultFont", 10)).pack(side=tk.LEFT)

        # Left video: Raw
        raw_panel = ttk.LabelFrame(right, text="Raw Video", padding=2)
        raw_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(6, 6))
        raw_panel.columnconfigure(0, weight=1)
        raw_panel.rowconfigure(0, weight=1)
        self.raw_label = ttk.Label(raw_panel, text="No video loaded.", anchor="center", relief="sunken")
        self.raw_label.grid(row=0, column=0, sticky="nsew")
        self.video_label = self.raw_label # for compatibility

        # Right video: Visualized
        vis_panel = ttk.LabelFrame(right, text="Visualized Video", padding=2)
        vis_panel.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(6, 6))
        vis_panel.columnconfigure(0, weight=1)
        vis_panel.rowconfigure(0, weight=1)
        self.vis_label = ttk.Label(vis_panel, text="No visualized video.", anchor="center", relief="sunken")
        self.vis_label.grid(row=0, column=0, sticky="nsew")

        # Playback controls
        ctrl = ttk.Frame(right)
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        ttk.Button(ctrl, text="▶ Play Both",
                   command=self._toggle_play).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="⟲ Replay",
                   command=self._replay).pack(side=tk.LEFT)

        # Scoring reference
        ref = ttk.LabelFrame(right, text="Scoring Reference", padding=6)
        ref.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(ref, text=(
            "0-20: Terrible — safety concern  |  20-40: Poor — multiple obvious problems\n"
            "40-60: Below avg — noticeable issues  |  60-75: Decent — minor imperfections\n"
            "75-90: Good — clean movement  |  90-100: Excellent — textbook form"
        ), font=("TkDefaultFont", 8), wraplength=600).pack(anchor="w")

        # Heuristic Reference
        heur_frame = ttk.LabelFrame(right, text="Heuristic Results (Reference)", padding=6)
        heur_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.heuristic_var = tk.StringVar(value="Load a video to see heuristic results here.")
        ttk.Label(heur_frame, textvariable=self.heuristic_var, font=("TkDefaultFont", 9), foreground="blue", wraplength=600).pack(anchor="w")

        # Pipeline log (shown when auto-running pipeline)
        self.log_frame = ttk.LabelFrame(right, text="Pipeline Log", padding=4)
        self.log_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.log_frame.grid_remove()  # hidden by default

        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, height=5,
                                 font=("Consolas", 8))
        self.log_text.pack(fill=tk.X)

    # ============================================================ Folder
    def _pick_folder(self) -> None:
        default = str(PROJECT_ROOT / "squat" / "dataset_videos_all")
        selected = filedialog.askdirectory(
            initialdir=default if Path(default).exists() else str(PROJECT_ROOT),
            title="Select folder of squat videos",
        )
        if not selected:
            return
        self.videos_folder = Path(selected)
        self.folder_var.set(str(self.videos_folder))
        self._scan_folder()

    def _scan_folder(self, preserve_order: bool = False) -> None:
        if self.videos_folder is None or not self.videos_folder.exists():
            return
        exts = {".mp4", ".mov", ".avi", ".mkv", ".flv"}
        if self._last_scanned_folder != self.videos_folder:
            previous_loaded = 0
        else:
            previous_loaded = self.loaded_video_count
        self._last_scanned_folder = self.videos_folder
        
        if not preserve_order:
            self.video_files = sorted(
                [p for p in self.videos_folder.iterdir()
                if p.is_file() and p.suffix.lower() in exts]
            )
        total_videos = len(self.video_files)
        self._initial_load_target = min(max(previous_loaded, VIDEO_SCAN_BATCH_SIZE), total_videos)
        self.loaded_video_count = 0
        self._scan_in_progress = True
        self._processed_vids_cache = set()
        self.video_listbox.delete(0, tk.END)
        self._update_load_more_button_state()
        self.status_var.set(
            f"Scanning first {self._initial_load_target} of {total_videos} videos in background..."
        )
        
        # Run the UI-blocking scan in a background thread
        thread = threading.Thread(target=self._scan_folder_thread, daemon=True)
        thread.start()

    def _scan_folder_thread(self) -> None:
        # 1. Quickly find all processed videos by scanning RUNS_ROOT once
        processed_vids = set()
        self.video_scores = {}
        for runs_root in (RUNS_ROOT, LEGACY_RUNS_ROOT):
            if not runs_root.exists():
                continue
            for run_dir in list(runs_root.iterdir()):
                if not run_dir.is_dir():
                    continue
                # Fix: Check for both segmentation and scores
                seg_root = run_dir / "workspace" / "squat" / "segmented_reps"
                if seg_root.exists():
                    for f in seg_root.rglob("*_segmented.json"):
                        vid_id = f.stem.replace("_segmented", "")
                        processed_vids.add(vid_id)

                score_root = run_dir / "workspace" / "squat" / "aqa_analysis_simple"
                if score_root.exists():
                    for f in score_root.rglob("*_aqa_simple.json"):
                        vid_id = f.stem.replace("_aqa_simple", "")
                        try:
                            with open(f, "r", encoding="utf-8") as json_f:
                                score_data = json.load(json_f)
                                if "overall_score" in score_data:
                                    self.video_scores[vid_id] = float(score_data["overall_score"])
                        except Exception:
                            pass

        self._processed_vids_cache = processed_vids

        # 2. Process only the first chunk and send back to UI thread
        self._scan_video_chunk(0, self._initial_load_target)
        
    def _scan_video_chunk(self, start_idx: int, end_idx: int) -> None:
        chunk = self.video_files[start_idx:end_idx]
        for vf in chunk:
            vid = vf.stem
            ann_path = ANNOTATIONS_VIDEOS_DIR / f"{vid}.json"
            marker = ""
            is_processed = False

            if ann_path.exists():
                is_processed = True
                try:
                    with open(ann_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    reps = data.get("reps", [])
                    scored = sum(1 for r in reps if r.get("human_score") is not None)
                    total = len(reps)
                    if scored == total and total > 0:
                        marker = "  ✓"
                    elif scored > 0:
                        marker = f"  ({scored}/{total})"
                except Exception:
                    pass
            else:
                if vid in self._processed_vids_cache:
                    is_processed = True

            text = f"{vf.name}{marker}"
            self.root.after(0, self._add_video_to_listbox, text, is_processed)

        self.root.after(0, self._on_chunk_loaded, end_idx)

    def _on_chunk_loaded(self, end_idx: int) -> None:
        self.loaded_video_count = min(end_idx, len(self.video_files))
        self._scan_in_progress = False
        self._update_load_more_button_state()
        self.status_var.set(
            f"Loaded {self.loaded_video_count}/{len(self.video_files)} videos. "
            f"Click one to annotate."
        )

    def _load_more_videos(self) -> None:
        if self._scan_in_progress:
            return
        total = len(self.video_files)
        if self.loaded_video_count >= total:
            self._update_load_more_button_state()
            return

        start_idx = self.loaded_video_count
        end_idx = min(start_idx + VIDEO_SCAN_BATCH_SIZE, total)
        self._scan_in_progress = True
        self._update_load_more_button_state()
        self.status_var.set(f"Loading videos {start_idx + 1}-{end_idx} of {total}...")

        thread = threading.Thread(
            target=self._scan_video_chunk,
            args=(start_idx, end_idx),
            daemon=True,
        )
        thread.start()

    def _update_load_more_button_state(self) -> None:
        total = len(self.video_files)
        remaining = max(0, total - self.loaded_video_count)

        if self._scan_in_progress:
            self.load_more_btn.configure(state="disabled", text="Loading...")
            return

        if remaining <= 0:
            self.load_more_btn.configure(state="disabled", text="All Loaded")
            return

        next_count = min(VIDEO_SCAN_BATCH_SIZE, remaining)
        self.load_more_btn.configure(state="normal", text=f"Load More ({next_count})")

    def _add_video_to_listbox(self, text: str, is_processed: bool) -> None:
        # Determine marker and score suffix more reliably
        # The text coming in from _scan_video_chunk already has markers if they were found in index
        # But for new scans, we want to show heuristic score if available
        vid_id = ""
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".flv"]:
             if ext in text:
                 vid_id = text.split(ext)[0]
                 break
        if not vid_id: vid_id = text.split("  ")[0] # fallback

        if vid_id in self.video_scores:
            score = self.video_scores[vid_id]
            if "  " in text: # has existing marker
                 parts = text.split("  ")
                 text = f"{parts[0]} | Heur: {score:.1f}  {parts[1]}"
            else:
                 text = f"{text} | Heur: {score:.1f}"

        idx = self.video_listbox.size()
        self.video_listbox.insert(tk.END, text)
        if is_processed:
            self.video_listbox.itemconfig(idx, bg="#e6ffe6", fg="#006600") # Green highlight
        else:
            self.video_listbox.itemconfig(idx, bg="#ffe6e6", fg="#990000") # Red highlight

    def _sort_list_by_score(self) -> None:
        """Sorts the current video_files list by heuristic score (descending)."""
        if not self.video_files:
            return
        
        # Sort video_files based on score in video_scores
        # Videos without scores go to the bottom
        self.video_files.sort(key=lambda f: self.video_scores.get(f.stem, -1.0), reverse=True)
        
        # Refresh the listbox while preserving the sort order
        self._scan_folder(preserve_order=True)
        self.status_var.set(f"Sorted {len(self.video_files)} videos by heuristic score.")

    # ============================================================ Video selection
    def _on_video_double_click(self, event=None) -> None:
        self._annotate_selected()

    def _annotate_selected(self) -> None:
        sel = self.video_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self.video_files):
            return
        video_path = self.video_files[idx]
        self._load_video_for_annotation(video_path)

    def _next_video(self) -> None:
        if self.current_video_idx is not None:
            idx = self.current_video_idx
        else:
            sel = self.video_listbox.curselection()
            if sel:
                idx = sel[0]
            else:
                if self.video_files:
                    self._load_with_selection(0)
                return
        
        next_idx = idx + 1
        if next_idx < len(self.video_files):
            self._load_with_selection(next_idx)
        else:
            self.feedback_var.set("You have reached the end of the video list.")

    def _prev_video(self) -> None:
        if self.current_video_idx is not None:
            idx = self.current_video_idx
        else:
            sel = self.video_listbox.curselection()
            if sel:
                idx = sel[0]
            else:
                if self.video_files:
                    last_idx = len(self.video_files) - 1
                    self._load_with_selection(last_idx)
                return
        
        prev_idx = idx - 1
        if prev_idx >= 0:
            self._load_with_selection(prev_idx)
        else:
            self.feedback_var.set("You are at the beginning of the video list.")

    def _load_with_selection(self, idx: int) -> None:
        if idx >= self.loaded_video_count:
            self._load_more_videos()
            self.feedback_var.set("Loaded more videos. Press Next again to continue.")
            return

        self.video_listbox.selection_clear(0, tk.END)
        self.video_listbox.selection_set(idx)
        self.video_listbox.see(idx)
        self.current_video_idx = idx
        self._load_video_for_annotation(self.video_files[idx])

    def _reprocess_selected_videos(self) -> None:
        self._process_selected_videos(force_reprocess=True)

    def _get_selected_annotation_extraction_mode(self) -> str:
        selected_label = self.annotation_extraction_mode_var.get()
        return self._annotation_extraction_mode_map.get(selected_label, "filtered")

    def _clear_pipeline_references(self, video_id: str) -> None:
        ann_path = ANNOTATIONS_VIDEOS_DIR / f"{video_id}.json"
        if not ann_path.exists():
            return

        with open(ann_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        if not isinstance(annotation, dict):
            raise ValueError("Annotation file is not a JSON object")

        annotation["pipeline_run"] = ""
        annotation["pipeline_outputs"] = {}

        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

        if self.current_video_id == video_id and isinstance(self.current_annotation, dict):
            self.current_annotation["pipeline_run"] = ""
            self.current_annotation["pipeline_outputs"] = {}

    def _process_selected_videos(self, force_reprocess: bool = False) -> None:
        if self.pipeline_running:
            messagebox.showwarning("Busy", "Wait for the current pipeline run to finish.")
            return

        sel = self.video_listbox.curselection()
        if not sel:
            messagebox.showinfo("No Selection", "Please select one or more videos to process.")
            return

        selected_videos = [self.video_files[idx] for idx in sel]

        if force_reprocess:
            confirm = messagebox.askyesno(
                "Reprocess Selected Videos",
                "This will run a fresh pipeline for the selected video(s). Existing annotation files will be kept, and only pipeline reference fields will be reset. Continue?",
            )
            if not confirm:
                return

            videos_to_process = selected_videos
            reset_failures: list[str] = []
            for video_path in videos_to_process:
                try:
                    self._clear_pipeline_references(video_path.stem)
                except Exception as exc:
                    reset_failures.append(f"{video_path.name}: {exc}")

            if reset_failures:
                messagebox.showerror(
                    "Cannot Reset Pipeline References",
                    "Failed to reset pipeline references for:\n" + "\n".join(reset_failures),
                )
                return
        else:
            videos_to_process = []
            for video_path in selected_videos:
                run_path, _ = self._find_existing_pipeline_output(video_path.stem)
                if not run_path:
                    videos_to_process.append(video_path)

        if not videos_to_process:
            if force_reprocess:
                self.status_var.set("No videos selected for reprocessing.")
            else:
                self.status_var.set("All selected videos already have pipeline outputs.")
            return

        mode_label = "reprocessing" if force_reprocess else "processing"
        extraction_mode = self._get_selected_annotation_extraction_mode()
        self.status_var.set(
            f"Queueing {len(videos_to_process)} video(s) for {mode_label} [{extraction_mode}]..."
        )
        self.loading_bar.grid()
        self.loading_bar.start(10)
        self.log_frame.grid()  # show log
        self.log_text.delete("1.0", tk.END)

        self.pipeline_running = True
        thread = threading.Thread(
            target=self._batch_pipeline_thread, args=(videos_to_process,), daemon=True
        )
        thread.start()

    def _load_video_for_annotation(self, video_path: Path) -> None:
        self._stop_playback()
        self.current_video_path = video_path
        self.current_video_id = video_path.stem
        self.current_rep_idx = 0
        self.feedback_var.set("")
        
        # Ensure current_video_idx is set if loaded outside of _load_with_selection (e.g., annotate_selected)
        try:
            self.current_video_idx = self.video_files.index(video_path)
        except ValueError:
            pass

        # Check for existing annotation file
        ann_path = ANNOTATIONS_VIDEOS_DIR / f"{self.current_video_id}.json"
        if ann_path.exists():
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    loaded_annotation = json.load(f)

                if isinstance(loaded_annotation, dict) and not self._annotation_pipeline_refs_stale(loaded_annotation):
                    self.current_annotation = loaded_annotation
                    self.pipeline_run_used = self.current_annotation.get("pipeline_run", "")
                    self._advance_to_next_unannotated()
                    self._show_current_rep()
                    return

                self.current_annotation = loaded_annotation if isinstance(loaded_annotation, dict) else None
            except Exception:
                pass

        # No annotation file — need pipeline output
        # Search for existing pipeline run
        run_path, run_name = self._find_existing_pipeline_output(self.current_video_id)

        if run_path is not None:
            self.status_var.set(f"Found pipeline output in: {run_name}")
            self.pipeline_run_used = run_name
            if self._build_annotation_from_run(run_path, run_name):
                self._show_current_rep()
            else:
                self.status_var.set(f"No segmentation found for {self.current_video_id}.")
        else:
            # Need to run the pipeline
            self.status_var.set(
                f"No pipeline output found for {self.current_video_id}. "
                f"Running pipeline (2.5→4→5→8)…"
            )
            self._run_pipeline_for_video(video_path)

    def _find_existing_pipeline_output(
        self, video_id: str
    ) -> tuple[Path | None, str]:
        """
        Search pipeline_ui_runs/ for a completed run that has
        segmented + scored output for this video_id.
        Returns (run_workspace_path, run_name) or (None, "").
        """
        candidate_run_dirs: list[Path] = []
        for runs_root in (RUNS_ROOT, LEGACY_RUNS_ROOT):
            if not runs_root.exists():
                continue
            candidate_run_dirs.extend([d for d in runs_root.iterdir() if d.is_dir()])

        if not candidate_run_dirs:
            return None, ""

        # Check runs in reverse chronological order (newest first)
        run_dirs = sorted(candidate_run_dirs, key=lambda d: d.name, reverse=True)

        for run_dir in run_dirs:
            workspace = run_dir / "workspace"
            if not workspace.exists():
                continue

            # Check for segmented JSON
            seg_root = workspace / "squat" / "segmented_reps"
            if not seg_root.exists():
                continue

            seg_matches = list(seg_root.rglob(f"{video_id}_segmented.json"))
            if not seg_matches:
                continue

            # Check for scoring JSON
            score_root = workspace / "squat" / "aqa_analysis_simple"
            if score_root.exists():
                score_matches = list(score_root.rglob(f"{video_id}_aqa_simple.json"))
                if score_matches:
                    return workspace, run_dir.name

            # Has segmentation but no scoring — still usable (just no heuristic scores)
            return workspace, run_dir.name

        return None, ""

    def _annotation_pipeline_refs_stale(self, annotation: dict) -> bool:
        pipeline_run = annotation.get("pipeline_run")
        pipeline_outputs = annotation.get("pipeline_outputs")
        if not isinstance(pipeline_outputs, dict):
            return True
        segmented_json = pipeline_outputs.get("segmented_json")
        return not pipeline_run or not segmented_json

    def _load_existing_annotation(self, video_id: str) -> dict | None:
        ann_path = ANNOTATIONS_VIDEOS_DIR / f"{video_id}.json"
        if not ann_path.exists():
            return None

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _build_annotation_payload_from_run(
        self,
        workspace: Path,
        run_name: str,
        video_id: str,
        source_video_path: Path | None,
    ) -> dict | None:
        """Build annotation payload from a pipeline run for any target video."""
        existing_annotation = self._load_existing_annotation(video_id) or {}

        # Find segmented JSON
        seg_root = workspace / "squat" / "segmented_reps"
        seg_matches = list(seg_root.rglob(f"{video_id}_segmented.json"))
        if not seg_matches:
            return None
        seg_path = seg_matches[0]
        quality = seg_path.parent.name

        with open(seg_path, "r", encoding="utf-8") as f:
            seg_data = json.load(f)

        # Find scoring JSON (recursive into nested dirs)
        score_root = workspace / "squat" / "aqa_analysis_simple"
        score_data = None
        score_matches = []
        if score_root.exists():
            score_matches = list(score_root.rglob(f"{video_id}_aqa_simple.json"))
            if score_matches:
                with open(score_matches[0], "r", encoding="utf-8") as f:
                    score_data = json.load(f)

        # Find features JSON path
        feat_root = workspace / "squat" / "extracted_features_clean"
        features_path = ""
        if feat_root.exists():
            feat_matches = list(feat_root.rglob(f"{video_id}.json"))
            if feat_matches:
                features_path = str(feat_matches[0])

        # Build rep-level score lookup
        rep_scores: dict[int, dict] = {}
        if score_data:
            for rep_entry in score_data.get("repetitions", []):
                rid = rep_entry.get("rep_id")
                if rid is not None:
                    rep_scores[rid] = rep_entry

        view = seg_data.get("info", {}).get("view", "unknown")
        fps = seg_data.get("info", {}).get("fps", 30.0)

        # Find visualization video path
        vis_root = workspace / "squat" / "visualized_segmentation"
        vis_video = ""
        for suffix in ("_phases.mp4", "_segmented.mp4", "_phases.avi", "_segmented.avi"):
            candidate = vis_root / quality / f"{video_id}{suffix}"
            if candidate.exists():
                vis_video = str(candidate)
                break
        if not vis_video:
            # Fallback to raw video
            raw = workspace / "squat" / "dataset_videos_all" / f"{video_id}.mp4"
            if raw.exists():
                vis_video = str(raw)

        # Make paths relative to PROJECT_ROOT for portability
        def to_rel(p: str) -> str:
            if not p: return ""
            p_obj = Path(p)
            if p_obj.is_relative_to(PROJECT_ROOT):
                # Always format as forward-slash relative paths
                return str(p_obj.relative_to(PROJECT_ROOT)).replace("\\", "/")
            return str(p_obj).replace("\\", "/")

        # Keep human annotation fields across reprocess by rep_id.
        preserved_reps: dict[int, dict] = {}
        for old_rep in existing_annotation.get("reps", []):
            if not isinstance(old_rep, dict):
                continue
            old_rep_id = old_rep.get("rep_id")
            if old_rep_id is None:
                continue
            preserved_reps[old_rep_id] = {
                "human_score": old_rep.get("human_score"),
                "human_flags": old_rep.get("human_flags", old_rep.get("flags")),
                "human_flag_severities": old_rep.get("human_flag_severities", old_rep.get("flag_severities")),
                "annotator_confidence": old_rep.get("annotator_confidence"),
                "annotation_notes": old_rep.get("annotation_notes"),
                "human_metric_scores": old_rep.get("human_metric_scores"),
                "flags": old_rep.get("flags", old_rep.get("human_flags")),
                "flag_severities": old_rep.get("flag_severities", old_rep.get("human_flag_severities")),
            }

        # Build reps list
        reps = []
        frame_phases = seg_data.get("frame_phases", [])
        signals = seg_data.get("signals", {})
        
        for rep in seg_data.get("repetitions", []):
            rep_id = rep.get("rep_id", 0)
            rep_score_entry = rep_scores.get(rep_id, {})
            score_obj = rep_score_entry.get("score", {})
            metrics = rep_score_entry.get("metrics", {})
            
            start_f = rep.get("start_frame", 0)
            end_f = rep.get("end_frame", 0)
            duration_s = (end_f - start_f + 1) / fps if fps > 0 else 0.0
            
            # Slice temporal phases sequence
            rep_phases = frame_phases[start_f : end_f + 1] if frame_phases else []
            
            # Slice the 4 continuous biomechanical signals 
            rep_signals = {}
            if signals:
                for sig_key, sig_arr in signals.items():
                    if isinstance(sig_arr, list):
                        rep_signals[sig_key] = sig_arr[start_f : end_f + 1]

            rep_payload = {
                "rep_id": rep_id,
                "start_frame": start_f,
                "end_frame": end_f,
                "bottom_frame": rep.get("bottom_frame", 0),
                "duration_seconds": round(duration_s, 3),
                "phases": rep_phases,
                "signals": rep_signals,
                "heuristic_score": score_obj.get("overall_score", 0.0),
                "heuristic_metrics": {
                    "knee_valgus": metrics.get("knee_valgus"),
                    "forward_lean": metrics.get("forward_lean"),
                    "min_knee_angle": metrics.get("min_knee_angle"),
                    "squat_depth": metrics.get("squat_depth"),
                },
                "heuristic_metric_scores": score_obj.get("metric_scores", {}),
                "human_score": None,
                "flags": None,
            }

            preserved = preserved_reps.get(rep_id)
            if preserved:
                for key, value in preserved.items():
                    if value is not None:
                        rep_payload[key] = value

            reps.append(rep_payload)

        source_path_str = ""
        if source_video_path is not None:
            source_path_str = to_rel(str(source_video_path))
        elif isinstance(existing_annotation, dict):
            source_path_str = existing_annotation.get("source_video_path", "")

        return {
            "video_id": video_id,
            "source_video_path": source_path_str,
            "pipeline_run": run_name,
            "pipeline_outputs": {
                "features_json": to_rel(features_path),
                "segmented_json": to_rel(str(seg_path)),
                "scoring_json": to_rel(str(score_matches[0])) if score_data and score_matches else "",
                "visualization_video": to_rel(vis_video)
            },
            "view": view,
            "quality_rating": quality,
            "fps": fps,
            "calibration": seg_data.get("info", {}).get("calibration", {}),
            "graph_metadata": {
                "active_joints": [0, 1, 2, 11, 12, 23, 24, 25, 26, 27, 28],
                "zeroed_joints": [3, 4, 5, 6, 7, 8, 9, 10, 29, 30, 31, 32],
                "foot_consolidated": True
            },
            "total_reps": len(reps),
            "annotated_at": existing_annotation.get("annotated_at"),
            "reps": reps,
        }

    def _build_annotation_from_run(self, workspace: Path, run_name: str) -> bool:
        """Build and save annotation for the currently loaded video from pipeline output."""
        if not self.current_video_id:
            return False

        payload = self._build_annotation_payload_from_run(
            workspace=workspace,
            run_name=run_name,
            video_id=self.current_video_id,
            source_video_path=self.current_video_path,
        )
        if payload is None:
            return False

        self.current_annotation = payload
        self._save_current_annotation()
        return True

    def _refresh_annotation_for_video_from_run(self, workspace: Path, run_name: str, video_path: Path) -> bool:
        """Build and save annotation metadata for any processed video during batch runs."""
        payload = self._build_annotation_payload_from_run(
            workspace=workspace,
            run_name=run_name,
            video_id=video_path.stem,
            source_video_path=video_path,
        )
        if payload is None:
            return False

        self._save_annotation(video_path.stem, payload)
        return True

    # ============================================================ Auto-pipeline
    def _run_pipeline_for_video(self, video_path: Path) -> None:
        """Run Stages 2.5→4→5→8 for a single video, then load results."""
        self.log_frame.grid()  # show log
        self.log_text.delete("1.0", tk.END)
        self._pipeline_log(f"Running pipeline for: {video_path.name}")

        self.pipeline_running = True
        thread = threading.Thread(
            target=self._pipeline_thread, args=(video_path,), daemon=True
        )
        thread.start()

    def _terminate_subprocess_tree(
        self,
        process: subprocess.Popen | None,
        reason: str = "",
    ) -> None:
        """Terminate a subprocess and all children to avoid orphaned stage workers."""
        if process is None:
            return

        if process.poll() is not None:
            return

        if reason:
            self._pipeline_log(f"  ⚠ {reason}")

        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            else:
                process.terminate()
                process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

        try:
            process.wait(timeout=2)
        except Exception:
            pass

    def _run_pipeline_stage(self, stage: Stage, workspace: Path, video_stem: str) -> int:
        """Run a single stage with timeout and robust cleanup for child processes."""
        if self.pipeline_process is not None and self.pipeline_process.poll() is None:
            self._terminate_subprocess_tree(
                self.pipeline_process,
                reason="Detected lingering stage process. Cleaning it before starting next stage.",
            )
        self.pipeline_process = None

        stage_args = list(stage.args)
        if stage.key == "extract_selected_features":
            extraction_mode = self._get_selected_annotation_extraction_mode()
            
            # Handle dual mode: run filtered first (silently fail if Poor quality), then unfiltered
            if extraction_mode == "dual":
                # Run filtered first (may fail for Poor quality videos)
                try:
                    filtered_args = ["filtered"] + stage_args
                    filtered_cmd = [sys.executable, str(stage.script_path), *filtered_args]
                    
                    self._pipeline_log("[Dual Mode] Running filtered extraction (may skip for Poor quality)...")
                    result_filtered = subprocess.run(
                        filtered_cmd,
                        cwd=str(workspace),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if result_filtered.returncode == 0:
                        self._pipeline_log("[Dual Mode] Filtered extraction succeeded.")
                    else:
                        self._pipeline_log("[Dual Mode] Filtered extraction skipped (likely Poor quality - continuing with unfiltered).")
                except Exception as e:
                    self._pipeline_log(f"[Dual Mode] Filtered extraction failed: {e}. Continuing with unfiltered.")
                
                # Always run unfiltered (unconditional)
                stage_args = ["unfiltered"] + stage_args
            else:
                # Single mode (either filtered or unfiltered)
                stage_args = [extraction_mode] + stage_args
        elif stage.key == "scoring":
            stage_args = [video_stem]

        cmd = [sys.executable, str(stage.script_path), *stage_args]
        env = os.environ.copy()
        env["EXEVISION_MODEL_PATH"] = str(SHARED_MODEL_PATH)
        env["EXEVISION_FACE_MODEL_PATH"] = str(SHARED_FACE_MODEL_PATH)

        process = subprocess.Popen(
            cmd,
            cwd=str(workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.pipeline_process = process

        timeout_sec = int(self.stage_timeouts_sec.get(stage.key, 180))
        timed_out = threading.Event()

        def _watchdog() -> None:
            try:
                process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out.set()
                self._terminate_subprocess_tree(
                    process,
                    reason=f"{stage.label} timed out after {timeout_sec}s; terminating process tree.",
                )

        threading.Thread(target=_watchdog, daemon=True).start()

        important_tokens = (
            "error",
            "failed",
            "warning",
            "completed",
            "success",
            "classified",
            "score",
            "summary",
            "segmenting videos",
            "processing summary",
            "results",
        )

        try:
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip("\n")
                    if not line:
                        continue

                    lowered = line.lower()
                    if stage.key == "temporal_segmentation" or any(t in lowered for t in important_tokens):
                        self._pipeline_log(f"  {line}")

            rc = process.wait()
            if timed_out.is_set():
                return 124
            return rc
        finally:
            if process.poll() is None:
                self._terminate_subprocess_tree(
                    process,
                    reason=f"Cleaning up unfinished process for {stage.label}.",
                )
            self.pipeline_process = None

    def _pipeline_thread(self, video_path: Path) -> None:
        run_name = datetime.now().strftime("annotation_run_%Y%m%d_%H%M%S")
        run_root = RUNS_ROOT / run_name
        workspace = run_root / "workspace"
        logs_root = run_root / "logs"

        try:
            # Prepare workspace
            squat_dir = workspace / "squat"
            dataset_target = squat_dir / "dataset_videos_all"
            dataset_target.mkdir(parents=True, exist_ok=True)
            logs_root.mkdir(parents=True, exist_ok=True)

            # Copy single video
            shutil.copy2(video_path, dataset_target / video_path.name)
            self._pipeline_log(f"Workspace: {workspace}")

            # Run stages in order
            stages_to_run = [s for s in STAGES if s.key in (
                "extract_selected_features", "classify_views",
                "temporal_segmentation", "scoring"
            )]

            failed = False

            for i, stage in enumerate(stages_to_run, 1):
                self._pipeline_log(f"\n[{i}/{len(stages_to_run)}] {stage.label}")

                rc = self._run_pipeline_stage(stage, workspace, video_path.stem)
                if rc != 0:
                    failed = True
                    self._pipeline_log(f"\n  ❌ Stage {stage.key} FAILED with exit code {rc}")
                    if rc == 124:
                        self._pipeline_log(f"  ⚠ Reason: TIMEOUT (process took longer than allotted time)")
                    self._pipeline_log(f"  Skipping remaining stages.")
                    break

            if not failed:
                self._pipeline_log("\n✅ Pipeline complete.")
                self.pipeline_run_used = run_name

                # Load results
                self.root.after(0, self._on_pipeline_complete, workspace, run_name)
            else:
                self.root.after(
                    0, lambda: self.status_var.set(f"Pipeline failed for {video_path.name}. Check log output above.")
                )

        except Exception as exc:
            self._pipeline_log(f"\n❌ Pipeline failed: {exc}")
            self.root.after(
                0, lambda: self.status_var.set(f"Pipeline failed: {exc}")
            )
        finally:
            self._terminate_subprocess_tree(self.pipeline_process)
            self.pipeline_running = False
            self.pipeline_process = None

    def _batch_pipeline_thread(self, video_paths: list[Path]) -> None:
        """Run the pipeline sequentially for a list of videos."""
        total = len(video_paths)
        for i, video_path in enumerate(video_paths, 1):
            if not self.pipeline_running:
                break

            self.root.after(0, lambda v=video_path, c=i, t=total: self.status_var.set(
                f"Processing {v.name} ({c}/{t})..."
            ))
            
            self._pipeline_log(f"\n\n{'='*50}")
            self._pipeline_log(f"BATCH PROCESSING VIDEO {i}/{total}: {video_path.name}")
            self._pipeline_log(f"{'='*50}")
            
            # Use the existing single-video logic (blocking execution)
            run_name = datetime.now().strftime("annotation_run_%Y%m%d_%H%M%S")
            run_root = RUNS_ROOT / run_name
            workspace = run_root / "workspace"
            logs_root = run_root / "logs"

            try:
                # Prepare workspace
                squat_dir = workspace / "squat"
                dataset_target = squat_dir / "dataset_videos_all"
                dataset_target.mkdir(parents=True, exist_ok=True)
                logs_root.mkdir(parents=True, exist_ok=True)

                # Copy single video
                shutil.copy2(video_path, dataset_target / video_path.name)
                self._pipeline_log(f"Workspace: {workspace}")

                stages_to_run = [s for s in STAGES if s.key in (
                    "extract_selected_features", "classify_views",
                    "temporal_segmentation", "scoring"
                )]

                failed = False

                for j, stage in enumerate(stages_to_run, 1):
                    self._pipeline_log(f"\n[{j}/{len(stages_to_run)}] {stage.label}")

                    rc = self._run_pipeline_stage(stage, workspace, video_path.stem)
                    if rc != 0:
                        failed = True
                        self._pipeline_log(f"  ⚠ Stage exited with code {rc}. Skipping remaining stages for this video.")
                        break

                if failed:
                    self._pipeline_log(f"\n❌ Pipeline failed for {video_path.name}.")
                    continue

                self._pipeline_log(f"\n✅ Pipeline complete for {video_path.name}.")

                refreshed = self._refresh_annotation_for_video_from_run(workspace, run_name, video_path)
                if refreshed:
                    self._pipeline_log(f"  ✅ Annotation refreshed for {video_path.name}.")
                else:
                    self._pipeline_log(f"  ⚠ Pipeline completed, but annotation refresh failed for {video_path.name}.")

                # Keep live UI behavior only for the currently viewed video.
                if self.current_video_path == video_path:
                     self.pipeline_run_used = run_name
                     self.root.after(0, self._on_pipeline_complete, workspace, run_name)

            except Exception as exc:
                self._pipeline_log(f"\n❌ Pipeline failed for {video_path.name}: {exc}")
            finally:
                self._terminate_subprocess_tree(self.pipeline_process)
                self.pipeline_process = None
                
        self.pipeline_running = False
        self.root.after(0, lambda: self.status_var.set("Batch processing complete."))
        self.root.after(0, lambda: self.log_frame.grid_remove())
        self.root.after(0, lambda: self.loading_bar.stop())
        self.root.after(0, lambda: self.loading_bar.grid_remove())
        self.root.after(0, self._scan_folder)
        self.root.after(0, lambda: messagebox.showinfo("Processing Complete", f"Finished batch processing {total} video(s)."))

    def _pipeline_log(self, text: str) -> None:
        def _append():
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(0, _append)

    def _on_pipeline_complete(self, workspace: Path, run_name: str) -> None:
        if not self._build_annotation_from_run(workspace, run_name):
            self.status_var.set(f"No segmentation found for {self.current_video_id}.")
            return
        self.log_frame.grid_remove()  # hide log
        self.status_var.set(f"Pipeline complete. Annotating {self.current_video_id}.")
        self._advance_to_next_unannotated()
        self._show_current_rep()

    # ============================================================ Rep navigation
    def _advance_to_next_unannotated(self) -> None:
        """Set current_rep_idx to the first unannotated rep."""
        if not self.current_annotation:
            return
        reps = self.current_annotation.get("reps", [])
        for i, rep in enumerate(reps):
            if rep.get("human_score") is None:
                self.current_rep_idx = i
                return
        # All annotated — show the last one
        self.current_rep_idx = max(0, len(reps) - 1)

    def _show_current_rep(self) -> None:
        self._stop_playback()
        if not self.current_annotation:
            return

        reps = self.current_annotation.get("reps", [])
        if not reps:
            self.rep_info_var.set("No reps found in segmentation.")
            self.video_label.configure(image="", text="No reps detected.")
            self._photo = None
            return

        idx = self.current_rep_idx
        if idx >= len(reps):
            idx = len(reps) - 1
            self.current_rep_idx = idx

        rep = reps[idx]
        total = len(reps)
        scored = sum(1 for r in reps if r.get("human_score") is not None)

        self.rep_info_var.set(
            f"Video: {self.current_video_id}  |  "
            f"Rep {rep['rep_id']} of {total}  |  "
            f"View: {self.current_annotation.get('view', '?')}  |  "
            f"Frames {rep['start_frame']}→{rep['end_frame']}"
        )

        self.progress_var.set(f"Scored: {scored} / {total} reps in this video")

        # View combobox
        current_view = self.current_annotation.get("view", "unknown")
        self.view_var.set(current_view)
        
        # Populate heuristic info
        h_score = rep.get("heuristic_score", 0)
        metrics = rep.get("heuristic_metrics") or {}
        
        heur_text = f"Overall Score: {h_score:.1f}\n"
        metrics_strs = []
        if metrics.get("knee_valgus") is not None:
             metrics_strs.append(f"Knee Valgus: {metrics['knee_valgus']:.1f}°")
        if metrics.get("forward_lean") is not None:
             metrics_strs.append(f"Forward Lean: {metrics['forward_lean']:.1f}°")
        if metrics.get("squat_depth") is not None:
             metrics_strs.append(f"Depth Ratio: {metrics['squat_depth']:.2f}")
        if metrics.get("min_knee_angle") is not None:
             metrics_strs.append(f"Min Knee Angle: {metrics['min_knee_angle']:.1f}°")
             
        if metrics_strs:
            heur_text += " | ".join(metrics_strs)
        else:
            heur_text += "Metrics not available."
            
        self.heuristic_var.set(heur_text)

        # Show existing score or clear
        if rep.get("human_score") is not None:
            self.score_entry.configure(state="normal")
            self.score_entry.delete(0, "end")
            self.score_entry.insert(0, str(int(rep["human_score"])))
            h = rep.get("heuristic_score", 0)
            diff = rep["human_score"] - h
            # Restore flags
            rep_flags = rep.get("human_flags", rep.get("flags", {}))
            for k, v in self.flag_vars.items():
                v.set(rep_flags.get(k, False))
            
            # Restore Severities
            rep_sevs = rep.get("human_flag_severities", rep.get("flag_severities", {}))
            for k, v in self.flag_severity_vars.items():
                # Round and set because it's a DoubleVar
                val = rep_sevs.get(k, 1 if rep_flags.get(k) else 0)
                v.set(float(val))
                    
            # Restore Confidence
            if rep.get("annotator_confidence"):
                self.confidence_var.set(str(rep["annotator_confidence"]))
            else:
                self.confidence_var.set("4")
                
            # Restore Notes
            self.notes_var.set(rep.get("annotation_notes", ""))
            
            # Restore Metrics
            if rep.get("human_metric_scores"):
                hms = rep["human_metric_scores"]
                for m_key, m_var in self.metric_vars.items():
                    val = hms.get(m_key)
                    if val is not None:
                        # Assuming integers for simplicity but float works too
                        m_var.set(str(int(val)) if val.is_integer() else str(val))
                    else:
                        m_var.set("")
            else:
                for m_var in self.metric_vars.values():
                    m_var.set("")

        else:
            self.score_entry.configure(state="normal")
            self.score_entry.delete(0, "end")
            self.feedback_var.set("")
            for v in self.flag_vars.values():
                v.set(False)
            for v in self.flag_severity_vars.values():
                v.set(0)
            self.confidence_var.set("4")
            self.notes_var.set("")
            for v in self.metric_vars.values():
                v.set("")

        self.score_entry.focus_set()

        # Load both raw and visualized side-by-side
        self._load_dual_videos()

    def _prev_rep(self) -> None:
        if self.current_rep_idx > 0:
            self.current_rep_idx -= 1
            self._show_current_rep()

    def _next_rep(self) -> None:
        if not self.current_annotation:
            return
        reps = self.current_annotation.get("reps", [])
        if self.current_rep_idx < len(reps) - 1:
            self.current_rep_idx += 1
            self._show_current_rep()
        else:
            # All reps done for this video
            scored = sum(1 for r in reps if r.get("human_score") is not None)
            self.feedback_var.set(
                f"All {len(reps)} reps viewed! {scored} scored. "
                f"Select another video."
            )
            self._scan_folder()  # refresh list markers

    # ============================================================ Video playback
    def _load_dual_videos(self) -> None:
        if not self.current_annotation:
            return

        # Ensure playback loop is fully stopped before mutating capture objects.
        self._stop_playback()

        def _resolve_existing_path(path_text: str) -> str:
            if not path_text:
                return ""
            p = Path(path_text)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return str(p)

            # Migration compatibility: historical pipeline outputs may have moved
            # from pipeline_ui_runs/ to _hidden_legacy/pipeline_ui_runs/.
            normalized = path_text.replace("\\", "/")
            if normalized.startswith("pipeline_ui_runs/"):
                legacy_rel = normalized.replace("pipeline_ui_runs/", "", 1)
                legacy_path = LEGACY_RUNS_ROOT / legacy_rel
                if legacy_path.exists():
                    return str(legacy_path)

            return ""

        def _find_overlay_from_pipeline_outputs(video_id: str, quality: str) -> str:
            outputs = self.current_annotation.get("pipeline_outputs", {}) if self.current_annotation else {}
            seg_path_text = outputs.get("segmented_json", "")
            seg_path = Path(_resolve_existing_path(seg_path_text)) if seg_path_text else None

            candidates = []
            if seg_path and seg_path.exists():
                # .../workspace/squat/segmented_reps/<quality>/<video>_segmented.json
                squat_root = seg_path.parent.parent.parent
                vis_dir = squat_root / "visualized_segmentation" / quality
                candidates.extend([
                    vis_dir / f"{video_id}_phases.mp4",
                    vis_dir / f"{video_id}_segmented.mp4",
                    vis_dir / f"{video_id}_phases.avi",
                    vis_dir / f"{video_id}_segmented.avi",
                ])

            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
            return ""

        # 1. Load Raw
        raw_path = str(self.current_video_path) if self.current_video_path and self.current_video_path.exists() else None
        with self._annotation_cap_lock:
            if self.cap_raw:
                self.cap_raw.release()
            self.cap_raw = cv2.VideoCapture(raw_path) if raw_path else None

        # 2. Load Vis
        vis_path = self.current_annotation.get("pipeline_outputs", {}).get("visualization_video", "")
        resolved_vis_path = _resolve_existing_path(vis_path)

        # If the saved path is stale (or points to raw fallback), recover from pipeline outputs.
        if (not resolved_vis_path) or (raw_path and Path(resolved_vis_path) == Path(raw_path)):
            resolved_vis_path = _find_overlay_from_pipeline_outputs(
                self.current_annotation.get("video_id", ""),
                self.current_annotation.get("quality_rating", "").lower(),
            )

        with self._annotation_cap_lock:
            if self.cap_vis:
                self.cap_vis.release()
            self.cap_vis = cv2.VideoCapture(resolved_vis_path) if resolved_vis_path else None

        # Reset labels if not opened
        if not self.cap_raw or not self.cap_raw.isOpened():
            self.raw_label.configure(image="", text="Raw video not found.")
            self._photo_raw = None
            self.cap_raw = None
        if not self.cap_vis or not self.cap_vis.isOpened():
            self.vis_label.configure(image="", text="Visualized video not found.")
            self._photo_vis = None
            self.cap_vis = None

        # Set up rep boundaries
        reps = self.current_annotation.get("reps", [])
        if self.current_rep_idx < len(reps):
            rep = reps[self.current_rep_idx]
            self.rep_start = rep.get("start_frame", 0)
            self.rep_end = rep.get("end_frame", 0)
            self.fps = self.current_annotation.get("fps", 30.0) or 30.0

            if self.cap_raw: self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
            if self.cap_vis: self.cap_vis.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
            
            self._display_current_frames()
        
        self._start_playback()

    def _display_current_frames(self) -> None:
        """Display one frame from whichever captures are open."""
        with self._annotation_cap_lock:
            if self.cap_raw:
                ok, frame = self.cap_raw.read()
                if ok:
                    self._display_single_frame(frame, self.raw_label, "raw")
            if self.cap_vis:
                ok, frame = self.cap_vis.read()
                if ok:
                    self._display_single_frame(frame, self.vis_label, "vis")

    def _display_single_frame(self, frame, label_widget, kind: str) -> None:
        if Image is None or ImageTk is None:
            label_widget.configure(image="", text="Pillow not installed.")
            return
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Smaller size for side-by-side
        max_w, max_h = 480, 320
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        label_widget.configure(image=photo, text="")
        if kind == "raw":
            self._photo_raw = photo
        else:
            self._photo_vis = photo

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop_playback()
        else:
            if self.cap_raw or self.cap_vis:
                # Check if we need to loop
                with self._annotation_cap_lock:
                    cap = self.cap_raw or self.cap_vis
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if cur >= self.rep_end:
                        if self.cap_raw: self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
                        if self.cap_vis: self.cap_vis.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
                self._start_playback()

    def _replay(self) -> None:
        self._stop_playback()
        time.sleep(0.05)
        with self._annotation_cap_lock:
            if self.cap_raw: self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
            if self.cap_vis: self.cap_vis.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
        self._start_playback()

    def _start_playback(self) -> None:
        self.playing = True
        self.play_stop_event.clear()
        self.play_thread = threading.Thread(
            target=self._playback_loop, daemon=True)
        self.play_thread.start()

    def _stop_playback(self) -> None:
        self.playing = False
        self.play_stop_event.set()
        if self.play_thread is not None and self.play_thread.is_alive():
            try:
                self.play_thread.join(timeout=0.3)
            except Exception:
                pass
            self.play_thread = None

    def _playback_loop(self) -> None:
        while self.playing and not self.play_stop_event.is_set():
            with self._annotation_cap_lock:
                if not self.cap_raw and not self.cap_vis:
                    return

                # Check for loop in master capture (Raw if available, else Vis)
                master = self.cap_raw if self.cap_raw else self.cap_vis
                cur = int(master.get(cv2.CAP_PROP_POS_FRAMES))
                if cur >= self.rep_end:
                    if self.cap_raw: self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
                    if self.cap_vis: self.cap_vis.set(cv2.CAP_PROP_POS_FRAMES, self.rep_start)
            
            # Read and display (Tkinter calls must be on main thread)
            self.root.after(0, self._display_current_frames)
            
            time.sleep(max(1.0 / self.fps, 0.01))


    # ============================================================ Scoring
    def _on_scale_changed(self, value: str, key: str) -> None:
        """Handle manual slider dragging -> tick the box if > 0, snap to integer."""
        v = float(value)
        int_val = round(v)
        
        # Snap visually to integer to enforce discrete 0-5 steps
        if abs(v - int_val) > 0.01:
            self.flag_severity_vars[key].set(float(int_val))

        if int_val > 0:
            if self.flag_vars[key].get() == False:
                self.flag_vars[key].set(True)
                self._on_error_toggled(key)
        else:
            if self.flag_vars[key].get() == True:
                self.flag_vars[key].set(False)
                self._on_error_toggled(key)

    def _on_checkbox_toggled(self, key: str) -> None:
        """Handle manual checkbox toggle -> snap slider to 1 or 0."""
        if self.flag_vars[key].get():
            if self.flag_severity_vars[key].get() < 1:
                self.flag_severity_vars[key].set(1.0)
        else:
            self.flag_severity_vars[key].set(0.0)

    def _on_error_toggled(self, key: str) -> None:
        pass

    def _toggle_flag(self, key: str) -> None:
        var = self.flag_vars.get(key)
        if var is not None:
            var.set(not var.get())
            self._on_error_toggled(key)

    def _submit_score(self, event=None) -> None:
        if not self.current_annotation:
            return

        reps = self.current_annotation.get("reps", [])
        if self.current_rep_idx >= len(reps):
            return

        text = self.score_entry.get().strip()
        try:
            score = float(text)
            if not (0 <= score <= 100):
                raise ValueError
        except ValueError:
            self.feedback_var.set("⚠ Enter a number between 0 and 100.")
            return

        # Save view label update globally for the video
        new_view = getattr(self, "view_var", None)
        if new_view:
            val = new_view.get()
            if val:
                self.current_annotation["view"] = val

        rep = reps[self.current_rep_idx]
        
        h = rep.get("heuristic_score", 0.0)
        
        rep["human_score"] = score
        
        # Human Flags and Severities
        rep["human_flags"] = {k: v.get() for k, v in self.flag_vars.items()}
        rep["human_flag_severities"] = {k: int(round(v.get())) for k, v in self.flag_severity_vars.items()}
        
        # Annotator Confidence
        try:
            rep["annotator_confidence"] = int(self.confidence_var.get())
        except ValueError:
            rep["annotator_confidence"] = 4
            
        # Annotation Notes
        rep["annotation_notes"] = self.notes_var.get().strip()
        
        # Human Metric Scores (convert blanks to null)
        hms = {}
        for m_key, m_var in self.metric_vars.items():
            val_str = m_var.get().strip()
            if not val_str:
                hms[m_key] = None
            else:
                try:
                    hms[m_key] = float(val_str)
                except ValueError:
                    hms[m_key] = None
        rep["human_metric_scores"] = hms
        
        # Keep old 'flags' and 'flag_severities' fields for backward compatibility, 
        # or mirror human_flags to it to avoid breaking downstream scripts yet.
        rep["flags"] = rep["human_flags"]
        rep["flag_severities"] = rep["human_flag_severities"]

        # Update timestamp
        self.current_annotation["annotated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S")

        self._save_current_annotation()
        self._update_index()

        # Reveal heuristic
        h = rep.get("heuristic_score", 0.0)
        diff = score - h
        direction = "higher" if diff > 0 else "lower"
        self.feedback_var.set(
            f"Your score: {score:.0f}  |  "
            f"Heuristic: {h:.0f}  |  "
            f"Δ {diff:+.1f} pts ({direction})"
        )

        scored = sum(1 for r in reps if r.get("human_score") is not None)
        self.progress_var.set(f"Scored: {scored} / {len(reps)} reps in this video")

        # Auto-advance after delay
        self.root.after(1500, self._next_rep)

    def _skip_rep(self) -> None:
        self._next_rep()

    # ============================================================ Persistence
    def _manual_save(self) -> None:
        if self.current_annotation:
            self._save_current_annotation()
            self.feedback_var.set("✅ Saved annotation progress manually!")
        else:
            self.feedback_var.set("No video loaded to save.")

    def _save_current_annotation(self) -> None:
        if not self.current_annotation:
            return
        self._save_annotation(self.current_video_id, self.current_annotation)

    def _save_annotation(self, video_id: str, annotation: dict) -> None:
        ANNOTATIONS_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        path = ANNOTATIONS_VIDEOS_DIR / f"{video_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

    def _update_index(self) -> None:
        """Update the master index.json with summary stats."""
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

        index: dict = {}
        if ANNOTATIONS_INDEX.exists():
            try:
                with open(ANNOTATIONS_INDEX, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except Exception:
                index = {}

        if "videos" not in index:
            index["videos"] = {}

        ann = self.current_annotation
        if ann:
            reps = ann.get("reps", [])
            scored = sum(1 for r in reps if r.get("human_score") is not None)
            scores = [r["human_score"] for r in reps if r.get("human_score") is not None]
            index["videos"][self.current_video_id] = {
                "total_reps": len(reps),
                "scored_reps": scored,
                "view": ann.get("view", "unknown"),
                "pipeline_run": ann.get("pipeline_run", ""),
                "avg_human_score": round(sum(scores) / len(scores), 1) if scores else None,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        # Global stats
        total_vids = len(index["videos"])
        total_scored = sum(v.get("scored_reps", 0) for v in index["videos"].values())
        total_reps = sum(v.get("total_reps", 0) for v in index["videos"].values())
        index["summary"] = {
            "total_videos": total_vids,
            "total_reps": total_reps,
            "total_scored": total_scored,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(ANNOTATIONS_INDEX, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    # ============================================================ Analysis
    def _run_analysis(self) -> None:
        if not ANNOTATIONS_VIDEOS_DIR.exists():
            messagebox.showinfo("No Data", "No annotations saved yet.")
            return

        all_human = []
        all_heuristic = []
        for ann_file in ANNOTATIONS_VIDEOS_DIR.glob("*.json"):
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rep in data.get("reps", []):
                    if rep.get("human_score") is not None:
                        all_human.append(rep["human_score"])
                        all_heuristic.append(rep.get("heuristic_score", 0))
            except Exception:
                continue

        if not all_human:
            messagebox.showinfo("No Data", "No scored reps found.")
            return

        import numpy as np
        human = np.array(all_human)
        heuristic = np.array(all_heuristic)
        disagreement = human - heuristic
        corr = np.corrcoef(human, heuristic)[0, 1] if len(human) > 1 else 0

        lines = [
            "═" * 45,
            "  Annotation Quality Report",
            "═" * 45,
            f"  Total scored reps: {len(human)}",
            f"  Videos:            {len(list(ANNOTATIONS_VIDEOS_DIR.glob('*.json')))}",
            "",
            f"  Human score range: {human.min():.0f} – {human.max():.0f}",
            f"  Human score mean:  {human.mean():.1f} ± {human.std():.1f}",
            f"  Heuristic mean:    {heuristic.mean():.1f} ± {heuristic.std():.1f}",
            "",
            f"  Mean disagreement: {disagreement.mean():+.1f} pts",
            f"  Disagreement std:  {disagreement.std():.1f} pts",
            f"  Correlation (r):   {corr:.3f}",
            "",
            "  Training Balance Metric (Target: 15/bucket):",
        ]

        training_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        target_per_bucket = 15
        total_deficit = 0
        for lo, hi in training_bins:
            count = np.sum((human >= lo) & (human < hi + (1 if hi == 100 else 0)))
            deficit = max(0, target_per_bucket - count)
            total_deficit += deficit
            bar_len = int(min(20, count))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            deficit_str = f" (Deficit: {deficit:2d})" if deficit > 0 else " (OK)"
            lines.append(f"    {lo:3d}-{hi:3d}: {count:3d}  {bar} {deficit_str}")
        
        if total_deficit > 0:
            lines.append(f"\n  ⚠ Total deficit: {total_deficit} more needed.")
        else:
            lines.append("\n  ✅ All buckets meet target of 15.")
        lines.append("")

        if disagreement.std() < 3.0:
            lines.append("  ⚠ Low disagreement — scores match heuristic closely.")
        if human.std() < 10.0:
            lines.append("  ⚠ Low score variance — use full 0-100 range.")
        if corr > 0.95:
            lines.append("  ⚠ Very high correlation — focus on temporal quality.")
        if len(human) < 50:
            lines.append(f"  ⚠ Only {len(human)} scored — consider annotating more.")
        if not any("⚠" in l for l in lines):
            lines.append("  ✅ Annotations look good for training!")
        lines.append("═" * 45)

        top = tk.Toplevel(self.root)
        top.title("Annotation Quality Analysis")
        top.geometry("460x420")
        tw = tk.Text(top, wrap=tk.WORD, padx=10, pady=10,
                     font=("Consolas", 10))
        tw.pack(fill=tk.BOTH, expand=True)
        tw.insert(tk.END, "\n".join(lines))
        tw.configure(state="disabled")

    # ============================================================ Cleanup
    def cleanup(self) -> None:
        self._stop_playback()
        if self.cap_raw is not None:
            self.cap_raw.release()
            self.cap_raw = None
        if self.cap_vis is not None:
            self.cap_vis.release()
            self.cap_vis = None


# ---------------------------------------------------------------------------
# Main entry point — Notebook with Inference + Annotation tabs
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.title("ExeVision Pipeline")
    root.geometry("1400x900")
    root.resizable(True, True)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Tab 1: Inference (existing pipeline runner)
    inference_frame = ttk.Frame(notebook)
    notebook.add(inference_frame, text="  Inference  ")

    # Tab 2: Annotation
    annotation_frame = ttk.Frame(notebook)
    notebook.add(annotation_frame, text="  Annotation  ")

    # Build Inference UI — PipelineRunnerUI expects to own root,
    # so we pass root but transplant its children into the inference_frame.
    pipeline_ui = PipelineRunnerUI(root)

    # Move all PipelineRunnerUI widgets into the inference tab
    for child in list(root.winfo_children()):
        if child is not notebook:
            child.pack_forget()
            child.grid_forget()
            child.place_forget()
            child.pack(in_=inference_frame, fill=tk.BOTH, expand=True)

    # Build Annotation UI
    annotation_ui = AnnotationToolUI(annotation_frame)


    def _on_close() -> None:
        annotation_ui.cleanup()
        pipeline_ui._on_close()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
