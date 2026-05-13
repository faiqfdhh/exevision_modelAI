"""OHP Phase 3 annotation tool.

Completely separate from squat AnnotationToolUI. app.py imports and delegates here.

Discovers videos and pipeline outputs automatically from a single home directory:
  {home}/
    {video_id}.mp4                                         ← raw videos (root)
    {exercise}/
      visualized_poses_clean/{tier}/{id}_annotated.mp4
      visualized_segmentation/{tier}/{id}_phases.mp4
      analysis_reports/{tier}/...
      aqa_analysis_simple/{tier}/{id}/{id}_aqa_simple.json
      extracted_features_clean/{tier}/{id}.json
      segmented_reps/{tier}/{id}_segmented.json

No target JSON file required. Set home dir → Scan → pick video from dropdown.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import sys
import cv2
import numpy as np
from PIL import Image, ImageTk

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CONFIG_EXERCISES_DIR = _REPO / "core" / "exevision" / "config" / "exercises"
OHP_PHASE3_ANNOTATIONS_DIR = _REPO / "training_dataset" / "ohp_phase3_annotations" / "videos"

_DEFAULT_HOME = Path(os.environ.get(
    "EXEVISION_OHP_HOME",
    r"D:\FitnessAQA\ohp_phase3\personal_videos",
))

_EXTRACT_SCRIPT = _REPO / "core" / "exevision" / "stages" / "extract_selected_features.py"
_POSE_MODEL     = _REPO / "models" / "pose_landmarker_heavy.task"
_FACE_MODEL     = _REPO / "models" / "blaze_face_short_range.tflite"


def _load_ohp_config() -> dict:
    path = CONFIG_EXERCISES_DIR / "overhead_press.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_annotation(video_id: str) -> Optional[dict]:
    path = OHP_PHASE3_ANNOTATIONS_DIR / f"{video_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


class OHPPhase3AnnotatorWindow:
    """OHP Phase 3 annotation widget.

    Can be embedded in any Tk container (Frame or Toplevel).

    Layout:
      Workspace bar  — home dir, exercise selector, video dropdown, Scan
      Rep info panel — read-only pipeline metadata
      Annotation     — quality, temporal (BiLSTM), spatial (ST-GCN), meta
      Nav buttons    — Previous / Skip / Submit
    """

    def __init__(self, master: tk.Widget) -> None:
        self.master = master
        if isinstance(master, (tk.Toplevel, tk.Tk)):
            master.title("OHP Phase 3 Annotator")
            master.resizable(True, True)

        self._cfg = _load_ohp_config()

        # ── Core state ─────────────────────────────────────────────────────────
        self._home_dir = _DEFAULT_HOME
        self._home_dir_var       = tk.StringVar(value=str(self._home_dir))
        self._exercise_var       = tk.StringVar(value="overhead_press")
        self._selected_video_var = tk.StringVar(value="")
        self._discovered_videos: list[str] = []

        self._target_reps: list[dict] = []
        self._rep_index    = 0
        self._current_rep: dict = {}
        self._current_view = "front"

        self._video_frames:   list[np.ndarray] = []
        self._skel_frames:    list[np.ndarray] = []
        self._feature_frames: list[dict]       = []
        self._frame_index = 0

        self._extract_proc  = None
        self._extract_queue: list[str] = []
        self._extract_done  = 0
        self._extract_total = 0

        self._after_id: Optional[str] = None
        self._playing  = False
        self._fps      = 30.0

        self._display_raw:  Optional[tk.Widget] = None
        self._display_skel: Optional[tk.Widget] = None

        # ── Annotation vars ────────────────────────────────────────────────────
        self._quality_var     = tk.DoubleVar(value=50)
        self._lockout_var     = tk.IntVar(value=1)
        self._confidence_var  = tk.StringVar(value="3")
        self._notes_var       = tk.StringVar(value="")
        self._phase_label_var = tk.StringVar(value="—")
        self._heuristic_var   = tk.StringVar(value="")
        self._status_var      = tk.StringVar(value="Set home directory and click Scan.")
        self._info_view_var            = tk.StringVar(value="—")
        self._info_knee_var            = tk.StringVar(value="—")
        self._info_phases_var          = tk.StringVar(value="—")
        self._info_heuristic_metrics_var = tk.StringVar(value="—")
        self._view_annotation_var      = tk.StringVar(value="unknown")

        self._build_ui()
        if self._home_dir.exists():
            self.master.after(200, self._scan_home_dir)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Status bar
        top = ttk.Frame(self.master)
        top.pack(fill=tk.X, padx=5, pady=(5, 2))
        ttk.Label(top, textvariable=self._status_var, foreground="gray",
                  wraplength=300, font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._heuristic_var, foreground="navy",
                  font=("TkDefaultFont", 8)).pack(side=tk.RIGHT)

        # Workspace panel
        ws = ttk.LabelFrame(self.master, text="Workspace", padding=6)
        ws.pack(fill=tk.X, padx=5, pady=(2, 4))
        ws.columnconfigure(1, weight=1)

        ttk.Label(ws, text="Home dir:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ws, textvariable=self._home_dir_var).grid(
            row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(ws, text="…", width=2, command=self._pick_home_dir).grid(
            row=0, column=2, padx=(4, 0))

        ttk.Label(ws, text="Exercise:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Combobox(
            ws, textvariable=self._exercise_var,
            values=["overhead_press", "seated_overhead_press"],
            state="readonly", width=26,
        ).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(4, 0))

        ttk.Label(ws, text="Video:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._video_combo = ttk.Combobox(
            ws, textvariable=self._selected_video_var, state="readonly")
        self._video_combo.grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(4, 0))
        self._video_combo.bind("<<ComboboxSelected>>", self._on_video_selected)

        btn_bar = ttk.Frame(ws)
        btn_bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(btn_bar, text="Scan",              command=self._scan_home_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_bar, text="Extract This Video", command=self._extract_current).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(btn_bar, text="Batch Extract…",    command=self._batch_extract).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Rep info (read-only)
        info_frame = ttk.LabelFrame(self.master, text="Rep Info", padding=4)
        info_frame.pack(fill=tk.X, padx=5, pady=(2, 4))
        info_frame.columnconfigure(1, weight=1)

        def _irow(row: int, label: str, var: tk.StringVar) -> None:
            ttk.Label(info_frame, text=label, foreground="gray",
                      font=("TkDefaultFont", 8)).grid(row=row, column=0, sticky="w", padx=(0, 6))
            ttk.Label(info_frame, textvariable=var,
                      font=("TkDefaultFont", 8)).grid(row=row, column=1, sticky="w")

        _irow(0, "View:",       self._info_view_var)
        _irow(1, "Knee error:", self._info_knee_var)
        _irow(2, "Heuristics:", self._info_heuristic_metrics_var)
        _irow(3, "Phases:",     self._info_phases_var)

        # Annotation controls
        controls = ttk.LabelFrame(self.master, text="Annotation")
        controls.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._add_score_row(controls, "Quality (overall)", self._quality_var)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(controls, text="── Temporal (BiLSTM) ──", foreground="gray").pack(anchor=tk.W)

        _BILSTM_KEYS = {"smoothness", "control"}
        all_metrics: dict = self._cfg.get("annotation_metrics_phase3", {})
        self._metric_vars: dict[str, tk.DoubleVar] = {}
        for key, label in all_metrics.items():
            if key in _BILSTM_KEYS:
                var = tk.DoubleVar(value=50)
                self._add_score_row(controls, label, var)
                self._metric_vars[key] = var

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(controls, text="── Spatial (ST-GCN) ──", foreground="gray").pack(anchor=tk.W)

        l_row = ttk.Frame(controls)
        l_row.pack(fill=tk.X, pady=2)
        ttk.Label(l_row, text="Full Lockout:", width=20).pack(side=tk.LEFT)
        tk.Radiobutton(l_row, text="Yes (lockout achieved)", variable=self._lockout_var, value=1).pack(side=tk.LEFT)
        tk.Radiobutton(l_row, text="No (incomplete)",        variable=self._lockout_var, value=0).pack(side=tk.LEFT)

        _STGCN_KEYS = {"elbow_flare", "grip_ratio", "rom_top", "rom_bottom"}
        null_views: list[str] = self._cfg.get("annotation_grip_null_views", [])
        for key, label in all_metrics.items():
            if key not in _STGCN_KEYS:
                continue
            var = tk.DoubleVar(value=50)
            scale, _ = self._add_score_row(controls, label, var)
            self._metric_vars[key] = var
            if key == "grip_ratio":
                self._grip_scale     = scale
                self._grip_null_views = null_views

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(controls, text="── Meta ──", foreground="gray").pack(anchor=tk.W)

        v_row = ttk.Frame(controls)
        v_row.pack(fill=tk.X, pady=2)
        ttk.Label(v_row, text="View (annotated):", width=20).pack(side=tk.LEFT)
        ttk.Combobox(v_row, textvariable=self._view_annotation_var,
                     values=["front", "back", "side", "front_side", "back_side", "unknown"],
                     width=12, state="readonly").pack(side=tk.LEFT)

        c_row = ttk.Frame(controls)
        c_row.pack(fill=tk.X, pady=2)
        ttk.Label(c_row, text="Confidence (1-5):", width=20).pack(side=tk.LEFT)
        ttk.Combobox(c_row, textvariable=self._confidence_var,
                     values=["1", "2", "3", "4", "5"], width=5, state="readonly").pack(side=tk.LEFT)

        n_row = ttk.Frame(controls)
        n_row.pack(fill=tk.X, pady=2)
        ttk.Label(n_row, text="Notes:", width=20).pack(side=tk.LEFT)
        ttk.Entry(n_row, textvariable=self._notes_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        frame_nav = ttk.Frame(controls)
        frame_nav.pack(fill=tk.X, pady=4)
        ttk.Button(frame_nav, text="◀", width=3, command=self._prev_frame).pack(side=tk.LEFT)
        self._play_btn = ttk.Button(frame_nav, text="▶ Play", width=8, command=self._toggle_play)
        self._play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_nav, text="▶", width=3, command=self._next_frame).pack(side=tk.LEFT)
        ttk.Label(frame_nav, textvariable=self._phase_label_var, foreground="gray").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_row = ttk.Frame(self.master)
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_row, text="◀ Previous rep", command=self._prev_rep).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Skip",            command=self._advance_rep).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Submit ▶",        command=self._on_submit,
                   style="Accent.TButton").pack(side=tk.RIGHT)

    # ── Discovery ──────────────────────────────────────────────────────────────

    def _scan_home_dir(self) -> None:
        """Scan home dir root for raw video files and populate dropdown."""
        home = Path(self._home_dir_var.get())
        self._home_dir = home
        if not home.exists():
            messagebox.showwarning("Scan", f"Home directory does not exist:\n{home}")
            return

        seen: set[str] = set()
        ordered: list[str] = []
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            for p in sorted(home.glob(f"*{ext}")):
                vid = p.stem
                if vid not in seen:
                    seen.add(vid)
                    ordered.append(vid)

        self._discovered_videos = ordered
        self._video_combo["values"] = ordered
        n = len(ordered)
        self._status_var.set(f"Found {n} video{'s' if n != 1 else ''} in {home.name}/")
        if ordered:
            self._selected_video_var.set(ordered[0])
            self._load_video_by_id(ordered[0])

    def _on_video_selected(self, _event) -> None:
        vid = self._selected_video_var.get()
        if vid:
            self._load_video_by_id(vid)

    def _load_video_by_id(self, video_id: str) -> None:
        """Build _target_reps from AQA + segmented JSONs for video_id, then load rep 0."""
        self._home_dir = Path(self._home_dir_var.get())

        seg_data: dict = {}
        seg_path = self._find_segmented(video_id)
        if seg_path:
            try:
                seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        aqa_data: dict = {}
        aqa_path = self._find_aqa(video_id)
        if aqa_path:
            try:
                aqa_data = json.loads(aqa_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # view/fps: AQA top-level preferred, fall back to segmented info block
        seg_info = seg_data.get("info", {})
        view = (aqa_data.get("view")
                or seg_info.get("view")
                or seg_data.get("view", "unknown"))
        fps  = float(
            aqa_data.get("fps") or seg_info.get("fps") or seg_data.get("fps", 30.0) or 30.0
        )

        # AQA uses "repetitions"; segmented may use "reps" or "repetitions"
        aqa_reps: list[dict] = aqa_data.get("repetitions", [])
        seg_reps: list[dict] = seg_data.get("reps", seg_data.get("repetitions", []))

        # Build phases lookup from segmented by rep_id (both int and str keys)
        phases_by_id: dict[str, list] = {}
        for sr in seg_reps:
            rid = str(sr.get("rep_id", ""))
            if rid:
                phases_by_id[rid] = sr.get("phases", [])

        if aqa_reps:
            # Primary: AQA reps have start/end/score; supplement with phases from segmented
            self._target_reps = []
            for ar in aqa_reps:
                rep_id = str(ar.get("rep_id", ""))
                score  = ar.get("score", {})
                self._target_reps.append({
                    "video_id":                video_id,
                    "rep_id":                  rep_id,
                    "start_frame":             ar.get("start_frame", 0),
                    "end_frame":               ar.get("end_frame", -1),
                    "phases":                  phases_by_id.get(rep_id, []),
                    "heuristic_score":         score.get("overall_score", ar.get("overall_score")),
                    "heuristic_metric_scores": score.get("metric_scores", {}),
                    "knee_error":              None,
                    "view":                    view,
                    "fps":                     fps,
                    "stratum":                 "auto",
                })
        elif seg_reps:
            # AQA not yet run — use segmented reps for boundaries only
            self._target_reps = [{
                "video_id":                video_id,
                "rep_id":                  str(sr.get("rep_id", f"rep_{i + 1}")),
                "start_frame":             sr.get("start_frame", 0),
                "end_frame":               sr.get("end_frame", -1),
                "phases":                  sr.get("phases", []),
                "heuristic_score":         None,
                "heuristic_metric_scores": {},
                "knee_error":              sr.get("knee_error"),
                "view":                    view,
                "fps":                     fps,
                "stratum":                 "auto",
            } for i, sr in enumerate(seg_reps)]
        else:
            # Neither available — single full-video entry so annotation is still possible
            self._target_reps = [{
                "video_id":                video_id,
                "rep_id":                  "full",
                "start_frame":             0,
                "end_frame":               -1,
                "phases":                  [],
                "heuristic_score":         aqa_data.get("overall_score"),
                "heuristic_metric_scores": {},
                "knee_error":              None,
                "view":                    view,
                "fps":                     fps,
                "stratum":                 "auto",
            }]

        self._rep_index = 0
        self._load_rep(0)

    # ── Path finders ───────────────────────────────────────────────────────────

    def _ex(self) -> str:
        return self._exercise_var.get()

    def _find_video(self, video_id: str) -> Optional[Path]:
        home = self._home_dir
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            p = home / f"{video_id}{ext}"
            if p.exists():
                return p
        # Recursive fallback, skip pipeline subdirs
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            for p in home.rglob(f"{video_id}{ext}"):
                if self._ex() not in str(p):
                    return p
        return None

    def _find_visualized(self, video_id: str) -> Optional[Path]:
        base = self._home_dir / self._ex() / "visualized_poses_clean"
        for p in base.rglob(f"{video_id}_annotated.mp4"):
            return p
        return None

    def _find_segmented(self, video_id: str) -> Optional[Path]:
        base = self._home_dir / self._ex() / "segmented_reps"
        for p in base.rglob(f"{video_id}_segmented.json"):
            return p
        return None

    def _find_features(self, video_id: str) -> Optional[Path]:
        base = self._home_dir / self._ex() / "extracted_features_clean"
        for p in base.rglob(f"{video_id}.json"):
            return p
        return None

    def _find_aqa(self, video_id: str) -> Optional[Path]:
        base = self._home_dir / self._ex() / "aqa_analysis_simple"
        for p in base.rglob(f"{video_id}_aqa_simple.json"):
            return p
        return None

    # ── Directory picker ───────────────────────────────────────────────────────

    def _pick_home_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select home directory")
        if selected:
            self._home_dir_var.set(selected)
            self._home_dir = Path(selected)
            self._scan_home_dir()

    # ── Rep loading ────────────────────────────────────────────────────────────

    def _load_rep(self, index: int) -> None:
        self._playing = False
        if hasattr(self, "_play_btn"):
            self._play_btn.config(text="▶ Play")
        if self._after_id:
            self.master.after_cancel(self._after_id)
            self._after_id = None

        if not self._target_reps:
            self._status_var.set("No reps found. Select a video or run pipeline first.")
            return
        if index < 0 or index >= len(self._target_reps):
            self._status_var.set("All reps in this video reviewed.")
            return

        self._rep_index = index
        target   = self._target_reps[index]
        video_id = target.get("video_id", "")
        rep_id   = target.get("rep_id", "")
        self._fps = float(target.get("fps", 30.0)) or 30.0

        self._status_var.set(
            f"Rep {index + 1}/{len(self._target_reps)} | {video_id}  rep={rep_id}"
        )
        self._heuristic_var.set("")

        self._current_rep  = dict(target)
        self._current_view = target.get("view", "front")

        if hasattr(self, "_grip_scale"):
            if self._current_view in getattr(self, "_grip_null_views", []):
                self._grip_scale.config(state="disabled")
                if "grip_ratio" in self._metric_vars:
                    self._metric_vars["grip_ratio"].set(0)
            else:
                self._grip_scale.config(state="normal")

        phases = target.get("phases", [])
        if phases:
            self._phase_label_var.set(" → ".join(p.get("phase", "?") for p in phases)[:80])
        else:
            self._phase_label_var.set(f"view: {self._current_view}")

        self._info_view_var.set(self._current_view)
        self._view_annotation_var.set(self._current_view or "unknown")
        knee = target.get("knee_error")
        self._info_knee_var.set("—" if knee is None else f"{knee:.2f}")
        hm = target.get("heuristic_metric_scores") or {}
        self._info_heuristic_metrics_var.set(
            "  ".join(f"{k}={v:.0f}" for k, v in hm.items() if v is not None) or "—")
        self._info_phases_var.set(
            " → ".join(p.get("phase", "?") for p in phases) if phases else "—")

        # Pre-fill if already annotated
        existing = _load_existing_annotation(video_id)
        if existing:
            for er in existing.get("reps", []):
                if str(er.get("rep_id", "")) == str(rep_id) and er.get("human_score") is not None:
                    self._quality_var.set(int(er.get("human_score", 50)))
                    hms = er.get("human_metric_scores") or {}
                    for k, var in self._metric_vars.items():
                        val = hms.get(k)
                        var.set(int(val) if val is not None else 50)
                    self._lockout_var.set(1 if er.get("human_flags", {}).get("lockout", True) else 0)
                    self._confidence_var.set(str(er.get("annotator_confidence", 3)))
                    self._notes_var.set(er.get("annotation_notes", ""))
                    self._view_annotation_var.set(er.get("annotated_view", self._current_view) or "unknown")
                    break

        # Load frames
        self._video_frames   = []
        self._skel_frames    = []
        self._feature_frames = []

        video_path = self._find_video(video_id)
        if video_path:
            self._load_video_frames(video_path, target)
        else:
            self._status_var.set(self._status_var.get() + " | ⚠ raw video not found")

        vis_path = self._find_visualized(video_id)
        if vis_path:
            self._load_skel_frames(vis_path, target)
        else:
            feat_path = self._find_features(video_id)
            if feat_path:
                try:
                    self._feature_frames = json.loads(
                        feat_path.read_text(encoding="utf-8")).get("frames", [])
                except Exception:
                    self._feature_frames = []

        self._frame_index = 0
        self._show_frame(0)

    def _load_video_frames(self, video_path: Path, rep_dict: dict) -> None:
        start_f = int(rep_dict.get("start_frame", 0))
        end_f   = int(rep_dict.get("end_frame", -1))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames: list[np.ndarray] = []
        idx = start_f
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            idx += 1
            if end_f >= 0 and idx > end_f:
                break
            if len(frames) > 300:
                break
        cap.release()
        self._video_frames = frames

    def _load_skel_frames(self, video_path: Path, rep_dict: dict) -> None:
        start_f = int(rep_dict.get("start_frame", 0))
        end_f   = int(rep_dict.get("end_frame", -1))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames: list[np.ndarray] = []
        idx = start_f
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            idx += 1
            if end_f >= 0 and idx > end_f:
                break
            if len(frames) > 300:
                break
        cap.release()
        self._skel_frames = frames

    # ── Extraction ─────────────────────────────────────────────────────────────

    def _build_extract_cmd(self, extra_args: list[str]) -> list[str]:
        return [
            sys.executable, str(_EXTRACT_SCRIPT),
            "unfiltered",
            "--exercise", self._ex(),
            "--video-dir", str(self._home_dir),
            "--no-report",
        ] + extra_args

    def _start_next_extraction(self) -> None:
        import subprocess
        if not self._extract_queue:
            self._status_var.set(
                f"Batch extraction complete: {self._extract_done}/{self._extract_total} done.")
            if self._target_reps:
                self._load_rep(self._rep_index)
            return
        video_id = self._extract_queue.pop(0)
        self._extract_done += 1
        self._home_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["EXEVISION_MODEL_PATH"] = str(_POSE_MODEL)
        if _FACE_MODEL.exists():
            env["EXEVISION_FACE_MODEL_PATH"] = str(_FACE_MODEL)
        cmd = self._build_extract_cmd(["--video-id", video_id])
        remaining = len(self._extract_queue)
        self._status_var.set(
            f"Extracting {video_id} ({self._extract_done}/{self._extract_total})… {remaining} left")
        try:
            self._extract_proc = subprocess.Popen(
                cmd, cwd=str(self._home_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except Exception as e:
            messagebox.showerror("Extraction failed", str(e))
            self._extract_queue.clear()
            return
        self._poll_extraction(video_id)

    def _poll_extraction(self, video_id: str) -> None:
        if self._extract_proc is None:
            return
        rc = self._extract_proc.poll()
        if rc is None:
            self.master.after(500, lambda: self._poll_extraction(video_id))
            return
        if rc != 0:
            self._status_var.set(f"Extraction failed for {video_id} (rc={rc}). Continuing…")
        self._extract_proc = None
        if self._target_reps:
            current_vid = self._target_reps[self._rep_index].get("video_id", "")
            if video_id == current_vid:
                self._load_rep(self._rep_index)
        self._start_next_extraction()

    def _extract_current(self) -> None:
        vid = self._selected_video_var.get()
        if not vid:
            messagebox.showwarning("Extract", "No video selected.")
            return
        if self._extract_proc and self._extract_proc.poll() is None:
            messagebox.showinfo("Extraction", "Extraction already running — please wait.")
            return
        self._extract_queue = [vid]
        self._extract_done  = 0
        self._extract_total = 1
        self._start_next_extraction()

    def _batch_extract(self) -> None:
        if self._extract_proc and self._extract_proc.poll() is None:
            messagebox.showinfo("Extraction", "Extraction already running — please wait.")
            return
        from tkinter.simpledialog import askinteger
        n = askinteger(
            "Batch Extract",
            "How many videos to extract?\n(From current position, skips already-extracted)",
            minvalue=1, maxvalue=5000, initialvalue=10,
        )
        if n is None:
            return
        seen: set[str] = set()
        queue: list[str] = []
        start_idx = (self._discovered_videos.index(self._selected_video_var.get())
                     if self._selected_video_var.get() in self._discovered_videos else 0)
        for vid in self._discovered_videos[start_idx:]:
            if vid in seen:
                continue
            if self._find_visualized(vid) is not None:
                seen.add(vid)
                continue
            seen.add(vid)
            queue.append(vid)
            if len(queue) >= n:
                break
        if not queue:
            messagebox.showinfo("Batch Extract", "All videos in range already extracted.")
            return
        self._extract_queue = queue
        self._extract_done  = 0
        self._extract_total = len(queue)
        self._start_next_extraction()

    # ── Slider builder ─────────────────────────────────────────────────────────

    def _add_score_row(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.DoubleVar,
        from_: float = 0.0,
        to: float = 100.0,
        step: float = 1.0,
    ) -> tuple:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)

        scale = tk.Scale(
            row, from_=from_, to=to, variable=var,
            orient=tk.HORIZONTAL, resolution=1,
            showvalue=False, sliderlength=16, highlightthickness=0,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        spinbox = ttk.Spinbox(
            row, from_=from_, to=to, increment=step,
            textvariable=var, width=6, format="%.0f",
        )
        spinbox.pack(side=tk.LEFT)

        def _on_focusout(event, v=var, lo=from_, hi=to):
            try:
                v.set(int(round(max(lo, min(hi, float(v.get()))))))
            except (tk.TclError, ValueError):
                v.set(int(round((lo + hi) / 2)))

        spinbox.bind("<FocusOut>", _on_focusout)
        spinbox.bind("<Return>",   _on_focusout)

        def _on_scroll(event, v=var, lo=from_, hi=to, s=step):
            delta = s if (event.delta > 0 or event.num == 4) else -s
            try:
                v.set(max(lo, min(hi, int(round(v.get() + delta)))))
            except (tk.TclError, ValueError):
                pass
            return "break"

        for w in (row, scale, spinbox):
            w.bind("<MouseWheel>", _on_scroll)
            w.bind("<Button-4>",   _on_scroll)
            w.bind("<Button-5>",   _on_scroll)

        return scale, spinbox

    def _add_slider(self, parent: ttk.Frame, label: str, var: tk.DoubleVar) -> tuple:
        return self._add_score_row(parent, label, var)

    # ── Frame display ──────────────────────────────────────────────────────────

    def set_display_labels(
        self,
        raw_label: tk.Widget,
        skel_label: Optional[tk.Widget] = None,
    ) -> None:
        self._display_raw  = raw_label
        self._display_skel = skel_label
        self.master.after(150, self._deferred_rerender)

    def _deferred_rerender(self) -> None:
        if self._video_frames or self._feature_frames:
            self._show_frame(self._frame_index)
        elif self._display_raw is not None:
            try:
                self._display_raw.configure(image="", text="No video loaded.")
            except Exception:
                pass

    def _show_frame(self, frame_idx: int) -> None:
        n = max(len(self._video_frames), len(self._skel_frames), len(self._feature_frames), 0)
        if n == 0:
            return
        frame_idx = max(0, min(frame_idx, n - 1))
        self._frame_index = frame_idx

        base_text = self._phase_label_var.get().split(" |")[0]
        self._phase_label_var.set(f"{base_text} | frame {frame_idx + 1}/{n}")

        raw_rgb = None
        if self._video_frames:
            raw = self._video_frames[min(frame_idx, len(self._video_frames) - 1)]
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        skel_rgb = None
        if self._skel_frames:
            skel = self._skel_frames[min(frame_idx, len(self._skel_frames) - 1)]
            skel_rgb = cv2.cvtColor(skel, cv2.COLOR_BGR2RGB)
        elif self._feature_frames and frame_idx < len(self._feature_frames):
            base = raw_rgb if raw_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            skel_rgb = self._render_skeleton_frame(base, self._feature_frames[frame_idx])

        self._push_frame(self._display_raw,  raw_rgb)
        self._push_frame(self._display_skel, skel_rgb if skel_rgb is not None else raw_rgb)

    def _push_frame(self, widget: Optional[tk.Widget], img_rgb: Optional[np.ndarray]) -> None:
        if widget is None or img_rgb is None:
            return
        try:
            w = widget.winfo_width()  or 640
            h = widget.winfo_height() or 480
            if w < 10: w = 640
            if h < 10: h = 480
            pil_img = Image.fromarray(img_rgb).resize((w, h), Image.LANCZOS)
            photo   = ImageTk.PhotoImage(image=pil_img)
            widget.configure(image=photo, text="")
            widget.image = photo
        except Exception:
            pass

    def _prev_frame(self) -> None:
        self._playing = False
        if hasattr(self, "_play_btn"):
            self._play_btn.config(text="▶ Play")
        self._show_frame(self._frame_index - 1)

    def _next_frame(self) -> None:
        self._playing = False
        if hasattr(self, "_play_btn"):
            self._play_btn.config(text="▶ Play")
        self._show_frame(self._frame_index + 1)

    def _toggle_play(self) -> None:
        n = max(len(self._video_frames), len(self._skel_frames), len(self._feature_frames))
        if n == 0:
            return
        self._playing = not self._playing
        if self._playing:
            self._play_btn.config(text="⏸ Pause")
            self._play_tick()
        else:
            self._play_btn.config(text="▶ Play")

    def _play_tick(self) -> None:
        if not self._playing:
            return
        n = max(len(self._video_frames), len(self._skel_frames), len(self._feature_frames))
        if n == 0:
            self._playing = False
            return
        next_idx = self._frame_index + 1
        if next_idx >= n:
            next_idx = 0
        self._show_frame(next_idx)
        self._after_id = self.master.after(max(16, int(1000 / self._fps)), self._play_tick)

    def _render_skeleton_frame(self, frame: np.ndarray, frame_data: dict) -> np.ndarray:
        try:
            from core.exevision.utils.skeleton_overlay import draw_skeleton, extract_keypoints_from_frame
            xy, conf = extract_keypoints_from_frame(frame_data)
            return draw_skeleton(frame, xy, conf)
        except Exception:
            return frame

    # ── Annotation actions ─────────────────────────────────────────────────────

    def _on_submit(self) -> None:
        view = self._current_view
        rep  = dict(self._current_rep)

        rep["human_score"] = int(self._quality_var.get())
        rep["human_metric_scores"] = {
            k: (
                None if (k == "grip_ratio" and view in self._cfg.get("annotation_grip_null_views", []))
                else int(self._metric_vars[k].get())
            )
            for k in self._metric_vars
        }
        rep["human_flags"]          = {"lockout": bool(self._lockout_var.get())}
        rep["human_flag_severities"] = {}
        rep["annotated_view"]        = self._view_annotation_var.get()
        rep["annotator_confidence"]  = int(self._confidence_var.get())
        rep["annotation_notes"]      = self._notes_var.get().strip()

        video_id = rep.get("video_id") or (
            self._target_reps[self._rep_index].get("video_id") if self._target_reps else "unknown"
        )
        try:
            self._save_annotation(video_id, rep, view)
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return

        self._reveal_heuristic()
        self.master.after(800, self._advance_rep)

    def _save_annotation(self, video_id: str, rep: dict, view: str) -> None:
        OHP_PHASE3_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OHP_PHASE3_ANNOTATIONS_DIR / f"{video_id}.json"

        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {"video_id": video_id, "view": view, "reps": []}
        else:
            existing = {"video_id": video_id, "view": view, "reps": []}

        reps   = existing.get("reps", [])
        rep_id = str(rep.get("rep_id", ""))
        for i, er in enumerate(reps):
            if str(er.get("rep_id", "")) == rep_id:
                reps[i] = rep
                break
        else:
            reps.append(rep)
        existing["reps"] = reps

        out_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    def _reveal_heuristic(self) -> None:
        target    = self._target_reps[self._rep_index] if self._target_reps else {}
        heuristic = target.get("heuristic_score")
        if heuristic is not None:
            human = self._quality_var.get()
            delta = round(human - float(heuristic), 1)
            sign  = "+" if delta >= 0 else ""
            self._heuristic_var.set(f"Heuristic: {heuristic:.1f} | Δ = {sign}{delta}")

    def _advance_rep(self) -> None:
        next_idx = self._rep_index + 1
        if next_idx < len(self._target_reps):
            self._load_rep(next_idx)
        else:
            # Last rep in this video — advance to next video in dropdown
            self._advance_video()

    def _advance_video(self) -> None:
        vid = self._selected_video_var.get()
        if vid in self._discovered_videos:
            i = self._discovered_videos.index(vid)
            if i + 1 < len(self._discovered_videos):
                next_vid = self._discovered_videos[i + 1]
                self._selected_video_var.set(next_vid)
                self._load_video_by_id(next_vid)
            else:
                self._status_var.set("All videos reviewed.")
        else:
            self._status_var.set("All reps reviewed.")

    def _prev_rep(self) -> None:
        self._load_rep(self._rep_index - 1)
