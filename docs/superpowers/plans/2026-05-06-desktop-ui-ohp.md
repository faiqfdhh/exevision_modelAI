# Desktop UI — OHP & Seated OHP Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the desktop UI (`apps/desktop-ui/app.py`) to support `overhead_press` and `seated_overhead_press` in both the Inference tab (pipeline stages, results display) and the Annotation tab (exercise selector, config-driven flags, pipeline runner). All changes are surgical; squat behaviour must be unchanged.

**Architecture:** There are two independent exercise selectors — `self.exercise_var` (inference tab) and `self._annotation_exercise_var` (annotation tab). Every hardcoded `"squat"` path segment must become `self.exercise_var.get()` or `self._annotation_exercise_var.get()` depending on which tab the code belongs to. A helper `_config_file_for_exercise()` routes `seated_overhead_press → overhead_press.json`.

**Tech Stack:** Python 3.10, tkinter, existing stage scripts in `core/exevision/stages/`, exercise configs in `core/exevision/config/exercises/`.

**Baseline — what the file looks like NOW (before any edits):**
- `_build_stages()` returns a plain tuple including `analyze_results` unconditionally
- Inference exercise dropdown has `["squat", "overhead_press"]` only
- All output discovery functions (`_find_score_json`, `_find_annotated_videos`, `_update_metadata_display`, etc.) are hardcoded to `"squat"`
- Annotation pipeline thread (`_pipeline_thread`, `_batch_pipeline_thread`) hardcodes `workspace / "squat"` and passes no `--exercise` flag
- `_build_rep_diagnostics` hardcodes squat metric names; `_build_metric_diagnostic` KeyErrors on OHP metrics via `get_view_thresholds`

---

## File Map

| File | Change |
|------|--------|
| `apps/desktop-ui/app.py` | All UI changes — 11 targeted tasks |
| `core/exevision/config/exercises/squat.json` | Add `annotation_flags` + `annotation_metrics` |

---

## Key Constants (already defined in app.py)

```python
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]   # → exevision_modelAI/
STAGES_DIR     = WORKSPACE_ROOT / "core" / "exevision" / "stages"
RUNTIME_ROOT   = WORKSPACE_ROOT / "_hidden_legacy"
RUNS_ROOT      = WORKSPACE_ROOT / "pipeline_ui_runs"
```

Config exercises directory (added in Task 2):
```python
CONFIG_EXERCISES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "config" / "exercises"
```

---

## Task 1: Add `annotation_flags` + `annotation_metrics` to `squat.json`

**Files:** `core/exevision/config/exercises/squat.json`

`overhead_press.json` already has these keys. `squat.json` needs them so `_build_annotation_flag_defs()` works uniformly for both exercises.

- [ ] **Step 1: Add the two sections to squat.json**

Open `core/exevision/config/exercises/squat.json`. Before the final closing `}`, add a comma after the last existing key and insert:

```json
,
"annotation_flags": {
  "insufficient_squat_depth": "Insufficient Squat Depth",
  "knee_valgus":              "Knee Valgus",
  "lumbar_flexion":           "Lumbar Flexion",
  "heel_rise":                "Heel Rise",
  "asymmetric_descent":       "Asymmetric Descent",
  "forward_lean":             "Forward Lean"
},
"annotation_metrics": {
  "depth":      "Depth Quality",
  "stability":  "Stability",
  "smoothness": "Smoothness",
  "control":    "Control"
}
```

- [ ] **Step 2: Verify JSON parses**

```powershell
& ".venv\Scripts\python.exe" -c "import json; d=json.load(open('core/exevision/config/exercises/squat.json')); print('annotation_flags:', list(d['annotation_flags'].keys()))"
```

Expected: `annotation_flags: ['insufficient_squat_depth', 'knee_valgus', 'lumbar_flexion', 'heel_rise', 'asymmetric_descent', 'forward_lean']`

- [ ] **Step 3: Commit**

```
git add core/exevision/config/exercises/squat.json
git commit -m "config: add annotation_flags and annotation_metrics to squat.json"
```

---

## Task 2: Add `CONFIG_EXERCISES_DIR` constant + `_config_file_for_exercise()` helper

**Files:** `apps/desktop-ui/app.py` (after the `LEGACY_RUNS_ROOT` line, ~line 93)

- [ ] **Step 1: Insert after `LEGACY_RUNS_ROOT` line**

Find:
```python
RUNS_ROOT = WORKSPACE_ROOT / "pipeline_ui_runs"
LEGACY_RUNS_ROOT = RUNTIME_ROOT / "pipeline_ui_runs"
SHARED_MODEL_PATH = WORKSPACE_ROOT / "models" / "pose_landmarker_heavy.task"
```

Replace with:
```python
RUNS_ROOT = WORKSPACE_ROOT / "pipeline_ui_runs"
LEGACY_RUNS_ROOT = RUNTIME_ROOT / "pipeline_ui_runs"
CONFIG_EXERCISES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "config" / "exercises"


def _config_file_for_exercise(exercise: str) -> str:
    """Return config JSON stem for the given exercise.
    seated_overhead_press shares its config with overhead_press.
    """
    if exercise == "seated_overhead_press":
        return "overhead_press"
    return exercise


SHARED_MODEL_PATH = WORKSPACE_ROOT / "models" / "pose_landmarker_heavy.task"
```

- [ ] **Step 2: Smoke-test**

```powershell
& ".venv\Scripts\python.exe" -c "
import sys; sys.path.insert(0, 'apps/desktop-ui')
from app import _config_file_for_exercise, CONFIG_EXERCISES_DIR
assert _config_file_for_exercise('squat') == 'squat'
assert _config_file_for_exercise('overhead_press') == 'overhead_press'
assert _config_file_for_exercise('seated_overhead_press') == 'overhead_press'
assert CONFIG_EXERCISES_DIR.exists(), f'Missing: {CONFIG_EXERCISES_DIR}'
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "feat(ui): add CONFIG_EXERCISES_DIR constant and _config_file_for_exercise() helper"
```

---

## Task 3: Inference tab — dropdown + `_build_stages` + `_on_exercise_changed`

**Files:** `apps/desktop-ui/app.py`

Three changes: add `seated_overhead_press` to the inference dropdown, make `analyze_results` squat-only in `_build_stages`, and reset `dataset_var` in `_on_exercise_changed`.

- [ ] **Step 1: Update inference exercise dropdown (~line 270)**

Find:
```python
            values=["squat", "overhead_press"],
```

Replace with:
```python
            values=["squat", "overhead_press", "seated_overhead_press"],
```

- [ ] **Step 2: Replace `_build_stages` to make `analyze_results` squat-only (~lines 33–82)**

Find the entire function body (from `def _build_stages` to its closing `)`):
```python
def _build_stages(exercise: str = "squat") -> tuple[Stage, ...]:
    """Build stage definitions with exercise-specific paths."""
    return (
        Stage(
            key="extract_selected_features",
            label="2.5 Extract Selected Features",
            script_path=STAGES_DIR / "extract_selected_features.py",
            args=(),
            output_paths=(
                f"{exercise}/extracted_features_clean",
                f"{exercise}/visualized_poses_clean",
                f"{exercise}/analysis_reports",
            ),
        ),
        Stage(
            key="classify_views",
            label="4 Classify Views",
            script_path=STAGES_DIR / "classify_views.py",
            args=(),
            output_paths=(f"{exercise}/extracted_features_clean",),
        ),
        Stage(
            key="temporal_segmentation",
            label="5 Temporal Segmentation",
            script_path=STAGES_DIR / "temporal_segmentation.py",
            args=(),
            output_paths=(f"{exercise}/segmented_reps", f"{exercise}/visualized_segmentation"),
        ),
        Stage(
            key="scoring",
            label="8 Scoring",
            script_path=STAGES_DIR / "scoring.py",
            args=("*",),
            output_paths=(f"{exercise}/aqa_analysis_simple",),
        ),
        Stage(
            key="analyze_results",
            label="9 Analyze Results",
            script_path=RUNTIME_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py",
            args=(),
            output_paths=(f"{exercise}/aqa_analysis_simple/analysis_visualizations",),
        ),
        Stage(
            key="neural_fusion",
            label="9 Neural Fusion Scoring",
            script_path=STAGES_DIR / "neural_fusion_inference.py",
            args=(),
            output_paths=(f"{exercise}/neural_analysis",),
        ),
    )
```

Replace with:
```python
def _build_stages(exercise: str = "squat") -> tuple[Stage, ...]:
    """Build stage definitions with exercise-specific paths."""
    stages: list[Stage] = [
        Stage(
            key="extract_selected_features",
            label="2.5 Extract Selected Features",
            script_path=STAGES_DIR / "extract_selected_features.py",
            args=(),
            output_paths=(
                f"{exercise}/extracted_features_clean",
                f"{exercise}/visualized_poses_clean",
                f"{exercise}/analysis_reports",
            ),
        ),
        Stage(
            key="classify_views",
            label="4 Classify Views",
            script_path=STAGES_DIR / "classify_views.py",
            args=(),
            output_paths=(f"{exercise}/extracted_features_clean",),
        ),
        Stage(
            key="temporal_segmentation",
            label="5 Temporal Segmentation",
            script_path=STAGES_DIR / "temporal_segmentation.py",
            args=(),
            output_paths=(f"{exercise}/segmented_reps", f"{exercise}/visualized_segmentation"),
        ),
        Stage(
            key="scoring",
            label="8 Scoring",
            script_path=STAGES_DIR / "scoring.py",
            args=("*",),
            output_paths=(f"{exercise}/aqa_analysis_simple",),
        ),
    ]
    if exercise == "squat":
        stages.append(Stage(
            key="analyze_results",
            label="9 Analyze Results",
            script_path=RUNTIME_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py",
            args=(),
            output_paths=(f"{exercise}/aqa_analysis_simple/analysis_visualizations",),
        ))
    stages.append(Stage(
        key="neural_fusion",
        label="9 Neural Fusion Scoring",
        script_path=STAGES_DIR / "neural_fusion_inference.py",
        args=(),
        output_paths=(f"{exercise}/neural_analysis",),
    ))
    return tuple(stages)
```

- [ ] **Step 3: Update `_on_exercise_changed` to reset `dataset_var` (~lines 219–229)**

Find:
```python
    def _on_exercise_changed(self) -> None:
        """Update STAGES when exercise selection changes."""
        global STAGES
        STAGES = _build_stages(self.exercise_var.get())
        self._log(f"Exercise changed to: {self.exercise_var.get()}")
        # Update custom stage checkboxes
        for key in self.stage_checks:
            self.stage_checks[key].set(False)
        for stage in STAGES if STAGES else []:
            if stage.key in self.stage_checks:
                self.stage_checks[stage.key].set(True)
```

Replace with:
```python
    def _on_exercise_changed(self) -> None:
        """Update STAGES when exercise selection changes."""
        global STAGES
        exercise = self.exercise_var.get()
        STAGES = _build_stages(exercise)
        self._log(f"Exercise changed to: {exercise}")
        for key in self.stage_checks:
            self.stage_checks[key].set(False)
        for stage in STAGES if STAGES else []:
            if stage.key in self.stage_checks:
                self.stage_checks[stage.key].set(True)
        # Squat has a known default dataset path; OHP videos are browsed manually
        if exercise == "squat":
            self.dataset_var.set(str(RUNTIME_ROOT / "squat" / "dataset_videos_all"))
        else:
            self.dataset_var.set("")
```

- [ ] **Step 4: Launch and verify**

```powershell
& ".venv\Scripts\python.exe" apps/desktop-ui/app.py
```

- Inference tab → change exercise to `overhead_press` → `analyze_results` checkbox disappears, dataset path clears
- Change to `seated_overhead_press` → same behaviour
- Change back to `squat` → `analyze_results` reappears, default dataset path restores

- [ ] **Step 5: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "feat(ui): add seated_overhead_press to inference dropdown; make analyze_results squat-only; reset dataset path on exercise change"
```

---

## Task 4: Fix inference hardcoded `"squat"` path references

**Files:** `apps/desktop-ui/app.py`

All these replace a literal `"squat"` directory segment with `self.exercise_var.get()` so pipeline output discovery works for any exercise. Do each step in order; the exact surrounding context in the "Find" blocks makes each replacement unique.

- [ ] **Step 1: Guard `analyze_src` copy for squat only (~line 749)**

Find:
```python
        # Ensure local copy of analyze_results.py exists inside workspace
        exercise = self.exercise_var.get()
        analyze_src = RUNTIME_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py"
        analyze_dst = workspace_root / exercise / "aqa_analysis_simple" / "analyze_results.py"
        analyze_dst.parent.mkdir(parents=True, exist_ok=True)
        if analyze_src.exists():
            shutil.copy2(analyze_src, analyze_dst)
```

Replace with:
```python
        # Ensure local copy of analyze_results.py exists inside workspace (squat only)
        exercise = self.exercise_var.get()
        if exercise == "squat":
            analyze_src = RUNTIME_ROOT / "squat" / "aqa_analysis_simple" / "analyze_results.py"
            analyze_dst = workspace_root / exercise / "aqa_analysis_simple" / "analyze_results.py"
            analyze_dst.parent.mkdir(parents=True, exist_ok=True)
            if analyze_src.exists():
                shutil.copy2(analyze_src, analyze_dst)
```

- [ ] **Step 2: Fix `_find_score_json` (~line 1041)**

Find:
```python
    def _find_score_json(self, run_root: Path, video_id: str) -> Path | None:
        search_roots = [
            run_root / "workspace" / "squat" / "aqa_analysis_simple",
            run_root / "stage_outputs",
        ]
```

Replace with:
```python
    def _find_score_json(self, run_root: Path, video_id: str) -> Path | None:
        search_roots = [
            run_root / "workspace" / self.exercise_var.get() / "aqa_analysis_simple",
            run_root / "stage_outputs",
        ]
```

- [ ] **Step 3: Fix `_find_analysis_summary_json` (~line 1055)**

Find:
```python
    def _find_analysis_summary_json(self, run_root: Path, video_id: str) -> Path | None:
        search_roots = [
            run_root / "workspace" / "squat" / "aqa_analysis_simple" / "analysis_visualizations",
            run_root / "stage_outputs",
        ]
```

Replace with:
```python
    def _find_analysis_summary_json(self, run_root: Path, video_id: str) -> Path | None:
        search_roots = [
            run_root / "workspace" / self.exercise_var.get() / "aqa_analysis_simple" / "analysis_visualizations",
            run_root / "stage_outputs",
        ]
```

- [ ] **Step 4: Fix `_find_neural_json` (~line 1069)**

Find:
```python
    def _find_neural_json(self, run_root: Path, video_id: str) -> Path | None:
        """Find neural fusion scoring JSON for a video."""
        search_roots = [
            run_root / "workspace" / "squat" / "neural_analysis",
            run_root / "stage_outputs",
        ]
```

Replace with:
```python
    def _find_neural_json(self, run_root: Path, video_id: str) -> Path | None:
        """Find neural fusion scoring JSON for a video."""
        search_roots = [
            run_root / "workspace" / self.exercise_var.get() / "neural_analysis",
            run_root / "stage_outputs",
        ]
```

- [ ] **Step 5: Fix `_find_annotated_videos` search roots (~line 967)**

Find:
```python
        search_roots = [
            run_root / "workspace" / "squat" / "visualized_poses_clean",
            run_root / "workspace" / "squat" / "visualized_segmentation",
            run_root / "stage_outputs",
        ]
```

Replace with:
```python
        _ex = self.exercise_var.get()
        search_roots = [
            run_root / "workspace" / _ex / "visualized_poses_clean",
            run_root / "workspace" / _ex / "visualized_segmentation",
            run_root / "stage_outputs",
        ]
```

- [ ] **Step 6: Fix `dataset_dir` in `_set_preview_outputs` (~line 1011)**

Find:
```python
        dataset_dir = run_root / "workspace" / "squat" / "dataset_videos_all"
```

Replace with:
```python
        dataset_dir = run_root / "workspace" / self.exercise_var.get() / "dataset_videos_all"
```

- [ ] **Step 7: Fix `dataset_root` in `_find_original_video_for_overlay` (~line 1401)**

Find:
```python
            dataset_root = run_root / "workspace" / "squat" / "dataset_videos_all"
```

Replace with:
```python
            dataset_root = run_root / "workspace" / self.exercise_var.get() / "dataset_videos_all"
```

- [ ] **Step 8: Fix `segmented_reps` search dir in `_load_selected_preview` for missing outputs (~line 1431)**

Find:
```python
                 search_dirs = [
                     run_root / "stage_outputs" / "03_temporal_segmentation",
                     run_root / "workspace" / "squat" / "segmented_reps"
                 ]
```

Replace with:
```python
                 search_dirs = [
                     run_root / "stage_outputs" / "03_temporal_segmentation",
                     run_root / "workspace" / self.exercise_var.get() / "segmented_reps"
                 ]
```

- [ ] **Step 9: Fix `dummy_path` in `_load_selected_preview` (~line 1459)**

Find:
```python
            dummy_path = run_root / "workspace" / "squat" / "segmented_reps" / "excellent" / f"{stem}_segmented.json" if run_root else Path(f"{stem}_segmented.json")
```

Replace with:
```python
            dummy_path = run_root / "workspace" / self.exercise_var.get() / "segmented_reps" / "excellent" / f"{stem}_segmented.json" if run_root else Path(f"{stem}_segmented.json")
```

- [ ] **Step 10: Fix `_update_metadata_display` — rename `squat_root` → `exercise_root` and update downstream uses (~lines 1532–1567)**

Find:
```python
            # Search upwards for the 'squat' directory to anchor our path
            squat_root = None
            for parent in overlay_path.parents:
                if parent.name == "squat":
                    squat_root = parent
                    break
            
            if squat_root is None:
                return
```

Replace with:
```python
            # Search upwards for the exercise directory to anchor our path
            _exercise_names = {"squat", "overhead_press", "seated_overhead_press"}
            exercise_root = None
            for parent in overlay_path.parents:
                if parent.name in _exercise_names:
                    exercise_root = parent
                    break

            if exercise_root is None:
                return
```

Then find (still within `_update_metadata_display`):
```python
                # .../squat/segmented_reps/<quality>/<vid>_segmented.json
                json_path = squat_root / "segmented_reps" / quality_folder / f"{stem_clean}_segmented.json"
```

Replace with:
```python
                # .../{exercise}/segmented_reps/<quality>/<vid>_segmented.json
                json_path = exercise_root / "segmented_reps" / quality_folder / f"{stem_clean}_segmented.json"
```

Then find:
```python
                # .../squat/extracted_features_clean/<quality>/<vid>.json
                json_path = squat_root / "extracted_features_clean" / quality_folder / f"{stem_clean}.json"
```

Replace with:
```python
                # .../{exercise}/extracted_features_clean/<quality>/<vid>.json
                json_path = exercise_root / "extracted_features_clean" / quality_folder / f"{stem_clean}.json"
```

- [ ] **Step 11: Verify no remaining hardcoded `"squat"` path segments in inference methods**

```powershell
& ".venv\Scripts\python.exe" -c "
src = open('apps/desktop-ui/app.py', encoding='utf-8').read()
lines = src.splitlines()
hits = [(i+1, l.strip()) for i, l in enumerate(lines)
        if '\"squat\"' in l and 'annotation' not in l.lower()
        and any(seg in l for seg in ['dataset_videos_all', 'segmented_reps', 'visualized', 'aqa_analysis', 'neural_analysis'])]
for ln, txt in hits:
    print(f'  line {ln}: {txt}')
print(f'Remaining hits: {len(hits)}')
"
```

Expected: `Remaining hits: 0`

- [ ] **Step 12: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "fix(ui): replace hardcoded squat path segments with exercise_var in all inference output discovery methods"
```

---

## Task 5: Make scoring display exercise-aware (`_metric_specs` + `_build_rep_diagnostics` + `_build_metric_diagnostic`)

**Files:** `apps/desktop-ui/app.py`

Three methods need changes. Failing to fix all three means OHP scores display no metrics and can crash with `KeyError` on `get_view_thresholds`.

- [ ] **Step 1: Replace `_metric_specs` (~line 1101)**

Find:
```python
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
```

Replace with:
```python
    def _metric_specs(self) -> dict[str, dict[str, str]]:
        if self.exercise_var.get() in ("overhead_press", "seated_overhead_press"):
            return {
                "grip_ratio": {
                    "label": "Grip width",
                    "source_key": "grip_ratio",
                    "unit": "ratio",
                    "evaluation": "direct",
                },
                "rom": {
                    "label": "ROM (elbow flexion)",
                    "source_key": "min_elbow_angle",
                    "unit": "deg",
                    "evaluation": "direct",
                },
                "lockout": {
                    "label": "Lockout extension",
                    "source_key": "max_elbow_angle",
                    "unit": "deg",
                    "evaluation": "direct",
                },
                "elbow_flare": {
                    "label": "Elbow flare",
                    "source_key": "elbow_flare_mean",
                    "unit": "deg",
                    "evaluation": "direct",
                },
            }
        # squat (default)
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
```

- [ ] **Step 2: Fix `_build_metric_diagnostic` — guard `get_view_thresholds` for missing metric keys (~line 1153)**

`get_view_thresholds` only contains squat metrics. For OHP metrics it would `KeyError`. Fix by returning `None` when the metric has no threshold entry.

Find:
```python
        threshold = get_view_thresholds(view)[metric_name]
        evaluated_value = abs(raw_value) if specs["evaluation"] == "absolute" else raw_value
```

Replace with:
```python
        threshold = get_view_thresholds(view).get(metric_name)
        if threshold is None:
            return None
        evaluated_value = abs(raw_value) if specs["evaluation"] == "absolute" else raw_value
```

- [ ] **Step 3: Fix `_build_rep_diagnostics` — use dynamic metric names instead of hardcoded squat list (~line 1177)**

Find:
```python
    def _build_rep_diagnostics(self, rep: dict, view: str) -> list[dict]:
        diagnostics = []
        for metric_name in ("knee_valgus", "forward_lean", "depth", "squat_depth"):
            detail = self._build_metric_diagnostic(metric_name, rep, view)
            if detail is not None:
                diagnostics.append(detail)
        diagnostics.sort(key=lambda item: item["metric_score"])
        return diagnostics
```

Replace with:
```python
    def _build_rep_diagnostics(self, rep: dict, view: str) -> list[dict]:
        diagnostics = []
        for metric_name in self._metric_specs():
            detail = self._build_metric_diagnostic(metric_name, rep, view)
            if detail is not None:
                diagnostics.append(detail)
        diagnostics.sort(key=lambda item: item["metric_score"])
        return diagnostics
```

- [ ] **Step 4: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "feat(ui): make scoring display exercise-aware (metric_specs, build_rep_diagnostics, guard get_view_thresholds)"
```

---

## Task 6: Add annotation exercise state variable + UI combobox

**Files:** `apps/desktop-ui/app.py` (~`AnnotationToolUI.__init__` and `_build_layout`)

The annotation tab gets its own exercise selector, independent of the inference tab.

- [ ] **Step 1: Add `_annotation_exercise_var` to `AnnotationToolUI.__init__`**

Find (in `AnnotationToolUI.__init__`):
```python
        self.annotation_extraction_mode_var = tk.StringVar(value="Filtered")
```

Insert immediately before that line:
```python
        self._annotation_exercise_var = tk.StringVar(value="squat")
```

- [ ] **Step 2: Add exercise combobox inside `folder_frame`**

Find:
```python
        self.folder_var = tk.StringVar(value="")
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        folder_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_frame, text="Browse…", width=8,
                   command=self._pick_folder).grid(row=0, column=1, padx=(4, 0))
```

Replace with:
```python
        self.folder_var = tk.StringVar(value="")
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        folder_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_frame, text="Browse…", width=8,
                   command=self._pick_folder).grid(row=0, column=1, padx=(4, 0))

        # Exercise selector — row 1 inside folder_frame
        ttk.Label(folder_frame, text="Exercise:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ann_exercise_combo = ttk.Combobox(
            folder_frame,
            textvariable=self._annotation_exercise_var,
            values=["squat", "overhead_press", "seated_overhead_press"],
            state="readonly",
            width=22,
        )
        ann_exercise_combo.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        ann_exercise_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._on_annotation_exercise_changed(),
        )
```

- [ ] **Step 3: Update status_var default message**

Find:
```python
        self.status_var = tk.StringVar(value="Pick a folder of squat videos to begin.")
```

Replace with:
```python
        self.status_var = tk.StringVar(value="Pick a folder of videos to begin.")
```

- [ ] **Step 4: Add `_on_annotation_exercise_changed()` method to `AnnotationToolUI`**

Add this method near the other annotation event handlers:

```python
    def _on_annotation_exercise_changed(self) -> None:
        """Rebuild annotation flags when exercise selection changes in annotation tab."""
        self._rebuild_annotation_flags()
        exercise = self._annotation_exercise_var.get()
        self.status_var.set(f"Exercise changed to {exercise}. Pick a folder of videos.")
        self.video_files = []
        self.folder_var.set("")
        if hasattr(self, "_video_listbox"):
            self._video_listbox.delete(0, tk.END)
```

- [ ] **Step 5: Launch and verify**

```powershell
& ".venv\Scripts\python.exe" apps/desktop-ui/app.py
```

Open Annotation tab → "Video Folder" section shows "Exercise:" dropdown with squat / overhead_press / seated_overhead_press.

- [ ] **Step 6: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "feat(ui): add exercise selector to annotation tab with _on_annotation_exercise_changed handler"
```

---

## Task 7: Config-driven annotation flags

**Files:** `apps/desktop-ui/app.py`

Replace the hardcoded squat flag list with a rebuildable method that loads from the exercise config JSON.

- [ ] **Step 1: Store `flags_frame` as instance variable**

Find:
```python
        flags_frame = ttk.LabelFrame(left, text="Form Errors", padding=8)
        flags_frame.grid(row=8, column=0, sticky="ew", pady=(8, 0))
```

Replace with:
```python
        self._flags_frame = ttk.LabelFrame(left, text="Form Errors", padding=8)
        self._flags_frame.grid(row=8, column=0, sticky="ew", pady=(8, 0))
```

Then inside the `_toggle_list` closure or any other place `flags_frame` is referenced (search the file for remaining `flags_frame.grid` / `flags_frame.grid_remove`), replace each with `self._flags_frame.grid` / `self._flags_frame.grid_remove`.

- [ ] **Step 2: Add `_build_annotation_flag_defs()` method**

```python
    def _build_annotation_flag_defs(self) -> list[tuple[str, str]]:
        """Load annotation flag definitions from the exercise config JSON."""
        exercise = self._annotation_exercise_var.get()
        config_stem = _config_file_for_exercise(exercise)
        config_path = CONFIG_EXERCISES_DIR / f"{config_stem}.json"
        if not config_path.exists():
            return []
        try:
            import json as _json
            with open(config_path, encoding="utf-8") as f:
                cfg = _json.load(f)
            return list(cfg.get("annotation_flags", {}).items())
        except Exception:
            return []
```

- [ ] **Step 3: Add `_rebuild_annotation_flags()` method**

```python
    def _rebuild_annotation_flags(self) -> None:
        """Destroy existing flag widgets and rebuild from current exercise config."""
        for widget in self._flags_frame.winfo_children():
            widget.destroy()

        self.flag_vars = {}
        self.flag_severity_vars = {}

        for i, (key, label) in enumerate(self._build_annotation_flag_defs()):
            var = tk.BooleanVar(value=False)
            self.flag_vars[key] = var
            sev_var = tk.DoubleVar(value=0)
            self.flag_severity_vars[key] = sev_var

            row_frame = ttk.Frame(self._flags_frame)
            row_frame.grid(row=i, column=0, sticky="ew", pady=2)
            row_frame.columnconfigure(0, weight=1)

            ttk.Checkbutton(
                row_frame, text=label, variable=var,
                command=lambda k=key: self._on_checkbox_toggled(k),
            ).grid(row=0, column=0, sticky="w")

            sev_var_label = tk.DoubleVar(value=0)
            self.flag_severity_vars[key] = sev_var_label
            ttk.Scale(
                row_frame, from_=0, to=5, variable=sev_var_label,
                orient=tk.HORIZONTAL, length=80,
                command=lambda v, k=key: self._on_scale_changed(v, k),
            ).grid(row=0, column=1, padx=(10, 0))

            ttk.Label(row_frame, textvariable=sev_var_label, width=3).grid(row=0, column=2, padx=(4, 0))

        # Rebind F1–F7 hotkeys to the new flag list
        flag_keys_list = list(self.flag_vars.keys())
        for idx in range(min(7, len(flag_keys_list))):
            self.root.bind(f"<F{idx + 1}>", lambda e, k=flag_keys_list[idx]: self._toggle_flag(k))
```

- [ ] **Step 4: Replace the hardcoded flag block with `_rebuild_annotation_flags()` call**

Find the entire hardcoded block:
```python
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
```

Replace with:
```python
        self.flag_vars: dict[str, tk.BooleanVar] = {}
        self.flag_severity_vars: dict[str, tk.DoubleVar] = {}
        self._rebuild_annotation_flags()
```

- [ ] **Step 5: Launch and verify**

```powershell
& ".venv\Scripts\python.exe" apps/desktop-ui/app.py
```

- Annotation tab → squat flags: "Insufficient Squat Depth", "Knee Valgus", etc.
- Change to `overhead_press` → flags change to: "Incomplete Lockout", "Elbow Flare / Winging", "Excessive Layback", "Bar Path Drift", "Wrist Bent Back", "Knee Instability (standing only)"
- Change to `seated_overhead_press` → same OHP flags (reads from `overhead_press.json`)
- F1–F7 hotkeys toggle first 7 flags in each mode

- [ ] **Step 6: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "feat(ui): config-driven annotation flags via _rebuild_annotation_flags; reads from overhead_press.json / squat.json"
```

---

## Task 8: Fix annotation tab hardcoded `"squat"` paths — payload builders

**Files:** `apps/desktop-ui/app.py`

- [ ] **Step 1: Fix `_pick_folder` default (~line 2262)**

Find:
```python
    def _pick_folder(self) -> None:
        default = str(PROJECT_ROOT / "squat" / "dataset_videos_all")
        selected = filedialog.askdirectory(
            initialdir=default if Path(default).exists() else str(PROJECT_ROOT),
            title="Select folder of squat videos",
        )
```

Replace with:
```python
    def _pick_folder(self) -> None:
        exercise = self._annotation_exercise_var.get()
        default = str(PROJECT_ROOT / exercise / "dataset_videos_all")
        selected = filedialog.askdirectory(
            initialdir=default if Path(default).exists() else str(PROJECT_ROOT),
            title=f"Select folder of {exercise.replace('_', ' ')} videos",
        )
```

- [ ] **Step 2: Fix `_find_existing_pipeline_output` — segmented + score root (~lines 2683, 2692)**

Find:
```python
            # Check for segmented JSON
            seg_root = workspace / "squat" / "segmented_reps"
            if not seg_root.exists():
                continue

            seg_matches = list(seg_root.rglob(f"{video_id}_segmented.json"))
            if not seg_matches:
                continue

            # Check for scoring JSON
            score_root = workspace / "squat" / "aqa_analysis_simple"
```

Replace with:
```python
            # Check for segmented JSON
            _ann_ex = self._annotation_exercise_var.get()
            seg_root = workspace / _ann_ex / "segmented_reps"
            if not seg_root.exists():
                continue

            seg_matches = list(seg_root.rglob(f"{video_id}_segmented.json"))
            if not seg_matches:
                continue

            # Check for scoring JSON
            score_root = workspace / _ann_ex / "aqa_analysis_simple"
```

- [ ] **Step 3: Fix `_build_annotation_payload_from_run` — seg/score/feat roots (~lines 2734, 2745, 2755)**

Find:
```python
        # Find segmented JSON
        seg_root = workspace / "squat" / "segmented_reps"
```

Replace with:
```python
        # Find segmented JSON
        _ann_ex = self._annotation_exercise_var.get()
        seg_root = workspace / _ann_ex / "segmented_reps"
```

Find (still in same method):
```python
        # Find scoring JSON (recursive into nested dirs)
        score_root = workspace / "squat" / "aqa_analysis_simple"
```

Replace with:
```python
        # Find scoring JSON (recursive into nested dirs)
        score_root = workspace / _ann_ex / "aqa_analysis_simple"
```

Find (still in same method):
```python
        # Find features JSON path
        feat_root = workspace / "squat" / "extracted_features_clean"
```

Replace with:
```python
        # Find features JSON path
        feat_root = workspace / _ann_ex / "extracted_features_clean"
```

- [ ] **Step 4: Fix `_build_annotation_payload_from_run` — visualization paths (~lines 2780, 2790, 2802)**

These three are in the same method, in the visualization discovery block. They follow the `_ann_ex` variable already set in Step 3.

Find:
```python
        # (1) Pose-landmark annotated video
        poses_root = workspace / "squat" / "visualized_poses_clean"
```

Replace with:
```python
        # (1) Pose-landmark annotated video
        poses_root = workspace / _ann_ex / "visualized_poses_clean"
```

Find:
```python
        # (2) Segmentation phase overlay
        if not vis_video:
            seg_vis_root = workspace / "squat" / "visualized_segmentation"
```

Replace with:
```python
        # (2) Segmentation phase overlay
        if not vis_video:
            seg_vis_root = workspace / _ann_ex / "visualized_segmentation"
```

Find:
```python
        # (3) Raw video fallback
        if not vis_video:
            raw = workspace / "squat" / "dataset_videos_all" / f"{video_id}.mp4"
```

Replace with:
```python
        # (3) Raw video fallback
        if not vis_video:
            raw = workspace / _ann_ex / "dataset_videos_all" / f"{video_id}.mp4"
```

- [ ] **Step 5: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "fix(ui): replace hardcoded squat paths in annotation payload builders with annotation exercise variable"
```

---

## Task 9: Fix annotation `_load_video_folder` processed-video scan

**Files:** `apps/desktop-ui/app.py`

The annotation tab scans previous run workspaces to mark videos as processed. It currently only looks in `squat/` subdirectories.

- [ ] **Step 1: Fix processed-video scan (~lines 2314, 2320)**

Find:
```python
                # Fix: Check for both segmentation and scores
                seg_root = run_dir / "workspace" / "squat" / "segmented_reps"
                if seg_root.exists():
                    for f in seg_root.rglob("*_segmented.json"):
                        vid_id = f.stem.replace("_segmented", "")
                        processed_vids.add(vid_id)

                score_root = run_dir / "workspace" / "squat" / "aqa_analysis_simple"
```

Replace with:
```python
                # Check for both segmentation and scores across all exercise namespaces
                for _ex_ns in ("squat", "overhead_press", "seated_overhead_press"):
                    seg_root = run_dir / "workspace" / _ex_ns / "segmented_reps"
                    if seg_root.exists():
                        for f in seg_root.rglob("*_segmented.json"):
                            vid_id = f.stem.replace("_segmented", "")
                            processed_vids.add(vid_id)

                score_root = run_dir / "workspace" / self._annotation_exercise_var.get() / "aqa_analysis_simple"
```

- [ ] **Step 2: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "fix(ui): annotation processed-video scan checks all exercise namespaces for segmented reps"
```

---

## Task 10: Fix annotation pipeline threads — exercise-aware workspace + `--exercise` flag

**Files:** `apps/desktop-ui/app.py`

The annotation tab auto-runs stages 2.5→4→5→8 when a video hasn't been processed. Both `_pipeline_thread` and `_batch_pipeline_thread` hardcode the workspace to `squat/` and `_run_pipeline_stage` never passes `--exercise` to the stage scripts.

- [ ] **Step 1: Fix `_pipeline_thread` workspace directory (~line 3120)**

Find:
```python
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
```

Replace with:
```python
        try:
            # Prepare workspace
            _ann_exercise = self._annotation_exercise_var.get()
            exercise_dir = workspace / _ann_exercise
            dataset_target = exercise_dir / "dataset_videos_all"
            dataset_target.mkdir(parents=True, exist_ok=True)
            logs_root.mkdir(parents=True, exist_ok=True)

            # Copy single video
            shutil.copy2(video_path, dataset_target / video_path.name)
            self._pipeline_log(f"Workspace: {workspace}")

            # Run stages in order
            stages_to_run = [s for s in _build_stages(_ann_exercise) if s.key in (
                "extract_selected_features", "classify_views",
                "temporal_segmentation", "scoring"
            )]
```

- [ ] **Step 2: Fix `_batch_pipeline_thread` workspace directory (~line 3193)**

Find:
```python
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
```

Replace with:
```python
            try:
                # Prepare workspace
                _ann_exercise = self._annotation_exercise_var.get()
                exercise_dir = workspace / _ann_exercise
                dataset_target = exercise_dir / "dataset_videos_all"
                dataset_target.mkdir(parents=True, exist_ok=True)
                logs_root.mkdir(parents=True, exist_ok=True)

                # Copy single video
                shutil.copy2(video_path, dataset_target / video_path.name)
                self._pipeline_log(f"Workspace: {workspace}")

                stages_to_run = [s for s in _build_stages(_ann_exercise) if s.key in (
                    "extract_selected_features", "classify_views",
                    "temporal_segmentation", "scoring"
                )]
```

- [ ] **Step 3: Add `--exercise` to `_run_pipeline_stage` cmd (~line 3043)**

Find:
```python
        cmd = [sys.executable, str(stage.script_path), *stage_args]
        env = os.environ.copy()
        env["EXEVISION_MODEL_PATH"] = str(SHARED_MODEL_PATH)
        env["EXEVISION_FACE_MODEL_PATH"] = str(SHARED_FACE_MODEL_PATH)
```

Replace with:
```python
        _ann_exercise = self._annotation_exercise_var.get()
        cmd = [sys.executable, str(stage.script_path), "--exercise", _ann_exercise, *stage_args]
        env = os.environ.copy()
        env["EXEVISION_MODEL_PATH"] = str(SHARED_MODEL_PATH)
        env["EXEVISION_FACE_MODEL_PATH"] = str(SHARED_FACE_MODEL_PATH)
```

- [ ] **Step 4: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "fix(ui): annotation pipeline threads use annotation exercise variable for workspace path and --exercise flag"
```

---

## Task 11: Fix `_find_overlay_from_pipeline_outputs` — exercise-agnostic path anchor

**Files:** `apps/desktop-ui/app.py`

This closure (inside a larger annotation method) anchors visualization paths via `squat_root = seg_path.parent.parent.parent`. The name is misleading but the structure `workspace/{exercise}/segmented_reps/{quality}/` means `.parent.parent.parent` correctly resolves to `workspace/{exercise}` for any exercise. Only the variable name and downstream path literals need updating.

- [ ] **Step 1: Fix variable name and downstream paths (~line 3453)**

Find:
```python
            # .../workspace/squat/segmented_reps/<quality>/<video>_segmented.json
            squat_root = seg_path.parent.parent.parent

            # Priority 1: pose-landmark annotated video
            poses_root = squat_root / "visualized_poses_clean"
            for viz_q in ("raw_unfiltered", quality):
                candidate = poses_root / viz_q / f"{video_id}_annotated.mp4"
                if candidate.exists():
                    return str(candidate)

            # Priority 2: segmentation phase overlay
            for viz_q in ("raw_unfiltered", quality):
                vis_dir = squat_root / "visualized_segmentation" / viz_q
                for suffix in ("_phases.mp4", "_segmented.mp4", "_phases.avi", "_segmented.avi"):
                    candidate = vis_dir / f"{video_id}{suffix}"
                    if candidate.exists():
                        return str(candidate)
```

Replace with:
```python
            # .../workspace/{exercise}/segmented_reps/<quality>/<video>_segmented.json
            exercise_root = seg_path.parent.parent.parent

            # Priority 1: pose-landmark annotated video
            poses_root = exercise_root / "visualized_poses_clean"
            for viz_q in ("raw_unfiltered", quality):
                candidate = poses_root / viz_q / f"{video_id}_annotated.mp4"
                if candidate.exists():
                    return str(candidate)

            # Priority 2: segmentation phase overlay
            for viz_q in ("raw_unfiltered", quality):
                vis_dir = exercise_root / "visualized_segmentation" / viz_q
                for suffix in ("_phases.mp4", "_segmented.mp4", "_phases.avi", "_segmented.avi"):
                    candidate = vis_dir / f"{video_id}{suffix}"
                    if candidate.exists():
                        return str(candidate)
```

- [ ] **Step 2: Final scan — confirm no remaining squat path hardcodes in annotation methods**

```powershell
& ".venv\Scripts\python.exe" -c "
src = open('apps/desktop-ui/app.py', encoding='utf-8').read()
lines = src.splitlines()
hits = [(i+1, l.strip()) for i, l in enumerate(lines)
        if '\"squat\"' in l and
        any(seg in l for seg in ['dataset_videos_all', 'segmented_reps', 'extracted_features', 'aqa_analysis', 'visualized', 'neural_analysis'])]
for ln, txt in hits:
    print(f'  line {ln}: {txt}')
print(f'Remaining: {len(hits)}')
"
```

Expected: `Remaining: 0`

- [ ] **Step 3: End-to-end manual test**

```powershell
& ".venv\Scripts\python.exe" apps/desktop-ui/app.py
```

1. **Inference — OHP:** Select `overhead_press` → browse to OHP video folder → run pipeline → scoring panel shows OHP metrics (Grip width, ROM, Lockout, Elbow flare)
2. **Inference — Seated OHP:** Select `seated_overhead_press` → same metrics, paths route to `seated_overhead_press/`
3. **Inference — Squat (regression):** Select `squat` → run pipeline → squat metrics appear, `analyze_results` stage visible, default dataset path restored
4. **Annotation — OHP flags:** Annotation tab → select `overhead_press` → Form Errors shows "Incomplete Lockout", "Elbow Flare / Winging", etc.
5. **Annotation — Seated OHP flags:** Select `seated_overhead_press` → same OHP flags
6. **Annotation — Squat regression:** Select `squat` → squat flags appear; browse to squat video folder → double-click unprocessed video → pipeline auto-runs → results load

- [ ] **Step 4: Commit**

```
git add apps/desktop-ui/app.py
git commit -m "fix(ui): _find_overlay_from_pipeline_outputs uses exercise_root instead of squat_root"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| `seated_overhead_press` in inference dropdown | Task 3 |
| `analyze_results` stage squat-only | Task 3 |
| `dataset_var` resets for OHP | Task 3 |
| All inference output discovery paths parameterised | Task 4 (12 paths across 8 methods) |
| OHP scoring metrics display (no KeyError) | Task 5 — all 3 methods |
| Annotation tab has its own exercise selector | Task 6 |
| Exercise selector independent from inference tab | Task 6 |
| Annotation flags rebuild on exercise change | Task 7 |
| OHP flags from `overhead_press.json` | Task 7 |
| Seated OHP flags from `overhead_press.json` (shared) | Task 7 — `_config_file_for_exercise` routing |
| Squat flags from `squat.json` | Tasks 1 + 7 |
| F1–F7 hotkeys rebind on exercise change | Task 7 |
| Annotation folder picker parameterised | Task 8 |
| Annotation payload builders parameterised (6 paths) | Task 8 |
| Annotation processed-video scan covers all exercises | Task 9 |
| Annotation pipeline workspace uses correct exercise dir | Task 10 |
| Annotation pipeline passes `--exercise` to stage scripts | Task 10 |
| Overlay discovery closure uses exercise-agnostic variable | Task 11 |
| Squat pipeline fully untouched | All tasks guard on `exercise == "squat"` or use exercise variable |

**Type consistency:**
- `_annotation_exercise_var` defined Task 6, used Tasks 7, 8, 9, 10, 11 — consistent
- `self._flags_frame` assigned Task 7 Step 1, used in `_rebuild_annotation_flags` — consistent
- `_config_file_for_exercise()` defined Task 2, called in Task 7 `_build_annotation_flag_defs` — consistent
- `CONFIG_EXERCISES_DIR` defined Task 2, used in Task 7 `_build_annotation_flag_defs` — consistent
- `_build_stages(_ann_exercise)` in Tasks 10 replaces `STAGES` (global) so annotation always builds correct stage list
