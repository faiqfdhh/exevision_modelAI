# CLAUDE.MD — ExeVision AI Project Knowledge Base

_Last updated: 2026-03-17 11:20:00 (latest: annotation extraction-mode selector + reprocess reference reset behavior + doc sync)_

## 1) Project Identity

**Project name:** ExeVision AI  
**Workspace root:** `exevision_modelAI`  
**Domain:** AI-assisted exercise form analysis (current focus: squat)  
**Core approach:** MediaPipe pose extraction + rule-based biomechanical analysis + stage-wise pipeline orchestration.

This repository is a practical implementation of a multi-stage human movement analysis pipeline, with both:
- **Production-style modular code** under `src/` (stages 1–3 integrated), and
- **Script-driven full pipeline** under `scripts/` + `pipeline_ui/` (stages through scoring/analytics).

---

## 2) Vision and Goal

### Vision
Build a robust, explainable movement-quality assessment system that can:
1. Understand exercise execution from video,
2. Detect form faults with biomechanical reasoning,
3. Produce interpretable quality scores and actionable feedback,
4. Scale to multiple exercises over time.

### Current concrete goal (observed in code)
Deliver reliable squat analysis from raw video using staged processing:
- Pose extraction,
- View classification,
- Temporal phase/repetition segmentation,
- Rule-based scoring,
- Results analysis/visualization.

### Intended broader roadmap (from `progress.md`)
A hybrid AQA system combining:
- Deterministic symbolic rule engine (primary), and
- Learned temporal scoring (BiLSTM secondary),
then fusing into a final 0–100 score + evidence-based feedback.

---

## 3) Workspace Inventory (What exists)

## Core authored code
- `src/`: modular pipeline package (`config`, `pose_extractor`, `view_classifier`, `temporal_segmenter`, `main`)
- `scripts/`: script-based pipeline stages (`1`, `2`, `2.5`, `4`, `5`, `8`, utilities)
  - `scripts/select_annotation_samples.py` *(NEW 2026-03-11)*: strategic rep subset selection for annotation (boundary reps, view-stratified, extremes, random fill; handles small pools gracefully)
  - `scripts/analyze_annotations.py` *(NEW 2026-03-11)*: standalone CLI tool to check annotation quality (distributions, human-heuristic correlation, disagreement stats, training-readiness warnings)
- `pipeline_ui/`: Tkinter desktop runner — now has two tabs: **Inference** (existing pipeline) + **Annotation** (new annotation tool)
- `squat_analysis/`: dedicated rule-based squat AQA module and microprograms
- `rule_based_programs_sample/`: legacy/reference diving AQA implementation

## Data/artifacts
- `squat/dataset_videos_all/`: raw dataset videos
- `models/pose_landmarker_heavy.task`: MediaPipe model file
- `pipeline_ui_runs/`: historical run folders with logs, snapshots, isolated workspaces
- `dataset/annotations/` *(NEW 2026-03-11)*: centralized annotation store (created on first annotation save)
  - `dataset/annotations/index.json`: master list of all annotated videos + global summary stats
  - `dataset/annotations/videos/{video_id}.json`: one file per video with all reps, heuristic scores, human scores, and flags

## Documentation present
- `README.md` (root)
- `pipeline_ui/README.md`
- `squat_analysis/README.md`
- `rule_based_programs_sample/README.md`
- `progress.md`

## Notable infra observations
- No `requirements.txt`, `pyproject.toml`, or `setup.py` committed.
- `.gitignore` excludes `*.json`, `*.mp4`, `*.txt`, `*.png`, so many outputs/logs are intentionally non-versioned.

---

## 4) Architecture and Pipeline

## A) Modular `src/` pipeline (package design)
`src/main.py` orchestrates:
1. `PoseExtractor` (`src/pose_extractor.py`)
2. `ViewClassifier` (`src/view_classifier.py`)
3. `TemporalSegmenter` (`src/temporal_segmenter.py`)

Outputs are serialized into a pipeline JSON via `PipelineResult` dataclasses.

**Status:** stages 1–3 implemented and connected; stage 4–7 explicitly printed as pending in `src/main.py`.

## B) Script pipeline (operationally fuller)
Primary run order in UI:
1. `scripts/2.5_extract_selected_features.py`
2. `scripts/4_classify_views.py`
3. `scripts/5_temporal_segmentation.py`
4. `scripts/8_scoring.py`
5. `analyze_results.py` (copied into workspace if available)

This path currently represents the most complete practical processing chain.

## C) UI orchestration model
`pipeline_ui/app.py` provides:
- Full/individual/custom stage execution,
- Dataset or single-video mode,
- Isolated run workspace per execution,
- Per-stage logs and snapshot outputs,
- Video preview for annotated outputs (with explicit UI errors surfaced if pipeline stages drop videos),
- Stop/terminate support while running.

Runs are stored under `pipeline_ui_runs/<run_name>/` with `workspace/`, `logs/`, and `stage_outputs/`.

---

## 5) Current Feature Set (Implemented)

## Pose extraction and quality gating
- MediaPipe Pose Landmarker extraction into image/world keypoints.
- Visibility-based quality scoring and quality folder routing (`excellent/good/fair`, with poor handling logic).
- Unfiltered mode preserves raw BlazePose output under `extracted_features_clean/raw_unfiltered/` with no smoothing or landmark filtering.
- Mandatory chain gating for critical lower-body visibility continuity.
- Optional annotated overlays and analysis report generation.

## View classification (Stage 4)
**Recent refactor (2026-03-05):** Switched from rotation-angle voting to **visibility-based classification**.
**Latest enhancement (2026-03-07):** Restored **nose-vs-hip Z-depth** for diagonal direction disambiguation.
- **Signals:** Shoulder width (X-axis collapse) + face visibility for obvious front/back cases + nose-vs-hip depth for diagonal direction.
- **Logic:**
  - Very narrow shoulders (< 0.08) → `side` regardless of face.
  - No face visible:
    - Narrow shoulders (0.08–0.18 diagonal zone): compare `nose_z` vs average hip `z`
      - **nose_z < hip_z** → `front_side` (nose is physically closer to camera)
      - **nose_z > hip_z** → `back_side` (nose is physically farther from camera)
    - Wide shoulders (≥ 0.18) → `back` (full back view)
  - Both eyes visible + wide shoulders → `front`; narrow → use nose-vs-hip depth.
  - Partial face (one eye/nose) → use nose-vs-hip depth.
  - Zero trackable frames → `unknown` (no fallback to side).
- **Reliability:** Face visibility is a model-confidence signal and can hallucinate frontal features on back-side clips. Nose-vs-hip depth is a geometric cue and is more reliable for front-side vs back-side separation.
- **Output:** One of 5 labels (`front`, `back`, `side`, `front_side`, `back_side`) or `unknown`.

## Temporal segmentation (Stage 5)
- Biomechanical window-based FSM segmentation with hysteresis and phase constraints.
- Phases include `idle`, `eccentric`, `isometric`, `concentric`, `unknown`.
- Rep extraction and phase transition analysis.
- Optional visualization outputs.
- **Update (2026-03-05):** Now accepts `unknown` view labels (treats as best-effort)—will process if pose data is trackable, fail gracefully otherwise.
- **Update (2026-03-11):** Raw unfiltered inputs are now discovered from `extracted_features_clean/raw_unfiltered/` and routed to `segmented_reps/raw_unfiltered/` plus `visualized_segmentation/raw_unfiltered/`.

## Scoring / AQA
Two related scoring paths exist:
1. `scripts/8_scoring.py`: main operational squat AQA/scoring pipeline.
2. `squat_analysis/aqa_metaProgram_squat.py`: richer dedicated module with microprogram error functions and score aggregation.

- **Update (2026-03-11):** Unfiltered runs are now discovered from `extracted_features_clean/raw_unfiltered/` and matched against `segmented_reps/raw_unfiltered/` with no scoring-rule changes.

**`calculate_vertical_depth()` — updated 2026-03-11:**
- **Previous approach:** Required all four joints (L_HIP, R_HIP, L_KNEE, R_KNEE); averaged bilateral hip-knee displacement. One drifted knee dragged the average negative even when the other leg correctly showed depth.
- **Current approach:** Computes each leg independently; returns `max()` of valid candidates. Only one valid leg pair needed — best leg wins.
- **Why:** From diagonal/front-side views, the farther knee's Z-depth distorts its projected 2D Y coordinate; high confidence does not guarantee correct position. The near leg's Y coordinate is reliable.
- **Evidence (video 49226_1):** Frame 47 — `L_Disp: +0.0263` (correct), `R_Disp: -0.0299` (Z-drift artifact), bilateral average: **-0.0018** (false negative). Per-leg max: **+0.0263** (correct positive depth).
- **Thresholds (front_side):** `squat_depth good: 0.10, bad: -0.05`; weight: `0.30` of rep score.
- **`below_parallel` flag:** `depth_val > 0.0`; now correctly set from a single clean leg.

## Post-analysis tooling
- `analyze_results.py` exists in historical run workspaces and generates aggregate visual analytics.

## Annotation tool (NEW 2026-03-11)

Added as the **Annotation tab** in `pipeline_ui/app.py` alongside the existing Inference tab.

**Workflow:**
1. User picks any folder of raw squat videos (e.g., `squat/dataset_videos_all/`)
2. Tool shows a browsable video list with annotation-status markers (`✓` = complete, `(2/3)` = partial)
3. User clicks a video — tool auto-detects existing pipeline output in `pipeline_ui_runs/`:
   - **Found:** Loads segmentation JSON + scoring JSON immediately
   - **Not found:** Runs Stages 2.5→4→5→8 automatically in a background thread, with a live log panel
4. Tool presents each rep clip (from `_phases.mp4` or `_segmented.mp4`; **never uses `_annotated.mp4`** due to potential frame-index mismatch from quality gating; falls back to raw source video)
5. User types a 0–100 score + Enter; optional flags via `F1–F6` keys
6. Heuristic score is **hidden until after submission** to avoid bias — revealed with Δ disagreement
7. Annotations saved incrementally after every submit to `dataset/annotations/videos/{video_id}.json`
8. Annotation processing controls now include an extraction mode selector:
  - `Filtered` (default)
  - `Unfiltered (raw)`
  This mode is used by both **Process Selected** and **Reprocess Selected** batch actions.

**Annotation file format (per video):**
```json
{
  "video_id": "25713_3",
  "source_video_path": "squat/dataset_videos_all/25713_3.mp4",
  "pipeline_run": "run_20260307_230531",
  "pipeline_outputs": {
      "features_json": "pipeline_ui_runs/annotation_run_20260315.../workspace/squat/extracted_features_clean/good/25713_3.json",
      "segmented_json": "pipeline_ui_runs/..."
  },
  "view": "side", "quality_rating": "good", "fps": 30.0, "total_reps": 3,
  "annotated_at": "2026-03-11T14:30:00",
  "calibration": {
      "body_scale": 0.247,
      "standing_hip_height": 0.312
  },
  "graph_metadata": {
      "active_joints": [0, 1, 2, 11, 12, 23, 24, 25, 26, 27, 28],
      "zeroed_joints": [3, 4, 5, 6, 7, 8, 9, 10, 29, 30, 31, 32],
      "foot_consolidated": true
  },
  "reps": [
    {
      "rep_id": 1, "start_frame": 45, "end_frame": 112, "bottom_frame": 78,
      "duration_seconds": 2.233,
      "phases": ["eccentric", "eccentric", "isometric", "concentric", "concentric"],
      "signals": {
          "normalized_hip_displacement": [0.0, 0.012, 0.031],
          "window_velocity": [0.001, 0.003],
          "knee_angles": [165.2, 161.8],
          "landmark_confidence": [0.92, 0.91]
      },
      "heuristic_score": 72.0,
      "heuristic_metrics": { "knee_valgus": null, "forward_lean": 21.0, "min_knee_angle": 88.6, "squat_depth": 0.9 },
      "heuristic_metric_scores": { "forward_lean": 100.0, "depth": 62.7 },
      "human_score": 65.0,
      "flags": { "insufficient_squat_depth": true, "lumbar_flexion": false },
      "flag_severities": { "insufficient_squat_depth": 3 }
    }
  ]
}
```

**Key design decisions:**
- `pipeline_run` field in every annotation file ensures traceability to the exact pipeline version that produced the segmentation/scoring.
- Video fallback order is staged and recovery-aware:
  - Use stored `pipeline_outputs.visualization_video` when valid.
  - If stale/missing, search overlay candidates in this order: `_phases.mp4` → `_segmented.mp4` → `_phases.avi` → `_segmented.avi`.
  - Final fallback to raw source video.
  - `_annotated.mp4` is explicitly excluded because quality gating can drop frames and misalign frame indices.
- Nested score folder structure (`aqa_analysis_simple/{quality}/{tier}/`) is handled via `rglob()` — not flat folder scan.
- Runs are searched newest-first; the most recent complete output for a given video is preferred.
- Auto-pipeline feature copies the single video into a new isolated `annotation_run_{timestamp}/` workspace and runs the relevant stages — does not interfere with existing runs.
- Reprocess behavior preserves annotation JSON files and resets only pipeline reference fields (`pipeline_run`, `pipeline_outputs`) before re-running pipeline stages.
- **Spectrum Error Severities:** 
  - Replaced binary error flags with 0-5 sliders for granular severity tracking.
  - Added **Forward Lean** to the set of trackable form faults.
  - Intertwined sliders and checkboxes: dragging a slider > 0 automatically checks the box; clicking a checkbox snaps the slider to 1 or 0.
  - Layout improved with a much taller video listbox (h=35) and equal-sized side-by-side video columns.
- Per-rep annotation fields include:
  - `annotator_confidence` (1-5)
  - `annotation_notes` (free text)
  - `human_metric_scores` (`depth`, `knee_tracking`, `forward_lean`, `smoothness`, `control_at_bottom`; blank inputs stored as `null`)
  - Backward-compat aliases `flags` and `flag_severities` mirror `human_flags` and `human_flag_severities`.

**Keyboard shortcuts & Interactions:**
- `Enter`: submit score
- `F1–F6`: toggle flags (automatically slides severity to 1 or 0)
- `← Prev Rep` button to go back and edit a previously scored rep
- `✅ Force Save Current` explicitly saves the JSON payload (although auto-saves trigger on every submission)
- `Double-Click` or explicit button handles batch selection processing.

---

## 6) Progress Assessment (As of 2026-03-17)

### Recent Session Summary (2026-03-17, annotation process controls)
**Focus:** Make annotation processing mode-selectable and keep annotation labels safe during reprocess.

**What was built:**
1. Added an Annotation-tab extraction mode dropdown with `Filtered` default and `Unfiltered (raw)` option.
2. Wired the selected mode into stage 2.5 execution for both single-video and batch pipeline paths.
3. Changed **Reprocess Selected** semantics to preserve annotation files while resetting only pipeline reference fields (`pipeline_run`, `pipeline_outputs`) before re-running.
4. Updated status messaging to show the active extraction mode while queueing jobs.

### Recent Session Summary (2026-03-15, UI Polish & Batch Setup)
**Focus:** Vastly improve the UI responsiveness, throughput efficiency, and nuance of data collection in the Annotation tool.

**What was built:**
1. **Multi-Video Batching:** The video listbox now supports extended selection. Users can select dozens of raw videos and queue them for processing sequentially on a background thread (`_batch_pipeline_thread`).
2. **Asynchronous Folder Scanning:** Moving folder scanning to a background thread (`_scan_folder_thread`), meaning dropping thousands of videos into the source directory no longer freezes the UI upon startup. 
3. **Color-Coded Status UI:** The video listbox automatically paints rows green (processed/pipeline outputs found), red (unprocessed), or appends a checkmark (fully annotated) via thread-safe UI callbacks.
4. **Indeterminate Progress Loading:** Integrated a `ttk.Progressbar` and `messagebox` success alert to visually indicate when a batch queue is running and completing.
5. **Heuristic Reference UI:** Exposed the script's raw biomechanical metrics (Depth Ratio, Knee Angle, Valgus, Trunk Lean) onto the right-side UI panel so the annotator can use the algorithm's raw geometric findings as a reference baseline while scoring a rep.
6. **Spectrum Severity Scaling:** 
   - Replaced boolean error checkboxes with `ttk.Scale` sliders mapped from `0` to `5` for 6 specific biomechanical faults.
   - Removed the mutually exclusive "Correct Form" flag, recognizing that movement quality is a continuous spectrum, not binary.
   - Intertwined sliders and checkboxes: dragging a slider > 0 automatically checks the box; unchecking the box snaps the slider to 0.
7. **`analyze_annotations.py` Update:** Modified the CLI script to sum and average the continuous dataset severities instead of just reporting binary error counts.

### Recent Session Summary (2026-03-11, annotation tool)
**Focus:** Design and implement a human annotation tool integrated into `pipeline_ui/app.py`.

**What was built:**
1. **`AnnotationToolUI` class** added to `pipeline_ui/app.py` — entire annotation workflow in a new Notebook tab:
   - Folder-based video browser with status markers
   - Auto-detection of existing pipeline output (newest-first search in `pipeline_ui_runs/`)
   - Auto-pipeline execution (stages 2.5→4→5→8) in a background thread if no output exists
   - Rep-by-rep video playback using the safer visualization videos
   - Bias-blind scoring: heuristic revealed only after human score is submitted
   - F1–F5 keyboard shortcuts for annotation flags
   - Per-video annotation files at `dataset/annotations/videos/` with full metadata
   - `index.json` master list updated on every submit
   - Resume support: reloads existing annotation files on video click

2. **`scripts/select_annotation_samples.py`** — strategic rep subset selection (importable or standalone CLI):
   - Priority 1: boundary reps (scores 45–55, 65–75, 80–90)
   - Priority 2: view-stratified equal sampling
   - Priority 3: score extremes (top + bottom)
   - Priority 4: random fill
   - Graceful: if pool < target, annotates everything

3. **`scripts/analyze_annotations.py`** — standalone CLI quality self-check:
   - Score distributions, human-heuristic correlation, disagreement stats
   - Warns if: correlation > 0.95, variance < 10 pts, disagreement std < 3 pts, count < 50

4. **`main()` modified** to use `ttk.Notebook` with two tabs: *Inference* (existing PipelineRunnerUI unchanged) and *Annotation* (new AnnotationToolUI).


---

### Recent Session Summary (2026-03-11, later)
**Focus:** Restore end-to-end support for UI Unfiltered runs without changing scoring, segmentation, or view-classification logic.

**Root-cause diagnosis & fix:**
1. **Pipeline path mismatch for raw BlazePose outputs:**
  - **Symptom:** UI Unfiltered runs produced raw feature JSONs but downstream stages did not consistently discover them.
  - **Observed behavior:** `2.5_extract_selected_features.py` writes unfiltered outputs to `squat/extracted_features_clean/raw_unfiltered/`, while stages 4, 5, and 8 were written around the quality-tier folders `excellent/`, `good/`, and `fair`.
  - **Root cause:** `scripts/4_classify_views.py`, `scripts/5_temporal_segmentation.py`, and `scripts/8_scoring.py` hardcoded folder discovery and output routing for quality-tier folders only.
  - **Fix applied:** Added `raw_unfiltered` discovery/routing to all three downstream stages so raw BlazePose outputs flow through the existing pipeline unchanged.
  - **Scope control:** No scoring thresholds, geometry calculations, temporal segmentation rules, or view-classification decisions were changed. The fix is limited to path discovery/output routing.

### Recent Session Summary (2026-03-11)
**Focus:** Diagnose and fix squat depth false-negative in `scripts/8_scoring.py`.

**Root-cause diagnosis & fix:**
1. **Stage 8 (`8_scoring.py`) — `calculate_vertical_depth()` bilateral averaging bug:**
   - **Symptom:** Video 49226_1 scored `squat_depth = -0.099`, `below_parallel = false` despite visible below-parallel squat.
   - **Investigation:** Scanned all 71 rep frames; bilateral average of hip-knee displacement was always negative. Confidence was high (hip 99.9%, knee 93.8%), ruling out visibility issues.
   - **Per-leg diagnosis at frame 47:**
     - `L_Hip_Y: 0.8397, L_Knee_Y: 0.8134` → `L_Disp: +0.0263` ✅ (left hip below left knee)
     - `R_Hip_Y: 0.8429, R_Knee_Y: 0.8728` → `R_Disp: -0.0299` ❌ (right knee Y higher than expected due to Z-drift in front-side view)
     - Bilateral average: `(0.0263 − 0.0299) / 2 ≈ −0.0018` → false negative
   - **Root cause:** From a diagonal/front-side camera angle, the farther knee's Z-depth projects into a drifted 2D Y coordinate. High confidence flags do not guarantee geometric accuracy.
   - **Fix applied:** Replaced bilateral averaging with per-leg independent computation. Collect valid `(left, right)` candidates; return `max()`. One valid leg is sufficient.
   - **Validated outcome:** Left leg correctly produces `+0.0263`; `below_parallel = true`; `squat_depth` score is now non-zero for this rep.

### Recent Session Summary (2026-03-07)
**Focus:** Fix view classification edge cases and implement robust phase-recovery for temporal segmentation.

**Root-cause diagnosis & fixes:**
1. **Stage 4 (view classification) diagonal ambiguity:**
   - **Problem:** The visibility-based classifier could misclassify both diagonal directions because MediaPipe sometimes hallucinates face landmarks with high confidence even when the person is actually back-side.
   - **Evidence A:** Video 25728_1 in run_20260307_230531 was angled front-view but had all face landmarks at confidence 0.0 while remaining trackable in profile-like posture.
   - **Evidence B:** Video 25886_1 in run_20260307_231001 was truly back_side but was labeled `front_side` because nose + both eyes were reported at ~0.999 confidence on all first 60 frames even though the nose depth stayed far behind the hip plane (`avg nose_rel_z ≈ +0.392`).
   - **Root cause:** The script pipeline had drifted to treat face visibility/confidence as the source of truth for diagonal direction. Those values measure model certainty, not actual camera-facing geometry.
   - **Fix applied:** Replaced diagonal front_side/back_side disambiguation in `scripts/4_classify_views.py` with **nose-vs-hip Z-depth**:
     - `nose_z < avg_hip_z` → `front_side`
     - `nose_z > avg_hip_z` → `back_side`
   - **Why this works:** The nose is anatomically on the front of the body, so its depth relative to the hip plane directly reveals whether the subject's front half-space or back half-space faces the camera.
   - **Validation:** Re-running the updated classifier on 25886_1 yields `back_side` with 60/60 frame votes.

2. **Stage 5 (temporal segmentation) startup-phase lockup (from 2026-03-07 earlier session):**
   - **Problem (already fixed in prior session):** Savitzky-Golay smoothing reintroduced small negative values after clamping; first small negative velocity on frame 0 locked FSM into `concentric` phase, preventing valid `eccentric→concentric→eccentric` rep cycles.
   - **Fix applied (in prior session):** 
     - Post-smoothing clamping: `hip_y = np.maximum(0, hip_y)` (line ~201)
     - Leading-phase normalization: convert opening `concentric`/`isometric` runs to `idle` before forward-pass ordering (lines ~238–245, new `_normalize_leading_phase()` at lines 276–297)
   - **Status:** Deployed; expecting improved rep recovery on clips with early-frame noise.

### Recent Session Summary (2026-03-05)
**Focus:** Improve robustness of view classification (stage 4) and pipeline continuity.

**Changes:**
1. **Stage 4 refactor** (`scripts/4_classify_views.py`):
   - Replaced rotation-angle/facing-signal voting with **visibility-based frame classification**.
   - Eliminated hard fallback-to-`side`; returns `unknown` only when zero trackable frames exist.
   - Per-frame voting: face/eye/ear visibility + shoulder width → 5 concrete labels or `unknown`.

2. **Stage 5 update** (`scripts/5_temporal_segmentation.py`):
   - Added `'unknown'` to `VALID_VIEWS` set.
   - Simplified view validation: accepts all valid views including `unknown`, permits best-effort processing.
   - No longer rejects downstream based on questionable view; lets pose quality determine success.

**Rationale:**
- Old approach (z-coordinate rotation/facing) was brittle on back views (MediaPipe hallucinates nose).
- New approach (visibility) is intuitive and robust: "can I see the face?" is ground truth.
- Removing hard `side` fallback means bad data is now honest (`unknown`) rather than hidden.
- Stage 5 gracefully degrades instead of hard-failing on `unknown`.

**Expected outcome:**
- Pipeline continuity improved: stage 4 outputs a concrete label or `unknown` (never surprises).
- Stage 5 processes or fails based on actual pose quality, not view label semantics.
- Diagnostic clarity: `unknown` signals poor frame tracking; stage 5 logs will show where actual failure occurs.

### Maturity by layer (updated 2026-03-15)
- **Data processing pipeline:** Strongly progressed (end-to-end script stages fully integrated).
- **Stage 4 (view classification):** Enhanced—robust visibility-based logic with nose-vs-hip Z-depth for diagonal disambiguation.
- **Stage 5 (temporal segmentation):** Improved—graceful handling of ambiguous/unknown views; defensive startup-phase recovery deployed; raw-unfiltered routing supported.
- **Stage 8 (scoring / AQA):** Improved—`calculate_vertical_depth()` now uses per-leg best-reading; depth false-negatives from Z-drift on diagonal views eliminated; raw-unfiltered discovery supported.
- **Annotation tooling:** High Maturity — folder-based background async workflow, batch processing, per-video JSON storage, heuristic reveal, continuous 0-5 severity scales, and analysis tools.
- **Modular package (`src`) maturity:** Medium (stages 1–3 clean, later stages not yet ported).
- **Productization/ops maturity:** Medium (good UI, missing lock file; stages 4/5/8 now more resilient).
- **Research roadmap completion:** Partial (symbolic path active; annotation dataset being built; BiLSTM/fusion remains roadmap).

## Objective run-history evidence
From `pipeline_ui_runs`:
- **Run folders detected:** 50
- **`extract_selected_features.log`:** 50 (stage used heavily)
- **`classify_views.log`:** 11
- **`temporal_segmentation.log`:** 9
- **`scoring.log`:** 4

Signal-level reliability observations from logs:
- 1 historical extraction traceback (`ModuleNotFoundError: tqdm`) indicates an early environment setup issue.
- Several extraction runs report **unreliable video / mandatory chain gate** warnings (data quality gating working as designed).
- Temporal segmentation has mixed outcomes in some runs (`Failed: 1` occurs in a subset), indicating edge-case robustness work remains.
- Scoring logs include successful summaries with zero failed items in observed cases.

Overall interpretation:
- The project has moved beyond prototype snippets into repeated operational execution.
- Bottlenecks are now mostly in **data quality variability, stage robustness, and pipeline consistency**, not core concept feasibility.

---

## 7) What Is Complete vs Incomplete

## Completed / working
- Core pose-view-segmentation logic (both script and modular implementations).
- Rule-based squat scoring and metrics extraction.
- Pipeline desktop UI with run management, logging, and preview.
- Extensive experimental run evidence.
- **Annotation tool** (2026-03-15): optimized background async folder scanning, batching queue logic, continuous severity sliders, heuristic context parsing, and per-video JSON storage.

## Partially complete / fragmented
- Two parallel code paths (`src/` and `scripts/`) are not fully unified.
- Stage 4–7 in `src/main.py` are placeholders relative to full script pipeline.
- `analyze_results.py` is not present in root source tree; only found in historical run workspaces.

## Not yet realized (vs stated ambition)
- BiLSTM temporal scorer integrated in executable pipeline.
- Deterministic fusion of symbolic + neural scores in a single production orchestrator.
- Fully reproducible environment specification file(s).
- Human annotation dataset — **in progress** (annotation tool now exists; data collection not yet started).

---

## 8) Key Technical Risks / Debt

1. **Pipeline divergence:** `src/` architecture and `scripts/` operational flow may drift.
2. **Missing dependency manifest:** onboarding and reproducibility risk.
3. **File-path coupling:** several scripts assume fixed relative paths.
4. **Artifact dependency:** some expected runtime files (e.g., analysis script) may only exist in old run workspaces.
5. ~~**Quality-gate sensitivity**~~ **[FURTHER REDUCED 2026-03-07]:** Ear-visibility tiebreaker in diagonal zone now robustly distinguishes angled front views from genuine back views; misclassification risk greatly reduced.
6. ~~**Depth false-negative on diagonal views**~~ **[FIXED 2026-03-11]:** Bilateral averaging in `calculate_vertical_depth()` could yield a negative net displacement even when one leg clearly showed below-parallel depth. Fixed by per-leg `max()` computation — one valid leg is sufficient.
7. ~~**Unfiltered pipeline breakage**~~ **[FIXED 2026-03-11]:** Raw BlazePose outputs were saved into `raw_unfiltered/`, but stages 4/5/8 only scanned `excellent/good/fair`. Fixed by adding raw-unfiltered discovery/routing only; core analysis logic unchanged.

---

## 9) Recommended Next Milestones

## Immediate (ongoing)
1. Port scoring/analysis stages into `src/` and extend `src/main.py` to true end-to-end parity.

## Short-term (stabilization)
1. Add `requirements.txt` or `pyproject.toml` with exact package versions.
2. Make `analyze_results.py` a first-class tracked source file in root (not only in run snapshots).
3. Standardize stage interfaces and JSON schemas across `src/` and `scripts/`.
4. Add failure reason logging in segmentation/scoring summaries for faster debugging.
5. **Start annotating** — run the pipeline on your video folder, open the Annotation tab, score reps. Target 100+ scored reps as a first milestone.

## Mid-term (consolidation)
1. Port scoring/analysis stages into `src/` and extend `src/main.py` to true end-to-end parity.
2. Create regression test fixtures with a few representative videos and expected outputs.
3. Add config profiles (strict/lenient quality gates) for real-world variability.
4. **Neural training** — once 150+ reps annotated, implement a BiLSTM or MLP trained on `heuristic_metrics` + `heuristic_metric_scores` → predict `human_score`.

## Long-term (roadmap alignment)
1. Integrate learned temporal scorer as secondary scorer alongside heuristic rules.
2. Implement explicit symbolic+neural fusion policy with calibration.
3. Expand exercise support beyond squat via microprogram architecture.

---

## 10) Quick File/Module Reference

- Entry pipeline package: `src/main.py`
- Config/dataclasses: `src/config.py`
- Pose extraction: `src/pose_extractor.py`
- View classification: `src/view_classifier.py`
- Temporal segmentation: `src/temporal_segmenter.py`
- UI runner (Inference + Annotation tabs): `pipeline_ui/app.py`
  - `PipelineRunnerUI` — existing inference pipeline orchestrator (unchanged)
  - `AnnotationToolUI` *(NEW)* — folder-based rep annotation tool (lines ~1571–3000; layout polished and severity sliders implemented)
- UI docs: `pipeline_ui/README.md`
- Squat AQA orchestrator: `squat_analysis/aqa_metaProgram_squat.py`
- Squat scoring logic: `squat_analysis/scoring_functions.py`
- Scripted extraction/classification/segmentation/scoring: `scripts/2.5_extract_selected_features.py`, `scripts/4_classify_views.py`, `scripts/5_temporal_segmentation.py`, `scripts/8_scoring.py`
- Annotation sample selection *(NEW)*: `scripts/select_annotation_samples.py`
- Annotation quality self-check *(NEW)*: `scripts/analyze_annotations.py`
- Annotation data: `dataset/annotations/index.json`, `dataset/annotations/videos/{video_id}.json`
- Project structure overview: `README.md`
- Roadmap snippet: `progress.md`

---

## 11) Bottom-line Status

ExeVision AI is in a **late prototyping / early pre-production** state for squat analysis:
- Core algorithmic pipeline is implemented and repeatedly executed,
- UI-based operational workflow exists and is highly usable via batch processing additions,
- **Annotation tooling is fully mature** — the system can scalably process bulk videos on background threads and collect granular spectrum-scaled human quality scores for neural training dataset construction.
- Main remaining work is: (1) start collecting voluminous annotations using the optimized tool, (2) train the neural scoring model using extracted `severity` metrics, (3) consolidate source code and orchestrate the unified architecture.
