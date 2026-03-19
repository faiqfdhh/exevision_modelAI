# CLAUDE.MD — ExeVision AI Project Knowledge Base

_Last updated: 2026-03-19 (reorganized for current workflow clarity; all historical detail preserved in appendix)_

---

## 0) Latest Status (2026-03-17 — Current)

ExeVision AI is in **late prototyping / early pre-production** state for squat analysis. The production pipeline is operational and actively used (50+ runs documented in `pipeline_ui_runs/`). Core workflow is UI-driven via `pipeline_ui/app.py` with two active tabs: **Inference** (traditional pipeline orchestration) and **Annotation** (human scoring tool for dataset collection).

**Key components:** `scripts/2.5` (pose extraction) → `scripts/4` (view classification) → `scripts/5` (temporal segmentation with strict phase sequencing as of 2026-03-17) → `scripts/8` (rule-based AQA scoring). All stages are routed through `pipeline_ui/app.py`, which manages isolated workspaces, logging, and preview rendering.

**Annotation ecosystem** reached maturity on 2026-03-15 with async batch processing, continuous severity scales (0–5), per-video JSON storage, and heuristic bias-hiding. Data collection is now the immediate next milestone (target 100+ human-scored reps to support BiLSTM training).

---

## 1) Project Identity

**Project name:** ExeVision AI  
**Workspace root:** `exevision_modelAI`  
**Domain:** AI-assisted exercise form analysis (current focus: squat)  
**Core approach:** MediaPipe pose extraction + rule-based biomechanical analysis + stage-wise pipeline orchestration.

This repository is a practical implementation of a multi-stage human movement analysis pipeline, with both:
- **Production-style modular code** under `src/` (stages 1–3 integrated), and
- **Script-driven full pipeline** under `scripts/` + `pipeline_ui/` (stages through scoring/analytics).

---

## 2) Vision and Goals

### Vision
Build a robust, explainable movement-quality assessment system that can:
1. Understand exercise execution from video,
2. Detect form faults with biomechanical reasoning,
3. Produce interpretable quality scores and actionable feedback,
4. Scale to multiple exercises over time.

### Current Concrete Goal
Deliver reliable squat analysis from raw video using a deterministic multi-stage pipeline:
- Pose extraction (MediaPipe landmarks),
- View classification (camera angle detection),
- Temporal phase segmentation (eccentric/concentric/isometric reps),
- Rule-based scoring and form feedback.

### Intended Hybrid Architecture (Roadmap)
Symbolic rule engine (currently active) + learned temporal scorer (BiLSTM; planned), fused into a single 0–100 score with evidence-based feedback.

---

## 3) Production vs Legacy: Active Components

### ✅ PRODUCTION (Active, Deployed)

**Core pipeline (stages):**
- `scripts/2.5_extract_selected_features.py` — MediaPipe pose extraction + quality gating
- `scripts/4_classify_views.py` — Visibility-based view classification (front/back/side/front_side/back_side/unknown) [Latest: 2026-03-07]
- `scripts/5_temporal_segmentation.py` — FSM phase segmentation with strict legal sequences [Latest: 2026-03-17]
- `scripts/8_scoring.py` — Rule-based AQA scoring and metrics [Latest: 2026-03-11]

**UI & Orchestration:**
- `pipeline_ui/app.py` — Dual-tab Tkinter runner (Inference + Annotation) [Latest: 2026-03-15]
  - `PipelineRunnerUI` class: traditional dataset/single-video pipeline execution with preview
  - `AnnotationToolUI` class: folder-based rep annotation with bias-blind heuristic reveal

**Annotation Ecosystem:**
- `scripts/select_annotation_samples.py` [NEW 2026-03-11] — Strategic rep subset selection
- `scripts/analyze_annotations.py` [NEW 2026-03-11] — Annotation quality self-check (correlation, disagreement stats)
- `dataset/annotations/index.json` [NEW 2026-03-11] — Master list of all annotated videos + global stats
- `dataset/annotations/videos/{video_id}.json` — Per-video annotation file with all reps, human scores, and flags

**Evidence:** 50+ runs in `pipeline_ui_runs/`; heavy daily use through March 2026

### ⚠️ LEGACY / EXPERIMENTAL (Rarely Used)

| Component | Status | Notes |
|-----------|--------|-------|
| `src/` (modular pipeline) | Medium | Stages 1–3 clean; stages 4–7 remain placeholders. Not actively invoked by UI. |
| `squat_analysis/` | Active reference | Dedicated AQA module with microprograms; maintained separately from `scripts/8_scoring.py`. |
| `rule_based_programs_sample/` | Reference only | Diving AQA; completely separate exercise domain. |
| `scripts/1, 2, 3, 5.5, 5.6, 7` | Superseded | Pre-dating refactored stages 2.5/4/5/8. Some visualization utilities still useful. |
| Old feature folders | Deprecated | `extracted_features/`, `extracted_features1/` replaced by `extracted_features_clean/` with quality tiers. |
| `analyze_results.py` | Snapshot-only | Exists in historical run workspaces but not tracked in source tree. |

### Key Fragmentation Points

1. **Parallel code paths:** `src/` (modular, clean) vs. `scripts/` (operational, actively maintained)
2. **Missing dependency manifest:** No `requirements.txt`, `pyproject.toml`, or `setup.py` versioning
3. **Path coupling:** Scripts assume fixed relative paths; onboarding friction
4. **Artifact drift:** Some runtime expectations rely on historical workspace snapshots

---

## 4) Current Production Flow

The UI (`pipeline_ui/app.py`) is the single entry point for all production workloads. It provides two tabs for different workflows:

### **Inference Tab: Full Pipeline Execution**

**Input:** Dataset folder or single video  
**Output:** Annotated videos, segmentation JSON, AQA scores, predictions, and preview renderings

**Execution flow:**
```
User selects input → clicks "Start" → spawns background thread
  ├─ Creates isolated workspace under pipeline_ui_runs/{run_name}/
  ├─ Executes stages sequentially (or by user-selected subset):
  │  1. Stage 2.5: Extract pose features (MediaPipe landmarks)
  │  2. Stage 4: Classify camera view
  │  3. Stage 5: Detect reps and phases (temporal segmentation)
  │  4. Stage 8: Score reps and generate AQA report
  ├─ Stages output JSON + video overlays to workspace
  ├─ Logs written per-stage to logs/{stage_key}.log
  └─ User can preview annotated videos + score summaries in UI
```

**Stage Selection:** User can run full pipeline, individual stages, or custom subsets; stages execute in dependency order.

**Extraction Mode:** Dropdown selector (lines ~200 in app.py) chooses `Filtered` (default, quality gating) or `Unfiltered (raw)` (lightweight smoothing only); passed to Stage 2.5 CLI.

### **Annotation Tab: Human Scoring Workflow**

**Input:** Folder of raw videos  
**Output:** Per-video annotation JSON files with human scores, flags, and metadata

**Execution flow:**
```
User picks folder → async background scan for processed videos
  ├─ Color-codes listbox: green (processed), red (unprocessed), ✓ (fully annotated)
  ├─ User double-clicks a video
  │  ├─ Check if existing annotation JSON found → load if present
  │  └─ Check if pipeline outputs found in pipeline_ui_runs/
  │      ├─ If found: load segmentation + scoring immediately
  │      └─ If not found: auto-pipeline stages 2.5→4→5→8 in background
  ├─ Display current rep with dual videos (raw + visualized)
  ├─ User scores 0–100 + optional flags (F1–F6 shortcuts) + confidence + notes
  ├─ Heuristic score hidden until submission → revealed with Δ disagreement
  ├─ Save to dataset/annotations/videos/{video_id}.json
  └─ Auto-advance to next rep (or next unannotated video)

Batch mode (multi-select):
  ├─ User selects multiple videos
  ├─ Clicks "Process Selected" or "Reprocess Selected"
  └─ Sequential background execution of stages 2.5→4→5→8 for each
```

**Auto-Pipeline:** If video lacks existing pipeline outputs, annotation tab spawns stages 2.5/4/5/8 in an isolated workspace (does not interfere with existing runs).

**Data Format:** Each annotated video produces `dataset/annotations/videos/{video_id}.json` with:
- Video metadata (source, view, quality, FPS)
- Per-rep: rep ID, start/end frames, duration, phases, heuristic metrics, human score, human flags, severity scales (0–5), confidence, notes

**Annotation Index:** Master `dataset/annotations/index.json` tracks all annotated videos, totals, and summary statistics.

---

## 5) Pipeline Logic by Stage

### **Stage 2.5 — Pose Extraction**

**Input:** Raw squat videos  
**Output:** Feature JSON + visualizations (annotated video, analysis report)

**Processing:**
- MediaPipe Pose Landmarker extracts 33 landmarks (image + world coordinates)
- Landmark confidence filtering (general ≥0.4, key joints ≥0.5)
- **Filtered mode** (default):
  - Savitzky-Golay smoothing (window 5 frames)
  - Landmark stability filtering
  - Planted-foot recovery
  - Mandatory chain gating (hip→knee→ankle continuity ≥80%)
  - Quality tier routing: excellent/good/fair/poor
- **Unfiltered mode** (raw):
  - Lightweight One Euro Filter (XYZ coords only)
  - Skips heavy filtering and recovery
  - Routes to `raw_unfiltered/` folder

**Output paths:**
```
squat/extracted_features_clean/{excellent|good|fair|raw_unfiltered}/{video_id}.json
squat/visualized_poses_clean/{quality}_{video_id}_annotated.mp4
squat/analysis_reports/{quality}/{video_id}_analysis.json
```

**Quality gating thresholds:**
- `MIN_LANDMARK_CONFIDENCE = 0.4`
- `MIN_KEY_JOINT_CONFIDENCE = 0.5` (hip, knee, ankle)
- `MANDATORY_CHAIN_MIN_RATIO ≥ 80%`

---

### **Stage 4 — View Classification**

**Input:** Feature JSON from Stage 2.5 (any quality tier, including raw_unfiltered)  
**Output:** Feature JSON + view label (front/back/side/front_side/back_side/unknown)

**Classification logic (visibility-based):**
- **Front:** Both eyes visible + shoulder width ≥0.18 (wide X-spread)
- **Back:** No face visible + shoulder width ≥0.18 → confirmed by nose-vs-hip Z-depth (`nose_z > hip_z`)
- **Front-side:** Shoulder width 0.08–0.18 (narrow zone) + `nose_z < hip_z` (nose closer to camera)
- **Back-side:** Shoulder width 0.08–0.18 + `nose_z > hip_z` (nose farther from camera)
- **Side:** Shoulder width <0.08 (very narrow)
- **Unknown:** Zero trackable frames (honest signal; no hardcoded fallback)

**Why nose-vs-hip Z-depth?** More geometrically reliable than face visibility/confidence: nose is anatomically front-facing, so its depth relative to hip plane directly reveals subject's orientation to camera.

**Output:** Updates feature JSON in-place with `view` field; no new files created.

---

### **Stage 5 — Temporal Segmentation**

**Input:** Feature JSON + view label from previous stages  
**Output:** Rep segmentation JSON + phase visualization videos

**Core algorithm:**
- 15-frame biomechanical window @ 30fps (500ms per analysis window)
- Primary signal: normalized hip displacement (Y-coordinate motion)
- Phase FSM: `idle` ↔ `eccentric` ↔ `[isometric]` ↔ `concentric` ↔ `idle`

**Strict phase sequencing (as of 2026-03-17):**
- Legal cycle: `idle → eccentric → [isometric] → concentric → idle`
- `isometric` only enters from `eccentric` after >1s stillness with bent knees
- `concentric` only enters from `eccentric` or `isometric` on confirmed upward motion
- Return to `idle` gated by knee extension + top-height checks
- **Illegal transitions auto-repaired in-place** after raw detection and after short-glitch merging

**Shallow-squat tuning (as of 2026-03-17):**
- `DOWNWARD_VELOCITY_THRESHOLD = 0.0015`
- `UPWARD_VELOCITY_THRESHOLD = -0.0015`
- `MOTION_CONFIRM_FRAMES = 2` (low latency)
- `GLITCH_MERGE_FRAMES = 3` (only ultra-short flickers)

**Output paths:**
```
squat/segmented_reps/{quality}/{video_id}_segmented.json
squat/visualized_segmentation/{quality}/
  ├─ {video_id}_phases.mp4       (color phase overlay: idle=gray, eccentric=red, isometric=cyan, concentric=green)
  └─ {video_id}_segmented.mp4    (concatenated individual rep clips)
```

---

### **Stage 8 — Scoring & AQA**

**Input:** Feature JSON + segmented reps JSON  
**Output:** AQA score JSON with per-rep and overall metrics

**Rule-based metrics (computed at rep bottom):**
- `knee_angle`: 2D angle (left/right average)
- `knee_valgus_ratio`: Knee spread / ankle spread (< 1.0 = inward collapse risk)
- `forward_lean_deg`: Hip-to-shoulder angle from vertical
- `squat_depth`: Normalized hip displacement; >0.0 = below-parallel
  - **Per-leg independent computation** (as of 2026-03-11): collects valid (left, right) candidates; returns `max()` to avoid Z-drift false negatives on diagonal views

**Scoring thresholds (view-dependent):**
- **Default:** depth good=0.1/bad=-0.1, forward_lean good=20°/bad=45°, valgus good=0.95/bad=0.75
- **Side view:** stricter forward_lean (good=30°/bad=60°), looser valgus (bad=0.70)

**Overall rep score:** 0–100 (aggregated from weighted metrics)

**Output paths:**
```
squat/aqa_analysis_simple/{quality}/{video_id}/
  ├─ {video_id}_aqa_simple.json    (scores, rep breakdown, metrics)
  └─ scoreboard.json              (summary across all videos)
```

---

## 6) Annotation Workflow & Data Artifacts

### Data Flow: Video → Pipeline → Annotation JSON

```
Raw Video (squat/dataset_videos_all/)
  ↓ [Stage 2.5]
Feature JSON (extracted_features_clean/{quality}/)
  ↓ [Stage 4]
Feature JSON + View (same file, updated)
  ↓ [Stage 5]
Segmented Reps JSON + Phase Videos (segmented_reps/, visualized_segmentation/)
  ↓ [Stage 8]
AQA Scores JSON (aqa_analysis_simple/{quality}/)
  ↓ [Annotation Tab]
Per-Video Annotation JSON (dataset/annotations/videos/{video_id}.json)
  +
Master Index JSON (dataset/annotations/index.json)
```

### Annotation File Format (Per-Video JSON)

```json
{
  "video_id": "25713_3",
  "source_video_path": "squat/dataset_videos_all/25713_3.mp4",
  "pipeline_run": "annotation_run_20260315_000320",
  "pipeline_outputs": {
    "features_json": "pipeline_ui_runs/.../workspace/squat/extracted_features_clean/good/25713_3.json",
    "segmented_json": "pipeline_ui_runs/.../workspace/squat/segmented_reps/good/25713_3_segmented.json"
  },
  "view": "side",
  "quality_rating": "good",
  "fps": 30.0,
  "total_reps": 3,
  "annotated_at": "2026-03-11T14:30:00",
  "calibration": { "body_scale": 0.247, "standing_hip_height": 0.312 },
  "reps": [
    {
      "rep_id": 1,
      "start_frame": 45,
      "end_frame": 112,
      "duration_seconds": 2.233,
      "heuristic_score": 72.0,
      "heuristic_metrics": { "knee_valgus": null, "forward_lean": 21.0, "squat_depth": 0.9 },
      "human_score": 65,
      "human_flags": { "insufficient_squat_depth": true, "forward_lean": true },
      "human_flag_severities": { "insufficient_squat_depth": 3, "forward_lean": 2 },
      "annotator_confidence": 4,
      "annotation_notes": "Good control but slightly shallow.",
      "human_metric_scores": { "depth": 70, "forward_lean": 65 }
    }
  ]
}
```

### Key Design Decisions

1. **Traceability:** `pipeline_run` field ensures every annotation links to exact pipeline version that produced segmentation/scoring
2. **Video fallback:** Annotation tab searches for visualization videos in order: stored path → `_phases.mp4` → `_segmented.mp4` → raw source (never `_annotated.mp4` due to frame-index mismatch from quality gating)
3. **Auto-pipeline:** If video lacks pipeline outputs, annotation tab spawns stages 2.5/4/5/8 in isolated workspace
4. **Bias-blind UI:** Heuristic score hidden until human score submitted → revealed with Δ disagreement
5. **Continuous severity:** Replaced binary flags with 0–5 scales for granular form-fault tracking
6. **Incremental save:** Auto-saves JSON after every rep submission; master index updated in real-time
7. **Batch processing:** Multi-video selection queues sequential pipeline + annotation for bulk workflows

---

## 7) Workspace Inventory

### Core Authored Code

- **`src/`**: Modular pipeline package (stages 1–3 clean; 4–7 placeholders)
  - `config.py`, `pose_extractor.py`, `view_classifier.py`, `temporal_segmenter.py`, `main.py`
- **`scripts/`**: Operational script stages (actively maintained)
  - `2.5_extract_selected_features.py` — MediaPipe pose + quality gating
  - `4_classify_views.py` — Visibility-based view classification
  - `5_temporal_segmentation.py` — FSM phase segmentation + strict sequencing
  - `8_scoring.py` — Rule-based AQA scoring
  - `select_annotation_samples.py` [NEW 2026-03-11] — Rep subset selection for annotation
  - `analyze_annotations.py` [NEW 2026-03-11] — Annotation quality self-check
  - Legacy: `1`, `2`, `3`, `5.5`, `5.6`, `7` (visualization utilities, superseded stages)
- **`pipeline_ui/`**: Tkinter UI runner
  - `app.py` — Dual-tab interface (Inference + Annotation)
- **`squat_analysis/`**: Dedicated squat AQA module (maintained separately)
  - `aqa_metaProgram_squat.py`, `scoring_functions.py`
- **`rule_based_programs_sample/`**: Diving AQA reference (separate exercise domain)

### Data & Artifacts

- `squat/dataset_videos_all/` — Raw dataset videos
- `models/pose_landmarker_heavy.task` — MediaPipe model file
- `pipeline_ui_runs/` — Historical run folders (~50 documented runs)
  - Each run: `workspace/` (isolated), `logs/` (per-stage), `stage_outputs/` (archival snapshots)
- **`dataset/annotations/`** [NEW 2026-03-11] — Centralized annotation store
  - `index.json` — Master list + global stats
  - `videos/{video_id}.json` — Per-video annotation files

### Documentation

- `README.md`, `pipeline_ui/README.md`, `squat_analysis/README.md`, `progress.md`
- No `requirements.txt`, `pyproject.toml`, or `setup.py` (infrastructure gap)

---

## 8) What Is Complete vs Incomplete

### Completed / Working

- ✅ Core pose-view-segmentation pipeline (both script and modular implementations)
- ✅ Rule-based squat scoring and metrics extraction
- ✅ Pipeline UI with run management, logging, preview, and dual tabs
- ✅ Annotation tool with async batch processing, bias-blind UI, and per-video JSON storage
- ✅ 50+ documented production runs
- ✅ Annotation data ecosystem (`select_annotation_samples.py`, `analyze_annotations.py`, `dataset/annotations/`)

### Partially Complete / Fragmented

- ⚠️ Two parallel code paths: `src/` (modular, clean) vs. `scripts/` (operational, actively maintained)
- ⚠️ Stage 4–7 in `src/main.py` remain placeholders vs. full operational script pipeline
- ⚠️ `analyze_results.py` exists only in historical run workspaces, not in source tree

### Not Yet Realized (vs Stated Ambition)

- ❌ BiLSTM temporal scorer integrated in pipeline
- ❌ Fusion of symbolic + neural scores into production orchestrator
- ❌ Fully reproducible environment manifest (requirements.txt/pyproject.toml)
- ⏳ Human annotation dataset — Tool exists; data collection in progress (target 100+ reps)

---

## 9) Technical Risks & Current Status

### Active Issues

1. **Pipeline divergence:** `src/` and `scripts/` may drift; plan mid-term unification
2. **Missing dependency manifest:** Onboarding and reproducibility risk
3. **File-path coupling:** Scripts assume fixed relative paths
4. **Annotation dataset:** Not yet started; blocks BiLSTM training milestone
5. **Stage 5 sensitivity tuning:** Shallow-rep detection threshold balance remains iterative across dataset

### Recently Resolved (✓ Fixed)

| Issue | Date | Fix |
|-------|------|-----|
| Quality-gate sensitivity | 2026-03-07 | Nose-vs-hip Z-depth for diagonal view disambiguation (replaced face hallucination) |
| Depth false-negative on diagonal views | 2026-03-11 | Per-leg independent `max()` computation (replaced bilateral averaging) |
| Unfiltered pipeline breakage | 2026-03-11 | Added `raw_unfiltered` discovery/routing to stages 4/5/8 |
| Shallow-squat detection | 2026-03-17 | Low-motion thresholds + strict phase sequencing + transition auto-repair |

### Maturity by Layer (as of 2026-03-17)

- **Data processing pipeline:** Strongly progressed (end-to-end stages fully integrated)
- **Stage 4 (view classification):** Robust (visibility-based + nose-vs-hip Z-depth)
- **Stage 5 (temporal segmentation):** Improved (strict sequencing + transition repair)
- **Stage 8 (scoring/AQA):** Improved (per-leg depth, Z-drift false-negative fixed)
- **Annotation tooling:** High maturity (async batch, bias-blind, continuous severity scales)
- **Modular package (`src`):** Medium (clean stages 1–3; later placeholders)
- **Ops/productization:** Medium (good UI; missing lock file)
- **Roadmap completion:** Partial (symbolic active; dataset collection in progress; BiLSTM/fusion pending)

### Run History Evidence

From `pipeline_ui_runs/` (50 runs documented):
- `extract_selected_features.log`: 50 runs (heavily used)
- `classify_views.log`: 11 runs
- `temporal_segmentation.log`: 9 runs
- `scoring.log`: 4 runs

**Interpretation:** Project has transitioned from prototype snippets to repeated operational execution. Bottlenecks are now **data quality variability, stage robustness, and pipeline consistency**, not core feasibility.

---

## 10) Next Milestones

### Immediate (Ongoing)
1. Port scoring/analysis stages into `src/` and extend `src/main.py` to full end-to-end parity

### Short-Term (Stabilization)
1. Add `requirements.txt` or `pyproject.toml` with exact package versions
2. Promote `analyze_results.py` to first-class source file (not just workspace snapshots)
3. Standardize stage interfaces and JSON schemas across `src/` and `scripts/`
4. Add failure reason logging in segmentation/scoring summaries
5. **Start annotating** — target 100+ human-scored reps as first dataset milestone

### Mid-Term (Consolidation)
1. Unify `src/` and `scripts/` code paths (eliminate divergence)
2. Create regression test fixtures with representative videos + expected outputs
3. Add config profiles (strict/lenient quality gates) for real-world variability
4. **Neural training** — once 150+ reps annotated, train BiLSTM on heuristic metrics → predict human scores

### Long-Term (Roadmap Alignment)
1. Integrate learned temporal scorer as secondary alongside heuristic rules
2. Implement explicit symbolic+neural fusion policy with calibration
3. Expand exercise support beyond squat via microprogram architecture

---

## 11) File/Module Reference

**UI & Orchestration:**
- `pipeline_ui/app.py` — Dual-tab runner (Inference + Annotation)
  - `PipelineRunnerUI` — Inference pipeline executor
  - `AnnotationToolUI` — Annotation folder browser & scorer
- `pipeline_ui/README.md` — UI workflow documentation

**Modular Pipeline (`src/`):**
- `src/main.py` — Orchestrator entry point
- `src/config.py` — Configuration & dataclasses
- `src/pose_extractor.py` — Stage 1: Pose extraction
- `src/view_classifier.py` — Stage 2: View classification
- `src/temporal_segmenter.py` — Stage 3: Temporal segmentation

**Script Pipeline (`scripts/`):**
- `scripts/2.5_extract_selected_features.py` — Pose extraction + quality gating
- `scripts/4_classify_views.py` — View classification
- `scripts/5_temporal_segmentation.py` — Phase segmentation + strict sequencing
- `scripts/8_scoring.py` — AQA scoring
- `scripts/select_annotation_samples.py` — Strategic rep selection for annotation
- `scripts/analyze_annotations.py` — Annotation quality check

**Squat AQA Module:**
- `squat_analysis/aqa_metaProgram_squat.py` — Orchestrator
- `squat_analysis/scoring_functions.py` — Rule-based scoring logic
- `squat_analysis/README.md` — Module documentation

**Annotation Data:**
- `dataset/annotations/index.json` — Master annotation list + stats
- `dataset/annotations/videos/{video_id}.json` — Per-video annotation files

**Project Docs:**
- `README.md` — Root project overview
- `progress.md` — Roadmap & milestone tracking
- `CLAUDE.MD` — This file

---

## Appendix A: Chronological Session Changelog (Historical Development)

_All day-to-day development notes preserved for reference. Latest session (2026-03-17) summarized in §0. See §5 for full production feature details._

### Session 2026-03-17 — Stage 5 Strict Sequencing + Annotation Controls

**Part 1: Strict Phase Sequencing**  
**Focus:** Force deterministic phase ordering while preserving shallow-squat detection.

**What was built:**
1. Enforced strict global adjacency in `scripts/5_temporal_segmentation.py` with sanitizer passes after `detect_phases()` and after `_enforce_minimum_durations()`
2. Constrained legal cycle to: `idle -> eccentric -> [isometric] -> concentric -> idle`
3. Restricted `isometric` entry to `eccentric -> isometric` only, requiring >1 second of stillness while knees remain bent
4. Preserved shallow reps by reducing motion thresholds and lowering confirmation latency, while limiting anti-jitter merging to ultra-short flicker segments only
5. Added explicit metadata reporting via `analysis_params.strict_phase_sequence` and `analysis_params.illegal_transition_repairs`

**Part 2: Annotation Process Controls**  
**Focus:** Make annotation processing mode-selectable and keep annotation labels safe during reprocess.

**What was built:**
1. Added extraction mode dropdown to Annotation tab with `Filtered` (default) and `Unfiltered (raw)` options
2. Wired selected mode into stage 2.5 execution for both single-video and batch pipeline paths
3. Changed **Reprocess Selected** semantics to preserve annotation JSON files, resetting only pipeline reference fields before re-running
4. Updated status messaging to show active extraction mode while queueing jobs

---

### Session 2026-03-15 — UI Polish & Batch Setup

**Focus:** Vastly improve UI responsiveness, throughput efficiency, and data collection nuance in Annotation tool.

**What was built:**
1. **Multi-Video Batching:** Video listbox now supports extended selection; users can queue dozens of videos for sequential background processing
2. **Async Folder Scanning:** Moved folder scanning to background thread; dropping thousands of videos no longer freezes UI on startup
3. **Color-Coded Status UI:** Listbox auto-paints rows (green=processed, red=unprocessed, checkmark=annotated) via thread-safe callbacks
4. **Indeterminate Progress:** Added `ttk.Progressbar` and success messagebox to indicate batch queue completion
5. **Heuristic Reference Panel:** Exposed raw biomechanical metrics (Depth Ratio, Knee Angle, Valgus, Trunk Lean) on right side for annotator reference
6. **Spectrum Severity Scaling:**
   - Replaced binary checkboxes with 0–5 sliders for six biomechanical faults
   - Removed mutually exclusive "Correct Form" flag (movement quality is spectrum, not binary)
   - Intertwined sliders & checkboxes: dragging >0 auto-checks; unchecking snaps to 0
7. **`analyze_annotations.py` Update:** Modified CLI to sum/average continuous severities instead of binary counts

---

### Session 2026-03-11 (Part 1) — Annotation Tool Launch

**Focus:** Design and implement human annotation tool in `pipeline_ui/app.py`.

**What was built:**

1. **`AnnotationToolUI` class** (entire new workflow in Notebook tab):
   - Folder-based video browser with status markers (`✓` complete, `(2/3)` partial)
   - Auto-detection of existing pipeline output (newest-first search in `pipeline_ui_runs/`)
   - Auto-pipeline execution (stages 2.5→4→5→8) in background if no output exists
   - Rep-by-rep video playback using visualization videos (safer than annotated overlays)
   - Bias-blind scoring: heuristic hidden until human score submitted → revealed with Δ
   - F1–F5 keyboard shortcuts for annotation flags
   - Per-video JSON storage at `dataset/annotations/videos/{video_id}.json`
   - `index.json` master list updated on every submit
   - Resume support: reloads existing annotation files on video click

2. **`scripts/select_annotation_samples.py`** (strategic rep subset selection):
   - Priority 1: boundary reps (scores 45–55, 65–75, 80–90)
   - Priority 2: view-stratified equal sampling
   - Priority 3: score extremes (top + bottom)
   - Priority 4: random fill
   - Gracefully handles small pools

3. **`scripts/analyze_annotations.py`** (quality self-check):
   - Score distributions, human-heuristic correlation, disagreement stats
   - Training-readiness warnings (correlation >0.95, variance <10, disagreement std <3, count <50)

4. **`main()` modified** to use `ttk.Notebook` with two tabs: Inference (PipelineRunnerUI) and Annotation (AnnotationToolUI)

---

### Session 2026-03-11 (Part 2) — Pipeline Path Fixes

**Focus:** Restore end-to-end support for Unfiltered runs without changing downstream logic.

**Root-cause diagnosis & fix:**

**Issue: Pipeline path mismatch for raw BlazePose outputs**
- **Symptom:** UI Unfiltered runs produced feature JSONs but stages 4/5/8 didn't discover them
- **Cause:** Stage 2.5 writes unfiltered outputs to `raw_unfiltered/`, but stages 4/5/8 hardcoded discovery for quality-tier folders only
- **Fix:** Added `raw_unfiltered` discovery/routing to all three downstream stages
- **Scope:** Path discovery/routing only; no analysis logic changed

---

### Session 2026-03-11 (Part 3) — Depth Metric Bug Fix

**Focus:** Diagnose and fix squat depth false-negative in `scripts/8_scoring.py`.

**Root-cause diagnosis & fix:**

**Issue: `calculate_vertical_depth()` bilateral averaging bug**
- **Symptom:** Video 49226_1 scored `squat_depth = -0.099`, `below_parallel = false` despite visible below-parallel squat
- **Investigation:** All 71 rep frames showed negative bilateral average despite high confidence (hip 99.9%, knee 93.8%)
- **Per-leg diagnosis at frame 47:**
  - `L_Hip_Y: 0.8397, L_Knee_Y: 0.8134` → `L_Disp: +0.0263` ✅ (correct)
  - `R_Hip_Y: 0.8429, R_Knee_Y: 0.8728` → `R_Disp: -0.0299` ❌ (Z-drift artifact on far side)
  - Bilateral average: `-0.0018` (false negative)
- **Root cause:** Diagonal/front-side camera view: farther knee's Z-depth projects into drifted 2D Y; high confidence ≠ accurate position
- **Fix:** Per-leg independent computation; return `max()` of valid candidates; one valid leg sufficient
- **Outcome:** Left leg correctly produces `+0.0263`; `below_parallel = true`; rep score now non-zero

---

### Session 2026-03-07 — View Classification Robustness

**Focus:** Fix view classification edge cases; robust phase recovery for temporal segmentation.

**Root-cause analysis & fixes:**

**Issue: Stage 4 diagonal ambiguity**
- **Problem:** Visibility-based classifier could misclassify diagonals because MediaPipe hallucinates face landmarks even on back views
- **Evidence:**
  - Video 25728_1 (front-angled): face confidence 0.0 but remaining trackable → would be mistaken for back
  - Video 25886_1 (truly back): nose + eyes at ~0.999 confidence (hallucination) but nose depth far behind hips (`nose_rel_z ≈ +0.392`)
- **Root cause:** Face visibility/confidence = model certainty, not camera-facing geometry
- **Fix:** Diagonal disambiguation via **nose-vs-hip Z-depth** instead of face confidence
  - `nose_z < avg_hip_z` → `front_side` (nose physically closer to camera)
  - `nose_z > avg_hip_z` → `back_side` (nose physically farther)
- **Why:** Nose is anatomically front-facing; its depth relative to hip plane directly reveals subject orientation
- **Validation:** Video 25886_1 now yields `back_side` with 60/60 frame votes

**Issue: Stage 5 sequence robustness**
- **Status:** Now uses strict sequence sanitation + explicit transition constraints (not startup normalization helpers)
- **Behavior:** Illegal adjacency repaired in-place; `isometric` only from `eccentric` after sustained hold; shallow thresholds tuned for early eccentric capture
- **Deployed:** In `scripts/5_temporal_segmentation.py`

---

### Session 2026-03-05 — Visibility-Based View Refactor

**Focus:** Improve robustness of view classification (stage 4) and pipeline continuity.

**Changes:**

1. **Stage 4 refactor** (`scripts/4_classify_views.py`):
   - Replaced rotation-angle/facing-signal voting with **visibility-based frame classification**
   - Eliminated hard fallback-to-`side`; returns `unknown` only when zero trackable frames
   - Per-frame voting: face/eye/ear visibility + shoulder width → concrete label or `unknown`

2. **Stage 5 update** (`scripts/5_temporal_segmentation.py`):
   - Added `'unknown'` to `VALID_VIEWS` set
   - Accepts all valid views including `unknown`; lets pose quality determine success
   - No longer rejects on questionable view; gracefully degrades instead

**Rationale:**
- Old approach (Z-coordinate rotation): brittle on back views (MediaPipe hallucinates face)
- New approach (visibility): intuitive & robust; "can I see the face?" is ground truth
- Removing hard `side` fallback: bad data now honest (`unknown`) rather than hidden
- Stage 5 graceful degradation: processes or fails based on actual pose quality, not view label semantics

**Expected outcome:**
- Pipeline continuity: stage 4 outputs concrete label or `unknown` (never surprises)
- Diagnostic clarity: `unknown` signals poor frame tracking; stage 5 logs show actual failure points

---

## Appendix B: How to Update This Document

**After each development session:**
1. Add a new subsection under "Appendix A: Chronological Session Changelog" with date and focus
2. Summarize what was built/fixed in 3–5 bullet points
3. Include root-cause analysis for any bug fixes (keep full technical detail)
4. Update §0 "Latest Status" with current state snapshot (1–2 paragraphs)
5. Update section headers' "Latest: YYYY-MM-DD" dates if relevant
6. Keep all historical entries intact (no deletions)

**Sections that change rarely (no update needed unless major refactor):**
- §1–3 (Project Identity, Vision, Production vs Legacy)
- §4 (Production Flow) — update only if new tabs/modes added
- §6–7 (Annotation Workflow) — update if annotation file format changes
- §11 (File Reference) — update only on new modules/scripts

---

