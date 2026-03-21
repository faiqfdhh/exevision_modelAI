# CLAUDE.MD — ExeVision AI Project Knowledge Base

_Last updated: 2026-03-21 (Step 2 fusion architecture overhauled; correction collapse diagnosed and fixed; new best evaluation snapshot recorded)_

---

## 0) Latest Status (2026-03-21 — Current)

ExeVision AI is in **late prototyping / early pre-production** state for squat analysis. The production pipeline is operational and actively used (50+ runs documented in `pipeline_ui_runs/`). Core workflow is UI-driven via `pipeline_ui/app.py` with two active tabs: **Inference** (traditional pipeline orchestration) and **Annotation** (human scoring tool for dataset collection).

**Key components:** `scripts/2.5` (pose extraction) → `scripts/4` (view classification) → `scripts/5` (temporal segmentation with strict phase sequencing as of 2026-03-17) → `scripts/8` (rule-based AQA scoring). All stages are routed through `pipeline_ui/app.py`, which manages isolated workspaces, logging, and preview rendering.

**Annotation ecosystem** reached maturity on 2026-03-15 with async batch processing, continuous severity scales (0–5), per-video JSON storage, and heuristic bias-hiding.

**Step 2 (Neural Fusion) — major improvement recorded 2026-03-21.** A correction collapse bug in `HeuristicGuidedFusion` was diagnosed and fixed: the prior model was applying a near-constant −0.4 point offset regardless of input (residual std ≈ 0.05). After architecture and training changes (see §12 and Appendix A), the model now learns genuine per-rep corrections (residual std ≈ 14–15) and achieves:

| Metric | Before fix | After fix | Δ |
|---|---|---|---|
| Post-clamp Pearson | 0.8352 | **0.8737** | +0.038 |
| Post-clamp MAE | 10.68 | **9.04** | −1.64 pts |
| Pre-clamp Pearson | 0.7795 | **0.8552** | +0.076 |
| Pre-clamp MAE | 11.98 | **9.28** | −2.70 pts |
| Failure cases (>\|20\|) | 4 | **0** | −4 |
| vs linear baseline MAE | 10.39 (neural was worse) | **9.28 beats linear** | ✅ |

Integration into `pipeline_ui/app.py` inference flow remains pending.

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
- ✅ Step 2 neural infrastructure scripts implemented (`nn_models.py`, `finetune_models.py`, `evaluate_model.py`, `nn_utils.py`, `pretrain_bilstm.py`, `pretrain_stgcn.py`)
- ✅ Step 2 phased fine-tuning executed and checkpoints saved (`bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt`)
- ✅ Step 2 evaluation pipeline executed with report artifact (`results/evaluation_report.json`)
- ✅ Step 2 correction collapse diagnosed and fixed (2026-03-21): fusion now learns per-rep residuals (std ≈ 14–15 vs prior 0.05), beating linear baseline on MAE for the first time (9.28 vs 10.39)

### Partially Complete / Fragmented

- ⚠️ Two parallel code paths: `src/` (modular, clean) vs. `scripts/` (operational, actively maintained)
- ⚠️ Stage 4–7 in `src/main.py` remain placeholders vs. full operational script pipeline
- ⚠️ `analyze_results.py` exists only in historical run workspaces, not in source tree

### Not Yet Realized (vs Stated Ambition)

- ❌ BiLSTM/ST-GCN/fusion inference integrated into production UI orchestration path
- ❌ Symbolic + neural fused scoring exposed as first-class runtime mode in `pipeline_ui/app.py`
- ❌ Fully reproducible environment manifest (requirements.txt/pyproject.toml)
- ⏳ Human annotation dataset scale-up — tooling exists; current Step 2 run used 147 reps (train=121, test=26), still sparse for robust extreme-score generalization

---

## 9) Technical Risks & Current Status

### Active Issues

1. **Pipeline divergence:** `src/` and `scripts/` may drift; plan mid-term unification
2. **Missing dependency manifest:** Onboarding and reproducibility risk
3. **File-path coupling:** Scripts assume fixed relative paths
4. **Step 2 integration gap:** Neural scorer exists as standalone scripts but is not yet wired into production inference tab
5. **Dataset scale and distribution risk:** Current test set is 26 reps with no 0–20 bucket coverage; generalization to poor-form squats remains unvalidated
6. **Per-metric prediction weakness:** Smoothness/control are reasonable (MAE ~10), but depth/lean/knee-tracking MAE remains high (~24) — ST-GCN Phase 2 converges but the spatial signal is inherently noisy on diagonal views
7. **Stage 5 sensitivity tuning:** Shallow-rep detection threshold balance remains iterative across dataset

### Recently Resolved (✓ Fixed)

| Issue | Date | Fix |
|-------|------|-----|
| Quality-gate sensitivity | 2026-03-07 | Nose-vs-hip Z-depth for diagonal view disambiguation (replaced face hallucination) |
| Depth false-negative on diagonal views | 2026-03-11 | Per-leg independent `max()` computation (replaced bilateral averaging) |
| Unfiltered pipeline breakage | 2026-03-11 | Added `raw_unfiltered` discovery/routing to stages 4/5/8 |
| Shallow-squat detection | 2026-03-17 | Low-motion thresholds + strict phase sequencing + transition auto-repair |
| **Step 2 correction collapse** | **2026-03-21** | **Removed L1 reg + orphaned confidence head; unfroze encoders with diff LR; tanh×40 bounded residual. Pre-clamp MAE dropped from 11.98 → 9.28; beats linear baseline; 0 failure cases.** |
| **Linear baseline parity** | **2026-03-21** | **Pre-clamp MAE 9.28 now beats linear (10.39); neural is earning its complexity.** |

### Maturity by Layer (as of 2026-03-21)

- **Data processing pipeline:** Strongly progressed (end-to-end stages fully integrated)
- **Stage 4 (view classification):** Robust (visibility-based + nose-vs-hip Z-depth)
- **Stage 5 (temporal segmentation):** Improved (strict sequencing + transition repair)
- **Stage 8 (scoring/AQA):** Improved (per-leg depth, Z-drift false-negative fixed)
- **Annotation tooling:** High maturity (async batch, bias-blind, continuous severity scales)
- **Step 2 training/evaluation tooling:** Improved — fusion architecture fixed, per-rep corrections working, beats linear baseline
- **Step 2 production integration:** Pending (not yet part of UI runtime path)
- **Modular package (`src`):** Medium (clean stages 1–3; later placeholders)
- **Ops/productization:** Medium (good UI; missing lock file)
- **Roadmap completion:** Partial (symbolic active; dataset collection in progress; neural inference integration pending)

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
1. **Integrate Step 2 neural inference into `pipeline_ui/app.py`** — expose explicit mode switch (heuristic-only vs neural-fused) in the Inference tab; use saved checkpoints (`bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt`) for live prediction
2. **Expand annotations in 20–60 score range** — currently 51 reps in this range (out of 173 total); annotate 30+ more to cover poor-form squats and improve generalization of residual learning
3. Port scoring/analysis stages into `src/` and extend `src/main.py` to full end-to-end parity

### Short-Term (Stabilization)
1. Add `requirements.txt` or `pyproject.toml` with exact package versions
2. Promote `analyze_results.py` to first-class source file (not just workspace snapshots)
3. Standardize stage interfaces and JSON schemas across `src/` and `scripts/`
4. Add failure reason logging in segmentation/scoring summaries
5. Re-run Step 2 with broader annotation coverage (especially 0–20 bucket) and compare against baselines under the same post-clamp policy
6. Validate Phase 4 joint fine-tuning — now that Phase 3 is working correctly, Phase 4 may provide additional gains

### Mid-Term (Consolidation)
1. Unify `src/` and `scripts/` code paths (eliminate divergence)
2. Create regression test fixtures with representative videos + expected outputs
3. Add config profiles (strict/lenient quality gates) for real-world variability
4. Investigate ST-GCN spatial metric MAE (depth=24, knee_tracking=24) — possibly improve with more training data or view-conditioned normalization

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

**Step 2 Neural Infrastructure (`scripts/`):**
- `scripts/nn_models.py` — Shared PyTorch modules for fine-tuning and inference
  - `BiLSTMScorer`: temporal encoder + smoothness/control heads
  - `STGCNScorer`: 5-block ST-GCN encoder + 3 spatial heads + 4 auxiliary heuristic-raw heads
  - `HeuristicGuidedFusion`: heuristic-anchored gated residual model (`final = clamp(heuristic + tanh(residual_head) × 40, 0, 100)`); confidence head removed 2026-03-21
  - `build_heuristic_vector()`: 15D vector contract
  - `apply_safety_clamps()`: post-prediction rule caps
- `scripts/finetune_models.py` — End-to-end supervised fine-tuning orchestrator
  - Index-gated annotation loading from `dataset/annotations/index.json`
  - Video-level stratified split by score bucket (seed 42)
  - Sample weighting: confidence × inverse-bucket
  - Training phases: BiLSTM → ST-GCN → Fusion (+ optional joint phase 4)
- `scripts/evaluate_model.py` — Test evaluation and report generation
  - Primary metrics on post-clamp predictions
  - Pre-clamp diagnostics
  - Baselines: heuristic-only and linear regression on heuristic vector
  - Output report: `results/evaluation_report.json`
- `scripts/nn_utils.py` — Shared sequence and skeleton utilities
  - Active-joint graph (11 joints), adjacency matrix construction
  - ST-GCN rep extraction and normalization contract
  - Fixed length pad/truncate
- `scripts/pretrain_bilstm.py` — BiLSTM encoder pretraining (masked reconstruction)
- `scripts/pretrain_stgcn.py` — ST-GCN encoder pretraining (masked reconstruction)

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

**Current Step 2 Artifacts:**
- `dataset/splits.json` — Persisted train/val/test video split
- `models/bilstm_pretrained.pt` — BiLSTM pretraining checkpoint
- `models/stgcn_pretrained.pt` — ST-GCN pretraining checkpoint
- `models/stgcn_pretrained_encoder.pt` — ST-GCN encoder-only checkpoint
- `models/bilstm_finetuned.pt` — Phase 1 fine-tuned temporal model
- `models/stgcn_finetuned.pt` — Phase 2 fine-tuned spatial model
- `models/fusion_layer.pt` — Phase 3 fine-tuned fusion model
- `results/evaluation_report.json` — Latest evaluation snapshot

---

## 12) Step 2: Supervised Fine-Tuning + Neural Fusion (Current State)

### 12.1 What Is Implemented

Step 2 is implemented as a standalone training/evaluation stack under `scripts/` and is runnable in the current workspace. The flow is:

1. Load indexed annotations from `dataset/annotations/index.json`.
2. Resolve per-video annotation JSON and features JSON.
3. Build rep-level multimodal records (temporal signals + skeleton sequence + heuristic vector).
4. Split by `video_id` using stratified score buckets.
5. Train in phases (BiLSTM, then ST-GCN, then fusion).
6. Evaluate on test split and write `results/evaluation_report.json`.

Production wiring into `pipeline_ui/app.py` is not yet implemented.

### 12.2 Data Contract Used by Step 2

#### Source-of-truth loading policy
- Only videos listed in `dataset/annotations/index.json` are considered.
- Per-video annotations are read from `dataset/annotations/videos/{video_id}.json`.

#### Per-rep fields consumed
- `human_score` (required)
- `heuristic_score`
- `signals.normalized_hip_displacement[]`
- `signals.window_velocity[]`
- `signals.knee_angles[]`
- `signals.landmark_confidence[]`
- `human_metric_scores` (nullable)
- `heuristic_metrics` (nullable)
- `flags` / `heuristic_flags`
- `human_flags`
- `flag_severities`
- `annotator_confidence`

#### Canonical tensor contracts
- BiLSTM input: `(128, 4)`
- ST-GCN input (pre-permute): `(128, 11, 7)`
- Heuristic vector: `(15,)`
- Temporal target/mask: `(2,)`
- Spatial target/mask: `(3,)`
- Auxiliary target/mask: `(4,)`

### 12.3 Training Flow (Current Script Behavior)

#### Phase 1: BiLSTM fine-tuning
- Loads pretrained temporal encoder
- Optimizes masked weighted MSE for smoothness/control
- Output: `models/bilstm_finetuned.pt`

#### Phase 2: ST-GCN fine-tuning
- Loads pretrained ST-GCN encoder blocks
- Optimizes spatial loss + 0.3 × auxiliary loss
- Output: `models/stgcn_finetuned.pt`

#### Phase 3: Fusion fine-tuning (revised 2026-03-21)
- **Encoders unfrozen** with differential LR (enc_lr = fusion_lr × 0.05) so they adapt slowly toward residual prediction task
- Trains heuristic-guided bounded residual model
- Formula: `final = clamp(heuristic + tanh(residual_head(fused)) × 40, 0, 100)`
- Loss: pure weighted MSE — no L1 correction regularisation (removed; was causing collapse)
- Regularisation from: weight_decay=1e-4 on fusion params + Dropout(0.1) in residual_head + gradient clipping (max_norm=1.0)
- Early stopping patience = 25 epochs
- Default epochs = 60, lr = 5e-4
- Monitoring: `res_std` in per-epoch log — healthy value is std > 2.0 by epoch 10
- Output: `models/fusion_layer.pt`

#### Optional Phase 4
- Joint end-to-end fine-tuning behind CLI flag
- Not enabled by default in current runs

### 12.4 Evaluation Snapshots

#### Snapshot A — Pre-fix (old architecture, ~2026-03-21 morning)
> `heuristic + confidence * correction` formula; frozen encoders; L1 correction reg = 0.1; 40 epochs

- Counts: train = 121, test = 26
- Post-clamp: Pearson = 0.8352, MAE = 10.68
- Pre-clamp: Pearson = 0.7795, MAE = 11.98
- Heuristic baseline: Pearson = 0.7796, MAE = 12.08
- Linear baseline: Pearson = 0.7948, MAE = 10.39
- **Neural was WORSE than linear on MAE (10.68 vs 10.39)**
- Failure cases: **4 reps**
- Correction stats: mean = −0.77, std = 0.05 — **near-constant (collapsed)**

#### Snapshot B — Post-fix (current architecture, 2026-03-21)
> `heuristic + tanh(residual_head)×40` formula; unfrozen encoders + diff LR; no L1 reg; 60 epochs

- Counts: train = 121, test = 26
- Post-clamp: Pearson = **0.8737**, MAE = **9.04**
- Pre-clamp: Pearson = **0.8552**, MAE = **9.28**
- Heuristic baseline: Pearson = 0.7796, MAE = 12.08
- Linear baseline: Pearson = 0.7948, MAE = 10.39
- **Neural beats linear on MAE (9.28 vs 10.39)** ✅
- Failure cases: **0 reps** ✅
- Residual stats: `res_std` grew from 1.4 (epoch 1) to ~14–15 (epoch 60) — genuine per-rep corrections
- Phase 3 val loss: 483 (epoch 1) → 164 (epoch 60, best) — consistent decrease confirming learning

### 12.5 Safety Clamp Policy (Post-Fusion)

Clamp rules currently applied in `scripts/nn_models.py`:

- If `knee_valgus` severity ≥ 2: score cap 60
- If `forward_lean` severity ≥ 2: score cap 65
- If `insufficient_squat_depth` severity ≥ 3: score cap 50

Primary metrics are computed on post-clamp scores, with pre-clamp metrics preserved for diagnostics.

### 12.6 Known Step 2 Limitations (Current Snapshot — Post-fix)

1. **Test-set size is small (26 reps), with no 0–20 bucket coverage.** Generalization to poor-form squats is unvalidated; 0 failure cases is promising but inconclusive with this distribution.
2. **Per-metric spatial MAE remains high** — depth (≈24) and knee_tracking (≈24) vs smoothness/control (≈10). ST-GCN learns sub-metric embeddings from Phase 2, but spatial quality on diagonal views is inherently noisy. Increasing training data or view-conditional normalization would help.
3. **Runtime integration into production UI/pipeline flow is still pending.** The neural scorer runs as a standalone training+eval stack only.
4. **Phase 3 train/val gap is wide** (e.g., train=111 vs val=164 at epoch 60) — some overfitting is occurring. With only 121 training reps, this is expected; more data would help.
5. **Phase 4 joint fine-tuning not yet validated** — now that Phase 3 is working, Phase 4 may provide additional gains but also carries overfitting risk.

### 12.7 Agent Fast-Path (Onboarding Shortcut)

For rapid agent onboarding, inspect in this order:

1. `CLAUDE.MD` §0, §12, §9, §10 (status, contracts, risks, next actions)
2. `scripts/finetune_models.py` (training flow and split/weight policy)
3. `scripts/nn_models.py` (model formulas and clamp policy)
4. `scripts/evaluate_model.py` + `results/evaluation_report.json` (actual performance)
5. `pipeline_ui/app.py` and §4 (what is production-wired vs standalone)

---

## Appendix A: Chronological Session Changelog (Historical Development)

_All day-to-day development notes preserved for reference. Latest session (2026-03-21, fusion fix) summarized in §0. See §5 for full production feature details._

---

### Session 2026-03-21 — Step 2 Fusion Architecture Fix (Correction Collapse Resolved)

**Focus:** Diagnose why the fusion layer was not learning, cross-validate three external reviews against the actual code, and implement the consolidated fix.

#### Root-Cause Diagnosis

Three external reviews were collected and cross-validated against the actual code. Key finding: the code had **already been partially modified** between the training run that produced the reviewed results and the current source. The mismatch was confirmed mathematically:
- Training log showed `reg=0.100` (constant) at all 40 epochs → old code had `correction_reg = 0.1` constant, not the scheduled 0.05 in current code
- Per-rep table: `48.4 + 0.501 × (−0.654) = 48.07` ✓ → training ran with old `heuristic + confidence × correction` formula, not the current `heuristic + residual`

**Root causes confirmed:**

1. **Correction collapse (primary):** L1 regularization (`0.1 × |correction|`) penalised every deviation from zero. With heuristic already at Pearson=0.78, the global minimum was "apply a near-zero constant offset." Result: correction std = 0.05 (every rep got the same −0.4 adjustment).

2. **Orphaned confidence head:** The formula was changed from `heuristic + confidence × correction` to `heuristic + residual`, but the `confidence_head` was kept in the model and computed `confidence = self.confidence_head(fused)` with no use in the final score. It received gradient only from a tiny `0.001 × |conf − 0.5|` penalty and contributed nothing.

3. **Frozen encoders:** BiLSTM and ST-GCN were frozen during Phase 3. Phase 1-2 trained them on sub-metric prediction targets (smoothness, depth, etc.), not on "where does the heuristic get it wrong." The fusion layer received fixed embeddings that weren't aligned to the residual prediction task.

4. **Dropout too aggressive:** `Dropout(0.3)` in the correction head killed 30% of neurons per forward pass in batches of 16 from 121 training samples — too much noise for stable residual learning.

5. **Phase 3 training instability:** Val loss started at 406 at epoch 1 and immediately rose; the best checkpoint was saved at epoch 1 (near-zero correction magnitude). By epoch 40, corrections had grown to |corr|=15 but the checkpoint was already locked at the collapsed state.

#### What the Reviews Got Right vs Wrong

| Claim | Source | Verdict |
|---|---|---|
| Correction regularization too strong | All 3 | ✅ Correct — primary cause |
| Frozen encoders in Phase 3 | All 3 | ✅ Correct |
| Dataset imbalance (61% 80-100) | All 3 | ✅ Correct |
| Linear baseline beating neural is diagnostic | All 3 | ✅ Correct |
| "Early stopping triggered at epoch 11" | Reviews 2 & 3 | ❌ Phase 3 ran all 40 epochs (no early stopping in code at that time) |
| "Confidence not used in score" | Review 1 | ❌ During training it WAS used; only became true after partial code edit |
| "Just use linear regression" | Review 3 | ⚠️ Valid fallback but ignores architecture; neural now beats linear after fix |
| "Add explicit residual MSE auxiliary" | Reviews 1 & 2 | ❌ Algebraically identical to existing loss — real issue was L1 reg |

#### Changes Implemented (2026-03-21)

**`scripts/nn_models.py` — `HeuristicGuidedFusion`:**
- Removed `confidence_head` entirely (orphaned — output never used in final score)
- Renamed `correction_head` → `residual_head` with `nn.Tanh()` as final layer; scaled by `× 40` after the Sequential → bounds corrections to ±40 pts (covers any realistic heuristic error gap while preventing gradient explosion)
- Reduced dropout `0.3` → `0.1` in `residual_head`
- Return signature changed from `(final_score, correction, confidence)` to `(final_score, residual)` — 2-tuple throughout

**`scripts/finetune_models.py` — Phase 3 (`train_phase_fusion`):**
- **Unfroze both encoders** — BiLSTM and ST-GCN participate in Phase 3 gradient updates
- **Differential LR**: encoders at `lr × 0.05` (= 2.5e-5), fusion at full `lr` (= 5e-4) with `weight_decay=1e-4`
- **Removed all correction regularisation** (`reg_residual`, `reg_confidence` terms)
- **Simplified loss** to pure weighted MSE
- **Increased early_stop_patience** 10 → 25 epochs
- **Removed correction-collapse auto-adjustment logic** (no longer needed)
- Updated per-epoch log: prints `res_std` instead of `conf` — `std > 2.0` by epoch 10 is the health indicator

**`scripts/finetune_models.py` — CLI defaults:**
- `--epochs-fusion` 40 → 60
- `--lr-fusion` 2e-4 → 5e-4

**`scripts/evaluate_model.py`:**
- Updated fusion call to 2-tuple unpack (`pred, corr = fusion(...)`)
- Renamed `correction_statistics` → `residual_statistics`, removed `mean_confidence`
- Per-rep table: renamed `correction` → `residual`, removed `confidence` field
- Failure case entries: renamed `correction` → `residual`, removed `confidence`

#### Before vs After (2026-03-21)

| | Before (old arch) | After (new arch) |
|---|---|---|
| Phase 3 residual std at epoch 1 | ~0.05 (collapsed) | 1.4 |
| Phase 3 residual std at final epoch | ~0.05 (flat) | **14–15** |
| Phase 3 val loss trajectory | 406 → 450+ (diverged) | 483 → **164** (converged) |
| Post-clamp Pearson | 0.8352 | **0.8737** |
| Post-clamp MAE | 10.68 | **9.04** |
| Pre-clamp Pearson | 0.7795 | **0.8552** |
| Pre-clamp MAE | 11.98 | **9.28** |
| vs linear baseline MAE | 10.68 > 10.39 (worse) | **9.28 < 10.39 (better)** ✅ |
| Failure cases (>\|20\|) | 4 | **0** ✅ |

---

### Session 2026-03-21 — Step 2 Documentation Consolidation (Agent Onboarding Focus)

**Focus:** Reduce future coding-agent startup effort by making `CLAUDE.MD` a complete single-stop reference for current implementation state.

**What was documented:**
1. Added a dedicated Step 2 section with script-level responsibilities and end-to-end flow.
2. Added strict data contract summary for indexed loading, rep fields, tensor shapes, and phase behavior.
3. Synced latest evaluation snapshot directly from `results/evaluation_report.json` (post-clamp primary and pre-clamp diagnostics).
4. Added current artifacts map (splits/checkpoints/report) and explicit pending integration status.
5. Updated risks and milestones to reflect the real current bottlenecks (dataset coverage, extreme-score reliability, integration gap, reproducibility).

**Why this reduces onboarding effort:**
- New agents can now identify implementation state, active scripts, contracts, artifacts, and immediate next actions from one document, avoiding repeated full-code rediscovery.

---

### Session 2026-03-21 — Step 2 Implementation Kickoff (Document-First)

**Focus:** Enforce document-first development workflow and start Step 2 implementation for supervised fine-tuning + fusion.

**What was built/planned before coding:**
1. Added a hard process gate: update `CLAUDE.MD` before any source-code edits for Step 2 work.
2. Confirmed implementation scope for new scripts: `scripts/nn_models.py`, `scripts/finetune_models.py`, and `scripts/evaluate_model.py`.
3. Locked data loading policy to `dataset/annotations/index.json` registry only (ignore non-index partial annotation JSONs).
4. Locked evaluation policy to deployment-faithful post-safety-clamp metrics with additional pre-clamp diagnostics.
5. Locked training policy to phased protocol: BiLSTM fine-tuning, ST-GCN fine-tuning, then heuristic-guided gated fusion training.

**Implementation constraints adopted:**
- Split by `video_id` with stratified score buckets and seed 42.
- Use confidence × inverse-bucket sample weighting.
- Apply gradient clipping in every training phase.
- Auto-reduce fusion correction regularization if correction collapse is detected.

---

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

