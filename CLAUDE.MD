# CLAUDE.MD — ExeVision AI Project Knowledge Base

_Last updated: 2026-03-29 (Neural anchor bug fixed; three-judge scoring model; disk optimization; feedback template system designed; tier-language feedback implementation; score-band tone policy for issue cues)_

---

## 0) Latest Status (2026-03-29 — Current)

ExeVision AI remains in **late prototyping / early pre-production** for squat analysis, but the repository has now entered a **structure migration phase**. The codebase was reorganized to improve long-term deployment readiness while preserving existing capabilities.

**Current top-level structure (new):**
- `apps/` — application layer (`desktop-ui/`, `api/` inference server, `web/` scaffold)
- `core/` — modularized code scaffold under `core/exevision/` (`stages`, `neural`, `training`, `analysis`, `rules`, `config`)
- `adapters/` — compatibility bridge (`adapters/legacy-cli/`)
- `models/` — model/checkpoint artifacts
- `training_dataset/` — centralized training/annotation dataset artifacts
- `_hidden_legacy/` — archived legacy layout and historical runtime artifacts

**Operational note:** legacy runtime folders and historical artifacts were moved under `_hidden_legacy/` during cleanup. New scaffolded paths under `apps/` and `core/` now exist and are the intended future canonical structure.

**Step 2 (Neural Fusion) — major improvement recorded 2026-03-21 (unchanged):** A correction collapse bug in `HeuristicGuidedFusion` was diagnosed and fixed: the prior model was applying a near-constant −0.4 point offset regardless of input (residual std ≈ 0.05). After architecture and training changes (see §12 and Appendix A), the model now learns genuine per-rep corrections (residual std ≈ 14–15) and achieves:

| Metric | Before fix | After fix | Δ |
|---|---|---|---|
| Post-clamp Pearson | 0.8352 | **0.8737** | +0.038 |
| Post-clamp MAE | 10.68 | **9.04** | −1.64 pts |
| Pre-clamp Pearson | 0.7795 | **0.8552** | +0.076 |
| Pre-clamp MAE | 11.98 | **9.28** | −2.70 pts |
| Failure cases (>\|20\|) | 4 | **0** | −4 |
| vs linear baseline MAE | 10.39 (neural was worse) | **9.28 beats linear** | ✅ |

**Feedback Engine Tier Language (2026-03-29):** Metric-agnostic tier language system implemented. All metrics now mentioned in coaching narratives using unified tiers (excellent/strong/okay/needs_work/focus_here). New "stable mentions" for ≥75 metrics that didn't improve. Severity-tiered issue cues (needs_work for 60-74, focus_here for <60). Full implementation in `core/exevision/feedback/engine.py`, `core/exevision/config/templates/feedback_templates.json`, and `core/exevision/config/exercises/squat.json`.

**Feedback Issue Tone Policy (2026-03-29):** Rep-level issue tone is now score-band driven in `core/exevision/feedback/engine.py`: scores 80-100 use soft issue language, scores 70-79 use stricter coaching language, and scores below 70 use very strict language. This is a wording-layer change only; API payload schema is unchanged.

---

## Quick Start

### Launch Desktop UI
```bash
python apps/desktop-ui/app.py
```

### Launch Inference API Server (for web app integration)
```bash
python -m pip install -r apps/api/requirements.txt

# PowerShell (Windows):
$env:INFERENCE_API_SECRET="your-secret"; python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
# bash/Linux:
INFERENCE_API_SECRET=your-secret python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# Health check: curl http://localhost:8000/health
# Submit job:   curl -X POST http://localhost:8000/infer -H "Authorization: Bearer your-secret" \
#               -d '{"video_url":"<url>","job_id":"<uuid>"}'
```

### Run Training (Step 2 Neural)
```bash
# Fine-tune all phases (BiLSTM → ST-GCN → Fusion)
python core/exevision/training/finetune_models.py
# Evaluate on test split → writes results/evaluation_report.json
python core/exevision/training/evaluate_model.py
```

### Annotation Quality Check
```bash
python core/exevision/analysis/analyze_annotations.py
```

### Environment
- **Python:** 3.10+ recommended (MediaPipe may require <3.12 on some builds)
- **Key packages:** `mediapipe`, `torch`, `torchvision`, `opencv-python`, `numpy`, `scipy` (no `requirements.txt` yet — see §9)
- **MediaPipe model:** `models/pose_landmarker_heavy.task` must be present before running Stage 2.5
- **Dataset videos:** `squat/dataset_videos_all/` (not tracked in git; required for pipeline runs)

> **Note:** `adapters/legacy-cli/run_stage.py` is currently broken post-migration (points to deleted `scripts/` paths — see §9 issue #8). Use `apps/desktop-ui/app.py` or invoke stage scripts from `core/exevision/stages/` directly.

---

## 1) Project Identity

**Project name:** ExeVision AI  
**Workspace root:** `exevision_modelAI`  
**Domain:** AI-assisted exercise form analysis (current focus: squat)  
**Core approach:** MediaPipe pose extraction + rule-based biomechanical analysis + stage-wise pipeline orchestration.

This repository is now organized around a migration target architecture:
- **Application layer** under `apps/` (desktop UI and future web app)
- **Core modular logic** under `core/exevision/`
- **Compatibility adapters** under `adapters/`
- **Archived legacy runtime** under `_hidden_legacy/` for reference and rollback

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

### ✅ ACTIVE RUNTIME (During Migration)

**Core pipeline logic (canonical scaffold paths):**
- `core/exevision/stages/extract_selected_features.py` — MediaPipe pose extraction + quality gating
- `core/exevision/stages/classify_views.py` — Visibility-based view classification
- `core/exevision/stages/temporal_segmentation.py` — FSM phase segmentation with strict legal sequences
- `core/exevision/stages/scoring.py` — Rule-based AQA scoring and metrics
- `core/exevision/stages/neural_fusion_inference.py` — Neural fusion stage (Step 9)

**UI & Orchestration:**
- `apps/desktop-ui/app.py` — current desktop UI scaffold (Inference + Annotation flow)
- `apps/api/main.py` — FastAPI inference server (HTTP wrapper for web app integration)
- `apps/api/pipeline.py` — pipeline execution logic used by the API server
- `adapters/legacy-cli/launch_pipeline_ui.py` — compatibility launcher
- `adapters/legacy-cli/run_stage.py` — compatibility stage runner

**Annotation/Data Ecosystem:**
- `core/exevision/analysis/select_annotation_samples.py` — strategic rep subset selection
- `core/exevision/analysis/analyze_annotations.py` — annotation quality self-check
- `training_dataset/annotations/index.json` — master annotation list + stats
- `training_dataset/annotations/videos/{video_id}.json` — per-video annotation files

**Historical evidence:** 50+ runs preserved in `_hidden_legacy/pipeline_ui_runs/`

### ⚠️ LEGACY / ARCHIVED

| Component | Status | Notes |
|-----------|--------|-------|
| `_hidden_legacy/2026-03-28/folders/src/` | Archived | Previous modular pipeline reference snapshot. |
| `_hidden_legacy/2026-03-28/folders/rule_based_programs_sample/` | Archived | Reference-only diving AQA snapshot. |
| `_hidden_legacy/2026-03-28/scripts/` | Archived | Superseded scripts moved for cleanup visibility. |
| `_hidden_legacy/pipeline_ui/`, `_hidden_legacy/scripts/`, `_hidden_legacy/squat/` | Archived runtime snapshot | Preserved for rollback/reference while migration finalizes. |
| `_hidden_legacy/pipeline_ui_runs/` | Historical artifacts | Legacy run evidence retained. |

### Key Fragmentation Points

1. **Migration rewiring pending:** some runtime references still assume old path contracts.
2. **Missing dependency manifest:** no finalized `requirements.txt`/`pyproject.toml` lock strategy yet.
3. **Path coupling risk:** legacy assumptions need full migration to centralized config in `core/exevision/config/`.
4. **Dual-state complexity:** both scaffolded new structure and archived legacy snapshot currently coexist.

---

## 4) Current Production Flow

The desktop UI flow is transitioning to `apps/desktop-ui/app.py`, while legacy orchestration references are preserved under `_hidden_legacy/`; two-tab workflows (Inference + Annotation) remain the operating model.

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
  ├─ Save to training_dataset/annotations/videos/{video_id}.json
  └─ Auto-advance to next rep (or next unannotated video)

Batch mode (multi-select):
  ├─ User selects multiple videos
  ├─ Clicks "Process Selected" or "Reprocess Selected"
  └─ Sequential background execution of stages 2.5→4→5→8 for each
```

**Auto-Pipeline:** If video lacks existing pipeline outputs, annotation tab spawns stages 2.5/4/5/8 in an isolated workspace (does not interfere with existing runs).

**Data Format:** Each annotated video produces `training_dataset/annotations/videos/{video_id}.json` with:
- Video metadata (source, view, quality, FPS)
- Per-rep: rep ID, start/end frames, duration, phases, heuristic metrics, human score, human flags, severity scales (0–5), confidence, notes

**Annotation Index:** Master `training_dataset/annotations/index.json` tracks all annotated videos, totals, and summary statistics.

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

### **Stage 9.5 — Feedback Generation (API Post-Processing)**

**Input:** Merged rep outputs from Stage 8 (AQA) and Stage 9 (neural fusion)  
**Output:** Deterministic coaching text per rep + session summary

**Engine:** `core/exevision/feedback/engine.py` (`FeedbackEngine`)
- Loads exercise config: `core/exevision/config/exercises/squat.json`
- Loads template library: `core/exevision/config/templates/feedback_templates.json`
- Classifies score tier (`excellent|good|fair|poor|critical`) from overall score
- Surfaces issues where sub-score `< 75`, grouped by exercise taxonomy
- Adds positive reinforcement only for metrics `>= 75` that improved vs previous rep
- Applies mismatch handling when overall score and sub-metrics disagree
- Builds session summary (average score, trajectory, most improved metric, persistent issue)

**Response payload added under API result:**
```json
{
  "feedback": {
    "schema_version": "1.0",
    "exercise": "squat",
    "reps": [
      {
        "rep_id": 1,
        "score": 72.0,
        "tier": "fair",
        "text": "Decent effort, but a few things to address.",
        "wins": ["smoothness"],
        "issues": ["forward_lean", "hip_depth"]
      }
    ],
    "session": {
      "avg_score": 78.33,
      "trajectory": "improving",
      "most_improved_metric": "forward_lean",
      "persistent_issue": "hip_depth",
      "aggregate_text": "Most improved this session: forward lean. Keep working on: hip depth.",
      "coach_text": "Good session! You improved consistently. Next time, focus on Drop your hips lower - aim for parallel or below."
    }
  }
}
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
Per-Video Annotation JSON (training_dataset/annotations/videos/{video_id}.json)
  +
Master Index JSON (training_dataset/annotations/index.json)
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

### Active Code (post-migration canonical paths)

**Application Layer (`apps/`):**
- `apps/desktop-ui/app.py` — Tkinter dual-tab UI (Inference + Annotation)
  - `PipelineRunnerUI` — Stage orchestration, run management, preview
  - `AnnotationToolUI` — Video browser, rep-by-rep scoring, batch processing
- `apps/api/main.py` — FastAPI inference server (web app integration)
  - `POST /infer` — accept video URL + job_id, run pipeline in background
  - `GET /jobs/{job_id}` — poll status + retrieve result JSON
  - `GET /health` — model/stage readiness check
- `apps/api/pipeline.py` — pipeline execution logic (mirrors desktop UI subprocess flow)
- `apps/api/requirements.txt` — API-only dependencies (`fastapi`, `uvicorn`, `httpx`, `pydantic`)
- `apps/web/INTEGRATION_PROMPT.md` — full integration spec for the Next.js web app agent

**Pipeline Stages (`core/exevision/stages/`):**
- `extract_selected_features.py` — MediaPipe pose extraction + quality gating (Stage 2.5)
- `classify_views.py` — Visibility-based view classification (Stage 4)
- `temporal_segmentation.py` — FSM phase segmentation + strict sequencing (Stage 5)
- `scoring.py` — Rule-based AQA scoring (Stage 8)
- `neural_fusion_inference.py` — Neural fusion inference (Stage 9)

**Neural Models (`core/exevision/neural/`):**
- `nn_models.py` — PyTorch model classes (`BiLSTMScorer`, `STGCNScorer`, `HeuristicGuidedFusion`)
- `nn_utils.py` — Skeleton graph, sequence utilities, adjacency matrix

**Training & Evaluation (`core/exevision/training/`):**
- `finetune_models.py` — Phased fine-tuning orchestrator (BiLSTM → ST-GCN → Fusion)
- `evaluate_model.py` — Test evaluation → `results/evaluation_report.json`
- `pretrain_bilstm.py` — BiLSTM masked reconstruction pretraining
- `pretrain_stgcn.py` — ST-GCN masked reconstruction pretraining

**Analysis Tools (`core/exevision/analysis/`):**
- `select_annotation_samples.py` — Strategic rep subset selection for annotation
- `analyze_annotations.py` — Annotation quality self-check

**Compatibility Adapters (`adapters/legacy-cli/`):**
- `launch_pipeline_ui.py` — Legacy launcher shim
- `run_stage.py` — Stage runner CLI (**currently broken** — points to deleted `scripts/` paths; see §9 #8)

### Data & Artifacts

- `squat/dataset_videos_all/` — Raw dataset videos (not tracked in git)
- `models/pose_landmarker_heavy.task` — MediaPipe Pose Landmarker v2 heavy model
- `models/*.pt` — Neural checkpoints (pretrained + finetuned; see §11 artifacts list)
- `training_dataset/annotations/` — Centralized annotation store
  - `index.json` — Master list + global stats
  - `videos/{video_id}.json` — Per-video annotation files
- `training_dataset/splits.json` — Persisted train/val/test video split
- `_hidden_legacy/pipeline_ui_runs/` — Historical run folders (~50 documented runs)

### Documentation

- `README.md` — Project overview
- `progress.md` — Roadmap & milestone tracking
- `CHANGELOG.md` — Historical session logs (archived from Appendix A, 2026-03-28)
- No `requirements.txt`, `pyproject.toml`, or `setup.py` (infrastructure gap — see §9 #2)

---

## 8) What Is Complete vs Incomplete

### Completed / Working

- ✅ Core pose-view-segmentation pipeline (both script and modular implementations)
- ✅ Rule-based squat scoring and metrics extraction
- ✅ Pipeline UI with run management, logging, preview, and dual tabs
- ✅ Annotation tool with async batch processing, bias-blind UI, and per-video JSON storage
- ✅ 50+ documented production runs
- ✅ Annotation data ecosystem (`core/exevision/analysis/`, `training_dataset/annotations/`)
- ✅ Step 2 neural infrastructure implemented (`core/exevision/neural/`, `core/exevision/training/`)
- ✅ Step 2 phased fine-tuning executed and checkpoints saved (`bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt`)
- ✅ Step 2 evaluation pipeline executed with report artifact (`results/evaluation_report.json`)
- ✅ Step 2 correction collapse diagnosed and fixed (2026-03-21): fusion now learns per-rep residuals (std ≈ 14–15 vs prior 0.05), beating linear baseline on MAE for the first time (9.28 vs 10.39)

### Partially Complete / Fragmented

- ⚠️ Migration rewiring: `adapters/legacy-cli/run_stage.py` still points to deleted `scripts/` paths (broken — see §9 #8)
- ✅ `apps/desktop-ui/app.py` wiring to `core/exevision/stages/` — fixed 2026-03-28: removed `_detect_runtime_root()`, hardcoded `STAGES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "stages"`
- ⚠️ `analyze_results.py` exists only in historical run workspaces, not in source tree

### Not Yet Realized (vs Stated Ambition)

- ✅ Neural fusion inference (Stage 9) integrated into API pipeline (`apps/api/pipeline.py` `DEFAULT_STAGES`) — live since 2026-03-28
- ⚠️ Neural fusion NOT yet integrated into desktop UI (`apps/desktop-ui/app.py`) — desktop still runs heuristic-only
- ❌ Fully reproducible environment manifest (requirements.txt/pyproject.toml)
- ⏳ Human annotation dataset scale-up — tooling exists; current Step 2 run used 147 reps (train=121, test=26), still sparse for robust extreme-score generalization

---

## 9) Technical Risks & Current Status

### Active Issues

1. **Pipeline divergence:** legacy `src/` snapshot archived; `core/exevision/stages/` is canonical. `apps/desktop-ui/app.py` wiring fixed 2026-03-28 (`STAGES_DIR` hardcoded). `apps/api/pipeline.py` already wired correctly. Remaining risk: `adapters/legacy-cli/run_stage.py` still broken (see #8).
2. **Missing dependency manifest:** Onboarding and reproducibility risk
3. **File-path coupling:** Scripts assume fixed relative paths
4. **Step 2 integration gap:** Neural scorer exists as standalone scripts but is not yet wired into production inference tab
5. **Dataset scale and distribution risk:** Current test set is 26 reps with no 0–20 bucket coverage; generalization to poor-form squats remains unvalidated
6. **Per-metric prediction weakness:** Smoothness/control are reasonable (MAE ~10), but depth/lean/knee-tracking MAE remains high (~24) — ST-GCN Phase 2 converges but the spatial signal is inherently noisy on diagonal views
7. **Stage 5 sensitivity tuning:** Shallow-rep detection threshold balance remains iterative across dataset
8. **`adapters/legacy-cli/run_stage.py` is broken post-migration:** Maps stage keys to `scripts/*.py` paths that were deleted during the 2026-03-28 restructure (moved to `core/exevision/stages/`). The adapter will fail at runtime with "Stage script not found". Requires rewiring to point to `core/exevision/stages/` before the CLI is usable.
9. **`extract_selected_features.py` silent exit-0 on failure:** When all videos fail processing, `run_extraction()` returns normally without calling `sys.exit(1)` (line 1624). The stage subprocess exits 0, so `_run_stage` in `apps/api/pipeline.py` does not raise — the pipeline silently continues with empty `extracted_features_clean/` directories. Downstream stages then fail with misleading errors (e.g., "Features JSON not found" from the scoring stage). Fix: add `sys.exit(1)` when `len(errors) == len(video_paths)` (all failed), or validate output directory non-empty in `run_pipeline_sync()` after each stage.
10. **`extract_selected_features.py` error message truncated to 100 chars:** Line 1581 prints `error[:100]` — the full exception traceback (stored in `error_details`) is never logged. When diagnosing extraction failures via `pipeline_ui_runs/{job_id}/logs/extract_selected_features.log`, the root cause is cut off. Run the stage script directly with the `.venv` Python against the failing workspace to see the full traceback.

### Recently Resolved (✓ Fixed)

| Issue | Date | Fix |
|-------|------|-----|
| Quality-gate sensitivity | 2026-03-07 | Nose-vs-hip Z-depth for diagonal view disambiguation (replaced face hallucination) |
| Depth false-negative on diagonal views | 2026-03-11 | Per-leg independent `max()` computation (replaced bilateral averaging) |
| Unfiltered pipeline breakage | 2026-03-11 | Added `raw_unfiltered` discovery/routing to stages 4/5/8 |
| Shallow-squat detection | 2026-03-17 | Low-motion thresholds + strict phase sequencing + transition auto-repair |
| **Step 2 correction collapse** | **2026-03-21** | **Removed L1 reg + orphaned confidence head; unfroze encoders with diff LR; tanh×40 bounded residual. Pre-clamp MAE dropped from 11.98 → 9.28; beats linear baseline; 0 failure cases.** |
| **Linear baseline parity** | **2026-03-21** | **Pre-clamp MAE 9.28 now beats linear (10.39); neural is earning its complexity.** |
| **Neural heuristic anchor always 0 at inference** | **2026-03-29** | **Two-layer fix: (1) `neural_fusion_inference.py` `process_video()` was reading only the segmentation JSON, which has no `heuristic_score` — AND the AQA lookup searched `aqa_analysis_simple/{quality_tier}/` but Stage 8 writes to `aqa_analysis_simple/{source_quality}/{score_tier}/` where source_quality differs (e.g. `unknown` vs `raw_unfiltered`). Fixed by searching the whole `aqa_analysis_simple/` tree. (2) Added defensive correction in `pipeline.py` `collect_results()`: if `|neural - heuristic| > 40` (impossible with tanh×40), re-anchors the score at `heuristic + clamp(deviation, -40, 40)`.** |

### Maturity by Layer (as of 2026-03-29)

- **Data processing pipeline:** Strongly progressed (end-to-end stages fully integrated)
- **Stage 4 (view classification):** Robust (visibility-based + nose-vs-hip Z-depth)
- **Stage 5 (temporal segmentation):** Improved (strict sequencing + transition repair)
- **Stage 8 (scoring/AQA):** Improved (per-leg depth, Z-drift false-negative fixed)
- **Stage 9 (neural fusion inference):** Working — integrated into API pipeline; anchor bug fixed; three-judge output (BiLSTM/ST-GCN/Heuristic) added
- **Annotation tooling:** High maturity (async batch, bias-blind, continuous severity scales)
- **Step 2 training/evaluation tooling:** Improved — fusion architecture fixed, per-rep corrections working, beats linear baseline
- **Step 2 production integration:** API pipeline live (Stage 9 in `DEFAULT_STAGES`); desktop UI still heuristic-only
- **Modular package (`src`):** Medium (clean stages 1–3; later placeholders)
- **Ops/productization:** Medium (good UI + API server; missing lock file)
- **Roadmap completion:** Partial (symbolic active; dataset collection in progress; neural inference live in API, pending in desktop UI)

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
1. **Wire `apps/api/` into the Next.js web app** — ✅ integration implemented and first end-to-end test run completed (2026-03-28). Web app successfully submits jobs and receives results via callback. Fix extraction silent-exit bug (#9 above) so pipeline failures surface cleanly.
2. **Integrate Step 2 neural inference into `pipeline_ui/app.py`** — expose explicit mode switch (heuristic-only vs neural-fused) in the Inference tab; use saved checkpoints (`bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt`) for live prediction
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
- `apps/desktop-ui/app.py` — Dual-tab UI runner (Inference + Annotation)
  - `PipelineRunnerUI` — Inference pipeline executor
  - `AnnotationToolUI` — Annotation folder browser & scorer
- `apps/api/main.py` — FastAPI inference server
  - `POST /infer` — video URL + job_id → background pipeline run
  - `GET /jobs/{job_id}` — poll status; `result` populated when `status=done`
  - `GET /health` — stage scripts + model file readiness check
  - Auth: `Authorization: Bearer {INFERENCE_API_SECRET}` (env var)
  - **Gotcha:** inserts `apps/api/` into `sys.path` at import time so `from pipeline import ...` works when run as `uvicorn apps.api.main:app` from the project root (CWD is root, not `apps/api/`)
- `apps/api/pipeline.py` — pipeline execution logic (no Tkinter; mirrors `app.py` subprocess flow)
  - `run_pipeline_sync()` — runs all 5 stages, returns merged `AnalysisResult` dict
  - `collect_results()` — merges Stage 8 AQA JSON + Stage 9 neural JSON and appends `feedback`
  - `download_video()` — async download from Supabase signed URL
- `apps/web/INTEGRATION_PROMPT.md` — full Next.js integration spec for the web app agent

**Feedback Engine (`core/exevision/feedback/`):**
- `engine.py` — orchestration + dataclass output contracts (`RepFeedback`, `SessionSummary`, `FeedbackResult`)
  - `generate_feedback(rep_scores, video_id)` — main public method
  - `QualityChecker.detect_mismatch()` — mismatch handling (`normal`, `low_overall_no_issues`, `high_overall_has_issues`)
- `rep_comparator.py` — rep-over-rep deltas + improvement tiering
- `session_aggregator.py` — session trajectory, average, most-improved, persistent issue
- `template_renderer.py` — deterministic phrase selection + template slot filling

**Feedback Configs:**
- `core/exevision/config/exercises/squat.json` — score brackets, issue groups, thresholds
- `core/exevision/config/templates/feedback_templates.json` — phrase catalogs and session templates

**Compatibility Adapters:**
- `adapters/legacy-cli/launch_pipeline_ui.py` — Legacy launcher
- `adapters/legacy-cli/run_stage.py` — Stage runner CLI (**broken post-migration** — maps to deleted `scripts/` paths; see §9 #8)

**Pipeline Stages (`core/exevision/stages/`):**
- `extract_selected_features.py` — Pose extraction + quality gating (Stage 2.5)
- `classify_views.py` — View classification (Stage 4)
- `temporal_segmentation.py` — Phase segmentation + strict sequencing (Stage 5)
- `scoring.py` — AQA scoring (Stage 8)
- `neural_fusion_inference.py` — Neural fusion inference (Stage 9)

**Neural Models (`core/exevision/neural/`):**
- `nn_models.py` — Shared PyTorch modules
  - `BiLSTMScorer`: temporal encoder + smoothness/control heads
  - `STGCNScorer`: 5-block ST-GCN encoder + 3 spatial heads + 4 auxiliary heuristic-raw heads
  - `HeuristicGuidedFusion`: heuristic-anchored gated residual (`final = clamp(heuristic + tanh(residual_head) × 40, 0, 100)`)
  - `build_heuristic_vector()`: 15D vector contract
  - `apply_safety_clamps()`: post-prediction rule caps
- `nn_utils.py` — Shared sequence and skeleton utilities
  - Active-joint graph (11 joints), adjacency matrix construction
  - ST-GCN rep extraction and normalization contract
  - Fixed length pad/truncate

**Training & Evaluation (`core/exevision/training/`):**
- `finetune_models.py` — End-to-end supervised fine-tuning orchestrator
  - Index-gated annotation loading from `training_dataset/annotations/index.json`
  - Video-level stratified split by score bucket (seed 42)
  - Sample weighting: confidence × inverse-bucket
  - Training phases: BiLSTM → ST-GCN → Fusion (+ optional joint phase 4)
- `evaluate_model.py` — Test evaluation and report generation
  - Primary metrics on post-clamp predictions
  - Pre-clamp diagnostics
  - Baselines: heuristic-only and linear regression on heuristic vector
  - Output report: `results/evaluation_report.json`
- `pretrain_bilstm.py` — BiLSTM encoder pretraining (masked reconstruction)
- `pretrain_stgcn.py` — ST-GCN encoder pretraining (masked reconstruction)

**Analysis Tools (`core/exevision/analysis/`):**
- `select_annotation_samples.py` — Strategic rep selection for annotation
- `analyze_annotations.py` — Annotation quality check

**Annotation Data:**
- `training_dataset/annotations/index.json` — Master annotation list + stats
- `training_dataset/annotations/videos/{video_id}.json` — Per-video annotation files

**Project Docs:**
- `README.md` — Root project overview
- `progress.md` — Roadmap & milestone tracking
- `CHANGELOG.md` — Historical development session logs
- `CLAUDE.md` — This file

**Current Step 2 Artifacts:**
- `training_dataset/splits.json` — Persisted train/val/test video split
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

Step 2 is implemented as a standalone training/evaluation stack under `core/exevision/training/` and is runnable in the current workspace. The flow is:

1. Load indexed annotations from `training_dataset/annotations/index.json`.
2. Resolve per-video annotation JSON and features JSON.
3. Build rep-level multimodal records (temporal signals + skeleton sequence + heuristic vector).
4. Split by `video_id` using stratified score buckets.
5. Train in phases (BiLSTM, then ST-GCN, then fusion).
6. Evaluate on test split and write `results/evaluation_report.json`.

Production wiring into `apps/desktop-ui/app.py` is not yet implemented (neural inference only; heuristic pipeline is wired).

### 12.0 Score Interpretation for Consumers (Web App / UI)

`neural_score` **is** the fusion output — it is not a separate model. Formula:
```
neural_score = clamp(heuristic_score + tanh(residual_head(fused_encoding)) × 40,  0, 100)
```
The heuristic is the anchor; the neural model applies a correction of up to ±40 points based on what the BiLSTM + ST-GCN encoders detect that the rules miss.

**Display contract for the web app (three-judge model):**

The API returns three independent "judge" scores per rep, plus the fused overall:

| Field | Judge | What it measures | Display as |
|-------|-------|-----------------|-----------|
| `neural_score` | Fusion (all three) | Best single number — heuristic-anchored + neural correction | **Primary score** ("Overall Score") |
| `bilstm_score` | BiLSTM (temporal) | Average of smoothness + control — motion quality over time | "Temporal Judge" |
| `stgcn_score` | ST-GCN (spatial) | Average of depth + forward lean + knee tracking — body geometry quality | "Spatial Judge" |
| `heuristic_score` | Heuristic (rules) | Deterministic rule-based score from exact joint angles | "Rule-Based Judge" |
| `sub_scores` | Per-dimension breakdown | Individual head outputs (smoothness, control, depth, lean, knee tracking) | Expandable detail under each judge |
| `metrics` | Raw biomechanical measurements | Exact angles/ratios with targets | "Biomechanical Metrics" section |

**Overall-level fields** (averaged across reps): `overall_neural_score`, `overall_bilstm_score`, `overall_stgcn_score`, `overall_heuristic_score`.

**Recommended UI hierarchy:**
1. **Top:** Overall fusion score (large, primary)
2. **Three-judge panel:** BiLSTM | ST-GCN | Heuristic side-by-side (equal-weight visual)
3. **Expandable:** Sub-scores under each judge (smoothness/control under BiLSTM, depth/lean/knee under ST-GCN, metric_scores under Heuristic)
4. **Bottom:** Biomechanical metrics (raw numbers with target ranges)

**Why the scores differ:** Each judge sees different aspects:
- BiLSTM → temporal patterns (was the movement smooth and controlled?)
- ST-GCN → spatial quality (was the body positioned correctly?)
- Heuristic → exact geometry (angles, ratios, thresholds)
- Fusion → weighted combination that anchors on heuristic and corrects with neural evidence

**Defensive correction:** If `anchor_correction_applied` is true for a rep, the neural score was re-anchored (original broken value in `neural_score_raw`). The `any_anchor_corrections` flag at the top level signals a data-flow issue occurred.

### 12.2 Data Contract Used by Step 2

#### Source-of-truth loading policy
- Only videos listed in `training_dataset/annotations/index.json` are considered.
- Per-video annotations are read from `training_dataset/annotations/videos/{video_id}.json`.

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

Clamp rules currently applied in `core/exevision/neural/nn_models.py`:

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

1. `CLAUDE.md` §0, §12, §9, §10 (status, contracts, risks, next actions)
2. `core/exevision/training/finetune_models.py` (training flow and split/weight policy)
3. `core/exevision/neural/nn_models.py` (model formulas and clamp policy)
4. `core/exevision/training/evaluate_model.py` + `results/evaluation_report.json` (actual performance)
5. `apps/desktop-ui/app.py` and §4 (what is production-wired vs standalone)

---

## Appendix A: Chronological Session Changelog (Historical Development)

_Latest sessions kept here for quick reference. Older sessions archived to [CHANGELOG.md](CHANGELOG.md)._

---

### Session 2026-03-29 — Score-Band Issue Tone Policy (80+ Softening)

**Focus:** Reduce tone mismatch on high-scoring reps by softening issue wording when overall rep score is high.

**What was done:**
1. Updated `core/exevision/feedback/engine.py` to add score-band issue tone policy via `_resolve_issue_tone_mode()`:
  - `80-100` → `soft`
  - `70-79` → `strict`
  - `<70` → `very_strict`
2. Updated `_build_rep_feedback()` to pass rep `score` into issue-cue generation so tone is decided per-rep.
3. Updated `_group_issue_cues()` behavior by mode:
  - `soft`: avoids harsh cue tiering and prefixes issue lines with "Something to keep in mind:"
  - `strict`: preserves severity-based behavior
  - `very_strict`: forces strong cue tiering and prefixes with "Priority fix:"
4. Preserved backward compatibility and contract stability: no API schema changes, only narrative wording changes.
5. Ran live smoke validation over three reps (`score=91`, `score=74`, `score=65`) and confirmed expected tone output across all three bands.

---

### Session 2026-03-29 — Feedback Template System Designed

**Focus:** Design the slot-based feedback generation system for rep-level and session-level coaching output.

**What was done:**
1. Reviewed SRD §4.3.4 (Decision Feedback Generation) — confirmed Dynamic Template Slot-Filling approach, no generative AI.
2. Designed **Modular Feedback Engine** (Approach 2): exercise config layer + shared template library. Spec written to `docs/superpowers/specs/2026-03-29-feedback-template-design.md`.
3. Key design decisions locked:
   - **Tone-first:** Overall score dictates tone tier (5 brackets: excellent/good/fair/poor/critical)
   - **Issue surfacing:** All sub-metrics below 75 surfaced; severity-banded (±5 rule groups same-urgency issues); predefined taxonomy groups related errors into one cue
   - **Rep-relative progress tracking:** Each rep compares to previous (significant/moderate/slight tiers)
   - **Positive reinforcement:** Wins called out for metrics ≥75 that improved; relative language only (no raw numbers)
   - **Exercise-specific configs:** Each exercise has its own score brackets, metric thresholds, and issue group taxonomy
   - **Deterministic phrase selection:** `hash(video_id:rep_id:metric_key) % len(phrases)` — varied but repeatable, no randomness
   - **Quality check/fallback mode:** Handles mismatch between overall score and sub-metrics (two mismatch cases; defined which signal wins)
   - **Schema versioning:** `schema_version` field in exercise config, template library, and FeedbackResult payload

**New files/folders (design only — not yet implemented):**
- `core/exevision/feedback/` — FeedbackEngine, RepComparator, SessionAggregator, TemplateRenderer
- `core/exevision/config/exercises/squat.json` — squat exercise config
- `core/exevision/config/templates/feedback_templates.json` — shared template library
- `docs/superpowers/specs/2026-03-29-feedback-template-design.md` — full design spec

---

### Session 2026-03-29 — Metric-Agnostic Tier Language Feedback Implementation

**Focus:** Implement unified tier-language feedback narrative system so all metrics appear in coaching text with appropriate tone, not just wins and issues.

**What was done:**
1. **Updated `feedback_templates.json`:**
   - Split `win_phrases` into tier-aware categories: `improving_metric_excellent`, `improving_metric_strong`, `improving_metric_okay`
   - Added **new `stable_phrases` section** (excellent/strong/okay) for metrics ≥75 that aren't wins
   - Expanded `improvement_phrases` with additional variants per tier

2. **Updated `squat.json` exercise config:**
   - Migrated all `single_cues` (forward_lean, hip_depth, knee_valgus, knee_tracking) from flat strings to `{needs_work, focus_here}` dicts
   - Migrated all `combined_cue` entries to severity-tiered dicts
   - Backwards-compatible: dict structure falls back to `needs_work` tier if present

3. **Updated `engine.py` feedback logic:**
   - Added `_metric_phrase_tier()` helper: maps score 90+ → "excellent", 85-89 → "strong", <85 → "okay"
   - Updated `_build_win_texts()` to use tier-aware phrase keys (`improving_metric_{tier}`)
   - **Added `_build_stable_texts()` method** (new): generates brief mentions for ≥75 metrics that didn't improve
   - Updated `_group_issue_cues()` to read tier-aware cue dicts and select based on severity (`< 60 → focus_here`, else `needs_work`)
   - Updated `_resolve_issue_cue_text()` to safely handle both dict and legacy string cues
   - Wired stable mentions into narrative flow: Opener → Wins → Stable → Issues

4. **Narrative structure now covers all metrics:**
   - **Wins** (≥75, improved): Tier-appropriate opener + improvement phrase
   - **Stable** (≥75, no improvement / rep 1): Tier-appropriate brief mention
   - **Issues** (<75): Severity-tiered coaching cue (needs_work / focus_here)

5. **Code review completed:** Implementation matches spec (95% compliant, minor hyphen/em-dash punctuation variance, zero functional impact).

**Implementation plan saved to:** `docs/superpowers/plans/2026-03-29-tier-language.md` (4 tasks, no tests/git per user instruction)

**Example narratives (post-implementation):**
- Rep 1 (score 95): "Excellent squat form! Excellent depth! Excellent knee tracking! Strong forward lean!"
- Rep 2 (score 78): "Good squat overall. Strong depth - improving from last rep! Your knee tracking is okay. Work on keeping your torso more upright."
- Rep 3 (score 45): "Your form needs significant attention. Your descent needs significant work - stay upright and drop much deeper. Knee valgus is critical - your knees are collapsing inward and need immediate attention."

---

### Session 2026-03-28 — Web App Integration Live + Extraction Bug Discovery + Disk Optimization

**Focus:** Wire `apps/api/` into the Next.js web app, achieve first end-to-end test, diagnose pipeline failure, and minimize per-run disk usage for hosting readiness.

**What was done:**
1. Fixed `apps/api/main.py` import bug: added `sys.path.insert(0, str(Path(__file__).parent))` so `from pipeline import ...` works when run as `uvicorn apps.api.main:app` from the project root (CWD is root, not `apps/api/`).
2. Updated Quick Start commands in `CLAUDE.md` for Windows/PowerShell syntax (`python -m pip`, `$env:KEY="val"; python -m uvicorn ...`).
3. Cross-checked the Next.js agent's integration plan against the API contract; identified 7 gaps (signed URL expiry, callback URL env var, polling specifics, error states, safety clamp messages, score color coding, `mode` field flow).
4. First end-to-end test: web app submitted job → Python server received and processed → callback delivered `result_json` → web app displayed results.
5. **Bug discovered:** `extract_selected_features.py` exits 0 even when all videos fail (line 1624 — no `sys.exit(1)`). Pipeline continues silently with empty feature directories; scoring stage produces misleading "Features JSON not found" error. Also: error message truncated to 100 chars at line 1581.
6. Documented fusion score display contract: `neural_score` IS the fusion output (`heuristic + tanh(residual) × 40`). Web app should display it as the primary score; heuristic as secondary reference; `sub_scores` for motion quality breakdown; `metrics` for biomechanical explanation.
7. **Disk optimization for API runs** (`apps/api/pipeline.py` only, no stage logic changed):
   - Pass `--no-viz --no-report` to stage 2.5 (skips annotated MP4 + PNG plots — ~2.6 MB)
   - Pass `--no-viz` to stage 5 (skips phase overlay MP4 — ~1.7 MB)
   - Delete input video copy from workspace immediately after stage 2.5 completes (~1.2 MB)
   - Call `_cleanup_workspace()` after `collect_results()` — removes feature JSONs, segmented reps, and any remaining intermediate dirs; retains logs + final result JSONs
   - Net result: ~6.9 MB per run → ~25 KB per run (logs + 2 KB result JSONs)

**Key bugs documented (not yet fixed):**
- `extract_selected_features.py` line 1624: silent exit-0 on full-batch failure → see §9 issue #9
- `extract_selected_features.py` line 1581: `error[:100]` truncation → see §9 issue #10

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

_Older sessions (2026-03-05 through 2026-03-17) archived to [CHANGELOG.md](CHANGELOG.md)._

## Appendix B: How to Update This Document

**After each development session:**
1. Add a new subsection under "Appendix A: Chronological Session Changelog" with date and focus
2. Summarize what was built/fixed in 3–5 bullet points
3. Include root-cause analysis for any bug fixes (keep full technical detail)
4. Update §0 "Latest Status" with current state snapshot (1–2 paragraphs)
5. Update section headers' "Latest: YYYY-MM-DD" dates if relevant
6. Keep the most recent 1–2 sessions here; move older entries to `CHANGELOG.md`

**Sections that change rarely (no update needed unless major refactor):**
- §1–3 (Project Identity, Vision, Production vs Legacy)
- §4 (Production Flow) — update only if new tabs/modes added
- §6–7 (Annotation Workflow) — update if annotation file format changes
- §11 (File Reference) — update only on new modules/scripts

---

