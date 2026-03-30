# CLAUDE.MD — ExeVision AI Project Knowledge Base

_Last updated: 2026-03-30 (GCR deployment live; feedback fallback + config guardrails; neural fusion non-fatal fallback policy; OpenGL ES Dockerfile fix; json discovery fixes in stages 2.5, 8, and 9; web layout/schema contract updates)_

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

**GCR Deployment (2026-03-29):** The FastAPI inference server is now containerized and deployed to Google Cloud Run. Key files: `Dockerfile` (bakes all models into image, OpenGL ES libs added for MediaPipe), `requirements-runtime.txt` (full pipeline deps including tqdm, matplotlib), `cloudbuild.yaml` (CI/CD). Several runtime bugs were found and fixed during containerization: OpenGL ES missing libs, json file discovery path assumptions in stages 2.5/8/9, feedback config file availability in image builds, and feedback fallback handling in `pipeline.py` when narrative configs are unavailable. Web app integration plan is written and ready for execution (`docs/superpowers/plans/2026-03-29-webapp-gcr-integration.md`).

**Web app scope note (2026-03-29):** `apps/web/` in this repository currently contains integration prompt docs only (`INTEGRATION_PROMPT.md`, `FEEDBACK_INTEGRATION_PROMPT.md`), not the actual Next.js implementation files. UI layout and type references in those prompts are contracts for the separate web repo.

**Web results layout contract (2026-03-30):** In the web handoff contract, rep cards are narrative-first (`feedback.reps[N].text`), while judges + metrics sit in a collapsible details panel. For single-rep runs (`rep_count == 1`), render one focused card without tabs; for multi-rep runs, selector tabs remain but feedback should be bound by `rep_id`, not only array index.

**API schema extension (2026-03-30):** `apps/api/pipeline.py` now emits optional rep-level charting fields from Stage 5 alongside existing scores: `phase_timeline` (phase spans + tempo summary) and `kinematic_data` (ROM time-series as `{time, rom}` points). Frontend rendering should be conditional when these fields are absent.

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

### Deploy to Google Cloud Run
```bash
# One-time setup (set your project/region)
export GCP_PROJECT="your-gcp-project-id"
export GCR_REGION="asia-southeast1"
export IMAGE_NAME="exevision-modelai"
export REGISTRY="${GCR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${IMAGE_NAME}"

# Build and push via Cloud Build (builds on GCP — no local upload of model files)
gcloud builds submit --tag "${REGISTRY}/${IMAGE_NAME}:latest" --project=$GCP_PROJECT .

# Deploy
gcloud run deploy $IMAGE_NAME \
  --image="${REGISTRY}/${IMAGE_NAME}:latest" \
  --region=$GCR_REGION --memory=4Gi --cpu=2 --timeout=600 \
  --concurrency=1 --min-instances=0 --max-instances=3 \
  --set-env-vars="INFERENCE_API_SECRET=<secret>,CORS_ORIGINS=https://your-app.vercel.app"

# Local Docker test
docker build -t exevision-modelai:local .
docker run --rm -p 8000:8000 -e INFERENCE_API_SECRET=test-secret exevision-modelai:local
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

## 5) Pipeline Stages (Brief Reference)

**Stage 2.5 — Pose Extraction:** `core/exevision/stages/extract_selected_features.py`
Extract MediaPipe landmarks. Key thresholds: `MIN_LANDMARK_CONFIDENCE=0.4`, `MIN_KEY_JOINT_CONFIDENCE=0.5` (hip/knee/ankle), `MANDATORY_CHAIN_MIN_RATIO=80%`. Filtered mode (default) includes Savitzky-Golay smoothing; unfiltered mode uses One Euro Filter only. Output: `squat/extracted_features_clean/{quality}/{video_id}.json`

**Stage 4 — View Classification:** `core/exevision/stages/classify_views.py`
Determine camera view (front/back/side) using eye visibility + shoulder width + nose-vs-hip Z-depth. Output: view label in feature JSON.

**Stage 5 — Temporal Segmentation:** `core/exevision/stages/temporal_segmentation.py`
Detect reps and phases (idle/eccentric/isometric/concentric) via FSM with strict sequencing. Key params: `DOWNWARD_VELOCITY_THRESHOLD=0.0015`, `UPWARD_VELOCITY_THRESHOLD=-0.0015`, `MOTION_CONFIRM_FRAMES=2`. Output: `squat/segmented_reps/{quality}/{video_id}_segmented.json`

**Stage 8 — Scoring & AQA:** `core/exevision/stages/scoring.py`
Compute rule-based scores (0–100) from metrics: knee_angle, knee_valgus_ratio, forward_lean_deg, squat_depth. Thresholds are view-dependent (see code). Per-leg independent depth computation to avoid diagonal-view Z-drift. Output: `squat/aqa_analysis_simple/{quality}/{video_id}/aqa_simple.json`

**Stage 9 — Neural Fusion Inference:** `core/exevision/stages/neural_fusion_inference.py`
Load saved checkpoints (bilstm_finetuned.pt, stgcn_finetuned.pt, fusion_layer.pt) and apply corrections to heuristic scores. Formula: `neural_score = clamp(heuristic + tanh(residual) × 40, 0, 100)`. Output: merged AQA with neural scores + sub-metric predictions.

---

## 6) Annotation & Data

**Annotation flow:** Raw video → (Stages 2.5/4/5/8) → Annotation Tab → Per-video JSON stored in `training_dataset/annotations/videos/{video_id}.json`
Master index: `training_dataset/annotations/index.json`

**Key design:** Heuristic score hidden until human score submitted (bias-blind UI); continuous severity scales (0–5); auto-saves per rep; batch processing supported.

---

## 7) File Structure (Canonical Paths Post-Migration)

| Path | Purpose |
|------|---------|
| `apps/desktop-ui/app.py` | Desktop UI (dual-tab: Inference + Annotation) |
| `apps/api/` | FastAPI inference server (`main.py`, `pipeline.py`) |
| `core/exevision/stages/` | Pipeline stage scripts (2.5, 4, 5, 8, 9) |
| `core/exevision/neural/` | PyTorch models (nn_models.py, nn_utils.py) |
| `core/exevision/training/` | Fine-tuning & evaluation (finetune_models.py, evaluate_model.py) |
| `core/exevision/analysis/` | Analysis tools (analyze_annotations.py) |
| `training_dataset/annotations/` | Annotation store (index.json + per-video JSONs) |
| `models/` | Checkpoints (bilstm_finetuned.pt, stgcn_finetuned.pt, fusion_layer.pt) |
| `_hidden_legacy/` | Archived legacy code & runs (50+ documented) |
| `Dockerfile` | Container build (Python 3.10-slim + OpenGL ES + models baked in) |
| `.dockerignore` | Excludes training_dataset/, _hidden_legacy/, pretrain checkpoints |
| `requirements-runtime.txt` | Full pipeline runtime deps (fastapi, mediapipe, torch cpu, tqdm, etc.) |
| `cloudbuild.yaml` | GCP Cloud Build CI/CD config (build + push + deploy on trigger) |
| `apps/web/INTEGRATION_PROMPT.md` | Web app integration spec (handoff reference) |
| `apps/web/FEEDBACK_INTEGRATION_PROMPT.md` | Feedback narrative UI contract (rep card layout + fallback behavior) |
| `core/exevision/config/exercises/squat.json` | Required feedback exercise config at runtime |
| `core/exevision/config/templates/feedback_templates.json` | Required feedback template config at runtime |
| `docs/superpowers/plans/` | Implementation plans for GCR deployment and web app integration |

---

## 8) Status Summary

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
- ✅ GCR deployment live (2026-03-29): `Dockerfile`, `.dockerignore`, `requirements-runtime.txt`, `cloudbuild.yaml` created; API server deployed on Cloud Run (`asia-southeast1`)

### Partially Complete / Fragmented

- ⚠️ Migration rewiring: `adapters/legacy-cli/run_stage.py` still points to deleted `scripts/` paths (broken — see §9 #8)
- ✅ `apps/desktop-ui/app.py` wiring to `core/exevision/stages/` — fixed 2026-03-28: removed `_detect_runtime_root()`, hardcoded `STAGES_DIR = WORKSPACE_ROOT / "core" / "exevision" / "stages"`
- ⚠️ `analyze_results.py` exists only in historical run workspaces, not in source tree

### Not Yet Realized (vs Stated Ambition)

- ✅ Neural fusion inference (Stage 9) integrated into API pipeline (`apps/api/pipeline.py` `DEFAULT_STAGES`) — live since 2026-03-28
- ✅ Web app backend toggle integration plan written (`docs/superpowers/plans/2026-03-29-webapp-gcr-integration.md`) — pending execution in Next.js repo
- ⚠️ Neural fusion NOT yet integrated into desktop UI (`apps/desktop-ui/app.py`) — desktop still runs heuristic-only
- ✅ Runtime dependency manifest: `requirements-runtime.txt` covers all pipeline deps for container deployment
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
11. **GCR cold starts (~30–60s):** `--min-instances=0` means first request after idle triggers a cold start. The web app integration plan accounts for this with a "Warming up…" state + 90s retry loop. Set `--min-instances=1` to eliminate cold starts (~$10/month for 1 idle instance).
12. **Web app backend toggle not yet implemented:** Integration plan is written (`docs/superpowers/plans/2026-03-29-webapp-gcr-integration.md`) but changes have not been applied to the Next.js repo yet.
13. **Feedback config JSONs must exist in image:** Feedback generation depends on both config files existing at runtime (`core/exevision/config/exercises/squat.json`, `core/exevision/config/templates/feedback_templates.json`). Missing files caused `feedback=null` in Cloud Run logs. Dockerfile now includes build-time file assertions.
14. **Callback URL reachability from Cloud Run:** `callback_url` must be publicly reachable. `http://localhost:3000/...` from Cloud Run fails with connection refused and prevents callback persistence.
15. **Result payload shape gotcha for web app:** Supabase callback flow stores `result_json = payload.result` (result body only). Frontend code must read `result_json.feedback`; if full envelope is stored in another path, it would be `result_json.result.feedback`.
16. **Rep selector/feedback alignment gotcha:** Bind rep narratives by `rep_id` (not only array index), and guard rendering by `status === "done"` to avoid false "no feedback" states during running/failed transitions.
17. **Type/validation pattern for results pages:** Normalize payload shape before reads (`result_json?.result ?? result_json`), validate feedback (`schema_version`, array shape), and only evaluate no-feedback fallback when `status === "done"`.
18. **Conditional rendering (single vs multi-rep):** For `rep_count == 1`, bypass rep tabs and render one card directly; for `rep_count > 1`, show selector tabs and rep-specific narratives.
19. **Do not over-interpret fallback copy:** If `feedback` is missing, do not assume neural was skipped unless `neural_available === false` is explicitly confirmed.

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
| **OpenGL ES missing in container** | **2026-03-29** | **MediaPipe requires `libegl1` and `libgles2` at runtime. Added to `Dockerfile` system package install (`apt-get install libegl1 libgles2`).** |
| **Stage json file discovery path assumption** | **2026-03-29** | **`extract_selected_features.py` and `scoring.py` had hardcoded path assumptions that broke in containerized CWD. Fixed by making json discovery search the full `aqa_analysis_simple/` subtree rather than a specific nested path.** |
| **Neural fusion failure policy update** | **2026-03-29** | **`pipeline.py` now treats neural fusion failure as non-fatal, logs warning, and continues returning result payloads. Neural unavailability is surfaced via `result.neural_available`.** |
| **Feedback null-on-GCR hotfix** | **2026-03-29** | **When feedback config files are missing, `pipeline.py` now emits a schema-compatible fallback `feedback` payload instead of `null`. Dockerfile also asserts required config files exist during image build.** |

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
- **Ops/productization:** Good (API server containerized + deployed to GCR; `requirements-runtime.txt` created; CI/CD via `cloudbuild.yaml`)
- **Roadmap completion:** Partial (symbolic active; dataset collection in progress; neural inference live in API; GCR deployed; web app toggle integration pending)

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

## 11) Key Entry Points & Checkpoints

**UI & API:**
- `apps/desktop-ui/app.py` — Desktop UI (Inference + Annotation tabs)
- `apps/api/main.py` — FastAPI server (`POST /infer`, `GET /jobs/{id}`, `GET /health`; auth: `Authorization: Bearer {INFERENCE_API_SECRET}`)
- `apps/api/pipeline.py` — Pipeline execution; calls stages in sequence

**Stages (see code for full contracts):**
- `core/exevision/stages/extract_selected_features.py` — Stage 2.5
- `core/exevision/stages/classify_views.py` — Stage 4
- `core/exevision/stages/temporal_segmentation.py` — Stage 5
- `core/exevision/stages/scoring.py` — Stage 8
- `core/exevision/stages/neural_fusion_inference.py` — Stage 9

**Neural:**
- `core/exevision/neural/nn_models.py` — BiLSTMScorer, STGCNScorer, HeuristicGuidedFusion (formula: `clamp(heuristic + tanh(residual) × 40, 0, 100)`)
- `core/exevision/neural/nn_utils.py` — Skeleton graph, tensor contracts

**Training & Evaluation:**
- `core/exevision/training/finetune_models.py` — Phased fine-tuning (BiLSTM → ST-GCN → Fusion); loads from `training_dataset/annotations/index.json`; stratified split seed 42
- `core/exevision/training/evaluate_model.py` — Test evaluation → `results/evaluation_report.json`

**Feedback:**
- `core/exevision/feedback/engine.py` — Rep + session coaching text
- `core/exevision/config/exercises/squat.json` — Score brackets & thresholds
- `core/exevision/config/templates/feedback_templates.json` — Phrase catalogs

**Step 2 Checkpoints:**
- `models/bilstm_finetuned.pt`, `models/stgcn_finetuned.pt`, `models/fusion_layer.pt` — Saved weights
- `results/evaluation_report.json` — Latest test metrics

**Known issue:** `adapters/legacy-cli/run_stage.py` broken post-migration (see §9).

---

## 12) Step 2: Neural Fusion (Current State)

**Status:** Implemented in training/eval stack (`core/exevision/training/`); integrated into API; desktop UI still heuristic-only.

**Formula:** `neural_score = clamp(heuristic_score + tanh(residual_head) × 40, 0, 100)` (heuristic-anchored with ±40 correction)

**Training:** Phased fine-tuning (BiLSTM temporal + ST-GCN spatial) → fusion with unfrozen encoders (differential LR). Loads annotations from `training_dataset/annotations/index.json`; stratified split seed 42; train=121, test=26.

**Current performance (post-fix 2026-03-21):**
- Post-clamp: Pearson = 0.8737, MAE = 9.04 (vs heuristic baseline 12.08, linear baseline 10.39)
- Pre-clamp: Pearson = 0.8552, MAE = 9.28 (beats linear ✅)
- 0 failure cases ✅

**Known limitations:**
1. Small test set (26 reps, no 0–20 coverage) — poor-form generalization unvalidated
2. Per-metric spatial MAE high (≈24) on diagonal views — would benefit from more data or view-conditioned normalization
3. Not yet in desktop UI (only API)
4. Phase 3 train/val gap (121 samples → slight overfitting expected)
5. Phase 4 joint fine-tuning not yet validated

**For implementation work:** See `core/exevision/training/finetune_models.py` (flow), `core/exevision/neural/nn_models.py` (formulas), `results/evaluation_report.json` (metrics).

---

## Appendix: Session History

Detailed session history is in [CHANGELOG.md](CHANGELOG.md). For latest status, see §0.

