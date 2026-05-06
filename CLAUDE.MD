# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

_Last updated: 2026-05-06 — See [CHANGELOG.md](CHANGELOG.md) for full session history._

---

## Current Top-Level Structure

- `apps/` — application layer (`desktop-ui/` multi-exercise UI, `api/` inference server, `web/` scaffold)
- `core/exevision/` — modularized logic (`stages/` exercise-parameterized, `neural/`, `training/`, `analysis/`, `feedback/`, `config/` with per-exercise configs)
- `adapters/legacy-cli/` — compatibility bridge (⚠️ `run_stage.py` broken — see Active Issues #8)
- `models/` — model/checkpoint artifacts (per-exercise neural models: `bilstm_{exercise}.pt`, `stgcn_{exercise}.pt`, `fusion_{exercise}.pt`)
- `training_dataset/annotations/` — annotation store
- `_hidden_legacy/` — archived legacy code and 50+ historical run artifacts

---

## Quick Start

### Launch Desktop UI (Multi-Exercise)
```bash
python apps/desktop-ui/app.py
# New: Exercise selector dropdown in UI; choose squat or overhead_press
# New: Stage outputs appear in {exercise}/extracted_features_clean/, etc.
```

### Launch Inference API Server
```bash
pip install -r apps/api/requirements.txt

# PowerShell (Windows):
$env:INFERENCE_API_SECRET="your-secret"; .\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# cmd.exe (Windows):
set INFERENCE_API_SECRET=your-secret && .venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# bash/Linux:
INFERENCE_API_SECRET=your-secret python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# With --reload for development:
INFERENCE_API_SECRET=your-secret python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

# Health check:
curl http://localhost:8000/health
# Submit job (squat inference):
curl -X POST http://localhost:8000/infer -H "Authorization: Bearer your-secret" \
     -d '{"video_url":"<url>","job_id":"<uuid>","exercise":"squat"}'

# Submit job (overhead press or other exercise):
curl -X POST http://localhost:8000/infer -H "Authorization: Bearer your-secret" \
     -d '{"video_url":"<url>","job_id":"<uuid>","exercise":"overhead_press"}'
```

### Run a Single Stage Directly (for debugging)
```bash
# Stages use CWD as the workspace (pipeline.py runs them with cwd=workspace_root).
# Set env vars that pipeline.py normally provides:
export EXEVISION_MODEL_PATH=models/pose_landmarker_heavy.task
export EXEVISION_FACE_MODEL_PATH=models/blaze_face_short_range.tflite

cd pipeline_ui_runs/{run_id}/workspace/
# All stages accept --exercise parameter (defaults to "squat")
python ../../../core/exevision/stages/extract_selected_features.py filtered --video-id <id> --exercise squat
python ../../../core/exevision/stages/classify_views.py --video-id <id> --exercise squat
python ../../../core/exevision/stages/temporal_segmentation.py --video-id <id> --exercise squat
python ../../../core/exevision/stages/scoring.py <id> --exercise squat
python ../../../core/exevision/stages/neural_fusion_inference.py --workspace-root . --exercise squat

# For overhead press:
python ../../../core/exevision/stages/extract_selected_features.py filtered --video-id <id> --exercise overhead_press
# ... etc. (paths and models automatically adapt to exercise)

# Add --no-viz --no-report to skip video/image generation.
# Check an existing workspace for the expected {exercise}/ directory layout.
```

### Run Tests
```bash
pip install pytest

pytest tests/                                          # all tests
pytest tests/test_feedback_e2e.py -v                  # single file
pytest tests/test_feedback_e2e.py::test_feedback_e2e_squat -v  # single test
```

> Tests do NOT require video files or model checkpoints — they test config loading and feedback engine logic only.

### Deploy to Google Cloud Run
```bash
export GCP_PROJECT="your-gcp-project-id"
export GCR_REGION="asia-southeast1"
export IMAGE_NAME="exevision-modelai"
export REGISTRY="${GCR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${IMAGE_NAME}"

# Build + push via Cloud Build (no local model upload required)
gcloud builds submit --tag "${REGISTRY}/${IMAGE_NAME}:latest" --project=$GCP_PROJECT .

# Deploy
gcloud run deploy $IMAGE_NAME \
  --image="${REGISTRY}/${IMAGE_NAME}:latest" \
  --region=$GCR_REGION --memory=4Gi --cpu=2 --timeout=600 \
  --concurrency=1 --min-instances=0 --max-instances=3 \
  --set-env-vars="INFERENCE_API_SECRET=<secret>,CORS_ORIGINS=https://your-app.vercel.app,SUPABASE_URL=https://your-project.supabase.co,SUPABASE_SERVICE_KEY=<service-role-key>"

# Local Docker test (no Supabase — uses local file serving)
docker build -t exevision-modelai:local .
docker run --rm -p 8000:8000 -e INFERENCE_API_SECRET=test-secret exevision-modelai:local
```

### Run Training (Step 2 Neural)
```bash
python core/exevision/training/finetune_models.py   # BiLSTM → ST-GCN → Fusion
python core/exevision/training/evaluate_model.py    # → results/evaluation_report.json
```

### Annotation Quality Check
```bash
python core/exevision/analysis/analyze_annotations.py
```

### Environment
- **Python:** 3.10+ recommended (MediaPipe may require <3.12 on some builds)
- **Dependencies for full pipeline:** `requirements-runtime.txt` (mediapipe, torch, opencv, scipy, etc.)
- **Dependencies for API server only:** `apps/api/requirements.txt` (fastapi, uvicorn, httpx, pydantic)
- **When adding a new pipeline dep:** add to `requirements-runtime.txt` (used in Docker), not just `apps/api/requirements.txt`
- **MediaPipe model:** `models/pose_landmarker_heavy.task` must be present before running Stage 2.5
- **Dataset videos:** `squat/dataset_videos_all/` (not tracked in git; required for pipeline runs)

**API Server Environment Variables (see `apps/api/.env`):**
- `INFERENCE_API_SECRET` — shared secret (required)
- `API_PUBLIC_URL` — defaults to `http://localhost:8000`
- `CORS_ORIGINS` — comma-separated allowed origins, defaults to `*` (unsafe in production)
- `SUPABASE_URL` — if unset, visualizations are served locally via `/results/{job_id}/workspace/...`
- `SUPABASE_SERVICE_KEY` — required only when `SUPABASE_URL` is set

**Visualization Setup:**
- **Local testing:** leave Supabase vars unset → annotated videos served locally
- **Production on Cloud Run:** set both Supabase vars → videos uploaded to Supabase Storage bucket `inference-results` → `result.videos.with_landmarks` is a signed URL (1h expiry)

---

## Architecture & Data Flow

```
Video file/URL
    │
    ▼
Stage 2.5 — extract_selected_features.py
  MediaPipe landmarks → squat/extracted_features_clean/{quality}/{video_id}.json
    │
    ▼
Stage 4 — classify_views.py
  Visibility + Z-depth heuristics → updates feature JSON with view label
    │
    ▼
Stage 5 — temporal_segmentation.py
  FSM phase detection → squat/segmented_reps/{quality}/{video_id}_segmented.json
    │
    ▼
Stage 8 — scoring.py
  Rule-based AQA (knee/depth/lean metrics) → squat/aqa_analysis_simple/.../aqa_simple.json
    │
    ▼
Stage 9 — neural_fusion_inference.py  ← API pipeline only; NOT in desktop UI
  HeuristicGuidedFusion: clamp(heuristic + tanh(residual)×40, 0, 100)
  Loads: bilstm_finetuned.pt + stgcn_finetuned.pt + fusion_layer.pt
    │
    ▼
FeedbackEngine — core/exevision/feedback/engine.py
  Consumes AQA scores + config JSONs → narrative coaching text
    │
    ▼
Result payload — apps/api/pipeline.py → collect_results()
  → POST to callback_url  OR  return via GET /jobs/{id}
  → Supabase upload (annotated video) or local file serving fallback
```

Each stage runs as a **subprocess** inside an isolated workspace under `pipeline_ui_runs/{run_id}/workspace/`. Logs land at `pipeline_ui_runs/{run_id}/logs/{stage_key}.log`. The `RunPaths` dataclass in `core/exevision/config/paths.py` defines this layout.

---

## Multi-Exercise Support (2026-04-09)

### Architecture
- **Exercise Parameter Threading:** API request → pipeline → all stage scripts
- **Config-Driven Metrics:** Each exercise has `core/exevision/config/exercises/{exercise}.json` with thresholds, issue groups, field mappings
- **Workspace Isolation:** `workspace_root/{exercise}/extracted_features_clean/`, `{exercise}/segmented_reps/`, etc.
- **Per-Exercise Models:** `models/bilstm_{exercise}.pt`, `models/stgcn_{exercise}.pt`, `models/fusion_{exercise}.pt` (with fallback to shared models during transition)
- **Dynamic Stages:** `_build_stage_specs(exercise)` in pipeline.py, `_build_stages(exercise)` in app.py

### Current Exercises
1. **Squat** — Fully implemented, backward compatible
2. **Overhead Press** — Config exists (`overhead_press.json`), infrastructure ready for Phase 1 dataset prep; see [OVERHEAD_PRESS_PLAN.md](OVERHEAD_PRESS_PLAN.md) for 3-phase training roadmap

### Desktop UI Changes
- Exercise selector dropdown (new)
- `_on_exercise_changed()` handler rebuilds stage specs dynamically
- All stage invocations pass `--exercise` parameter

### API Changes
- `InferRequest` includes `exercise: str = "squat"`
- `/infer` endpoint validates exercise config exists
- Field mapping loaded per-exercise (metric name normalization)
- Model paths constructed with exercise context

### To Add Another Exercise
1. Create `core/exevision/config/exercises/{exercise_name}.json` with correct schema
2. (Optional) Create per-exercise models: `bilstm_{exercise_name}.pt`, etc.
3. Test: `POST /infer` with `"exercise": "{exercise_name}"`
4. Dockerfile validates automatically (checks all *.json files exist)

### FitnessAQA Dataset Integration (Overhead Press Phase 1)
**Context:** FitnessAQA dataset (~5,490 unlabeled + ~2,260 labeled videos) is being integrated for overhead press pre-training.

**Workflow (Phase 1 self-supervised pre-training):**
1. Run batch extraction on FitnessAQA unlabeled videos:
```powershell
Set-Location "D:\FitnessAQA\ohp_phase1\workspace"
$env:EXEVISION_MODEL_PATH = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\models\pose_landmarker_heavy.task"
$env:EXEVISION_FACE_MODEL_PATH = "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\models\blaze_face_short_range.tflite"

& "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\python.exe" `
    "C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\extract_selected_features.py" `
    unfiltered `
    --exercise overhead_press `
    --video-dir "D:\FitnessAQA\Overhead Press\Unlabeled_Dataset-OHP\Unlabeled_Dataset\videos\videos" `
    --max-videos 100 `
    --no-viz
```
2. Scale `--max-videos` parameter: start with 10–100 for testing, then run full dataset (5490 videos)
2. Outputs land in `overhead_press/extracted_features_clean/raw_unfiltered/` (one Euro filtering only; no squat-specific post-processing)
3. Seated variant (`seated_overhead_press/extracted_features_clean/raw_unfiltered/`) auto-generated for models robust to knee-occluded views
4. Non-fatal extraction: `--include-poor` saves marginal-quality videos; downstream models filter by confidence thresholds
5. Run stages 4→5→8 (classify_views, temporal_segmentation, scoring) with `--exercise overhead_press` to populate temporal/biomechanical annotations
6. Feed segmented reps + heuristic scores into BiLSTM/ST-GCN fine-tuning pipeline (Phase 2)

**Quality gates:**
- Frames with overall landmark visibility <0.60: entire video skipped
- Videos with >10% of frames showing 2+ people detected: skipped (training expects single-subject)
- Poor-quality frames within otherwise-good videos: included (scoring stage applies per-frame confidence gates)

**Data distribution targets:**
- Unlabeled pre-training: all 5,490 (maximize diversity)
- Labeled fine-tuning: 2,260 (via `prepare_ohp_dataset.py`, Phase 3)
- Seated variant: mirrors all extractions; enables knee-invariant model training

---

## Active Components

**Core pipeline stages:**
- `core/exevision/stages/extract_selected_features.py` — MediaPipe pose extraction + quality gating
- `core/exevision/stages/classify_views.py` — Visibility-based view classification
- `core/exevision/stages/temporal_segmentation.py` — FSM phase segmentation with strict legal sequences (squat + OHP paths)
- `core/exevision/stages/scoring.py` — Rule-based AQA scoring and metrics
  - **Squat scoring:** View-dependent weights, 4 metrics (knee valgus, forward lean, depth, below_parallel)
  - **OHP scoring (NEW):** View-independent, 5 biomechanical metrics in body-local 3D frame (grip_ratio, bar_path_deviation, ROM, lockout, elbow_flare); supports `overhead_press` and `seated_overhead_press` with identical metric logic
- `core/exevision/stages/neural_fusion_inference.py` — Neural fusion stage (Step 9)

**UI & Orchestration:**
- `apps/desktop-ui/app.py` — Desktop UI (Inference + Annotation tabs); heuristic-only (Stage 9 not wired yet)
- `apps/api/main.py` — FastAPI inference server (`POST /infer`, `GET /jobs/{id}`, `GET /health`)
- `apps/api/pipeline.py` — Pipeline execution; calls all 5 stages in sequence including Stage 9

**Annotation/Data:**
- `core/exevision/analysis/select_annotation_samples.py` — Strategic rep subset selection
- `core/exevision/analysis/analyze_annotations.py` — Annotation quality self-check
- `training_dataset/annotations/index.json` — Master annotation list + stats

**Feedback:**
- `core/exevision/feedback/engine.py` — Rep + session coaching text; score-band tone policy (80-100=soft, 70-79=strict, <70=very_strict); tier language (excellent/strong/okay/needs_work/focus_here)

**Training:**
- `core/exevision/training/finetune_models.py` — Phased fine-tuning (BiLSTM → ST-GCN → Fusion); stratified split seed 42; train=121, test=26
- `core/exevision/training/evaluate_model.py` → `results/evaluation_report.json`

> `_hidden_legacy/` contains archived legacy code and historical run workspaces — do not edit. `adapters/legacy-cli/run_stage.py` is broken post-migration (see Active Issues #8).

---

## Production Flow

### Inference Tab (Desktop UI + API Server)

Both entry points execute core stages (2.5→4→5→8) with Stage 9 (neural fusion) in API only:

```
Input video/folder
  → workspace created under pipeline_ui_runs/{run_id}/
  → stages 2.5 → 4 → 5 → 8 → 9 (neural_fusion, API only) execute as subprocesses
  → JSON outputs + video overlays written to workspace
  → logs at logs/{stage_key}.log
```

**Extraction Mode:** Stage 2.5 supports `Filtered` (default: Savitzky-Golay smoothing + quality gating) or `Unfiltered (raw)` (One Euro Filter only). The desktop UI exposes a dropdown selector (~line 200 in `app.py`).

### Annotation Tab (Desktop UI Only)

```
Folder scan → color-coded listbox (green=processed, red=unprocessed, ✓=annotated)
  → user double-clicks video
    → if pipeline outputs exist: load immediately
    → if not: auto-run stages 2.5→4→5→8 in background
  → dual video display (raw + visualized)
  → human scores 0–100 + flags (F1–F6) + confidence + notes
  → heuristic score hidden until submission (bias-blind)
  → save to training_dataset/annotations/videos/{video_id}.json
  → auto-advance to next rep
```

---

## Pipeline Stage Reference

**Stage 2.5 — Pose Extraction:** `core/exevision/stages/extract_selected_features.py`

Key thresholds: `MIN_LANDMARK_CONFIDENCE=0.4`, `MIN_KEY_JOINT_CONFIDENCE=0.5` (hip/knee/ankle), `MANDATORY_CHAIN_MIN_RATIO=80%`, `MIN_FRAME_VISIBILITY=0.50` (skip if any frame below this), `MAX_MULTI_PERSON_RATIO=10%` (skip if >10% of frames detect 2+ people). Output: `{exercise}/extracted_features_clean/{quality}/{video_id}.json`

**Stage 2.5 CLI Parameters** (in addition to mode `filtered|unfiltered` and `--video-id`):
- `--exercise {name}` — Target exercise config (default: "squat"); controls output path routing and seated variant behavior
- `--video-dir /path/to/videos` — Override default dataset root (e.g., for non-NTFS drives; default: `{exercise}/dataset_videos_all/`)
- `--max-videos N` — Stop after processing N unprocessed videos (cap before already-processed filter; useful for iteration)
- `--include-poor` — Save videos with poor-but-detectable landmarks to `raw_unfiltered/` output folder (default: skip)
- `--no-viz` — Skip visualization video generation (saves disk)
- `--no-report` — Skip HTML report generation

**Rejection Registry:** Skipped videos are recorded in `.skipped_videos_registry.json` (sibling to output folders). Stores reason + timestamp for each rejection. On re-runs, rejected videos are skipped immediately without re-processing. Registry survives across sessions/interruptions.

**Stage 2.5 Seated Variant** (Overhead Press only):
- When `--exercise overhead_press`, stage creates dual outputs:
  - Main: `overhead_press/extracted_features_clean/{quality}/{video_id}.json` (full landmarks)
  - Seated: `seated_overhead_press/extracted_features_clean/{quality}/{video_id}.json` (leg landmarks zeroed: indices 25–32, maintaining array shape)
- Both sets receive identical preprocessing (MediaPipe extraction, One Euro filtering, post-processing) except leg landmark nullification happens after segmentation
- Visualizations mirror to `{exercise}/visualized_poses_clean/{quality}/` and `seated_{exercise}/visualized_poses_clean/{quality}/`
- Use seated variant for training models robust to knee-occluded camera angles (common in seated shoulder press form videos)

**Stage 4 — View Classification:** `core/exevision/stages/classify_views.py`
Determine camera view (front/back/side) via eye visibility + shoulder width + nose-vs-hip Z-depth. Output: view label in feature JSON.

**Stage 5 — Temporal Segmentation:** `core/exevision/stages/temporal_segmentation.py` (Refactored 2026-05-05)

Exercise-parameterized phase detection with biomechanical control signals:
- **Squat:** normalized hip Y-displacement (positive = descending). FSM cycle: `IDLE → ECCENTRIC → [ISOMETRIC] → CONCENTRIC → IDLE`. Knee-angle gate for ECCENTRIC entry and IDLE return.
- **Overhead Press / Seated OHP:** inverted wrist Y-displacement (positive = wrists rising). FSM cycle: `IDLE → CONCENTRIC → [ISOMETRIC] → ECCENTRIC → IDLE`. No knee-angle checks — fully wrist/elbow signal driven.

**FSM Cycles:**
| Exercise | Cycle | Primary signal | Knee check |
|----------|-------|---------------|-----------|
| squat | IDLE→ECC→[ISO]→CONC→IDLE | hip Y descent | ✅ required |
| overhead_press / seated_overhead_press | IDLE→CONC→[ISO]→ECC→IDLE | wrist Y ascent | ❌ removed |

**Key Features:**
- Separate `_detect_phases_squat()` and `_detect_phases_ohp()` — zero shared squat logic in OHP path
- Exercise-specific valid transitions: `VALID_TRANSITIONS` (squat), `OHP_VALID_TRANSITIONS` (OHP)
- `_detect_repetitions()` and `_detect_repetitions_phase_only()` both route to exercise-specific implementations
- `_is_ohp` property on `SquatStateMachine` drives all routing
- Debug mode: `--debug-phases` outputs per-rep acceptance reasoning
- Backward compatible: all defaults remain "squat"

**CLI Parameters:**
```bash
python core/exevision/stages/temporal_segmentation.py \
  --video-id <id> \
  --exercise overhead_press \          # squat (default), overhead_press, seated_overhead_press
  --video-dir "/path/to/videos" \      # Search custom directory for videos (recursive)
  --debug-phases \                     # Enable phase detection debug logs
  --quality excellent \                # Filter by quality tier
  --no-viz                             # Skip visualization generation
```

**Video Discovery (Priority Order):**
1. Annotated poses from Stage 2.5 (if quality specified): `{exercise}/visualized_poses_clean/{quality}/{video_id}_annotated.mp4`
2. Custom video directory (if `--video-dir` provided): searches recursively
3. Default dataset directory: `{exercise}/dataset_videos_all/`

**Example Usage (FitnessAQA Dataset):**
```powershell
cd D:\FitnessAQA\ohp_phase1\workspace
& "path\to\venv\Scripts\Activate.ps1"
python "path\to\temporal_segmentation.py" `
  --video-id 80830_5 `
  --exercise seated_overhead_press `
  --video-dir "D:\FitnessAQA\Overhead Press\Unlabeled_Dataset-OHP\Unlabeled_Dataset\videos\videos" `
  --debug-phases
```

**Output:**
- Main: `{exercise}/segmented_reps/{quality}/{video_id}_segmented.json` — reps with phase breakdown
- Visualization: `{exercise}/visualized_segmentation/{quality}/{video_id}_phases.mp4` — phase annotations on video (if `--no-viz` not set)
- Debug: `{exercise}/debug_phases/{quality}/{video_id}_phases.json` — per-rep acceptance reasoning (if `--debug-phases` enabled)

Key params: `DOWNWARD_VELOCITY_THRESHOLD=0.0015`, `UPWARD_VELOCITY_THRESHOLD=-0.0015`, `MOTION_CONFIRM_FRAMES=2`

**Stage 8 — Scoring & AQA:** `core/exevision/stages/scoring.py`
- **Squat:** knee_angle, knee_valgus_ratio, forward_lean_deg, squat_depth. Thresholds view-dependent. Per-leg independent depth to avoid diagonal-view Z-drift. Output: `squat/aqa_analysis_simple/{quality}/{video_id}/aqa_simple.json`
- **OHP / Seated OHP:** 5 biomechanical metrics in body-local 3D frame (grip_ratio, bar_path_deviation, ROM, lockout, elbow_flare). View-independent; body frame built from shoulder/hip landmarks. Config-driven thresholds from `overhead_press.json`. Output: `{exercise}/aqa_analysis_simple/{quality}/{video_id}/aqa_simple.json`

**Stage 9 — Neural Fusion:** `core/exevision/stages/neural_fusion_inference.py`
`neural_score = clamp(heuristic + tanh(residual) × 40, 0, 100)`. Requires `bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt`. Output: merged AQA with neural scores + three-judge breakdown (BiLSTM/ST-GCN/Heuristic).

---

## Annotation & Data

**Flow:** Raw video → stages 2.5/4/5/8 → Annotation Tab → `training_dataset/annotations/videos/{video_id}.json`

**Per-video JSON contains:** video metadata, per-rep (rep_id, start/end frames, phases, heuristic metrics, human score, flags, severity scales 0–5, confidence, notes).

**Master index:** `training_dataset/annotations/index.json` — tracks all annotated videos + summary statistics.

**Bias-blind design:** heuristic score hidden until human score submitted; revealed with Δ disagreement.

---

## File Structure

| Path | Purpose |
|------|---------|
| `apps/desktop-ui/app.py` | Desktop UI (Inference + Annotation tabs) |
| `apps/api/main.py` | FastAPI server entry point |
| `apps/api/pipeline.py` | Pipeline execution (all 5 stages + result collection) |
| `core/exevision/stages/` | Pipeline stage scripts (2.5, 4, 5, 8, 9) |
| `core/exevision/neural/nn_models.py` | BiLSTMScorer, STGCNScorer, HeuristicGuidedFusion |
| `core/exevision/neural/nn_utils.py` | Skeleton graph, tensor contracts |
| `core/exevision/training/finetune_models.py` | Phased fine-tuning (BiLSTM → ST-GCN → Fusion) |
| `core/exevision/training/evaluate_model.py` | Test evaluation → `results/evaluation_report.json` |
| `core/exevision/feedback/engine.py` | Rep + session coaching text generator |
| `core/exevision/config/exercises/squat.json` | Score brackets & thresholds (required at runtime) |
| `core/exevision/config/templates/feedback_templates.json` | Phrase catalogs (required at runtime) |
| `core/exevision/config/paths.py` | `RunPaths` dataclass — workspace directory layout |
| `training_dataset/annotations/` | Annotation store (index.json + per-video JSONs) |
| `models/` | Checkpoints (bilstm_finetuned.pt, stgcn_finetuned.pt, fusion_layer.pt) |
| `results/evaluation_report.json` | Latest neural model test metrics |
| `Dockerfile` | Python 3.10-slim + OpenGL ES libs + models baked in |
| `requirements-runtime.txt` | Full pipeline deps for Docker (fastapi + mediapipe + torch cpu, etc.) |
| `apps/api/requirements.txt` | API server only (fastapi, uvicorn, httpx, pydantic) |
| `cloudbuild.yaml` | GCP Cloud Build CI/CD (build + push + deploy) |
| `apps/web/INTEGRATION_PROMPT.md` | Web app integration spec (handoff to Next.js repo) |
| `apps/web/FEEDBACK_INTEGRATION_PROMPT.md` | Feedback narrative UI contract (rep card layout) |
| `docs/superpowers/plans/` | Implementation plans (GCR deployment, web app integration) |

---

## Technical Risks & Active Issues

1. **`adapters/legacy-cli/run_stage.py` broken:** Maps stage keys to deleted `scripts/*.py` paths. Fails with "Stage script not found". Use `apps/desktop-ui/app.py` or invoke `core/exevision/stages/` directly.

2. **`extract_selected_features.py` silent exit-0 on failure:** When all videos fail, `run_extraction()` returns normally without `sys.exit(1)` (line 1624). Pipeline continues with empty output dirs; downstream stages fail with misleading "Features JSON not found". Fix: `sys.exit(1)` when all videos fail, or validate non-empty output after each stage.

3. **Error message truncated to 100 chars:** Line 1581 prints `error[:100]`; full traceback in `error_details` is never logged. When diagnosing extraction failures, run the stage script directly with the `.venv` Python to see full output.

4. **Stage 9 not in desktop UI:** Desktop UI (`apps/desktop-ui/app.py`) is heuristic-only; infrastructure ready for neural toggle. Stage 9 is fully active in `apps/api/pipeline.py`. Wire in desktop neural mode with explicit toggle before launch (deferred).

5. **GCR cold starts (~30–60s):** `--min-instances=0` triggers cold starts. Set `--min-instances=1` to eliminate (~$10/month). The web integration plan handles this with a "Warming up…" state + 90s retry.

6. **Feedback config JSONs must exist in image:** Both `core/exevision/config/exercises/squat.json` and `core/exevision/config/templates/feedback_templates.json` must be present. The Dockerfile now asserts this at build time and discovers available exercise configs.

7. **Callback URL must be public:** `callback_url` pointing to `http://localhost:3000/...` from Cloud Run will fail with "connection refused".

8. **Result payload shape:** Supabase callback stores `payload.result`. Frontend must read `result_json.feedback`, not `result_json.result.feedback`. Normalize with `result_json?.result ?? result_json` before reads.

9. **Rep feedback alignment:** Bind rep narratives by `rep_id` (not array index). Guard rendering by `status === "done"` to avoid false "no feedback" states.

10. **Single vs multi-rep rendering:** For `rep_count == 1`, bypass rep tabs; render one card directly. For `rep_count > 1`, show selector tabs.

11. **`neural_available` flag:** If `feedback` is missing, do not assume neural was skipped unless `neural_available === false` is explicitly set in the result.

12. **API schema optional fields:** `phase_timeline` and `kinematic_data` are optional in the result payload (emitted by Stage 5). Frontend rendering must be conditional on their presence.

13. **Dataset distribution risk:** Test set is 26 reps with no 0–20 coverage. Neural model generalization to poor-form squats remains unvalidated.

14. **Per-metric spatial MAE high (≈24):** On diagonal views, ST-GCN depth/knee-tracking predictions are noisy. Would benefit from more training data or view-conditioned normalization.

15. **Path coupling risk (RESOLVED 2026-04-07):** Stage scripts now fully parameterized. Paths computed at startup from exercise name; works correctly in container CWD `/app` and local workspaces.

16. **Neural fusion non-fatal policy:** Stage 9 failure does not abort the pipeline. Result is returned with `neural_available: false` and heuristic-only scores.

17. **Missing pyproject.toml / lockfile:** No frozen dependency manifest yet. `requirements-runtime.txt` covers runtime but is unpinned. Reproducibility risk.

18. **Analysis tooling:** Core analysis scripts are in `core/exevision/analysis/` (e.g., `analyze_annotations.py`, `select_annotation_samples.py`). Historical `analyze_results.py` may exist only in workspace snapshots.

19. **Single-exercise assumption (RESOLVED 2026-04-07):** Multi-exercise infrastructure now complete. Each exercise has config; workspace paths parameterized; all stages accept `--exercise` CLI arg with defaults for backward compat.

---

## Session Log

Full history in [CHANGELOG.md](CHANGELOG.md).

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
