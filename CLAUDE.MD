# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

_Last updated: 2026-05-14 — See [CHANGELOG.md](CHANGELOG.md) for full session history._

---

## Top-Level Structure

- `apps/` — `desktop-ui/` multi-exercise UI, `api/` inference server, `web/` scaffold
- `core/exevision/` — `stages/` (exercise-parameterized), `neural/`, `training/`, `analysis/`, `feedback/`, `config/`
- `models/` — per-exercise checkpoints: `bilstm_{ex}.pt`, `stgcn_{ex}.pt`, `fusion_{ex}.pt`
- `training_dataset/annotations/` — squat; `training_dataset/ohp_phase3_annotations/` — OHP Phase 3 (165 reps)
- `_hidden_legacy/` — archived; do not edit

---

## Quick Start

### Desktop UI
```bash
python apps/desktop-ui/app.py
# Exercises: squat, overhead_press, seated_overhead_press
# Inference Tab: Select exercise → pick video → "Full Neural Pipeline" runs all 5 stages
#   OHP neural output (lockout, elbow_flare, grip_ratio, ROM) shown in score report
# Annotation Tab: OHP Phase 3 annotator button → annotation_overhead_press.py
# _config_file_for_exercise(): seated_overhead_press → overhead_press.json for config/flags
```

**Tkinter gotchas (annotation UI):**
- `tk.Scale` has no `.state()` — use `.config(state="disabled"/"normal")`
- `ttk.Radiobutton` with `IntVar(value=0)` unreliable — use `tk.Radiobutton` for binary flags
- `ttk.Label` keeps text when `image=` set — pass `text=""` simultaneously
- `set_display_labels()` called before widget mapped — defer with `master.after(150, fn)`
- `core.exevision.*` imports fail unless `sys.path.insert(0, str(_REPO))` at module level

### Inference API
```bash
pip install -r apps/api/requirements.txt
# PowerShell:
$env:INFERENCE_API_SECRET="your-secret"; .\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
# Test:
curl -X POST http://localhost:8000/infer -H "Authorization: Bearer your-secret" \
     -d '{"video_url":"<url>","job_id":"<uuid>","exercise":"overhead_press"}'
```
**Env vars** (`apps/api/.env`): `INFERENCE_API_SECRET` (required), `SUPABASE_URL`+`SUPABASE_SERVICE_KEY` (production upload), `CORS_ORIGINS` (default `*`).

### Run a Single Stage
```bash
# PowerShell: apostrophe in "Faiq's FYP" breaks inline quoting — use $py/$script variables
# All stages expect CWD = workspace root; accept --exercise (default: squat)
export EXEVISION_MODEL_PATH=models/pose_landmarker_heavy.task
cd pipeline_ui_runs/{run_id}/workspace/
python .../extract_selected_features.py filtered --video-id <id> --exercise overhead_press
python .../classify_views.py --video-id <id> --exercise overhead_press  [--neural]
python .../temporal_segmentation.py --video-id <id> --exercise overhead_press [--debug-phases]
python .../scoring.py <id> --exercise overhead_press
python .../neural_fusion_inference.py --workspace-root . --exercise overhead_press
```

### Batch Pipeline (new)
```bash
python core/exevision/stages/run_pipeline.py \
  --video-dir /path/to/videos --exercise overhead_press \
  --output-dir /output [--mode unfiltered] [--include-neural] [--max-videos N]
```

### Tests
```bash
pytest tests/   # no videos or checkpoints needed — tests config + feedback engine only
```

### OHP Training (squat methodology — current recommended approach)
```powershell
$py = ".\.venv\Scripts\python.exe"
# Step 0: Populate heuristic scores from AQA JSONs (MUST run before training)
& $py core\exevision\analysis\populate_heuristic_scores.py `
    --aqa-root "D:\FitnessAQA\ohp_phase3\overhead_press\aqa_analysis_simple"

# Step 1: Stamp 3-way quality-stratified split (70/15/15, quality bucket)
& $py core\exevision\analysis\stamp_phase3_splits.py --dry-run   # preview
& $py core\exevision\analysis\stamp_phase3_splits.py             # write

# Step 2: Train (4 phases: BiLSTM → ST-GCN → Fusion → optional Joint)
& $py core\exevision\training\ohp\finetune_ohp.py `
    --annotation-dir training_dataset\ohp_phase3_annotations `
    --pretrain-bilstm models\runtime_neural_ohp\bilstm_ohp_phase2.pt `
    --pretrain-stgcn  models\runtime_neural_ohp\stgcn_ohp_phase2.pt `
    --output-dir      models\runtime_neural_ohp\

# Step 3: Evaluate on held-out test split
& $py core\exevision\training\ohp\evaluate_ohp.py `
    --annotation-dir training_dataset\ohp_phase3_annotations `
    --model-dir      models\runtime_neural_ohp\ `
    --output         results\ohp_eval.json
```

**Current eval results (Session 9h, 23-rep held-out test):**
- ✅ quality_mae=8.77 (threshold <12) — beats heuristic baseline (17.79) by 2×
- ✅ smoothness_mae=6.20, control_mae=5.23, elbow_flare_mae=12.06, grip_ratio_mae=8.68, rom_top_mae=11.87, rom_bottom_mae=9.60
- ❌ lockout_auc=0.742 (threshold >0.75) — misses by 0.008 on 23-rep test set (1 rep = threshold)
- `quality_pearson=0.546` vs heuristic `-0.111`; `residual_std=15.82` (model adapts beyond heuristic)

**⚠️ heuristic_score must be populated** in annotation JSONs before training. 108/166 reps had `heuristic_score=0` (AQA not run). Use `populate_heuristic_scores.py` after running Stage 5+8 on source videos.

**Phase 3 5-seed ensemble (legacy, superseded):**
```powershell
# Old approach — kept for reference only. finetune_ohp.py is now recommended.
& $py core\exevision\training\ohp\finetune_phase3.py `
    --annotation-dir training_dataset\ohp_phase3_annotations `
    --pretrain-bilstm models\runtime_neural_ohp\bilstm_ohp_phase2.pt `
    --pretrain-stgcn  models\runtime_neural_ohp\stgcn_ohp_phase2.pt `
    --output-dir      models\runtime_neural_ohp\ --final
```

### Deploy (GCR)
```bash
gcloud builds submit --tag "${GCR_REGION}-docker.pkg.dev/${GCP_PROJECT}/exevision-modelai/exevision-modelai:latest" .
gcloud run deploy exevision-modelai --image=... --memory=4Gi --cpu=2 --timeout=600 \
  --set-env-vars="INFERENCE_API_SECRET=<secret>,SUPABASE_URL=...,SUPABASE_SERVICE_KEY=..."
```

---

## Architecture & Data Flow

```
Video → Stage 2.5 (MediaPipe pose) → Stage 4 (view classify) → Stage 5 (FSM segment)
      → Stage 8 (rule-based AQA)  → Stage 9 (neural fusion, API only)
      → FeedbackEngine → result payload (POST callback or GET /jobs/{id})
```

Stages run as **subprocesses** in `pipeline_ui_runs/{run_id}/workspace/`. Logs at `logs/{stage_key}.log`.

---

## Multi-Exercise Support

- **Config-driven:** `core/exevision/config/exercises/{exercise}.json` — thresholds, issue groups, field mappings
- **Workspace isolation:** all outputs under `workspace_root/{exercise}/`
- **Models** (all OHP models in `models/runtime_neural_ohp/`):
  - Squat: `models/bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt`
  - **OHP finetuned (current, squat methodology):** `bilstm_ohp_finetuned.pt`, `stgcn_ohp_finetuned.pt`, `fusion_ohp_finetuned.pt` — single model, 8 heads, quality_mae=8.77
  - OHP Phase 3 ensemble (legacy): `bilstm/stgcn/fusion_ohp_phase3_seed{42,7,19,3,99}.pt` — superseded by finetuned
  - OHP Phase 2 (fallback): `bilstm/stgcn/fusion_ohp_phase2.pt` — quality + knee_error only
  - OHP Seated: heuristic-only; leg landmarks zeroed (indices 25–32)
- **Inference routing** (Stage 9): exercise handler from `registry.py` uses handler-defined default checkpoint paths. OHP defaults to `models/runtime_neural_ohp/{model}_ohp_finetuned.pt`. No manual routing needed — handler pattern handles it.
- **To add exercise:** create `config/exercises/{name}.json`, optionally add model checkpoints

---

## Stage Reference

**Stage 2.5 — Pose Extraction** (`extract_selected_features.py`)
- Visibility <0.60 any frame → skip video; >10% multi-person → skip
- `--exercise overhead_press` auto-generates seated variant (legs zeroed) alongside main output
- Output: `{exercise}/extracted_features_clean/{quality}/{video_id}.json`

**Stage 4 — View Classification** (`classify_views.py`)
- Writes `info.view` in-place. `--neural` activates `view_classifier_ohp.pt`; falls back to heuristic if confidence < 0.7.
- Landmarks: nose (0), eyes (2,5), ears (7,8), shoulders (11,12), elbows (13,14), wrists (15,16), hips (23,24)

**Signal stack (priority order):**
1. **Torso-lean gate:** `shoulder_width < 0.08` AND `hip_width < 0.08` → `side` (AND-gate prevents false-side on OHP lockout frames where shoulders occlude)
2. **Side gray-zone rescue:** both widths < 0.12 + single arm visible + asym > 0.40 + no face + far arm vis < 0.20 → `side` (face gate + far-arm gate both required)
3. **Arm-asymmetry diagonal forcing:** `arm_asym > 0.35` → diagonal branch even in wide-shoulder zone
4. **Pure-view rescue:** inside diagonal, both arms visible + `arm_asym < 0.10` + definitive depth → re-promote to pure front/back
5. **Back demotion:** wide+back+`arm_asym > 0.20`, not both arms visible → `back_side`
6. **Nose-Z depth:** `nose_z < hip_z` → facing camera; primary front/back discriminator
7. **BlazeFace fallback:** depth=None → `face_detected` flag, then `any_face` (nose/eye vis)

**Video-level overrides** in `get_view_label_with_probs()`:
- `forward_depth_count > backward_depth_count` → override majority back/* → front/* (OHP head-tilt fix)
- `face_detected / valid_frames ≥ 0.25` → override back/* → front/*
- `face_detected / valid_frames < 0.20` → demote front/* → back/* (MediaPipe hallucinates nose-Z forward from back angles)

**⚠️ `front_side`/`back_side` → `diagonal`, `front`/`back` → `straight` display normalization:**
Backend keeps raw labels. `app.py` and `pipeline.py` normalize via `_display_view()` at user-facing boundaries: `front_side`/`back_side` → `"diagonal"`, `front`/`back` → `"straight"`. API consumers/web/desktop UI always see display labels. Annotation UI retains raw. Always pass new display sites through `_display_view()`.

**⚠️ Score discrepancy:** Desktop UI runs Stage 4 before scoring; CLI without Stage 4 uses `view=unknown` (includes all metrics). `front_side` excludes grip_ratio + lockout → scores can differ dramatically for the same video.

**Stage 5 — Temporal Segmentation** (`temporal_segmentation.py`)

| Exercise | FSM Cycle | Primary signal | Knee check |
|----------|-----------|----------------|-----------|
| squat | IDLE→ECC→[ISO]→CONC→IDLE | hip Y descent | required |
| overhead_press / seated | IDLE→CONC→[ISO]→ECC→IDLE | wrist Y ascent | none |

Key params: `DOWNWARD_VELOCITY_THRESHOLD=0.0015`, `UPWARD_VELOCITY_THRESHOLD=-0.0015`, `MOTION_CONFIRM_FRAMES=2`. Use `--debug-phases` for per-rep acceptance logs.

**Stage 8 — Scoring** (`scoring.py`)
- **Squat:** knee_angle, knee_valgus_ratio, forward_lean_deg, squat_depth (view-dependent thresholds)
- **OHP:** `grip_ratio` (excluded for `side`), `rom` (2-component: top extension + elbow-below-shoulder), `lockout` (2D angles for `side` view), `elbow_flare` (front/back only — foreshortening on diagonals)
- Output: `{exercise}/aqa_analysis_simple/{quality}/{video_id}/aqa_simple.json`

**Stage 9 — Neural Fusion** (`neural_fusion_inference.py`)
- `neural_score = clamp(heuristic + tanh(residual)×40, 0, 100)`
- Exercise-agnostic handler pattern: `get_exercise_handler(exercise)` from `registry.py` returns adjacency, fusion, heuristic builders + default checkpoint paths
- To add an exercise: one entry in `_EXERCISE_HANDLER_FACTORIES` + model classes in `_REGISTRY_FACTORIES` in `registry.py`
- `seated_overhead_press`: `suppress_knee=True` in handler, same OHP model
- Stage 9 failure is non-fatal: pipeline continues with heuristic scores

---

## File Structure

| Path | Purpose |
|------|---------|
| `apps/desktop-ui/app.py` | Desktop UI — Inference + Annotation tabs |
| `apps/desktop-ui/annotation_overhead_press.py` | OHP Phase 3 annotator (8 multi-task metrics, auto-discovery) |
| `apps/api/main.py` + `pipeline.py` | FastAPI server + pipeline execution (all 5 stages) |
| `core/exevision/stages/run_pipeline.py` | Batch pipeline runner (stages 2.5→4→5→8) |
| `core/exevision/neural/nn_models.py` | Squat: BiLSTMScorer, STGCNScorer, HeuristicGuidedFusion |
| `core/exevision/neural/ohp/models.py` | OHPBiLSTMScorer, OHPSTGCNScorer (Phase 2/3 multi-task) |
| `core/exevision/neural/ohp/view_classifier.py` | Neural view classifier MLP for OHP |
| `core/exevision/neural/registry.py` | Exercise → model class + inference handler registry (add entry + model file = new exercise) |
| `core/exevision/training/ohp/finetune_ohp.py` | **Current**: 4-phase squat-methodology trainer (BiLSTM→ST-GCN→Fusion→Joint) |
| `core/exevision/training/ohp/evaluate_ohp.py` | **Current**: comprehensive eval — quality Pearson, residual diagnostics, per-bucket MAE |
| `core/exevision/training/ohp/finetune_phase3.py` | Legacy 5-seed ensemble trainer (kept for reference) |
| `core/exevision/training/ohp/evaluate_phase3.py` | Legacy 8-metric eval; `--all-data` flag |
| `core/exevision/training/ohp/data_phase3.py` | Dataset loader; default feat/seg dirs under `ohp_phase3_annotations/` |
| `core/exevision/analysis/stamp_phase3_splits.py` | CLI tool — stamps `fitnessaqa_split=train/val/test`, stratified by quality bucket |
| `core/exevision/analysis/populate_heuristic_scores.py` | CLI tool — back-fills `heuristic_score` from AQA JSONs into annotation files |
| `core/exevision/feedback/engine.py` | Coaching text; tone: 80-100=soft, 70-79=strict, <70=very_strict |
| `core/exevision/config/exercises/{ex}.json` | Per-exercise thresholds + Phase 3 annotation schema |
| `models/` | Squat: `bilstm_finetuned.pt`, `stgcn_finetuned.pt`, `fusion_layer.pt` |
| `models/runtime_neural_ohp/` | OHP finetuned + Phase 2 + Phase 3 seeds + Phase 3 CV folds |
| `training_dataset/ohp_phase3_annotations/` | 165 annotations + `extracted_features/` + `segmented_reps/` (consolidated) |
| `Dockerfile` / `requirements-runtime.txt` | Container; asserts config JSONs at build time |
| `docs/superpowers/plans/` | Implementation plans |
| `apps/web/INTEGRATION_PROMPT.md` | Web app integration spec |

---

## Active Issues

1. **`adapters/legacy-cli/run_stage.py` broken** — maps to deleted `scripts/*.py`. Use `core/exevision/stages/` directly.
2. **`extract_selected_features.py` silent exit-0** — when all videos fail, returns exit-0; downstream stages fail with misleading "Features JSON not found". Run stage directly with `.venv` Python for full traceback.
3. **Error message truncated 100 chars** — `error[:100]` at line 1581; run stage directly for full output.
4. **GCR cold starts (~30–60s)** — `--min-instances=0`; set `=1` to eliminate (~$10/month).
5. **Feedback config JSONs must exist in image** — `squat.json` + `feedback_templates.json` must be in container; Dockerfile asserts at build time.
6. **Callback URL must be public** — `localhost` callback from Cloud Run → "connection refused".
7. **Result payload shape** — Supabase stores `payload.result`; normalize with `result_json?.result ?? result_json` before reading `.feedback`.
8. **Rep feedback alignment** — bind by `rep_id`, not array index; guard by `status === "done"`.
9. **Squat dataset distribution** — test set (26 reps) has no 0–20 coverage; poor-form generalization unvalidated.
10. **No lockfile** — `requirements-runtime.txt` unpinned; reproducibility risk.
11. **Dead code audit** — Several modules confirmed unused: `core/exevision/config/paths.py` + `settings.py` (never imported), `neural/ohp/inference.py` (zero importers), `rules/squat_analysis/` (unreferenced), `legacy_src_reference/` (unreferenced), `training/ohp/{finetune.py,evaluate.py,data.py,finetune_phase3.py,evaluate_phase3.py,tta.py}` (superseded), `training/squat/` + `training/overhead_press/` + `training/seated_overhead_press/` (standalone, never imported). See CHANGELOG 2026-05-14 for full details.

---

## Session Log

Full history in [CHANGELOG.md](CHANGELOG.md).

## graphify

Knowledge graph at `graphify-out/`.
- Before architecture/cross-module questions, read `graphify-out/GRAPH_REPORT.md`
- Use `graphify query/path/explain` for "how does X relate to Y" (traverses EXTRACTED + INFERRED edges)
- After modifying code files, run `graphify update .` (AST-only, no API cost)
