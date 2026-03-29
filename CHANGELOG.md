# ExeVision AI — Development Session Log

> Older sessions archived here from `CLAUDE.md` Appendix A.
> Latest sessions are kept inline in `CLAUDE.md` Appendix A for quick reference.

---

## Session 2026-03-29 — Feedback Reliability Hotfix (Cloud Run)

**Focus:** Resolve production "No feedback data in result_json" behavior and align docs/deployment assumptions with live Cloud Run runtime.

**What was done:**

1. **Backend fallback feedback payload** (`apps/api/pipeline.py`):
   - Added schema-compatible fallback `feedback` object when merged reps exist but narrative config files are missing.
   - Prevents `result.feedback = null` in done-state responses.

2. **Feedback config build guardrails** (`Dockerfile`):
   - Added build-time assertions for:
     - `/app/core/exevision/config/exercises/squat.json`
     - `/app/core/exevision/config/templates/feedback_templates.json`
   - Build now fails early if required feedback config files are absent.

3. **Cloud Build naming alignment** (`cloudbuild.yaml`):
   - Updated substitutions to match live service naming:
     - `_IMAGE_NAME: exevision-modelai`
     - `_SERVICE_NAME: exevision-modelai`

4. **Live runtime diagnosis captured (from Cloud Run logs):**
   - `neural_available=True` with `has_feedback=False` traced to missing feedback config files in container runtime.
   - Callback persistence also failed when callback URL pointed to localhost (`connection refused` from Cloud Run context).

5. **Web integration gotchas documented:**
   - `result_json` shape handling (`payload.result` vs full envelope) must be explicit in frontend parsing.
   - Rep selector should align feedback by `rep_id` and only render narrative on `status === 'done'`.

**Behavioral note:**
- Current API policy is non-fatal for Stage 9 failures (returns fallback-capable responses and surfaces neural state via `result.neural_available`).

---

## Session 2026-03-29 — GCR Deployment + Runtime Bug Fixes

**Focus:** Containerize and deploy the FastAPI inference server to Google Cloud Run. Found and fixed several runtime bugs surfaced during containerization.

**What was done:**

1. **Created `requirements-runtime.txt`** — full pipeline runtime deps beyond the API-only `apps/api/requirements.txt`. Includes `mediapipe`, `opencv-python-headless`, `numpy`, `scipy`, `tqdm`, `matplotlib`. Torch/torchvision kept separate in Dockerfile to force CPU-only wheels.

2. **Created `Dockerfile`** — Python 3.10-slim base. System packages: `libgomp1`, `libglib2.0-0`, `libgl1`, `libegl1`, `libgles2` (OpenGL ES — required by MediaPipe at runtime, discovered during container testing), `ffmpeg`. Models baked into image. Working directory `/app` = repo root (required for subprocess `core.exevision.*` imports). Reads `PORT`/`INFERENCE_API_SECRET`/`CORS_ORIGINS`/`EXEVISION_MODEL_PATH` from env at runtime.

3. **Created `.dockerignore`** — excludes `training_dataset/`, `_hidden_legacy/`, pretrain checkpoints (`bilstm_pretrained.pt`, `stgcn_pretrained.pt`, `stgcn_pretrained_encoder.pt`), Python cache, dev artifacts.

4. **Created `cloudbuild.yaml`** — Google Cloud Build CI/CD config. Build + push to Artifact Registry + deploy to Cloud Run in one `gcloud builds submit` trigger.

5. **Fixed: OpenGL ES missing libs in container** (`Dockerfile`) — MediaPipe requires `libegl1` and `libgles2`. Container would start but crash on first pipeline job without these.

6. **Fixed: json file discovery path assumption** (`core/exevision/stages/extract_selected_features.py`, `scoring.py`) — Stages had hardcoded nested path assumptions (`aqa_analysis_simple/{quality}/`) that worked locally but broke in the containerized CWD. Fixed by walking the full `aqa_analysis_simple/` subtree.

7. **Fixed: neural fusion silent skip** (`apps/api/pipeline.py`) — Added guard in `run_pipeline_sync()`: if neural fusion was in `stages` but `neural_available=False` or no rep carries a `neural_score`, raises `RuntimeError` immediately. Prevents silent heuristic-only output masquerading as a successful neural job.

8. **Added `__main__` block to `apps/api/main.py`** — Respects `PORT` env var (GCR standard) when run directly via `python apps/api/main.py`.

9. **GCR service deployed** — `asia-southeast1`, `--memory=4Gi`, `--cpu=2`, `--timeout=600`, `--concurrency=1`, `--min-instances=0`.

10. **Written web app integration plan** — `docs/superpowers/plans/2026-03-29-webapp-gcr-integration.md`: 10-task plan for the Next.js repo covering `BackendConfig`, `BackendContext`, `BackendToggle` component, health proxy route, env vars, and GCR cold-start UX. Ready for execution in the web app repo.

**New files in this session:**
- `Dockerfile`
- `.dockerignore`
- `requirements-runtime.txt`
- `cloudbuild.yaml`
- `docs/superpowers/plans/2026-03-29-gcr-deployment.md`
- `docs/superpowers/plans/2026-03-29-webapp-gcr-integration.md`

**Modified files:**
- `apps/api/main.py` — PORT env var + `__main__` block
- `apps/api/pipeline.py` — neural fusion silent-skip guard
- `core/exevision/stages/extract_selected_features.py` — json discovery fix
- `core/exevision/stages/scoring.py` — json discovery fix

---

## Session 2026-03-05 — Visibility-Based View Refactor

**Focus:** Improve robustness of view classification (stage 4) and pipeline continuity.

**Changes:**

1. **Stage 4 refactor** (`core/exevision/stages/classify_views.py`):
   - Replaced rotation-angle/facing-signal voting with **visibility-based frame classification**
   - Eliminated hard fallback-to-`side`; returns `unknown` only when zero trackable frames
   - Per-frame voting: face/eye/ear visibility + shoulder width → concrete label or `unknown`

2. **Stage 5 update** (`core/exevision/stages/temporal_segmentation.py`):
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

## Session 2026-03-07 — View Classification Robustness

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
- **Deployed:** In `core/exevision/stages/temporal_segmentation.py`

---

## Session 2026-03-11 (Part 1) — Annotation Tool Launch

**Focus:** Design and implement human annotation tool in `apps/desktop-ui/app.py`.

**What was built:**

1. **`AnnotationToolUI` class** (entire new workflow in Notebook tab):
   - Folder-based video browser with status markers (`✓` complete, `(2/3)` partial)
   - Auto-detection of existing pipeline output (newest-first search in `pipeline_ui_runs/`)
   - Auto-pipeline execution (stages 2.5→4→5→8) in background if no output exists
   - Rep-by-rep video playback using visualization videos (safer than annotated overlays)
   - Bias-blind scoring: heuristic hidden until human score submitted → revealed with Δ
   - F1–F5 keyboard shortcuts for annotation flags
   - Per-video JSON storage at `training_dataset/annotations/videos/{video_id}.json`
   - `index.json` master list updated on every submit
   - Resume support: reloads existing annotation files on video click

2. **`core/exevision/analysis/select_annotation_samples.py`** (strategic rep subset selection):
   - Priority 1: boundary reps (scores 45–55, 65–75, 80–90)
   - Priority 2: view-stratified equal sampling
   - Priority 3: score extremes (top + bottom)
   - Priority 4: random fill
   - Gracefully handles small pools

3. **`core/exevision/analysis/analyze_annotations.py`** (quality self-check):
   - Score distributions, human-heuristic correlation, disagreement stats
   - Training-readiness warnings (correlation >0.95, variance <10, disagreement std <3, count <50)

4. **`main()` modified** to use `ttk.Notebook` with two tabs: Inference (PipelineRunnerUI) and Annotation (AnnotationToolUI)

---

## Session 2026-03-11 (Part 2) — Pipeline Path Fixes

**Focus:** Restore end-to-end support for Unfiltered runs without changing downstream logic.

**Root-cause diagnosis & fix:**

**Issue: Pipeline path mismatch for raw BlazePose outputs**
- **Symptom:** UI Unfiltered runs produced feature JSONs but stages 4/5/8 didn't discover them
- **Cause:** Stage 2.5 writes unfiltered outputs to `raw_unfiltered/`, but stages 4/5/8 hardcoded discovery for quality-tier folders only
- **Fix:** Added `raw_unfiltered` discovery/routing to all three downstream stages
- **Scope:** Path discovery/routing only; no analysis logic changed

---

## Session 2026-03-11 (Part 3) — Depth Metric Bug Fix

**Focus:** Diagnose and fix squat depth false-negative in `core/exevision/stages/scoring.py`.

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

## Session 2026-03-15 — UI Polish & Batch Setup

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

## Session 2026-03-17 — Stage 5 Strict Sequencing + Annotation Controls

**Part 1: Strict Phase Sequencing**
**Focus:** Force deterministic phase ordering while preserving shallow-squat detection.

**What was built:**
1. Enforced strict global adjacency in `core/exevision/stages/temporal_segmentation.py` with sanitizer passes after `detect_phases()` and after `_enforce_minimum_durations()`
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

## Session 2026-03-28 — Web App Integration Live + Extraction Bug Discovery + Disk Optimization

**Focus:** Wire `apps/api/` into the Next.js web app, achieve first end-to-end test, diagnose pipeline failure, and minimize per-run disk usage for hosting readiness.

**What was done:**
1. Fixed `apps/api/main.py` import bug: added `sys.path.insert(0, str(Path(__file__).parent))` so `from pipeline import ...` works when run as `uvicorn apps.api.main:app` from the project root.
2. Updated Quick Start commands in `CLAUDE.md` for Windows/PowerShell syntax.
3. Cross-checked the Next.js agent's integration plan against the API contract; identified 7 gaps.
4. First end-to-end test: web app submitted job → Python server received and processed → callback delivered `result_json` → web app displayed results.
5. **Bug discovered:** `extract_selected_features.py` exits 0 even when all videos fail (line 1624). Pipeline continues silently with empty feature directories; scoring stage produces misleading "Features JSON not found" error. Error message also truncated to 100 chars at line 1581.
6. Documented fusion score display contract: `neural_score` IS the fusion output (`heuristic + tanh(residual) × 40`). Three-judge model (BiLSTM / ST-GCN / Heuristic) documented.
7. **Disk optimization for API runs:** Pass `--no-viz --no-report` to stages 2.5 and 5. Delete input video after extraction. Call `_cleanup_workspace()` post-results. Net: ~6.9 MB → ~25 KB per run.

---

## Session 2026-03-29 — Metric-Agnostic Tier Language Feedback Implementation

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

---

## Session 2026-03-29 — Score-Band Issue Tone Policy (80+ Softening)

**Focus:** Reduce tone mismatch on high-scoring reps by softening issue wording when overall rep score is high.

**What was done:**
1. Updated `core/exevision/feedback/engine.py` with score-band issue tone policy in `_resolve_issue_tone_mode()`:
   - `80-100` -> `soft`
   - `70-79` -> `strict`
   - `<70` -> `very_strict`
2. Updated `_build_rep_feedback()` to pass rep score into issue-cue generation so issue tone is selected per rep.
3. Updated `_group_issue_cues()` behavior by mode:
   - `soft`: avoids harsh cue tiering and prefixes issue cues with "Something to keep in mind:"
   - `strict`: keeps prior severity-based behavior
   - `very_strict`: enforces stronger cue tiering and prefixes with "Priority fix:"
4. Preserved output contract: API payload schema unchanged; only narrative text changes.
5. Verified with runtime smoke check using three synthetic reps (91/74/65), confirming soft/strict/very-strict output transitions.

---

