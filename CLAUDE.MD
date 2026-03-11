# CLAUDE.MD — ExeVision AI Project Knowledge Base

_Last updated: 2026-03-07_

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
- `pipeline_ui/`: Tkinter desktop runner for orchestrating scripts and previewing outputs
- `squat_analysis/`: dedicated rule-based squat AQA module and microprograms
- `rule_based_programs_sample/`: legacy/reference diving AQA implementation

## Data/artifacts
- `squat/dataset_videos_all/`: raw dataset videos
- `models/pose_landmarker_heavy.task`: MediaPipe model file
- `pipeline_ui_runs/`: historical run folders with logs, snapshots, isolated workspaces

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

## Scoring / AQA
Two related scoring paths exist:
1. `scripts/8_scoring.py`: simple operational squat AQA/scoring pipeline.
2. `squat_analysis/aqa_metaProgram_squat.py`: richer dedicated module with microprogram error functions and score aggregation.

## Post-analysis tooling
- `analyze_results.py` exists in historical run workspaces and generates aggregate visual analytics.

---

## 6) Progress Assessment (As of 2026-03-07)

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

### Maturity by layer (updated 2026-03-07)
- **Data processing pipeline:** Strongly progressed (end-to-end script stages fully integrated).
- **Stage 4 (view classification):** Enhanced—now robust, intuitive visibility-based logic with ear-visibility tiebreaker for diagonal ambiguity.
- **Stage 5 (temporal segmentation):** Improved—graceful handling of ambiguous/unknown views; defensive startup-phase recovery mechanisms deployed.
- **Modular package (`src`) maturity:** Medium (stages 1–3 clean, later stages not yet ported).
- **Productization/ops maturity:** Medium (good UI, missing lock file; stage 4/5 now more resilient).
- **Research roadmap completion:** Partial (symbolic path active; BiLSTM/fusion remains roadmap).

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

## Partially complete / fragmented
- Two parallel code paths (`src/` and `scripts/`) are not fully unified.
- Stage 4–7 in `src/main.py` are placeholders relative to full script pipeline.
- `analyze_results.py` is not present in root source tree; only found in historical run workspaces.

## Not yet realized (vs stated ambition)
- BiLSTM temporal scorer integrated in executable pipeline.
- Deterministic fusion of symbolic + neural scores in a single production orchestrator.
- Fully reproducible environment specification file(s).

---

## 8) Key Technical Risks / Debt

1. **Pipeline divergence:** `src/` architecture and `scripts/` operational flow may drift.
2. **Missing dependency manifest:** onboarding and reproducibility risk.
3. **File-path coupling:** several scripts assume fixed relative paths.
4. **Artifact dependency:** some expected runtime files (e.g., analysis script) may only exist in old run workspaces.
5. ~~**Quality-gate sensitivity**~~ **[FURTHER REDUCED 2026-03-07]:** Ear-visibility tiebreaker in diagonal zone now robustly distinguishes angled front views from genuine back views; misclassification risk greatly reduced.

---

## 9) Recommended Next Milestones

## Short-term (stabilization)
1. Add `requirements.txt` or `pyproject.toml` with exact package versions.
2. Make `analyze_results.py` a first-class tracked source file in root (not only in run snapshots).
3. Standardize stage interfaces and JSON schemas across `src/` and `scripts/`.
4. Add failure reason logging in segmentation/scoring summaries for faster debugging.

## Mid-term (consolidation)
1. Port scoring/analysis stages into `src/` and extend `src/main.py` to true end-to-end parity.
2. Create regression test fixtures with a few representative videos and expected outputs.
3. Add config profiles (strict/lenient quality gates) for real-world variability.

## Long-term (roadmap alignment)
1. Integrate BiLSTM (or equivalent temporal model) as secondary scorer.
2. Implement explicit symbolic+neural fusion policy with calibration.
3. Expand exercise support beyond squat via microprogram architecture.

---

## 10) Quick File/Module Reference

- Entry pipeline package: `src/main.py`
- Config/dataclasses: `src/config.py`
- Pose extraction: `src/pose_extractor.py`
- View classification: `src/view_classifier.py`
- Temporal segmentation: `src/temporal_segmenter.py`
- UI runner: `pipeline_ui/app.py`
- UI docs: `pipeline_ui/README.md`
- Squat AQA orchestrator: `squat_analysis/aqa_metaProgram_squat.py`
- Squat scoring logic: `squat_analysis/scoring_functions.py`
- Scripted extraction/classification/segmentation/scoring: `scripts/2.5_extract_selected_features.py`, `scripts/4_classify_views.py`, `scripts/5_temporal_segmentation.py`, `scripts/8_scoring.py`
- Project structure overview: `README.md`
- Roadmap snippet: `progress.md`

---

## 11) Bottom-line Status

ExeVision AI is in a **late prototyping / early pre-production** state for squat analysis:
- Core algorithmic pipeline is implemented and repeatedly executed,
- UI-based operational workflow exists and is usable,
- Main remaining work is consolidation, robustness hardening, and alignment of code architecture with full intended roadmap.
