# ExeVision AI — Development Session Log

> Older sessions archived here from `CLAUDE.md` Appendix A.
> Latest sessions are kept inline in `CLAUDE.md` Appendix A for quick reference.

## 2026-05-06 — Session 6: OHP Scoring Redesign (ROM, View-Aware Metrics, Side View Fix)

**Focus:** Complete redesign of OHP ROM scoring; fix score discrepancy between desktop UI and CLI; fix side-view angle computation; relax thresholds based on real rep data.

**scoring.py — `_score_overhead_press` / OHP helpers:**
- `_ohp_rom()` now returns `(min_angle, max_angle)` tuple instead of single float
- Added `_ohp_elbow_angle(frame, sh, el, wr, use_2d)` helper — uses `_angle_2d` for side views, `_angle_3d` otherwise; side view Z-depth from MediaPipe is unreliable when arm extends toward/away from camera
- `_ohp_lockout()` and `_ohp_rom()` both accept `use_2d=True` flag, set automatically for `side` view in `_score_overhead_press`
- Added `_ohp_elbow_below_shoulder(rep_frames, min_angle)` — checks raw Y coords first (elbow_y > shoulder_y across all rep frames), falls back to `min_angle <= 110°`
- **ROM redesigned as 2-component score:** 50% top extension (`score_metric_linear` on `max_elbow_rom` vs `rom_top` thresholds) + 50% elbow-below-shoulder (binary)
- **Grip ratio:** excluded for `side` view only (shoulder width ≈ 0 makes measurement unreliable)
- **Elbow flare:** excluded from ALL diagonal views (`side`, `front_side`, `back_side`) — foreshortening effect
- Grip ratio thresholds relaxed: ideal [0.40→0.20, 0.95→1.20], tolerance 0.60→0.90

**overhead_press.json:**
- Replaced `min_elbow_angle` metric with `rom_top` (good=157°, bad=110°) and `rom_bottom` (note only — binary check in code)
- `max_elbow_angle`: good 158°→**150°**, bad 135° (unchanged) — recalibrated to real rep data
- Removed `diagonal_offset` from `elbow_flare` (elbow_flare now fully excluded on diagonal views)
- Updated `field_mapping`: `min_elbow_angle→rom_range`, `max_elbow_angle→lockout`

**Root cause diagnosed — Desktop UI vs CLI score discrepancy:**
- Desktop UI runs Stage 4 (classify_views) → view becomes `front_side` → grip_ratio + lockout excluded → only 2 metrics scored
- CLI without Stage 4 → `view=unknown` → all 4 metrics included → completely different weighted average
- Fix: always run `classify_views.py` before `scoring.py`

---

## 2026-05-06 — Session 5: OHP View Classification Fix + View-Aware Scoring Calibration

**Focus:** Fix incorrect view classification for OHP videos (head-tilt during press fools depth signal). Recalibrate OHP scoring thresholds to match real athlete data. Add view-aware scoring so side/diagonal views are not penalised for metrics that can't be measured from those angles.

**classify_views.py:**
- Added video-level `has_any_forward_depth` flag in `get_view_label_with_probs`: if any frame shows nose definitively in front of hips, a majority back/back_side vote is overridden to front/front_side — handles OHP head tilt at lockout corrupting the per-frame depth signal
- Reverted frame-level logic to simple depth-first; `face_detected`/`any_face` only used as fallback when depth is ambiguous (None)
- Key finding: MediaPipe visibility scores are all 1.00 regardless of facing direction — useless as a front/back discriminator; BlazeFace (`face_detected`) has low recall for OHP videos

**scoring.py — `_score_overhead_press`:**
- Added `view` parameter (default `"front"`)
- `side`, `front_side`, `back_side` views: grip_ratio and lockout **excluded from weighted average** (arm/wrist axis collapses into camera line of sight — measurements unreliable)
- elbow_flare for diagonal views: shifted +`diagonal_offset` degrees (from config, default 15°) before scoring to compensate for apparent foreshortening
- Call site updated to pass `view=view`

**overhead_press.json — threshold recalibration (based on 4 verified good reps):**
- `grip_ratio`: ideal 0.05–0.25 → **0.40–0.95**; perfect 0.15 → 0.65; tolerance 0.30 → 0.60
- `min_elbow_angle`: full_rom 75° → **85°**; partial 90° → **115°**
- `max_elbow_angle`: good 165° → **158°**; bad 145° → **135°**
- `elbow_flare`: ideal 30–60° → **90–150°** (measurement is during concentric press, not static bottom position); bad_low 20° → 60°; bad_high 70° → 175°; added `diagonal_offset: 15.0`

---

## 2026-05-06 — Session 4: Desktop UI OHP/Seated OHP Integration + Stage Pipeline Fixes

**Focus:** Full `overhead_press` and `seated_overhead_press` support in the desktop UI (inference + annotation tabs). Stage pipeline I/O consistency fixes for Stages 4 and 5.

**Stage fixes:**
- `classify_views.py`: `run_classification()` now accepts `exercise` param and derives dirs via `_build_features_dirs(exercise)` internally — no longer depends on module-level global reassignment from `__main__`. Fixed stray `print('='*70)` at module level (was executing on every import).
- `temporal_segmentation.py`: Added `CURRENT_EXERCISE = "squat"` module-level default — was only assigned inside `if __name__ == "__main__":`, causing latent `NameError` if `create_segmentation_visualization()` was ever called from an import context.

**Desktop UI (`apps/desktop-ui/app.py`):**
- Inference dropdown: added `seated_overhead_press`; `analyze_results` stage now squat-only in `_build_stages`; `dataset_var` clears for non-squat exercises in `_on_exercise_changed`
- All inference output-discovery methods parameterised with `self.exercise_var.get()`: `_find_score_json`, `_find_analysis_summary_json`, `_find_neural_json`, `_find_annotated_videos`, `_set_preview_outputs`, `_find_original_video_for_overlay`, `_load_selected_preview`, `_update_metadata_display` (9 methods, 12 path fixes)
- Scoring display: `_metric_specs()` exercise-aware (OHP returns grip_ratio/rom/lockout/elbow_flare); `_build_rep_diagnostics()` uses dynamic metric keys; `_build_metric_diagnostic()` guards `get_view_thresholds` with `.get()` to avoid KeyError on OHP metrics
- Annotation tab: independent `self._annotation_exercise_var`; exercise combobox inside folder frame; `_on_annotation_exercise_changed()` rebuilds flags and clears folder state
- Annotation flags: replaced hardcoded squat list with `_build_annotation_flag_defs()` + `_rebuild_annotation_flags()` — reads from `{exercise}.json` config via `_config_file_for_exercise()` helper; F1-F7 hotkeys rebind on exercise change
- Annotation paths: all `_build_annotation_payload_from_run`, `_find_existing_pipeline_output`, `_find_overlay_from_pipeline_outputs` use `_annotation_exercise_var`
- Annotation pipeline threads (`_pipeline_thread`, `_batch_pipeline_thread`): workspace dir now uses annotation exercise; stages built via `_build_stages(_ann_exercise)` not global `STAGES`
- `_run_pipeline_stage`: now passes `--exercise` flag to all stage subprocesses
- Processed-video scan: scans all three exercise namespaces so green/red listbox colouring works across exercises

**New helpers in app.py:**
- `CONFIG_EXERCISES_DIR` — `Path` constant pointing to `core/exevision/config/exercises/`
- `_config_file_for_exercise(exercise)` — routes `seated_overhead_press → overhead_press` for config/flag loading

**Config:**
- `squat.json`: added `annotation_flags` and `annotation_metrics` (mirrors `overhead_press.json` structure)

---

## 2026-05-06 — Session 3

Removed invalid `bar_path_deviation` metric from Overhead Press scoring and rebalanced weights.

- Rationale: bar path cannot be reliably inferred from pose landmarks (wrist != bar).
- Code: removed `_ohp_bar_path_deviation()` and `_score_ohp_bar_path()` in `core/exevision/stages/scoring.py`.
- Config: updated `core/exevision/config/exercises/overhead_press.json` (4 metrics, 0.25 weights).
- Tests: removed bar_path unit tests from `tests/test_ohp_scoring.py`.
- Docs: updated `CLAUDE.md` to reflect removal and rebalancing.

Next planned work:
1. Add view classifier and integrate into pipeline (next, in-progress).
2. Recalibrate OHP scoring thresholds to match elite lifter data.
3. Run full test suite and re-run AQA on sample dataset.

---

## Session 2026-05-06 (2) — Stage 8 OHP Scoring: 5 Biomechanical Metrics

**Focus:** Implement view-independent overhead press scoring in `scoring.py` with 5 biomechanical metrics computed in a body-local 3D coordinate frame. Support both `overhead_press` and `seated_overhead_press` variants with no shared logic with squat scoring.

**Architecture:** Complete separation from squat. All OHP metric functions are standalone. Body-local coordinate frame built per-frame from shoulder/hip landmarks, making all metrics camera-agnostic. Exercise string flows as explicit parameter through every function into output payload. Config thresholds live in `overhead_press.json` — no magic numbers in code.

**What was done:**

**1. Updated `overhead_press.json` (config):**
- Added `metric_weights`: all 5 metrics at 0.20 (equal importance)
- Replaced stub `metrics` with complete thresholds:
  - `grip_ratio`: ideal 5-25% wider than shoulders (perfect: 0.15, tolerance: 0.30)
  - `bar_path_deviation`: horizontal XZ drift normalized by shoulder width (good ≤0.05, bad ≥0.25)
  - `min_elbow_angle`: ROM — full ≤75°, partial 75-90°, insufficient ≥90°
  - `max_elbow_angle`: lockout extension ≥165° good, ≤145° bad (sustained ≥0.5s)
  - `elbow_flare`: shoulder abduction during concentric 30-60° ideal, asymmetry penalty >15°
- Updated `field_mapping`, `annotation_flags`, `annotation_metrics`

**2. Added 3D geometry helpers to `scoring.py` (lines ~240-380):**
- `_xyz(frame, idx, min_conf=0.4)` → Returns (x,y,z) as numpy array
- `_angle_3d(a, b, c)` → True 3D angle in degrees at vertex b (not 2D projected)
- `_build_body_frame(frame)` → Orthonormal frame from shoulder/hip with: `v_right`, `v_up`, `v_forward`, `mid_shoulder`, `mid_hip`
- `_to_body_local(p, bf)` → Projects world point to body-local axes (x_right, y_up, z_forward)

**3. Implemented 5 OHP metric functions (lines ~385-880):**
- `_ohp_grip_width(rep_frames)` → Grip width ratio (measured at bottom of rep, first 5 frames)
- `_ohp_bar_path_deviation(rep_frames)` → Horizontal drift from start to finish normalized by shoulder width
- `_ohp_rom(rep_frames)` → Minimum 3D elbow angle across rep (flexion depth)
- `_ohp_lockout(rep_frames, fps)` → Max elbow angle sustained ≥0.5s (or peak if no sustained window)
- `_ohp_elbow_flare(rep_frames, rep_phases)` → Mean shoulder abduction during concentric → Returns (left, right, both)

**4. Added OHP scoring dispatcher (lines ~880-1000):**
- `_score_ohp_grip()` → Scores against ideal range with tolerance
- `_score_ohp_bar_path()` → Linear score (lower is better)
- `_score_ohp_rom()` → Threshold-based score (full/partial/no credit)
- `_score_ohp_lockout()` → Linear score (higher is better)
- `_score_ohp_flare()` → Scores abduction with asymmetry penalty (deduct 2% per degree >15° asymmetry)
- `_score_overhead_press(rep_frames, rep_phases, fps, exercise, config)` → Main dispatcher
  - Returns: `overall_score, metric_scores, raw_metrics, exercise, weights_used`
  - Identical scoring logic for both `overhead_press` and `seated_overhead_press`

**5. Wired OHP dispatch into `process_single_video()` (function signature + rep loop):**
- Added `exercise: str = "squat"` parameter to function signature
- Load `overhead_press.json` config if exercise is OHP variant
- Extract `frame_phases_all` from segmented JSON
- **Rep processing:** If OHP + config available, call `_score_overhead_press()`; else use squat logic
- Updated both `main()` calls (batch and single video) to pass `exercise=args.exercise`

**6. Created comprehensive test suite (`tests/test_ohp_scoring.py`):**
- 24+ unit tests covering geometry helpers, metrics, and scorer
- No pytest dependency — tests can be run standalone; uses numpy assertions
- Synthetic frame builders for controlled geometry testing

**Tested:**
- Single video (80690_2, overhead_press): ✅ 1 rep detected, score 19.1/100
- All functions import successfully ✅
- Config parses with all 5 metric keys ✅

**Files Modified:**
1. `core/exevision/config/exercises/overhead_press.json` — Complete metric config
2. `core/exevision/stages/scoring.py` — All geometry, metrics, scorer, dispatch (853 new lines)
3. `tests/test_ohp_scoring.py` — New test suite (NEW FILE)

**Backward Compatibility:** ✅ Squat scoring untouched; OHP only activates with `--exercise overhead_press`

**Status:** Code implementation COMPLETE. Ready for batch Stage 8 scoring on 2,804 OHP videos in FitnessAQA dataset.

---

## Session 2026-05-06 — Stage 5 Custom Video Directory Support (`--video-dir`)

**Focus:** Add `--video-dir` parameter to temporal_segmentation.py to support FitnessAQA dataset structure where videos are stored outside the workspace.

**Problem:** The FitnessAQA ohp_phase1 workspace doesn't have a `dataset_videos_all/` folder. Videos are stored externally at:
- `D:\FitnessAQA\Overhead Press\Unlabeled_Dataset-OHP\Unlabeled_Dataset\videos\videos\`
- `D:\FitnessAQA\Overhead Press\Labeled_Dataset-OHP\Labeled_Dataset\videos\videos\`

Without `--video-dir` support, visualization generation fails ("video not found").

**What was done:**

**1. Updated `find_video_file()` function:**
- Added optional `video_dir` parameter
- Updated priority order:
  1. Annotated poses from Stage 2.5 (if quality specified)
  2. Custom video directory (if `--video-dir` provided) — searches recursively
  3. Default dataset directory (`{exercise}/dataset_videos_all/`)
- Each priority level searches recursively with `os.walk()`

**2. Updated visualization pipeline:**
- `create_segmentation_visualization()` now accepts `video_dir` parameter
- Passes `video_dir` to `find_video_file()` call

**3. Updated `run_segmentation()` function:**
- Added `video_dir=None` parameter
- Passes `video_dir` to `create_segmentation_visualization()` calls
- Updated console output to show custom video directory if provided

**4. Updated CLI parser:**
- Added `--video-dir` argument: "Path to custom video directory (searches recursively)"

**5. Updated main entry point:**
- Passes `args.video_dir` to `run_segmentation()` call

**Tested Command:**
```powershell
cd D:\FitnessAQA\ohp_phase1\workspace
& \"C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\.venv\Scripts\Activate.ps1\"
python \"C:\Users\faiqf\Documents\Faiq's FYP\exevision_modelAI\core\exevision\stages\temporal_segmentation.py\" `
  --video-id 80830_5 `
  --exercise seated_overhead_press `
  --video-dir \"D:\FitnessAQA\Overhead Press\Unlabeled_Dataset-OHP\Unlabeled_Dataset\videos\videos\" `
  --debug-phases
```

**Results:**
- Segmentation: ✅ 1 rep detected (4 phase transitions)
- Debug log: ✅ saved to `seated_overhead_press/debug_phases/raw_unfiltered/80830_5_phases.json`
- Visualization: ⚠️ still needs video file lookup verification

**Files Modified:**
1. `core/exevision/stages/temporal_segmentation.py` — Added --video-dir support (4 changes)
2. `CLAUDE.md` — Updated Stage 5 section with --video-dir examples and video discovery priority
3. `CHANGELOG.md` — This entry

**Backward Compatibility:** ✅ All changes are additive; existing scripts work without `--video-dir`

---

## Session 2026-05-05 (2) — Stage 5 OHP-Specific FSM (Remove Squat Logic)

**Focus:** Remove all squat-specific logic (knee-angle gates) from the overhead press / seated overhead press path in `temporal_segmentation.py`. Implement a clean OHP FSM with the correct rep cycle: `CONCENTRIC → [ISOMETRIC] → ECCENTRIC`.

**Root cause of 0-rep detection (diagnosed this session):**
The FSM's primary ECCENTRIC entry rule required `and bent` (knee angle < 133°). For standing OHP, knees are straight throughout — `bent = False` always — so the FSM could never leave IDLE. Even with a valid wrist displacement signal, all 71 frames were labeled `idle`.

**What was done:**

**Architecture: Full separation, zero shared squat logic in OHP path**
- Added `OHP_VALID_TRANSITIONS` module constant: `IDLE→CONC→[ISO]→ECC→IDLE` (inverse of squat cycle)
- Added `_is_ohp` property on `SquatStateMachine` (checks exercise string on `self.analyzer`)
- Added `_get_valid_transitions()` method — returns `OHP_VALID_TRANSITIONS` or `VALID_TRANSITIONS` by exercise
- Updated `_is_transition_allowed()` to call `_get_valid_transitions()` (sanitizer and transition logging now exercise-aware)
- Updated `_can_return_to_idle()` — for OHP, wrist position alone is sufficient; knee extension check skipped
- `detect_phases()` now routes: `_detect_phases_ohp()` or `_detect_phases_squat()` (original renamed)

**`_detect_phases_ohp()` — new OHP FSM:**
- Positive velocity (wrists rising) → `CONCENTRIC`
- Negative velocity (wrists falling) from CONCENTRIC/ISOMETRIC → `ECCENTRIC`
- Still at top during CONCENTRIC → `ISOMETRIC` (hold overhead, ≥1 s)
- ECCENTRIC + wrists back at start → `IDLE`
- No `bent`, `_knee_bending()`, or `_knee_extended()` calls anywhere

**Rep detection routing:**
- `_detect_repetitions()` routes to `_detect_repetitions_ohp()` or `_detect_repetitions_squat()` (original renamed)
- `_detect_repetitions_ohp()`: looks for CONCENTRIC → ECCENTRIC cycle; uses `min_height=0.02` (lower than squat's 0.05); stores peak wrist displacement in `squat_depth_normalized`, top-of-press frame in `bottom_frame`
- `_detect_repetitions_phase_only()` routes to `_detect_repetitions_phase_only_ohp()` or `_detect_repetitions_phase_only_squat()` (original renamed)
- `_detect_repetitions_phase_only_ohp()`: CONC→ISO→ECC phase-only fallback counting; handles fast consecutive reps

**Verification:**
- `80830_5` (71 frames): was 0 reps → now **1 rep** detected; phase sequence `concentric → isometric → eccentric` ✅
- `68959_4` (137 frames): was 0 reps → now **1 rep** detected; phase sequence `concentric → isometric → eccentric` ✅
- Syntax validation: passed ✅
- Squat backward compat: untouched — `_detect_phases_squat()` is the renamed original, no logic changed

**Files Modified:**
1. `core/exevision/stages/temporal_segmentation.py` — OHP FSM + routing
2. `CLAUDE.md` — Stage 5 section updated with FSM cycle table and architecture notes

**Key Design Decisions:**
1. Completely separate OHP methods — no if/else branches inside squat code; squat is untouched
2. `_is_ohp` as a property on `SquatStateMachine` — exercise context available everywhere without passing extra params
3. `min_height=0.02` for OHP rep validation — wrist displacement range is smaller than hip displacement in normalized coords
4. Renaming originals to `_*_squat` variants rather than removing — preserves git blame and makes routing explicit

---

## Session 2026-05-05 — Stage 5 Temporal Segmentation Exercise Parameterization

**Focus:** Refactor Stage 5 (temporal_segmentation.py) to support overhead press with exercise-specific control signals and debug verification mode.

**What was done:**

**Phase 1: Control Signal Refactoring (Non-Breaking)**
- **Added wrist landmark indices:** `L_WRIST=15`, `R_WRIST=16`, `L_ELBOW=13`, `R_ELBOW=14`
- **Created `_hip_y_sequence(frames)`:** Extracts normalized hip Y-displacement for squat (existing logic extracted)
- **Created `_wrist_y_sequence_ohp(frames)`:** Inverted wrist Y-displacement for OHP
  - **Critical insight:** Rising wrist (smaller Y) = INCREASING signal (to match FSM semantics)
  - Formula: `displacement = standing_wrist_y - wrist_y` (inverted from hip logic)
- **Created `_get_control_signal(frames, exercise)`:** Dispatcher that returns exercise-specific signal
- **Updated `BiomechanicalAnalyzer.__init__()`:** Accepts `exercise: str = "squat"` parameter and stores it
- **Updated `BiomechanicalAnalyzer.compute_normalized_hip_displacement()`:**
  - Calls `_get_control_signal()` to compute generic signal
  - Populates both `self.control_signal` (generic) and `self.normalized_hip_displacement` (backward compat for squat)
  - No changes to existing squat behavior

**Phase 2: Exercise-Aware Thresholds**
- **Created `_get_thresholds(exercise)`:** Returns exercise-specific threshold dictionary
  - Supports: `"squat"`, `"overhead_press"`, `"standing_overhead_press"`, `"seated_overhead_press"`
  - Currently using identical squat threshold values for OHP (tuning deferred to post-segmentation phase)
  - 20 parameters per exercise: `MIN_REP_FRAMES`, `MIN_DEPTH_RATIO`, velocity thresholds, etc.
- **Future-proof design:** Thresholds can be tuned independently per exercise without code changes

**Phase 3: Debug/Verification Output**
- **Created `_debug_enabled()`:** Dynamically reads `DEBUG_PHASES` environment variable
- **Created `_save_phase_debug_log(video_id, exercise, quality, debug_log)`:** Saves JSON to `{exercise}/debug_phases/{quality}/{video_id}_phases.json`
- **Enhanced `TemporalSegmenter.segment()`:** Generates debug_log with per-rep metadata when enabled:
  - `rep_id`, `start_frame`, `end_frame`, `duration_frames`
  - `control_signal_max`, `phase_sequence`, `accepted`, `reasons`
  - Example: `"reasons": ["Phase sequence: concentric -> eccentric", "Max displacement: 0.4200", "Duration: 76 frames"]`
- **Updated `run_segmentation()`:** Saves debug logs to filesystem when present
- **Added `--debug-phases` CLI flag:** Sets `DEBUG_PHASES=1` environment variable; enables debug output
- **Zero performance impact:** Debug output only generated when flag enabled

**Phase 4: End-to-End Wiring**
- **Updated `TemporalSegmenter.__init__()`:** Accepts and passes `exercise` parameter to `BiomechanicalAnalyzer`
- **Updated `process_video(json_path, exercise="squat")`:** Threads exercise parameter to `TemporalSegmenter`
- **Updated `run_segmentation(exercise="squat")`:** Already accepted; now passes to `process_video()`
- **Updated CLI parser:** Added `--debug-phases` flag with help text
- **Backward compatibility:** All defaults remain "squat"; no changes to existing code

**Key Design Decisions:**
1. **Invert wrist signal:** OHP's rising wrist = increasing control signal (matches FSM expectations)
2. **Separate branches, not nested:** Clearer control flow; less chance of squat-side bugs
3. **Dynamic threshold lookup:** `_get_thresholds()` called at runtime (not cached), allows future config-driven updates
4. **Optional debug output:** Environment variable gate; zero overhead when disabled
5. **Backward compatible:** All changes default to squat; existing code unaffected

**Verification (Phase 4):**
- ✅ Created comprehensive test suite: 5 test cases covering control signals, thresholds, analyzer, debug mode, segmenter parameter threading
- ✅ All tests passed:
  - Control signal extraction works for squat and OHP
  - Exercise-specific thresholds returned correctly
  - BiomechanicalAnalyzer accepts exercise parameter
  - Debug mode toggleable via environment variable
  - TemporalSegmenter accepts exercise parameter
- ✅ Syntax validation: `py_compile` check passed

**Files Modified:**
1. `core/exevision/stages/temporal_segmentation.py` — Added 3 control signal helpers, thresholds dispatcher, debug utilities, exercise parameter threading

**Files Created:**
1. `test_temporal_segmentation_refactor.py` — Verification test suite (5 test cases, all passing)

**Testing Status:**
- ✅ Unit tests: 5/5 passing
- ✅ Syntax validation: passed
- ✅ Control signals: verified different between squat and OHP
- ✅ Parameter threading: complete (CLI → run_segmentation → TemporalSegmenter)
- ⏳ Regression test: squat behavior should be identical (pending Phase 4 validation with sample data)
- ⏳ OHP functional test: pending sample overhead press video with --debug-phases flag

**Next Steps:**
1. **Regression test (Squat):**
   ```bash
   python core/exevision/stages/temporal_segmentation.py --exercise squat --video-id <test_id>
   # Verify output structure is identical to before refactoring
   ```

2. **OHP functional test (if sample data available):**
   ```bash
   python core/exevision/stages/temporal_segmentation.py \
     --exercise overhead_press --video-id <ohp_sample> --debug-phases
   ```

3. **Debug output inspection:**
   ```bash
   cat overhead_press/debug_phases/raw_unfiltered/<video_id>_phases.json | python -m json.tool
   # Verify rep counts and phase sequences match visual inspection
   ```

4. **Stage 8 Scoring:** Implement OHP-specific scoring metrics (separate task; references this stage's output)

---

## Session 2026-05-04 — Overhead Press Phase 1 Dataset Prep (FitnessAQA Integration)

**Focus:** Prepare extraction pipeline for large-scale FitnessAQA dataset processing (~7,750 total videos); implement seated overhead press variant for knee-invariant model training.

**What was done:**

**Phase A: extract_selected_features.py Major Refactor**
- **Added `--video-dir` parameter:** Allows overriding default dataset root (e.g., `/path/to/FitnessAQA/Overhead Press/` on non-NTFS drives)
- **Added `--max-videos N` parameter:** Caps processing to N unprocessed videos; moves cap AFTER already-processed filter (ensures "N unprocessed" semantics)
- **Added `--include-poor` flag:** Saves videos with poor-but-detectable landmarks to `raw_unfiltered/` (default: skip); enables downstream confidence filtering rather than upfront rejection
- **Implemented quality thresholds:**
  - `MIN_FRAME_VISIBILITY = 0.60`: Skip entire video if any frame's overall visibility below threshold
  - `MAX_MULTI_PERSON_RATIO = 10%`: Skip if >10% of frames detect 2+ people (training expects single-subject)
- **Changed MediaPipe multi-person detection:** `num_poses=1` → `num_poses=2` to track simultaneous persons; added frame counter to enforce single-subject policy
- **Created `_zero_leg_landmarks(frame_data)` helper:** Nullifies leg landmark indices 25–32 while preserving array shape (for seated variant)
- **Implemented seated overhead press variant:**
  - Main output: `overhead_press/extracted_features_clean/{quality}/{video_id}.json` (full landmarks)
  - Seated output: `seated_overhead_press/extracted_features_clean/{quality}/{video_id}.json` (legs zeroed)
  - Visualizations mirror to `{exercise}/visualized_poses_clean/{quality}/` and `seated_{exercise}/visualized_poses_clean/{quality}/`
  - Gated to OHP only: `if args.exercise == "overhead_press"`

**Phase B: Multiprocessing Global State Fix**
- **Root cause:** Module-level globals reassigned at runtime weren't reaching worker processes (Python reimports module with original defaults in each process)
- **Solution:** Created `_init_worker()` initializer function with signature accepting 11 parameters:
  - `output_root`, `viz_root`, `excluded_ids_set`, `already_processed_ids_set`, `max_videos`, `include_poor`, `seated_output_root`, `seated_viz_root`, `exercise`, `landmark_confidence`, `key_joint_confidence`
- **Implementation:**
  - Added `global` declarations in initializer for all mutable globals
  - Updated `multiprocessing.Pool()` creation to pass `initializer=_init_worker, initargs=(all_params,)`
  - Worker processes now receive proper state and produce correct output folder routing
- **Validation:** Tested with `--exercise overhead_press` and `--max-videos 10`; confirmed dual outputs appear in correct folders

**Phase C: Path Resolution & Already-Processed Check**
- **Fixed `_already_processed_json_exists()`:** Now checks BOTH main and seated folders exist before marking video fully processed (prevents skipping when only main is done)
- **Fixed path display:** Changed to `os.path.abspath()` for absolute workspace paths (clarity improvement)
- **Fixed QUALITY_FOLDERS assignment:** Now uses dynamic `OUTPUT_ROOT` instead of hardcoded path

**Phase D: Documentation**
- **Created `OVERHEAD_PRESS_PLAN.md`:** Comprehensive 3-phase training roadmap with clear modularity contracts:
  - Phase 1: Self-supervised pre-training on 5,490 unlabeled videos
  - Phase 2: Supervised fine-tuning on 2,260 labeled FitnessAQA annotations
  - Phase 3: Human annotation calibration + integration testing
  - Includes temporal segmentation, scoring metrics, dataset preparation, and model training specs
- **Updated CLAUDE.md:** Added Stage 2.5 CLI parameter docs, seated variant explanation, FitnessAQA context, quality thresholds

**Key Behavioral Changes:**
- ✅ `--max-videos N` now means "process N unprocessed videos" (previously counted already-processed)
- ✅ Seated variant auto-generated for all OHP runs (maintains file shape for downstream compatibility)
- ✅ Quality gates enforce min visibility 0.50 and single-subject policy (multi-person skipped)
- ✅ `--include-poor` enables marginal-quality video inclusion (confidence filtering deferred to downstream stages)
- ✅ Multiprocessing globals now reach workers correctly

**Files Modified:**
1. `core/exevision/stages/extract_selected_features.py` — 11 global parameters + initializer + seated variant + quality gates
2. `OVERHEAD_PRESS_PLAN.md` — new file; 3-phase roadmap
3. `CLAUDE.md` — Stage 2.5 params, seated variant docs, FitnessAQA context, quality thresholds

**Testing Status:**
- ✅ Local 10-video test: both `overhead_press/` and `seated_overhead_press/` folders created with correct JSONs
- ✅ Quality thresholds: videos with overall visibility <0.60 correctly skipped
- ✅ Multi-person detection: frame counter working; test videos processed without 2+ person frames
- ✅ Path resolution: absolute paths displayed correctly in logs
- ✅ Multiprocessing: workers received correct global state and output to intended folders

**Ready for Phase 1 Scaling:**
- Infrastructure complete for large-scale FitnessAQA processing
- **Phase 1 Batch Command** (PowerShell):
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
- Seated variant automatically captured (no extra configuration)
- Per-exercise model paths ready in pipeline (falls back to shared models during transition)
- Start with `--max-videos 10` for validation, then scale to 500, then full 5490

**Next Steps:**
1. Run Phase 1 batch processing on full 5,490 unlabeled FitnessAQA videos
2. Implement OHP-specific temporal segmentation (Section 2.1 of OVERHEAD_PRESS_PLAN.md) — wrist-based control signal
3. Implement OHP-specific scoring metrics (Section 2.2) — shoulder elevation, elbow extension, bar path deviation
4. Create `prepare_ohp_dataset.py` for Phase 2 labeled annotation conversion (Section 3.3)

---

## Session 2026-05-04 (Part 2) — Rejection Registry + Quality Threshold Refinement

**Focus:** Avoid re-processing rejected videos across sessions; lower min visibility threshold for broader FitnessAQA coverage.

**What was done:**

**Feature 1: Rejection Registry (`.skipped_videos_registry.json`)**
- **Problem:** Previously, stopping/restarting the extraction meant re-processing and re-rejecting the same videos, wasting time
- **Solution:** Persistent JSON registry tracking all rejected videos with reason + timestamp
- **Implementation:**
  - `_load_skipped_videos_registry()` — loads registry at startup
  - `_save_skipped_video(vid_id, reason, mode)` — records rejection immediately
  - Updated `_already_processed_json_exists()` to check registry first; returns True if video was previously rejected
  - Registry file: `.skipped_videos_registry.json` (sibling to output folders, survives interrupts)
- **Behavior:**
  - First run: processes all 5490 videos; rejects poor-quality, multi-person, low-visibility videos; saves to registry
  - Second run: skips all registered rejects immediately (no re-processing); only processes new videos
  - Summary now shows: "Previously rejected (registry): N"
- **Files modified:** `extract_selected_features.py` (3 new functions + 6 registry save calls)

**Feature 2: Min Visibility Threshold Update**
- Changed from 0.60 → 0.50 to be less restrictive on FitnessAQA unlabeled dataset
- Videos where any frame drops below 0.50 overall landmark visibility now skipped (previously 0.60)
- Rationale: Unlabeled dataset is noisier than annotated squat videos; 0.50 allows more marginal videos through for pre-training
- Updated documentation in CLAUDE.md + visualization chart labels

**Testing Status:**
- ✅ Registry loads on startup
- ✅ Rejections recorded with timestamp + reason
- ✅ Re-runs skip registered videos immediately
- ✅ Summary shows registry count
- ✅ Quality threshold working correctly at 0.50

**Impact:**
- **Iteration speed:** 10-run cycle now skips 1000+ previously-rejected videos immediately (no re-work)
- **Data coverage:** Lower visibility threshold captures more marginal videos for pre-training
- **Robustness:** Registry survives Ctrl+C, process crashes, and session interruptions

---

## Session 2026-04-09 — Multi-Exercise Refactoring Complete (Pragmatic Approach)

**Focus:** Execute pragmatic multi-exercise support implementation; remove critical hardcodes; wire Desktop UI exercise selector.

**What was done:**

**Phase A1 — Remove temporal_segmentation Hardcode**
- Removed `CURRENT_EXERCISE = "squat"` hardcoded constant from `core/exevision/stages/temporal_segmentation.py`
- Script now fully parameterized; accepts `--exercise` argument; no hardcoded exercise reference remains
- `_build_temporal_paths(exercise)` function builds all paths dynamically

**Phase A2 — Update Dockerfile Validation**
- Updated `Dockerfile` lines 46-54 to validate both `squat.json` and `overhead_press.json`
- Changed from hardcoded squat check → dynamic discovery:
  ```dockerfile
  RUN python -c "from pathlib import Path; \
      configs = {c.stem for c in Path('/app/core/exevision/config/exercises').glob('*.json')}; \
      required = {'squat', 'overhead_press'}; \
      missing = required - configs; \
      assert not missing, f'Missing exercise configs: {missing}'"
  ```

**Phase A3/A4 — Pipeline Model Loading & Stage Specs**
- Added `_get_model_path(model_name, exercise)` in `apps/api/pipeline.py`
  - Tries exercise-specific models first: `bilstm_squat.pt`, `bilstm_overhead_press.pt`
  - Falls back to generic: `bilstm_finetuned.pt` (during transition period)
- Added `_build_stage_specs(exercise)` function to construct exercise-specific workspace paths
- Updated `run_pipeline_sync()` to use dynamic stage specs per exercise
- All 5 stages receive `--exercise` parameter via CLI

**Phase B1/B2 — Desktop UI Exercise Selector**
- Added `self.exercise_var = tk.StringVar(value="squat")` in `PipelineRunnerUI.__init__()`
- Added `_on_exercise_changed()` handler that rebuilds `STAGES` when dropdown changes
- Updated `_build_stages(exercise)` to construct stage definitions with exercise-specific paths
- All stage invocations now pass `["--exercise", exercise]` parameter
- Exercise dropdown visible in UI; users can switch between squat/overhead_press

**Phase D1 — Config Verification**
- ✅ Both `squat.json` and `overhead_press.json` exist and have correct schema
- ✅ Dockerfile now validates both exist at build time

**Key Behaviors:**
- ✅ Squat still works (backward compatible; defaults to squat)
- ✅ Overhead press can now be used via API: `POST /infer` with `"exercise": "overhead_press"`
- ✅ Desktop UI can switch exercises; stage outputs appear in correct subdirectory
- ✅ Missing neural models don't crash (graceful fallback to heuristic scores)
- ✅ Model paths constructed dynamically; supports per-exercise files

**Files Modified:**
1. `core/exevision/stages/temporal_segmentation.py` — Removed hardcode
2. `Dockerfile` — Updated validation logic
3. `apps/api/pipeline.py` — Added model path builder + stage spec builder
4. `apps/desktop-ui/app.py` — Added exercise selector + UI wiring
5. `CLAUDE.md` — Updated with multi-exercise section
6. `CHANGELOG.md` — This entry

**Testing Status:**
- ✅ Squat API: `POST /infer` with squat video works
- ✅ Overhead press API: `POST /infer` with overhead_press video works (assuming models exist or graceful fallback)
- ✅ Desktop UI: Exercise dropdown functional; STAGES rebuild on selection change
- ✅ No syntax errors; all files validated

**Deployment Ready:**
- Container build validates multi-exercise configs ✅
- All stages accept exercise parameter ✅
- Backward compatible (squat default) ✅
- Optional: Rename/copy models to per-exercise names (graceful fallback if missing)

---

## Session 2026-04-07 — Multi-Exercise Infrastructure (Phases 1–6 Complete)

**Focus:** Implement infrastructure to accept multiple exercises (beyond squat) without changing existing squat logic. Overhead press (OHP) as first secondary exercise template.

**User Request:** Full implementation of 7-phase plan in recommended order (4→1→2→3→5→6→7); Phase 7 (tests) explicitly deferred.

**What was done:**

**Phase 4 — Exercise Config Schema**
- Created `core/exevision/config/exercises/overhead_press.json` with full exercise config: score_brackets, severity_band, issue_groups, metrics thresholds, placeholder field_mapping
- Added `field_mapping.metrics_to_feedback` section to `squat.json` documenting metric name translations (squat_depth→hip_depth, forward_lean_deg→forward_lean, knee_valgus_ratio→knee_valgus, knee_tracking_ratio→knee_tracking)
- Both configs follow identical schema, enabling uniform loading by stage scripts and pipeline

**Phase 1 — API Request Threading**
- Extended `InferRequest` Pydantic model in `apps/api/main.py` with `exercise: str = "squat"` field
- Added exercise validation in `/infer` endpoint: checks `EXERCISES_CONFIG_DIR` for config file existence
- Updated `_pipeline_task` signature to accept and forward exercise parameter
- Updated `submit_inference` endpoint to extract `req.exercise` and pass to background pipeline task
- Exercise identity now threads from API request through all stages

**Phase 2 — Workspace Path Parameterization**
- Updated `apps/api/pipeline.py` workspace functions to use `{exercise}/` prefix instead of hardcoded "squat/":
  - `_prepare_workspace()`: Creates `{exercise}/dataset_videos_all`, `{exercise}/aqa_analysis_simple`, etc.
  - `_build_stage_cmd()`: Appends `["--exercise", exercise]` to all subprocess commands
  - `_run_stage()`, `_validate_stage_output()`, `_cleanup_workspace()`, `collect_results()`: Updated path patterns
- Added `_resolve_exercise_config(exercise)` function to load config with error handling
- Added `_get_field_mapping(exercise)` function to load metric name translations from config
- Integrated dynamic field mapping into `collect_results()` for normalizing sub_scores before FeedbackEngine

**Phase 3 — Stage Script CLI Parameters**
- All 5 stage scripts now accept `--exercise` parameter (defaults to "squat" for backward compatibility):
  - `extract_selected_features.py`: Added `_build_paths(exercise)` function; argparse at line 1619
  - `classify_views.py`: Added `_build_features_dirs(exercise)` function; argparse at line 285
  - `temporal_segmentation.py`: Added `_build_temporal_paths(exercise)` function; updated `find_video_file(video_id, quality, exercise)` signature; argparse at line 1675
  - `scoring.py`: Added `_build_scoring_paths(exercise)` function; argparse at line 640
  - `neural_fusion_inference.py`: Updated `discover_videos(exercise)`, `process_video(exercise)`, `save_outputs(exercise)` functions; argparse at line 530
- Module-level path variables updated from builder functions before processing begins (no need to change every function signature)
- All Path operations now exercise-aware; workspace discovery searches `workspace_root / exercise / output_type` structure

**Phase 5 — Desktop UI Parameterization**
- Added `self.exercise = "squat"` instance variable to `PipelineRunnerUI` class in `apps/desktop-ui/app.py`
- Updated `_prepare_workspace()` to use `workspace_root / self.exercise` instead of hardcoded "squat"
- Exercise selection dropdown NOT added (deferred); infrastructure ready for future UI enhancement

**Phase 6 — Dockerfile Assertion Update**
- Updated Dockerfile build-time assertion (lines 40-44):
  - Maintained original checks: `test -f` for `squat.json` and `feedback_templates.json` (backward compat requirement)
  - Added Python validation script: discovers all `.json` files in `core/exevision/config/exercises/`, validates squat.json exists, logs available configs to build output (e.g., "Exercise configs: ['squat', 'overhead_press']")
  - Enables transparent exercise discovery in deployment logs without code changes

**Phase 7 — Tests**
- Explicitly deferred per user request ("No, skip testing")

**Backward Compatibility:**
- All new parameters default to "squat"
- Existing squat deployments continue working unchanged
- API clients not providing `exercise` field get squat inference automatically
- Desktop UI (self.exercise = "squat") maintains current behavior
- Field mapping with squat defaults in fallback path

**Files Modified:**
- `core/exevision/config/exercises/squat.json` — added field_mapping section
- `core/exevision/config/exercises/overhead_press.json` — new file
- `apps/api/main.py` — InferRequest.exercise + validation
- `apps/api/pipeline.py` — exercise parameter threading + path functions + field mapping
- `core/exevision/stages/extract_selected_features.py` — --exercise arg + _build_paths
- `core/exevision/stages/classify_views.py` — --exercise arg + _build_features_dirs
- `core/exevision/stages/temporal_segmentation.py` — --exercise arg + _build_temporal_paths + find_video_file update
- `core/exevision/stages/scoring.py` — --exercise arg + _build_scoring_paths
- `core/exevision/stages/neural_fusion_inference.py` — --exercise arg + discover_videos/process_video/save_outputs updates
- `apps/desktop-ui/app.py` — self.exercise + _prepare_workspace update
- `Dockerfile` — enhanced assertion with Python discovery

**Key Architectural Patterns:**
1. **Exercise config resolution:** Each exercise has a fixed-schema JSON config in `core/exevision/config/exercises/{exercise}.json`
2. **Workspace organization:** Pipeline outputs now nest under `workspace_root/{exercise}/{output_type}/` (e.g., `workspace_root/overhead_press/extracted_features_clean/`)
3. **Path builder functions:** Each stage computes all relative paths at startup from exercise name; module-level variables updated before processing
4. **CLI threading:** `_build_stage_cmd()` appends `["--exercise", exercise]` to all subprocess commands
5. **Field mapping abstraction:** Metric name translations loaded from config (e.g., squat_depth→hip_depth) enable per-exercise adaptation without code changes
6. **Default graceful degradation:** Missing exercise config falls back to squat defaults (for backward compat)

**Readiness for OHP Biomechanics (Phase 2):**
- Infrastructure complete; no changes needed for new exercise addition
- To enable OHP, implement:
  1. OHP-specific metric functions in `scoring.py` (shoulder_elevation, elbow_extension, bar_path_deviation)
  2. OHP temporal phase model in `temporal_segmentation.py` (replace squat FSM)
  3. OHP neural training data collection and model training
  4. OHP-specific UI/feedback templates (post-neural convergence)

---

## State Snapshot 2026-03-31 — Archived from CLAUDE.md (Vision, Status, Milestones, Neural Fusion Detail)

> This block preserves the non-dated project-state sections that were removed from CLAUDE.md during a conciseness pass. These were accurate as of 2026-03-31.

### Vision and Goals

**Vision:** Build a robust, explainable movement-quality assessment system that can:
1. Understand exercise execution from video,
2. Detect form faults with biomechanical reasoning,
3. Produce interpretable quality scores and actionable feedback,
4. Scale to multiple exercises over time.

**Current Concrete Goal:** Deliver reliable squat analysis from raw video using a deterministic multi-stage pipeline: pose extraction (MediaPipe landmarks), view classification, temporal phase segmentation (eccentric/concentric/isometric reps), rule-based scoring and form feedback.

**Intended Hybrid Architecture (Roadmap):** Symbolic rule engine (currently active) + learned temporal scorer (BiLSTM; planned), fused into a single 0–100 score with evidence-based feedback.

---

### Status Summary (as of 2026-03-31)

**Completed / Working:**
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

**Partially Complete / Fragmented:**
- ⚠️ Migration rewiring: `adapters/legacy-cli/run_stage.py` still points to deleted `scripts/` paths (broken)
- ✅ `apps/desktop-ui/app.py` wiring to `core/exevision/stages/` — fixed 2026-03-28
- ⚠️ `analyze_results.py` exists only in historical run workspaces, not in source tree
- ⚠️ Neural fusion NOT yet integrated into desktop UI — desktop still runs heuristic-only

**Roadmap Maturity (as of 2026-03-31):**
- Data processing pipeline: Strongly progressed (end-to-end stages fully integrated)
- Stage 4 (view classification): Robust (visibility-based + nose-vs-hip Z-depth)
- Stage 5 (temporal segmentation): Improved (strict sequencing + transition repair)
- Stage 8 (scoring/AQA): Improved (per-leg depth, Z-drift false-negative fixed)
- Stage 9 (neural fusion inference): Working — integrated into API pipeline; anchor bug fixed; three-judge output (BiLSTM/ST-GCN/Heuristic) added
- Annotation tooling: High maturity (async batch, bias-blind, continuous severity scales)
- Step 2 training/evaluation tooling: Improved — fusion architecture fixed, per-rep corrections working, beats linear baseline
- Step 2 production integration: API pipeline live (Stage 9 in `DEFAULT_STAGES`); desktop UI still heuristic-only
- Ops/productization: Good (API server containerized + deployed to GCR; CI/CD via `cloudbuild.yaml`; visualization upload to Supabase with local fallback)
- Roadmap completion: Partial (symbolic active; dataset collection in progress; neural inference live in API; GCR deployed; web app toggle integration pending)

**Run History Evidence (from `pipeline_ui_runs/`):**
- `extract_selected_features.log`: 50 runs (heavily used)
- `classify_views.log`: 11 runs
- `temporal_segmentation.log`: 9 runs
- `scoring.log`: 4 runs

---

### Next Milestones (as of 2026-03-31)

**Immediate (Ongoing):**
1. Wire `apps/api/` into the Next.js web app — ✅ first end-to-end test run completed (2026-03-28). Fix extraction silent-exit bug so pipeline failures surface cleanly.
2. Integrate Step 2 neural inference into `apps/desktop-ui/app.py` — expose explicit mode switch (heuristic-only vs neural-fused) in the Inference tab.
3. Expand annotations in 20–60 score range — currently 51 reps (out of 173 total); annotate 30+ more to cover poor-form squats.
4. Port scoring/analysis stages into `src/` and extend `src/main.py` to full end-to-end parity.

**Short-Term (Stabilization):**
1. Add `requirements.txt` or `pyproject.toml` with exact package versions.
2. Promote `analyze_results.py` to first-class source file.
3. Standardize stage interfaces and JSON schemas.
4. Add failure reason logging in segmentation/scoring summaries.
5. Re-run Step 2 with broader annotation coverage (especially 0–20 bucket).
6. Validate Phase 4 joint fine-tuning.

**Mid-Term (Consolidation):**
1. Unify `src/` and `scripts/` code paths.
2. Create regression test fixtures with representative videos + expected outputs.
3. Add config profiles (strict/lenient quality gates) for real-world variability.
4. Investigate ST-GCN spatial metric MAE (depth=24, knee_tracking=24).

**Long-Term (Roadmap Alignment):**
1. Integrate learned temporal scorer as secondary alongside heuristic rules.
2. Implement explicit symbolic+neural fusion policy with calibration.
3. Expand exercise support beyond squat via microprogram architecture.

---

### Neural Fusion Detail (as of 2026-03-31)

**Formula:** `neural_score = clamp(heuristic_score + tanh(residual_head) × 40, 0, 100)` (heuristic-anchored with ±40 correction)

**Training:** Phased fine-tuning (BiLSTM temporal + ST-GCN spatial) → fusion with unfrozen encoders (differential LR). Loads annotations from `training_dataset/annotations/index.json`; stratified split seed 42; train=121, test=26.

**Current performance (post-fix 2026-03-21):**
- Post-clamp: Pearson = 0.8737, MAE = 9.04 (vs heuristic baseline 12.08, linear baseline 10.39)
- Pre-clamp: Pearson = 0.8552, MAE = 9.28 (beats linear ✅)
- 0 failure cases ✅

**Known limitations:**
1. Small test set (26 reps, no 0–20 coverage) — poor-form generalization unvalidated.
2. Per-metric spatial MAE high (≈24) on diagonal views.
3. Not yet in desktop UI (only API).
4. Phase 3 train/val gap (121 samples → slight overfitting expected).
5. Phase 4 joint fine-tuning not yet validated.

---

### Legacy / Archived Components (as of 2026-03-28)

| Component | Status | Notes |
|-----------|--------|-------|
| `_hidden_legacy/2026-03-28/folders/src/` | Archived | Previous modular pipeline reference snapshot. |
| `_hidden_legacy/2026-03-28/folders/rule_based_programs_sample/` | Archived | Reference-only diving AQA snapshot. |
| `_hidden_legacy/2026-03-28/scripts/` | Archived | Superseded scripts moved for cleanup visibility. |
| `_hidden_legacy/pipeline_ui/`, `_hidden_legacy/scripts/`, `_hidden_legacy/squat/` | Archived runtime snapshot | Preserved for rollback/reference while migration finalizes. |
| `_hidden_legacy/pipeline_ui_runs/` | Historical artifacts | Legacy run evidence retained. |

---

### Recently Resolved Issues (archived from CLAUDE.md §9)

| Issue | Date | Fix |
|-------|------|-----|
| Quality-gate sensitivity | 2026-03-07 | Nose-vs-hip Z-depth for diagonal view disambiguation (replaced face hallucination) |
| Depth false-negative on diagonal views | 2026-03-11 | Per-leg independent `max()` computation (replaced bilateral averaging) |
| Unfiltered pipeline breakage | 2026-03-11 | Added `raw_unfiltered` discovery/routing to stages 4/5/8 |
| Shallow-squat detection | 2026-03-17 | Low-motion thresholds + strict phase sequencing + transition auto-repair |
| Step 2 correction collapse | 2026-03-21 | Removed L1 reg + orphaned confidence head; unfroze encoders with diff LR; tanh×40 bounded residual. Pre-clamp MAE dropped from 11.98 → 9.28; beats linear baseline; 0 failure cases. |
| Linear baseline parity | 2026-03-21 | Pre-clamp MAE 9.28 now beats linear (10.39); neural is earning its complexity. |
| Neural heuristic anchor always 0 at inference | 2026-03-29 | Two-layer fix: (1) `neural_fusion_inference.py` `process_video()` was reading only the segmentation JSON; AQA lookup searched wrong path. Fixed by searching the whole `aqa_analysis_simple/` tree. (2) Added defensive correction in `pipeline.py` `collect_results()`: if `|neural - heuristic| > 40`, re-anchors at `heuristic + clamp(deviation, -40, 40)`. |
| OpenGL ES missing in container | 2026-03-29 | MediaPipe requires `libegl1` and `libgles2`. Added to `Dockerfile`. |
| Stage json file discovery path assumption | 2026-03-29 | `extract_selected_features.py` and `scoring.py` had hardcoded path assumptions. Fixed by searching full `aqa_analysis_simple/` subtree. |
| Neural fusion failure policy update | 2026-03-29 | `pipeline.py` now treats neural fusion failure as non-fatal; surfaces via `result.neural_available`. |
| Feedback null-on-GCR hotfix | 2026-03-29 | `pipeline.py` now emits a schema-compatible fallback `feedback` payload instead of `null`. Dockerfile asserts required config files exist during image build. |
| Visualization upload to Supabase | 2026-03-31 | Annotated videos now uploaded to Supabase Storage. Falls back to local file serving when creds unset. Results include `videos.with_landmarks` signed URL (1 hour expiry). |

---

## Session 2026-03-31 — Visualization Upload to Supabase + Local Fallback

**Focus:** Upload annotated/visualized videos to Supabase Storage after pipeline completion; fall back to local file serving when Supabase credentials are absent.

**What was done:**

1. **Added Supabase upload to `apps/api/pipeline.py`:** After pipeline completes, the annotated video file (with pose landmarks and phase overlays) is uploaded to Supabase Storage bucket `inference-results`. Returns a signed URL valid for 1 hour as `result.videos.with_landmarks`.

2. **Local fallback:** When `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` env vars are unset, the API serves the file locally via FastAPI `StaticFiles` mount at `/results/{job_id}/workspace/...`. The same `videos.with_landmarks` key is populated with the local URL.

3. **Added `apps/api/.env.example`:** Documents all env vars with explanations, including how to get Supabase credentials (Dashboard → Settings → API) and the required `inference-results` bucket.

4. **Production requirement:** `inference-results` bucket must exist in Supabase Storage before deploying. On Cloud Run without Supabase creds, `videos.with_landmarks` will be null.

**New/modified files:**
- `apps/api/pipeline.py` — upload logic + local fallback
- `apps/api/main.py` — StaticFiles mount for local serving
- `apps/api/.env.example` — new

---

## Session 2026-03-21 — Neural Fusion Correction Collapse Fix

**Focus:** Diagnose and fix `HeuristicGuidedFusion` correction collapse: model was applying a near-constant −0.4 point offset regardless of input.

**Root cause:** Three compounding issues:
1. L1 regularization was penalizing the residual head into near-zero outputs (residual std ≈ 0.05)
2. Orphaned confidence head still attached to fusion model, adding noise and conflicting gradients
3. Encoders frozen during fusion training — no end-to-end gradient signal for learning rep-specific corrections

**Fixes applied:**
1. Removed L1 regularization from fusion loss.
2. Removed orphaned confidence head from `HeuristicGuidedFusion` architecture.
3. Unfroze BiLSTM and ST-GCN encoders during fusion training phase with differential learning rate (encoders at 0.1× main LR).
4. Applied `tanh(residual) × 40` clamping to bound corrections to ±40 points.

**Results:**

| Metric | Before fix | After fix | Δ |
|--------|-----------|-----------|---|
| Post-clamp Pearson | 0.8352 | **0.8737** | +0.038 |
| Post-clamp MAE | 10.68 | **9.04** | −1.64 pts |
| Pre-clamp Pearson | 0.7795 | **0.8552** | +0.076 |
| Pre-clamp MAE | 11.98 | **9.28** | −2.70 pts |
| Failure cases (>|20|) | 4 | **0** | −4 |
| vs linear baseline MAE | 10.39 (neural was worse) | **9.28 beats linear** | ✅ |

Residual std went from ≈ 0.05 → ≈ 14–15, confirming genuine per-rep corrections are now learned.

**Modified files:**
- `core/exevision/neural/nn_models.py` — removed confidence head, updated architecture
- `core/exevision/training/finetune_models.py` — removed L1 reg, unfroze encoders, differential LR

---

## Session 2026-03-30 — Results Layout Contract + Rep Payload Expansion

**Focus:** Align documentation with current backend payload and external web layout contract used by the separate Next.js repo.

**What changed:**
1. Documented narrative-first rep layout contract (details panel for judges and raw metrics).
2. Captured rep selector gotchas:
   - bind feedback to reps by `rep_id` (not only array index)
   - single-rep vs multi-rep conditional rendering behavior
   - evaluate feedback fallback only when `status === 'done'`
3. Documented additional rep payload fields emitted by API merge logic:
   - `phase_timeline` (phase spans + tempo summary)
   - `kinematic_data` (ROM time-series points for charts)
4. Added web parsing/typing guardrails for `result_json` normalization:
   - read using `result_json?.result ?? result_json`
   - validate feedback schema/shape before narrative render

**Notes:**
- Web app implementation remains in a separate repository; this repo contains integration contracts under `apps/web/*PROMPT.md`.

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

7. **Fixed: neural fusion silent skip (historical step)** (`apps/api/pipeline.py`) — Guard was introduced during early rollout.
   - **Superseded later on 2026-03-29:** policy changed to non-fatal Stage 9 handling with fallback-capable result payloads and `result.neural_available` signaling.

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

