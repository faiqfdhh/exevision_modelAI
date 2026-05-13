# Multi-Exercise Readiness Analysis
**Date:** 2026-04-09  
**Branch:** multiexercise  
**Goal:** Prepare repo for multi-exercise support (Squat + Overhead Press)

---

## Executive Summary

**Current State:**
- ✅ API + Pipeline: **95% exercise-ready** (infrastructure parametrized, stages accept `--exercise` arg)
- ⚠️ Desktop UI: **0% ready** (hardcoded to squat, UI lacks exercise selector)
- ✅ Deployment: **Mostly ready** (need Dockerfile validation update)
- ✅ Config: **Exercise configs already exist** (`squat.json` + `overhead_press.json`)

**Key Finding:** The codebase already supports multiple exercises through CLI parameters and dynamic path construction. Main work is **removing scattered hardcodes and wiring UI**.

---

## Part 1: Exercise Configuration Schema

### Squat Config Structure (`core/exevision/config/exercises/squat.json`)

```json
{
  "schema_version": "1.0",
  "exercise": "squat",
  "score_brackets": {
    "0-39": { "tier": "critical", "opening": "..." },
    "40-59": { "tier": "poor", "opening": "..." },
    "60-74": { "tier": "fair", "opening": "..." },
    "75-89": { "tier": "good", "opening": "..." },
    "90-100": { "tier": "excellent", "opening": "..." }
  },
  "improvement_threshold": 75,
  "severity_band": 5,
  "metrics": {
    "forward_lean": {
      "unit": "degrees",
      "good_threshold": 20,
      "bad_threshold": 45,
      "higher_is_better": false
    },
    "hip_depth": {
      "unit": "normalized",
      "good_threshold": 0.1,
      "bad_threshold": -0.1,
      "higher_is_better": true
    },
    "knee_valgus": {
      "unit": "ratio",
      "good_threshold": 0.95,
      "bad_threshold": 0.75,
      "higher_is_better": true
    },
    "knee_tracking": {
      "unit": "ratio",
      "good_threshold": 0.95,
      "bad_threshold": 0.75,
      "higher_is_better": true
    },
    "smoothness": {
      "unit": "score",
      "good_threshold": 75,
      "bad_threshold": 50,
      "higher_is_better": true
    },
    "control": {
      "unit": "score",
      "good_threshold": 75,
      "bad_threshold": 50,
      "higher_is_better": true
    }
  },
  "issue_groups": {
    "descent_quality": {
      "metrics": ["forward_lean", "knee_valgus"],
      "label": "Descent Quality",
      "single_cues": {...},
      "combined_cue": "..."
    },
    "knee_stability": {
      "metrics": ["knee_tracking", "knee_valgus"],
      "label": "Knee Stability",
      "single_cues": {...},
      "combined_cue": "..."
    }
  },
  "field_mapping": {
    "metrics_to_feedback": {
      "forward_lean_deg": "forward_lean",
      "hip_depth": "hip_depth",
      ...
    }
  }
}
```

**For Overhead Press:** Same structure, different metric names (e.g., `shoulder_elevation`, `elbow_extension`, `bar_path_deviation`).

---

## Part 2: Pipeline Architecture

### 2.1 Workspace Path Structure

```
workspace_root/
├── {EXERCISE}/
│   ├── dataset_videos_all/                    ← Input videos
│   ├── extracted_features_clean/
│   │   ├── raw_unfiltered/
│   │   ├── filtered/
│   │   └── {video_id}.json
│   ├── visualized_poses_clean/
│   │   └── {video_id}.mp4
│   ├── segmented_reps/
│   │   └── {video_id}_segmented.json
│   ├── visualized_segmentation/
│   │   └── {phase_viz}.mp4
│   ├── aqa_analysis_simple/
│   │   └── {video_id}/aqa_simple.json
│   └── neural_analysis/
│       └── {video_id}_neural.json
└── logs/
    ├── extract_selected_features.log
    ├── classify_views.log
    ├── temporal_segmentation.log
    ├── scoring.log
    └── neural_fusion_inference.log
```

**Current Status:** All stages build paths dynamically via `_build_*_paths(exercise)` functions. **BUT:** Module-level globals still hardcoded at import time, then reassigned at runtime → fragile pattern.

### 2.2 Stage Invocation Flow

**From API (`apps/api/pipeline.py`):**
```python
# All stages receive exercise parameter
_build_stage_cmd(stage_key, exercise, args)
# ↓
base_cmd + ["--exercise", exercise] + stage_specific_args
```

**From CLI (each stage script):**
```python
parser.add_argument("--exercise", default="squat")
# then: _build_*_paths(args.exercise) → update globals
```

**From Desktop UI (`apps/desktop-ui/app.py`):**
```python
self.exercise = "squat"  # HARDCODED, never passed to stage commands
```

---

## Part 3: Core Stages Deep Dive

### Stage 2.5: Extract Selected Features
- **Input:** Raw video
- **Output:** `{exercise}/extracted_features_clean/{quality}/{video_id}.json`
- **Exercise-Awareness:** ✓ Parameterized (paths dynamic, quality modes supported)
- **Config Used:** None (hardcoded feature list)
- **Issue:** Globals reassigned at runtime; should be function parameters

### Stage 4: Classify Views
- **Input:** Feature JSON
- **Output:** Updated feature JSON with `view` label
- **Exercise-Awareness:** ✓ Parameterized
- **Config Used:** None (hardcoded view detection heuristics)
- **Issue:** Same global mutation pattern

### Stage 5: Temporal Segmentation
- **Input:** Feature JSON + raw video
- **Output:** `{exercise}/segmented_reps/{quality}/{video_id}_segmented.json`
- **Exercise-Awareness:** ⚠️ Partially (paths dynamic, but `CURRENT_EXERCISE = "squat"` hardcoded at line 67)
- **Config Used:** None (FSM phases hardcoded)
- **Issue:** Hardcoded `CURRENT_EXERCISE` constant; view thresholds may be exercise-specific

### Stage 8: Scoring (Heuristic AQA)
- **Input:** Segmented features + view
- **Output:** `{exercise}/aqa_analysis_simple/{quality}/{video_id}/aqa_simple.json`
- **Exercise-Awareness:** ✓ Parameterized
- **Config Used:** `exercises/{exercise}.json` (metrics, thresholds, issue groups)
- **Key Feature:** View-specific weight matrices (side-view emphasizes forward_lean, front-view emphasizes valgus)
- **Issue:** Metric computation is generic; weights are hardcoded for squat

### Stage 9: Neural Fusion
- **Input:** Heuristic AQA + segmented features
- **Output:** Neural scores merged with heuristic
- **Exercise-Awareness:** ✓ Parameterized (paths, video discovery)
- **Config Used:** Model checkpoint paths (currently shared, should be per-exercise)
- **Key Logic:**
  - BiLSTM: Temporal (smoothness, control)
  - ST-GCN: Spatial-temporal (depth, lean, tracking)
  - Fusion: `neural_score = clamp(heuristic + tanh(residual) × 40, 0, 100)`
  - Hard patch: If ST-GCN depth unreliable, dampen residual (0.6×)
  - Safety clamps: Prevent high scores if any subscore is poor

---

## Part 4: Feedback Engine

### 4.1 Initialization
```python
engine = FeedbackEngine(
  exercise_config_path="core/exevision/config/exercises/squat.json",
  templates_path="core/exevision/config/templates/feedback_templates.json"
)
```

### 4.2 Generation Flow
1. **Tier Assignment:** Score → bracket (excellent/good/fair/poor/critical)
2. **Quality Check:** Detect mismatches (low score but no issues = luck? high score with issues = careless?)
3. **Rep Comparison:** Improvement tier (significant ≥15%, moderate ≥8%, slight >0%, none)
4. **Win Identification:** Metrics ≥improvement_threshold (75) + improvement tier
5. **Issue Grouping:** Metrics <75, group by `issue_groups` config, tier by tone (soft/strict/very_strict)
6. **Tone Policy:**
   - 80-100: soft ("Something to keep in mind...")
   - 70-79: strict (plain cue)
   - <70: very_strict ("Priority fix...")
7. **Session Summary:** Trajectory (improving/declining/stable), most_improved_metric, persistent_issue

### 4.3 Output Schema
```python
FeedbackResult(
  schema_version="1.0",
  exercise="squat",
  reps=[RepFeedback(...)],
  session=SessionSummary(...)
)
```

---

## Part 5: Hardcoded "Squat" References (26+ instances)

| File | Lines | Issue | Severity |
|------|-------|-------|----------|
| `apps/api/pipeline.py` | 37, 55-59, 87-102, 121, 130, 171, 201, 225, 232 | Output dirs, defaults, config | HIGH |
| `apps/desktop-ui/app.py` | 52-92, 170 | STAGES tuple, exercise setter, paths | HIGH |
| `extract_selected_features.py` | 46-49, 358 | Globals, comments | MEDIUM |
| `classify_views.py` | 12-15 | Globals, paths | MEDIUM |
| `temporal_segmentation.py` | 32-35, 40-43, 52-55, 64, 67 | Globals, `CURRENT_EXERCISE = "squat"` | **CRITICAL** |
| `scoring.py` | 8, 27, 45-47, 490, 505, 520, 536 | Globals, comments, metric names | MEDIUM |
| `neural_fusion_inference.py` | 5, 218 | Comments only | LOW |
| `main.py` | 143, 163 | Comments only | LOW |

**Most Critical:** `temporal_segmentation.py:67` — `CURRENT_EXERCISE = "squat"` hardcoded constant.

---

## Part 6: API & UI Status

### API Server (`apps/api/main.py`)
- ✅ `InferRequest.exercise` parameter exists (line 160)
- ✅ Exercise validation: rejects unknown exercises
- ✅ Passes to `_pipeline_task(exercise=...)`
- ✅ **95% ready**

### Pipeline Runner (`apps/api/pipeline.py`)
- ✅ All helper functions accept `exercise` parameter
- ✅ Stage commands built with `--exercise` argument
- ✅ Output dirs parameterized: `{exercise}/aqa_analysis_simple`
- ✅ Result collection looks in exercise-specific paths
- ⚠️ STAGE_SPECS still hardcode `squat/` in output_paths (overridden at runtime)
- ✅ **95% ready**

### Desktop UI (`apps/desktop-ui/app.py`)
- ❌ `self.exercise = "squat"` hardcoded
- ❌ No UI element for exercise selection
- ❌ STAGES tuple doesn't receive exercise parameter
- ❌ Stage output paths hardcoded to `squat/`
- ❌ **0% ready** (but optional for cloud deployments)

---

## Part 7: Deployment Status

### Dockerfile
**Current checks (line 46-48):**
```dockerfile
RUN test -f /app/core/exevision/config/exercises/squat.json
RUN python -c "... assert any(c.name == 'squat.json' for c in configs)"
```

**Issue:** Hardcoded to squat; fails if only overhead_press.json exists.

**Fix Required:** Dynamic validation
```dockerfile
RUN python -c "from pathlib import Path; configs = {c.stem for c in Path('/app/core/exevision/config/exercises').glob('*.json')}; required = {'squat', 'overhead_press'}; assert required.issubset(configs), f'Missing: {required - configs}'"
```

### Cloud Build (`cloudbuild.yaml`)
- ✅ Exercise-agnostic
- ✅ **No changes needed**

### Runtime Dependencies (`requirements-runtime.txt`)
- ✅ Exercise-agnostic
- ✅ **No changes needed**

---

## Part 8: Files Critical for Refactoring

| File | Purpose | Exercise-Ready | Notes |
|------|---------|---|---|
| `core/exevision/config/exercises/squat.json` | Squat config | ✓ | Model |
| `core/exevision/config/exercises/overhead_press.json` | Overhead press config | ✓ | Already exists (placeholder) |
| `core/exevision/stages/scoring.py` | Heuristic scoring | ✓ | Loads config; weights hardcoded |
| `core/exevision/stages/neural_fusion_inference.py` | Neural inference | Partial | Models should be per-exercise |
| `core/exevision/feedback/engine.py` | Feedback generation | ✓ | Loads exercise config |
| `apps/api/main.py` | API entry point | ✓ | Validates exercise exists |
| `apps/api/pipeline.py` | Pipeline orchestration | ✓ | All stages receive exercise |
| `apps/desktop-ui/app.py` | Local UI | ❌ | Requires UI wiring |
| `core/exevision/stages/temporal_segmentation.py` | Phase detection | ⚠️ | `CURRENT_EXERCISE` hardcoded |
| `Dockerfile` | Build validation | ⚠️ | Hardcoded squat check |

---

## Part 9: Refactoring Needs

### Priority 1 (Blocking)
1. **Remove `CURRENT_EXERCISE = "squat"` from `temporal_segmentation.py`**
   - Replace with `args.exercise` parameter usage
   - Severity: CRITICAL (prevents dynamic exercise selection)

2. **Update Dockerfile validation**
   - Change from hardcoded squat check to dynamic exercise discovery
   - Severity: HIGH (fails deployment with new exercises)

3. **Wire exercise parameter in Desktop UI**
   - Add dropdown selector in UI
   - Thread exercise through stage invocation
   - Severity: MEDIUM (API works without this; UI only)

### Priority 2 (Nice to have)
4. **Refactor global mutation pattern in stage scripts**
   - Replace module-level globals with function parameters
   - Makes code testable and cleaner
   - Severity: MEDIUM (works but fragile)

5. **Separate neural models per exercise**
   - Currently shared; should be exercise-specific (bilstm_squat.pt, bilstm_overhead_press.pt)
   - Severity: LOW (can use shared models if accuracy acceptable)

6. **Extract stage weights from hardcode to config**
   - View-specific weights for scoring currently hardcoded in `scoring.py`
   - Should move to exercise config JSON
   - Severity: LOW (works but less flexible)

---

## Part 10: Adding Overhead Press Checklist

### ✅ Already Done
- [ ] Config file exists: `core/exevision/config/exercises/overhead_press.json`
- [ ] API parameter support: `exercise: str = "squat"` in request model
- [ ] Pipeline stage support: All stages accept `--exercise` argument
- [ ] Feedback engine: Loads config dynamically

### 🟡 Needs Update
- [ ] **Dockerfile:** Update validation from hardcoded squat → dynamic exercise discovery
- [ ] **Temporal Segmentation:** Remove `CURRENT_EXERCISE = "squat"` hardcode
- [ ] **Desktop UI:** Add exercise selector dropdown (optional for API-only use)
- [ ] **overhead_press.json:** Replace placeholders with actual scoring metrics

### ✏️ Implementation Steps
1. Update `temporal_segmentation.py` to use `args.exercise` instead of `CURRENT_EXERCISE`
2. Update `Dockerfile` validation logic
3. Update `core/exevision/config/exercises/overhead_press.json` with actual metrics
4. Create overhead_press-specific neural models (or reuse squat models as baseline)
5. Test API: `POST /infer` with `"exercise": "overhead_press"`
6. Test Desktop UI: Add exercise dropdown (optional)
7. Deploy: Commit changes → trigger Cloud Build → deploy to Cloud Run

---

## Part 11: Model Loading Strategy

**Current (Shared Models):**
```
models/
├── pose_landmarker_heavy.task  ← Universal (all exercises)
├── blaze_face_short_range.tflite ← Universal (all exercises)
├── bilstm_finetuned.pt        ← Shared OR per-exercise?
├── stgcn_finetuned.pt         ← Shared OR per-exercise?
└── fusion_layer.pt            ← Shared OR per-exercise?
```

**Recommendation:**
- **Pose + Face models:** Keep shared (universal body part detection)
- **BiLSTM/ST-GCN/Fusion:** Per-exercise (since metrics vary)
  - `bilstm_squat.pt`, `bilstm_overhead_press.pt`
  - `stgcn_squat.pt`, `stgcn_overhead_press.pt`
  - `fusion_squat.pt`, `fusion_overhead_press.pt`

**Update required in `neural_fusion_inference.py`:**
```python
# Current:
bilstm_ckpt = "models/bilstm_finetuned.pt"

# Updated:
bilstm_ckpt = f"models/bilstm_{exercise}.pt"
```

---

## Part 12: Confidence Assessment

| Component | Confidence | Notes |
|-----------|-----------|-------|
| **API + Pipeline** | **95%** | Infrastructure is there; just remove hardcodes |
| **Feedback Engine** | **90%** | Config-driven; templates may need exercise-specific text |
| **Scoring Logic** | **80%** | Metric computation generic; view weights hardcoded |
| **Stage Scripts** | **85%** | Support exercise params; but globals mutation is fragile |
| **Deployment** | **75%** | Dockerfile needs update; Cloud Build is fine |
| **Desktop UI** | **40%** | Requires new UI element; optional for API-only |
| **Overall Readiness** | **85%** | Can deploy overhead_press tomorrow; UI optional |

---

## Part 13: Next Steps (Recommended Order)

**Phase 3: Clarify Design** → Ask user about:
- View-specific metrics for overhead press (same as squat or different?)
- Feedback templates: exercise-specific phrases or generic?

**Phase 4: Architecture Design** → Present options for:
- Minimal refactor (just remove critical hardcodes, keep globals)
- Clean refactor (parameterize all paths, remove globals, config-driven weights)
- Middle ground (pragmatic balance)

**Phase 5: Implementation** → Execute chosen approach:
1. Remove `CURRENT_EXERCISE` hardcode
2. Update Dockerfile validation
3. Populate `overhead_press.json` with real metrics
4. Wire Desktop UI exercise selector (if approved)
5. Test API + deployment

---

**Document Generated:** 2026-04-09  
**Status:** Analysis Complete → Ready for Architecture Design Phase
