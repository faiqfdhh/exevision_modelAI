# Multi-Exercise Implementation Handoff

**Prepared for:** Another agent/developer  
**Date:** 2026-04-09  
**Branch:** `multiexercise`  
**Objective:** Enable multi-exercise support (Squat + Overhead Press) with pragmatic refactoring  
**Scope:** Remove critical hardcodes, implement exercise selector in Desktop UI, ensure squat still works

---

## Architecture Overview

### Decisions Made

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| **Refactoring Scope** | Pragmatic (minimal disruption) | Remove only CRITICAL hardcodes; defer nice-to-have refactors |
| **Feedback Templates** | Reuse + Exercise-Specific | Inherit squat's generic tier language; add exercise-specific metrics to feedback_templates.json |
| **View Detection** | Same Heuristics | No new logic; existing eye/shoulder detection works for overhead press |
| **Desktop UI** | Implement Now | Add exercise dropdown selector; thread parameter through pipeline |
| **Neural Models** | Per-Exercise Files | `bilstm_squat.pt`, `bilstm_overhead_press.pt`, `stgcn_squat.pt`, etc. |
| **Global Mutation** | Keep As-Is | Stage scripts still reassign globals at runtime (simplest path) |

### High-Level Changes

```
Blocking Issues Fixed:
├── 1. temporal_segmentation.py: Remove CURRENT_EXERCISE = "squat" hardcode
├── 2. Dockerfile: Update validation from hardcoded squat → dynamic exercise discovery
├── 3. apps/api/pipeline.py: Fix model path construction for per-exercise files
├── 4. apps/desktop-ui/app.py: Add exercise dropdown + thread through stage invocation
├── 5. core/exevision/config/templates/feedback_templates.json: Add exercise-specific feedback
└── 6. neural_fusion_inference.py: Load models with exercise parameter

Non-Blocking (Deferred):
├── Refactor global mutation pattern in stage scripts
├── Extract view-specific weights from scoring.py to config
└── Add per-exercise view thresholds
```

---

## Step-by-Step Implementation

### Phase A: Core Blockers (API Focus)

#### A1. Fix `temporal_segmentation.py`

**File:** `core/exevision/stages/temporal_segmentation.py`

**Current Problem:** Line 67 has `CURRENT_EXERCISE = "squat"` hardcoded.

**Change:**
- Line 67: **DELETE** `CURRENT_EXERCISE = "squat"`
- Search file for `CURRENT_EXERCISE` usage
  - If used in logic → replace with `args.exercise` or `exercise` parameter
  - If only in comments → just remove/update comment

**Validation:**
- Script should accept `python temporal_segmentation.py --video-id test --exercise overhead_press`
- Should not crash or reference hardcoded "squat"

---

#### A2. Fix `Dockerfile`

**File:** `Dockerfile`

**Current Problem:** Lines 46-48 hardcode squat.json validation.

**Current code:**
```dockerfile
RUN test -f /app/core/exevision/config/exercises/squat.json
RUN python -c "... configs = list(Path('core/exevision/config/exercises').glob('*.json')); assert any(c.name == 'squat.json' for c in configs)"
```

**Replace with:**
```dockerfile
# Verify all required exercise configs exist
RUN python -c "from pathlib import Path; \
    configs = {c.stem for c in Path('/app/core/exevision/config/exercises').glob('*.json')}; \
    required = {'squat', 'overhead_press'}; \
    missing = required - configs; \
    assert not missing, f'Missing exercise configs: {missing}'; \
    print(f'Available exercises: {sorted(configs)}')"
```

**Validation:**
- Build image: `docker build -t exevision-test:latest .`
- Should succeed if both squat.json and overhead_press.json exist
- Should fail with clear message if either is missing

---

#### A3. Fix Model Loading in `neural_fusion_inference.py`

**File:** `core/exevision/stages/neural_fusion_inference.py`

**Current Problem:** Models loaded from hardcoded shared paths.

**Locate:** Lines where models are loaded (search for `bilstm_ckpt`, `stgcn_ckpt`, `fusion_ckpt`)

**Current Pattern:**
```python
bilstm_ckpt = "models/bilstm_finetuned.pt"
stgcn_ckpt = "models/stgcn_finetuned.pt"
fusion_ckpt = "models/fusion_layer.pt"
```

**Change to:**
```python
# Accept exercise parameter from CLI or environment
exercise = args.exercise  # or getenv("EXERCISE", "squat")

bilstm_ckpt = f"models/bilstm_{exercise}.pt"
stgcn_ckpt = f"models/stgcn_{exercise}.pt"
fusion_ckpt = f"models/fusion_{exercise}.pt"
```

**Handle Missing Models:**
```python
def check_model_exists(path, exercise):
    if not Path(path).exists():
        logger.warning(f"Model not found: {path}. Skipping neural fusion for {exercise}.")
        return False
    return True

# In main():
if not all([
    check_model_exists(bilstm_ckpt, exercise),
    check_model_exists(stgcn_ckpt, exercise),
    check_model_exists(fusion_ckpt, exercise)
]):
    logger.info("Neural fusion disabled; returning heuristic scores only")
    # Set a flag or return early
```

**Validation:**
- Script should run with squat models: `python neural_fusion_inference.py --exercise squat --video-id test`
- If overhead_press models missing, should gracefully skip neural (not crash)
- Result should have `neural_available: false` if models missing

---

#### A4. Update `apps/api/pipeline.py` to Pass Models

**File:** `apps/api/pipeline.py`

**Current Problem:** Hardcoded model paths don't account for exercise-specific files.

**Locate:** Where models are passed to neural_fusion stage (search for `BILSTM_CKPT`, `STGCN_CKPT`)

**Current Pattern (approx. line 34-36):**
```python
BILSTM_CKPT = WORKSPACE_ROOT / "models" / "bilstm_finetuned.pt"
STGCN_CKPT = WORKSPACE_ROOT / "models" / "stgcn_finetuned.pt"
FUSION_CKPT = WORKSPACE_ROOT / "models" / "fusion_layer.pt"
```

**Change:**
```python
def get_model_path(model_name, exercise):
    """Construct exercise-specific model path."""
    if model_name in ["bilstm", "stgcn", "fusion"]:
        return WORKSPACE_ROOT / "models" / f"{model_name}_{exercise}.pt"
    return WORKSPACE_ROOT / "models" / f"{model_name}.pt"

# In _build_stage_cmd() or where neural_fusion is invoked:
# Build args with exercise-specific models
bilstm_path = get_model_path("bilstm", exercise)
stgcn_path = get_model_path("stgcn", exercise)
fusion_path = get_model_path("fusion", exercise)

# Add to CLI: --bilstm-ckpt {bilstm_path} --stgcn-ckpt {stgcn_path} --fusion-ckpt {fusion_path}
```

**Validation:**
- POST to `/infer` with squat video → uses `bilstm_squat.pt`
- POST to `/infer` with overhead_press video + `"exercise": "overhead_press"` → uses `bilstm_overhead_press.pt`
- If models missing, pipeline continues with heuristic scores

---

### Phase B: Desktop UI (Optional, But Recommended)

#### B1. Add Exercise Selector to UI

**File:** `apps/desktop-ui/app.py`

**Current Problem:** `self.exercise = "squat"` hardcoded; no UI element.

**Locate:** Main app class `ExeVisionApp` (around line 150-200)

**Add Exercise Dropdown (after dataset selector, around line 185):**

```python
# Exercise Selection
tk.Label(controls_frame, text="Exercise:").pack(side=tk.LEFT, padx=5)
self.exercise_var = tk.StringVar(value="squat")
exercise_choices = ["squat", "overhead_press"]  # Auto-discover from config/ later
exercise_dropdown = ttk.Combobox(
    controls_frame, 
    textvariable=self.exercise_var, 
    values=exercise_choices,
    state="readonly",
    width=15
)
exercise_dropdown.pack(side=tk.LEFT, padx=5)
```

**Update `self.exercise` assignment:**

**Old (line ~170):**
```python
self.exercise = "squat"
```

**New:**
```python
self.exercise = self.exercise_var.get()  # Read from dropdown
```

---

#### B2. Thread Exercise Through Stage Execution

**File:** `apps/desktop-ui/app.py`

**Locate:** Where stages are invoked (search for `_execute_stage` or stage command building)

**Current Pattern (approx. line 300-350):**
```python
def run_stage(stage_key):
    cmd = [python, stage_script, args...]
    # BUG: exercise not passed
    subprocess.run(cmd)
```

**Change:**
```python
def run_stage(stage_key):
    exercise = self.exercise_var.get()  # Get current selection
    cmd = [python, stage_script, "--exercise", exercise, args...]
    subprocess.run(cmd)
```

**Update STAGE_SPECS (lines 52-92):**

**Old Pattern:**
```python
STAGES = {
    "extract_selected_features": {
        "script": "core/exevision/stages/extract_selected_features.py",
        "output_paths": ["squat/extracted_features_clean", ...]
    }
}
```

**New Pattern:**
```python
def get_stage_specs(exercise="squat"):
    """Return stage specs with correct exercise paths."""
    return {
        "extract_selected_features": {
            "script": "core/exevision/stages/extract_selected_features.py",
            "output_paths": [f"{exercise}/extracted_features_clean", ...]
        },
        # ... other stages
    }

# In main UI class:
self.stage_specs = get_stage_specs(self.exercise_var.get())

# Update when exercise changes:
self.exercise_var.trace("w", lambda *args: self._refresh_stage_specs())

def _refresh_stage_specs(self):
    self.stage_specs = get_stage_specs(self.exercise_var.get())
```

**Validation:**
- Launch Desktop UI
- Select "overhead_press" from dropdown
- Run a stage → verify logs show exercise parameter
- Verify output appears in `overhead_press/extracted_features_clean/`, not `squat/`
- Switch back to squat → verify paths revert

---

### Phase C: Feedback Templates (Exercise-Specific)

#### C1. Add Overhead Press Metrics to `feedback_templates.json`

**File:** `core/exevision/config/templates/feedback_templates.json`

**Current:** Squat-specific metrics like "forward_lean", "knee_valgus", etc.

**Add:** Overhead press metrics (shoulder_elevation, elbow_extension, bar_path_deviation, etc.)

**Current Structure (example):**
```json
{
  "improvement_phrases": {
    "significant": ["Great progress!", "Nice improvement..."],
    "moderate": [...],
    "slight": [...]
  },
  "win_phrases": {
    "metric_name": {
      "excellent": "Perfect [METRIC_LABEL]! [IMPROVEMENT_PHRASE]",
      "strong": "Strong [METRIC_LABEL]. [IMPROVEMENT_PHRASE]",
      "okay": "[METRIC_LABEL] looking good. [IMPROVEMENT_PHRASE]"
    }
  },
  "metric_labels": {
    "forward_lean": "Forward Lean",
    "knee_valgus": "Knee Stability",
    ...
  }
}
```

**Add (after squat metrics):**
```json
{
  "metric_labels": {
    "forward_lean": "Forward Lean",
    "knee_valgus": "Knee Stability",
    
    // OVERHEAD PRESS
    "shoulder_elevation": "Shoulder Position",
    "elbow_extension": "Arm Extension",
    "bar_path_deviation": "Bar Path Control",
    "wrist_alignment": "Wrist Alignment",
    "core_stability": "Core Engagement",
    ...
  },
  "metric_specific_cues": {
    // Squat-specific
    "forward_lean": {
      "needs_work": "Focus on keeping your torso more upright",
      "focus_here": "Chest should be up; avoid excessive forward lean"
    },
    "knee_valgus": {
      "needs_work": "Knees drifting inward; push them outward",
      "focus_here": "Keep knees tracking over your toes"
    },
    
    // Overhead Press-specific
    "shoulder_elevation": {
      "needs_work": "Shoulders need more shrug at lockout",
      "focus_here": "Fully shrug and stabilize at the top"
    },
    "elbow_extension": {
      "needs_work": "Elbows not fully locked; extend more",
      "focus_here": "Complete your rep with full arm extension"
    },
    "bar_path_deviation": {
      "needs_work": "Bar path is inconsistent; stabilize the press",
      "focus_here": "Drive bar straight up without drifting forward"
    },
    ...
  }
}
```

**Validation:**
- Load feedback engine with overhead_press.json
- Verify metric labels and cues render correctly in feedback output

---

### Phase D: Confirm Overhead Press Config

#### D1. Verify/Update `overhead_press.json`

**File:** `core/exevision/config/exercises/overhead_press.json`

**Expected Structure (should match squat.json):**
```json
{
  "schema_version": "1.0",
  "exercise": "overhead_press",
  "score_brackets": {
    "0-39": { "tier": "critical", "opening": "..." },
    ...
  },
  "improvement_threshold": 75,
  "severity_band": 5,
  "metrics": {
    "shoulder_elevation": {
      "unit": "degrees",
      "good_threshold": 15,
      "bad_threshold": 0,
      "higher_is_better": true
    },
    "elbow_extension": {
      "unit": "ratio",
      "good_threshold": 0.95,
      "bad_threshold": 0.80,
      "higher_is_better": true
    },
    ... (other metrics)
  },
  "issue_groups": {
    "lockout_quality": {
      "metrics": ["shoulder_elevation", "elbow_extension"],
      "label": "Lockout Quality",
      "single_cues": {...},
      "combined_cue": "..."
    },
    ...
  },
  "field_mapping": {
    "metrics_to_feedback": {
      // Map AQA output field names to config metric keys
      "shoulder_elevation_deg": "shoulder_elevation",
      ...
    }
  }
}
```

**Action:**
- If overhead_press.json doesn't exist, copy from squat.json as template
- Update metric names and thresholds for overhead press (use placeholder values; will be tuned later)
- Ensure schema_version, score_brackets, issue_groups follow same structure as squat.json

**Validation:**
- Load config: `json.load(open("core/exevision/config/exercises/overhead_press.json"))`
- Verify no schema errors; all required fields present

---

### Phase E: Create/Update Model Files

#### E1. Prepare Model Files

**Action (Manual):**
You'll need to:
1. Rename existing models (if shared currently):
   - `models/bilstm_finetuned.pt` → `models/bilstm_squat.pt`
   - `models/stgcn_finetuned.pt` → `models/stgcn_squat.pt`
   - `models/fusion_layer.pt` → `models/fusion_squat.pt`

2. For overhead press (placeholder for now):
   - Copy squat models: `cp models/bilstm_squat.pt models/bilstm_overhead_press.pt`
   - (Later, replace with actual overhead press fine-tuned models)

**Or** (if agent can't access filesystem):
- Document the rename steps in a DEPLOYMENT_NOTES.md
- Leave code to handle missing models gracefully (done in A3)

**Validation:**
- List models: `ls -la models/bilstm_*.pt models/stgcn_*.pt models/fusion_*.pt`
- Should show: bilstm_squat.pt, bilstm_overhead_press.pt, etc.

---

## Testing Checklist

### Unit Tests

- [ ] `temporal_segmentation.py --exercise squat`: Should not reference CURRENT_EXERCISE
- [ ] `temporal_segmentation.py --exercise overhead_press`: Should execute without hardcode reference
- [ ] `neural_fusion_inference.py --exercise squat`: Loads bilstm_squat.pt
- [ ] `neural_fusion_inference.py --exercise overhead_press`: Loads bilstm_overhead_press.pt (or skips if missing)
- [ ] Dockerfile builds without errors
- [ ] Dockerfile validates both squat.json and overhead_press.json exist

### Integration Tests

- [ ] API: POST `/infer` with squat video → returns heuristic + neural scores
- [ ] API: POST `/infer` with overhead_press video + `"exercise": "overhead_press"` → returns scores
- [ ] API: POST `/infer` with unknown exercise → returns 400 error
- [ ] Desktop UI: Launch app → exercise dropdown visible with squat/overhead_press options
- [ ] Desktop UI: Select overhead_press → run stage → outputs appear in `overhead_press/` dir
- [ ] Desktop UI: Switch to squat → run stage → outputs appear in `squat/` dir
- [ ] Feedback: Generate feedback for overhead_press rep → uses overhead_press.json metrics

### Regression Tests

- [ ] Squat pipeline still works end-to-end (no breaking changes)
- [ ] Existing squat videos still produce same output paths
- [ ] API backward compatible: requests without exercise parameter default to squat
- [ ] Desktop UI defaults to squat on startup

---

## Deployment Validation

### Pre-Deploy Checklist

- [ ] All CRITICAL files modified:
  - [x] temporal_segmentation.py
  - [x] Dockerfile
  - [x] neural_fusion_inference.py
  - [x] apps/api/pipeline.py
  - [x] apps/desktop-ui/app.py
  - [x] feedback_templates.json
  - [x] overhead_press.json

- [ ] Models renamed/copied:
  - [x] bilstm_squat.pt, bilstm_overhead_press.pt (etc.)

- [ ] All tests passing (see Testing Checklist above)

- [ ] No regression: Squat still works

### Deploy to GCP Cloud Run

```bash
# 1. Commit changes
git add -A
git commit -m "Enable multi-exercise support: add overhead press config, wire desktop UI, fix model loading"

# 2. Build & Push
gcloud builds submit --tag asia-southeast1-docker.pkg.dev/YOUR_PROJECT/exevision-modelai/exevision-modelai:latest --project=YOUR_PROJECT .

# 3. Deploy
gcloud run deploy exevision-modelai \
  --image=asia-southeast1-docker.pkg.dev/YOUR_PROJECT/exevision-modelai/exevision-modelai:latest \
  --region=asia-southeast1 --memory=4Gi --cpu=2 \
  --set-env-vars="INFERENCE_API_SECRET=<secret>,CORS_ORIGINS=...,..."

# 4. Smoke Test
curl -X POST http://localhost:8000/infer \
  -H "Authorization: Bearer your-secret" \
  -d '{"video_url":"<url>","exercise":"squat"}'

curl -X POST http://localhost:8000/infer \
  -H "Authorization: Bearer your-secret" \
  -d '{"video_url":"<url>","exercise":"overhead_press"}'
```

---

## Handoff Prompt for Agent

**Copy this prompt and send to your agent for implementation:**

---

### Agent Handoff Prompt

```
Your task: Implement multi-exercise support for ExeVision model following the "IMPLEMENTATION_HANDOFF.md" document in the repo root.

**Scope:**
Pragmatic refactoring to unblock Overhead Press support while keeping Squat working.

**Key Files to Modify (in order):**

1. **core/exevision/stages/temporal_segmentation.py** (CRITICAL)
   - Remove CURRENT_EXERCISE = "squat" hardcode (line 67)
   - Replace usage with args.exercise parameter
   - Validate: `python temporal_segmentation.py --exercise overhead_press` runs without hardcode reference

2. **Dockerfile** (CRITICAL)
   - Replace hardcoded squat.json validation (lines 46-48)
   - Update to: Verify both squat.json and overhead_press.json exist
   - See IMPLEMENTATION_HANDOFF.md section A2 for exact code

3. **core/exevision/stages/neural_fusion_inference.py** (CRITICAL)
   - Update model loading to use exercise-specific paths: bilstm_{exercise}.pt
   - Add graceful fallback if models missing (don't crash)
   - See section A3 for code pattern

4. **apps/api/pipeline.py** (CRITICAL)
   - Update model path construction to support per-exercise files
   - See section A4 for get_model_path() function

5. **apps/desktop-ui/app.py** (RECOMMENDED)
   - Add exercise dropdown selector (section B1)
   - Thread exercise parameter through stage invocation (section B2)
   - Update STAGE_SPECS to use dynamic paths per exercise

6. **core/exevision/config/templates/feedback_templates.json** (RECOMMENDED)
   - Add overhead press metric labels and cues (section C1)
   - Keep reusing squat's generic tier language

7. **core/exevision/config/exercises/overhead_press.json** (REQUIRED)
   - Verify config exists and has correct schema (section D1)
   - Use placeholder metrics for now; will be tuned later

8. **models/** (MANUAL - may be deferred)
   - Rename: bilstm_finetuned.pt → bilstm_squat.pt (and others)
   - Copy squat models as placeholder for overhead_press

**Testing:**
Run all tests in Testing Checklist (section "Testing Checklist" in handoff doc)

**Success Criteria:**
- [ ] Squat pipeline still works (no regression)
- [ ] Overhead press pipeline works via API: `POST /infer` with `"exercise": "overhead_press"`
- [ ] Desktop UI has exercise dropdown; can switch between squat/overhead_press
- [ ] Dockerfile builds without errors
- [ ] All 8 files modified correctly

**Do NOT:**
- Refactor global mutation pattern (keep as-is for pragmatic approach)
- Extract view weights to config (nice-to-have, defer)
- Add new view detection logic (reuse squat heuristics)

**Reference:** Read MULTIEXERCISE_ANALYSIS.md for context on current architecture.

Estimated time: 2-3 hours
```

---

## Summary

| Phase | Changes | Files | Effort | Blocking |
|-------|---------|-------|--------|----------|
| **A: Core Blockers** | Remove hardcodes, fix model loading | 4 files | 1.5h | YES |
| **B: Desktop UI** | Add exercise selector, thread parameter | 1 file | 1h | NO (nice-to-have) |
| **C: Feedback** | Add exercise-specific templates | 1 file | 0.5h | NO |
| **D: Config** | Verify overhead_press.json | 1 file | 0.25h | YES |
| **E: Models** | Rename/copy model files | 0 files (manual) | 0.25h | NO (can skip initial) |

**Total Estimated Effort:** 3-3.5 hours

**Critical Path:** A (Core Blockers) → D (Config) → Testing → Deploy

---

**Document Status:** Ready for Agent Handoff  
**Next:** Agent implements using this handoff prompt  
**Then:** Code review → Testing → Deploy to Cloud Run
