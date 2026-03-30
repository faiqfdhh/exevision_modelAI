# Squat Guard — Design Spec
**Date:** 2026-03-30
**Status:** Approved
**Scope:** Minimal, non-destructive guard added to `apps/api/pipeline.py` only

---

## Problem

The inference pipeline assumes the input is a valid squat video. If a user submits a bicep curl, a static standing video, or a clip with the body mostly off-frame, the pipeline runs all five stages and either produces meaningless scores or fails with a cryptic downstream error (e.g., "Features JSON not found" or zero reps with feedback).

The guard should catch two cases early — after Stage 2.5 writes the features JSON, before any remaining stages run:

1. **Not a squat** — lower body shows no meaningful vertical movement or knee flexion
2. **Body not sufficiently visible** — key joints (hips, knees, shoulders) are absent from too many frames to evaluate

---

## Approach: Post-Stage-2.5 validation in `pipeline.py`

After `extract_selected_features` completes and its output is validated, a new `_check_squat_viability()` function reads the already-written features JSON and raises `PipelineRejectionError` if either condition fails. The remaining stages (4, 5, 8, 9) are never invoked for rejected jobs.

**Why not inside a stage script?** Keeping it in `pipeline.py` means zero changes to any stage script. The guard is purely a pipeline orchestration concern.

**Why not Stage 5?** Stage 5 detects eccentric/concentric motion flow, not exercise type. A bicep curl or row has motion too — Stage 5 would produce reps or partial reps for non-squat exercises, giving false confidence.

---

## New additions (minimal surface area)

### 1. `PipelineRejectionError` — new exception class in `pipeline.py`

```python
class PipelineRejectionError(Exception):
    """Raised when a video is structurally unsuitable for squat evaluation."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
```

### 2. `_check_squat_viability(features_json_path)` — new function in `pipeline.py`

Reads the features JSON produced by Stage 2.5. Uses the same landmark indexing as `scoring.py`:

```
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP           = 23, 24
L_KNEE, R_KNEE         = 25, 26
L_ANKLE, R_ANKLE       = 27, 28
```

Each frame is a list of landmarks; each landmark is `[x, y, z, visibility, ...]`.

**Check A — Key joint visibility:**
- For each frame, check if hips (23, 24), knees (25, 26), and at least one shoulder (11 or 12) all have `visibility >= 0.4`
- Require at least **40% of frames** to pass
- Rejection message: `"Body not sufficiently visible. Ensure hips, knees, and shoulders are fully in frame."`

**Check B — Hip vertical displacement (conservative threshold):**
- Collect hip Y coordinate (`avg(left_hip_y, right_hip_y)`) for all valid frames (hip confidence >= 0.4)
- Compute `hip_y_range = max(hip_y_values) - min(hip_y_values)` (normalized 0–1 in MediaPipe coords)
- Threshold: `< 0.06` (6% of frame height)
- Real squats typically produce 20–40% range. 6% catches static poses, lateral exercises, and seated shots.

**Check C — Knee flexion range (conservative threshold):**
- Compute knee angle per frame using the same 2D angle formula as `scoring.py` (hip–knee–ankle)
- Collect valid angles (confidence >= 0.4 on all three joints)
- Compute `knee_angle_range = max(angles) - min(angles)` in degrees
- Threshold: `< 15°`
- Real squats produce 60–90° of flexion. 15° catches non-squat exercises where knees barely move.

**Rejection logic (conservative — Approach A):**
- Reject on Check A **alone** (visibility is a hard prerequisite regardless of exercise type)
- Reject if **both** Check B AND Check C fail together (requires *both* hip motion AND knee flexion to be absent — avoids false positives on partial-range squats or awkward camera angles)

```python
visibility_ok = (fraction_of_valid_frames >= 0.40)
motion_ok = (hip_y_range >= 0.06) or (knee_angle_range >= 15.0)

if not visibility_ok:
    raise PipelineRejectionError("Body not sufficiently visible. Ensure hips, knees, and shoulders are fully in frame.")
if not motion_ok:
    raise PipelineRejectionError("No squat movement detected. This video does not appear to contain a squat exercise.")
```

### 3. Call site in `run_pipeline_sync`

After the `extract_selected_features` stage block (just before the `_delete_input_video` call):

```python
if key == "extract_selected_features":
    # Guard: reject non-squats and visibility failures before running downstream stages
    features_json = _find_features_json(workspace_root, video_id)  # glob helper
    if features_json:
        _check_squat_viability(features_json)   # raises PipelineRejectionError if bad
    _delete_input_video(workspace_root, video_path.name)
```

`PipelineRejectionError` is **not caught** in `run_pipeline_sync` — it propagates to the caller.

### 4. `main.py` catches `PipelineRejectionError`

In the job execution handler (wherever `run_pipeline_sync` is called), add one `except` branch:

```python
except PipelineRejectionError as exc:
    # Update job status to "rejected" with the reason
    jobs[job_id] = {"status": "rejected", "reason": exc.reason}
    # Fire callback if present
    if callback_url:
        httpx.post(callback_url, json={"job_id": job_id, "status": "rejected", "reason": exc.reason})
```

The frontend can check `status === "rejected"` and show `reason` to the user — no code changes needed on the web side beyond handling a new status value.

---

## What is NOT changed

- No stage scripts modified (`extract_selected_features.py`, `scoring.py`, `temporal_segmentation.py`, etc.)
- No changes to the features JSON schema
- No new stage in `DEFAULT_STAGES`
- No changes to `collect_results` or any downstream logic
- Desktop UI (`apps/desktop-ui/app.py`) is unaffected — guard lives only in the API pipeline

---

## Edge cases

| Scenario | Outcome |
|----------|---------|
| Partial squat (very short ROM) | Passes if hip OR knee range exceeds threshold |
| Diagonal camera angle (hip partially occluded) | Passes if 40% of frames have sufficient visibility |
| User crouching but not squatting (e.g., picking up object) | Likely passes — the guard is conservative. False negatives are acceptable. |
| Multiple people in frame | Stage 2.5 already picks the most prominent person — guard applies to that person |
| Features JSON missing (Stage 2.5 failed) | `_find_features_json` returns None → guard skips → Stage 2.5 failure propagates normally via `_validate_stage_output` |
| All frames below visibility threshold | Rejected with visibility message |

---

## Files changed

| File | Change |
|------|--------|
| `apps/api/pipeline.py` | Add `PipelineRejectionError`, `_check_squat_viability()`, call site in `run_pipeline_sync` |
| `apps/api/main.py` | Add `except PipelineRejectionError` branch in job handler |

**Total: 2 files, ~60 lines added, 0 lines modified in existing logic.**
