# Backend Visualization Integration — Implementation Memo

**Date:** 2026-03-30  
**Status:** ✅ COMPLETE  
**Target:** Make visualization output reachable by frontend when `generate_viz=true`

---

## Summary of Changes

The backend has been updated to fully honor the `generate_viz` parameter across the inference pipeline and expose visualization artifacts via a consistent callback and result schema.

### Changes Made

#### 1. **Fixed Stage 5 (temporal_segmentation) to Respect `generate_viz`**

**File:** `apps/api/pipeline.py`, `_build_stage_cmd()` (lines 107–110)

**Before:**
```python
elif key == "temporal_segmentation":
    return base + ["--video-id", video_id, "--no-viz"]  # UNCONDITIONAL --no-viz
```

**After:**
```python
elif key == "temporal_segmentation":
    cmd = base + ["--video-id", video_id]
    if not generate_viz:
        cmd.append("--no-viz")  # CONDITIONAL --no-viz
    return cmd
```

**Impact:** Temporal phase visualization (segmentation overlay) is now generated when `generate_viz=true` in the infer request, matching the behavior of Stage 2.5 (pose extraction).

---

#### 2. **Made Workspace Cleanup Conditional on `generate_viz`**

**File:** `apps/api/pipeline.py`, `_cleanup_workspace()` (lines 190–228)

**Before:**
```python
def _cleanup_workspace(workspace_root: Path) -> None:
    subdirs_to_remove = [
        "squat/visualized_segmentation",
        "squat/analysis_reports",
        "squat/extracted_features_clean",
        "squat/segmented_reps",
    ]
    # UNCONDITIONALLY removes visualization directories
```

**After:**
```python
def _cleanup_workspace(workspace_root: Path, generate_viz: bool = True) -> None:
    subdirs_to_remove = [
        "squat/analysis_reports",
        "squat/extracted_features_clean",
        "squat/segmented_reps",
    ]
    # Only remove visualization directories if visualization was NOT requested
    if not generate_viz:
        subdirs_to_remove.extend([
            "squat/visualized_segmentation",
            "squat/visualized_poses_clean",
        ])
```

**Impact:**
- When `generate_viz=true`: Visualization directories are preserved after pipeline completion, allowing the frontend to serve them from `/results/{job_id}/workspace/squat/visualized_poses_clean/...`
- When `generate_viz=false`: Visualization directories are cleaned up to save disk space (no unnecessary artifacts retained)
- Call site updated in `run_pipeline_sync()` to pass the `generate_viz` parameter: `_cleanup_workspace(workspace_root, generate_viz=generate_viz)`

---

#### 3. **Extended Result Schema with Visualization Metadata**

**File:** `apps/api/pipeline.py`, `collect_results()` (lines 785–808)

**Added fields to result dict:**
```python
result = {
    # ... existing fields ...
    "videos": videos_dict,  # Existing: raw + with_landmarks URLs
    
    # NEW fields for visualization metadata
    "visualization_available": bool(videos_dict.get("with_landmarks")),
    "visualization_url": videos_dict.get("with_landmarks"),
}
```

**Schema Contract:**

```json
{
  "video_id": "sample_video",
  "view": "front",
  "quality": "raw_unfiltered",
  "rep_count": 3,
  "overall_heuristic_score": 75.5,
  "overall_neural_score": 76.2,
  "neural_available": true,
  "reps": [ /* rep details */ ],
  "feedback": { /* coaching narratives */ },
  "videos": {
    "raw": "http://api.local:8000/results/..../workspace/squat/dataset_videos_all/sample_video.mp4",
    "with_landmarks": "http://api.local:8000/results/.../workspace/squat/visualized_poses_clean/filtered/sample_video_annotated.mp4"
  },
  "visualization_available": true,
  "visualization_url": "http://api.local:8000/results/.../workspace/squat/visualized_poses_clean/filtered/sample_video_annotated.mp4"
}
```

**Impact:** Frontend can now directly check `result.visualization_available` and access `result.visualization_url` without parsing the `videos` dict structure.

---

#### 4. **Enhanced Callback Payload with Visualization Reference**

**File:** `apps/api/main.py`, `_pipeline_task()` (lines 237–247 for success, 251–263 for failure)

**Before (Success):**
```python
if callback_url:
    _fire_callback(callback_url, {"job_id": job_id, "status": "done", "result": result})
```

**After (Success):**
```python
if callback_url:
    callback_payload = {
        "job_id": job_id,
        "status": "done",
        "result": result,
        "visualization_url": result.get("visualization_url"),
        "visualization_available": result.get("visualization_available", False),
    }
    _fire_callback(callback_url, callback_payload)
```

**Before (Failure):**
```python
if callback_url:
    _fire_callback(callback_url, {"job_id": job_id, "status": "failed", "error": str(exc)})
```

**After (Failure):**
```python
if callback_url:
    callback_payload = {
        "job_id": job_id,
        "status": "failed",
        "error": str(exc),
        "visualization_url": None,
        "visualization_available": False,
    }
    _fire_callback(callback_url, callback_payload)
```

**Callback Payload Contract:**

```json
{
  "job_id": "uuid-string",
  "status": "done|failed",
  "result": { /* Full result object */ },
  "visualization_url": "http://.../annotated_video.mp4",
  "visualization_available": true,
  "error": null  // Only on failure; null on success
}
```

**Benefits:**
- **Consistent shape:** Callback always includes `visualization_url` and `visualization_available` fields (null/false on failure)
- **Frontend convenience:** Can access visualization reference without nested parsing
- **Backward compatible:** Full result dict is still included for complete data

---

## No Unguarded `.exists()` Calls

**Status:** ✅ All `.exists()` calls in pipeline are properly guarded

Audit findings:
- **pipeline.py:** All 7 `.exists()` calls are preceded by conditionals or proper null checks
- **main.py:** All 5 `.exists()` calls (model checkpoint verification) are guarded conditionally
- **Intentional unguarded calls:**
  - Line 854 in pipeline.py: `spec.script.exists()` — *unguarded by design* to raise FileNotFoundError if stage script is missing (fatal error, should not be silenced)

---

## Testing

Test file: `tests/test_visualization_integration.py`

**Covered scenarios:**
1. ✅ `generate_viz=true` → Stage 2.5 and Stage 5 produce visualizations
2. ✅ `generate_viz=false` → `--no-viz` flags passed to both stages
3. ✅ Cleanup preserves visualization directories when `generate_viz=true`
4. ✅ Cleanup removes visualization directories when `generate_viz=false`
5. ✅ Result schema includes `visualization_available` and `visualization_url`
6. ✅ Callback payload includes visualization metadata on success
7. ✅ Callback payload includes null visualization fields on failure

---

## Frontend Integration Points

### 1. **Infer Request** (Next.js app → Backend API)
```typescript
POST /infer
{
  "video_url": "https://signed-url...",
  "job_id": "...",
  "generate_viz": true,  // ← Feature flag: request visualization
  "callback_url": "https://..."
}
```

### 2. **Status Endpoint** (Next.js app ← Backend API)
```typescript
GET /jobs/{jobId}
Response: {
  "result": {
    "visualization_url": "http://api:8000/results/.../annotated_video.mp4",
    "visualization_available": true,
    // ... other fields
  }
}
```

### 3. **Callback** (Backend → Next.js Supabase trigger)
```typescript
POST {callback_url}
{
  "job_id": "...",
  "status": "done",
  "visualization_url": "http://api:8000/results/.../annotated_video.mp4",
  "visualization_available": true,
  "result": { /* Full result */ }
}
```

### 4. **Rendering Recommendations**

**When `generate_viz=true`:**
```typescript
if (result?.visualization_available) {
  return <video src={result.visualization_url} controls />;
} else {
  return <p>Visualization failed to generate</p>;
}
```

**When `generate_viz=false`:**
```typescript
if (result?.videos?.raw) {
  return <video src={result.videos.raw} controls />;
}
```

---

## Performance & Scale Notes

- **Visualization directory size:** ~20–50 MB per video (depends on video length and encoding)
- **Disk retention:** Visualization directories now persist in `/results/{job_id}/workspace/squat/` when `generate_viz=true`
  - If served from a static mount (nginx/CloudFront), total disk usage for 100 jobs = ~2–5 GB
  - Cleanup is automatic for `generate_viz=false` runs
- **Callback timing:** No change; callback still fires immediately after pipeline completion (~30–120s from start)

---

## Deployment Checklist

- [ ] Merge backend changes to main
- [ ] Deploy updated `apps/api/main.py` and `apps/api/pipeline.py` to GCP Cloud Run
- [ ] **Important:** Update Next.js repo to:
  - [x] Send `generate_viz` in infer payload (already done in inference-client.ts:36)
  - [x] Read `visualization_url` from status endpoint (ready after this merge)
  - [x] Read `visualization_url` from callback payload (ready after this merge)
  - [ ] Update results page to display visualization when available
- [ ] Test end-to-end: submit job → receive callback → render visualization

---

## Summary

The backend now fully honors the `generate_viz` parameter throughout the pipeline:
- ✅ Stage 2.5 & 5 conditionally generate visualizations
- ✅ Cleanup preserves visualization artifacts when requested
- ✅ Result schema exposes visualization metadata
- ✅ Callback includes visualization reference
- ✅ All `.exists()` calls are safe (no NoneType crashes)

**Frontend can now display annotated exercise videos by:**
1. Setting `generate_viz=true` in infer request
2. Reading `result.visualization_url` from status endpoint or callback
3. Rendering `<video src={result.visualization_url} />`
