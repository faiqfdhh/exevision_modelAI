# Visualization Setup Guide

This document explains how to set up visualized video delivery for local development and production Cloud Run deployments.

## Quick Start

### Local Testing (Recommended for Development)

No setup needed! The API will automatically serve visualizations from disk.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API (no Supabase env vars needed)
$env:INFERENCE_API_SECRET="test-secret"
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# 3. Submit a job
# Visualizations will be served via http://localhost:8000/results/{job_id}/workspace/...
```

**How it works:**
- Annotated videos are generated locally in `pipeline_ui_runs/{job_id}/workspace/squat/visualized_poses_clean/`
- FastAPI's built-in `StaticFiles` middleware serves them via `/results` endpoint
- Results include `videos.with_landmarks = "http://localhost:8000/results/{job_id}/workspace/squat/visualized_poses_clean/{video_id}_annotated.mp4"`

### Production on Cloud Run with Supabase

Visualizations are uploaded to Supabase Storage for durability and availability.

#### 1. Set up Supabase Storage bucket

Go to your Supabase project dashboard:
1. **Storage** → **Create a new bucket** (or use existing)
2. Name it: `inference-results`
3. Set **Public** permission (web app needs to access signed URLs)

#### 2. Get your credentials

From **Settings** → **API** → **Keys and URLs**:
- Copy **Project URL** → `SUPABASE_URL`
- Copy **Service Role** key → `SUPABASE_SERVICE_KEY`

⚠️ **Important:** Service Role is a server-only key. Never expose it to the browser.

#### 3. Deploy to Cloud Run with env vars

```bash
gcloud run deploy exevision-modelai \
  --image="..." \
  --region=asia-southeast1 \
  --set-env-vars="INFERENCE_API_SECRET=<secret>,CORS_ORIGINS=https://your-app.vercel.app,SUPABASE_URL=https://your-project.supabase.co,SUPABASE_SERVICE_KEY=<service-key>"
```

**How it works:**
- Pipeline generates annotated videos in ephemeral Cloud Run disk
- `_upload_visualization_to_supabase()` uploads to Supabase Storage
- Returns signed URL (valid 1 hour): `https://...supabase.co/storage/v1/object/sign/visualizations/{job_id}/{video_id}_annotated.mp4?token=...`
- Results include `videos.with_landmarks` pointing to the signed URL
- Web app fetches and displays the video

#### 4. Local Docker testing with Supabase

```bash
docker build -t exevision-modelai:local .
docker run --rm -p 8000:8000 \
  -e INFERENCE_API_SECRET=test-secret \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_SERVICE_KEY=your-key \
  exevision-modelai:local
```

## Result Schema

Both local and Supabase deployments return results with this structure:

```json
{
  "video_id": "user_squat_1",
  "videos": {
    "raw": "http://localhost:8000/results/job-id/workspace/squat/dataset_videos_all/user_squat_1.mp4",
    "with_landmarks": "http://localhost:8000/results/job-id/workspace/squat/visualized_poses_clean/user_squat_1_annotated.mp4"
  },
  "visualization_available": true,
  "visualization_url": "http://localhost:8000/results/job-id/workspace/squat/visualized_poses_clean/user_squat_1_annotated.mp4",
  "reps": [...],
  "feedback": {...}
}
```

**For web app integration:**
- Use `result.videos.with_landmarks` (contains the signed URL if on Supabase)
- Fallback to `result.visualization_url` for backward compatibility
- Check `result.visualization_available` to show/hide video player

## Troubleshooting

### Local testing: "Video not found" error

**Problem:** Visualization video exists but `/results` returns 404

**Solution:**
1. Check that `pipeline_ui_runs/{job_id}/workspace/squat/visualized_poses_clean/` exists
2. Verify FastAPI `/results` mount is working: `curl http://localhost:8000/results/`
3. Check full path: `curl http://localhost:8000/results/{job_id}/workspace/squat/visualized_poses_clean/{video_id}_annotated.mp4`

### Cloud Run: Visualization upload fails

**Problem:** Logs show `Failed to upload visualization to Supabase`

**Solution:**
1. Verify `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set: `gcloud run services describe exevision-modelai --region asia-southeast1`
2. Test Supabase credentials locally:
   ```python
   from supabase import create_client
   client = create_client("https://your-project.supabase.co", "your-key")
   # Should not raise
   ```
3. Check bucket exists: Supabase Dashboard → Storage → `inference-results` should be listed
4. Ensure bucket is **Public** (Settings → Policies)

### Cloud Run: Signed URL is invalid after 1 hour

This is expected. Signed URLs expire after 1 hour. Web app should handle this gracefully:
- Fetch the file immediately after receiving the URL
- Cache locally if needed
- Show user-friendly error if URL is stale

## Environment Variables Reference

| Variable | Local | Cloud Run | Purpose |
|----------|-------|-----------|---------|
| `INFERENCE_API_SECRET` | ✅ Required | ✅ Required | Shared secret with web app |
| `API_PUBLIC_URL` | Optional (default: `http://localhost:8000`) | Optional | Public URL for result links |
| `SUPABASE_URL` | ❌ (omit for local) | ✅ Required for uploads | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ❌ (omit for local) | ✅ Required for uploads | Supabase server key |
| `CORS_ORIGINS` | Optional (default: `*`) | ✅ Recommended | Web app origin |

## See Also

- `.env.example` — Full env var documentation
- `CLAUDE.md` — API deployment & setup details
- `apps/api/pipeline.py` — `_upload_visualization_to_supabase()` implementation
