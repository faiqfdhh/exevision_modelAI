# ExeVision Web App — Integration Agent Prompt

> Pass this entire file as a task to a coding agent working in the Next.js + Supabase repo.

---

## Context

You are integrating a **Next.js + Supabase** web application with an external **Python FastAPI inference server** that performs AI-based squat form analysis from uploaded videos.

The Python inference server (`exevision_modelAI/apps/api/`) is already built and running separately. Your job is to wire the web app to it: upload handling, job tracking, results display.

**Do not modify the Python inference server or any pipeline scripts.**

---

## What the Inference Server Does

The Python server runs a 5-stage ML pipeline on a squat video:
1. Pose extraction (MediaPipe)
2. View classification (front/side/back)
3. Temporal segmentation (rep detection: eccentric/isometric/concentric phases)
4. Rule-based scoring (heuristic 0–100 score per rep)
5. Neural fusion scoring (BiLSTM + ST-GCN, residual correction on top of heuristic)

Processing time: **30–120 seconds** per video depending on length and hardware.

---

## Inference API Contract

**Base URL:** `INFERENCE_API_URL` env var (e.g. `http://localhost:8000` in dev)

**Authentication:** `Authorization: Bearer {INFERENCE_API_SECRET}` header on every request.

### `POST /infer`
Submit a video for analysis.

**Request body:**
```json
{
  "video_url": "https://...supabase.co/storage/v1/object/sign/...",
  "job_id": "uuid-v4-string",
  "stages": ["extract_selected_features","classify_views","temporal_segmentation","scoring","neural_fusion"],
  "mode": "filtered",
  "callback_url": "https://your-app.com/api/inference/callback"
}
```

- `video_url` — Supabase Storage **signed URL** (must be downloadable by the Python server; min 300s expiry)
- `job_id` — supply the Supabase `inference_jobs.id` UUID so results map back to the row
- `stages` — optional; omit to run all 5 stages
- `mode` — `"filtered"` (default, quality-gated) or `"unfiltered"` (faster, lighter)
- `callback_url` — optional; the server will POST `{ job_id, status, result }` here when done

**Response `202`:**
```json
{ "job_id": "...", "status": "queued" }
```

### `GET /jobs/{job_id}`
Poll for status.

**Response:**
```json
{
  "job_id": "...",
  "status": "queued | running | done | failed",
  "queued_at": "2026-03-28T15:30:00Z",
  "started_at": "2026-03-28T15:30:05Z",
  "completed_at": "2026-03-28T15:31:20Z",
  "result": { /* see Result Schema below */ },
  "error": null
}
```

### `GET /health`
Liveness check — call this before showing the analysis UI.

```json
{ "status": "ok | degraded", "models_ok": true, "stages_dir_ok": true, "missing_models": [] }
```

---

## Result Schema

When `status === "done"`, `result` contains:

```typescript
interface AnalysisResult {
  video_id: string;           // filename stem (e.g. "user_squat_1")
  // _display_view() in pipeline.py collapses raw labels before the API response:
  //   front / back         → "straight"
  //   front_side/back_side → "diagonal"
  // Raw labels are kept internally for view-specific scoring; API always emits display labels.
  view: "side" | "diagonal" | "straight" | "unknown";
  quality: "excellent" | "good" | "fair" | "poor" | "unknown";
  rep_count: number;
  overall_heuristic_score: number;   // 0–100 rule-based score averaged over reps
  overall_neural_score: number | null; // 0–100 fused score averaged over reps; null if neural stage skipped
  overall_bilstm_score: number | null; // 0–100 temporal judge (avg of smoothness+control across reps)
  overall_stgcn_score: number | null;  // 0–100 spatial judge (avg of depth+lean+knee across reps)
  neural_available: boolean;
  any_anchor_corrections: boolean;     // true if any rep had a broken neural anchor (see anchor_correction_applied)

  reps: RepResult[];
}

interface RepResult {
  rep_id: number;
  start_frame: number;
  end_frame: number;
  duration_seconds: number;

  heuristic_score: number;           // Rule-based score 0–100 ("Rule-Based Judge")
  neural_score: number | null;       // Fused score 0–100 ("Overall Score")
  neural_score_raw: number | null;   // Only set when anchor_correction_applied=true (original broken value)
  neural_score_pre_clamp: number | null;
  residual: number | null;           // Neural correction applied on top of heuristic
  anchor_correction_applied: boolean; // true if |neural - heuristic| > 40 was corrected

  bilstm_score: number | null;      // 0–100 temporal judge — avg(smoothness, control)
  stgcn_score: number | null;       // 0–100 spatial judge — avg(depth, forward_lean, knee_tracking)

  metrics: {
    knee_valgus: number | null;      // Ratio (< 1.0 = inward collapse risk)
    forward_lean: number | null;     // Degrees from vertical
    min_knee_angle: number | null;   // Degrees (lower = deeper squat)
    squat_depth: number | null;      // Normalized hip displacement (> 0 = below parallel)
    below_parallel: boolean | null;
  };

  metric_scores: {
    knee_valgus?: number;            // 0–100 component score
    forward_lean?: number;
    depth?: number;
    squat_depth?: number;
  };

  sub_scores: {                      // Only present when neural_available=true
    smoothness: number;              // BiLSTM head — temporal smoothness 0–100
    control: number;                 // BiLSTM head — movement control 0–100
    depth: number;                   // ST-GCN head — squat depth score 0–100
    forward_lean: number;            // ST-GCN head — trunk lean score 0–100
    knee_tracking: number;           // ST-GCN head — knee alignment 0–100
  } | null;

  safety_clamps: string[];           // Reasons a safety cap was applied, e.g. ["knee_valgus_severity>=2"]
}
```

### Three-Judge Display Model

The scoring system uses three independent "judges" whose outputs are fused into the overall neural score:

| Judge | Field | Sub-scores | What it evaluates |
|-------|-------|-----------|-------------------|
| **BiLSTM** (Temporal) | `bilstm_score` | smoothness, control | Was the movement smooth and controlled over time? |
| **ST-GCN** (Spatial) | `stgcn_score` | depth, forward_lean, knee_tracking | Was the body positioned correctly at each point? |
| **Heuristic** (Rules) | `heuristic_score` | metric_scores (knee_valgus, forward_lean, depth, squat_depth) | Do the exact joint angles meet biomechanical thresholds? |

**Recommended UI layout:**
1. **Primary:** Overall fusion score (`neural_score`) — large gauge/number
2. **Three-judge panel:** BiLSTM | ST-GCN | Heuristic — three equal score cards
3. **Expandable detail:** Sub-scores under each judge card
4. **Bottom:** Biomechanical metrics with target ranges

---

## Supabase Schema to Create

Run these SQL migrations:

```sql
-- Storage bucket for squat videos
insert into storage.buckets (id, name, public) values ('squat-videos', 'squat-videos', false);

-- RLS: only the owner can upload / read their own videos
create policy "owner upload" on storage.objects for insert
  with check (bucket_id = 'squat-videos' and auth.uid()::text = (storage.foldername(name))[1]);

create policy "owner read" on storage.objects for select
  using (bucket_id = 'squat-videos' and auth.uid()::text = (storage.foldername(name))[1]);

-- Inference jobs table
create table inference_jobs (
  id               uuid primary key default gen_random_uuid(),
  user_id          uuid references auth.users not null,
  video_path       text not null,              -- Storage path: {user_id}/{filename}
  status           text not null default 'queued'
                   check (status in ('queued','running','done','failed')),
  mode             text not null default 'filtered',
  created_at       timestamptz default now(),
  started_at       timestamptz,
  completed_at     timestamptz,
  result_json      jsonb,                      -- Full AnalysisResult when done
  error_message    text
);

-- RLS: users see only their own jobs
alter table inference_jobs enable row level security;
create policy "owner all" on inference_jobs for all
  using (user_id = auth.uid()) with check (user_id = auth.uid());
```

---

## Environment Variables Required

Add these to your `.env.local` and Vercel / hosting env:

```env
# Python inference server
INFERENCE_API_URL=http://localhost:8000
INFERENCE_API_SECRET=your-shared-secret-here   # same value on the Python server

# Supabase (already configured)
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...                   # server-only, for admin DB writes
```

---

## Next.js Routes to Create

### `POST /api/inference/analyze`

Client calls this to start a new analysis run.

**What it does:**
1. Validate authenticated user (check Supabase session)
2. Create a signed upload URL for Supabase Storage at `{user_id}/{uuid}.mp4`
3. **Wait** for the client to upload the video to Storage (client uploads directly)
4. Create an `inference_jobs` row (`status: 'queued'`, `video_path`)
5. Generate a Supabase **signed download URL** (300s expiry) for the Python server
6. `POST {INFERENCE_API_URL}/infer` with:
   - `video_url` = signed download URL
   - `job_id` = the UUID from the DB row
   - `callback_url` = `{NEXTAUTH_URL}/api/inference/callback`
7. Return `{ job_id, upload_url }` to the client

```typescript
// Pseudo-code
// POST /api/inference/analyze
export async function POST(req: Request) {
  const session = await getServerSession()  // or Supabase auth
  const { filename, mode } = await req.json()

  const jobId = crypto.randomUUID()
  const storagePath = `${session.user.id}/${jobId}${ext}`

  // Create DB row
  await supabaseAdmin
    .from('inference_jobs')
    .insert({ id: jobId, user_id: session.user.id, video_path: storagePath, mode })

  // Get upload URL for client-side direct upload
  const { data: uploadData } = await supabaseAdmin.storage
    .from('squat-videos')
    .createSignedUploadUrl(storagePath)

  return Response.json({ job_id: jobId, upload_url: uploadData.signedUrl, path: storagePath })
}
```

### `POST /api/inference/submit`

Called by the client **after** the video has been uploaded to Storage.

**What it does:**
1. Generate signed download URL for the video
2. Call `POST {INFERENCE_API_URL}/infer` with `job_id`, `video_url`, `callback_url`
3. Update DB row to confirm submission
4. Return `{ job_id }`

### `GET /api/inference/status/[jobId]`

Client polls this every 3 seconds.

**What it does:**
1. Read `inference_jobs` row from Supabase (by `id`, RLS enforces ownership)
2. If `status !== 'done' && status !== 'failed'`, optionally also poll `GET {INFERENCE_API_URL}/jobs/{jobId}` and sync status to DB
3. Return `{ job_id, status, result_json?, error_message? }`

### `POST /api/inference/callback`

The Python server calls this when a job completes (via `callback_url`).

**What it does:**
1. Verify the request comes from the inference server (check `Authorization: Bearer {INFERENCE_API_SECRET}`)
2. Update `inference_jobs` row:
   - `status = payload.status` ('done' or 'failed')
   - `result_json = payload.result`
   - `error_message = payload.error`
   - `completed_at = now()`
3. Return `200 OK`

```typescript
// POST /api/inference/callback
export async function POST(req: Request) {
  const authHeader = req.headers.get('authorization')
  if (authHeader !== `Bearer ${process.env.INFERENCE_API_SECRET}`) {
    return new Response('Unauthorized', { status: 401 })
  }
  const { job_id, status, result, error } = await req.json()
  await supabaseAdmin.from('inference_jobs').update({
    status,
    result_json: result ?? null,
    error_message: error ?? null,
    completed_at: new Date().toISOString(),
    ...(status === 'running' ? { started_at: new Date().toISOString() } : {}),
  }).eq('id', job_id)
  return new Response('OK')
}
```

---

## UI Flow to Implement

### Page: `/analyze` (or `/dashboard/analyze`)

**Step 1 — Upload**
- Drag-and-drop or file picker (`.mp4`, `.mov`, `.avi`)
- Show video preview
- Mode selector: "Quality (filtered)" / "Fast (unfiltered)"
- "Analyze" button

**On submit:**
1. `POST /api/inference/analyze` → get `{ job_id, upload_url, path }`
2. Upload file directly to Supabase Storage via the signed upload URL (PUT request with the video binary)
3. `POST /api/inference/submit` with `{ job_id, path }` → triggers inference
4. Redirect to or show `/analyze/results/[jobId]`

**Step 2 — Processing (poll page)**
- Show animated skeleton/spinner
- Poll `GET /api/inference/status/[jobId]` every 3 seconds
- Show current stage in progress if possible (parse Python server logs — optional)
- Once `status === 'done'` → show results inline or redirect

**Step 3 — Results**

Show a results dashboard with:

```
┌─────────────────────────────────────────────────────┐
│  Overall Score                                       │
│  Neural: 72/100  │  Heuristic: 74/100               │
│  View: Side  │  Quality: Good  │  3 reps detected   │
├─────────────────────────────────────────────────────┤
│  Rep 1   Rep 2   Rep 3                              │  ← tab selector
├─────────────────────────────────────────────────────┤
│  Rep 2 — Neural: 68/100  Heuristic: 72/100         │
│                                                      │
│  Sub-scores (neural):                               │
│  Smoothness    ████████░░  80                       │
│  Control       ███████░░░  70                       │
│  Depth         ██████░░░░  60                       │
│  Forward Lean  ██████░░░░  62                       │
│  Knee Tracking █████████░  88                       │
│                                                      │
│  Biomechanical Metrics:                             │
│  • Below parallel: ✓                               │
│  • Forward lean: 22.5°  (target < 30°)             │
│  • Knee valgus ratio: 0.94  (good ≥ 0.95)          │
│  • Squat depth: 0.12                               │
│                                                      │
│  ⚠ Safety caps applied: none                       │
└─────────────────────────────────────────────────────┘
```

**Score color coding:**
- ≥ 80: green
- 60–79: yellow/amber
- < 60: red

**Safety clamp display:**
If `safety_clamps` is non-empty, show a warning banner per rep:
- `"knee_valgus_severity>=2"` → "⚠ Score capped at 60 due to significant knee cave"
- `"forward_lean_severity>=2"` → "⚠ Score capped at 65 due to excessive forward lean"
- `"insufficient_squat_depth_severity>=3"` → "⚠ Score capped at 50 due to insufficient depth"

### Page: `/dashboard` (history)

List all `inference_jobs` for the authenticated user:
- Date, filename, overall neural score, status badge
- Click row → `/analyze/results/[jobId]`

---

## Key Implementation Notes

**Video upload size:**
- Configure `next.config.ts` body size limit for the API routes (videos can be 50–200MB)
- Use Supabase direct signed upload URL (client-to-storage) to avoid routing the binary through Next.js server

**Polling vs callback:**
- Implement both: callback updates DB immediately, polling is the fallback
- Poll from client at 3s interval with exponential backoff after 60s
- Stop polling at `status === 'done' || status === 'failed'`

**Signed URL expiry:**
- Upload URLs: 120 seconds (user needs to upload promptly)
- Download URLs for inference server: **minimum 300 seconds** (pipeline can take up to 2 minutes)
- View/display URLs: 3600 seconds

**Error handling:**
- `status === 'failed'`: show `error_message` in a red alert; offer "Try again" button
- `quality === 'poor'`: show a warning: "Low pose quality detected. Try a well-lit video filmed from 2–3m away."
- `view === 'unknown'`: show: "Camera angle could not be determined. Use a side or front view for best results."
- `neural_available === false`: show heuristic score only with a note that neural scoring was skipped

**TypeScript types:**
Create `types/analysis.ts` with the full `AnalysisResult` and `RepResult` interfaces from the Result Schema section above.

---

## Recommended File Structure

```
app/
├── analyze/
│   ├── page.tsx                    # Upload UI (Step 1)
│   └── results/[jobId]/
│       └── page.tsx                # Results display (Steps 2+3)
├── dashboard/
│   └── page.tsx                    # Job history list
└── api/
    └── inference/
        ├── analyze/route.ts         # POST: create job + get upload URL
        ├── submit/route.ts          # POST: trigger inference after upload
        ├── status/[jobId]/route.ts  # GET: poll job status
        └── callback/route.ts        # POST: receive result from Python server

lib/
└── inference-client.ts             # Typed wrapper for all inference API calls

types/
└── analysis.ts                     # AnalysisResult, RepResult interfaces

components/
└── analysis/
    ├── ScoreGauge.tsx               # Circular score display
    ├── RepSelector.tsx              # Tab-based rep navigation
    ├── SubScoreBar.tsx              # Labelled progress bar for sub-scores
    ├── MetricsTable.tsx             # Biomechanical metrics display
    └── SafetyClampBanner.tsx        # Warning for capped scores
```

---

## Testing the Integration

1. Start Python server: `cd exevision_modelAI && uvicorn apps.api.main:app --port 8000`
2. Check health: `curl http://localhost:8000/health`
3. In Next.js `.env.local`:
   ```env
   INFERENCE_API_URL=http://localhost:8000
   INFERENCE_API_SECRET=dev-secret
   ```
4. On the Python server: `INFERENCE_API_SECRET=dev-secret uvicorn apps.api.main:app --port 8000`
5. Upload a test MP4 via the web UI and verify the job flow end-to-end

A successful end-to-end flow produces a `pipeline_ui_runs/{job_id}/` folder in the inference repo
with the full workspace, stage output JSONs, and logs — inspect these for debugging.
