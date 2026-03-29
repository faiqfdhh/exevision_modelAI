# GCR Deployment — ExeVision AI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize and deploy the `exevision_modelAI` FastAPI server to Google Cloud Run so the Next.js web app can switch between this cloud endpoint and a local server.

**Architecture:** A single `Dockerfile` at the repo root bundles the FastAPI server (`apps/api/`), all pipeline stages (`core/exevision/stages/`), and all model artifacts (`models/`) into one container image. Google Cloud Run serves this image. All configuration (secret, CORS origins, PORT) is injected via environment variables at runtime.

**Tech Stack:** Python 3.10, FastAPI, uvicorn, Docker, Google Cloud Run (`gcloud` CLI), GitHub Actions (optional CI/CD)

---

## Role & Context for the Executing Agent

You are implementing GCR deployment for a Python FastAPI server that wraps an AI video-analysis pipeline (MediaPipe + PyTorch). Here is what you need to know before you start:

> **Execution boundary:** Implement this plan in the AI backend repo. The web-app backend toggle integration is handled in a separate plan executed in the Next.js repo.

**Repo structure relevant to this task:**
```
exevision_modelAI/
├── apps/api/
│   ├── main.py          # FastAPI app (entry point)
│   ├── pipeline.py      # Pipeline runner (spawns stage subprocesses)
│   └── requirements.txt # API-only deps (fastapi, uvicorn, httpx, pydantic)
├── core/exevision/
│   ├── stages/          # Stage scripts run as subprocesses by pipeline.py
│   ├── neural/          # PyTorch model classes
│   ├── training/        # Training scripts (NOT needed at runtime)
│   ├── feedback/        # Feedback engine (needed at runtime)
│   └── config/          # Exercise configs, feedback templates
├── models/              # Model artifacts (~67MB total, ALL needed at runtime)
│   ├── pose_landmarker_heavy.task  # 30MB — MediaPipe model
│   ├── bilstm_finetuned.pt         # BiLSTM checkpoint
│   ├── stgcn_finetuned.pt          # ST-GCN checkpoint
│   ├── fusion_layer.pt             # Fusion checkpoint
│   └── blaze_face_short_range.tflite
├── training_dataset/    # NOT needed at runtime — exclude from image
└── _hidden_legacy/      # NOT needed at runtime — exclude from image
```

**Critical runtime dependencies:**
- `pipeline.py` runs stage scripts as **subprocesses** via `subprocess.run()`. The container must have Python available on PATH and all stage script imports must work.
- Stage scripts import from `core.exevision.*` — the container's working directory must be the repo root so Python can resolve these imports.
- The server reads model paths from `EXEVISION_MODEL_PATH` and `EXEVISION_FACE_MODEL_PATH` environment variables (set in `_run_stage()` in `pipeline.py`).
- The server reads `INFERENCE_API_SECRET` for auth, `CORS_ORIGINS` for CORS, and `PORT` for the listen port (standard GCR env var).
- `apps/api/main.py` inserts `apps/api/` into `sys.path` at startup so `from pipeline import ...` works. Do NOT change this.

**What this plan does NOT include:**
- Web app changes (handled in a separate plan)
- Training workflow or annotation tooling
- Desktop UI changes

**Definition of done for this plan:**
- `docker build` succeeds locally
- `docker run` starts the server and `GET /health` returns `{"status": "ok"}`
- `gcloud run deploy` succeeds and `/health` returns `{"status": "ok"}` at the GCR URL

---

## File Map (files to create or modify)

| File | Action | Purpose |
|------|--------|---------|
| `Dockerfile` | **Create** | Multi-stage build: install deps, copy source + models |
| `.dockerignore` | **Create** | Exclude training data, legacy, notebooks, dev artifacts |
| `requirements-runtime.txt` | **Create** | All runtime Python deps (FastAPI + pipeline deps) |
| `apps/api/main.py` | **Modify** | Read `PORT` env var (GCR standard) instead of hardcoded port |
| `cloudbuild.yaml` | **Create** | Google Cloud Build config for CI/CD (optional but recommended) |
| `.github/workflows/deploy-gcr.yml` | **Create** | GitHub Actions workflow — build + push + deploy on push to main |
| `CLAUDE.md` | **Update** | Add GCR deployment section to Quick Start and §9 risks |
| `CHANGELOG.md` | **Update** | Log this session's changes |

---

## Task 1: Create Runtime Requirements File

**Files:**
- Create: `requirements-runtime.txt`

Context: `apps/api/requirements.txt` only lists API deps. The container also needs `mediapipe`, `opencv-python-headless`, `numpy`, `scipy` for stage scripts. `torch`/`torchvision` will be installed separately in the Dockerfile as CPU-only wheels.

- [ ] **Step 1.1: Audit imports in stage scripts**

Run from repo root:
```bash
grep -rh "^import\|^from" core/exevision/stages/ | sort -u
```
Note which packages need adding beyond what's in `apps/api/requirements.txt`.

- [ ] **Step 1.2: Create `requirements-runtime.txt`**

Create `requirements-runtime.txt` at repo root with this exact content:

```txt
# API server
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
pydantic>=2.7.0

# Pipeline stage dependencies
mediapipe>=0.10.0
opencv-python-headless>=4.9.0  # headless: no GUI, smaller image
numpy>=1.26.0
scipy>=1.13.0
```

> **Note on torch:** Keep `torch` and `torchvision` out of `requirements-runtime.txt`. Install them in the Dockerfile using `--index-url https://download.pytorch.org/whl/cpu` to force CPU wheels and avoid duplicate resolver passes.

- [ ] **Step 1.3: Commit**

```bash
git add requirements-runtime.txt
git commit -m "feat: add consolidated runtime requirements for container deployment"
```

---

## Task 2: Create `.dockerignore`

**Files:**
- Create: `.dockerignore`

Exclude heavy non-runtime directories to keep build context small and fast.

- [ ] **Step 2.1: Create `.dockerignore`**

Create `.dockerignore` at repo root:

```dockerignore
# Dev artifacts and training data (not needed at runtime)
training_dataset/
_hidden_legacy/
squat/
pipeline_ui_runs/

# Model training checkpoints (keep only finetuned inference models)
models/bilstm_pretrained.pt
models/stgcn_pretrained.pt
models/stgcn_pretrained_encoder.pt

# Python cache
__pycache__/
**/__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Virtual environments
.venv/
venv/
env/

# IDE and OS artifacts
.idea/
.vscode/
*.DS_Store
*.log

# Git
.git/
.gitignore

# Docs and test artifacts
docs/
tests/
*.md
```

> **Important:** `models/pose_landmarker_heavy.task`, `models/blaze_face_short_range.tflite`, `models/bilstm_finetuned.pt`, `models/stgcn_finetuned.pt`, `models/fusion_layer.pt` are **NOT** excluded. They are required at runtime.

- [ ] **Step 2.2: Verify expected files are included**

```bash
# Check what would be sent to Docker (requires docker installed)
docker build --no-cache --dry-run . 2>/dev/null | head -50
# OR: manually verify models directory is intact
ls models/
```

Expected output includes: `bilstm_finetuned.pt`, `blaze_face_short_range.tflite`, `fusion_layer.pt`, `pose_landmarker_heavy.task`, `stgcn_finetuned.pt`

- [ ] **Step 2.3: Commit**

```bash
git add .dockerignore
git commit -m "feat: add .dockerignore for container build"
```

---

## Task 3: Create `Dockerfile`

**Files:**
- Create: `Dockerfile`

The container must:
1. Start from an official Python 3.10 slim base
2. Install system deps for MediaPipe and OpenCV (libgomp, etc.)
3. Install Python runtime deps from `requirements-runtime.txt`
4. Copy the entire repo (minus dockerignore exclusions) into `/app`
5. Set working directory to `/app` (repo root) — critical for relative imports
6. Expose the port and launch uvicorn

- [ ] **Step 3.1: Create `Dockerfile`**

Create `Dockerfile` at repo root:

```dockerfile
# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System packages needed by MediaPipe, OpenCV, and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements-runtime.txt .

# Install CPU-only torch first (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir \
    torch>=2.3.0 \
    torchvision>=0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining runtime deps (requirements-runtime.txt excludes torch/torchvision)
RUN pip install --no-cache-dir -r requirements-runtime.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy source tree (training_dataset/, _hidden_legacy/, etc. excluded by .dockerignore)
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
# GCR injects PORT; default to 8000 for local docker run
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Model paths (resolved relative to /app which is the repo root)
ENV EXEVISION_MODEL_PATH=/app/models/pose_landmarker_heavy.task
ENV EXEVISION_FACE_MODEL_PATH=/app/models/blaze_face_short_range.tflite

# ── Launch ────────────────────────────────────────────────────────────────────
# Working directory /app = repo root. This ensures `core.exevision.*` imports work
# both in the FastAPI app and in subprocess-spawned stage scripts.
CMD uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT
```

- [ ] **Step 3.2: Build the image locally to verify it compiles**

```bash
docker build -t exevision-api:local .
```

Expected: build completes without error. It will take several minutes on first run (torch download ~1GB). Subsequent builds use layer cache.

If you see `ImportError` for a missing package, add it to `requirements-runtime.txt` and rebuild.

- [ ] **Step 3.3: Run the container locally and test health endpoint**

```bash
docker run --rm -p 8000:8000 \
  -e INFERENCE_API_SECRET=test-secret \
  exevision-api:local
```

In a second terminal:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "stages_dir_ok": true, "models_ok": true}
```

If `stages_dir_ok` or `models_ok` is `false`, check that:
- `core/exevision/stages/` files are present in the image (`docker exec <container_id> ls /app/core/exevision/stages/`)
- Model files are present (`docker exec <container_id> ls /app/models/`)

- [ ] **Step 3.4: Test a real inference call (optional but recommended)**

```bash
curl -X POST http://localhost:8000/infer \
  -H "Authorization: Bearer test-secret" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "<any-valid-video-url>", "job_id": "local-test-1"}'
```

Expected: `{"job_id": "local-test-1", "status": "queued"}`

- [ ] **Step 3.5: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for GCR deployment"
```

---

## Task 4: Update `apps/api/main.py` — PORT Env Var

**Files:**
- Modify: `apps/api/main.py` (startup only — no logic changes)

Google Cloud Run injects `PORT` as an environment variable. The `Dockerfile` already passes it to uvicorn via `CMD uvicorn ... --port $PORT`. However, if anyone runs `python apps/api/main.py` directly (not via uvicorn), it should also respect `PORT`.

Currently `main.py` does not have a `__main__` block. Add one so direct execution also works on GCR.

- [ ] **Step 4.1: Add `__main__` block to `apps/api/main.py`**

Open `apps/api/main.py`. At the very end of the file (after all route definitions), add:

```python
# ── Direct execution (fallback; prefer uvicorn CLI) ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    _port = int(os.environ.get("PORT", 8000))
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=_port, reload=False)
```

- [ ] **Step 4.2: Verify no existing `__main__` block**

Before adding, run:
```bash
grep -n "__main__" apps/api/main.py
```
Expected: no output (none exists). If one exists, update it rather than duplicating.

- [ ] **Step 4.3: Commit**

```bash
git add apps/api/main.py
git commit -m "feat: respect PORT env var for direct execution (GCR standard)"
```

---

## Task 5: Deploy to Google Cloud Run

**Prerequisites (agent must confirm these exist before running):**
- Google Cloud SDK (`gcloud`) installed and authenticated: `gcloud auth login`
- A GCP project exists: `gcloud projects list`
- Artifact Registry API enabled: `gcloud services enable artifactregistry.googleapis.com`
- Cloud Run API enabled: `gcloud services enable run.googleapis.com`
- Cloud Build API enabled: `gcloud services enable cloudbuild.googleapis.com`

**Files:**
- No new files for this task. All steps are CLI commands.

- [ ] **Step 5.1: Set shell variables (replace placeholders)**

```bash
export GCP_PROJECT="your-gcp-project-id"        # e.g. exevision-ai-prod
export GCR_REGION="asia-southeast1"              # or us-central1, europe-west1
export IMAGE_NAME="exevision-api"
export SERVICE_NAME="exevision-api"
export REGISTRY="${GCR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${IMAGE_NAME}"
```

- [ ] **Step 5.2: Create Artifact Registry repository**

```bash
gcloud artifacts repositories create $IMAGE_NAME \
  --repository-format=docker \
  --location=$GCR_REGION \
  --project=$GCP_PROJECT
```

Expected: `Created repository [...]`

- [ ] **Step 5.3: Build and push image via Cloud Build**

```bash
gcloud builds submit \
  --tag "${REGISTRY}/${IMAGE_NAME}:latest" \
  --project=$GCP_PROJECT \
  .
```

This uploads the build context to GCP and builds there (avoids slow local upload of model files). Expected: `SUCCESS` at the end of the output after several minutes.

- [ ] **Step 5.4: Deploy to Cloud Run**

```bash
gcloud run deploy $SERVICE_NAME \
  --image="${REGISTRY}/${IMAGE_NAME}:latest" \
  --platform=managed \
  --region=$GCR_REGION \
  --project=$GCP_PROJECT \
  --allow-unauthenticated \
  --memory=4Gi \
  --cpu=2 \
  --timeout=600 \
  --concurrency=1 \
  --set-env-vars="INFERENCE_API_SECRET=<your-secret>,CORS_ORIGINS=https://your-web-app.vercel.app" \
  --min-instances=0 \
  --max-instances=3
```

**Parameter notes:**
- `--memory=4Gi`: MediaPipe + torch need ~3GB peak. 4GB gives headroom.
- `--timeout=600`: Pipeline can take 2-5 minutes for a full video.
- `--concurrency=1`: Each instance handles one video at a time (pipeline is single-threaded).
- `--min-instances=0`: Scale to zero when idle (cost-saving). Expect cold starts of 30-60s.
- `--allow-unauthenticated`: Auth is handled by `INFERENCE_API_SECRET` in the app itself.

Expected output includes the service URL: `Service URL: https://exevision-api-xxxx-as.a.run.app`

- [ ] **Step 5.5: Verify health endpoint on GCR**

```bash
GCR_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$GCR_REGION \
  --project=$GCP_PROJECT \
  --format="value(status.url)")

curl "${GCR_URL}/health"
```

Expected:
```json
{"status": "ok", "stages_dir_ok": true, "models_ok": true}
```

If `status` is `"degraded"` or `"error"`, check Cloud Run logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" \
  --project=$GCP_PROJECT \
  --limit=50 \
  --format="value(textPayload)"
```

- [ ] **Step 5.6: Note the GCR URL**

Record the URL: `https://exevision-api-xxxx-as.a.run.app`

This is `NEXT_PUBLIC_GCR_BACKEND_URL` needed by the web app in the next plan.

---

## Task 6: Create `cloudbuild.yaml` (CI/CD for Future Deploys)

**Files:**
- Create: `cloudbuild.yaml`

So future code changes can be re-deployed without manually running CLI commands.

- [ ] **Step 6.1: Create `cloudbuild.yaml`**

Create `cloudbuild.yaml` at repo root:

```yaml
# cloudbuild.yaml — Triggered on push to main branch
# Usage: gcloud builds submit OR connect via Cloud Build trigger in GCP console
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_IMAGE_NAME}/${_IMAGE_NAME}:$COMMIT_SHA'
      - '-t'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_IMAGE_NAME}/${_IMAGE_NAME}:latest'
      - '.'

  # Push both tags
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '--all-tags'
      - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_IMAGE_NAME}/${_IMAGE_NAME}'

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - '${_SERVICE_NAME}'
      - '--image=${_REGION}-docker.pkg.dev/$PROJECT_ID/${_IMAGE_NAME}/${_IMAGE_NAME}:$COMMIT_SHA'
      - '--region=${_REGION}'
      - '--platform=managed'

substitutions:
  _REGION: asia-southeast1
  _IMAGE_NAME: exevision-api
  _SERVICE_NAME: exevision-api

options:
  logging: CLOUD_LOGGING_ONLY
```

- [ ] **Step 6.2: Commit**

```bash
git add cloudbuild.yaml
git commit -m "feat: add cloudbuild.yaml for automated GCR deploys"
```

---

## Task 7: Update `CLAUDE.md` and `CHANGELOG.md`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 7.1: Update `CLAUDE.md` Quick Start section**

In `CLAUDE.md`, add under `## Quick Start` a new subsection:

```markdown
### Deploy to Google Cloud Run
```bash
# One-time setup (set your project/region)
export GCP_PROJECT="your-gcp-project-id"
export GCR_REGION="asia-southeast1"
export IMAGE_NAME="exevision-api"

# Build and deploy
gcloud builds submit --tag "${GCR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${IMAGE_NAME}/${IMAGE_NAME}:latest" .
gcloud run deploy exevision-api \
  --image="${GCR_REGION}-docker.pkg.dev/${GCP_PROJECT}/${IMAGE_NAME}/${IMAGE_NAME}:latest" \
  --region=$GCR_REGION --memory=4Gi --cpu=2 --timeout=600 --concurrency=1
```
```

Also add to `## 9) Technical Risks & Current Status` under Active Issues:

```markdown
11. **GCR cold starts (~30-60s):** `--min-instances=0` means first request after idle period triggers a cold start. The web app should show a loading state and retry for up to 90s. Set `--min-instances=1` to eliminate cold starts (incurs constant cost ~$10/month for 1 idle instance).
```

- [ ] **Step 7.2: Update `CHANGELOG.md`**

Prepend a new entry at the top of `CHANGELOG.md`:

```markdown
### Session 2026-03-29 — GCR Deployment Infrastructure

**Focus:** Containerize the FastAPI inference server for Google Cloud Run deployment.

**What was done:**
1. Created `requirements-runtime.txt` — consolidated runtime deps (fastapi, mediapipe, opencv-python-headless, numpy, scipy)
2. Created `Dockerfile` — multi-stage image baking models into container; reads `PORT`/`INFERENCE_API_SECRET`/`CORS_ORIGINS` from env
3. Created `.dockerignore` — excludes training_dataset/, _hidden_legacy/, pretrain checkpoints, and dev artifacts
4. Added `__main__` block to `apps/api/main.py` to respect `PORT` env var on direct execution
5. Created `cloudbuild.yaml` — automated GCR build + deploy triggered on push to main
6. Deployed to `https://exevision-api-xxxx-as.a.run.app` (GCR region: asia-southeast1)

**New env vars required on GCR:**
- `INFERENCE_API_SECRET` — shared secret for auth
- `CORS_ORIGINS` — comma-separated allowed origins (e.g. `https://your-app.vercel.app`)
```

- [ ] **Step 7.3: Commit**

```bash
git add CLAUDE.md CHANGELOG.md
git commit -m "docs: document GCR deployment in CLAUDE.md and CHANGELOG.md"
```

---

## Self-Review Checklist

### Spec Coverage
- ✅ Dockerfile created (Task 3)
- ✅ .dockerignore created (Task 2)
- ✅ Runtime requirements (Task 1)
- ✅ GCR deploy commands (Task 5)
- ✅ PORT env var respected (Task 4)
- ✅ CI/CD pipeline (Task 6)
- ✅ Docs updated (Task 7)

### Critical Reminders for the Executing Agent
1. **Working directory must be `/app` (repo root)** — subprocess stage scripts resolve `core.exevision.*` relative to CWD
2. **Torch must be CPU-only** — GCR has no GPU; use `--index-url https://download.pytorch.org/whl/cpu`
3. **`opencv-python-headless`** not `opencv-python` — headless avoids pulling GUI libs into the container
4. **`INFERENCE_API_SECRET` must be set as GCR env var** — never hardcode it
5. **After deploy, note the GCR URL** — needed for the web app plan (`NEXT_PUBLIC_GCR_BACKEND_URL`)
