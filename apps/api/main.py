"""
ExeVision AI — FastAPI Inference Server

Wraps the squat analysis pipeline for consumption by the Next.js web app.

Endpoints
─────────
POST /infer
    Body: { "video_url": str, "job_id": str?, "stages": str[]?, "mode": "filtered"|"unfiltered" }
    Returns: { "job_id": str, "status": "queued" }

GET /jobs/{job_id}
    Returns: { "job_id", "status": "queued|running|done|failed", "result"?, "error"? }

GET /health
    Returns: { "status": "ok", "stages_dir_ok": bool, "models_ok": bool }

Usage
─────
    uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
    # or from apps/api/:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure apps/api/ is on sys.path so `pipeline` is importable when running
# as `uvicorn apps.api.main:app` from the project root.
_API_DIR = Path(__file__).resolve().parent
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, HttpUrl

from pipeline import (
    BILSTM_CKPT,
    FUSION_CKPT,
    SHARED_FACE_MODEL_PATH,
    SHARED_MODEL_PATH,
    STAGES_DIR,
    STGCN_CKPT,
    DEFAULT_STAGES,
    download_video,
    run_pipeline_sync,
)

# ── Auth ───────────────────────────────────────────────────────────────────────
# Shared secret between this server and the Next.js app.
# Set INFERENCE_API_SECRET in both services' environment variables.
_API_SECRET = os.environ.get("INFERENCE_API_SECRET", "")
_bearer = HTTPBearer(auto_error=False)


def _verify_secret(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if not _API_SECRET:
        return  # Secret not configured → open (dev mode only)
    if creds is None or creds.credentials != _API_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API secret")


# ── In-memory job store ────────────────────────────────────────────────────────
# For production, replace with Supabase row updates via the callback_url mechanism.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ExeVision AI Inference API",
    description="Squat form analysis pipeline — wraps MediaPipe + rule-based AQA + neural fusion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class InferRequest(BaseModel):
    video_url: str                          # Supabase signed URL or any direct download URL
    job_id: str | None = None               # Optional; generated if not provided
    stages: list[str] | None = None         # Subset of DEFAULT_STAGES; None = run all
    mode: Literal["filtered", "unfiltered"] = "filtered"
    callback_url: str | None = None         # Optional: POST result here when done


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    queued_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# ── Background task ────────────────────────────────────────────────────────────
def _pipeline_task(job_id: str, video_url: str, stages: list[str], mode: str, callback_url: str | None) -> None:
    """Downloads the video and runs the full pipeline. Runs in a background thread."""
    import asyncio
    import httpx

    _update_job(job_id, status="running", started_at=datetime.now(timezone.utc).isoformat())

    try:
        # Download video to a temp directory
        with tempfile.TemporaryDirectory(prefix=f"exevision_{job_id}_") as tmp:
            tmp_path = Path(tmp)

            # Download (run async download in a new event loop for the thread)
            loop = asyncio.new_event_loop()
            try:
                video_path = loop.run_until_complete(download_video(video_url, tmp_path))
            finally:
                loop.close()

            # Run pipeline stages synchronously
            result = run_pipeline_sync(
                job_id=job_id,
                video_path=video_path,
                stages=stages,
                mode=mode,
            )

        completed_at = datetime.now(timezone.utc).isoformat()
        _update_job(job_id, status="done", result=result, completed_at=completed_at)

        # Fire-and-forget callback to Next.js / Supabase if requested
        if callback_url:
            try:
                with httpx.Client(timeout=10) as client:
                    client.post(callback_url, json={"job_id": job_id, "status": "done", "result": result})
            except Exception:
                pass  # Callback failure must not affect job status

    except Exception as exc:
        completed_at = datetime.now(timezone.utc).isoformat()
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=completed_at,
        )
        if callback_url:
            try:
                with httpx.Client(timeout=10) as client:
                    client.post(callback_url, json={"job_id": job_id, "status": "failed", "error": str(exc)})
            except Exception:
                pass


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/infer", status_code=status.HTTP_202_ACCEPTED, dependencies=[Security(_verify_secret)])
def submit_inference(req: InferRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Accept a video URL and enqueue a pipeline run.
    Returns immediately with a job_id; poll GET /jobs/{job_id} for results.
    """
    job_id = req.job_id or str(uuid.uuid4())
    stages = req.stages or DEFAULT_STAGES
    now = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "queued_at": now,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

    background_tasks.add_task(
        _pipeline_task,
        job_id=job_id,
        video_url=req.video_url,
        stages=stages,
        mode=req.mode,
        callback_url=req.callback_url,
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", dependencies=[Security(_verify_secret)])
def get_job(job_id: str) -> JobStatus:
    """Poll for job status. Once status='done', result contains the full analysis."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatus(**job)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + readiness check. Verifies that all model files and stage scripts exist."""
    models_ok = all(p.exists() for p in [SHARED_MODEL_PATH, SHARED_FACE_MODEL_PATH, BILSTM_CKPT, STGCN_CKPT, FUSION_CKPT])
    stages_ok = STAGES_DIR.exists() and all(
        (STAGES_DIR / s).exists()
        for s in [
            "extract_selected_features.py",
            "classify_views.py",
            "temporal_segmentation.py",
            "scoring.py",
            "neural_fusion_inference.py",
        ]
    )
    return {
        "status": "ok" if (models_ok and stages_ok) else "degraded",
        "stages_dir_ok": stages_ok,
        "models_ok": models_ok,
        "missing_models": [
            str(p) for p in [SHARED_MODEL_PATH, SHARED_FACE_MODEL_PATH, BILSTM_CKPT, STGCN_CKPT, FUSION_CKPT]
            if not p.exists()
        ],
    }


# ── Direct execution (fallback; prefer uvicorn CLI) ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    _port = int(os.environ.get("PORT", 8000))
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=_port, reload=False)
