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
from dotenv import load_dotenv
load_dotenv()                    # .env (shared/base config)
load_dotenv(".env.local", override=True)  # .env.local overrides .env
import logging
import sys
from pathlib import Path
import re

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
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
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
    RUNS_ROOT,
)


# ── Auth ───────────────────────────────────────────────────────────────────────
# Shared secret between this server and the Next.js app.
# Set INFERENCE_API_SECRET in both services' environment variables.
_API_SECRET = os.environ.get("INFERENCE_API_SECRET", "")
_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


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


def _sanitize_error(msg: str, max_len: int = 200) -> str:
    """Strip file paths and truncate for safe user-facing errors."""
    msg = re.sub(r'[A-Z]:(?:\\[^\\\s]+)+', '<path>', msg)
    msg = re.sub(r'(?:/[^/\s]+)+/', '<path>/', msg)
    if len(msg) > max_len:
        msg = msg[:max_len] + '…'
    return msg


def _normalize_stage_selection(requested_stages: list[str] | None) -> list[str]:
    """
    Validate and normalize requested stages.

    Rules:
    - Unknown stage names are rejected.
    - Dependencies are auto-included and execution is forced into canonical order.
    - If scoring is requested, neural_fusion is auto-included so coaching feedback
      is not silently downgraded to heuristic-only output.
    """
    if not requested_stages:
        return list(DEFAULT_STAGES)

    allowed = set(DEFAULT_STAGES)
    requested = set(requested_stages)
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}")

    # Dependency closure in canonical pipeline order.
    deps: dict[str, tuple[str, ...]] = {
        "extract_selected_features": tuple(),
        "classify_views": ("extract_selected_features",),
        "temporal_segmentation": ("extract_selected_features", "classify_views"),
        "scoring": ("extract_selected_features", "classify_views", "temporal_segmentation"),
        "neural_fusion": ("extract_selected_features", "classify_views", "temporal_segmentation", "scoring"),
    }

    # Coaching payload expects neural outputs whenever scoring is requested.
    if "scoring" in requested:
        requested.add("neural_fusion")

    expanded = set(requested)
    changed = True
    while changed:
        changed = False
        for stage in list(expanded):
            for dep in deps.get(stage, tuple()):
                if dep not in expanded:
                    expanded.add(dep)
                    changed = True

    return [stage for stage in DEFAULT_STAGES if stage in expanded]


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ExeVision AI Inference API",
    description="Squat form analysis pipeline — wraps MediaPipe + rule-based AQA + neural fusion",
    version="1.0.0",
)

# Create results directory if it doesn't exist (for local dev static file serving)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RUNS_ROOT)), name="results")

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
    exercise: str = "squat"                 # Exercise type; defaults to "squat" for backward compatibility
    stages: list[str] | None = None         # Subset of DEFAULT_STAGES; None = run all
    mode: Literal["filtered", "unfiltered"] = "filtered"
    generate_viz: bool = True               # Generate visuals (annotated video outputs from MediaPipe)
    callback_url: str | None = None         # Optional: POST result here when done
    rep_boundaries: list[dict] | None = None  # Per-rep boundaries (start_sec, end_sec) from browser real-time detection


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    queued_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# ── Callback helper ───────────────────────────────────────────────────────────
def _fire_callback(callback_url: str, payload: dict[str, Any]) -> None:
    """
    POST the job result to the Next.js callback endpoint.
    Includes the Authorization header required by callback/route.ts.
    Failures are logged but never propagate — the job result is already
    persisted in the in-memory store and returned via /jobs/{job_id}.
    """
    import httpx

    headers = {"Content-Type": "application/json"}
    if _API_SECRET:
        headers["Authorization"] = f"Bearer {_API_SECRET}"

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(callback_url, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "[callback] POST to %s returned HTTP %s — result may not be persisted in DB.",
                callback_url,
                resp.status_code,
            )
        else:
            logger.info("[callback] POST to %s succeeded (HTTP %s).", callback_url, resp.status_code)
    except Exception as exc:
        logger.warning(
            "[callback] POST to %s failed (network error): %s — result not persisted via callback.",
            callback_url,
            exc,
        )


# ── Background task ────────────────────────────────────────────────────────────
def _pipeline_task(job_id: str, video_url: str, stages: list[str], mode: str, callback_url: str | None, generate_viz: bool, exercise: str = "squat", rep_boundaries: list[dict] | None = None) -> None:
    """Downloads the video and runs the full pipeline. Runs in a background thread."""
    import asyncio

    _update_job(job_id, status="running", started_at=datetime.now(timezone.utc).isoformat())

    try:
        with tempfile.TemporaryDirectory(prefix=f"exevision_{job_id}_") as tmp:
            tmp_path = Path(tmp)

            # Run async download in a new event loop for the thread
            loop = asyncio.new_event_loop()
            try:
                video_path = loop.run_until_complete(download_video(video_url, tmp_path))
            finally:
                loop.close()

            result = run_pipeline_sync(
                job_id=job_id,
                video_path=video_path,
                stages=stages,
                mode=mode,
                generate_viz=generate_viz,
                exercise=exercise,
                rep_boundaries=rep_boundaries,
            )

        if result.get("extraction_failed"):
            reason = result.get("extraction_failure_reason", "Pose extraction failed")
            _update_job(job_id, status="failed", error=reason, completed_at=datetime.now(timezone.utc).isoformat())
            if callback_url:
                _fire_callback(callback_url, {"job_id": job_id, "status": "failed", "error": reason, "visualization_url": None, "visualization_available": False})
            return

        _update_job(job_id, status="done", result=result, completed_at=datetime.now(timezone.utc).isoformat())

        # Fire-and-forget callback to Next.js / Supabase if requested
        if callback_url:
            callback_payload = {
                "job_id": job_id,
                "status": "done",
                "result": result,
                "visualization_url": result.get("visualization_url"),
                "visualization_available": result.get("visualization_available", False),
            }
            _fire_callback(callback_url, callback_payload)

    except Exception as exc:
        err_msg = _sanitize_error(str(exc))
        _update_job(
            job_id,
            status="failed",
            error=err_msg,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        if callback_url:
            callback_payload = {
                "job_id": job_id,
                "status": "failed",
                "error": err_msg,
                "visualization_url": None,
                "visualization_available": False,
            }
            _fire_callback(callback_url, callback_payload)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/infer", status_code=status.HTTP_202_ACCEPTED, dependencies=[Security(_verify_secret)])
def submit_inference(req: InferRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Accept a video URL and enqueue a pipeline run.
    Returns immediately with a job_id; poll GET /jobs/{job_id} for results.
    """
    from pipeline import EXERCISES_CONFIG_DIR, _resolve_exercise_config

    job_id = req.job_id or str(uuid.uuid4())

    # Validate exercise config exists (aliases like seated_overhead_press → overhead_press resolved here)
    try:
        _resolve_exercise_config(req.exercise)
    except FileNotFoundError:
        available = sorted(p.stem for p in EXERCISES_CONFIG_DIR.glob("*.json"))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported exercise: '{req.exercise}'. Available: {available}",
        )

    try:
        stages = _normalize_stage_selection(req.stages)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        generate_viz=req.generate_viz,
        exercise=req.exercise,
        rep_boundaries=req.rep_boundaries,
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
