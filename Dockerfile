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
