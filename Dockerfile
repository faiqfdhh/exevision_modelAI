# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Use an explicit virtual environment inside the container for deterministic runtime.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"

# System packages needed by MediaPipe, OpenCV, and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements-runtime.txt .

# Install CPU-only torch first (avoids pulling CUDA wheels)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
    torch>=2.3.0 \
    torchvision>=0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining runtime deps (requirements-runtime.txt excludes torch/torchvision)
# NOTE: requirements-runtime.txt includes `supabase` so containerized Cloud Run deployments
# have the Supabase Python client available for visualization uploads.
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy source tree (training_dataset/, _hidden_legacy/, etc. excluded by .dockerignore)
COPY . .

# Ensure feedback narrative config files and exercise configs are always present in the runtime image.
# If these are missing, API responses will omit `result.feedback`.
RUN test -f /app/core/exevision/config/exercises/squat.json
RUN test -f /app/core/exevision/config/exercises/overhead_press.json
RUN test -f /app/core/exevision/config/templates/feedback_templates.json
RUN python -c "from pathlib import Path; \
    configs = {c.stem for c in Path('/app/core/exevision/config/exercises').glob('*.json')}; \
    required = {'squat', 'overhead_press'}; \
    missing = required - configs; \
    assert not missing, f'Missing exercise configs: {missing}'; \
    print(f'Available exercises: {sorted(configs)}')"

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
CMD /opt/venv/bin/python -m uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT
