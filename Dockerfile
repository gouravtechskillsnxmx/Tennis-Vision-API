# Optimized Dockerfile for Render (faster builds, smaller image)
FROM python:3.10-slim-bookworm

# Python + pip runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS deps:
# - ffmpeg: video decoding/encoding
# - libgl1 + libglib2.0-0: commonly needed by opencv/mediapipe wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (best caching)
COPY requirements.txt /app/requirements.txt

# If Render uses BuildKit (usually yes), this speeds up repeat installs by caching pip downloads
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# (Optional) create data dirs if your code writes there
RUN mkdir -p /data/videos /data/results /data/overlays

# Render binds to $PORT
EXPOSE 8000

# Keep your existing envs (ok to leave even if not used)
ENV YOLO_WEIGHTS_PATH="yolo_tennis_best.pt" \
    MOVENET_MODEL_DIR="movenet_saved_model" \
    SHOT_CLASSIFIER_WEIGHTS="shot_classifier_best.pt"

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
