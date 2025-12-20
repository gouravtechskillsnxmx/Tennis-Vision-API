# filename: Dockerfile
FROM python:3.10-bookworm

# Prevents Python from writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System deps (OpenCV, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Ensure shared data dirs exist inside the container
RUN mkdir -p /data/videos /data/results /data/overlays

# optional, kept for clarity; Render uses $PORT, not this

EXPOSE 8000  

ENV YOLO_WEIGHTS_PATH="yolo_tennis_best.pt"
ENV MOVENET_MODEL_DIR="movenet_saved_model"
ENV SHOT_CLASSIFIER_WEIGHTS="shot_classifier_best.pt"

# IMPORTANT: bind to $PORT (Render sets this)
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]

