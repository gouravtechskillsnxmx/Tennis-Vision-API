# filename: Dockerfile
FROM python:3.10-slim

# Prevents Python from writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt /app/

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Expose the API port
EXPOSE 8000

# Default envs (override on Render)
ENV YOLO_WEIGHTS_PATH="yolo_tennis_best.pt"
ENV MOVENET_MODEL_DIR="movenet_saved_model"
ENV SHOT_CLASSIFIER_WEIGHTS="shot_classifier_best.pt"

# Start FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
