import os

# Paths
YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "yolo_tennis_best.pt")
MOVENET_SAVED_MODEL_DIR = os.getenv("MOVENET_MODEL_DIR", "movenet_saved_model")
SHOT_CLASSIFIER_WEIGHTS = os.getenv("SHOT_CLASSIFIER_WEIGHTS", "shot_classifier_best.pt")

# OpenAI / GPT
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Video
DEFAULT_FPS = 30
MAX_FRAMES = 0  # 0 = process entire video
