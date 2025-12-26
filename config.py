# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENABLE_SHOT_CLASSIFIER = os.getenv("ENABLE_SHOT_CLASSIFIER", "0").strip() == "1"


YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.pt")
MOVENET_SAVED_MODEL_DIR = os.getenv("MOVENET_MODEL_DIR", "movenet_saved_model")
SHOT_CLASSIFIER_WEIGHTS = os.getenv("SHOT_CLASSIFIER_WEIGHTS", "shot_classifier_best.pt")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_FPS = int(os.getenv("DEFAULT_FPS", "30"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0"))  # 0 = default cap in utils

MAX_SECONDS = int(os.getenv("MAX_SECONDS", "8"))   # default 8 seconds
TARGET_FPS = int(os.getenv("TARGET_FPS", "12"))   # default 12 fps
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "640"))    # default 640px



