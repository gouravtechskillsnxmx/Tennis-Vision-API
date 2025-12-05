# yolo_infer.py
import os
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        # If explicit path provided, use it; otherwise fall back to env or yolov8n.pt
        model_name = (
            weights_path
            or os.getenv("YOLO_WEIGHTS_PATH")
            or "yolov8n.pt"  # default built-in model
        )
        print(f"[YOLODetector] Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame: np.ndarray):
        results = self.model(
            frame,
            device=self.device,
            conf=0.3,
            verbose=False
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections
