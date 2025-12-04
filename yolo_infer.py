# yolo_infer.py
import logging
from typing import List, Dict, Any

from ultralytics import YOLO
from config import YOLO_WEIGHTS_PATH

logger = logging.getLogger("yolo_detector")


class YOLODetector:
    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path or YOLO_WEIGHTS_PATH
        logger.info(f"[YOLODetector] Loading YOLO weights from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run YOLO inference on a single frame.
        Returns a list of detections with bbox, class_id, confidence, etc.
        """
        results = self.model(frame, verbose=False)
        detections = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append(
                {
                    "bbox": xyxy,
                    "class_id": cls_id,
                    "confidence": conf,
                }
            )
        return detections
