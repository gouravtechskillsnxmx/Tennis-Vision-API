# filename: yolo_infer.py
from ultralytics import YOLO
import numpy as np
from typing import List, Dict
from config import YOLO_WEIGHTS_PATH

class YOLODetector:
    """
    Wrapper around Ultralytics YOLO for detecting player, racket, ball.
    Assumes your model’s classes are:
      0: player
      1: racket
      2: ball
    """

    def __init__(self, weights_path: str = YOLO_WEIGHTS_PATH, conf: float = 0.25):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns list of detections:
        [
          {"cls": "player", "cls_id": 0, "conf": 0.91, "bbox": [x1,y1,x2,y2]},
          ...
        ]
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = self.model.names[cls_id]
            detections.append(
                {
                    "cls": cls_name,
                    "cls_id": cls_id,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections
