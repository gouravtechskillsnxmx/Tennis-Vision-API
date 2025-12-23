# filename: pose_movenet.py
import cv2
import numpy as np
from typing import Dict, Any
from config import MOVENET_SAVED_MODEL_DIR

class MoveNetPose:
    """
    Wrapper for MoveNet (singlepose) saved model.
    Input: BGR frame + bbox of player (optional crop).
    Output: 17 keypoints (x, y, score) normalised to [0,1].
    """

    def __init__(self, model_dir: str = MOVENET_SAVED_MODEL_DIR):
        self.model = tf.saved_model.load(model_dir)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        img = img[None, ...]
        return img

    def estimate(self, frame: np.ndarray, bbox: list | None = None) -> Dict[str, Any]:
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(x1, 0); y1 = max(y1, 0)
            x2 = min(x2, frame.shape[1]); y2 = min(y2, frame.shape[0])
            crop = frame[y1:y2, x1:x2].copy()
        else:
            crop = frame

        inp = self._preprocess(crop)
        outputs = self.model.signatures["serving_default"](tf.constant(inp))
        keypoints = outputs["output_0"].numpy()[0, 0, :, :]  # (17,3)

        # (x,y) normalised to frame coordinates (0..1)
        # Remember we resized to 256x256, so x,y already ~[0,1]
        return {"keypoints": keypoints}
