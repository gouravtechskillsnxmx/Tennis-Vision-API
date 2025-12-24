# pose_movenet.py
# Drop-in replacement for Option 1: NO TensorFlow.
# Keeps class name MoveNetPose but uses MediaPipe Pose internally.

from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
import mediapipe as mp


class MoveNetPose:
    """
    Compatibility wrapper:
    main_pipeline imports MoveNetPose; we keep that name but implement pose
    estimation using MediaPipe Pose (no TensorFlow).

    estimate(frame, bbox) -> {"keypoints": np.ndarray (17,3)} with (x, y, score)
    where x,y are in image pixel coordinates and score is 0..1.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Map MediaPipe 33 landmarks -> COCO-like 17 keypoints
        self._coco17_from_mp33 = [
            0,   # nose
            2,   # left eye
            5,   # right eye
            7,   # left ear
            8,   # right ear
            11,  # left shoulder
            12,  # right shoulder
            13,  # left elbow
            14,  # right elbow
            15,  # left wrist
            16,  # right wrist
            23,  # left hip
            24,  # right hip
            25,  # left knee
            26,  # right knee
            27,  # left ankle
            28,  # right ankle
        ]

    def estimate(self, frame: np.ndarray, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"keypoints": np.zeros((17, 3), dtype=np.float32)}

        h, w = frame.shape[:2]

        # Optional crop for speed
        x1, y1, x2, y2 = 0, 0, w, h
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(1, min(w, x2)))
            y2 = int(max(1, min(h, y2)))
            if x2 <= x1 or y2 <= y1:
                return {"keypoints": np.zeros((17, 3), dtype=np.float32)}

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"keypoints": np.zeros((17, 3), dtype=np.float32)}

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self._pose.process(crop_rgb)

        kps = np.zeros((17, 3), dtype=np.float32)

        if not res.pose_landmarks:
            return {"keypoints": kps}

        lm = res.pose_landmarks.landmark  # 33 landmarks (normalized to crop)
        cw = float(x2 - x1)
        ch = float(y2 - y1)

        for i, mp_idx in enumerate(self._coco17_from_mp33):
            l = lm[mp_idx]
            px = x1 + l.x * cw
            py = y1 + l.y * ch
            score = float(max(0.0, min(1.0, l.visibility)))
            kps[i] = (px, py, score)

        return {"keypoints": kps}
