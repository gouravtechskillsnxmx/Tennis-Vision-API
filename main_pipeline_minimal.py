# filename: main_pipeline_minimal.py
from typing import Dict, Any, List
import numpy as np

from yolo_infer import YOLODetector
from utils import extract_frames


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Minimal pipeline:
    - Extract a limited number of downscaled frames
    - Run YOLO on them
    - Return a dummy JSON response summarizing detections

    This is just to verify that the backend and YOLO run end-to-end on Render.
    """

    # Extract frames at low FPS and small resolution (utils should already downscale)
    frames: List[np.ndarray] = extract_frames(video_path, target_fps=10)

    if not frames:
        return {
            "type_of_shot": "unknown",
            "strengths": [],
            "improvements": ["No frames could be read from the video."],
            "score": 0,
            "overlay_url": "",
        }

    # Use small YOLO model (you should have yolo_infer.py hard-coded to yolov8n.pt)
    detector = YOLODetector()

    num_frames = min(30, len(frames))  # only first 30 frames for safety
    det_counts = []

    for i in range(num_frames):
        dets = detector.detect(frames[i])
        det_counts.append(len(dets))

    avg_dets = sum(det_counts) / max(1, len(det_counts))

    # Return something simple so we know it worked
    return {
        "type_of_shot": "unknown",
        "strengths": [
            "End-to-end pipeline ran successfully on Render.",
            f"Average detections per frame in the first {num_frames} frames: {avg_dets:.2f}",
        ],
        "improvements": [
            "Re-enable pose estimation and shot classifier once this minimal version is stable."
        ],
        "score": 50,
        "overlay_url": "",
    }
