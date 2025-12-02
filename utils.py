# filename: utils.py
import cv2
from typing import List
from config import DEFAULT_FPS, MAX_FRAMES

def extract_frames(video_path: str, target_fps: int = DEFAULT_FPS) -> List:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(1, int(round(src_fps / target_fps)))
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frames.append(frame)
            if MAX_FRAMES > 0 and len(frames) >= MAX_FRAMES:
                break
        i += 1
    cap.release()
    return frames

def smooth_trajectory(points, window=5):
    if len(points) < window:
        return points
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window + 1)
        window_pts = points[start:i+1]
        xs = [p[0] for p in window_pts]
        ys = [p[1] for p in window_pts]
        smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return smoothed
