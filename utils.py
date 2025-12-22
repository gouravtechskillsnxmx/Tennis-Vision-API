# filename: utils.py
# utils.py
import cv2
from typing import List, Optional

from config import TARGET_FPS, MAX_SECONDS, MAX_WIDTH, MAX_FRAMES


def extract_frames(
    video_path: str,
    target_fps: int = TARGET_FPS,
    max_seconds: float = MAX_SECONDS,
    max_width: int = MAX_WIDTH,
    max_frames: int = MAX_FRAMES,
) -> List:
    """
    Extract frames from video with strict caps for synchronous API usage:
    - sample to target_fps
    - stop after max_seconds
    - downscale to max_width
    - optional hard cap max_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = float(target_fps)

    # Determine how often to grab frames to approximate target_fps
    frame_interval = max(1, int(round(src_fps / float(target_fps))))

    # Compute maximum frames allowed by time
    time_cap_frames = None
    if max_seconds and max_seconds > 0:
        time_cap_frames = int(max_seconds * float(target_fps))

    # If MAX_FRAMES is not set, pick the smallest sensible cap
    # so we never process too much in a web request.
    caps = [c for c in [time_cap_frames, max_frames if max_frames and max_frames > 0 else None] if c is not None]
    effective_cap = min(caps) if caps else 300  # fallback cap

    frames = []
    i = 0
    kept = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % frame_interval == 0:
            # Downscale
            h, w = frame.shape[:2]
            if max_width and max_width > 0 and w > max_width:
                scale = max_width / float(w)
                new_w = max_width
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frames.append(frame)
            kept += 1

            if kept >= effective_cap:
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
