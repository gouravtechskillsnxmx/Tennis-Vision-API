# filename: overlay_video.py
import cv2
import os
import uuid
from typing import Dict, List, Any

def create_overlay(
    video_path: str,
    frame_results: List[Dict[str, Any]],
    shot_type: str,
    score: int,
    output_dir: str = "overlays",
) -> str:
    """
    frame_results: list of dicts, each:
      {
        "detections": [ {cls, cls_id, conf, bbox, track_id}, ... ],
        "poses": {track_id: keypoints(17,3)},  # optional
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_dir, f"overlay_{uuid.uuid4().hex}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frame_results):
            break

        info = frame_results[frame_idx]
        dets = info.get("detections", [])
        poses = info.get("poses", {})

        # Draw detections
        for det in dets:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_name = det["cls"]
            track_id = det.get("track_id", -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            text = f"{cls_name}#{track_id}"
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

        # Draw simple pose keypoints for players
        for tid, kps in poses.items():
            for (x, y, s) in kps:
                cx = int(x * w)
                cy = int(y * h)
                if s > 0.3:
                    cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

        # Top bar with shot & score
        cv2.rectangle(frame, (0,0), (w,40), (0,0,0), -1)
        cv2.putText(frame, f"{shot_type} | Score: {score}/100", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path
