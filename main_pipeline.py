# filename: main_pipeline.py
from typing import Dict, Any, List
import numpy as np

from yolo_infer import YOLODetector
from tracker import SimpleTracker
from pose_movenet import MoveNetPose
from utils import extract_frames
from feature_extraction import aggregate_features
from shot_classifier_infer import ShotTypeInferencer
from scoring_engine import score_shot, strengths_and_improvements
from gpt_feedback import generate_gpt_feedback
from overlay_video import create_overlay
from config import DEFAULT_FPS

def process_frames(
    frames: List,
    yolo: YOLODetector,
    tracker: SimpleTracker,
    pose_model: MoveNetPose
) -> Dict[str, Any]:
    """
    Returns:
      {
        "frame_results": [...],
        "pose_seq": [np.ndarray(17,3)],
        "ball_positions": [(x,y)...],
        "racket_positions": [(x,y)...],
    }
    """
    frame_results = []
    pose_seq = []
    ball_positions = []
    racket_positions = []

    for frame in frames:
        dets = yolo.detect(frame)
        dets = tracker.update(dets)

        # Pose estimation for players (cls=0 by assumption)
        poses_for_frame = {}
        for det in dets:
            if det["cls"] == "player":
                track_id = det["track_id"]
                bbox = det["bbox"]
                pose = pose_model.estimate(frame, bbox=bbox)
                kps = pose["keypoints"]  # (17,3)
                poses_for_frame[track_id] = kps
                pose_seq.append(kps)

        # Ball position: pick top-conf ball detection
        ball_det = next((d for d in sorted(dets, key=lambda x: x["conf"], reverse=True)
                         if d["cls"] == "ball"), None)
        if ball_det:
            x1,y1,x2,y2 = ball_det["bbox"]
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            ball_positions.append((cx, cy))

        # Racket position: top-conf racket
        racket_det = next((d for d in sorted(dets, key=lambda x: x["conf"], reverse=True)
                           if d["cls"] == "racket"), None)
        if racket_det:
            x1,y1,x2,y2 = racket_det["bbox"]
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            racket_positions.append((cx, cy))

        frame_results.append(
            {
                "detections": dets,
                "poses": poses_for_frame,
            }
        )

    return {
        "frame_results": frame_results,
        "pose_seq": pose_seq,
        "ball_positions": ball_positions,
        "racket_positions": racket_positions,
    }

def analyze_video(video_path: str) -> Dict[str, Any]:
    frames = extract_frames(video_path, target_fps=DEFAULT_FPS)
    yolo = YOLODetector()
    tracker = SimpleTracker()
    pose_model = MoveNetPose()
    shot_infer = ShotTypeInferencer()

    results = process_frames(frames, yolo, tracker, pose_model)

    # Aggregate features from entire shot
    pose_seq_np = [np.array(kps) for kps in results["pose_seq"]]
    features = aggregate_features(
        pose_seq_np,
        results["ball_positions"],
        results["racket_positions"],
        fps=DEFAULT_FPS,
    )

    # Convert features dict to fixed-length vector for classifier
    feat_keys = [
        "elbow_angle_left",
        "elbow_angle_right",
        "shoulder_angle_left",
        "shoulder_angle_right",
        "ball_speed",
        "swing_arc",
        "pose_frames",
    ]
    vec = np.array([features.get(k, 0.0) for k in feat_keys], dtype=np.float32)
    # For classifier input_dim=16, we can tile / pad:
    if vec.shape[0] < 16:
        vec = np.pad(vec, (0, 16 - vec.shape[0]))
    elif vec.shape[0] > 16:
        vec = vec[:16]

    shot_type = shot_infer.predict_from_features(vec)
    score = score_shot(features, shot_type)
    strengths, improvements = strengths_and_improvements(features, shot_type, score)
    feedback_text = generate_gpt_feedback(shot_type, features, strengths, improvements, score)
    overlay_path = create_overlay(video_path, results["frame_results"], shot_type, score)

    return {
        "type_of_shot": shot_type,
        "strengths": strengths,
        "improvements": improvements,
        "score": score,
        "overlay_url": overlay_path,
        "feedback": feedback_text,
    }

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <video.mp4>")
        sys.exit(1)
    res = analyze_video(sys.argv[1])
    print(json.dumps(res, indent=2))
