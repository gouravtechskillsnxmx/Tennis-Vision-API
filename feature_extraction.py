# filename: feature_extraction.py
import numpy as np
from typing import List, Dict

def compute_joint_angles(pose_seq: List[np.ndarray]) -> Dict[str, float]:
    """
    Example: compute average elbow & shoulder angles as a proxy for swing quality.
    pose_seq: list of (17,3) keypoint arrays.
    """
    if not pose_seq:
        return {"elbow_angle": 0.0, "shoulder_angle": 0.0}

    def angle(a, b, c):
        # angle at b given 3 points
        ba = a - b
        bc = c - b
        num = np.dot(ba, bc)
        den = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        cosang = np.clip(num / den, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    left_elbows = []
    right_elbows = []
    left_shoulders = []
    right_shoulders = []

    for kps in pose_seq:
        # Using COCO-like indices: shoulder(5,6) elbow(7,8) wrist(9,10)
        # You can adjust to MoveNet indexing as needed.
        try:
            ls, rs = kps[5, :2], kps[6, :2]
            le, re = kps[7, :2], kps[8, :2]
            lw, rw = kps[9, :2], kps[10, :2]

            left_elbows.append(angle(ls, le, lw))
            right_elbows.append(angle(rs, re, rw))
            left_shoulders.append(angle(le, ls, rs))
            right_shoulders.append(angle(re, rs, ls))
        except Exception:
            continue

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "elbow_angle_left": avg(left_elbows),
        "elbow_angle_right": avg(right_elbows),
        "shoulder_angle_left": avg(left_shoulders),
        "shoulder_angle_right": avg(right_shoulders),
    }

def compute_ball_speed(ball_positions: List[tuple], fps: int = 30) -> float:
    if len(ball_positions) < 2:
        return 0.0
    dist_px = 0.0
    for i in range(1, len(ball_positions)):
        x1, y1 = ball_positions[i-1]
        x2, y2 = ball_positions[i]
        dist_px += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    # px/frame * fps → px/sec (you can calibrate to m/s if court dimensions known)
    avg_speed_px_per_frame = dist_px / max(1, len(ball_positions)-1)
    return float(avg_speed_px_per_frame * fps)

def compute_swing_arc(racket_positions: List[tuple]) -> float:
    """
    Total angular change of racket path (rough proxy).
    """
    if len(racket_positions) < 3:
        return 0.0
    angles = []
    for i in range(1, len(racket_positions)):
        x1, y1 = racket_positions[i-1]
        x2, y2 = racket_positions[i]
        angles.append(np.degrees(np.arctan2(y2-y1, x2-x1)))
    if not angles:
        return 0.0
    return float(max(angles) - min(angles))

def aggregate_features(
    pose_seq: List[np.ndarray],
    ball_positions: List[tuple],
    racket_positions: List[tuple],
    fps: int = 30
) -> Dict[str, float]:
    features = {}
    features.update(compute_joint_angles(pose_seq))
    features["ball_speed"] = compute_ball_speed(ball_positions, fps)
    features["swing_arc"] = compute_swing_arc(racket_positions)
    features["pose_frames"] = len(pose_seq)
    return features
