# filename: scoring_engine.py
from typing import Dict, List

def score_shot(features: Dict[str, float], shot_type: str) -> int:
    """
    Rule-based scoring from 0 to 100.
    You can tweak these heuristics based on coach domain knowledge.
    """
    # Example features:
    # elbow_angle_right, shoulder_angle_right, ball_speed, swing_arc, pose_frames
    prep_score = 0.0
    timing_score = 0.0
    swing_score = 0.0
    balance_score = 0.0
    follow_score = 0.0

    # Preparation: some shoulder rotation is good (say 20–80 degrees)
    sh = features.get("shoulder_angle_right", 0.0)
    if 20 < sh < 80:
        prep_score = 90
    elif 10 < sh < 100:
        prep_score = 70
    else:
        prep_score = 40

    # Swing arc: not too small, not crazy big
    arc = features.get("swing_arc", 0.0)
    if 30 < arc < 140:
        swing_score = 90
    elif 15 < arc < 180:
        swing_score = 70
    else:
        swing_score = 40

    # Ball speed proxy
    speed = features.get("ball_speed", 0.0)
    if speed > 15:
        timing_score = 90
    elif speed > 5:
        timing_score = 70
    else:
        timing_score = 50

    # Balance heuristic: use pose_frames as proxy (longer controlled sequence)
    frames = features.get("pose_frames", 0)
    if frames >= 15:
        balance_score = 85
    elif frames >= 8:
        balance_score = 70
    else:
        balance_score = 50

    # Follow-through: derived from swing arc and frames
    if arc > 60 and frames > 10:
        follow_score = 90
    else:
        follow_score = 70

    score = (
        0.20 * prep_score +
        0.25 * swing_score +
        0.25 * timing_score +
        0.15 * balance_score +
        0.10 * follow_score +
        0.05 * 80  # dummy outcome score
    )

    final_score = int(round(max(0, min(100, score))))
    return final_score

def strengths_and_improvements(features: Dict[str, float], shot_type: str, score: int):
    strengths: List[str] = []
    improvements: List[str] = []

    sh = features.get("shoulder_angle_right", 0.0)
    arc = features.get("swing_arc", 0.0)
    speed = features.get("ball_speed", 0.0)

    if 20 < sh < 80:
        strengths.append("Good shoulder rotation during preparation.")
    else:
        improvements.append("Rotate shoulders more for better preparation.")

    if 30 < arc < 140:
        strengths.append("Swing arc looks efficient and controlled.")
    else:
        improvements.append("Refine swing arc to avoid being too short or too long.")

    if speed > 15:
        strengths.append("Shot generates strong pace on the ball.")
    elif speed < 5:
        improvements.append("Try accelerating the racket through contact to produce more pace.")

    if score < 60:
        improvements.append("Work on timing your contact in front of the body.")
    elif score > 80:
        strengths.append("Overall shot quality is high with consistent mechanics.")

    return strengths, improvements
