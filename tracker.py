# filename: tracker.py
import numpy as np
from typing import List, Dict

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

class Track:
    def __init__(self, track_id, bbox, cls_id):
        self.track_id = track_id
        self.bbox = bbox
        self.cls_id = cls_id
        self.hits = 1
        self.missed = 0

class SimpleTracker:
    """
    Class-agnostic multi-object tracker with IOU matching.
    Good enough for stable player, racket, ball IDs over a rally.
    """

    def __init__(self, iou_threshold=0.3, max_missed=10):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: list of dicts from YOLODetector.detect
        Returns: list with added "track_id" field.
        """
        assigned_tracks = set()
        assigned_dets = set()

        # Compute iou matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                if track.cls_id != det["cls_id"]:
                    continue
                iou_matrix[t_idx, d_idx] = iou(track.bbox, det["bbox"])

        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break
            t_idx, d_idx = divmod(iou_matrix.argmax(), iou_matrix.shape[1])
            if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                break
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                iou_matrix[t_idx, d_idx] = -1
                continue

            track = self.tracks[t_idx]
            det = detections[d_idx]
            track.bbox = det["bbox"]
            track.hits += 1
            track.missed = 0
            det["track_id"] = track.track_id

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)
            iou_matrix[t_idx, d_idx] = -1

        # New tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx in assigned_dets:
                continue
            new_track = Track(self.next_id, det["bbox"], det["cls_id"])
            det["track_id"] = self.next_id
            self.tracks.append(new_track)
            self.next_id += 1

        # Increment missed count for unmatched tracks and remove dead ones
        active_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in assigned_tracks:
                track.missed += 1
            if track.missed <= self.max_missed:
                active_tracks.append(track)
        self.tracks = active_tracks

        return detections
