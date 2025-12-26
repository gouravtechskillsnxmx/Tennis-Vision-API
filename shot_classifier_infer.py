# shot_classifier_infer.py
import os
import torch
import numpy as np
from typing import List, Optional

from shot_classifier_model import ShotClassifier
from config import SHOT_CLASSIFIER_WEIGHTS

SHOT_LABELS = [
    "forehand",
    "backhand",
    "serve",
    "volley",
    "slice",
    "smash",
    "drop_shot",
    "bandeja"
]


class ShotTypeInferencer:
    def __init__(self, weights_path: str = SHOT_CLASSIFIER_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enabled = False
        self.model: Optional[ShotClassifier] = None

        # If weights file does not exist, DISABLE classifier gracefully
        if not weights_path or not os.path.exists(weights_path):
            print(
                f"[ShotTypeInferencer] Weights not found at '{weights_path}'. "
                "Shot classification disabled (returning 'unknown').",
                flush=True,
            )
            return

        # Initialize model only if weights exist
        self.model = ShotClassifier(
            input_dim=16,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(SHOT_LABELS),
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.eval()
        self.enabled = True
        print("[ShotTypeInferencer] Shot classifier loaded successfully.", flush=True)

    def predict_from_features(self, feature_vector: np.ndarray) -> str:
        if not self.enabled or self.model is None:
            return "unknown"

        if feature_vector.ndim == 1:
            feature_vector = feature_vector[None, :]  # (1, 16)

        x = torch.tensor(
            feature_vector[None, :, :], dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            pred = logits.argmax(dim=1).item()

        return SHOT_LABELS[pred]
