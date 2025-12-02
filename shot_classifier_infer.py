# filename: shot_classifier_infer.py
import torch
import numpy as np
from typing import List
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
        # Assuming input_dim 16 (e.g., aggregated features)
        self.model = ShotClassifier(input_dim=16, hidden_dim=128, num_layers=2,
                                    num_classes=len(SHOT_LABELS)).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict_from_features(self, feature_vector: np.ndarray) -> str:
        """
        feature_vector: shape (seq_len, 16) or (16,) -> we will handle both.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector[None, :]  # (1,16)
        x = torch.tensor(feature_vector[None, :, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            pred = logits.argmax(dim=1).item()
        return SHOT_LABELS[pred]
