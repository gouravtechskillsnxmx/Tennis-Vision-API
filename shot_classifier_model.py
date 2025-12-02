# filename: shot_classifier_model.py
import torch
import torch.nn as nn

class ShotClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, num_layers=2, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
