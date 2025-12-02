# filename: shot_classifier_train.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
from typing import List, Dict
from shot_classifier_model import ShotClassifier

class ShotDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def train_shot_classifier(
    train_sequences: List[np.ndarray],
    train_labels: List[int],
    val_sequences: List[np.ndarray],
    val_labels: List[int],
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    save_path: str = "shot_classifier_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShotClassifier(input_dim=train_sequences[0].shape[-1],
                           hidden_dim=128,
                           num_layers=2,
                           num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_ds = ShotDataset(train_sequences, train_labels)
    val_ds = ShotDataset(val_sequences, val_labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
        avg_val_loss = val_loss / len(val_ds)
        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch+1}/{epochs} - train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")
