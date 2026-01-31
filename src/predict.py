from pathlib import Path
import math
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn

OUT_DIR = Path("outputs")
MODEL_PATH = OUT_DIR / "cnn_ddos.pt"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

def row_to_image(row_vec: np.ndarray, size: int) -> np.ndarray:
    total = size * size
    padded = np.zeros((total,), dtype=np.float32)
    padded[: row_vec.shape[0]] = row_vec
    return padded.reshape(size, size)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = joblib.load(OUT_DIR / "scaler.joblib")

    # Example: load a small CSV you create with same feature columns (post-drop)
    sample = pd.read_csv("sample_row.csv")
    X = sample.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    Xs = scaler.transform(X)

    n_features = Xs.shape[1]
    size = math.ceil(math.sqrt(n_features))

    img = row_to_image(Xs[0], size)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    label = "DDoS" if pred == 1 else "BENIGN"
    print("Prediction:", label)

if __name__ == "__main__":
    main()

