from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix


# Absolute, stable paths (independent of PyCharm working directory)
ROOT = Path(__file__).resolve().parents[1]   # .../netflow2img_cnn
OUT_DIR = ROOT / "outputs"
IMG_DIR = OUT_DIR / "images"
MODEL_PATH = OUT_DIR / "cnn_ddos.pt"
METRICS_PATH = OUT_DIR / "metrics.txt"


class SimpleCNN(nn.Module):
    """
    Small CNN suitable for tiny grayscale "feature-images" (e.g., 10x10, 12x12, etc.).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
    return np.array(ys), np.array(ps)


def assert_imagefolder_layout(img_root: Path):
    """
    Validates that outputs/images has ImageFolder-compatible structure:
      images/train/<class>/*.png, images/val/<class>/*.png, images/test/<class>/*.png
    """
    if not img_root.exists():
        raise FileNotFoundError(f"IMG_DIR does not exist: {img_root}")

    for split in ["train", "val", "test"]:
        split_dir = img_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            raise RuntimeError(
                f"{split_dir} has no class subfolders. Expected e.g. {split_dir}/benign and {split_dir}/ddos."
            )

        # Ensure each class has at least 1 image
        for cd in class_dirs:
            imgs = list(cd.glob("*.png")) + list(cd.glob("*.jpg")) + list(cd.glob("*.jpeg"))
            if len(imgs) == 0:
                raise RuntimeError(f"Class folder is empty: {cd} (no images found)")


def main():
    # Ensure outputs exists (for saving model + metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")

    print("Device:", device)
    print("ROOT:", ROOT)
    print("IMG_DIR:", IMG_DIR)
    print("IMG_DIR exists:", IMG_DIR.exists())

    # Validate folder structure before loading datasets
    assert_imagefolder_layout(IMG_DIR)

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # scales to 0..1
    ])

    train_ds = ImageFolder(IMG_DIR / "train", transform=tfm)
    val_ds = ImageFolder(IMG_DIR / "val", transform=tfm)
    test_ds = ImageFolder(IMG_DIR / "test", transform=tfm)

    # Enforce consistent class mapping across splits
    # (ImageFolder builds class_to_idx from folder names)
    if val_ds.class_to_idx != train_ds.class_to_idx or test_ds.class_to_idx != train_ds.class_to_idx:
        raise RuntimeError(
            "Class mapping mismatch across splits.\n"
            f"train: {train_ds.class_to_idx}\n"
            f"val:   {val_ds.class_to_idx}\n"
            f"test:  {test_ds.class_to_idx}\n"
            "Fix by ensuring train/val/test have identical class folder names."
        )

    class_names = list(train_ds.class_to_idx.keys())
    print("Class mapping:", train_ds.class_to_idx)
    print("Class names:", class_names)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = SimpleCNN(num_classes=len(class_names)).to(device)

    # ---- Class weights (handles imbalance) ----
    labels = [y for _, y in train_ds.samples]
    counts = np.bincount(labels, minlength=len(class_names)).astype(np.float64)
    # Inverse frequency; normalize for stability
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv * (len(inv) / inv.sum())
    class_weights = torch.tensor(inv, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, 11):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / max(total, 1)
        train_loss = loss_sum / max(total, 1)

        yv, pv = evaluate(model, val_loader, device)
        val_acc = float((yv == pv).mean()) if len(yv) else 0.0

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val acc: {best_val_acc:.4f} (epoch {best_epoch})")

    # Load best model for test evaluation
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    yt, pt = evaluate(model, test_loader, device)
    cm = confusion_matrix(yt, pt)
    report = classification_report(yt, pt, target_names=class_names)

    print("\nConfusion matrix:\n", cm)
    print("\nReport:\n", report)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Device: {device}\n")
        f.write(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch})\n\n")
        f.write("Class mapping:\n")
        f.write(str(train_ds.class_to_idx) + "\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Report:\n")
        f.write(report)


if __name__ == "__main__":
    main()

