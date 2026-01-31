from pathlib import Path
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

OUT_DIR = Path("outputs")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def to_square_images(X: np.ndarray, size: int) -> np.ndarray:
    n, f = X.shape
    total = size * size
    if f > total:
        raise ValueError(f"Too many features ({f}) for size {size}x{size}. Increase size.")
    padded = np.zeros((n, total), dtype=np.float32)
    padded[:, :f] = X
    imgs = padded.reshape(n, size, size)
    return imgs

def save_pngs(imgs: np.ndarray, y: np.ndarray, split_name: str):
    split_dir = IMG_DIR / split_name
    (split_dir / "benign").mkdir(parents=True, exist_ok=True)
    (split_dir / "ddos").mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(imgs)), desc=f"Saving {split_name}"):
        arr = imgs[i]
        # MinMax already 0..1 → scale to 0..255
        im = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        cls = "ddos" if y[i] == 1 else "benign"
        im.save(split_dir / cls / f"{split_name}_{i:07d}.png")

def main():
    X_train = np.load(OUT_DIR / "X_train.npy")
    y_train = np.load(OUT_DIR / "y_train.npy")
    X_val = np.load(OUT_DIR / "X_val.npy")
    y_val = np.load(OUT_DIR / "y_val.npy")
    X_test = np.load(OUT_DIR / "X_test.npy")
    y_test = np.load(OUT_DIR / "y_test.npy")

    n_features = X_train.shape[1]
    size = math.ceil(math.sqrt(n_features))  # minimal square

    print("n_features =", n_features, "→ image size =", size, "x", size)

    train_imgs = to_square_images(X_train, size)
    val_imgs = to_square_images(X_val, size)
    test_imgs = to_square_images(X_test, size)

    save_pngs(train_imgs, y_train, "train")
    save_pngs(val_imgs, y_val, "val")
    save_pngs(test_imgs, y_test, "test")

    with open(OUT_DIR / "image_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"size={size}\n")

if __name__ == "__main__":
    main()
