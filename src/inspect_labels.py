import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../netflow2img_cnn
scripts = [
    ROOT / "src" / "inspect_labels.py",
    ROOT / "src" / "preprocess.py",
    ROOT / "src" / "make_images.py",
    ROOT / "src" / "train_cnn.py",
]

print("RUNNING:", __file__)
print("ROOT:", ROOT)

for script in scripts:
    print(f"\n▶ Running {script}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline stopped at {script}")
