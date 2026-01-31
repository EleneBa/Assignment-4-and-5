import subprocess
import sys

scripts = [
    "src/inspect_labels.py",
    "src/preprocess.py",
    "src/make_images.py",
    "src/train_cnn.py"
]

for script in scripts:
    print(f"\n▶ Running {script}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline stopped at {script}")
