from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_FILE = Path("data") / "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def find_label_col(columns):
    if " Label" in columns:
        return " Label"
    if "Label" in columns:
        return "Label"
    raise ValueError("Label column not found. Check CSV header.")

def main():
    # Load with pandas (100MB is usually OK), but chunking is also possible.
    df = pd.read_csv(DATA_FILE, low_memory=False)
    label_col = find_label_col(df.columns)

    # Strip whitespace in labels
    df[label_col] = df[label_col].astype(str).str.strip()

    # Keep only BENIGN and DDoS (common requirement for this specific file)
    df = df[df[label_col].isin(["BENIGN", "DDoS"])].copy()

    # Drop obvious non-feature columns if present
    drop_candidates = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port", "Dst Port", "Protocol"]
    for c in drop_candidates:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Separate X/y
    y = (df[label_col] == "DDoS").astype(int).values
    X = df.drop(columns=[label_col])

    # Convert to numeric robustly
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace inf with nan then fill
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.values, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Scale using train-only fit
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Persist artifacts
    np.save(OUT_DIR / "X_train.npy", X_train_s)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy", X_val_s)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "X_test.npy", X_test_s)
    np.save(OUT_DIR / "y_test.npy", y_test)

    joblib.dump(scaler, OUT_DIR / "scaler.joblib")

    # Save feature count for image shaping
    with open(OUT_DIR / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"n_features={X.shape[1]}\n")

    print("Done.")
    print("Shapes:",
          X_train_s.shape, y_train.shape,
          X_val_s.shape, y_val.shape,
          X_test_s.shape, y_test.shape)

if __name__ == "__main__":
    main()
