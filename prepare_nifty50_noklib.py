# prepare_nifty50_noklib.py
import os, pickle
import pandas as pd

OUTPUT_DIR = "./data/processed_datasets"
LOOKBACK, PREDICT = 90, 10

# --- Load your data however you have it ---
# Option A: one CSV per symbol (columns: datetime, open, high, low, close)
def load_from_csvs(data_dir):
    all_data = {}
    for f in os.listdir(data_dir):
        if not f.endswith(".csv"): continue
        sym = f.replace(".csv", "")
        df = pd.read_csv(f"{data_dir}/{f}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df[["open", "high", "low", "close"]].dropna()
        if len(df) >= LOOKBACK + PREDICT + 1:
            all_data[sym] = df
    return all_data

all_data = load_from_csvs("./nifty50_csv")

# --- Split by date (overlap for lookback buffer) ---
splits = {
    "train": (None,         "2023-09-30"),
    "val":   ("2023-07-01", "2024-09-30"),  # 3-month overlap
    "test":  ("2024-07-01", None),           # 3-month overlap
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for name, (start, end) in splits.items():
    split = {}
    for sym, df in all_data.items():
        mask = pd.Series(True, index=df.index)
        if start: mask &= df.index >= start
        if end:   mask &= df.index <= end
        sliced = df[mask]
        if len(sliced) >= LOOKBACK + PREDICT + 1:
            split[sym] = sliced
    with open(f"{OUTPUT_DIR}/{name}_data.pkl", "wb") as f:
        pickle.dump(split, f)
    print(f"{name}: {len(split)} symbols saved")