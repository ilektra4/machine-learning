# split_and_aggregate_kaggle.py
# Load kaggle_openface_merged.csv
# Split train/test by video_id (no leakage)
# Aggregate frame-level features to video-level (mean/std/max)
# Optionally compute DAU (delta AU) and aggregate those too

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Settings

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"
CSV_PATH = os.path.join(BASE_DIR, "kaggle_openface_merged.csv")

TEST_SIZE = 0.2
SEED = 42

ENABLE_DAU = True               # set False to skip DAU
TIME_CANDIDATES = ["timestamp", "frame"]  # for DAU ordering

# -------------------------
# Load
# -------------------------
def load_frames() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    print("Loading:", CSV_PATH)
    return pd.read_csv(CSV_PATH, low_memory=False)

df = load_frames()

required = ["video_id", "label", "success", "confidence"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------
# Split by video_id
# -------------------------
video_ids = df["video_id"].dropna().unique()
train_vids, test_vids = train_test_split(
    video_ids, test_size=TEST_SIZE, random_state=SEED, shuffle=True
)

train_frames = df[df["video_id"].isin(train_vids)].copy()
test_frames  = df[df["video_id"].isin(test_vids)].copy()

# Save split ids (reproducible)
with open(os.path.join(BASE_DIR, "train_video_ids.txt"), "w", encoding="utf-8") as f:
    for v in train_vids:
        f.write(f"{v}\n")

with open(os.path.join(BASE_DIR, "test_video_ids.txt"), "w", encoding="utf-8") as f:
    for v in test_vids:
        f.write(f"{v}\n")

print(f"Train videos: {len(train_vids)} | Test videos: {len(test_vids)}")
print(f"Train frames: {len(train_frames)} | Test frames: {len(test_frames)}")

# -------------------------
# Feature columns
# -------------------------
au_cols   = [c for c in df.columns if c.endswith("_r")]
pose_cols = [c for c in df.columns if c.startswith("pose_")]
gaze_cols = [c for c in df.columns if c.startswith("gaze_")]

feat_cols = au_cols + pose_cols + gaze_cols
if not feat_cols:
    raise ValueError("No features found (expected *_r, pose_*, gaze_*).")

# -------------------------
# Static aggregation (video-level)
# -------------------------
def aggregate_static(frames: pd.DataFrame) -> pd.DataFrame:
    X = frames.groupby("video_id")[feat_cols].agg(["mean", "std", "max"])
    X.columns = [f"{col}_{stat}" for col, stat in X.columns]
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()

train_static = aggregate_static(train_frames)
test_static  = aggregate_static(test_frames)

train_static.to_csv(os.path.join(BASE_DIR, "train_agg_static_v2.csv"), index=False)
test_static.to_csv(os.path.join(BASE_DIR, "test_agg_static_v2.csv"), index=False)
train_static.to_pickle(os.path.join(BASE_DIR, "train_agg_static.pkl"))
test_static.to_pickle(os.path.join(BASE_DIR, "test_agg_static.pkl"))

print("Saved static aggregated datasets.")

# -------------------------
# Optional: DAU (delta AU)
# -------------------------
def find_time_col(frames: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in frames.columns:
            return c
    raise ValueError(f"DAU requires a time column. None found in: {TIME_CANDIDATES}")

def add_dau(frames: pd.DataFrame) -> pd.DataFrame:
    time_col = find_time_col(frames)
    frames = frames.sort_values(["video_id", time_col]).copy()
    for c in au_cols:
        frames[f"d_{c}"] = frames.groupby("video_id")[c].diff().fillna(0)
    return frames

def aggregate_dau(frames: pd.DataFrame) -> pd.DataFrame:
    dau_cols = [f"d_{c}" for c in au_cols if f"d_{c}" in frames.columns]
    if not dau_cols:
        raise ValueError("No DAU columns found. Did add_dau() run?")
    X = frames.groupby("video_id")[dau_cols].agg(["mean", "std", "max"])
    X.columns = [f"{col}_{stat}" for col, stat in X.columns]
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()

if ENABLE_DAU:
    train_frames_d = add_dau(train_frames)
    test_frames_d  = add_dau(test_frames)

    train_dau = aggregate_dau(train_frames_d)
    test_dau  = aggregate_dau(test_frames_d)

    train_dau.to_csv(os.path.join(BASE_DIR, "train_agg_dau.csv"), index=False)
    test_dau.to_csv(os.path.join(BASE_DIR, "test_agg_dau.csv"), index=False)
    train_dau.to_pickle(os.path.join(BASE_DIR, "train_agg_dau.pkl"))
    test_dau.to_pickle(os.path.join(BASE_DIR, "test_agg_dau.pkl"))

    print("Saved DAU aggregated datasets.")

print("\nDONE.")
print("Static train/test rows:", len(train_static), len(test_static))
