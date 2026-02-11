from __future__ import annotations
import os
import re
import pandas as pd


# -------------------------------------------------
# Load useful info
# -------------------------------------------------
from config import (
    DOLOS_OPENFACE_CSV, OUT_DIR,
    VIDEO_COL, FACE_COL, LABEL_COL, FRAME_COL, TIME_COL,
    REQUIRE_SUCCESS, MIN_CONFIDENCE, FEATURE_LIST_JSON
)

from io_utils import ensure_dir, save_parquet, save_json, safe_read_csv


# -------------------------------------------------
# Feature columns
# -------------------------------------------------
def build_feature_column_lists(df: pd.DataFrame) -> dict:
    au_cols = [c for c in df.columns if c.endswith("_r")]
    gaze_pose_cols = [c for c in df.columns if re.search(r"(gaze|pose_R|pose_T)", c)]
    return {"au_cols": au_cols, "gaze_pose_cols": gaze_pose_cols}


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess() -> None:

    ensure_dir(OUT_DIR)

    print("Loading OpenFace merged CSV...")
    df = safe_read_csv(DOLOS_OPENFACE_CSV)

    # -------------------------------------------------
    # Rename field
    # -------------------------------------------------
    if "file_name" not in df.columns:
        raise ValueError("Column 'file_name' not found in CSV.")

    df["video_id"] = df["file_name"].astype(str).str.replace(r"\.csv$", "", regex=True)

    # -------------------------------------------------
    # Checks
    # -------------------------------------------------
    if LABEL_COL not in df.columns:
        raise ValueError("Column 'label' not found in CSV.")

    if FACE_COL not in df.columns:
        print("face_id not found → face_id=0")
        df[FACE_COL] = 0

    # -------------------------------------------------
    # Filter feature quality
    # -------------------------------------------------
    if REQUIRE_SUCCESS and "success" in df.columns:
        df = df[df["success"] == 1]

    if "confidence" in df.columns:
        df = df[df["confidence"] >= MIN_CONFIDENCE]

    # -------------------------------------------------
    # Count Frames to use later
    # -------------------------------------------------
    if FRAME_COL not in df.columns:
        if TIME_COL in df.columns:
            print("Frame column missing → creating from timestamp ordering")
            df = df.sort_values(["video_id", FACE_COL, TIME_COL]).copy()
            df[FRAME_COL] = df.groupby(["video_id", FACE_COL]).cumcount() + 1
        else:
            raise ValueError("Need 'frame' or 'timestamp' column.")

    # -------------------------------------------------
    # Save clean data to reuse
    # -------------------------------------------------
    out_frames = os.path.join(OUT_DIR, "dolos_frames_clean.parquet")
    save_parquet(df, out_frames)

    # -------------------------------------------------
    # Save feature list to ensure uniformity later in tests
    # -------------------------------------------------
    feat_lists = build_feature_column_lists(df)
    out_features = os.path.join(OUT_DIR, FEATURE_LIST_JSON)
    save_json(feat_lists, out_features)

    print("\nDONE PREPROCESSING")
    print("Saved frames →", out_frames)
    print("Saved feature list →", out_features)
    print("AU columns:", len(feat_lists["au_cols"]))
    print("Gaze/Pose columns:", len(feat_lists["gaze_pose_cols"]))
    print("Videos:", df["video_id"].nunique())


if __name__ == "__main__":
    preprocess()
