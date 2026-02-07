from __future__ import annotations

# -------------------------------------------------
# File path
# -------------------------------------------------
DOLOS_OPENFACE_CSV = r"D:\video_project\dolos_openface_merged_final.csv"
KAGGLE_OPENFACE_CSV = r"D:\video_project\kaggle_openface_merged.csv"

OUT_DIR = r"D:\video_project\dolos_pipeline_out"

# -------------------------------------------------
# Useful columns
# -------------------------------------------------
VIDEO_COL = "video_id"
FACE_COL  = "face_id"
LABEL_COL = "label"
FRAME_COL = "frame"
TIME_COL  = "timestamp"

# -------------------------------------------------
# Quality Thresholds
# -------------------------------------------------
REQUIRE_SUCCESS = True
MIN_CONFIDENCE = 0.80
MIN_FRAMES_PER_TRACK = 30


# -------------------------------------------------
# Isolate AU features 
# -------------------------------------------------
FEAT_STATIC_AU   = "features_static_au.parquet"
FEAT_TEMPORAL_AU = "features_temporal_au.parquet"
FEAT_TEMPORAL_AU_Z = "features_temporal_au_z.parquet"

# -------------------------------------------------
# Feature list
# -------------------------------------------------
FEATURE_LIST_JSON = "feature_columns.json"

# -------------------------------------------------
# Label Ordering
# -------------------------------------------------
LABELS_ORDER = ["lie", "truth"]

# -------------------------------------------------
# Seed
# -------------------------------------------------
SEED = 42
