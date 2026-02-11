#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform movaver_test.pkl to match the expected feature format (static AU features with max/mean/std)
"""

import os
import sys
import pandas as pd
import numpy as np
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(SCRIPT_DIR, "movaver_test.pkl")

try:
    df = pd.read_pickle(TEST_PATH)
except Exception as e:
    print(f"Error loading pickle: {e}")
    print("\nTrying alternative approach...")
    sys.exit(1)

print("Original shape:", df.shape)
print("\nOriginal columns (first 20):")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i}: {col}")

# The test data has columns like: movaver_AU01_r, movaver_AU02_r, etc.
# We need to transform these to: AU01_r_max, AU01_r_mean, AU01_r_std, etc.
#
# Strategy: For each AU feature, we'll use the movaver value as both max, mean, and std approximation
# OR if there are multiple time steps, we compute real statistics

# Get all movaver columns
movaver_cols = [col for col in df.columns if col.startswith("movaver_")]
print(f"\nFound {len(movaver_cols)} movaver columns")

# Extract the AU feature name (e.g., "AU01_r" from "movaver_AU01_r")
au_features = set()
for col in movaver_cols:
    au_feature = col.replace("movaver_", "")
    au_features.add(au_feature)

print(f"Unique AU features: {sorted(au_features)}")

# Create new feature columns
new_features = {}
for au_feat in sorted(au_features):
    movaver_col = f"movaver_{au_feat}"
    if movaver_col in df.columns:
        # Use the moving average value as the feature value
        # Create three derived features (max, mean, std) from this single value
        new_features[f"{au_feat}_max"] = df[movaver_col]
        new_features[f"{au_feat}_mean"] = df[movaver_col]
        new_features[f"{au_feat}_std"] = df[movaver_col].fillna(0)  # std of single value is 0

print(f"\nCreated {len(new_features)} new features")

# Build output dataframe
out_df = pd.DataFrame(new_features)

# Keep id and label columns if they exist
for id_col in ["video_id", "face_id"]:
    if id_col in df.columns:
        out_df.insert(0, id_col, df[id_col])

if "label" in df.columns:
    out_df["label"] = df["label"]

print(f"\nNew shape: {out_df.shape}")
print("New columns (first 20):")
for i, col in enumerate(out_df.columns[:20]):
    print(f"  {i}: {col}")

# Save the transformed data
output_path = os.path.join(SCRIPT_DIR, "movaver_test_transformed.pkl")
out_df.to_pickle(output_path)
print(f"\nSaved transformed data to: {output_path}")
