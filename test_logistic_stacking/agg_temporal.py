from __future__ import annotations
import os
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------------
# Aggregation by video,face,label capturing change with time - keeping meaningful metrics instead of noisy plain frame rows
# -------------------------------------------------------------------------------------------------------------------------

from config import OUT_DIR, VIDEO_COL, FACE_COL, LABEL_COL, FRAME_COL, MIN_FRAMES_PER_TRACK
from io_utils import load_parquet, load_json, save_parquet

def aggregate_temporal(include_gaze_pose: bool = True) -> str:
    df = load_parquet(os.path.join(OUT_DIR, "dolos_frames_clean.parquet"))
    feat = load_json(os.path.join(OUT_DIR, "feature_columns.json"))

    au_cols = feat["au_cols"]
    gp_cols = feat["gaze_pose_cols"] if include_gaze_pose else []

    keys = [VIDEO_COL, FACE_COL, LABEL_COL]

    df = df.sort_values([VIDEO_COL, FACE_COL, FRAME_COL]).copy()
    g = df.groupby(keys, sort=False)

    for c in au_cols:
        df[f"d_{c}"] = g[c].diff()
        df[f"ad_{c}"] = df[f"d_{c}"].abs()

    for c in gp_cols:
        df[f"gd_{c}"] = g[c].diff()
        df[f"agd_{c}"] = df[f"gd_{c}"].abs()

    diff_au = [f"d_{c}" for c in au_cols]
    adiff_au = [f"ad_{c}" for c in au_cols]
    diff_gp = [f"gd_{c}" for c in gp_cols]
    adiff_gp = [f"agd_{c}" for c in gp_cols]

    base = df.groupby(keys).size().reset_index(name="n_frames")
    base = base[base["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    def agg_stats(cols: list[str], prefix: str) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(columns=keys)
        gg = df.groupby(keys)[cols]
        out = pd.concat([
            gg.mean().fillna(0).add_prefix(f"{prefix}mean_"),
            gg.min().fillna(0).add_prefix(f"{prefix}min_"),
            gg.max().fillna(0).add_prefix(f"{prefix}max_"),
            gg.var().fillna(0).add_prefix(f"{prefix}var_"),
        ], axis=1).reset_index()
        return out

    out = base
    out = out.merge(agg_stats(diff_au,  "au_d_"), on=keys, how="left")
    out = out.merge(agg_stats(adiff_au, "au_ad_"), on=keys, how="left")
    out = out.merge(agg_stats(diff_gp,  "gp_d_"), on=keys, how="left")
    out = out.merge(agg_stats(adiff_gp, "gp_ad_"), on=keys, how="left")

    out = out.fillna(0)
    out_path = os.path.join(OUT_DIR, "features_temporal.parquet")
    save_parquet(out, out_path)
    print("Saved:", out_path, "rows:", len(out), "cols:", out.shape[1])
    return out_path

if __name__ == "__main__":
    aggregate_temporal(include_gaze_pose=True)
