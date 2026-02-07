from __future__ import annotations
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------
# Aggregation by video,face,label - keeping meaningful metrics instead of noisy plain frame rows
# --------------------------------------------------------------------------------------------------

from config import OUT_DIR, VIDEO_COL, FACE_COL, LABEL_COL, MIN_FRAMES_PER_TRACK
from io_utils import load_parquet, load_json, save_parquet

def aggregate_static(include_gaze_pose: bool = True, activation_thr: float = 1.0) -> str:
    df = load_parquet(os.path.join(OUT_DIR, "dolos_frames_clean.parquet"))
    feat = load_json(os.path.join(OUT_DIR, "feature_columns.json"))

    au_cols = feat["au_cols"]
    gp_cols = feat["gaze_pose_cols"] if include_gaze_pose else []

    keys = [VIDEO_COL, FACE_COL, LABEL_COL]

    base = df.groupby(keys).size().reset_index(name="n_frames")
    base = base[base["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    def agg_block(cols: list[str], prefix: str = "") -> pd.DataFrame:
        g = df.groupby(keys)[cols]
        out = pd.concat([
            g.mean().add_prefix(f"{prefix}mean_"),
            g.min().add_prefix(f"{prefix}min_"),
            g.max().add_prefix(f"{prefix}max_"),
            g.var().fillna(0).add_prefix(f"{prefix}var_"),
            g.apply(lambda x: (x > activation_thr).mean()).add_prefix(f"{prefix}act_"),
        ], axis=1).reset_index()
        return out

    blocks = [agg_block(au_cols, prefix="au_")]
    if gp_cols:
        g = df.groupby(keys)[gp_cols]
        gp = pd.concat([
            g.mean().add_prefix("gp_mean_"),
            g.min().add_prefix("gp_min_"),
            g.max().add_prefix("gp_max_"),
            g.var().fillna(0).add_prefix("gp_var_"),
        ], axis=1).reset_index()
        blocks.append(gp)

    out = base
    for b in blocks:
        out = out.merge(b, on=keys, how="left")

    out = out.fillna(0)
    out_path = os.path.join(OUT_DIR, "features_static.parquet")
    save_parquet(out, out_path)
    print("Saved:", out_path, "rows:", len(out), "cols:", out.shape[1])
    return out_path

if __name__ == "__main__":
    aggregate_static(include_gaze_pose=True, activation_thr=1.0)
