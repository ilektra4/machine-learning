from __future__ import annotations
import os
import numpy as np
import pandas as pd

from config import (
    OUT_DIR, VIDEO_COL, FACE_COL, LABEL_COL, FRAME_COL,
    MIN_FRAMES_PER_TRACK, FEAT_TEMPORAL_AU_Z
)
from io_utils import ensure_dir, load_parquet, save_parquet

def main():
    ensure_dir(OUT_DIR)

    frames_path = os.path.join(OUT_DIR, "dolos_frames_clean.parquet")
    df = load_parquet(frames_path)

    au_cols = [c for c in df.columns if c.endswith("_r")]
    if not au_cols:
        raise ValueError("No AU *_r columns found.")

    keys = [VIDEO_COL, FACE_COL, LABEL_COL]

    df = df.sort_values(keys + [FRAME_COL]).copy()
    grp = df.groupby(keys, sort=False)

    # z-score per track (avoid div0)
    for c in au_cols:
        mu = grp[c].transform("mean")
        sd = grp[c].transform("std")
        sd = sd.replace(0, np.nan)
        z = (df[c] - mu) / sd
        df[f"z_{c}"] = z.fillna(0)

    # diffs on z
    for c in au_cols:
        df[f"dz_{c}"] = grp[f"z_{c}"].diff().abs()

    dz_cols = [f"dz_{c}" for c in au_cols]

    out = grp[dz_cols].agg(["mean", "var", "max"]).reset_index()
    out.columns = ["_".join([x for x in col if x]) for col in out.columns.to_flat_index()]

    n_frames = grp.size().reset_index(name="n_frames")
    out = out.merge(n_frames, on=keys, how="left")
    out = out[out["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    out = out.fillna(0)

    out_path = os.path.join(OUT_DIR, FEAT_TEMPORAL_AU_Z)
    save_parquet(out, out_path)

    print("Saved:", out_path)
    print("Rows:", len(out), "Cols:", out.shape[1])

if __name__ == "__main__":
    main()
