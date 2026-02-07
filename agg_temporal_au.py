from __future__ import annotations
import os
import pandas as pd

from config import (
    OUT_DIR, VIDEO_COL, FACE_COL, LABEL_COL, FRAME_COL,
    MIN_FRAMES_PER_TRACK, FEAT_TEMPORAL_AU
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

    # diffs
    for c in au_cols:
        df[f"d_{c}"] = grp[c].diff().abs()

    d_cols = [f"d_{c}" for c in au_cols]

    out = grp[d_cols].agg(["mean", "var", "max"]).reset_index()
    out.columns = ["_".join([x for x in col if x]) for col in out.columns.to_flat_index()]

    n_frames = grp.size().reset_index(name="n_frames")
    out = out.merge(n_frames, on=keys, how="left")
    out = out[out["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    out = out.fillna(0)

    out_path = os.path.join(OUT_DIR, FEAT_TEMPORAL_AU)
    save_parquet(out, out_path)

    print("Saved:", out_path)
    print("Rows:", len(out), "Cols:", out.shape[1])

if __name__ == "__main__":
    main()
