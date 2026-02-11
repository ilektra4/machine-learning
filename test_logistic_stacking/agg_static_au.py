from __future__ import annotations
import os
import pandas as pd

from config import (
    OUT_DIR, VIDEO_COL, FACE_COL, LABEL_COL,
    MIN_FRAMES_PER_TRACK, FEAT_STATIC_AU
)
from io_utils import ensure_dir, load_parquet, save_parquet

def main():
    ensure_dir(OUT_DIR)

    frames_path = os.path.join(OUT_DIR, "dolos_frames_clean.parquet")
    df = load_parquet(frames_path)

    # AU intensity columns
    au_cols = [c for c in df.columns if c.endswith("_r")]
    if not au_cols:
        raise ValueError("No AU *_r columns found.")

    keys = [VIDEO_COL, FACE_COL, LABEL_COL]
    grp = df.groupby(keys, sort=False)

    # base stats
    stat = grp[au_cols].agg(["mean", "var", "max"]).reset_index()
    stat.columns = ["_".join([x for x in col if x]) for col in stat.columns.to_flat_index()]

    # activation rate (% frames AU > 1)
    act = grp[au_cols].apply(lambda x: (x > 1).mean()).reset_index()
    act = act.rename(columns={c: f"act_{c}" for c in au_cols})

    # n_frames + filter
    n_frames = grp.size().reset_index(name="n_frames")
    out = stat.merge(n_frames, on=keys, how="left")
    out = out[out["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    out = out.merge(act, on=keys, how="left").fillna(0)

    out_path = os.path.join(OUT_DIR, FEAT_STATIC_AU)
    save_parquet(out, out_path)

    print("Saved:", out_path)
    print("Rows:", len(out), "Cols:", out.shape[1])

if __name__ == "__main__":
    main()
