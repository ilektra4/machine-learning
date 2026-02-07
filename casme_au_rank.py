import os
import numpy as np
import pandas as pd

CASME_CSV = r"D:\video_project\casme_ii_openface_merged.csv"
OUT_DIR = r"D:\video_project\dolos_pipeline_out"
TOP_K = 12  # change if you want more/less

def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score (robust to outliers)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CASME_CSV)
    # Ensure identifiers exist
    if "video_id" not in df.columns:
        if "file_name" in df.columns:
            df["video_id"] = df["file_name"].astype(str).str.replace(r"\.csv$", "", regex=True)
        else:
            df["video_id"] = "vid0"
    if "face_id" not in df.columns:
        df["face_id"] = 0

    # Basic cleaning: keep successful, confident frames if available
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.80]

    # Pick AU intensity columns (AU##_r)
    au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    if not au_cols:
        raise RuntimeError("No AU*_r columns found in CASME CSV.")

    # Sort for temporal diffs
    sort_cols = [c for c in ["video_id", "face_id", "frame", "timestamp"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Compute per-video AU stats
    # Motion metrics:
    #  - var: within-video variance
    #  - mean_abs_diff: mean |x_t - x_{t-1}| (frame-to-frame)
    #  - max: peak intensity (captures bursts)
    groups = df.groupby(["video_id", "face_id"], sort=False)

    rows = []
    for (vid, fid), g in groups:
        if len(g) < 3:
            continue
        x = g[au_cols].to_numpy(dtype=float)

        var = np.nanvar(x, axis=0)
        maxv = np.nanmax(x, axis=0)

        dx = np.abs(np.diff(x, axis=0))
        madiff = np.nanmean(dx, axis=0)

        rows.append({
            "video_id": vid,
            "face_id": fid,
            **{f"var_{c}": v for c, v in zip(au_cols, var)},
            **{f"madiff_{c}": v for c, v in zip(au_cols, madiff)},
            **{f"max_{c}": v for c, v in zip(au_cols, maxv)},
        })

    if not rows:
        raise RuntimeError("No valid groups found (check CASME CSV content).")

    per_vid = pd.DataFrame(rows)

    # Aggregate across videos (mean of each metric)
    var_cols = [c for c in per_vid.columns if c.startswith("var_AU")]
    mad_cols = [c for c in per_vid.columns if c.startswith("madiff_AU")]
    max_cols = [c for c in per_vid.columns if c.startswith("max_AU")]

    var_mean = per_vid[var_cols].mean(axis=0).to_numpy()
    mad_mean = per_vid[mad_cols].mean(axis=0).to_numpy()
    max_mean = per_vid[max_cols].mean(axis=0).to_numpy()

    # Robust standardize each metric then combine
    zv = robust_z(var_mean)
    zd = robust_z(mad_mean)
    zx = robust_z(max_mean)

    # Motion score: weighted combo (tweak weights if you want)
    score = 0.45 * zd + 0.35 * zv + 0.20 * zx

    # Build ranking table
    # var_cols are like "var_AU12_r" -> AU12_r
    aus = [c.replace("var_", "") for c in var_cols]

    rank = pd.DataFrame({
        "au": aus,
        "score": score,
        "z_madiff": zd,
        "z_var": zv,
        "z_max": zx,
        "mean_var": var_mean,
        "mean_madiff": mad_mean,
        "mean_max": max_mean,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(OUT_DIR, "casme_au_motion_rank.csv")
    rank.to_csv(out_csv, index=False)

    top = rank.head(TOP_K)["au"].tolist()
    out_txt = os.path.join(OUT_DIR, "casme_top_aus.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for au in top:
            f.write(au + "\n")

    print(f"Saved AU ranking: {out_csv}")
    print(f"Saved top AUs list: {out_txt}")
    print("\nTop AUs by CASME motion score:")
    for i, au in enumerate(top, 1):
        print(f"{i:02d}. {au}")

if __name__ == "__main__":
    main()
