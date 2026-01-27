import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from joblib import dump

from skimage.feature import hog, local_binary_pattern

# ---------- Helpers ----------
def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s == "truth":
        return 0
    if s in {"deception", "lie"}:
        return 1
    if s.startswith("t"):
        return 0
    if s.startswith(("d", "l")):
        return 1
    return None

def pick_k_uniform(items, k):
    if k <= 0 or len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]

def load_gray_resized(path, size_wh):
    im = Image.open(path).convert("L")
    im = im.resize(size_wh, Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr

def frame_feature_hog(gray_01, orientations=9, pixels_per_cell=8, cells_per_block=2):
    return hog(
        gray_01,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)

def frame_feature_lbp_hist(gray_01, P=16, R=2, method="uniform"):
    # gray_01 in [0,1] -> convert to [0,255] uint8 for stable LBP
    img = (gray_01 * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(img, P=P, R=R, method=method)
    n_bins = P + 2 if method == "uniform" else int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    return hist.astype(np.float32)

def find_column(df_cols, contains_any):
    cols = list(df_cols)
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in contains_any):
            return c
    return None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to Dolos.xlsx")
    ap.add_argument("--rgb_dir", required=True, help="Path to rgb_frames_download folder")
    ap.add_argument("--out", default="features.joblib", help="Output features file (joblib)")
    ap.add_argument("--method", choices=["hog", "lbp"], default="hog")
    ap.add_argument("--k_frames", type=int, default=8, help="How many frames per clip (0=all)")
    ap.add_argument("--img_size", type=int, default=128, help="Resize (square) before features")
    # LBP params
    ap.add_argument("--lbp_P", type=int, default=16)
    ap.add_argument("--lbp_R", type=int, default=2)
    args = ap.parse_args()

    # Read DOLOS excel (your file has 2 header rows -> header=2 worked for you)
    df = pd.read_excel(args.xlsx, header=2)

    # Try to locate columns robustly
    file_col = find_column(df.columns, ["file name of the video clip"])
    label_col = find_column(df.columns, ['label "truth" or "deception"', "label"])
    part_col = find_column(df.columns, ["participants name", "participant"])

    if file_col is None or label_col is None:
        raise RuntimeError("Couldn't find required columns in Dolos.xlsx (file_name/label).")

    if part_col is None:
        # If participant missing, we can still split randomly, but leakage risk exists.
        part_col = "__no_participant__"
        df[part_col] = "UNKNOWN"

    df = df.rename(columns={file_col: "file_name"})
    df["y"] = df[label_col].apply(normalize_label)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # keep only clips that exist
    def clip_exists(name):
        p = os.path.join(args.rgb_dir, str(name).strip())
        return os.path.isdir(p)

    df = df[df["file_name"].apply(clip_exists)].copy()
    df = df[df[part_col].notna()].copy()

    print("Usable clips:", len(df))

    X_list, y_list, g_list, clip_list = [], [], [], []
    size_wh = (args.img_size, args.img_size)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {args.method.upper()}"):
        clip = str(row["file_name"]).strip()
        folder = os.path.join(args.rgb_dir, clip)

        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        imgs.sort()
        if not imgs:
            continue

        chosen = pick_k_uniform(imgs, args.k_frames)
        feats = []
        for fn in chosen:
            p = os.path.join(folder, fn)
            gray = load_gray_resized(p, size_wh)

            if args.method == "hog":
                feat = frame_feature_hog(gray)
            else:
                feat = frame_feature_lbp_hist(gray, P=args.lbp_P, R=args.lbp_R, method="uniform")

            feats.append(feat)

        clip_feat = np.mean(np.stack(feats, axis=0), axis=0)

        X_list.append(clip_feat)
        y_list.append(int(row["y"]))
        g_list.append(str(row[part_col]))
        clip_list.append(clip)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(g_list, dtype=object)
    clips = np.array(clip_list, dtype=object)

    print("Feature dim:", X.shape[1])
    print("Saved samples:", X.shape[0])

    dump({"X": X, "y": y, "groups": groups, "clips": clips}, args.out)
    print("Saved to:", args.out)

if __name__ == "__main__":
    main()
