import os
import pandas as pd

# =========================
# Settings
# =========================
BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

KAGGLE_PATH = os.path.join(BASE_DIR, "kaggle_openface_merged.csv")

# Για DAU (delta AU)
ENABLE_DAU = True
TIME_CANDIDATES = ["timestamp", "frame"]


# =========================
# Load kaggle file
# =========================
def load_kaggle() -> pd.DataFrame:
    if not os.path.exists(KAGGLE_PATH):
        raise FileNotFoundError(f"Δεν βρήκα το αρχείο: {KAGGLE_PATH}")
    print("Loading kaggle csv:", KAGGLE_PATH)
    df = pd.read_csv(KAGGLE_PATH, low_memory=False)
    return df


# =========================
# Helpers for DAU
# =========================
def find_time_col(frames: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in frames.columns:
            return c
    raise ValueError(
        f"Για DAU χρειάζομαι μία στήλη χρόνου. "
        f"Δεν βρέθηκε καμία από: {TIME_CANDIDATES}"
    )


def add_dau(frames: pd.DataFrame, au_cols) -> pd.DataFrame:
    """
    Προσθέτει στήλες d_AUxx_r = διαφορά από προηγούμενο frame
    μέσα στο ίδιο video_id, ταξινομημένο χρονικά.
    """
    time_col = find_time_col(frames)
    frames = frames.sort_values(["video_id", time_col]).copy()

    for c in au_cols:
        frames[f"d_{c}"] = frames.groupby("video_id")[c].diff().fillna(0)

    return frames


def aggregate_static(frames: pd.DataFrame, feat_cols) -> pd.DataFrame:
    """
    Aggregation AU+pose+gaze: mean / std / max ανά video_id.
    """
    X = frames.groupby("video_id")[feat_cols].agg(["mean", "std", "max"])
    # flatten columns (multiindex -> simple names)
    X.columns = [f"{c}_{stat}" for c, stat in X.columns]
    # label ανά video (υποθέτουμε σταθερή ετικέτα ανά video)
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()


def aggregate_dau(frames: pd.DataFrame, au_cols) -> pd.DataFrame:
    """
    Aggregation πάνω στα d_AUxx_r: mean / std / max ανά video_id.
    """
    dau_cols = [f"d_{c}" for c in au_cols if f"d_{c}" in frames.columns]
    if not dau_cols:
        raise ValueError("Δεν βρέθηκαν DAU columns. Κάτι πήγε στραβά στο add_dau().")

    X = frames.groupby("video_id")[dau_cols].agg(["mean", "std", "max"])
    X.columns = [f"{c}_{stat}" for c, stat in X.columns]
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()


def main():
    df = load_kaggle()

    # basic checks
    need = ["video_id", "label", "success", "confidence"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Λείπουν βασικές στήλες: {missing}")
    print("Rows:", len(df))

    # =========================
    # Feature columns (AU + pose + gaze)
    # =========================
    au_cols   = [c for c in df.columns if c.endswith("_r")]
    pose_cols = [c for c in df.columns if c.startswith("pose_")]
    gaze_cols = [c for c in df.columns if c.startswith("gaze_")]

    feat_cols = au_cols + pose_cols + gaze_cols
    if not feat_cols:
        raise ValueError("Δεν βρέθηκαν feature columns (AU *_r / pose_ / gaze_).")

    print("AU cols:", len(au_cols), "| pose cols:", len(pose_cols), "| gaze cols:", len(gaze_cols))
    print("Total feature cols:", len(feat_cols))

    # =========================
    # STATIC aggregation (AU+pose+gaze)
    # =========================
    print(">>> Κάνω STATIC aggregation (AU+pose+gaze) ανά video_id...")
    kaggle_static = aggregate_static(df, feat_cols)
    print("Static aggregated shape:", kaggle_static.shape)

    static_csv = os.path.join(BASE_DIR, "kaggle_agg_static.csv")
    static_pkl = os.path.join(BASE_DIR, "kaggle_agg_static.pkl")
    kaggle_static.to_csv(static_csv, index=False)
    kaggle_static.to_pickle(static_pkl)
    print("Saved static aggregated kaggle to:")
    print(" -", static_csv)
    print(" -", static_pkl)

    # =========================
    # DAU aggregation (optional)
    # =========================
    if ENABLE_DAU:
        print(">>> Υπολογίζω DAU (delta AU) και κάνω aggregation ανά video_id...")
        df_dau = add_dau(df, au_cols)
        kaggle_dau = aggregate_dau(df_dau, au_cols)
        print("DAU aggregated shape:", kaggle_dau.shape)

        dau_csv = os.path.join(BASE_DIR, "kaggle_agg_dau.csv")
        dau_pkl = os.path.join(BASE_DIR, "kaggle_agg_dau.pkl")
        kaggle_dau.to_csv(dau_csv, index=False)
        kaggle_dau.to_pickle(dau_pkl)
        print("Saved DAU aggregated kaggle to:")
        print(" -", dau_csv)
        print(" -", dau_pkl)

    print("\nDONE.")


if __name__ == "__main__":
    main()
