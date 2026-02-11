# split_and_aggregate.py
# Φορτώνει το cleaned dataset από C:\Users\ilekt\Downloads\dolos_outputs
# Κάνει split σε TRAIN/TEST σε επίπεδο video_id
# Κάνει aggregation (mean/std/max) για AU+pose+gaze
# (Προαιρετικά) υπολογίζει και DAU (delta AU) και κάνει aggregation και γι’ αυτά
#
# Outputs (στο ίδιο folder):
# - train_agg_static.csv / test_agg_static.csv
# - train_agg_static.pkl / test_agg_static.pkl
# - train_video_ids.txt / test_video_ids.txt
# - (αν ENABLE_DAU=True) train_agg_dau.csv / test_agg_dau.csv + pkl

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# Settings
# =========================
BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

# προτιμάμε parquet αν υπάρχει, αλλιώς pickle/csv
PARQUET_PATH = os.path.join(BASE_DIR, "dolos_frames_clean.parquet")
PKL_PATH     = os.path.join(BASE_DIR, "dolos_frames_clean.pkl")
CSV_PATH     = os.path.join(BASE_DIR, "dolos_frames_clean.csv")

TEST_SIZE = 0.2
SEED = 42

# Για DAU (delta AU): αν δεν το θες τώρα, βάλ’ το False
ENABLE_DAU = True

# για DAU χρειάζεται χρονική σειρά: προτιμά timestamp, αλλιώς frame
TIME_CANDIDATES = ["timestamp", "frame"]

# =========================
# Load (robust)
# =========================
def load_clean_frames() -> pd.DataFrame:
    if os.path.exists(PKL_PATH):
        print("Loading pickle:", PKL_PATH)
        return pd.read_pickle(PKL_PATH)

    if os.path.exists(PARQUET_PATH):
        print("Loading parquet:", PARQUET_PATH)
        try:
            return pd.read_parquet(PARQUET_PATH)
        except Exception as e:
            print("\n[WARN] Δεν μπόρεσα να διαβάσω parquet:", repr(e))
            print("Λύση: εγκατάστησε fastparquet ->  py -m pip install fastparquet")
            print("Ή σώσε/χρησιμοποίησε pickle (dolos_frames_clean.pkl).")
            raise

    if os.path.exists(CSV_PATH):
        print("Loading csv:", CSV_PATH)
        return pd.read_csv(CSV_PATH, low_memory=False)

    raise FileNotFoundError(
        "Δεν βρέθηκε κανένα cleaned αρχείο. Περίμενα ένα από:\n"
        f"- {PKL_PATH}\n- {PARQUET_PATH}\n- {CSV_PATH}"
    )

df = load_clean_frames()

# basic checks
need = ["video_id", "label", "success", "confidence"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Λείπουν βασικές στήλες: {missing}")

# =========================
# Split by VIDEO
# =========================
video_ids = df["video_id"].dropna().unique()
train_vids, test_vids = train_test_split(
    video_ids, test_size=TEST_SIZE, random_state=SEED, shuffle=True
)

train_frames = df[df["video_id"].isin(train_vids)].copy()
test_frames  = df[df["video_id"].isin(test_vids)].copy()

# save ids (για reproducibility)
with open(os.path.join(BASE_DIR, "train_video_ids.txt"), "w", encoding="utf-8") as f:
    for v in train_vids:
        f.write(str(v) + "\n")
with open(os.path.join(BASE_DIR, "test_video_ids.txt"), "w", encoding="utf-8") as f:
    for v in test_vids:
        f.write(str(v) + "\n")

print("Train videos:", len(train_vids), "| Test videos:", len(test_vids))
print("Train frames:", len(train_frames), "| Test frames:", len(test_frames))

# =========================
# Feature columns (AU + pose + gaze)
# =========================
au_cols   = [c for c in df.columns if c.endswith("_r")]
pose_cols = [c for c in df.columns if c.startswith("pose_")]
gaze_cols = [c for c in df.columns if c.startswith("gaze_")]

feat_cols = au_cols + pose_cols + gaze_cols
if not feat_cols:
    raise ValueError("Δεν βρέθηκαν feature columns (AU *_r / pose_ / gaze_).")

def aggregate_static(frames: pd.DataFrame) -> pd.DataFrame:
    X = frames.groupby("video_id")[feat_cols].agg(["mean", "std", "max"])
    X.columns = [f"{c}_{stat}" for c, stat in X.columns]  # flatten
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()

train_static = aggregate_static(train_frames)
test_static  = aggregate_static(test_frames)

# save static
train_static.to_csv(os.path.join(BASE_DIR, "train_agg_static_v2.csv"), index=False)
test_static.to_csv(os.path.join(BASE_DIR, "test_agg_static_v2.csv"), index=False)
train_static.to_pickle(os.path.join(BASE_DIR, "train_agg_static.pkl"))
test_static.to_pickle(os.path.join(BASE_DIR, "test_agg_static.pkl"))

print("Saved static aggregated datasets.")

# =========================
# Optional: DAU (delta AU) aggregation
# =========================
def find_time_col(frames: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in frames.columns:
            return c
    raise ValueError(f"Για DAU χρειάζομαι μία στήλη χρόνου. Δεν βρέθηκε καμία από: {TIME_CANDIDATES}")

def add_dau(frames: pd.DataFrame) -> pd.DataFrame:
    time_col = find_time_col(frames)
    frames = frames.sort_values(["video_id", time_col]).copy()

    # δημιουργεί d_AUxx_r = διαφορά από προηγούμενο frame μέσα στο ίδιο video
    for c in au_cols:
        frames[f"d_{c}"] = frames.groupby("video_id")[c].diff().fillna(0)

    return frames

def aggregate_dau(frames: pd.DataFrame) -> pd.DataFrame:
    # aggregates πάνω στα d_AU (συνήθως mean/std/max είναι χρήσιμα)
    dau_cols = [f"d_{c}" for c in au_cols if f"d_{c}" in frames.columns]
    if not dau_cols:
        raise ValueError("Δεν βρέθηκαν DAU columns. Κάτι πήγε στραβά στο add_dau().")

    X = frames.groupby("video_id")[dau_cols].agg(["mean", "std", "max"])
    X.columns = [f"{c}_{stat}" for c, stat in X.columns]
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()

if ENABLE_DAU:
    train_frames_d = add_dau(train_frames)
    test_frames_d  = add_dau(test_frames)

    train_dau = aggregate_dau(train_frames_d)
    test_dau  = aggregate_dau(test_frames_d)

    train_dau.to_csv(os.path.join(BASE_DIR, "train_agg_dau.csv"), index=False)
    test_dau.to_csv(os.path.join(BASE_DIR, "test_agg_dau.csv"), index=False)
    train_dau.to_pickle(os.path.join(BASE_DIR, "train_agg_dau.pkl"))
    test_dau.to_pickle(os.path.join(BASE_DIR, "test_agg_dau.pkl"))

    print("Saved DAU aggregated datasets.")

print("\nDONE.")
print("Static train/test rows:", len(train_static), len(test_static))
