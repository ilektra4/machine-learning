import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Φάκελος στο οποίο βρίσκεται το ίδιο το script
BASE_DIR = os.path.dirname(__file__)

# Το αρχικό dataset (πρέπει να είναι στον ίδιο φάκελο)
DATA_PATH = os.path.join(BASE_DIR, "movaver_dataset.pkl")

# Ποσοστό που θα πάει στο test
TEST_SIZE = 0.2      # 20%
RANDOM_STATE = 42    # για αναπαραγωγιμότητα

def main():
    print(">>> Φορτώνω το movaver_dataset.pkl από:", DATA_PATH)
    df = pd.read_pickle(DATA_PATH)
    print("Σχήμα dataset:", df.shape)

    # Αν υπάρχει στήλη 'label', κάνουμε stratify για να είναι balanced το split
    stratify_col = df["label"] if "label" in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_col
    )

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)

    # Αποθήκευση σε δύο νέα αρχεία
    train_path = os.path.join(BASE_DIR, "movaver_train.pkl")
    test_path  = os.path.join(BASE_DIR, "movaver_test.pkl")

    train_df.to_pickle(train_path)
    test_df.to_pickle(test_path)

    print("\n>>> Έσωσα:")
    print(" -", train_path)
    print(" -", test_path)
    print("ΤΕΛΟΣ.")

if __name__ == "__main__":
    main()
