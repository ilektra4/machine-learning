import os
import pandas as pd

# Φάκελος που είναι τα movaver_*.pkl
BASE_DIR = r"C:\Users\ilekt\Downloads\final_dolos"

def convert(name: str):
    pkl_path = os.path.join(BASE_DIR, f"{name}.pkl")
    csv_path = os.path.join(BASE_DIR, f"{name}.csv")

    print(f">>> Φορτώνω {pkl_path}")
    df = pd.read_pickle(pkl_path)
    print(f"{name} shape:", df.shape)

    print(f">>> Σώζω σε {csv_path}")
    df.to_csv(csv_path, index=False)
    print("OK.\n")

def main():
    convert("movaver_train")
    convert("movaver_test")
    print("ΤΕΛΟΣ - και τα δύο έγιναν CSV.")

if __name__ == "__main__":
    main()
