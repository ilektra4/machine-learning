import os
import pandas as pd

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"
csv_path = os.path.join(BASE_DIR, "data", "kaggle_openface_merged.csv")

df = pd.read_csv(csv_path)

print("Σχήμα (rows, cols):", df.shape)
print("Γραμμές (rows):", len(df))
print("Στήλες (cols):", len(df.columns))

print("\nΟνόματα στηλών:")
print(df.columns.tolist()[:50])  # δείξε τις πρώτες 50 για να μην πλημμυρίσει
