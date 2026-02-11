import matplotlib
matplotlib.use("Agg")  

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = r"D:\video_project\dolos_eda"
CSV_PATH = r"D:\video_project\dolos_openface_merged_final.csv"

os.makedirs(OUT_DIR, exist_ok=True)

print("Reading DOLOS CSV...")
df = pd.read_csv(CSV_PATH)

# ----------------------------
# Label creation
# ----------------------------
def label_from_name(name):
    n = str(name).lower()
    if "truth" in n:
        return "truth"
    if "lie" in n:
        return "lie"
    return None

name_col = "file_name" if "file_name" in df.columns else "video_id"
df["label"] = df[name_col].apply(label_from_name)
df = df.dropna(subset=["label"])

print(df["label"].value_counts())

# ----------------------------
# Keep only successful frames
# ----------------------------
if "success" in df.columns:
    df = df[df["success"] == 1]
if "confidence" in df.columns:
    df = df[df["confidence"] >= 0.80]

# ----------------------------
# Keep only numeric OpenFace features
# ----------------------------
non_features = ["frame","timestamp","face_id","confidence","success","label",name_col]
features = [c for c in df.columns if c not in non_features and df[c].dtype != "object"]

print("Feature count:", len(features))

# =====================================================
# 1) CLASS BALANCE PLOT
# =====================================================
plt.figure()
df["label"].value_counts().plot(kind="bar")
plt.title("Class distribution (frame level)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
plt.close()

# =====================================================
# 2) AU DISTRIBUTIONS (Top 10 variance features)
# =====================================================
var_series = df[features].var().sort_values(ascending=False)
top_feats = var_series.head(10).index

for feat in top_feats:
    plt.figure()
    sns.kdeplot(data=df, x=feat, hue="label", fill=True)
    plt.title(f"Distribution: {feat}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dist_{feat}.png", dpi=150)
    plt.close()

# =====================================================
# 3) CORRELATION HEATMAP
# =====================================================
sample = df[features].sample(5000, random_state=42)
corr = sample.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# =====================================================
# 4) PCA VISUALIZATION (pattern discovery)
# =====================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[features].sample(10000, random_state=42)
y = df.loc[X.index, "label"]

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)

plt.figure()
sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=y, alpha=0.4)
plt.title("PCA projection of frames")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_projection.png", dpi=150)
plt.close()

# =====================================================
# 5) OUTLIER DETECTION (Isolation Forest)
# =====================================================
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42)
preds = iso.fit_predict(X_scaled)

plt.figure()
sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=(preds==-1), alpha=0.4)
plt.title("Outlier detection (Isolation Forest)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/outliers.png", dpi=150)
plt.close()

print("\nEDA DONE → check:", OUT_DIR)
