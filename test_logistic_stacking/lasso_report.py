import os
import numpy as np
import pandas as pd

from io_utils import load_parquet
from config import OUT_DIR, SEED
from run_experiments import load_feature_set, split_by_video, make_xy
from model_lasso import fit_lasso

def main(feature_set="temporal"):
    feats = load_feature_set(feature_set)
    tr, va, te = split_by_video(feats, seed=SEED)
    Xtr, ytr = make_xy(tr)

    model = fit_lasso(Xtr, ytr, seed=SEED)

    # get coefficients
    clf = model.named_steps["clf"]
    cols = Xtr.columns.to_list()
    coefs = clf.coef_[0]  # binary
    nz = np.sum(coefs != 0)

    df = pd.DataFrame({"feature": cols, "coef": coefs, "abs_coef": np.abs(coefs)})
    df = df.sort_values("abs_coef", ascending=False)

    out_csv = os.path.join(OUT_DIR, f"lasso_top_features_{feature_set}.csv")
    df.head(30).to_csv(out_csv, index=False)

    print("Non-zero coefficients:", int(nz), "out of", len(cols))
    print("Saved:", out_csv)
    print(df.head(15).to_string(index=False))

if __name__ == "__main__":
    main("temporal")
