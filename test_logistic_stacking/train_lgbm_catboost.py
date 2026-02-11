# train_lgbm_catboost.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# reuse EVERYTHING from your working pipeline
from xgboost_v2 import (
    build_video_table,
    split_train_valid_test,
    align_and_impute,
    find_best_threshold_f1,
    save_roc,
    FEATURE_MODE, FACE_MODE, SEED,
    DOLOS_FILE, KAGGLE_FILE, OUT_DIR
)

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def evaluate_model(model, name, Xtr,ytr,Xva,yva,Xte,yte,Xk,yk,feat_mode):
    print("\n============================")
    print("MODEL:", name)
    print("============================")

    model.fit(Xtr,ytr)

    # threshold from DOLOS validation
    p_val = model.predict_proba(Xva)[:,1]
    thr, f1_val = find_best_threshold_f1(yva, p_val)
    print("Best threshold:", round(thr,3),"F1:",round(f1_val,3))

    # DOLOS TEST
    p_te = model.predict_proba(Xte)[:,1]
    pred_te = (p_te>=thr).astype(int)

    print("\nDOLOS TEST")
    print("AUC:",roc_auc_score(yte,p_te))
    print("ACC:",accuracy_score(yte,pred_te))
    print(confusion_matrix(yte,pred_te))

    save_roc(yte,p_te,
        os.path.join(OUT_DIR,f"roc_dolos_{name}.png"),
        f"DOLOS ROC {name}")

    # KAGGLE GENERALIZATION
    p_k = model.predict_proba(Xk)[:,1]
    pred_k = (p_k>=thr).astype(int)

    print("\nKAGGLE GENERALIZATION")
    print("AUC:",roc_auc_score(yk,p_k))
    print("ACC:",accuracy_score(yk,pred_k))
    print(confusion_matrix(yk,pred_k))

    save_roc(yk,p_k,
        os.path.join(OUT_DIR,f"roc_kaggle_{name}.png"),
        f"Kaggle ROC {name}")



def main():
    os.makedirs(OUT_DIR,exist_ok=True)

    print("Building DOLOS features")
    dolos, feat_cols = build_video_table(DOLOS_FILE, FEATURE_MODE, FACE_MODE)
    train_df, valid_df, test_df = split_train_valid_test(dolos, seed=SEED)

    Xtr = train_df[feat_cols].to_numpy(float)
    Xva = valid_df[feat_cols].to_numpy(float)
    Xte = test_df[feat_cols].to_numpy(float)

    ytr = (train_df.label=="truth").astype(int).to_numpy()
    yva = (valid_df.label=="truth").astype(int).to_numpy()
    yte = (test_df.label=="truth").astype(int).to_numpy()

    print("Building Kaggle features")
    kag,_ = build_video_table(KAGGLE_FILE, FEATURE_MODE, FACE_MODE)
    kag = align_and_impute(train_df, kag, feat_cols)

    Xk = kag[feat_cols].to_numpy(float)
    yk = (kag.label=="truth").astype(int).to_numpy()

    # ---------------- LightGBM ----------------
    lgbm = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )
    evaluate_model(lgbm,"lightgbm",Xtr,ytr,Xva,yva,Xte,yte,Xk,yk,FEATURE_MODE)

    # ---------------- CatBoost ----------------
    cat = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False
    )
    evaluate_model(cat,"catboost",Xtr,ytr,Xva,yva,Xte,yte,Xk,yk,FEATURE_MODE)


if __name__ == "__main__":
    main()
