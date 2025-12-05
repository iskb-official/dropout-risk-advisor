# file: dropout_models_step2.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# You may need: pip install xgboost
from xgboost import XGBClassifier


def load_data():
    path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
    file_name = "students_dropout_academic_success.csv"
    df = pd.read_csv(path / file_name)

    # Binary label: 1 = Dropout, 0 = Non‑dropout
    df["y"] = df["target"].map(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0,
        }
    )

    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]

    return X, y, feature_cols, path


def make_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(name, model, X_val, y_val, X_test, y_test):
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    print(f"\n=== {name} – Validation ===")
    print(classification_report(y_val, y_val_pred, digits=3))
    print("ROC-AUC (val):", roc_auc_score(y_val, y_val_prob))

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} – Test ===")
    print(classification_report(y_test, y_test_pred, digits=3))
    print("ROC-AUC (test):", roc_auc_score(y_test, y_test_prob))

    return y_test_pred, y_test_prob


def main():
    X, y, feature_cols, path = load_data()

    print("Data shape:", X.shape)
    print("Positive class proportion (dropout):", y.mean())

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    print("\nSplit sizes:")
    print("Train:", X_train.shape[0])
    print("Val:  ", X_val.shape[0])
    print("Test: ", X_test.shape[0])

    # 1) Logistic Regression baseline (same as before, but inside this script)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    lr_test_pred, lr_test_prob = evaluate_model(
        "Logistic Regression", lr, X_val, y_val, X_test, y_test
    )

    # 2) XGBoost model – tuned modestly, not overfitted
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        tree_method="hist",  # fast on CPU
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    xgb_test_pred, xgb_test_prob = evaluate_model(
        "XGBoost", xgb, X_val, y_val, X_test, y_test
    )

    # 3) Save predictions and feature importances for later steps
    np.savez(
        path / "models_step2_predictions.npz",
        y_test=y_test.values,
        lr_pred=lr_test_pred,
        lr_prob=lr_test_prob,
        xgb_pred=xgb_test_pred,
        xgb_prob=xgb_test_prob,
    )
    print("\nSaved predictions to models_step2_predictions.npz")

    # Save XGBoost feature importances with names (for early interpretability)
    importances = xgb.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)
    fi_path = path / "xgb_feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    print("Saved XGBoost feature importances to", fi_path)


if __name__ == "__main__":
    main()
