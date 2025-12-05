# file: baseline_dropout.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def main():
    # 1. Load dataset
    path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
    file_name = "students_dropout_academic_success.csv"
    df = pd.read_csv(path / file_name)

    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    # 2. Create binary label: 1 = Dropout, 0 = Non‑dropout
    df["y"] = df["target"].map(
        {
            "Dropout": 1,
            "Graduate": 0,
            "Enrolled": 0,
        }
    )

    # Sanity check: label distribution
    print("\nLabel distribution (counts):")
    print(df["y"].value_counts())
    print("\nLabel distribution (proportions):")
    print(df["y"].value_counts(normalize=True))

    # 3. Define features and label
    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]

    # 4. Train/validation/test split (60 / 20 / 20), stratified
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

    print("\nSplit sizes:")
    print("Train:", X_train.shape[0])
    print("Val:  ", X_val.shape[0])
    print("Test: ", X_test.shape[0])

    # 5. Baseline model: Logistic Regression
    # class_weight='balanced' helps with class imbalance
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # 6. Evaluate on validation set (optional)
    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[:, 1]
    print("\n=== Validation performance (baseline LR) ===")
    print(classification_report(y_val, y_val_pred, digits=3))
    print("ROC-AUC (val):", roc_auc_score(y_val, y_val_prob))

    # 7. Final evaluation on test set
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]

    print("\n=== Test performance (baseline LR) ===")
    print(classification_report(y_test, y_test_pred, digits=3))
    print("ROC-AUC (test):", roc_auc_score(y_test, y_test_prob))

    # 8. Save basic results for later comparison
    results = {
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
    }
    np.savez(path / "baseline_lr_results.npz", **results)
    print("\nSaved test predictions to baseline_lr_results.npz")

if __name__ == "__main__":
    main()
