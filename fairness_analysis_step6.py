# file: fairness_analysis_step6.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
CSV_NAME = "students_dropout_academic_success.csv"
DECISIONS_FILE = "dropout_decisions_test.csv"  # from step4


def main():
    # 1. Load original data and recreate test split to get group attributes
    orig = pd.read_csv(DATA_PATH / CSV_NAME)
    orig["y"] = orig["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})

    feature_cols = [c for c in orig.columns if c not in ["target", "y"]]
    X = orig[feature_cols]
    y = orig["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Keep only the test meta info we need
    X_test_reset = X_test.reset_index(drop=True)
    meta = X_test_reset[["Gender", "Scholarship holder"]].copy()
    meta["y_true"] = y_test.reset_index(drop=True)

    # 2. Load decisions (must have risk_band and action_final, y_true in same order)
    decisions = pd.read_csv(DATA_PATH / DECISIONS_FILE).reset_index(drop=True)

    # Sanity check
    assert decisions.shape[0] == meta.shape[0], "Row count mismatch between decisions and test set."

    # Attach group attributes
    for col in ["Gender", "Scholarship holder"]:
        decisions[col] = meta[col]

    # 3. Define binary prediction: 1 if risk_band == 'high'
    decisions["y_pred_high"] = (decisions["risk_band"] == "high").astype(int)

    # 4. Function to compute group stats
    def group_fairness(df, attr):
        rows = []
        for g, sub in df.groupby(attr):
            tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred_high"]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            high_rate = sub["y_pred_high"].mean()
            intensive_rate = (sub["action_final"] == "intensive_mentoring").mean()
            rows.append(
                {
                    "group": g,
                    "n": len(sub),
                    "TPR_high": tpr,
                    "FPR_high": fpr,
                    "Pct_high_band": high_rate,
                    "Pct_intensive_mentoring": intensive_rate,
                }
            )
        return pd.DataFrame(rows)

    gender_stats = group_fairness(decisions, "Gender")
    schol_stats = group_fairness(decisions, "Scholarship holder")

    print("\n=== Fairness metrics by Gender (0 = male, 1 = female) ===")
    print(gender_stats)

    print("\n=== Fairness metrics by Scholarship holder (0 = no, 1 = yes) ===")
    print(schol_stats)

    # 5. Save to CSV for use in the paper
    gender_stats.to_csv(DATA_PATH / "fairness_gender_test.csv", index=False)
    schol_stats.to_csv(DATA_PATH / "fairness_scholarship_test.csv", index=False)
    print("\nSaved fairness summaries to fairness_gender_test.csv and fairness_scholarship_test.csv")


if __name__ == "__main__":
    main()
