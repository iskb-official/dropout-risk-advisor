# file: dropout_interventions_step4.py

from pathlib import Path

import numpy as np
import pandas as pd


LOW_TH = 0.2
HIGH_TH = 0.5
INTENSIVE_CAPACITY = 200  # max students for intensive mentoring per semester


def assign_risk_band(p: float) -> str:
    if p < LOW_TH:
        return "low"
    elif p < HIGH_TH:
        return "medium"
    else:
        return "high"


def basic_action_for_band(band: str) -> str:
    if band == "high":
        return "intensive_mentoring_candidate"
    elif band == "medium":
        return "workshop_monitoring"
    else:
        return "no_targeted_action"
    

def main():
    path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")

    # Load original data and recreate splits to get test indices
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path / "students_dropout_academic_success.csv")
    df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})

    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Load XGBoost test probabilities from step2
    preds = np.load(path / "models_step2_predictions.npz")
    xgb_test_prob = preds["xgb_prob"]
    y_test_array = preds["y_test"]

    assert len(xgb_test_prob) == X_test.shape[0]

    # Build dataframe for test set with probabilities
    df_test = X_test.copy()
    df_test = df_test.reset_index(drop=True)
    df_test["y_true"] = y_test_array
    df_test["p_dropout"] = xgb_test_prob

    # 1) Assign risk band
    df_test["risk_band"] = df_test["p_dropout"].apply(assign_risk_band)

    # 2) Initial action type by band
    df_test["action_initial"] = df_test["risk_band"].apply(basic_action_for_band)

    # 3) Apply capacity constraint within "high" band
    high_mask = df_test["risk_band"] == "high"
    df_high = df_test[high_mask].copy()

    # sort descending by dropout probability
    df_high = df_high.sort_values("p_dropout", ascending=False)
    df_high["action_final"] = "standard_advising"
    df_high.iloc[:INTENSIVE_CAPACITY, df_high.columns.get_loc("action_final")] = (
        "intensive_mentoring"
    )

    # merge back into main df_test
    df_test["action_final"] = "no_targeted_action"
    df_test.loc[df_test["risk_band"] == "medium", "action_final"] = "workshop_monitoring"
    df_test.loc[df_high.index, "action_final"] = df_high["action_final"]

    # 4) Simple textual rationale (placeholder – SHAP will refine later)
    def make_rationale(row):
        band = row["risk_band"]
        p = row["p_dropout"]
        if band == "high":
            return (
                f"High predicted dropout risk ({p:.2f}); "
                "prioritized for mentoring based on academic performance and workload."
            )
        elif band == "medium":
            return (
                f"Moderate dropout risk ({p:.2f}); recommended for workshops and monitoring."
            )
        else:
            return f"Low dropout risk ({p:.2f}); no targeted action recommended."

    df_test["rationale"] = df_test.apply(make_rationale, axis=1)

    # 5) Summary statistics
    print("\nCounts per action_final (test):")
    print(df_test["action_final"].value_counts())

    print("\nDropout rate per action_final (test):")
    print(
        df_test.groupby("action_final")["y_true"]
        .mean()
        .sort_index()
    )

    # 6) Save full decision table
    out_file = path / "dropout_decisions_test.csv"
    df_test.to_csv(out_file, index=False)
    print("\nSaved detailed test decisions to", out_file)


if __name__ == "__main__":
    main()
