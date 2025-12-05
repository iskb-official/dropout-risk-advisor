# file: dropout_shap_step5.py

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

import shap  # pip install shap


DATA_PATH = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
CSV_NAME = "students_dropout_academic_success.csv"
DECISION_FILE = "dropout_decisions_test.csv"  # from step4


def load_data():
    df = pd.read_csv(DATA_PATH / CSV_NAME)
    df["y"] = df["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})
    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    X = df[feature_cols]
    y = df["y"]
    return df, X, y, feature_cols


def make_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb(X_train, y_train, X_val, y_val):
    # Same hyperparameters as step2 to keep results consistent
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    # Quick check that performance is similar
    val_prob = xgb.predict_proba(X_val)[:, 1]
    y_val = y_val.values
    print("ROC-AUC (val, SHAP model):", roc_auc_score(y_val, val_prob))
    return xgb


def compute_shap_for_test(model, X_train, X_test):
    # TreeExplainer is ideal for XGBoost models
    explainer = shap.TreeExplainer(model)
    # Use a small background sample for speed if needed
    # explainer = shap.TreeExplainer(model, data=X_train.sample(500, random_state=42))
    shap_values = explainer.shap_values(X_test)
    base_values = explainer.expected_value
    return shap_values, base_values


def top_k_features_for_instance(shap_row, feature_names, k=3):
    # shap_row: 1D array of SHAP values for a single instance
    idx = np.argsort(-np.abs(shap_row))[:k]
    return [(feature_names[i], shap_row[i]) for i in idx]


def build_rationale(row, top_feats):
    p = row["p_dropout"]
    band = row["risk_band"]
    action = row["action_final"]

    risk_text = f"Predicted dropout risk {p:.2f} ({band}). "
    action_map = {
        "intensive_mentoring": "Prioritized for intensive mentoring and academic counseling. ",
        "standard_advising": "Recommended for standard academic advising. ",
        "workshop_monitoring": "Recommended for skills workshop and monitoring. ",
        "no_targeted_action": "No targeted intervention beyond general support. ",
    }
    action_text = action_map.get(action, "")

    # Build explanation from top features
    feat_parts = []
    for name, val in top_feats:
        direction = "increasing" if val > 0 else "reducing"
        feat_parts.append(f"{name} ({direction} risk)")
    expl_text = "Main drivers: " + "; ".join(feat_parts) + "."

    return risk_text + action_text + expl_text


def main():
    df, X, y, feature_cols = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    # 1. Train XGBoost model (same style as before)
    model = train_xgb(X_train, y_train, X_val, y_val)

    # 2. Compute SHAP values on test set
    shap_values, base_values = compute_shap_for_test(model, X_train, X_test)
    # shap_values has shape (n_test, n_features)
    print("SHAP shape:", shap_values.shape)

    # 3. Load decisions from step4 and align indices
    decisions = pd.read_csv(DATA_PATH / DECISION_FILE)
    decisions = decisions.reset_index(drop=True)
    assert decisions.shape[0] == X_test.shape[0]

    # 4. Build top-3 feature explanations per instance
    top_features_list = []
    for i in range(X_test.shape[0]):
        top_feats = top_k_features_for_instance(shap_values[i], feature_cols, k=3)
        top_features_list.append(top_feats)

    # 5. Add a new column with feature-based rationale
    detailed_rationales = []
    for i in range(decisions.shape[0]):
        row = decisions.iloc[i]
        top_feats = top_features_list[i]
        rationale = build_rationale(row, top_feats)
        detailed_rationales.append(rationale)

    decisions["rationale_shap"] = detailed_rationales

    # 6. Save enhanced decisions table
    out_file = DATA_PATH / "dropout_decisions_with_shap_test.csv"
    decisions.to_csv(out_file, index=False)
    print("Saved SHAP-enhanced decisions to", out_file)

    # Optional: quick global SHAP summary file (averaged abs values)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
    )
    global_importance.to_csv(DATA_PATH / "shap_global_importance.csv", index=False)
    print("Saved global SHAP importance to shap_global_importance.csv")


if __name__ == "__main__":
    main()
