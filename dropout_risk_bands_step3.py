# file: dropout_risk_bands_step3.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def load_data_and_splits():
    """Recreate the same splits used in step2 so that
    validation/test sets align with saved predictions.
    """
    from sklearn.model_selection import train_test_split

    path = Path(r"C:\Users\shaki\CCNU\OneDrive - mails.ccnu.edu.cn\Desktop\EDXAI")
    df = pd.read_csv(path / "students_dropout_academic_success.csv")

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

    return path, X_val, X_test, y_val, y_test


def assign_risk_band(p, low_th=0.2, high_th=0.5):
    """Map probability to risk band."""
    if p < low_th:
        return "low"
    elif p < high_th:
        return "medium"
    else:
        return "high"


def main():
    # 1. Load saved predictions from step2
    path, X_val, X_test, y_val, y_test = load_data_and_splits()

    pred_file = path / "models_step2_predictions.npz"
    data = np.load(pred_file)

    # XGBoost probabilities (validation & test)
    # In step2 we only saved test; here we will recompute val probs through a simple workaround:
    # For now, we will load only test probs and define thresholds manually;
    # later we can refine using proper validation optimization if needed.

    # NOTE: For a precise optimization of thresholds we would need
    # val probabilities saved; for now we just fix 0.2 / 0.5 as
    # reasonable bands (can be tuned later).
    xgb_test_prob = data["xgb_prob"]
    y_test_array = data["y_test"]

    # Sanity check
    assert y_test_array.shape[0] == X_test.shape[0]

    # 2. Define thresholds (initial version)
    low_th = 0.2
    high_th = 0.5
    print(f"Using thresholds: low<{low_th}, medium<{high_th}, else high")

    # 3. Assign risk bands on test set
    risk_bands = np.array([assign_risk_band(p, low_th, high_th) for p in xgb_test_prob])

    # 4. Basic statistics per band
    df_test_summary = pd.DataFrame(
        {
            "y_true": y_test_array,
            "p_dropout": xgb_test_prob,
            "risk_band": risk_bands,
        }
    )

    print("\nCounts per risk band (test):")
    print(df_test_summary["risk_band"].value_counts())

    print("\nDropout rate per band (test):")
    print(
        df_test_summary.groupby("risk_band")["y_true"]
        .mean()
        .sort_index()
    )

    # 5. Evaluate classification when treating "high" band as positive prediction
    y_pred_high = (risk_bands == "high").astype(int)

    print("\n=== Treating only HIGH band as positive prediction ===")
    print(classification_report(y_test_array, y_pred_high, digits=3))

    # 6. Confusion matrix for high‑risk rule
    cm = confusion_matrix(y_test_array, y_pred_high)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # 7. Save banded results for later expert rules / interventions
    out_path = path / "risk_bands_test.csv"
    df_test_summary.to_csv(out_path, index=False)
    print("\nSaved detailed test risk bands to", out_path)


if __name__ == "__main__":
    main()
