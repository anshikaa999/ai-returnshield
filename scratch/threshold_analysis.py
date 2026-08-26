import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def run_threshold_analysis():
    # 1. Load dataset
    data_path = os.path.join("data", "raw", "return_requests.csv")
    df = pd.read_csv(data_path)

    TARGET = "is_suspicious"
    X = df.drop(columns=["customer_id", "order_id", TARGET])
    y = df[TARGET]

    # 2. Reproduce exact 70/15/15 split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    val_size_ratio = 15.0 / 85.0
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_ratio, stratify=y_train_val, random_state=42
    )

    # 3. Load saved XGBoost pipeline model
    model_path = os.path.join("models", "returnshield_pipeline.joblib")
    pipeline = joblib.load(model_path)

    # 4. Predict probabilities on Validation Set ONLY
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    total_val_samples = len(y_val)

    results = []
    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        # Flagged count & percentage of total validation return requests sent for manual review
        review_count = tp + fp
        review_pct = (review_count / total_val_samples) * 100.0

        results.append({
            "Threshold": thresh,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Review Count": review_count,
            "Review Rate (%)": review_pct
        })

    results_df = pd.DataFrame(results)
    print("=== VALIDATION SET THRESHOLD ANALYSIS ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    run_threshold_analysis()
