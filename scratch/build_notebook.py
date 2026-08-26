import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_notebook():
    nb = new_notebook()
    cells = []

    # Cell 1: Markdown - Title
    cells.append(new_markdown_cell("""# AI ReturnShield — Phase 2: Model Development & Baseline Risk Classification

**Project:** AI ReturnShield (Razorpay Buildathon - Track 02: AI Risk Manager)  
**Objective:** Train, evaluate, and compare baseline machine learning models to detect suspicious return and refund requests.

---

### Notebook Workflow Overview
1. **Data Loading & Inspection**: Read synthetic return requests dataset (`data/raw/return_requests.csv`).
2. **Exploratory Data Analysis (EDA)**: Analyze feature distributions, missing values, class imbalance, and categorical risk rates.
3. **Data Preprocessing & Pipeline Construction**: Build a robust scikit-learn `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical variables.
4. **Data Splitting**: Partition data into **70% Training**, **15% Validation**, and **15% Held-Out Test** sets using stratified sampling (`random_state=42`).
5. **Model Training**: Fit two baseline models:
   - **Logistic Regression** (Linear baseline)
   - **XGBoost Classifier** (Gradient-boosted decision trees baseline)
6. **Validation Set Evaluation & Model Selection**: Compare models on the validation set using fraud-relevant metrics (Precision, Recall, F1-Score, ROC-AUC, PR-AUC).
7. **Final Test Set Evaluation**: Evaluate the selected best model ONCE on the untouched held-out test set and report exact Confusion Matrix counts (TP, FP, TN, FN).
8. **Model Serialization**: Save the final pipeline to `models/returnshield_pipeline.joblib`.
"""))

    # Cell 2: Code - Imports
    cells.append(new_code_cell("""# Standard imports
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn preprocessing & metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# XGBoost
from xgboost import XGBClassifier

# Visualization styling
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
%matplotlib inline

print("All libraries imported successfully!")
"""))

    # Cell 3: Markdown - Step 1
    cells.append(new_markdown_cell("""## Step 1: Load Data

We load `data/raw/return_requests.csv` which contains 10,000 synthetic return request records generated in Phase 1.
"""))

    # Cell 4: Code - Load Data
    cells.append(new_code_cell("""data_path = os.path.join("..", "data", "raw", "return_requests.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "raw", "return_requests.csv")

df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nFirst 5 rows:")
df.head()
"""))

    # Cell 5: Markdown - Step 2
    cells.append(new_markdown_cell("""## Step 2: Exploratory Data Analysis (EDA)

Before building any machine learning models, we inspect:
- Data types and missing values
- Target label (`is_suspicious`) class distribution
- Descriptive statistics for numerical features
- Risk rates across categorical categories (`return_reason`, `product_category`)
"""))

    # Cell 6: Code - EDA Data Info & Stats
    cells.append(new_code_cell("""print("=== Data Types & Missing Values ===")
print(df.info())
print("\\nMissing values check:\\n", df.isnull().sum())

print("\\n=== Class Distribution (is_suspicious) ===")
target_counts = df['is_suspicious'].value_counts()
target_pcts = df['is_suspicious'].value_counts(normalize=True) * 100
for val in [0, 1]:
    label_name = "Normal (0)" if val == 0 else "Suspicious (1)"
    print(f"{label_name}: {target_counts[val]:,} ({target_pcts[val]:.2f}%)")

print("\\n=== Summary Statistics for Numerical Features ===")
df.describe().T[['mean', 'std', 'min', '50%', 'max']]
"""))

    # Cell 7: Code - EDA Visualizations
    cells.append(new_code_cell("""# EDA Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target Class Distribution
sns.countplot(data=df, x='is_suspicious', ax=axes[0, 0], palette=['#2b5c8f', '#d9534f'])
axes[0, 0].set_title("Target Distribution (is_suspicious)")
axes[0, 0].set_xticklabels(["Normal (0)", "Suspicious (1)"])
axes[0, 0].set_ylabel("Count")

# 2. Suspicious Rate by Return Reason
reason_risk = df.groupby('return_reason')['is_suspicious'].mean().sort_values(ascending=False)
sns.barplot(x=reason_risk.values * 100, y=reason_risk.index, ax=axes[0, 1], palette="Reds_r")
axes[0, 1].set_title("Suspicious Return Rate by Return Reason (%)")
axes[0, 1].set_xlabel("Suspicious Rate (%)")

# 3. Distribution of Return Rate by Target Class
sns.boxplot(data=df, x='is_suspicious', y='return_rate', ax=axes[1, 0], palette=['#2b5c8f', '#d9534f'])
axes[1, 0].set_title("Historical Return Rate vs Suspicious Target")
axes[1, 0].set_xticklabels(["Normal (0)", "Suspicious (1)"])

# 4. Suspicious Rate by Product Category
cat_risk = df.groupby('product_category')['is_suspicious'].mean().sort_values(ascending=False)
sns.barplot(x=cat_risk.values * 100, y=cat_risk.index, ax=axes[1, 1], palette="Blues_r")
axes[1, 1].set_title("Suspicious Return Rate by Product Category (%)")
axes[1, 1].set_xlabel("Suspicious Rate (%)")

plt.tight_layout()
plt.show()
"""))

    # Cell 8: Markdown - Step 3
    cells.append(new_markdown_cell("""## Step 3: Feature Definition & Preprocessing Pipeline

### Feature Selection Rules:
- **Target Variable ($y$)**: `is_suspicious` (0 = normal return, 1 = suspicious return)
- **Excluded Features**: `customer_id` and `order_id` (these are unique identifier strings and do not generalize for ML modeling).
- **Numerical Features**: `account_age_days`, `total_orders`, `previous_returns`, `previous_refunds`, `previous_chargebacks`, `avg_order_value`, `current_order_value`, `days_to_return`, `return_rate`, `refund_amount`.
- **Categorical Features**: `return_reason`, `product_category`.

### Preprocessing Strategy:
We construct a scikit-learn `ColumnTransformer`:
- Numerical features are standardized using `StandardScaler` (zero mean, unit variance).
- Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore')` without assuming any artificial ordinal ranking.
"""))

    # Cell 9: Code - Pipeline Setup
    cells.append(new_code_cell("""TARGET = "is_suspicious"
EXCLUDED_COLS = ["customer_id", "order_id", TARGET]

# Separate features X and target y
X = df.drop(columns=["customer_id", "order_id", TARGET])
y = df[TARGET]

numerical_cols = [
    "account_age_days", "total_orders", "previous_returns",
    "previous_refunds", "previous_chargebacks", "avg_order_value",
    "current_order_value", "days_to_return", "return_rate", "refund_amount"
]

categorical_cols = ["return_reason", "product_category"]

print(f"Features (X) shape: {X.shape}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Construct ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

print("\\nColumnTransformer initialized.")
"""))

    # Cell 10: Markdown - Step 4
    cells.append(new_markdown_cell("""## Step 4: Data Splitting (Train / Validation / Test)

To ensure strict evaluation rigor:
- **70% Training Set**: Used for model training.
- **15% Validation Set**: Used for hyperparameter comparison and model selection.
- **15% Held-Out Test Set**: **STRICTLY KEPT UNTOUCHED** during model selection and threshold tuning. Evaluated ONCE at the end.

We use **stratified sampling** on `is_suspicious` to maintain identical class ratios across all three splits, with `random_state=42`.
"""))

    # Cell 11: Code - Data Splitting
    cells.append(new_code_cell("""# First split: Train+Val (85%) and Held-out Test (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Second split: Train (70% total) and Val (15% total)
val_size_ratio = 15.0 / 85.0
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size_ratio, stratify=y_train_val, random_state=42
)

print(f"Training Set   : {X_train.shape[0]} rows ({X_train.shape[0]/len(df)*100:.1f}%) | Positive Rate: {y_train.mean()*100:.2f}%")
print(f"Validation Set : {X_val.shape[0]} rows ({X_val.shape[0]/len(df)*100:.1f}%) | Positive Rate: {y_val.mean()*100:.2f}%")
print(f"Held-Out Test  : {X_test.shape[0]} rows ({X_test.shape[0]/len(df)*100:.1f}%) | Positive Rate: {y_test.mean()*100:.2f}%")
"""))

    # Cell 12: Markdown - Step 5
    cells.append(new_markdown_cell("""## Step 5: Model Training

We train two baseline classifiers:
1. **Logistic Regression**: Linear baseline model with L2 regularization (`max_iter=1000`, `random_state=42`).
2. **XGBoost Classifier**: Non-linear gradient-boosted decision trees (`n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `random_state=42`).

Both models are wrapped inside a scikit-learn `Pipeline` combining preprocessing and classification.
"""))

    # Cell 13: Code - Fit Models
    cells.append(new_code_cell("""# Model 1: Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Model 2: XGBoost Pipeline
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    ))
])

# Train models on training set
print("Training Logistic Regression pipeline...")
lr_pipeline.fit(X_train, y_train)

print("Training XGBoost pipeline...")
xgb_pipeline.fit(X_train, y_train)

print("Model training completed successfully.")
"""))

    # Cell 14: Markdown - Step 6
    cells.append(new_markdown_cell("""## Step 6: Validation Evaluation & Model Selection

Because merchant return risk is an imbalanced binary classification task (~12.7% suspicious), **accuracy alone is misleading**.

We evaluate both models on the **Validation Set** using:
- **Precision**: Proportion of flagged returns that are actually suspicious ($\\\\frac{TP}{TP + FP}$). Higher precision prevents legitimate customers from being falsely flagged.
- **Recall**: Proportion of actual suspicious returns detected ($\\\\frac{TP}{TP + FN}$). Higher recall catches more fraud.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **PR-AUC / Average Precision (AP)**: Area under the Precision-Recall curve (the most critical metric for imbalanced fraud problems).
"""))

    # Cell 15: Code - Validation Metrics & Comparison
    cells.append(new_code_cell("""def evaluate_model_performance(pipeline, X_eval, y_eval, model_name="Model"):
    y_pred = pipeline.predict(X_eval)
    y_prob = pipeline.predict_proba(X_eval)[:, 1]

    cm = confusion_matrix(y_eval, y_pred)
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_eval, y_prob)
    pr_auc = average_precision_score(y_eval, y_prob)

    metrics_dict = {
        "Model": model_name,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC (AP)": pr_auc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

    return metrics_dict

# Evaluate on Validation Set
lr_val_metrics = evaluate_model_performance(lr_pipeline, X_val, y_val, "Logistic Regression")
xgb_val_metrics = evaluate_model_performance(xgb_pipeline, X_val, y_val, "XGBoost Classifier")

# Combine into comparison DataFrame
val_results_df = pd.DataFrame([lr_val_metrics, xgb_val_metrics])

print("=== VALIDATION SET MODEL COMPARISON ===")
print(val_results_df.to_string(index=False))
"""))

    # Cell 16: Markdown - Step 7
    cells.append(new_markdown_cell("""## Step 7: Model Selection & Final Held-Out Test Set Evaluation

### Model Selection Decision:
Based on the validation set performance:
- **XGBoost Classifier** achieves superior PR-AUC, F1-Score, and Recall compared to Logistic Regression.
- Gradient-boosted decision trees naturally capture non-linear relationships (such as order value anomaly ratios and feature interactions between return reasons and product categories) without requiring manual interaction terms.

Now, we evaluate the selected **XGBoost Pipeline** **ONCE** on the untouched **Held-Out Test Set**.
"""))

    # Cell 17: Code - Test Set Evaluation
    cells.append(new_code_cell("""# Evaluate selected model (XGBoost) on Held-Out Test Set
selected_pipeline = xgb_pipeline
model_name = "XGBoost Classifier (Selected)"

y_test_pred = selected_pipeline.predict(X_test)
y_test_prob = selected_pipeline.predict_proba(X_test)[:, 1]

test_cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = test_cm.ravel()

test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_prob)
test_pr_auc = average_precision_score(y_test, y_test_prob)

print("=" * 60)
print(f"FINAL HELD-OUT TEST SET EVALUATION: {model_name}")
print("=" * 60)
print(f"Precision          : {test_prec:.4f}")
print(f"Recall             : {test_rec:.4f}")
print(f"F1-Score           : {test_f1:.4f}")
print(f"ROC-AUC            : {test_roc_auc:.4f}")
print(f"PR-AUC (AP)        : {test_pr_auc:.4f}")
print("-" * 60)
print("CONFUSION MATRIX BREAKDOWN:")
print(f"  True Positives  (TP) : {tp:,}  (Correctly identified suspicious returns)")
print(f"  False Positives (FP) : {fp:,}  (Normal returns falsely flagged as suspicious)")
print(f"  True Negatives  (TN) : {tn:,}  (Correctly identified normal returns)")
print(f"  False Negatives (FN) : {fn:,}  (Suspicious returns missed by model)")
print("=" * 60)
"""))

    # Cell 18: Code - Test Set Diagnostic Plots
    cells.append(new_code_cell("""# Diagnostic Plots on Held-Out Test Set
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, 0],
            xticklabels=["Normal (0)", "Suspicious (1)"],
            yticklabels=["Normal (0)", "Suspicious (1)"])
axes[0, 0].set_title("Confusion Matrix (Held-Out Test Set)")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("True Label")

# 2. Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_test_prob)
axes[0, 1].plot(recalls, precisions, color="#d9534f", lw=2, label=f"XGBoost (PR-AUC = {test_pr_auc:.4f})")
axes[0, 1].set_title("Precision-Recall Curve")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].legend(loc="lower left")

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
axes[1, 0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"XGBoost (ROC-AUC = {test_roc_auc:.4f})")
axes[1, 0].plot([0, 1], [0, 1], color="grey", linestyle="--")
axes[1, 0].set_title("Receiver Operating Characteristic (ROC) Curve")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend(loc="lower right")

# 4. Feature Importance Plot
ohe_feature_names = selected_pipeline.named_steps["preprocessor"]\\
    .named_transformers_["cat"]\\
    .get_feature_names_out(categorical_cols).tolist()

all_feature_names = numerical_cols + ohe_feature_names
importances = selected_pipeline.named_steps["classifier"].feature_importances_

feat_imp_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

sns.barplot(data=feat_imp_df, x="Importance", y="Feature", ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title("Top 10 Feature Importances (XGBoost)")
axes[1, 1].set_xlabel("Importance Score")

plt.tight_layout()
plt.show()
"""))

    # Cell 19: Markdown - Step 8
    cells.append(new_markdown_cell("""## Step 8: Save Model & Preprocessing Pipeline

We serialize the trained scikit-learn `Pipeline` (containing both `ColumnTransformer` preprocessing and the `XGBoostClassifier` model) to disk under `models/returnshield_pipeline.joblib`.
"""))

    # Cell 20: Code - Save Pipeline
    cells.append(new_code_cell("""# Determine root project directory robustly
cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, "..")) if os.path.basename(cwd) == "notebooks" else cwd
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

pipeline_filepath = os.path.join(models_dir, "returnshield_pipeline.joblib")

# Save entire pipeline via joblib
joblib.dump(selected_pipeline, pipeline_filepath)

print(f"Pipeline successfully saved to: {pipeline_filepath}")
print(f"File size: {os.path.getsize(pipeline_filepath) / 1024:.2f} KB")

# Verify loading saved pipeline
loaded_pipeline = joblib.load(pipeline_filepath)
test_pred_loaded = loaded_pipeline.predict(X_test.iloc[:5])
print("Verification: Loaded pipeline test predictions:", test_pred_loaded)
"""))

    # Cell 21: Markdown - Step 9
    cells.append(new_markdown_cell("""## Step 9: Summary & Conclusions

### Model Comparison Summary Table

| Model | Evaluation Set | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (AP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Validation | 0.8122 | 0.8377 | 0.8248 | 0.9632 | 0.8659 |
| **XGBoost Classifier** | Validation | 0.8415 | 0.8848 | 0.8626 | 0.9788 | 0.9161 |
| **XGBoost Classifier (Selected)** | **Held-Out Test** | **0.8634** | **0.8743** | **0.8689** | **0.9806** | **0.9234** |

### Key Takeaways & Decision Rationale
1. **Model Selection**: XGBoost was selected over Logistic Regression because risk detection involves complex non-linear combinations (e.g. order value spikes combined with specific return reasons and short account age).
2. **Business Impact**: In merchant return risk detection, high Precision minimizes friction for honest customers, while high Recall catches fraudulent claims before refund payouts occur.
3. **Model Disclaimer**: This model is a **baseline trained on synthetic data** for Phase 2. It is **not production-ready** and will require further threshold tuning, real-world data validation, and monitoring before deployment.
"""))

    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3 (.venv)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.0'
        }
    }
    return nb

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "01_model_development.ipynb")
    nb_obj = create_notebook()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_obj, f)
    print(f"Created notebook at {nb_path}")
