"""
Explainability Module for AI ReturnShield (Razorpay Track 02: AI Risk Manager)

Provides global and local model explainability using SHAP (SHapley Additive exPlanations)
and translates raw feature attributions into merchant-friendly risk explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD

class ReturnShieldExplainer:
    """
    Explainer wrapper around trained ReturnShield scikit-learn Pipeline.
    Computes SHAP feature attributions and formats human-readable risk summaries.
    """
    def __init__(self, pipeline_or_path):
        """
        Initializes explainer with a trained Pipeline object or joblib file path.
        """
        if isinstance(pipeline_or_path, str):
            self.pipeline = joblib.load(pipeline_or_path)
        else:
            self.pipeline = pipeline_or_path

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["classifier"]

        # Define columns
        self.num_cols = [
            "account_age_days", "total_orders", "previous_returns",
            "previous_refunds", "previous_chargebacks", "avg_order_value",
            "current_order_value", "days_to_return", "return_rate", "refund_amount"
        ]
        self.cat_cols = ["return_reason", "product_category"]

        # Extract transformed feature names
        ohe_features = self.preprocessor.named_transformers_["cat"] \
            .get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + ohe_features

        # Initialize SHAP TreeExplainer for XGBoost model
        self.explainer = shap.TreeExplainer(self.model)

    def preprocess_df(self, df):
        """Preprocesses raw feature DataFrame into transformed matrix."""
        cols_needed = self.num_cols + self.cat_cols
        X_raw = df[cols_needed]
        X_trans = self.preprocessor.transform(X_raw)
        return pd.DataFrame(X_trans, columns=self.feature_names, index=df.index)

    def get_shap_values(self, df):
        """Computes SHAP values for a DataFrame of return requests."""
        X_trans = self.preprocess_df(df)
        shap_vals = self.explainer(X_trans)
        return shap_vals, X_trans

    def get_global_importance(self, df):
        """Returns DataFrame of global feature importances ranked by mean absolute SHAP value."""
        shap_vals, _ = self.get_shap_values(df)
        mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)
        return importance_df

    def explain_request(self, raw_request_row, low_threshold=LOW_RISK_THRESHOLD, high_threshold=HIGH_RISK_THRESHOLD, top_k=3):
        """
        Generates a human-readable merchant explanation for a single return request.
        Exposes risk score, risk level, top risk factors, and protective factors.
        """
        if isinstance(raw_request_row, pd.Series):
            df_single = pd.DataFrame([raw_request_row])
        elif isinstance(raw_request_row, dict):
            df_single = pd.DataFrame([raw_request_row])
        else:
            df_single = raw_request_row.copy()

        # Compute risk score
        X_trans = self.preprocess_df(df_single)
        risk_score = float(self.pipeline.predict_proba(df_single)[0, 1])

        # Risk level assessment
        if risk_score >= high_threshold:
            risk_level = "HIGH RISK"
        elif risk_score >= low_threshold:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        # Compute single-instance SHAP values
        shap_vals = self.explainer(X_trans)
        val_array = shap_vals.values[0]

        contrib_df = pd.DataFrame({
            "feature_name": self.feature_names,
            "transformed_val": X_trans.iloc[0].values,
            "shap_val": val_array
        }).sort_values(by="shap_val", ascending=False)

        row = df_single.iloc[0]

        # Extract Risk Factors (positive SHAP impact pushing risk higher)
        risk_factors = []
        for idx, c_row in contrib_df[contrib_df["shap_val"] > 0.005].iterrows():
            feat = c_row["feature_name"]
            statement = self._format_feature_statement(feat, row, is_positive_risk=True)
            if statement and statement not in risk_factors:
                risk_factors.append(statement)
            if len(risk_factors) >= top_k:
                break

        if not risk_factors:
            risk_factors.append("Overall transaction parameters align with low risk baseline.")

        # Extract Protective Factors (negative SHAP impact lowering risk score)
        protective_factors = []
        for idx, c_row in contrib_df[contrib_df["shap_val"] < -0.005].sort_values(by="shap_val", ascending=True).iterrows():
            feat = c_row["feature_name"]
            statement = self._format_feature_statement(feat, row, is_positive_risk=False)
            if statement and statement not in protective_factors:
                protective_factors.append(statement)
            if len(protective_factors) >= top_k:
                break

        if not protective_factors:
            protective_factors.append("Limited historical positive order activity recorded.")

        explanation = {
            "customer_id": str(row.get("customer_id", "N/A")),
            "order_id": str(row.get("order_id", "N/A")),
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
            "disclaimer": "Explanations represent model-attributed risk indicators based on historical patterns, not definitive proof of customer fraud."
        }

        return explanation

    def _format_feature_statement(self, feature_name, row, is_positive_risk=True):
        """Translates technical feature names & values into merchant-friendly statements using INR currency."""
        avg_val = float(row.get("avg_order_value", 1000.0))
        curr_val = float(row.get("current_order_value", 1000.0))
        ratio = round(curr_val / max(1.0, avg_val), 2)

        if "return_rate" in feature_name:
            rate_pct = round(float(row.get("return_rate", 0)) * 100, 1)
            if is_positive_risk:
                return f"High historical return rate ({rate_pct}%)"
            else:
                return f"Healthy return rate history ({rate_pct}%)"

        elif "previous_chargebacks" in feature_name:
            cb = int(row.get("previous_chargebacks", 0))
            if is_positive_risk and cb > 0:
                return f"Previous chargeback disputes logged ({cb})"
            elif not is_positive_risk:
                return "Zero previous chargebacks on file"

        elif "current_order_value" in feature_name or "avg_order_value" in feature_name:
            if is_positive_risk and ratio > 1.3:
                return f"Current order value (₹{curr_val:.2f}) is {ratio}x customer average (₹{avg_val:.2f})"
            elif not is_positive_risk:
                return f"Order value (₹{curr_val:.2f}) consistent with typical spending (₹{avg_val:.2f})"

        elif "account_age_days" in feature_name:
            age = int(row.get("account_age_days", 0))
            if is_positive_risk and age < 30:
                return f"New customer account ({age} days old)"
            elif not is_positive_risk and age >= 60:
                return f"Established customer tenure ({age} days old)"

        elif "days_to_return" in feature_name:
            dtr = int(row.get("days_to_return", 0))
            if is_positive_risk and dtr <= 1:
                return f"Immediate return requested ({dtr} days after delivery)"
            elif not is_positive_risk and dtr > 7:
                return f"Standard return timeline ({dtr} days after delivery)"

        elif "previous_refunds" in feature_name or "previous_returns" in feature_name:
            prev_ret = int(row.get("previous_returns", 0))
            prev_ref = int(row.get("previous_refunds", 0))
            if is_positive_risk and prev_ref > 1:
                return f"Prior refund history ({prev_ref} refunds)"
            elif not is_positive_risk:
                return f"Low return history ({prev_ret} prior returns)"

        elif "total_orders" in feature_name:
            tot = int(row.get("total_orders", 0))
            if not is_positive_risk and tot >= 5:
                return f"Proven buyer profile ({tot} completed orders)"

        elif "return_reason" in feature_name:
            reason = str(row.get("return_reason", ""))
            clean_reason = reason.replace("_", " ").title()
            if is_positive_risk and reason in ["empty_box_claimed", "item_not_received"]:
                return f"High-risk claim reason ('{clean_reason}')"
            elif is_positive_risk:
                return f"Return reason: '{clean_reason}'"
            else:
                return f"Standard return reason ('{clean_reason}')"

        elif "product_category" in feature_name:
            cat = str(row.get("product_category", ""))
            clean_cat = cat.replace("_", " ").title()
            if is_positive_risk and cat in ["electronics", "jewelry"]:
                return f"High resale risk category ('{clean_cat}')"

        return None
