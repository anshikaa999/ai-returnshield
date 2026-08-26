"""
Risk Assessment Service for AI ReturnShield Backend
(Razorpay Track 02: AI Risk Manager)

Encapsulates model inference, SHAP explainability generation,
and decision engine processing outside of API route handlers.
"""

from typing import Optional
import pandas as pd
from pathlib import Path

from src.config import MODEL_PATH, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from src.api.schemas import ReturnRiskRequest, ReturnRiskResponse
from src.evaluation.explainability import ReturnShieldExplainer
from src.models.decision_engine import evaluate_return_decision

class RiskAssessmentService:
    """
    Service responsible for loading the trained pipeline once during application startup
    and assessing individual return requests.
    """
    _instance: Optional["RiskAssessmentService"] = None

    def __init__(self, model_path: Optional[Path] = None):
        target_path = model_path or MODEL_PATH
        if not target_path.exists():
            raise RuntimeError(f"Trained model pipeline file not found at: {target_path}")

        # Initialize SHAP explainer (loads pipeline once)
        self.explainer = ReturnShieldExplainer(str(target_path))

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "RiskAssessmentService":
        """Singleton accessor ensuring model pipeline is loaded only once across requests."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    @classmethod
    def initialize(cls, model_path: Optional[Path] = None) -> None:
        """Explicit startup initialization method."""
        cls.get_instance(model_path=model_path)

    def assess(self, request: ReturnRiskRequest) -> ReturnRiskResponse:
        """
        Assesses a return request payload by:
        1. Transforming schema inputs to feature DataFrame (computing derived return_rate)
        2. Computing SHAP feature explanations
        3. Evaluating merchant decision rules
        4. Returning structured ReturnRiskResponse
        """
        # Calculate derived return_rate feature
        return_rate = (
            float(request.previous_returns) / float(request.total_orders)
            if request.total_orders > 0
            else 0.0
        )

        # Construct single-row DataFrame matching model expected columns
        raw_row = {
            "account_age_days": request.account_age_days,
            "total_orders": request.total_orders,
            "previous_returns": request.previous_returns,
            "previous_refunds": request.previous_refunds,
            "previous_chargebacks": request.previous_chargebacks,
            "avg_order_value": request.avg_order_value,
            "current_order_value": request.current_order_value,
            "days_to_return": request.days_to_return,
            "return_rate": return_rate,
            "refund_amount": request.refund_amount,
            "return_reason": request.return_reason,
            "product_category": request.product_category,
        }

        # Generate SHAP explanation statements
        explanation = self.explainer.explain_request(
            raw_row,
            low_threshold=LOW_RISK_THRESHOLD,
            high_threshold=HIGH_RISK_THRESHOLD
        )

        risk_score = explanation["risk_score"]
        top_risk_factors = explanation["top_risk_factors"]
        top_protective_factors = explanation["top_protective_factors"]

        # Run decision engine
        decision = evaluate_return_decision(
            risk_score=risk_score,
            top_risk_factors=top_risk_factors,
            top_protective_factors=top_protective_factors
        )

        return ReturnRiskResponse(
            risk_score=decision["risk_score"],
            risk_level=decision["risk_level"],
            recommended_action=decision["recommended_action"],
            review_required=decision["review_required"],
            top_risk_factors=decision["top_risk_factors"],
            top_protective_factors=decision["top_protective_factors"],
            merchant_reason=decision["merchant_reason"],
            customer_message=decision["customer_message"]
        )
