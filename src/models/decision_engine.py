"""
Decision Engine Module for AI ReturnShield (Razorpay Track 02: AI Risk Manager)

Converts model-predicted risk scores into structured merchant decision outputs,
action recommendations, and non-accusatory operational reasons.
"""

from typing import Dict, List, Any, Optional

from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD

# Operating Thresholds imported from central config
LOW_THRESHOLD = LOW_RISK_THRESHOLD
HIGH_THRESHOLD = HIGH_RISK_THRESHOLD

def evaluate_return_decision(
    risk_score: float,
    top_risk_factors: Optional[List[str]] = None,
    top_protective_factors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates a return request risk score and returns a structured merchant decision.

    Parameters:
        risk_score (float): Model probability of return abuse risk (0.0 to 1.0).
        top_risk_factors (list, optional): List of human-readable risk factors.
        top_protective_factors (list, optional): List of human-readable protective factors.

    Returns:
        dict: Structured decision output.
    """
    # 1. Strict Validation
    if not isinstance(risk_score, (int, float)):
        raise ValueError(f"Invalid risk score type: {type(risk_score)}. Risk score must be a numeric value between 0.0 and 1.0.")

    if risk_score < 0.0 or risk_score > 1.0:
        raise ValueError(f"Invalid risk score: {risk_score}. Risk score must be between 0.0 and 1.0 inclusive.")

    # Safe input normalization
    risk_factors = list(top_risk_factors) if top_risk_factors is not None else []
    protective_factors = list(top_protective_factors) if top_protective_factors is not None else []

    # 2. Threshold Decision Logic
    if risk_score < LOW_THRESHOLD:
        risk_level = "LOW"
        recommended_action = "APPROVE_RETURN"
        review_required = False
        customer_message = "Your return request has been approved. Return instructions have been generated."
        merchant_reason = "Low return-abuse risk. Automated approval recommended."

    elif LOW_THRESHOLD <= risk_score < HIGH_THRESHOLD:
        risk_level = "MEDIUM"
        recommended_action = "ADDITIONAL_VERIFICATION"
        review_required = True
        customer_message = "Your return request has been received and is under standard review. We will update you shortly."
        merchant_reason = "Elevated return-abuse risk detected. Additional verification and manual review recommended before refund approval."

    elif risk_score >= HIGH_THRESHOLD:
        risk_level = "HIGH"
        recommended_action = "ENHANCED_VERIFICATION"
        review_required = True
        customer_message = "Your return request requires physical inspection upon arrival. Return details are being processed."
        merchant_reason = "High return-abuse risk pattern detected. Enhanced verification and mandatory manual inspection recommended before issuing refund payout."

    else:
        raise ValueError(f"Unknown decision state encountered for risk score: {risk_score}")

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "review_required": review_required,
        "top_risk_factors": risk_factors,
        "top_protective_factors": protective_factors,
        "customer_message": customer_message,
        "merchant_reason": merchant_reason,
        "disclaimer": "Explanations and decisions represent model-attributed risk indicators based on historical patterns, not definitive proof of customer fraud."
    }
