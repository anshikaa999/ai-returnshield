"""
Unit tests for AI ReturnShield Decision Engine (src/models/decision_engine.py)
"""

import pytest
from src.models.decision_engine import evaluate_return_decision

def test_low_risk_request():
    """Test low risk score (< 0.25) decision evaluation."""
    score = 0.10
    risk_factors = ["Return reason: 'Size Fit Issue'"]
    protective_factors = ["Healthy return rate history (0.0%)", "Established customer tenure (120 days old)"]

    result = evaluate_return_decision(score, risk_factors, protective_factors)

    assert result["risk_score"] == 0.10
    assert result["risk_level"] == "LOW"
    assert result["recommended_action"] == "APPROVE_RETURN"
    assert result["review_required"] is False
    assert len(result["top_risk_factors"]) == 1
    assert len(result["top_protective_factors"]) == 2
    assert "fraud" not in result["merchant_reason"].lower()

def test_medium_risk_request():
    """Test medium risk score (0.25 <= score < 0.40) decision evaluation."""
    score = 0.30
    risk_factors = ["Prior refund history (6 refunds)", "Immediate return requested (0 days)"]
    protective_factors = ["Zero previous chargebacks on file"]

    result = evaluate_return_decision(score, risk_factors, protective_factors)

    assert result["risk_score"] == 0.30
    assert result["risk_level"] == "MEDIUM"
    assert result["recommended_action"] == "ADDITIONAL_VERIFICATION"
    assert result["review_required"] is True
    assert "elevated return-abuse risk" in result["merchant_reason"].lower()

def test_high_risk_request():
    """Test high risk score (score >= 0.40) decision evaluation."""
    score = 0.65
    risk_factors = ["Previous chargeback disputes logged (2)", "High historical return rate (67.0%)"]
    protective_factors = ["Order value (₹2,500.00) consistent with typical spending (₹2,200.00)"]

    result = evaluate_return_decision(score, risk_factors, protective_factors)

    assert result["risk_score"] == 0.65
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "ENHANCED_VERIFICATION"
    assert result["review_required"] is True
    assert "high return-abuse risk" in result["merchant_reason"].lower()

def test_exact_boundary_low_to_medium():
    """Test exact boundary score of 0.25 (should fall into MEDIUM risk)."""
    score = 0.25
    result = evaluate_return_decision(score)

    assert result["risk_score"] == 0.25
    assert result["risk_level"] == "MEDIUM"
    assert result["recommended_action"] == "ADDITIONAL_VERIFICATION"
    assert result["review_required"] is True

def test_exact_boundary_medium_to_high():
    """Test exact boundary score of 0.40 (should fall into HIGH risk)."""
    score = 0.40
    result = evaluate_return_decision(score)

    assert result["risk_score"] == 0.40
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "ENHANCED_VERIFICATION"
    assert result["review_required"] is True

def test_invalid_score_below_zero():
    """Test that score < 0 raises ValueError."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        evaluate_return_decision(-0.05)

def test_invalid_score_above_one():
    """Test that score > 1 raises ValueError."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        evaluate_return_decision(1.05)
