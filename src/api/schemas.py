"""
Pydantic Schemas for AI ReturnShield API
(Razorpay Track 02: AI Risk Manager)
"""

from typing import List, Dict
from pydantic import BaseModel, Field, model_validator

class ReturnRiskRequest(BaseModel):
    """
    Request payload schema for return request risk assessment.
    Enforces non-negative counts, positive monetary amounts, and relational validity.
    """
    account_age_days: int = Field(..., ge=0, description="Customer account age in days (>= 0)")
    total_orders: int = Field(..., ge=0, description="Total historical completed orders (>= 0)")
    previous_returns: int = Field(..., ge=0, description="Total previous returns requested (>= 0)")
    previous_refunds: int = Field(..., ge=0, description="Total previous refunds granted (>= 0)")
    previous_chargebacks: int = Field(..., ge=0, description="Total previous chargeback disputes (>= 0)")
    avg_order_value: float = Field(..., gt=0, description="Historical average order value in INR (> 0)")
    current_order_value: float = Field(..., gt=0, description="Current order value in INR (> 0)")
    days_to_return: int = Field(..., ge=0, description="Days elapsed between delivery and return request (>= 0)")
    return_reason: str = Field(..., min_length=1, description="Customer-stated return reason")
    product_category: str = Field(..., min_length=1, description="Category of the product being returned")
    refund_amount: float = Field(..., gt=0, description="Requested refund amount in INR (> 0)")

    @model_validator(mode="after")
    def validate_relational_constraints(self) -> "ReturnRiskRequest":
        """
        Validates cross-field logical constraints:
        - previous_returns cannot exceed total_orders
        - previous_refunds cannot exceed previous_returns
        """
        if self.previous_returns > self.total_orders:
            raise ValueError(
                f"previous_returns ({self.previous_returns}) cannot exceed total_orders ({self.total_orders})"
            )
        if self.previous_refunds > self.previous_returns:
            raise ValueError(
                f"previous_refunds ({self.previous_refunds}) cannot exceed previous_returns ({self.previous_returns})"
            )
        return self


class ReturnRiskResponse(BaseModel):
    """
    Response schema for return request risk assessment.
    Abstracts model predictions into non-accusatory operational risk insights.
    """
    risk_score: float = Field(..., description="Model predicted return-abuse risk score between 0.0 and 1.0")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    recommended_action: str = Field(..., description="Merchant action: APPROVE_RETURN, ADDITIONAL_VERIFICATION, or ENHANCED_VERIFICATION")
    review_required: bool = Field(..., description="Whether manual merchant verification is required")
    top_risk_factors: List[str] = Field(..., description="Human-readable risk factors elevating risk score")
    top_protective_factors: List[str] = Field(..., description="Human-readable protective factors lowering risk score")
    merchant_reason: str = Field(..., description="Operational reason summary for merchant dashboard")
    customer_message: str = Field(..., description="Polite, non-accusatory message for customer view")


class HealthCheckResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(default="healthy")
    service: str = Field(default="AI ReturnShield")


class ModelInfoResponse(BaseModel):
    """Model information endpoint response schema."""
    model_type: str
    supported_risk_levels: List[str]
    current_operating_thresholds: Dict[str, float]
