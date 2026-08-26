"""
API Routes for AI ReturnShield FastAPI Backend
(Razorpay Track 02: AI Risk Manager)
"""

import logging
from fastapi import APIRouter, HTTPException, status

from src.config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from src.api.schemas import (
    ReturnRiskRequest,
    ReturnRiskResponse,
    HealthCheckResponse,
    ModelInfoResponse,
)
from src.services.risk_service import RiskAssessmentService

logger = logging.getLogger("returnshield.api")

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    tags=["System"]
)
def get_health_status() -> HealthCheckResponse:
    """
    Health check endpoint returning system status.
    """
    return HealthCheckResponse(status="healthy", service="AI ReturnShield")


@router.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Operating Thresholds",
    tags=["Metadata"]
)
def get_model_info() -> ModelInfoResponse:
    """
    Returns non-sensitive metadata regarding the active return-abuse risk model
    and operating risk thresholds. Internal filesystem paths are omitted.
    """
    return ModelInfoResponse(
        model_type="XGBoost Classifier Pipeline",
        supported_risk_levels=["LOW", "MEDIUM", "HIGH"],
        current_operating_thresholds={
            "LOW_RISK_THRESHOLD": LOW_RISK_THRESHOLD,
            "HIGH_RISK_THRESHOLD": HIGH_RISK_THRESHOLD,
        }
    )


@router.post(
    "/api/v1/assess",
    response_model=ReturnRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess Return Request Risk",
    tags=["Risk Manager"]
)
def assess_return_risk(request: ReturnRiskRequest) -> ReturnRiskResponse:
    """
    Assesses a customer return request payload using the AI ReturnShield pipeline.

    Processes feature attributes, calculates SHAP explanations, evaluates merchant
    decision engine rules, and returns non-accusatory operational risk recommendations.
    """
    try:
        service = RiskAssessmentService.get_instance()
        return service.assess(request)
    except Exception as exc:
        logger.error(f"Error during risk assessment execution: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal processing error occurred while evaluating the return risk assessment."
        )
