"""
Unit and Integration Tests for AI ReturnShield FastAPI Backend
(tests/test_api.py)
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    """Test client fixture with lifespan model initialization."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns status 200 and healthy payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AI ReturnShield"


def test_model_info_endpoint(client):
    """Test GET /api/v1/model-info returns 200 and safe metadata."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert data["supported_risk_levels"] == ["LOW", "MEDIUM", "HIGH"]
    assert data["current_operating_thresholds"]["LOW_RISK_THRESHOLD"] == 0.25
    assert data["current_operating_thresholds"]["HIGH_RISK_THRESHOLD"] == 0.40
    # Guarantee internal filesystem paths are not exposed
    assert "c:" not in str(data).lower()
    assert "users" not in str(data).lower()


def test_assess_valid_low_risk_request(client):
    """Test POST /api/v1/assess with valid LOW risk parameters."""
    payload = {
        "account_age_days": 180,
        "total_orders": 12,
        "previous_returns": 0,
        "previous_refunds": 0,
        "previous_chargebacks": 0,
        "avg_order_value": 2000.0,
        "current_order_value": 1800.0,
        "days_to_return": 10,
        "return_reason": "size_fit_issue",
        "product_category": "apparel",
        "refund_amount": 1800.0,
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] == "LOW"
    assert data["recommended_action"] == "APPROVE_RETURN"
    assert data["review_required"] is False
    assert isinstance(data["top_risk_factors"], list)
    assert isinstance(data["top_protective_factors"], list)
    assert "merchant_reason" in data
    assert "customer_message" in data
    assert "fraud" not in data["merchant_reason"].lower()


def test_assess_valid_medium_risk_request(client):
    """Test POST /api/v1/assess with MEDIUM risk profile parameters."""
    payload = {
        "account_age_days": 30,
        "total_orders": 3,
        "previous_returns": 1,
        "previous_refunds": 1,
        "previous_chargebacks": 0,
        "avg_order_value": 1000.0,
        "current_order_value": 2200.0,
        "days_to_return": 2,
        "return_reason": "defective_item",
        "product_category": "electronics",
        "refund_amount": 2200.0,
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["top_risk_factors"], list)
    assert isinstance(data["top_protective_factors"], list)


def test_assess_valid_high_risk_request(client):
    """Test POST /api/v1/assess with HIGH risk profile parameters."""
    payload = {
        "account_age_days": 3,
        "total_orders": 2,
        "previous_returns": 2,
        "previous_refunds": 2,
        "previous_chargebacks": 1,
        "avg_order_value": 500.0,
        "current_order_value": 4500.0,
        "days_to_return": 0,
        "return_reason": "empty_box_claimed",
        "product_category": "electronics",
        "refund_amount": 4500.0,
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert data["risk_level"] == "HIGH"
    assert data["recommended_action"] == "ENHANCED_VERIFICATION"
    assert data["review_required"] is True
    assert "high return-abuse risk" in data["merchant_reason"].lower()


def test_assess_invalid_negative_value(client):
    """Test POST /api/v1/assess with negative account_age_days returns 422."""
    payload = {
        "account_age_days": -5,
        "total_orders": 10,
        "previous_returns": 0,
        "previous_refunds": 0,
        "previous_chargebacks": 0,
        "avg_order_value": 1000.0,
        "current_order_value": 1000.0,
        "days_to_return": 5,
        "return_reason": "size_fit_issue",
        "product_category": "apparel",
        "refund_amount": 1000.0,
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 422


def test_assess_invalid_previous_returns_exceed_total(client):
    """Test POST /api/v1/assess when previous_returns > total_orders returns 422."""
    payload = {
        "account_age_days": 50,
        "total_orders": 2,
        "previous_returns": 5,
        "previous_refunds": 1,
        "previous_chargebacks": 0,
        "avg_order_value": 1000.0,
        "current_order_value": 1000.0,
        "days_to_return": 5,
        "return_reason": "size_fit_issue",
        "product_category": "apparel",
        "refund_amount": 1000.0,
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 422


def test_assess_invalid_previous_refunds_exceed_returns(client):
    """Test POST /api/v1/assess when previous_refunds > previous_returns returns 422."""
    payload = {
        "account_age_days": 50,
        "total_orders": 10,
        "previous_returns": 2,
        "previous_refunds": 4,
        "previous_chargebacks": 0,
        "avg_order_value": 1000.0,
        "current_order_value": 1000.0,
        "days_to_return": 5,
        "return_reason": "size_fit_issue",
        "product_category": "apparel",
        "refund_amount": 1000.0,
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 422
