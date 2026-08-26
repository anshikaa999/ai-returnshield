"""
Central Configuration for AI ReturnShield Backend
(Razorpay Track 02: AI Risk Manager)
"""

import os
from pathlib import Path

# Operating Risk Thresholds (derived from Validation Set Analysis)
LOW_RISK_THRESHOLD: float = 0.25
HIGH_RISK_THRESHOLD: float = 0.40

# Project Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODEL_PATH: Path = BASE_DIR / "models" / "returnshield_pipeline.joblib"

# API Metadata
API_TITLE: str = "AI ReturnShield API"
API_DESCRIPTION: str = (
    "Production-grade FastAPI backend for AI ReturnShield return-abuse risk assessment. "
    "Provides ML risk scoring, SHAP-based merchant explanations, and actionable operational recommendations."
)
API_VERSION: str = "1.0.0"
