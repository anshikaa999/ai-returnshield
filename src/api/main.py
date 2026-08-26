"""
Main FastAPI Application for AI ReturnShield Backend
(Razorpay Track 02: AI Risk Manager)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import API_TITLE, API_DESCRIPTION, API_VERSION
from src.services.risk_service import RiskAssessmentService
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads the ML pipeline ONCE during startup.
    Ensures the model is pre-warmed and ready before accepting requests.
    """
    # Startup: Load model pipeline into singleton service
    RiskAssessmentService.initialize()
    yield
    # Shutdown logic (if any) can be placed here


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration suitable for local React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
