# AI ReturnShield — E-Commerce Return Risk Manager

Developed for the **Razorpay Buildathon — Track 02: AI Risk Manager**

AI ReturnShield is an intelligent, privacy-preserving risk assessment platform designed to help e-commerce merchants identify potential return-abuse patterns before issuing refunds. By integrating machine learning classification with SHAP (SHapley Additive exPlanations), ReturnShield provides operational risk indicators and human-readable explanation statements to guide merchant review workflows without penalizing honest customers.

---

## 1. Problem Statement

E-commerce merchants lose billions annually to return abuse, wardrobing, empty-box claims, and serial refund exploits. Traditional rule-based systems rely on rigid thresholds that either block genuine buyers or fail to catch sophisticated return abuse patterns.

**AI ReturnShield** solves this by:
- Evaluating 11 operational signals (account age, order velocity, chargebacks, claim reasons, item value ratios).
- Scoring risk probabilities with an XGBoost ML pipeline.
- Translating complex mathematical SHAP attributions into clear business terms in INR (₹).
- Recommending automated approvals for low-risk requests while routing elevated-risk claims for physical inspection or manual verification.

---

## 2. End-to-End Architecture

```text
Customer / Merchant Return Payload
               │
               ▼
   [ React + Vite Frontend ] ── (HTTP POST /api/v1/assess)
               │
               ▼
     [ FastAPI Backend ]
               │
               ▼
[ Scikit-Learn / XGBoost Pipeline ] ── (Predict Probabilities)
               │
               ▼
  [ SHAP Tree Explainer ] ───────────── (Extract Feature Attributions)
               │
               ▼
    [ Decision Engine ] ────────────── (Apply Operating Thresholds)
               │
               ▼
Structured Merchant Decision & Customer View Message
```

---

## 3. Technology Stack

- **Machine Learning & Analytics**: Python 3.13, XGBoost, Scikit-Learn, SHAP, Pandas, NumPy, Joblib.
- **Backend API**: FastAPI, Uvicorn, Pydantic v2.
- **Frontend UI**: React 18, Vite, Vanilla CSS (Professional Fintech Aesthetic).
- **Testing & Quality Assurance**: Pytest, Starlette TestClient, HTTPX.

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and `npm`

### Step 1: Install Python Backend Dependencies

```bash
# Clone repository
git clone https://github.com/anshikaa999/ai-returnshield.git
cd ai-returnshield

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install Python requirements
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 5. Running the Application

### 1. Start the FastAPI Backend Server

Run the backend server from the project root directory:

```bash
uvicorn src.api.main:app --reload
```

The backend server will launch on `http://127.0.0.1:8000`.

- **Swagger Interactive API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Specifications**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Start the React Frontend Server

In a second terminal window, start the Vite dev server:

```bash
cd frontend
npm run dev
```

The merchant dashboard will open at [http://localhost:5173](http://localhost:5173).

---

## 6. API Endpoints Reference

| Method | Path | Summary | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health Check | Returns `{"status": "healthy", "service": "AI ReturnShield"}` |
| `GET` | `/api/v1/model-info` | Model Metadata | Returns safe model metadata and active operating risk thresholds |
| `POST` | `/api/v1/assess` | Assess Return Risk | Accepts `ReturnRiskRequest` and returns structured `ReturnRiskResponse` |

---

## 7. Risk Thresholds & Decision Engine Matrix

ReturnShield evaluates risk scores against centralized validation thresholds (`LOW_RISK_THRESHOLD = 0.25`, `HIGH_RISK_THRESHOLD = 0.40`):

| Risk Level | Score Range | Recommended Action | Review Required | Merchant Operational Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **LOW** | `score < 0.25` | `APPROVE_RETURN` | `False` | Automated return approval. Return label generated immediately. |
| **MEDIUM** | `0.25 <= score < 0.40` | `ADDITIONAL_VERIFICATION` | `True` | Standard manual review recommended before refund processing. |
| **HIGH** | `score >= 0.40` | `ENHANCED_VERIFICATION` | `True` | Mandatory physical warehouse inspection required upon receipt. |

---

## 8. Running the Test Suite

Run the unit and integration test suite using `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

The test suite covers health checks, model metadata, decision boundary logic, valid LOW/MEDIUM/HIGH requests, and negative/relational validation rules (15 tests total).

---

## 9. Example Workflow

1. Merchant opens the dashboard at `http://localhost:5173`.
2. Enter customer transaction details or click a sample preset (**Low Risk Sample**, **Medium Risk Sample**, or **High Risk Sample**).
3. Click **Assess Return**.
4. The system sends a POST payload to `http://localhost:8000/api/v1/assess`.
5. The backend pipeline calculates the risk probability, runs SHAP feature attribution, and formats human-readable risk/protective statements.
6. The merchant dashboard renders:
   - Risk score percentage and colored risk badge.
   - Recommended merchant action & manual review indicator.
   - Risk signals & protective factors.
   - Merchant operational reason & customer-safe messaging.

---

## 10. Responsible AI & Governance Statement

AI ReturnShield is designed strictly as a **decision-support tool** for merchant operations.
- Model explanations represent attributed statistical risk indicators based on historical patterns, not legal proof of fraud.
- System outputs strictly use non-accusatory operational terminology (*"elevated return-abuse risk"*, *"additional verification recommended"*).
- No customer is ever labeled or communicated to as "fraudulent".