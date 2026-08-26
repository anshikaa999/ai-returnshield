"""
AI ReturnShield Decision Engine & Explainability Demonstration (Razorpay Track 02)
Demonstrates end-to-end evaluation for Low, Medium, and High risk return requests using INR terminology.
"""

import sys
import json

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.decision_engine import evaluate_return_decision

def run_demo():
    print("=" * 70)
    print("AI RETURN SHIELD DECISION ENGINE DEMONSTRATION (INR TERMINOLOGY)")
    print("=" * 70 + "\n")

    cases = [
        {
            "title": "Low-Risk Standard Return",
            "score": 0.08,
            "risk_factors": ["Immediate return requested (1 day after delivery)"],
            "protective_factors": [
                "Healthy return rate history (0.0%)",
                "Established customer tenure (180 days old)",
                "Order value (INR 1,499.00) consistent with customer average (INR 1,350.00)"
            ]
        },
        {
            "title": "Medium-Risk Elevated Claim",
            "score": 0.32,
            "risk_factors": [
                "Prior refund history (4 refunds)",
                "Immediate return requested (0 days after delivery)",
                "High resale risk category ('Jewelry')"
            ],
            "protective_factors": [
                "Healthy return rate history (18.5%)",
                "Zero previous chargeback disputes on record"
            ]
        },
        {
            "title": "High-Risk Return Abuse Pattern",
            "score": 0.68,
            "risk_factors": [
                "Previous chargeback disputes logged (2)",
                "High historical return rate (62.5%)",
                "Current order value (INR 18,500.00) is 3.4x customer average (INR 5,400.00)",
                "New customer account (14 days old)"
            ],
            "protective_factors": [
                "Standard return timeline (12 days after delivery)"
            ]
        }
    ]

    for idx, c in enumerate(cases, 1):
        decision = evaluate_return_decision(
            risk_score=c["score"],
            top_risk_factors=c["risk_factors"],
            top_protective_factors=c["protective_factors"]
        )

        print(f"CASE {idx}: {c['title']}")
        print("-" * 70)
        print(f"  Risk Score        : {decision['risk_score']:.4f}")
        print(f"  Risk Level        : [{decision['risk_level']}]")
        print(f"  Recommended Action: {decision['recommended_action']}")
        print(f"  Review Required   : {decision['review_required']}")
        print(f"  Merchant Reason   : {decision['merchant_reason']}")
        print(f"  Customer Message  : {decision['customer_message']}")
        print("-" * 70)
        print("  Top Risk Factors:")
        for rf in decision["top_risk_factors"]:
            print(f"    * {rf}")
        print("  Top Protective Factors:")
        for pf in decision["top_protective_factors"]:
            print(f"    * {pf}")
        print(f"  [Disclaimer]: {decision['disclaimer']}")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    run_demo()
