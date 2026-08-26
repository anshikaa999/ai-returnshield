"""
Synthetic Dataset Generator for AI ReturnShield (Razorpay Track 02: AI Risk Manager)

Generates 10,000 realistic return/refund requests to emulate merchant transaction
and return behavior, creating a binary classification target (`is_suspicious`).
"""

import os
import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SAMPLES = 10000

def generate_synthetic_data(n_samples=N_SAMPLES):
    """
    Generates synthetic return request data with realistic relationships
    and risk indicators for return/refund fraud detection.
    """

    # 1. Customer and Order Identifiers
    customer_ids = [f"CUST_{i+1:05d}" for i in range(n_samples)]
    order_ids = [f"ORD_{i+1:05d}" for i in range(n_samples)]

    # 2. Account Age (1 to 1000 days, right-skewed)
    # Lognormal distribution to simulate many newer/medium accounts and fewer very old accounts
    account_age_days = np.random.lognormal(mean=4.5, sigma=1.0, size=n_samples)
    account_age_days = np.clip(account_age_days, 1, 1200).astype(int)

    # 3. Total Orders (Correlated with account age)
    # Older accounts generally have placed more orders
    base_orders_per_100_days = np.random.poisson(lam=4, size=n_samples) + 1
    total_orders = (account_age_days / 100.0 * base_orders_per_100_days).astype(int)
    total_orders = np.clip(total_orders, 1, 300)

    # 4. Previous Returns (Bounded by total_orders)
    # Most users have low return rates (5-15%), some high returners (30-70%)
    return_propensities = np.random.beta(a=1.5, b=8.0, size=n_samples)
    previous_returns = np.random.binomial(n=total_orders, p=return_propensities)

    # 5. Return Rate (previous_returns / total_orders)
    return_rate = np.round(previous_returns / total_orders, 4)

    # 6. Previous Refunds (Bounded by previous_returns)
    # Most returns lead to refunds, but some may be rejected or exchanged
    refund_ratio = np.random.uniform(0.7, 1.0, size=n_samples)
    previous_refunds = (previous_returns * refund_ratio).astype(int)

    # 7. Previous Chargebacks (Sparse count, correlated with high returns / bad actors)
    # Most customers have 0 chargebacks; risky profiles have 1+
    chargeback_prob = np.where(return_rate > 0.3, 0.15, 0.01)
    previous_chargebacks = np.random.poisson(lam=chargeback_prob)
    # Cap chargebacks at previous_refunds for consistency
    previous_chargebacks = np.minimum(previous_chargebacks, previous_refunds)

    # 8. Order Values (avg_order_value vs current_order_value)
    # Normal customer spending: Log-normal distribution centered around $80-$150
    avg_order_value = np.random.lognormal(mean=4.5, sigma=0.6, size=n_samples)
    avg_order_value = np.round(np.clip(avg_order_value, 10.0, 2000.0), 2)

    # Current order value ratio relative to customer's average order value
    # Occasional high spikes for fraud/risk testing
    order_value_mult = np.random.lognormal(mean=0.0, sigma=0.4, size=n_samples)
    current_order_value = np.round(avg_order_value * order_value_mult, 2)
    current_order_value = np.clip(current_order_value, 5.0, 5000.0)

    # 9. Days to Return (0 to 30 days after delivery)
    # Instant returns (0-1 days) can indicate specific suspicious patterns
    days_to_return = np.random.exponential(scale=7.0, size=n_samples).astype(int)
    days_to_return = np.clip(days_to_return, 0, 30)

    # 10. Return Reasons
    return_reasons = [
        "size_fit_issue",
        "changed_mind",
        "defective_item",
        "wrong_item_sent",
        "item_not_as_described",
        "empty_box_claimed",
        "item_not_received"
    ]
    # Probabilities for reasons
    reason_probs = [0.30, 0.25, 0.20, 0.10, 0.08, 0.04, 0.03]
    return_reason = np.random.choice(return_reasons, size=n_samples, p=reason_probs)

    # 11. Product Category
    product_categories = [
        "apparel",
        "electronics",
        "home_kitchen",
        "beauty",
        "jewelry",
        "books"
    ]
    category_probs = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    product_category = np.random.choice(product_categories, size=n_samples, p=category_probs)

    # 12. Refund Amount
    # Usually full refund of current_order_value, occasionally partial refund
    partial_refund_flag = np.random.binomial(n=1, p=0.1, size=n_samples)
    refund_ratio_current = np.where(partial_refund_flag == 1, np.random.uniform(0.4, 0.9, size=n_samples), 1.0)
    refund_amount = np.round(current_order_value * refund_ratio_current, 2)

    # -------------------------------------------------------------------------
    # 13. Determine Risk Score & Target Label (is_suspicious)
    # -------------------------------------------------------------------------
    # Create latent risk logit based on domain risk factors:
    # - High return rate
    # - Previous chargebacks & high previous refunds
    # - Sudden order value anomaly (current_order_value / avg_order_value)
    # - Very new account (< 30 days)
    # - Fast return (0-1 days)
    # - High risk return reasons ("empty_box_claimed", "item_not_received")
    # - High risk categories ("electronics", "jewelry")

    order_value_ratio = current_order_value / avg_order_value

    logit = (
        -3.5  # Base intercept (calibrates overall positive class rate ~12-15%)
        + 3.2 * (return_rate - 0.2)  # Return rate impact
        + 1.8 * previous_chargebacks  # Chargeback history
        + 0.8 * np.log1p(previous_refunds)  # High past refund count
        + 1.2 * np.maximum(0, order_value_ratio - 2.0)  # Value spike anomaly
        + 1.5 * (account_age_days < 30).astype(float)  # Brand new account risk
        + 0.7 * (days_to_return <= 1).astype(float)  # Rapid return
        + 1.8 * np.isin(return_reason, ["empty_box_claimed", "item_not_received"]).astype(float)  # High risk claim
        + 0.6 * np.isin(product_category, ["electronics", "jewelry"]).astype(float)  # High resale category
    )

    # Add realistic random noise to avoid deterministic target labels
    noise = np.random.normal(loc=0.0, scale=0.8, size=n_samples)
    noisy_logit = logit + noise

    # Convert logit to probability via Sigmoid function
    prob_suspicious = 1.0 / (1.0 + np.exp(-noisy_logit))

    # Bernoulli sampling to get final binary target (0 = normal, 1 = suspicious)
    is_suspicious = np.random.binomial(n=1, p=prob_suspicious)

    # -------------------------------------------------------------------------
    # Create DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "order_id": order_ids,
        "account_age_days": account_age_days,
        "total_orders": total_orders,
        "previous_returns": previous_returns,
        "previous_refunds": previous_refunds,
        "previous_chargebacks": previous_chargebacks,
        "avg_order_value": avg_order_value,
        "current_order_value": current_order_value,
        "days_to_return": days_to_return,
        "return_reason": return_reason,
        "product_category": product_category,
        "return_rate": return_rate,
        "refund_amount": refund_amount,
        "is_suspicious": is_suspicious
    })

    return df

def validate_dataset(df):
    """
    Validates dataset against strict integrity constraints before saving.
    """
    # 1. Exact row count check
    assert len(df) == 10000, f"Validation Failed: Expected 10,000 rows, got {len(df)}"

    # 2. Null / Missing value check
    assert df.isnull().sum().sum() == 0, "Validation Failed: Found missing values in dataset"

    # 3. Non-negative monetary values check
    assert (df['avg_order_value'] < 0).sum() == 0, "Validation Failed: Found negative avg_order_value"
    assert (df['current_order_value'] < 0).sum() == 0, "Validation Failed: Found negative current_order_value"
    assert (df['refund_amount'] < 0).sum() == 0, "Validation Failed: Found negative refund_amount"

    # 4. Count consistency checks
    assert (df['previous_returns'] > df['total_orders']).sum() == 0, \
        "Validation Failed: previous_returns > total_orders found"
    assert (df['previous_refunds'] > df['previous_returns']).sum() == 0, \
        "Validation Failed: previous_refunds > previous_returns found"
    assert (df['previous_chargebacks'] < 0).sum() == 0, \
        "Validation Failed: negative previous_chargebacks found"

    # 5. Return rate consistency check (return_rate == previous_returns / total_orders)
    expected_return_rate = np.round(df['previous_returns'] / df['total_orders'], 4)
    assert np.allclose(df['return_rate'], expected_return_rate, atol=1e-4), \
        "Validation Failed: return_rate inconsistency detected"

    print("Data Validation Passed Successfully!")

def print_summary(df):
    """
    Prints a concise dataset summary after generation.
    """
    total_rows = len(df)
    suspicious_count = df['is_suspicious'].sum()
    suspicious_pct = (suspicious_count / total_rows) * 100

    print("=" * 60)
    print("AI RETURN SHIELD - SYNTHETIC DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Rows Generated    : {total_rows:,}")
    print(f"Suspicious Cases (1)    : {suspicious_count:,}")
    print(f"Normal Cases (0)        : {total_rows - suspicious_count:,}")
    print(f"Suspicious Percentage   : {suspicious_pct:.2f}%")
    print("-" * 60)
    print("\nKey Numerical Feature Statistics:")
    num_cols = [
        "account_age_days", "total_orders", "previous_returns",
        "previous_chargebacks", "avg_order_value", "current_order_value",
        "return_rate", "refund_amount"
    ]
    print(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
    print("=" * 60)

def main():
    # Target directory and file path
    output_dir = os.path.join("data", "raw")
    output_filepath = os.path.join(output_dir, "return_requests.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    df = generate_synthetic_data(N_SAMPLES)

    # Validate data
    validate_dataset(df)

    # Print summary statistics
    print_summary(df)

    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Dataset successfully saved to: {output_filepath}\n")

if __name__ == "__main__":
    main()
