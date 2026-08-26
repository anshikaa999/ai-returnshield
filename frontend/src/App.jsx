import React, { useState } from 'react';

const SAMPLES = {
  LOW: {
    account_age_days: 180,
    total_orders: 15,
    previous_returns: 0,
    previous_refunds: 0,
    previous_chargebacks: 0,
    avg_order_value: 2500,
    current_order_value: 2000,
    days_to_return: 12,
    return_reason: 'size_fit_issue',
    product_category: 'apparel',
    refund_amount: 2000,
  },
  MEDIUM: {
    account_age_days: 35,
    total_orders: 4,
    previous_returns: 1,
    previous_refunds: 1,
    previous_chargebacks: 0,
    avg_order_value: 1200,
    current_order_value: 2800,
    days_to_return: 2,
    return_reason: 'defective_item',
    product_category: 'electronics',
    refund_amount: 2800,
  },
  HIGH: {
    account_age_days: 3,
    total_orders: 2,
    previous_returns: 2,
    previous_refunds: 2,
    previous_chargebacks: 1,
    avg_order_value: 500,
    current_order_value: 6000,
    days_to_return: 0,
    return_reason: 'empty_box_claimed',
    product_category: 'electronics',
    refund_amount: 6000,
  },
};

export default function App() {
  const [formData, setFormData] = useState(SAMPLES.LOW);
  const [activePreset, setActivePreset] = useState('LOW');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setActivePreset(null);
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? '' : Number(value)) : value,
    }));
  };

  const loadPreset = (presetKey) => {
    setFormData(SAMPLES[presetKey]);
    setActivePreset(presetKey);
    setError(null);
  };

  const validateForm = () => {
    if (formData.account_age_days < 0) return 'Account age cannot be negative.';
    if (formData.total_orders < 0) return 'Total orders cannot be negative.';
    if (formData.previous_returns < 0) return 'Previous returns cannot be negative.';
    if (formData.previous_refunds < 0) return 'Previous refunds cannot be negative.';
    if (formData.previous_chargebacks < 0) return 'Previous chargebacks cannot be negative.';
    if (formData.days_to_return < 0) return 'Days to return cannot be negative.';
    if (formData.avg_order_value <= 0) return 'Average order value must be greater than 0.';
    if (formData.current_order_value <= 0) return 'Current order value must be greater than 0.';
    if (formData.refund_amount <= 0) return 'Refund amount must be greater than 0.';

    if (formData.previous_returns > formData.total_orders) {
      return `Previous returns (${formData.previous_returns}) cannot exceed total orders (${formData.total_orders}).`;
    }
    if (formData.previous_refunds > formData.previous_returns) {
      return `Previous refunds (${formData.previous_refunds}) cannot exceed previous returns (${formData.previous_returns}).`;
    }
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/assess', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        if (response.status === 422) {
          const errData = await response.json();
          const detailMsg = Array.isArray(errData.detail)
            ? errData.detail.map((d) => d.msg).join('; ')
            : 'Validation error';
          throw new Error(`Invalid Request (422): ${detailMsg}`);
        } else {
          throw new Error(`Server Error (${response.status}): Unable to process assessment.`);
        }
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Backend Unavailable: Ensure FastAPI backend is running on http://127.0.0.1:8000.');
      } else {
        setError(err.message || 'An unexpected error occurred during risk assessment.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="header-title">ReturnShield</h1>
        <p className="header-subtitle">Return risk assessment for merchants</p>
      </header>

      {/* Preset Toolbar */}
      <div className="preset-bar">
        <span className="preset-label">Demo Presets:</span>
        <button
          type="button"
          className={`btn-preset ${activePreset === 'LOW' ? 'active' : ''}`}
          onClick={() => loadPreset('LOW')}
        >
          Low Risk Sample
        </button>
        <button
          type="button"
          className={`btn-preset ${activePreset === 'MEDIUM' ? 'active' : ''}`}
          onClick={() => loadPreset('MEDIUM')}
        >
          Medium Risk Sample
        </button>
        <button
          type="button"
          className={`btn-preset ${activePreset === 'HIGH' ? 'active' : ''}`}
          onClick={() => loadPreset('HIGH')}
        >
          High Risk Sample
        </button>
      </div>

      {/* Main Grid */}
      <main className="main-layout">
        {/* Left Column: Form */}
        <section className="card">
          <h2 className="card-title">Return details</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="account_age_days">Account age (days)</label>
                <input
                  type="number"
                  id="account_age_days"
                  name="account_age_days"
                  value={formData.account_age_days}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="total_orders">Total orders</label>
                <input
                  type="number"
                  id="total_orders"
                  name="total_orders"
                  value={formData.total_orders}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="previous_returns">Previous returns</label>
                <input
                  type="number"
                  id="previous_returns"
                  name="previous_returns"
                  value={formData.previous_returns}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="previous_refunds">Previous refunds</label>
                <input
                  type="number"
                  id="previous_refunds"
                  name="previous_refunds"
                  value={formData.previous_refunds}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="previous_chargebacks">Previous chargebacks</label>
                <input
                  type="number"
                  id="previous_chargebacks"
                  name="previous_chargebacks"
                  value={formData.previous_chargebacks}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="days_to_return">Days to return</label>
                <input
                  type="number"
                  id="days_to_return"
                  name="days_to_return"
                  value={formData.days_to_return}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="avg_order_value">Average order value (₹)</label>
                <input
                  type="number"
                  id="avg_order_value"
                  name="avg_order_value"
                  value={formData.avg_order_value}
                  onChange={handleChange}
                  step="0.01"
                  min="0.01"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="current_order_value">Current order value (₹)</label>
                <input
                  type="number"
                  id="current_order_value"
                  name="current_order_value"
                  value={formData.current_order_value}
                  onChange={handleChange}
                  step="0.01"
                  min="0.01"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="refund_amount">Refund amount (₹)</label>
                <input
                  type="number"
                  id="refund_amount"
                  name="refund_amount"
                  value={formData.refund_amount}
                  onChange={handleChange}
                  step="0.01"
                  min="0.01"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="return_reason">Return reason</label>
                <select
                  id="return_reason"
                  name="return_reason"
                  value={formData.return_reason}
                  onChange={handleChange}
                  required
                >
                  <option value="size_fit_issue">Size / Fit Issue</option>
                  <option value="defective_item">Defective Item</option>
                  <option value="wrong_item">Wrong Item Sent</option>
                  <option value="changed_mind">Changed Mind</option>
                  <option value="empty_box_claimed">Empty Box Claimed</option>
                  <option value="item_not_received">Item Not Received Claim</option>
                </select>
              </div>

              <div className="form-group full-width">
                <label htmlFor="product_category">Product category</label>
                <select
                  id="product_category"
                  name="product_category"
                  value={formData.product_category}
                  onChange={handleChange}
                  required
                >
                  <option value="apparel">Apparel</option>
                  <option value="footwear">Footwear</option>
                  <option value="electronics">Electronics</option>
                  <option value="jewelry">Jewelry</option>
                  <option value="home">Home & Living</option>
                  <option value="beauty">Beauty & Care</option>
                </select>
              </div>
            </div>

            <button type="submit" className="btn-submit" disabled={loading}>
              {loading && <span className="spinner"></span>}
              {loading ? 'Evaluating risk...' : 'Assess return'}
            </button>
          </form>
        </section>

        {/* Right Column: Assessment Output */}
        <section className="card" aria-live="polite">
          <h2 className="card-title">Risk assessment</h2>

          {error && (
            <div className="error-banner">
              <div className="error-title">Assessment Error</div>
              <div>{error}</div>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="empty-state">
              <div className="empty-state-icon">!</div>
              <div className="empty-state-title">Awaiting assessment</div>
              <div className="empty-state-desc">Enter return details on the left to evaluate risk.</div>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="spinner" style={{ borderColor: '#A8A29E', borderTopColor: '#C85A32', width: 24, height: 24 }}></div>
              <div className="empty-state-title" style={{ marginTop: 12 }}>Processing request...</div>
              <div className="empty-state-desc">Calculating risk score and generating SHAP signals.</div>
            </div>
          )}

          {result && (
            <div>
              {/* Header Badge */}
              <div className={`result-header risk-${result.risk_level}`}>
                <div>
                  <div style={{ fontSize: 12, opacity: 0.85, fontWeight: 600 }}>RISK LEVEL</div>
                  <div className="risk-level-badge">{result.risk_level} RISK</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 12, opacity: 0.85, fontWeight: 600 }}>RISK SCORE</div>
                  <div className="risk-score-display">{(result.risk_score * 100).toFixed(1)}%</div>
                </div>
              </div>

              {/* Decision Grid */}
              <div className="summary-grid">
                <div className="summary-item">
                  <div className="summary-label">Recommended Action</div>
                  <div className="summary-value">
                    {result.recommended_action === 'APPROVE_RETURN' && 'Approve return'}
                    {result.recommended_action === 'ADDITIONAL_VERIFICATION' && 'Additional verification'}
                    {result.recommended_action === 'ENHANCED_VERIFICATION' && 'Enhanced verification'}
                  </div>
                </div>

                <div className="summary-item">
                  <div className="summary-label">Review Required</div>
                  <div className="summary-value">
                    {result.review_required ? 'Yes (Manual Review)' : 'No (Automated Approval)'}
                  </div>
                </div>
              </div>

              {/* Risk Signals */}
              <div className="result-section">
                <div className="section-title">Risk signals</div>
                <ul className="signals-list">
                  {result.top_risk_factors.map((factor, idx) => (
                    <li key={idx} className="signal-item risk">
                      <span className="signal-bullet">•</span>
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Protective Signals */}
              <div className="result-section">
                <div className="section-title">Protective signals</div>
                <ul className="signals-list">
                  {result.top_protective_factors.map((factor, idx) => (
                    <li key={idx} className="signal-item protective">
                      <span className="signal-bullet">•</span>
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Merchant Recommendation */}
              <div className="result-section">
                <div className="section-title">Merchant recommendation</div>
                <div className="merchant-reason-box">{result.merchant_reason}</div>
              </div>

              {/* Customer Message */}
              <div className="result-section" style={{ marginBottom: 0 }}>
                <div className="section-title">Customer view message</div>
                <div className="customer-message-box">"{result.customer_message}"</div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        AI ReturnShield • Decision support, not definitive fraud detection
      </footer>
    </div>
  );
}
