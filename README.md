# Credit Risk Probability Model for Bati Bank

An end-to-end credit scoring system using alternative eCommerce transaction data
to enable a buy-now-pay-later service.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires banks to hold capital reserves proportional
to their credit risk exposure. To calculate that risk accurately, Basel II demands
that models be **transparent, validated, and auditable**. Specifically:

- **Interpretability**: Regulators and internal risk teams must be able to
  understand *why* a customer received a particular risk score. A black-box model
  that says "deny this loan" without explanation is not acceptable — it exposes
  the bank to legal and regulatory risk.
- **Documentation**: Every modeling choice — feature selection, proxy variable
  design, validation methodology — must be documented so auditors can reconstruct
  and challenge the process.
- **Monitoring**: Models must be monitored for drift over time. If the eCommerce
  platform's customer behavior changes, the model's predictions may degrade, and
  Basel II requires processes to detect and respond to this.

In practice, this means we will prefer models like **Logistic Regression with
Weight of Evidence (WoE)** encoding where possible, and when we use complex models
like Gradient Boosting, we will add explainability tools (e.g., SHAP values) to
satisfy interpretability requirements.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The raw dataset from the eCommerce platform contains transaction records and a
fraud flag — but **no field that says whether a customer ever failed to repay a
loan**. This is common when building credit models for new lending products where
no historical loan performance data exists yet.

A **proxy variable** is a measurable behavior that we believe is *correlated* with
the true outcome we care about (default). In our case, we use **RFM analysis**:

- **Recency**: How recently did the customer transact? (Disengaged customers
  may be harder to collect from)
- **Frequency**: How often do they transact? (Low frequency may signal
  disengagement or financial stress)
- **Monetary**: How much do they spend? (Very low spenders may have limited
  financial capacity)

Customers who score poorly on all three dimensions — low frequency, low monetary
value, and haven't transacted recently — are labeled as **high risk (1)**.
All others are labeled **low risk (0)**.

**Business risks this introduces:**

| Risk | Description |
|------|-------------|
| **Label noise** | RFM disengagement ≠ guaranteed default. Some disengaged customers are simply inactive, not insolvent. |
| **Self-fulfilling exclusion** | If the model denies loans to customers based on a flawed proxy, we never observe whether they would have repaid. |
| **Regulatory challenge** | Regulators may question whether the proxy is a fair and valid measure of creditworthiness. |
| **Bias** | If certain demographic groups naturally have lower transaction frequency (e.g., seasonal workers), the proxy may introduce discriminatory outcomes. |

This proxy must be explicitly disclosed in model documentation and treated as
an assumption to be validated as real loan performance data accumulates.

---

### 3. What are the key trade-offs between a simple, interpretable model (Logistic Regression with WoE) and a high-performance model (Gradient Boosting) in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|-----------|--------------------------|--------------------------------------|
| **Interpretability** | High — coefficients directly show feature impact | Low — requires SHAP or LIME to explain |
| **Regulatory acceptance** | Well-established in credit scoring, auditor-friendly | Harder to justify without explainability add-ons |
| **Predictive performance** | Moderate — may miss complex non-linear patterns | High — captures non-linear interactions automatically |
| **Feature engineering** | Requires careful WoE binning and IV analysis | Can handle raw features more flexibly |
| **Overfitting risk** | Low — simpler model | Higher — requires careful tuning and cross-validation |
| **Deployment complexity** | Low — lightweight, fast scoring | Higher — larger model files, more compute |
| **Monitoring** | Easy — coefficient stability is simple to track | Complex — feature importance can shift subtly |

**Our approach:** We will train both types, track all experiments with MLflow,
and select the best model balancing AUC-ROC performance against interpretability
requirements. For a regulated bank, a slightly lower-performing interpretable
model may be preferable to a black-box model with marginally better metrics.

---

## Project Structure