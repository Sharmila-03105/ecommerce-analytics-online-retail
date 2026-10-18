# Churn Model Card

Generated: 2026-10-18T11:52:53.789072+00:00

## Problem

Predict whether a customer will churn (no purchase in 90 days after cutoff).

## Walk-forward validation

| Fold | Train cutoff | Validate cutoff |
|---|---|---|
| 2010-06-30 -> 2010-12-31 | 2010-06-30 | 2010-12-31 |
| 2010-12-31 -> 2011-06-30 | 2010-12-31 | 2011-06-30 |

## Metrics per fold

| Fold | Model | ROC-AUC | PR-AUC | Prec@10% | Lift@10% |
|---|---|---|---|---|---|
| 2010-06-30->2010-12-31 | Logistic | 0.7917 | 0.867 | 0.9028 | 1.33 |
| 2010-06-30->2010-12-31 | XGBoost | 0.7697 | 0.8548 | 0.9213 | 1.36 |
| 2010-12-31->2011-06-30 | Logistic | 0.7995 | 0.8715 | 0.9462 | 1.43 |
| 2010-12-31->2011-06-30 | XGBoost | 0.7986 | 0.8661 | 0.9363 | 1.42 |

## Features

- recency_days
- tenure_days
- frequency
- monetary
- product_diversity
- avg_order_value
- purchase_cadence
- active_days

## Feature importance (SHAP mean abs)

| Feature | Mean abs SHAP |
|---|---|
| monetary | 0.3637 |
| active_days | 0.3367 |
| recency_days | 0.3118 |
| frequency | 0.2480 |
| purchase_cadence | 0.1650 |
| tenure_days | 0.1507 |
| product_diversity | 0.1483 |
| avg_order_value | 0.1245 |

## Limitations

- Dataset has no marketing/promo data; model can't account for campaigns.
- Q4 seasonality may inflate churn labels in Q1 cutoffs.
- Guest customers excluded from modeling.
- Model predicts 90-day churn on this retailer; may need re-tuning for other domains.
