# Architecture — Causal Digital Twin for MCI Prevention

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  synthetic_mci.py ──── ihdp_loader.py ──── preprocessing│
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Layer                            │
│  dml_nuisance.py ── macf.py ── risk_predictor.py        │
│  (Cross-fitted)    (Novel)    (LightGBM)                │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                 Pipeline Layer                           │
│  train.py ──── evaluate.py ──── e_value.py              │
│  (Orchestrate)  (PEHE/AUROC)   (Sensitivity)            │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              Deployment Layer                            │
│  onnx_export.py ──── FastTreeInference                  │
│  (JSON + int8 quant)  (<1ms per patient)                │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              Presentation Layer                          │
│  app.py (Flask) ── inference.py ── dashboard (HTML/JS)  │
│  (REST API)        (Real-time)     (Touch UI)           │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Missingness-Aware Splits (MACF)
Standard causal forests drop patients with missing data or require
imputation. Our MACF tries both "missing → left" and "missing → right"
at each split, selecting the direction that maximizes treatment effect
heterogeneity: (τ̂_left - τ̂_right)².

### 2. Honest Splitting
Each tree uses 50% of its subsample for structure (split decisions)
and 50% for estimation (leaf τ̂ values). This prevents overfitting.

### 3. Double Machine Learning
Before MACF training, we residualize outcomes and treatments using
5-fold cross-fitted LightGBM models. This removes confounding bias.

### 4. Edge Deployment via JSON
Instead of ONNX tree-ensemble operators (which don't support custom
split logic), we serialize trees to compact JSON and use a fast
Python traversal engine. This achieves <1ms inference.

## Treatment Effects

| Treatment | Display Name | Target τ |
|-----------|-------------|----------|
| hba1c_reduced | Lower HbA1c by 1% | -0.08 to -0.15 |
| bp_managed | Manage BP to <130/80 | -0.06 to -0.10 |
| activity_increased | 300 min/wk activity | -0.05 to -0.08 |
| ldl_reduced | Reduce LDL by 20 mg/dL | -0.03 to -0.06 |
