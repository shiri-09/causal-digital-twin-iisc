<p align="center">
  <h1 align="center">🧠 MindBridge: Causal Digital Twin for MCI Prevention</h1>
  <p align="center">
    <em>A Counterfactual Therapy Simulator Using Missingness-Aware Causal Forests</em>
  </p>
  <p align="center">
    <a href="https://aichallenge.cbr-iisc.ac.in/"><img src="https://img.shields.io/badge/IISc_CBR-AI_Challenge_2026-blue?style=for-the-badge" alt="CBR Challenge"/></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python" alt="Python"/></a>
    <a href="#"><img src="https://img.shields.io/badge/Edge-Raspberry_Pi_4-red?style=for-the-badge&logo=raspberrypi" alt="RPi4"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/></a>
  </p>
</p>

---

## 🎯 The Problem

Current ML models for dementia predict **risk** — *"This patient has a 34% chance of developing MCI."*

But clinicians in rural India need **action** — *"What specific intervention should I implement for THIS patient, RIGHT NOW?"*

## 💡 Our Solution

A **causal digital twin** that simulates counterfactual interventions in **<10ms** on a **₹6,500 Raspberry Pi 4** — no cloud, no GPU, no internet required.

**Example Output:**
> 📊 Patient Risk: **34%** (2-year MCI probability)
>
> | Intervention | New Risk | Reduction | Confidence |
> |---|---|---|---|
> | 🩸 Lower HbA1c by 1% | 22% | **-12pp** | E-value: 3.1 |
> | 💊 Reduce BP to <130/80 | 26% | **-8pp** | E-value: 2.8 |
> | 🏃 Increase activity to 300 min/wk | 28% | **-6pp** | E-value: 2.4 |
> | 🫀 Reduce LDL by 20mg/dL | 30% | **-4pp** | E-value: 2.1 |
> | 🔗 **Combined (all four)** | **18%** | **-16pp** | — |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (Cloud/GPU)                   │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  SANSCOG/TLSA │───▶│   DML with    │───▶│  Missingness-    │   │
│  │  Cohort Data  │    │  5-fold CV    │    │  Aware Causal    │   │
│  │  (n≈10,000)   │    │  (Nuisance)   │    │  Forest (MACF)   │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                                          ┌─────────▼─────────┐  │
│                                          │  ONNX Export +     │  │
│                                          │  int8 Quantization │  │
│                                          └─────────┬─────────┘  │
└────────────────────────────────────────────────────┼────────────┘
                                                     │
┌────────────────────────────────────────────────────┼────────────┐
│               INFERENCE PHASE (Raspberry Pi 4)     │            │
│                                                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Clinician    │◀──│  Counterfact. │◀──│  ONNX Runtime    │   │
│  │  Dashboard    │    │  Generator   │    │  (int8 model)    │   │
│  │  (Touch UI)   │    │  (<10ms)     │    │  4GB RAM         │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                 │
│  Hardware: RPi4 + 7" touchscreen = ₹6,500 total                │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Key Innovation: MACF

The **Missingness-Aware Causal Forest** is our core algorithmic contribution:

- Standard causal forests **drop patients** with missing data
- MACF treats missingness as **informative** — at each tree node, for features with NaN values, it tries both `missing → left` and `missing → right` and picks the split maximizing treatment effect heterogeneity: `(τ̂_left - τ̂_right)²`
- Handles **41% OCT missingness** and **28% gait speed missingness** without imputation bias

## 📊 Datasets

| Dataset | Role | Size |
|---|---|---|
| **SANSCOG** | Primary training | n ≈ 6,102 (V1 baseline) |
| **TLSA** | External validation only | n ≈ 1,449 (V3) |
| **IHDP** | Causal benchmark (known τ) | n = 747 |
| **ACTG-175** | Clinical RCT benchmark | n = 2,139 |
| **News** | High-dimensional benchmark | n = 5,000 |

> **Note:** Real SANSCOG/TLSA data is accessed only through CBR's secure infrastructure under DUA. This repository uses synthetic data that mirrors the SANSCOG feature structure for development and demonstration.

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/shiri-09/causal-digital-twin-iisc.git
cd causal-digital-twin-iisc

# Install dependencies
pip install -r requirements.txt

# Run full demo pipeline
python demo.py

# Launch clinician dashboard
python -m src.dashboard.app
```

## 📁 Project Structure

```
├── configs/
│   └── default.yaml          # Pipeline hyperparameters and targets
├── docs/
│   └── architecture.md       # System architecture documentation
├── src/
│   ├── config.py             # Centralized YAML config loader
│   ├── data/                 # Data generators and loaders
│   │   ├── synthetic_mci.py  # SANSCOG-like synthetic data
│   │   ├── ihdp_loader.py    # Real IHDP benchmark (CEVAE)
│   │   ├── actg175_loader.py # ACTG-175 clinical trial (UCI)
│   │   ├── news_loader.py    # News semi-synthetic benchmark
│   │   ├── addi_harmonize.py # ADDI C-Surv harmonization schema
│   │   ├── preprocessing.py  # Feature engineering & standardization
│   │   └── missing_indicators.py  # Shared missingness utilities
│   ├── models/               # Core ML models
│   │   ├── macf.py           # Missingness-Aware Causal Forest
│   │   ├── honest_tree.py    # Honest splitting base
│   │   ├── dml_nuisance.py   # Double ML estimation
│   │   └── risk_predictor.py # LightGBM baseline
│   ├── pipeline/             # Training & evaluation
│   │   ├── train.py          # Full training orchestrator
│   │   ├── evaluate.py       # PEHE, AUROC, coverage
│   │   ├── e_value.py        # E-value sensitivity
│   │   └── negative_controls.py  # Placebo/shuffle validation
│   ├── deployment/           # Edge deployment
│   │   ├── model_export.py   # ONNX + JSON model export
│   │   └── quantize.py       # int8 quantization for RPi4
│   ├── dashboard/            # Clinician-facing UI
│   │   ├── app.py            # Flask backend
│   │   ├── inference.py      # Real-time inference engine
│   │   ├── templates/        # HTML (touch-optimized)
│   │   └── static/           # CSS + JS
│   └── visualization/        # Plotting utilities
├── tests/
│   ├── test_data.py          # Data generation & IHDP tests
│   └── test_macf.py          # MACF algorithm tests
└── demo.py                   # One-command pipeline runner
```

## 📈 Validation Metrics

| Metric | Target | What It Validates |
|---|---|---|
| PEHE | < 0.08 | Treatment effect accuracy (IHDP benchmark) |
| CI Coverage | > 90% | Confidence interval reliability |
| AUROC | > 0.78 | Risk prediction accuracy |
| E-value | > 2.0 | Robustness to unmeasured confounding |
| Inference | < 10ms (p95) | Edge deployment feasibility |

## 🏥 Four Modifiable Interventions

1. **HbA1c Reduction** — Lower glycated hemoglobin by 1%
2. **Blood Pressure Management** — Target < 130/80 mmHg
3. **Physical Activity** — Increase to 150–300 min/week
4. **LDL Cholesterol Reduction** — Lower by ≥ 20 mg/dL

Each monitored with a ₹800 BP cuff + smartphone camera (gait pose estimation) — **zero additional infrastructure cost**.

## 👥 Team

**Team PESU-RF** — PES University, Bengaluru

Built for the [IISc CBR AI Challenge for Healthy Brain Aging 2026](https://aichallenge.cbr-iisc.ac.in/)

## 📚 References

- Wager, S. & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*.
- Chernozhukov, V. et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*.
- VanderWeele, T.J. & Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*.

## 📄 License

MIT License — see [LICENSE](LICENSE)
