"""
Pipeline package — training orchestration, evaluation, and sensitivity analysis.

Exports:
    train_full_pipeline: Complete training workflow
    evaluate_cate: CATE evaluation metrics (PEHE, coverage)
    evaluate_risk_predictor: Risk predictor evaluation (AUROC)
    compute_e_values: E-value sensitivity analysis
    run_all_negative_controls: Negative control validation
"""

from src.pipeline.evaluate import evaluate_cate, evaluate_risk_predictor
from src.pipeline.e_value import compute_e_values

__all__ = [
    'evaluate_cate',
    'evaluate_risk_predictor',
    'compute_e_values',
]
