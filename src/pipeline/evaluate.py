"""
Evaluation Metrics

Computes all validation metrics from the proposal:
    - PEHE (Precision in Estimation of Heterogeneous Effects)
    - CI Coverage
    - AUROC for risk prediction
    - ATE error
    - Inference timing
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Optional
import time


def evaluate_cate(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate CATE estimation quality.
    
    Args:
        tau_hat: Predicted treatment effects
        tau_true: True treatment effects (known in synthetic/benchmark data)
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
    
    Returns:
        Dictionary of metrics
    """
    # PEHE: sqrt(mean((τ̂ - τ)²))
    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
    
    # ATE error: |mean(τ̂) - mean(τ)|
    ate_hat = float(tau_hat.mean())
    ate_true = float(tau_true.mean())
    ate_error = float(abs(ate_hat - ate_true))
    
    # Bias
    bias = float(np.mean(tau_hat - tau_true))
    
    # Correlation
    if tau_true.std() > 1e-10 and tau_hat.std() > 1e-10:
        correlation = float(np.corrcoef(tau_hat, tau_true)[0, 1])
    else:
        correlation = 0.0
    
    results = {
        'pehe': pehe,
        'ate_hat': ate_hat,
        'ate_true': ate_true,
        'ate_error': ate_error,
        'bias': bias,
        'correlation': correlation,
    }
    
    # Coverage: proportion of true τ within CI
    if ci_lower is not None and ci_upper is not None:
        coverage = float(np.mean((tau_true >= ci_lower) & (tau_true <= ci_upper)))
        ci_width = float(np.mean(ci_upper - ci_lower))
        results['coverage'] = coverage
        results['ci_width'] = ci_width
    
    return results


def evaluate_risk_predictor(
    predictor,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the MCI risk predictor.
    
    Returns AUROC, AUPRC, calibration metrics.
    """
    y_prob = predictor.predict_proba(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    
    auroc = float(roc_auc_score(Y_test, y_prob))
    auprc = float(average_precision_score(Y_test, y_prob))
    
    # Simple calibration: compare mean predicted vs actual
    mean_predicted = float(y_prob.mean())
    mean_actual = float(Y_test.mean())
    calibration_error = float(abs(mean_predicted - mean_actual))
    
    # Accuracy
    accuracy = float(np.mean(y_pred == Y_test))
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'mean_predicted': mean_predicted,
        'mean_actual': mean_actual,
        'calibration_error': calibration_error,
    }


def measure_inference_time(
    model,
    X_sample: np.ndarray,
    n_runs: int = 100,
) -> Dict[str, float]:
    """
    Measure inference time for a single patient.
    
    Target: 95th percentile < 10ms on Raspberry Pi 4.
    """
    times = []
    
    for _ in range(n_runs):
        # Single patient inference
        x = X_sample[np.random.randint(len(X_sample))].reshape(1, -1)
        
        start = time.perf_counter()
        model.predict(x, return_ci=True)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        'mean_ms': float(times.mean()),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
    }


def full_patient_workup_time(
    macf_models: Dict,
    risk_predictor,
    X_sample: np.ndarray,
    n_runs: int = 50,
) -> Dict[str, float]:
    """
    Measure full patient workup time:
    risk score + 4 counterfactual predictions.
    
    Target: < 100ms on Raspberry Pi 4.
    """
    times = []
    
    for _ in range(n_runs):
        x = X_sample[np.random.randint(len(X_sample))].reshape(1, -1)
        
        start = time.perf_counter()
        
        # Risk prediction
        risk = risk_predictor.predict_proba(x)
        
        # 4 counterfactual predictions
        for t_name, macf in macf_models.items():
            tau, ci_lo, ci_hi = macf.predict(x, return_ci=True)
        
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        'mean_ms': float(times.mean()),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
    }
