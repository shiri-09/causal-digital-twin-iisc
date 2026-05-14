"""
E-value Sensitivity Analysis

Implements the VanderWeele & Ding (2017) E-value method for
assessing robustness of causal estimates to unmeasured confounding.

The E-value answers: "How strong would an unmeasured confounder need
to be (in terms of risk ratios with both treatment and outcome) to
fully explain away the observed treatment effect?"

Target: E-value > 2.0 for all claimed treatment effects.
"""

import numpy as np
from typing import Dict, Optional


def compute_e_value_from_rr(risk_ratio: float) -> float:
    """
    Compute E-value from a risk ratio.
    
    E-value = RR + sqrt(RR * (RR - 1))
    
    For protective effects (RR < 1), use 1/RR.
    """
    if risk_ratio < 1:
        risk_ratio = 1 / risk_ratio
    
    if risk_ratio <= 1.0:
        return 1.0
    
    return float(risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1)))


def ate_to_risk_ratio(
    ate: float,
    baseline_risk: float,
) -> float:
    """
    Convert Average Treatment Effect (risk difference) to Risk Ratio.
    
    RR = (baseline_risk + ate) / baseline_risk
    
    Args:
        ate: Average treatment effect (negative = protective)
        baseline_risk: P(Y=1) in untreated population
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        return 1.0
    
    treated_risk = max(0.001, min(0.999, baseline_risk + ate))
    rr = treated_risk / baseline_risk
    
    return float(rr)


def compute_e_values(
    ate: float,
    baseline_risk: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute E-value for an estimated treatment effect.
    
    Args:
        ate: Estimated ATE (negative = protective effect)
        baseline_risk: P(Y=1 | T=0) baseline MCI probability
        ci_lower: Lower bound of ATE confidence interval
        ci_upper: Upper bound of ATE confidence interval
    
    Returns:
        Dictionary with E-value and interpretation
    """
    rr = ate_to_risk_ratio(ate, baseline_risk)
    e_value = compute_e_value_from_rr(rr)
    
    results = {
        'ate': float(ate),
        'baseline_risk': float(baseline_risk),
        'risk_ratio': float(rr),
        'e_value': float(e_value),
    }
    
    # E-value for CI bound (more conservative)
    if ci_lower is not None and ci_upper is not None:
        # For protective effect, use the bound closest to null
        if ate < 0:
            bound_ate = ci_upper  # closer to 0
        else:
            bound_ate = ci_lower
        
        bound_rr = ate_to_risk_ratio(bound_ate, baseline_risk)
        bound_e = compute_e_value_from_rr(bound_rr)
        
        results['e_value_ci_bound'] = float(bound_e)
    
    # Interpretation
    if e_value >= 3.0:
        results['interpretation'] = (
            f"Strong: An unmeasured confounder would need risk ratio >{e_value:.1f} "
            f"with both treatment and outcome to explain away this effect."
        )
    elif e_value >= 2.0:
        results['interpretation'] = (
            f"Moderate: Unmeasured confounder needs RR >{e_value:.1f} to nullify. "
            f"Meets the >2.0 threshold."
        )
    elif e_value >= 1.5:
        results['interpretation'] = (
            f"Weak: E-value {e_value:.2f} is below 2.0 threshold. "
            f"Report as 'inconclusive' per proposal failure conditions."
        )
    else:
        results['interpretation'] = (
            f"Insufficient: E-value {e_value:.2f} too low. "
            f"Effect could easily be explained by unmeasured confounding."
        )
    
    return results


def run_sensitivity_analysis(
    treatment_effects: Dict[str, Dict],
    baseline_risk: float,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run E-value sensitivity analysis for all treatments.
    
    Args:
        treatment_effects: Dict mapping treatment → {ate, ci_lower, ci_upper}
        baseline_risk: Baseline MCI probability
    
    Returns:
        E-value results for each treatment
    """
    all_results = {}
    
    for t_name, t_data in treatment_effects.items():
        e_results = compute_e_values(
            ate=t_data['ate'],
            baseline_risk=baseline_risk,
            ci_lower=t_data.get('ci_lower'),
            ci_upper=t_data.get('ci_upper'),
        )
        
        all_results[t_name] = e_results
        
        if verbose:
            status = "✓" if e_results['e_value'] >= 2.0 else "⚠"
            print(f"  {status} {t_name}: E-value = {e_results['e_value']:.2f}")
            print(f"    {e_results['interpretation']}")
    
    return all_results


if __name__ == "__main__":
    print("E-value Sensitivity Analysis Demo")
    print("=" * 50)
    
    # Simulate treatment effects from our proposal
    effects = {
        'hba1c_reduced': {'ate': -0.12, 'ci_lower': -0.18, 'ci_upper': -0.06},
        'bp_managed': {'ate': -0.08, 'ci_lower': -0.14, 'ci_upper': -0.02},
        'activity_increased': {'ate': -0.06, 'ci_lower': -0.11, 'ci_upper': -0.01},
        'ldl_reduced': {'ate': -0.04, 'ci_lower': -0.09, 'ci_upper': 0.01},
    }
    
    results = run_sensitivity_analysis(effects, baseline_risk=0.34)
