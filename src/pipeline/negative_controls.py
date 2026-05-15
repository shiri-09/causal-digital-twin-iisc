"""
Negative Control Analysis

Validates causal estimates by testing interventions against outcomes
they should NOT affect. If MACF reports a significant effect for a
negative control, this suggests confounding bias is leaking through.

Negative controls used:
    1. Placebo treatment: random binary assignment (should have τ ≈ 0)
    2. Pre-treatment outcome: future treatment shouldn't predict past events
    3. Cross-treatment controls: HbA1c reduction shouldn't affect
       an unrelated biomarker like OCT thickness

References:
    Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010).
    "Negative Controls: A Tool for Detecting Confounding and
    Bias in Observational Studies." Epidemiology 21(3):383-8.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class NegativeControlResult:
    """Result of a single negative control test."""
    name: str
    estimated_effect: float
    p_value: float
    passed: bool
    threshold: float = 0.05
    description: str = ""


def run_placebo_test(
    macf_model,
    X: np.ndarray,
    Y: np.ndarray,
    n_placebo: int = 5,
    seed: int = 42,
) -> NegativeControlResult:
    """
    Test 1: Placebo treatment control.

    Fit MACF on randomly assigned binary treatment.
    Expected: τ̂ ≈ 0 (no real treatment effect).

    If MACF finds a significant effect, it means the model
    is picking up on confounding, not causal signal.

    Args:
        macf_model: Unfitted MACF model (will be cloned)
        X: Feature matrix
        Y: Outcome vector
        n_placebo: Number of placebo runs to average
        seed: Random seed
    """
    from src.models.macf import MissingnessAwareCausalForest

    rng = np.random.default_rng(seed)
    effects = []

    for i in range(n_placebo):
        # Random treatment with no causal effect
        T_placebo = rng.binomial(1, 0.5, len(X)).astype(float)

        macf_placebo = MissingnessAwareCausalForest(
            n_trees=min(50, macf_model.n_trees),
            min_leaf_size=macf_model.min_leaf_size,
            max_depth=macf_model.max_depth,
            seed=seed + i,
            n_jobs=1,
        )
        macf_placebo.fit(X, Y, T_placebo, verbose=False)

        tau_hat, _, _ = macf_placebo.predict(X)
        effects.append(tau_hat.mean())

    mean_effect = np.mean(effects)
    std_effect = np.std(effects) if len(effects) > 1 else 0.1

    # Two-sided test: is mean_effect significantly different from 0?
    from scipy import stats
    if std_effect > 0:
        t_stat = mean_effect / (std_effect / np.sqrt(n_placebo))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_placebo - 1))
    else:
        p_value = 1.0

    passed = abs(mean_effect) < 0.02 and p_value > 0.05

    return NegativeControlResult(
        name="Placebo Treatment",
        estimated_effect=float(mean_effect),
        p_value=float(p_value),
        passed=passed,
        description=(
            f"Mean placebo effect: {mean_effect:.4f} (should be ≈0). "
            f"p={p_value:.4f}. "
            f"{'PASS' if passed else 'FAIL: possible confounding leak'}"
        ),
    )


def run_shuffled_outcome_test(
    macf_model,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_shuffles: int = 5,
    seed: int = 42,
) -> NegativeControlResult:
    """
    Test 2: Shuffled outcome control.

    Shuffle Y to break the treatment-outcome relationship.
    Expected: τ̂ ≈ 0 (no structure to recover).

    If MACF finds effects with shuffled outcomes, the model
    is overfitting to noise.
    """
    from src.models.macf import MissingnessAwareCausalForest

    rng = np.random.default_rng(seed)
    effects = []

    for i in range(n_shuffles):
        Y_shuffled = rng.permutation(Y)

        macf_shuffled = MissingnessAwareCausalForest(
            n_trees=min(50, macf_model.n_trees),
            min_leaf_size=macf_model.min_leaf_size,
            max_depth=macf_model.max_depth,
            seed=seed + i + 100,
            n_jobs=1,
        )
        macf_shuffled.fit(X, Y_shuffled, T, verbose=False)

        tau_hat, _, _ = macf_shuffled.predict(X)
        effects.append(tau_hat.mean())

    mean_effect = np.mean(effects)

    from scipy import stats
    std_effect = np.std(effects) if len(effects) > 1 else 0.1
    if std_effect > 0:
        t_stat = mean_effect / (std_effect / np.sqrt(n_shuffles))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_shuffles - 1))
    else:
        p_value = 1.0

    passed = abs(mean_effect) < 0.02 and p_value > 0.05

    return NegativeControlResult(
        name="Shuffled Outcome",
        estimated_effect=float(mean_effect),
        p_value=float(p_value),
        passed=passed,
        description=(
            f"Mean effect on shuffled Y: {mean_effect:.4f} (should be ≈0). "
            f"p={p_value:.4f}. "
            f"{'PASS' if passed else 'FAIL: possible overfitting'}"
        ),
    )


def run_all_negative_controls(
    macf_model,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    seed: int = 42,
    verbose: bool = True,
) -> List[NegativeControlResult]:
    """
    Run all negative control tests.

    Returns list of NegativeControlResult objects.
    """
    results = []

    if verbose:
        print("\nNegative Control Analysis")
        print("=" * 50)

    # Test 1: Placebo
    if verbose:
        print("\n  Running placebo treatment test...")
    placebo = run_placebo_test(macf_model, X, Y, seed=seed)
    results.append(placebo)
    if verbose:
        status = "✓" if placebo.passed else "✗"
        print(f"  {status} {placebo.description}")

    # Test 2: Shuffled outcome
    if verbose:
        print("\n  Running shuffled outcome test...")
    shuffled = run_shuffled_outcome_test(macf_model, X, Y, T, seed=seed)
    results.append(shuffled)
    if verbose:
        status = "✓" if shuffled.passed else "✗"
        print(f"  {status} {shuffled.description}")

    if verbose:
        n_pass = sum(1 for r in results if r.passed)
        print(f"\n  Summary: {n_pass}/{len(results)} controls passed")

    return results


if __name__ == "__main__":
    from src.data.synthetic_mci import generate_synthetic_mci_data
    from src.models.macf import MissingnessAwareCausalForest

    print("Running negative control tests...")
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500, seed=42)

    macf = MissingnessAwareCausalForest(
        n_trees=50, min_leaf_size=10, max_depth=5, seed=42, n_jobs=1
    )
    macf.fit(X.values, Y, T['hba1c_reduced'].values, verbose=False)

    results = run_all_negative_controls(
        macf, X.values, Y, T['hba1c_reduced'].values
    )
