"""
Honest Tree Splitting — Base Module

Implements the honest splitting strategy from Athey & Imbens (2016):
  - A random subsample is split into STRUCTURE and ESTIMATION halves
  - The STRUCTURE half determines the tree splits (partition)
  - The ESTIMATION half populates leaf statistics (τ̂ and variance)

This prevents overfitting: the leaf estimates are computed on data
that was never used to choose the splits.

The MACFTree class in macf.py builds on this foundation and adds
missingness-aware splitting on top of honest estimation.

References:
    Athey, S. & Imbens, G. (2016). "Recursive Partitioning for
    Heterogeneous Causal Effects." PNAS 113(27):7353–7360.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HonestSplit:
    """Container for an honest train split."""
    X_struct: np.ndarray     # Features for tree structure
    Y_struct: np.ndarray     # Outcomes for tree structure
    T_struct: np.ndarray     # Treatments for tree structure
    X_est: np.ndarray        # Features for estimation
    Y_est: np.ndarray        # Outcomes for estimation
    T_est: np.ndarray        # Treatments for estimation
    struct_idx: np.ndarray   # Original indices of structure samples
    est_idx: np.ndarray      # Original indices of estimation samples


def create_honest_split(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    honesty_fraction: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> HonestSplit:
    """
    Create an honest split of the data into structure and estimation samples.

    Args:
        X: Feature matrix (n × p)
        Y: Outcome vector (n,)
        T: Treatment indicator (n,)
        honesty_fraction: Fraction used for structure (default 0.5)
        rng: Random number generator

    Returns:
        HonestSplit dataclass with partitioned data
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(X)
    indices = rng.permutation(n)
    n_struct = int(n * honesty_fraction)

    struct_idx = indices[:n_struct]
    est_idx = indices[n_struct:]

    return HonestSplit(
        X_struct=X[struct_idx],
        Y_struct=Y[struct_idx],
        T_struct=T[struct_idx],
        X_est=X[est_idx],
        Y_est=Y[est_idx],
        T_est=T[est_idx],
        struct_idx=struct_idx,
        est_idx=est_idx,
    )


def compute_leaf_estimate(
    Y: np.ndarray,
    T: np.ndarray,
    min_samples: int = 5,
) -> Tuple[float, float]:
    """
    Compute treatment effect estimate and variance in a leaf node.

    Uses simple difference-in-means:
        τ̂ = mean(Y | T=1) - mean(Y | T=0)
    with variance:
        Var(τ̂) = Var(Y|T=1)/n1 + Var(Y|T=0)/n0

    Args:
        Y: Outcome values in the leaf
        T: Treatment indicators in the leaf
        min_samples: Minimum treated/control required

    Returns:
        (tau_hat, tau_variance)
    """
    treated = T == 1
    control = T == 0

    n_treated = treated.sum()
    n_control = control.sum()

    if n_treated < min_samples or n_control < min_samples:
        return 0.0, float('inf')

    tau = Y[treated].mean() - Y[control].mean()

    var_t = Y[treated].var() / n_treated if n_treated > 1 else float('inf')
    var_c = Y[control].var() / n_control if n_control > 1 else float('inf')

    return float(tau), float(var_t + var_c)


# Re-export MACFTree for convenience (the full implementation
# lives in macf.py, which adds missingness-aware splitting)
from src.models.macf import MACFTree, TreeNode

__all__ = [
    'HonestSplit',
    'create_honest_split',
    'compute_leaf_estimate',
    'MACFTree',
    'TreeNode',
]
