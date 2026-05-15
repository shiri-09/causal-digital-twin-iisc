"""
Missing Indicators

Shared utility for creating binary missingness indicators and
median-filling NaN values. Used by both DML nuisance estimation
and the LightGBM risk predictor.

The key insight: missingness in clinical data is often informative
(MNAR — Missing Not At Random). By encoding missingness as explicit
binary features, models can learn from the pattern of *what* is
missing, not just the observed values.
"""

import numpy as np
from typing import Tuple, Optional


def add_missing_indicators(
    X: np.ndarray,
    threshold: float = 0.05,
    medians: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add binary missing indicators and fill NaN with column median.

    For each feature with missingness rate > threshold, appends a
    binary column (1 = missing, 0 = observed). Then fills all NaN
    values with column medians.

    Args:
        X: Feature matrix (n × p), may contain NaN
        threshold: Minimum missingness rate to create an indicator
        medians: Pre-computed medians (for test data). If None, compute from X.

    Returns:
        X_augmented: Feature matrix with indicators appended and NaN filled
        medians: Column medians used for filling (save for test-time use)
    """
    n, p = X.shape
    missing_mask = np.isnan(X)

    # Identify features with significant missingness
    miss_rates = missing_mask.mean(axis=0)
    high_miss_cols = np.where(miss_rates > threshold)[0]

    # Create binary indicators for high-missingness features
    indicators = missing_mask[:, high_miss_cols].astype(np.float64)

    # Fill NaN with column median
    X_filled = X.copy()
    if medians is None:
        medians = np.nanmedian(X_filled, axis=0)

    for col in range(p):
        nan_mask = np.isnan(X_filled[:, col])
        if nan_mask.any():
            X_filled[nan_mask, col] = medians[col]

    # Concatenate features + indicators
    X_augmented = np.hstack([X_filled, indicators])

    return X_augmented, medians


def get_missing_feature_indices(
    X: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Return indices of features with missingness above threshold.

    Useful for understanding which missing indicators were created.
    """
    missing_mask = np.isnan(X)
    miss_rates = missing_mask.mean(axis=0)
    return np.where(miss_rates > threshold)[0]


if __name__ == "__main__":
    from src.data.synthetic_mci import generate_synthetic_mci_data

    print("Testing missing indicators...")
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=1000)

    X_aug, medians = add_missing_indicators(X.values)
    n_indicators = X_aug.shape[1] - X.shape[1]

    print(f"  Original features: {X.shape[1]}")
    print(f"  Missing indicators added: {n_indicators}")
    print(f"  Augmented features: {X_aug.shape[1]}")
    print(f"  NaN remaining: {np.isnan(X_aug).sum()}")

    high_miss = get_missing_feature_indices(X.values)
    print(f"  High-missingness columns: {high_miss}")
