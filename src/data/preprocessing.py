"""
Data Preprocessing

Standardization, binary missingness indicator creation, and
feature validation for the MACF pipeline.

The key insight: we do NOT impute missing values for MACF.
Instead, we create explicit binary indicators and pass raw NaN
to the causal forest, which handles them via missingness-aware splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


FEATURE_NAMES = [
    'age', 'sex', 'education_years', 'hba1c', 'systolic_bp',
    'diastolic_bp', 'ldl', 'physical_activity_min_week',
    'gait_speed', 'oct_rnfl_thickness', 'mmse_score', 'hrv_sdnn'
]

# Features expected to have significant missingness (MNAR)
HIGH_MISS_FEATURES = ['oct_rnfl_thickness', 'gait_speed', 'mmse_score', 'hrv_sdnn']


def create_missing_indicators(X: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Create binary indicators for features with missingness above threshold.

    For each feature with >5% missing, adds a column 'feature_missing'
    with value 1 where the original is NaN, 0 otherwise.

    Args:
        X: Feature DataFrame
        threshold: Minimum missingness rate to create indicator

    Returns:
        DataFrame with original features + missing indicators
    """
    indicators = pd.DataFrame(index=X.index)

    for col in X.columns:
        miss_rate = X[col].isna().mean()
        if miss_rate > threshold:
            indicators[f"{col}_missing"] = X[col].isna().astype(int)

    return pd.concat([X, indicators], axis=1)


def standardize_features(
    X: pd.DataFrame,
    fit_stats: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Z-score standardize continuous features.

    Note: Binary features (sex) and missing indicators are NOT standardized.
    NaN values are preserved (not filled).

    Args:
        X: Feature DataFrame
        fit_stats: Pre-computed mean/std (for test set). If None, compute from X.

    Returns:
        Standardized DataFrame and fit statistics dict
    """
    binary_cols = ['sex'] + [c for c in X.columns if c.endswith('_missing')]
    continuous_cols = [c for c in X.columns if c not in binary_cols]

    if fit_stats is None:
        fit_stats = {}
        for col in continuous_cols:
            fit_stats[col] = {
                'mean': float(X[col].mean()),
                'std': float(X[col].std()),
            }

    X_out = X.copy()
    for col in continuous_cols:
        if col in fit_stats:
            mean = fit_stats[col]['mean']
            std = fit_stats[col]['std']
            if std > 1e-10:
                X_out[col] = (X_out[col] - mean) / std

    return X_out, fit_stats


def validate_features(X: pd.DataFrame) -> Dict:
    """
    Validate feature DataFrame for expected structure.

    Returns dict with validation results and any warnings.
    """
    results = {
        'valid': True,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'warnings': [],
    }

    # Check expected features exist
    for feat in FEATURE_NAMES:
        if feat not in X.columns:
            results['warnings'].append(f"Missing expected feature: {feat}")
            results['valid'] = False

    # Check for reasonable ranges
    ranges = {
        'age': (40, 100),
        'hba1c': (3.0, 16.0),
        'systolic_bp': (70, 250),
        'mmse_score': (0, 30),
    }

    for feat, (lo, hi) in ranges.items():
        if feat in X.columns:
            vals = X[feat].dropna()
            if len(vals) > 0:
                if vals.min() < lo or vals.max() > hi:
                    results['warnings'].append(
                        f"{feat} has values outside expected range [{lo}, {hi}]"
                    )

    # Check missingness rates
    for feat in HIGH_MISS_FEATURES:
        if feat in X.columns:
            miss = X[feat].isna().mean()
            results[f'{feat}_missing_pct'] = round(miss * 100, 1)

    return results


def preprocess_pipeline(
    X: pd.DataFrame,
    fit_stats: Optional[Dict] = None,
    add_indicators: bool = True,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline.

    1. Validate features
    2. Add missing indicators
    3. Standardize continuous features

    Args:
        X: Raw feature DataFrame
        fit_stats: Pre-computed standardization stats (for test data)
        add_indicators: Whether to add missing indicators
        standardize: Whether to standardize

    Returns:
        Preprocessed DataFrame and fit statistics
    """
    validation = validate_features(X)
    if not validation['valid']:
        import warnings
        for w in validation['warnings']:
            warnings.warn(w)

    if add_indicators:
        X = create_missing_indicators(X)

    if standardize:
        X, fit_stats = standardize_features(X, fit_stats)

    return X, fit_stats


if __name__ == "__main__":
    from src.data.synthetic_mci import generate_synthetic_mci_data

    print("Testing preprocessing pipeline...")
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500)

    # Validate
    validation = validate_features(X)
    print(f"  Valid: {validation['valid']}")
    print(f"  Features: {validation['n_features']}")

    # Preprocess
    X_proc, stats = preprocess_pipeline(X)
    print(f"  After preprocessing: {X_proc.shape[1]} features "
          f"(added {X_proc.shape[1] - X.shape[1]} indicators)")
    print(f"  NaN preserved: {X_proc.isna().any().any()}")
