"""
Double Machine Learning (DML) Nuisance Estimation

Implements doubly robust nuisance parameter estimation following
Chernozhukov et al. (2018) "Double/Debiased Machine Learning."

Two nuisance models are estimated via 5-fold cross-fitting:
    1. Outcome model: μ̂(X) = E[Y | X]  (risk of MCI given features)
    2. Propensity model: ê(X) = P(T=1 | X)  (probability of treatment)

These are used to residualize Y and T before feeding into MACF,
which removes confounding bias in the CATE estimates.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Optional, Dict
import warnings


def _get_lightgbm_model(task: str = 'regression'):
    """Create a LightGBM model with sensible defaults for clinical data."""
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        warnings.warn("LightGBM not available, falling back to sklearn GBM")
        if task == 'classification':
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    
    if task == 'classification':
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
    else:
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )


def _add_missing_indicators(X: np.ndarray) -> np.ndarray:
    """
    Add binary missing indicators for features with >5% missingness.
    Then fill NaN with column median for the nuisance models.
    
    This is the preprocessing step described in the proposal:
    "MACF receives raw missing values plus explicit binary missing indicators"
    """
    n, p = X.shape
    missing_mask = np.isnan(X)
    
    # Identify features with >5% missingness
    miss_rates = missing_mask.mean(axis=0)
    high_miss_cols = np.where(miss_rates > 0.05)[0]
    
    # Create missing indicators
    indicators = missing_mask[:, high_miss_cols].astype(np.float64)
    
    # Fill NaN with column median (for nuisance models only)
    X_filled = X.copy()
    for col in range(p):
        col_vals = X_filled[:, col]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            median_val = np.nanmedian(col_vals)
            col_vals[nan_mask] = median_val
    
    # Concatenate features + missing indicators
    X_augmented = np.hstack([X_filled, indicators])
    
    return X_augmented


def cross_fit_nuisance(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform 5-fold cross-fitted nuisance estimation.
    
    For each fold:
        - Train outcome model μ̂(X) on out-of-fold data
        - Train propensity model ê(X) on out-of-fold data
        - Predict on held-out fold
    
    This avoids overfitting bias in the nuisance estimates.
    
    Args:
        X: Features (n × p), may contain NaN
        Y: Binary outcome (n,)
        T: Binary treatment (n,)
        n_folds: Number of CV folds
        seed: Random seed
    
    Returns:
        Y_residual: Y - μ̂(X)  (residualized outcome)
        T_residual: T - ê(X)  (residualized treatment)
        mu_hat: Predicted E[Y|X] for all samples
        e_hat: Predicted P(T=1|X) for all samples
    """
    n = len(X)
    
    # Augment features with missing indicators
    X_aug = _add_missing_indicators(X)
    
    mu_hat = np.zeros(n)  # outcome predictions
    e_hat = np.zeros(n)   # propensity predictions
    
    # Stratified K-fold on treatment (to maintain T balance)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_aug, T)):
        X_train, X_val = X_aug[train_idx], X_aug[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        T_train, T_val = T[train_idx], T[val_idx]
        
        # Outcome model: E[Y | X]
        outcome_model = _get_lightgbm_model('classification')
        outcome_model.fit(X_train, Y_train)
        mu_hat[val_idx] = outcome_model.predict_proba(X_val)[:, 1]
        
        # Propensity model: P(T=1 | X)
        propensity_model = _get_lightgbm_model('classification')
        propensity_model.fit(X_train, T_train)
        e_hat[val_idx] = propensity_model.predict_proba(X_val)[:, 1]
    
    # Clip propensity to avoid extreme weights
    e_hat = np.clip(e_hat, 0.05, 0.95)
    
    # Residualize
    Y_residual = Y - mu_hat
    T_residual = T - e_hat
    
    return Y_residual, T_residual, mu_hat, e_hat


def fit_all_nuisance_models(
    X: np.ndarray,
    Y: np.ndarray,
    treatments: Dict[str, np.ndarray],
    n_folds: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Fit nuisance models for all four treatments.
    
    Args:
        X: Features
        Y: Outcome
        treatments: Dict mapping treatment name → binary array
        n_folds: CV folds
        seed: Random seed
    
    Returns:
        Dict mapping treatment name → {Y_residual, T_residual, mu_hat, e_hat}
    """
    results = {}
    
    for t_idx, (t_name, T) in enumerate(treatments.items()):
        if verbose:
            print(f"\nFitting nuisance for treatment: {t_name} ({t_idx+1}/{len(treatments)})")
        
        Y_res, T_res, mu_hat, e_hat = cross_fit_nuisance(
            X, Y, T, n_folds=n_folds, seed=seed + t_idx
        )
        
        results[t_name] = {
            'Y_residual': Y_res,
            'T_residual': T_res,
            'mu_hat': mu_hat,
            'e_hat': e_hat,
        }
        
        if verbose:
            print(f"  μ̂(X) range: [{mu_hat.min():.3f}, {mu_hat.max():.3f}]")
            print(f"  ê(X) range: [{e_hat.min():.3f}, {e_hat.max():.3f}]")
    
    return results


if __name__ == "__main__":
    from src.data.synthetic_mci import generate_synthetic_mci_data
    
    print("Testing DML nuisance estimation...")
    X, T_df, Y, tau, prob = generate_synthetic_mci_data(n_samples=1000)
    
    X_np = X.values
    treatments = {col: T_df[col].values for col in T_df.columns}
    
    results = fit_all_nuisance_models(X_np, Y, treatments, n_folds=3)
    
    for t_name, res in results.items():
        print(f"\n{t_name}:")
        print(f"  Y residual std: {res['Y_residual'].std():.4f}")
        print(f"  T residual std: {res['T_residual'].std():.4f}")
