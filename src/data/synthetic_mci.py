"""
Synthetic MCI Data Generator

Generates SANSCOG-like synthetic patient data with:
- 12 clinical features matching SANSCOG/TLSA cohort structure
- 4 binary treatments (HbA1c reduction, BP management, activity increase, LDL reduction)
- Known ground-truth CATE (Conditional Average Treatment Effect) for each treatment
- Realistic MNAR missingness (OCT: 41%, gait_speed: 28%, cognitive: 15%)
- Binary 2-year MCI outcome

This enables PEHE computation since true τ(x) is known by construction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


# Feature specification matching SANSCOG cohort
FEATURE_SPEC = {
    'age': {'mean': 62.0, 'std': 10.0, 'min': 45, 'max': 90},
    'sex': {'type': 'binary', 'p': 0.52},  # 52% female in SANSCOG
    'education_years': {'mean': 7.5, 'std': 4.5, 'min': 0, 'max': 20},
    'hba1c': {'mean': 6.2, 'std': 1.1, 'min': 4.0, 'max': 14.0},
    'systolic_bp': {'mean': 138.0, 'std': 20.0, 'min': 90, 'max': 220},
    'diastolic_bp': {'mean': 82.0, 'std': 12.0, 'min': 50, 'max': 130},
    'ldl': {'mean': 120.0, 'std': 35.0, 'min': 40, 'max': 250},
    'physical_activity_min_week': {'mean': 120.0, 'std': 90.0, 'min': 0, 'max': 600},
    'gait_speed': {'mean': 0.95, 'std': 0.25, 'min': 0.2, 'max': 1.8},
    'oct_rnfl_thickness': {'mean': 95.0, 'std': 15.0, 'min': 50, 'max': 140},
    'mmse_score': {'mean': 26.5, 'std': 3.0, 'min': 10, 'max': 30},
    'hrv_sdnn': {'mean': 42.0, 'std': 18.0, 'min': 5, 'max': 120},
}

# Missingness rates matching SANSCOG (MNAR)
MISSINGNESS_RATES = {
    'oct_rnfl_thickness': 0.41,   # 41% missing - highest
    'gait_speed': 0.28,           # 28% missing
    'mmse_score': 0.15,           # 15% missing
    'hrv_sdnn': 0.12,             # 12% missing
}

# Treatment definitions
TREATMENTS = ['hba1c_reduced', 'bp_managed', 'activity_increased', 'ldl_reduced']


def _generate_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate baseline clinical features for n patients."""
    data = {}
    
    for feat, spec in FEATURE_SPEC.items():
        if spec.get('type') == 'binary':
            data[feat] = rng.binomial(1, spec['p'], n).astype(float)
        else:
            values = rng.normal(spec['mean'], spec['std'], n)
            values = np.clip(values, spec['min'], spec['max'])
            data[feat] = values
    
    # Add correlations to make data realistic
    # Age correlates with higher HbA1c
    data['hba1c'] += 0.02 * (data['age'] - 62) + rng.normal(0, 0.1, n)
    data['hba1c'] = np.clip(data['hba1c'], 4.0, 14.0)
    
    # Age correlates with higher BP
    data['systolic_bp'] += 0.5 * (data['age'] - 62) + rng.normal(0, 2, n)
    data['systolic_bp'] = np.clip(data['systolic_bp'], 90, 220)
    
    # Lower education correlates with lower MMSE
    data['mmse_score'] += 0.15 * (data['education_years'] - 7.5)
    data['mmse_score'] = np.clip(data['mmse_score'], 10, 30)
    
    # Age negatively correlates with gait speed
    data['gait_speed'] -= 0.008 * (data['age'] - 62)
    data['gait_speed'] = np.clip(data['gait_speed'], 0.2, 1.8)
    
    # Lower OCT thickness with age
    data['oct_rnfl_thickness'] -= 0.3 * (data['age'] - 62)
    data['oct_rnfl_thickness'] = np.clip(data['oct_rnfl_thickness'], 50, 140)
    
    return pd.DataFrame(data)


def _generate_treatment_assignments(
    X: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate treatment assignments based on patient features.
    
    Treatments are NOT randomly assigned (observational data).
    Treatment probability depends on baseline features to simulate
    confounding — this is what makes causal inference necessary.
    """
    treatments = {}
    
    # HbA1c reduction: more likely if HbA1c is high
    p_hba1c = 1 / (1 + np.exp(-(X['hba1c'] - 7.0) / 0.8))
    treatments['hba1c_reduced'] = rng.binomial(1, p_hba1c).astype(float)
    
    # BP management: more likely if systolic BP is high
    p_bp = 1 / (1 + np.exp(-(X['systolic_bp'] - 140) / 15))
    treatments['bp_managed'] = rng.binomial(1, p_bp).astype(float)
    
    # Activity increase: more likely if younger and educated
    logit_activity = -0.03 * (X['age'] - 55) + 0.05 * (X['education_years'] - 7)
    p_activity = 1 / (1 + np.exp(-logit_activity))
    treatments['activity_increased'] = rng.binomial(1, p_activity).astype(float)
    
    # LDL reduction: more likely if LDL is high
    p_ldl = 1 / (1 + np.exp(-(X['ldl'] - 130) / 25))
    treatments['ldl_reduced'] = rng.binomial(1, p_ldl).astype(float)
    
    return pd.DataFrame(treatments)


def _compute_true_cate(X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute the TRUE Conditional Average Treatment Effect for each treatment.
    
    These are the ground-truth values we want MACF to recover.
    The CATE is heterogeneous — it varies by patient characteristics.
    
    Returns:
        Dictionary mapping treatment name → array of true τ(x) values
    """
    n = len(X)
    tau = {}
    
    # HbA1c reduction effect on MCI risk
    # Stronger effect for older patients and those with higher baseline HbA1c
    tau['hba1c_reduced'] = (
        -0.08                                        # base effect: 8pp risk reduction
        - 0.003 * (X['age'].values - 62)             # stronger for older
        - 0.02 * (X['hba1c'].values - 6.2)           # stronger for higher HbA1c
        + 0.005 * (X['education_years'].values - 7)  # weaker for more educated
    )
    
    # BP management effect
    # Stronger for those with higher BP and lower education
    tau['bp_managed'] = (
        -0.06
        - 0.002 * (X['systolic_bp'].values - 138)
        - 0.001 * (X['age'].values - 62)
    )
    
    # Physical activity increase effect
    # Stronger for sedentary patients and younger ones
    tau['activity_increased'] = (
        -0.05
        + 0.0003 * (X['physical_activity_min_week'].values - 120)  # weaker if already active
        + 0.001 * (X['age'].values - 70)  # weaker for very old
    )
    
    # LDL reduction effect
    # Moderate, relatively homogeneous
    tau['ldl_reduced'] = (
        -0.04
        - 0.001 * (X['ldl'].values - 120)
    )
    
    # Clip all CATE values to reasonable range
    for t in tau:
        tau[t] = np.clip(tau[t], -0.25, 0.05)
    
    return tau


def _generate_outcome(
    X: pd.DataFrame,
    T: pd.DataFrame,
    tau: Dict[str, np.ndarray],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate 2-year MCI outcome (binary) based on features, treatments, and true CATE.
    
    Y(1) = Y(0) + τ(x)  for each treatment
    The outcome is the probability of MCI, mapped through a logistic function.
    """
    # Baseline MCI risk (without any treatment effect)
    logit_baseline = (
        -2.5                                           # base intercept
        + 0.05 * (X['age'].values - 62)                # age increases risk
        - 0.03 * X['education_years'].values           # education protects
        + 0.15 * (X['hba1c'].values - 6.0)             # diabetes risk
        + 0.008 * (X['systolic_bp'].values - 130)      # hypertension risk
        - 0.4 * (X['gait_speed'].values - 0.95)        # slow gait = risk
        - 0.05 * (X['mmse_score'].values - 26)          # low cognition = risk
        + 0.005 * (X['ldl'].values - 120)               # cholesterol risk
        - 0.003 * X['physical_activity_min_week'].values # activity protects
        - 0.01 * (X['hrv_sdnn'].values - 42)            # low HRV = risk
    )
    
    # Add treatment effects
    total_treatment_effect = np.zeros(len(X))
    for treatment in TREATMENTS:
        # Treatment effect only applies to treated individuals
        total_treatment_effect += T[treatment].values * tau[treatment]
    
    # Convert to probability
    logit_y = logit_baseline + total_treatment_effect * 5  # scale to logit space
    prob_mci = 1 / (1 + np.exp(-logit_y))
    
    # Add noise and binarize
    prob_mci = np.clip(prob_mci + rng.normal(0, 0.02, len(X)), 0.01, 0.99)
    y = rng.binomial(1, prob_mci).astype(float)
    
    return y, prob_mci


def _inject_missingness(
    X: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Inject MNAR (Missing Not At Random) missingness into the dataset.
    
    Missingness depends on observed variables to simulate real-world patterns:
    - OCT: more likely missing for rural patients (low education proxy)
    - Gait speed: more likely missing for older/frailer patients
    - MMSE: more likely missing for patients who refuse testing
    """
    X_missing = X.copy()
    n = len(X)
    
    for feat, rate in MISSINGNESS_RATES.items():
        if feat == 'oct_rnfl_thickness':
            # MNAR: more likely missing if education < median (rural proxy)
            logit = -0.3 + 0.1 * (7.5 - X['education_years'].values)
            p_missing = rate * 2 * (1 / (1 + np.exp(-logit)))
            p_missing = np.clip(p_missing, 0, 0.95)
        elif feat == 'gait_speed':
            # MNAR: more likely missing for very old patients
            logit = -0.5 + 0.03 * (X['age'].values - 60)
            p_missing = rate * 2 * (1 / (1 + np.exp(-logit)))
            p_missing = np.clip(p_missing, 0, 0.95)
        else:
            # MAR: random missingness at specified rate
            p_missing = np.full(n, rate)
        
        mask = rng.binomial(1, p_missing).astype(bool)
        X_missing.loc[mask, feat] = np.nan
    
    return X_missing


def generate_synthetic_mci_data(
    n_samples: int = 6000,
    seed: int = 42,
    inject_missing: bool = True,
    return_ground_truth: bool = True
) -> Tuple:
    """
    Generate a complete synthetic MCI dataset mirroring SANSCOG structure.
    
    Args:
        n_samples: Number of synthetic patients (default 6000 ~ SANSCOG V1)
        seed: Random seed for reproducibility
        inject_missing: Whether to inject MNAR missingness
        return_ground_truth: Whether to return true CATE values
    
    Returns:
        If return_ground_truth:
            (X, T, Y, tau, prob_mci) where:
                X: Features DataFrame (n_samples × 12)
                T: Treatments DataFrame (n_samples × 4)  
                Y: Binary MCI outcome array
                tau: Dict of true CATE arrays per treatment
                prob_mci: True MCI probability (before binarization)
        Else:
            (X, T, Y)
    """
    rng = np.random.default_rng(seed)
    
    # Step 1: Generate baseline features
    X = _generate_features(n_samples, rng)
    
    # Step 2: Generate treatment assignments (confounded)
    T = _generate_treatment_assignments(X, rng)
    
    # Step 3: Compute true CATE (ground truth)
    tau = _compute_true_cate(X)
    
    # Step 4: Generate outcome based on features + treatments + CATE
    Y, prob_mci = _generate_outcome(X, T, tau, rng)
    
    # Step 5: Inject missingness (after generating outcome to avoid data leakage)
    if inject_missing:
        X = _inject_missingness(X, rng)
    
    if return_ground_truth:
        return X, T, Y, tau, prob_mci
    else:
        return X, T, Y


def generate_train_val_test_split(
    n_samples: int = 6000,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1
) -> Dict:
    """
    Generate synthetic data with stratified 80/10/10 split.
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        (X, T, Y, tau) tuples.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    
    X, T, Y, tau, prob_mci = generate_synthetic_mci_data(
        n_samples=n_samples, seed=seed, inject_missing=True, return_ground_truth=True
    )
    
    rng = np.random.default_rng(seed + 1)
    
    # Stratified split by MCI outcome
    idx_pos = np.where(Y == 1)[0]
    idx_neg = np.where(Y == 0)[0]
    
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    
    def _split_indices(indices):
        n = len(indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return (
            indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:]
        )
    
    train_pos, val_pos, test_pos = _split_indices(idx_pos)
    train_neg, val_neg, test_neg = _split_indices(idx_neg)
    
    train_idx = np.concatenate([train_pos, train_neg])
    val_idx = np.concatenate([val_pos, val_neg])
    test_idx = np.concatenate([test_pos, test_neg])
    
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    def _subset(idx):
        tau_sub = {k: v[idx] for k, v in tau.items()}
        return X.iloc[idx].reset_index(drop=True), T.iloc[idx].reset_index(drop=True), Y[idx], tau_sub
    
    return {
        'train': _subset(train_idx),
        'val': _subset(val_idx),
        'test': _subset(test_idx),
    }


def get_feature_names() -> list:
    """Return list of all feature names."""
    return list(FEATURE_SPEC.keys())


def get_treatment_names() -> list:
    """Return list of all treatment names."""
    return TREATMENTS.copy()


if __name__ == "__main__":
    # Quick test
    print("Generating synthetic MCI data...")
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=1000)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Treatments shape: {T.shape}")
    print(f"Outcome shape: {Y.shape}")
    print(f"MCI prevalence: {Y.mean():.2%}")
    
    print(f"\nMissingness rates:")
    for col in X.columns:
        miss_rate = X[col].isna().mean()
        if miss_rate > 0:
            print(f"  {col}: {miss_rate:.1%}")
    
    print(f"\nTreatment rates:")
    for col in T.columns:
        print(f"  {col}: {T[col].mean():.1%}")
    
    print(f"\nTrue CATE (mean ± std):")
    for t_name, t_vals in tau.items():
        print(f"  {t_name}: {t_vals.mean():.4f} ± {t_vals.std():.4f}")
