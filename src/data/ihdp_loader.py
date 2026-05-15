"""
IHDP Benchmark Data Loader — REAL DATA

Downloads and loads the Infant Health and Development Program (IHDP)
semi-synthetic benchmark dataset from the AMLab-Amsterdam CEVAE repository.

This is the standard causal inference benchmark used in:
  - Shalit et al. (2017) "Estimating individual treatment effects"
  - Louizos et al. (2017) "Causal Effect Inference with Deep Latent-Variable Models"
  - Hill (2011) "Bayesian Nonparametric Modeling for Causal Inference"

Dataset: 747 subjects, 25 covariates (6 continuous + 19 binary),
         binary treatment, factual + counterfactual outcomes.

Source: https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
import urllib.request
import hashlib

# ── Real IHDP dataset URLs (10 realizations from CEVAE repo) ────────────────
IHDP_BASE_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/"
    "master/datasets/IHDP/csv/ihdp_npci_{}.csv"
)

# Column specification for the IHDP NPCI CSV format
IHDP_COLUMNS = (
    ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
    + [f"x{i}" for i in range(1, 26)]
)

# Covariate metadata (from Hill 2011)
# x1-x6: continuous covariates
# x7-x25: binary covariates
COVARIATE_INFO = {
    "x1": {"type": "continuous", "description": "birth weight (grams, normalized)"},
    "x2": {"type": "continuous", "description": "head circumference (cm, normalized)"},
    "x3": {"type": "continuous", "description": "weeks pre-term"},
    "x4": {"type": "continuous", "description": "birth weight (grams, normalized alt)"},
    "x5": {"type": "continuous", "description": "neonatal health index"},
    "x6": {"type": "continuous", "description": "mother's age"},
    "x7": {"type": "binary", "description": "sex (1=male)"},
    "x8": {"type": "binary", "description": "twin status"},
    "x9": {"type": "binary", "description": "mother married"},
    "x10": {"type": "binary", "description": "mother education: high school"},
    "x11": {"type": "binary", "description": "mother education: some college"},
    "x12": {"type": "binary", "description": "mother education: college+"},
    "x13": {"type": "binary", "description": "mother worked during pregnancy"},
    "x14": {"type": "binary", "description": "prenatal care: 1st trimester"},
    "x15": {"type": "binary", "description": "prenatal care: 2nd trimester"},
    "x16": {"type": "binary", "description": "mother smoked during pregnancy"},
    "x17": {"type": "binary", "description": "mother used alcohol"},
    "x18": {"type": "binary", "description": "mother used drugs"},
    "x19": {"type": "binary", "description": "adequate prenatal care"},
    "x20": {"type": "binary", "description": "race: white"},
    "x21": {"type": "binary", "description": "race: black"},
    "x22": {"type": "binary", "description": "race: hispanic"},
    "x23": {"type": "binary", "description": "race: other"},
    "x24": {"type": "binary", "description": "Medicaid recipient"},
    "x25": {"type": "binary", "description": "site indicator"},
}

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


def _get_cache_path(realization: int) -> Path:
    """Get the cache file path for a given IHDP realization."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"ihdp_npci_{realization}.csv"


def _download_ihdp(realization: int = 1) -> pd.DataFrame:
    """
    Download real IHDP data from GitHub.

    Args:
        realization: Which realization to download (1-10).
                    Different realizations have different outcome surfaces
                    but the same covariates and treatment assignments.

    Returns:
        DataFrame with 747 rows and 30 columns
    """
    cache_path = _get_cache_path(realization)

    if cache_path.exists():
        print(f"  Loading cached IHDP realization {realization} from {cache_path}")
        data = pd.read_csv(cache_path, header=None)
    else:
        url = IHDP_BASE_URL.format(realization)
        print(f"  Downloading IHDP realization {realization} from {url}")
        try:
            data = pd.read_csv(url, header=None)
            # Cache for future use
            data.to_csv(cache_path, index=False, header=False)
            print(f"  Cached to {cache_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download IHDP data from {url}. "
                f"Check your internet connection. Error: {e}"
            )

    # Validate shape
    assert data.shape[1] == 30, (
        f"Expected 30 columns, got {data.shape[1]}. "
        f"The IHDP CSV format may have changed."
    )

    data.columns = IHDP_COLUMNS
    return data


def load_ihdp(
    realization: int = 1,
    remove_confounding: bool = False,
) -> Dict:
    """
    Load the real IHDP benchmark dataset.

    This dataset contains REAL covariates from the IHDP RCT (Hill 2011),
    with semi-synthetic outcomes generated via the NPCI response surfaces.
    The key property is that BOTH factual and counterfactual outcomes are
    available, enabling computation of true Individual Treatment Effects.

    Args:
        realization: Which outcome realization (1-10). Each uses different
                    response surface parameters but same covariates.
        remove_confounding: If True, removes subjects to create observational
                          selection bias (standard IHDP benchmark setup where
                          treated group has non-random selection removed).

    Returns:
        Dictionary with:
            'X': Covariate matrix (n × 25)
            'T': Treatment vector (n,)
            'Y': Observed (factual) outcome (n,)
            'Y_cf': Counterfactual outcome (n,)
            'mu0': True control response surface (n,)
            'mu1': True treated response surface (n,)
            'tau': True ITE = mu1 - mu0 (n,)
            'ATE': True Average Treatment Effect (scalar)
            'n_treated': Number of treated subjects
            'n_control': Number of control subjects
            'covariate_names': List of covariate names
            'covariate_info': Metadata about each covariate
    """
    print(f"Loading IHDP benchmark (realization {realization})...")
    raw = _download_ihdp(realization)

    # Extract components
    T = raw["treatment"].values.astype(float)
    Y_factual = raw["y_factual"].values.astype(float)
    Y_cfactual = raw["y_cfactual"].values.astype(float)
    mu0 = raw["mu0"].values.astype(float)
    mu1 = raw["mu1"].values.astype(float)

    covariate_cols = [f"x{i}" for i in range(1, 26)]
    X = raw[covariate_cols].values.astype(float)

    if remove_confounding:
        # Standard IHDP benchmark: remove a subset of treated subjects
        # to create selection bias (observational data setup)
        # Following Hill (2011): remove treated children from
        # non-white mothers (creates confounding)
        keep_mask = np.ones(len(T), dtype=bool)
        treated_nonwhite = (T == 1) & (X[:, 19] == 0)  # x20 = race: white
        # Remove ~2/3 of treated non-white subjects
        rng = np.random.default_rng(42)
        remove_idx = np.where(treated_nonwhite)[0]
        n_remove = len(remove_idx) * 2 // 3
        remove_idx = rng.choice(remove_idx, size=n_remove, replace=False)
        keep_mask[remove_idx] = False

        T = T[keep_mask]
        Y_factual = Y_factual[keep_mask]
        Y_cfactual = Y_cfactual[keep_mask]
        mu0 = mu0[keep_mask]
        mu1 = mu1[keep_mask]
        X = X[keep_mask]

    # Compute true treatment effects
    tau = mu1 - mu0  # True Individual Treatment Effect
    ATE = tau.mean()  # True Average Treatment Effect

    # Observed outcome (what we actually see)
    Y_observed = Y_factual

    result = {
        "X": X,
        "T": T,
        "Y": Y_observed,
        "Y_cf": Y_cfactual,
        "mu0": mu0,
        "mu1": mu1,
        "tau": tau,
        "ATE": ATE,
        "n_subjects": len(T),
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "covariate_names": covariate_cols,
        "covariate_info": COVARIATE_INFO,
    }

    print(f"  Loaded {result['n_subjects']} subjects "
          f"({result['n_treated']} treated, {result['n_control']} control)")
    print(f"  True ATE: {ATE:.4f}")
    print(f"  True ITE range: [{tau.min():.4f}, {tau.max():.4f}]")

    return result


def load_ihdp_for_causal_forest(
    realization: int = 1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dict:
    """
    Load IHDP formatted for causal forest training/evaluation.

    Splits into train/test and returns in a format compatible with
    our MACF pipeline, enabling direct PEHE computation.

    Args:
        realization: IHDP realization (1-10)
        test_fraction: Fraction of data for test set
        seed: Random seed for splitting

    Returns:
        Dictionary with 'train' and 'test' splits, each containing
        (X, T, Y, tau) for PEHE evaluation.
    """
    data = load_ihdp(realization=realization)

    rng = np.random.default_rng(seed)
    n = data["n_subjects"]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_fraction)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    result = {
        "train": {
            "X": data["X"][train_idx],
            "T": data["T"][train_idx],
            "Y": data["Y"][train_idx],
            "tau": data["tau"][train_idx],
        },
        "test": {
            "X": data["X"][test_idx],
            "T": data["T"][test_idx],
            "Y": data["Y"][test_idx],
            "tau": data["tau"][test_idx],
        },
        "ATE": data["ATE"],
        "covariate_names": data["covariate_names"],
    }

    print(f"\n  Train: {len(train_idx)} subjects, Test: {len(test_idx)} subjects")
    return result


def load_multiple_realizations(
    realizations: range = range(1, 11),
) -> list:
    """
    Load multiple IHDP realizations for robust evaluation.

    The standard practice is to evaluate across all 10 realizations
    and report mean ± std of PEHE (Precision in Estimation of
    Heterogeneous Effects).

    Returns:
        List of data dictionaries, one per realization.
    """
    results = []
    for r in realizations:
        try:
            data = load_ihdp(realization=r)
            results.append(data)
        except Exception as e:
            print(f"  Warning: Failed to load realization {r}: {e}")
    return results


if __name__ == "__main__":
    # Quick test — downloads and validates real IHDP data
    print("=" * 60)
    print("IHDP Benchmark Loader — Real Data Test")
    print("=" * 60)

    data = load_ihdp(realization=1)

    print(f"\nCovariate matrix shape: {data['X'].shape}")
    print(f"Treatment vector shape: {data['T'].shape}")
    print(f"Outcome vector shape: {data['Y'].shape}")

    print(f"\nDataset statistics:")
    print(f"  Subjects: {data['n_subjects']}")
    print(f"  Treated: {data['n_treated']}")
    print(f"  Control: {data['n_control']}")
    print(f"  True ATE: {data['ATE']:.4f}")
    print(f"  True ITE std: {data['tau'].std():.4f}")

    print(f"\nContinuous covariates (x1-x6) summary:")
    for i in range(6):
        col = data["X"][:, i]
        name = f"x{i+1}"
        info = COVARIATE_INFO[name]["description"]
        print(f"  {name} ({info}): "
              f"mean={col.mean():.3f}, std={col.std():.3f}")

    print(f"\nBinary covariates (x7-x25) prevalence:")
    for i in range(6, 25):
        col = data["X"][:, i]
        name = f"x{i+1}"
        info = COVARIATE_INFO[name]["description"]
        print(f"  {name} ({info}): {col.mean():.1%}")

    # Test train/test split
    print("\n" + "=" * 60)
    splits = load_ihdp_for_causal_forest(realization=1)
    print(f"  Train PEHE baseline (always predict ATE): "
          f"{np.sqrt(((splits['train']['tau'] - splits['ATE'])**2).mean()):.4f}")
