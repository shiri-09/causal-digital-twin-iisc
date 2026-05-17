"""
IHDP Benchmark Data Loader

Loads the REAL Infant Health and Development Program (IHDP) semi-synthetic dataset
from the CEVAE repository (Louizos et al. 2017), the standard benchmark for
causal ML (used by Shalit et al. 2017, Wager & Athey 2018).

Dataset structure (per Hill 2011 simulation design):
    - 747 subjects, 25 covariates (6 continuous + 19 binary)
    - Binary treatment assignment
    - y_factual: observed outcome
    - y_cfactual: counterfactual outcome (for PEHE computation)
    - mu0, mu1: potential outcome means (ground truth)
    - True ITE: tau = mu1 - mu0

Source: https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

# CEVAE repository: 1000 replications of IHDP (ihdp_npci_1.csv through ihdp_npci_1000.csv)
IHDP_BASE_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv"

# Column layout: treatment, y_factual, y_cfactual, mu0, mu1, x1...x25
IHDP_COLUMNS = (
    ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
    + [f"x{i}" for i in range(1, 26)]
)

# Local cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


def _download_ihdp_replication(
    replication: int = 1,
    cache: bool = True
) -> pd.DataFrame:
    """
    Download a single IHDP replication from the CEVAE repository.
    
    Args:
        replication: Replication number (1-1000). Default=1 is the most
                     commonly used in the literature.
        cache: Whether to cache the downloaded CSV locally.
    
    Returns:
        DataFrame with 747 rows and 30 columns.
    """
    assert 1 <= replication <= 1000, "IHDP has 1000 replications (1-1000)"
    
    # Check local cache first
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"ihdp_npci_{replication}.csv"
        if cache_path.exists():
            logger.info(f"Loading IHDP replication {replication} from cache: {cache_path}")
            df = pd.read_csv(cache_path, header=None)
            df.columns = IHDP_COLUMNS
            return df
    
    # Download from CEVAE repository
    url = f"{IHDP_BASE_URL}/ihdp_npci_{replication}.csv"
    logger.info(f"Downloading IHDP replication {replication} from {url}")
    
    try:
        df = pd.read_csv(url, header=None)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download IHDP data from {url}. "
            f"Check your internet connection. Error: {e}"
        )
    
    df.columns = IHDP_COLUMNS
    
    # Validate shape
    assert df.shape == (747, 30), (
        f"Unexpected IHDP shape: {df.shape}. Expected (747, 30). "
        f"The data source may have changed."
    )
    
    # Cache locally
    if cache:
        df.to_csv(cache_path, index=False, header=False)
        logger.info(f"Cached IHDP data to {cache_path}")
    
    return df


def load_ihdp_data(
    replication: int = 1,
    test_fraction: float = 0.2,
    seed: int = 42,
    cache: bool = True
) -> Dict:
    """
    Load REAL IHDP benchmark data with train/test split.
    
    This loads the actual IHDP semi-synthetic dataset from the CEVAE
    repository. Covariates are from the real IHDP study; outcomes are
    simulated following Hill (2011) Setting A to create known ground-truth
    treatment effects for PEHE evaluation.
    
    Args:
        replication: Which of the 1000 IHDP replications to use (1-1000).
                     Replication 1 is the standard benchmark.
        test_fraction: Fraction held out for testing (default 0.2).
        seed: Random seed for train/test split.
        cache: Whether to cache downloaded data locally.
    
    Returns:
        Dictionary with 'train' and 'test' keys, each containing:
        - X: covariates (n × 25)
        - T: treatment assignment (n,) — binary
        - Y: observed (factual) outcome (n,)
        - Y_cf: counterfactual outcome (n,) — for PEHE
        - mu0: E[Y(0)|X] potential outcome mean under control (n,)
        - mu1: E[Y(1)|X] potential outcome mean under treatment (n,)
        - tau: true individual treatment effect = mu1 - mu0 (n,)
    """
    df = _download_ihdp_replication(replication, cache=cache)
    
    # Extract components
    T = df["treatment"].values.astype(int)
    Y_factual = df["y_factual"].values
    Y_cfactual = df["y_cfactual"].values
    mu0 = df["mu0"].values
    mu1 = df["mu1"].values
    X = df[[f"x{i}" for i in range(1, 26)]].values
    
    # True individual treatment effect (ground truth for PEHE)
    tau = mu1 - mu0
    
    # Train/test split
    n = len(X)
    n_test = int(n * test_fraction)
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    def _subset(idx):
        return {
            "X": X[idx],
            "T": T[idx],
            "Y": Y_factual[idx],
            "Y_cf": Y_cfactual[idx],
            "mu0": mu0[idx],
            "mu1": mu1[idx],
            "tau": tau[idx],
        }
    
    result = {
        "train": _subset(train_idx),
        "test": _subset(test_idx),
        "metadata": {
            "replication": replication,
            "n_total": n,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": 25,
            "n_continuous": 6,
            "n_binary": 19,
            "source": f"{IHDP_BASE_URL}/ihdp_npci_{replication}.csv",
            "reference": "Hill (2011) Setting A; Louizos et al. (2017) CEVAE",
        }
    }
    
    logger.info(
        f"IHDP loaded: {len(train_idx)} train, {len(test_idx)} test, "
        f"ATE = {tau.mean():.4f}, CATE std = {tau.std():.4f}"
    )
    
    return result


def load_multiple_ihdp_replications(
    replications: list = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    cache: bool = True
) -> list:
    """
    Load multiple IHDP replications for robust PEHE estimation.
    
    Standard practice is to average PEHE across multiple replications
    (typically 100 or 1000) to get stable benchmark numbers.
    
    Args:
        replications: List of replication numbers. Default = [1..10].
        test_fraction: Fraction held out for testing.
        seed: Random seed.
        cache: Whether to cache.
    
    Returns:
        List of data dictionaries (one per replication).
    """
    if replications is None:
        replications = list(range(1, 11))  # Default: first 10
    
    results = []
    for rep in replications:
        try:
            data = load_ihdp_data(
                replication=rep,
                test_fraction=test_fraction,
                seed=seed,
                cache=cache
            )
            results.append(data)
            logger.info(f"Loaded replication {rep}/{ replications[-1]}")
        except Exception as e:
            logger.warning(f"Failed to load replication {rep}: {e}")
    
    return results


if __name__ == "__main__":
    print("Loading REAL IHDP benchmark data (replication 1)...")
    data = load_ihdp_data(replication=1)
    
    train = data["train"]
    test = data["test"]
    meta = data["metadata"]
    
    print(f"\nSource: {meta['source']}")
    print(f"Reference: {meta['reference']}")
    print(f"\nTrain: {train['X'].shape[0]} samples, {train['X'].shape[1]} features")
    print(f"Test:  {test['X'].shape[0]} samples")
    print(f"Treatment rate (train): {train['T'].mean():.2%}")
    print(f"True ATE (train): {train['tau'].mean():.4f}")
    print(f"True CATE std (train): {train['tau'].std():.4f}")
    
    # Quick PEHE sanity check (naive estimator)
    naive_tau = np.full_like(test["tau"], test["tau"].mean())
    pehe_naive = np.sqrt(np.mean((naive_tau - test["tau"]) ** 2))
    print(f"\nNaive PEHE (constant ATE): {pehe_naive:.4f}")
    print(f"(MACF should beat this significantly)")
