"""
Full Training Pipeline

Orchestrates the complete training workflow:
    1. Generate/load data
    2. Fit DML nuisance models (5-fold cross-fitting)
    3. Train MACF for each of 4 treatments
    4. Train LightGBM risk predictor
    5. Evaluate on held-out test set + IHDP benchmark
    6. Export results
"""

import numpy as np
import pandas as pd
import time
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.data.synthetic_mci import (
    generate_train_val_test_split,
    get_treatment_names,
    get_feature_names,
)
from src.data.ihdp_loader import load_ihdp_data
from src.models.macf import MissingnessAwareCausalForest
from src.models.dml_nuisance import fit_all_nuisance_models
from src.models.risk_predictor import MCIRiskPredictor
from src.pipeline.evaluate import evaluate_cate, evaluate_risk_predictor
from src.pipeline.e_value import compute_e_values


def train_full_pipeline(
    n_samples: int = 6000,
    n_trees: int = 500,
    max_depth: int = 8,
    min_leaf_size: int = 20,
    n_folds: int = 5,
    seed: int = 42,
    output_dir: str = "models",
    verbose: bool = True,
) -> Dict:
    """
    Execute the complete training pipeline.
    
    Args:
        n_samples: Number of synthetic patients
        n_trees: Trees per causal forest (proposal: 2000, PoC: 500)
        max_depth: Max tree depth
        min_leaf_size: Minimum samples per leaf
        n_folds: Cross-fitting folds
        seed: Random seed
        output_dir: Directory for saved models
        verbose: Print progress
    
    Returns:
        Dictionary containing all trained models and evaluation results
    """
    results = {}
    total_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ================================================================
    # Step 1: Generate Data
    # ================================================================
    if verbose:
        print("=" * 60)
        print("STEP 1: Generating Synthetic MCI Data")
        print("=" * 60)
    
    splits = generate_train_val_test_split(n_samples=n_samples, seed=seed)
    
    X_train, T_train, Y_train, tau_train = splits['train']
    X_val, T_val, Y_val, tau_val = splits['val']
    X_test, T_test, Y_test, tau_test = splits['test']
    
    if verbose:
        print(f"  Train: {len(X_train)} samples (MCI rate: {Y_train.mean():.2%})")
        print(f"  Val:   {len(X_val)} samples (MCI rate: {Y_val.mean():.2%})")
        print(f"  Test:  {len(X_test)} samples (MCI rate: {Y_test.mean():.2%})")
    
    results['data'] = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'mci_rate_train': float(Y_train.mean()),
    }
    
    # ================================================================
    # Step 2: Fit DML Nuisance Models
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Fitting DML Nuisance Models (5-fold cross-fitting)")
        print("=" * 60)
    
    treatment_names = get_treatment_names()
    treatments_dict = {name: T_train[name].values for name in treatment_names}
    
    nuisance_start = time.time()
    nuisance_results = fit_all_nuisance_models(
        X_train.values, Y_train, treatments_dict,
        n_folds=n_folds, seed=seed, verbose=verbose
    )
    nuisance_time = time.time() - nuisance_start
    
    if verbose:
        print(f"\n  Nuisance fitting time: {nuisance_time:.1f}s")
    
    results['nuisance_time_s'] = nuisance_time
    
    # ================================================================
    # Step 3: Train MACF for Each Treatment
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Training MACF for Each Treatment")
        print("=" * 60)
    
    macf_models = {}
    macf_start = time.time()
    
    for t_name in treatment_names:
        if verbose:
            print(f"\n  Training MACF for: {t_name}")
        
        nuisance = nuisance_results[t_name]
        
        macf = MissingnessAwareCausalForest(
            n_trees=n_trees,
            min_leaf_size=min_leaf_size,
            max_depth=max_depth,
            honesty_fraction=0.5,
            subsample_fraction=0.5,
            seed=seed,
            n_jobs=1,
        )
        
        # Train on residualized outcomes
        macf.fit(
            X_train.values,
            nuisance['Y_residual'],
            treatments_dict[t_name],
            verbose=verbose
        )
        
        macf_models[t_name] = macf
    
    macf_time = time.time() - macf_start
    if verbose:
        print(f"\n  Total MACF training time: {macf_time:.1f}s")
    
    results['macf_time_s'] = macf_time
    results['macf_models'] = macf_models
    
    # ================================================================
    # Step 4: Train Risk Predictor
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Training LightGBM Risk Predictor")
        print("=" * 60)
    
    risk_start = time.time()
    risk_predictor = MCIRiskPredictor(seed=seed)
    risk_predictor.fit(
        X_train.values, Y_train,
        eval_X=X_val.values, eval_Y=Y_val,
        verbose=verbose
    )
    risk_time = time.time() - risk_start
    
    results['risk_predictor'] = risk_predictor
    results['risk_time_s'] = risk_time
    
    # ================================================================
    # Step 5: Evaluate on Test Set
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 5: Evaluation on Test Set")
        print("=" * 60)
    
    # CATE evaluation
    cate_results = {}
    for t_name in treatment_names:
        if verbose:
            print(f"\n  Evaluating CATE for: {t_name}")
        
        tau_hat, ci_lo, ci_hi = macf_models[t_name].predict(X_test.values)
        tau_true = tau_test[t_name]
        
        metrics = evaluate_cate(tau_hat, tau_true, ci_lo, ci_hi)
        cate_results[t_name] = metrics
        
        if verbose:
            print(f"    PEHE:     {metrics['pehe']:.4f} (target: <0.08)")
            print(f"    Coverage: {metrics['coverage']:.2%} (target: >90%)")
            print(f"    ATE err:  {metrics['ate_error']:.4f}")
    
    results['cate_results'] = cate_results
    
    # Risk predictor evaluation
    risk_metrics = evaluate_risk_predictor(
        risk_predictor, X_test.values, Y_test
    )
    results['risk_metrics'] = risk_metrics
    
    if verbose:
        print(f"\n  Risk Predictor AUROC: {risk_metrics['auroc']:.4f} (target: >0.78)")
    
    # ================================================================
    # Step 6: E-value Sensitivity Analysis
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 6: E-value Sensitivity Analysis")
        print("=" * 60)
    
    e_value_results = {}
    for t_name in treatment_names:
        tau_hat, _, _ = macf_models[t_name].predict(X_test.values)
        ate = tau_hat.mean()
        
        e_val = compute_e_values(ate, Y_test.mean())
        e_value_results[t_name] = e_val
        
        if verbose:
            print(f"  {t_name}: E-value = {e_val['e_value']:.2f} (target: >2.0)")
    
    results['e_values'] = e_value_results
    
    # ================================================================
    # Step 7: IHDP Benchmark Validation
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STEP 7: IHDP Benchmark Validation")
        print("=" * 60)
    
    ihdp_data = load_ihdp_data(seed=seed)
    
    macf_ihdp = MissingnessAwareCausalForest(
        n_trees=min(200, n_trees),
        min_leaf_size=10,
        max_depth=6,
        seed=seed,
        n_jobs=1,
    )
    macf_ihdp.fit(
        ihdp_data['train']['X'],
        ihdp_data['train']['Y'],
        ihdp_data['train']['T'],
        verbose=verbose
    )
    
    tau_hat_ihdp, ci_lo_ihdp, ci_hi_ihdp = macf_ihdp.predict(ihdp_data['test']['X'])
    ihdp_metrics = evaluate_cate(
        tau_hat_ihdp, ihdp_data['test']['tau'],
        ci_lo_ihdp, ci_hi_ihdp
    )
    
    results['ihdp_metrics'] = ihdp_metrics
    
    if verbose:
        print(f"  IHDP PEHE:     {ihdp_metrics['pehe']:.4f}")
        print(f"  IHDP Coverage: {ihdp_metrics['coverage']:.2%}")
    
    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - total_start
    results['total_time_s'] = total_time
    
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Models trained: 4 MACF + 1 Risk Predictor")
        
        # Pass/fail summary
        all_pass = True
        for t_name, cr in cate_results.items():
            if cr['pehe'] > 0.08:
                print(f"  ⚠ {t_name} PEHE {cr['pehe']:.4f} > 0.08")
                all_pass = False
        if risk_metrics['auroc'] < 0.78:
            print(f"  ⚠ Risk AUROC {risk_metrics['auroc']:.4f} < 0.78")
            all_pass = False
        
        if all_pass:
            print("  ✓ All targets met!")
    
    return results


if __name__ == "__main__":
    results = train_full_pipeline(
        n_samples=2000,   # Smaller for quick test
        n_trees=100,       # Fewer trees for speed
        verbose=True
    )
