"""
Full Pipeline Demo

One-command demonstration of the entire Causal Digital Twin pipeline:
    1. Generate synthetic MCI data (SANSCOG-like)
    2. Fit DML nuisance models
    3. Train MACF for 4 treatments
    4. Train LightGBM risk predictor
    5. Evaluate (PEHE, AUROC, E-values)
    6. Export models to JSON
    7. Run inference benchmark
    8. Launch dashboard

Usage:
    python demo.py              # Full pipeline
    python demo.py --quick      # Quick test (fewer trees/samples)
    python demo.py --dashboard  # Skip training, launch dashboard
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo(quick: bool = False, dashboard_only: bool = False):
    """Run the full pipeline demonstration."""
    
    print("\n" + "=" * 64)
    print("  🧠 MINDBRIDGE — Causal Digital Twin for MCI Prevention")
    print("  Counterfactual Therapy Simulator — Full Pipeline Demo")
    print("=" * 64)
    
    if dashboard_only:
        print("\n→ Launching dashboard only...")
        from src.dashboard.app import run_dashboard
        run_dashboard(host='127.0.0.1', port=5000, debug=True)
        return
    
    # Configuration
    if quick:
        n_samples = 1000
        n_trees = 50
        print("\n⚡ QUICK MODE: Reduced samples and trees for fast testing")
    else:
        n_samples = 3000
        n_trees = 200
    
    total_start = time.time()
    
    # ================================================================
    # STEP 1: Data Generation
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 1/7: Generating Synthetic MCI Data")
    print("─" * 64)
    
    from src.data.synthetic_mci import (
        generate_train_val_test_split,
        get_treatment_names,
        get_feature_names,
    )
    
    splits = generate_train_val_test_split(n_samples=n_samples, seed=42)
    X_train, T_train, Y_train, tau_train = splits['train']
    X_val, T_val, Y_val, tau_val = splits['val']
    X_test, T_test, Y_test, tau_test = splits['test']
    
    print(f"  ✓ Train: {len(X_train)} patients | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  ✓ MCI prevalence: {Y_train.mean():.1%}")
    print(f"  ✓ Missing data: OCT={X_train['oct_rnfl_thickness'].isna().mean():.0%}, "
          f"Gait={X_train['gait_speed'].isna().mean():.0%}")
    
    # ================================================================
    # STEP 2: DML Nuisance Estimation
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 2/7: Fitting DML Nuisance Models (Cross-Fitted)")
    print("─" * 64)
    
    from src.models.dml_nuisance import fit_all_nuisance_models
    
    treatment_names = get_treatment_names()
    treatments_dict = {name: T_train[name].values for name in treatment_names}
    
    nuisance_start = time.time()
    nuisance_results = fit_all_nuisance_models(
        X_train.values, Y_train, treatments_dict,
        n_folds=3 if quick else 5, seed=42, verbose=True
    )
    print(f"\n  ✓ Nuisance fitting complete in {time.time()-nuisance_start:.1f}s")
    
    # ================================================================
    # STEP 3: Train MACF
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 3/7: Training Missingness-Aware Causal Forest (MACF)")
    print("─" * 64)
    
    from src.models.macf import MissingnessAwareCausalForest
    
    macf_models = {}
    macf_start = time.time()
    
    for t_name in treatment_names:
        print(f"\n  Training MACF for: {t_name}")
        nuisance = nuisance_results[t_name]
        
        macf = MissingnessAwareCausalForest(
            n_trees=n_trees,
            min_leaf_size=15,
            max_depth=6 if quick else 8,
            honesty_fraction=0.5,
            subsample_fraction=0.5,
            seed=42,
            n_jobs=1,
        )
        
        macf.fit(
            X_train.values,
            nuisance['Y_residual'],
            treatments_dict[t_name],
            verbose=True
        )
        macf_models[t_name] = macf
    
    print(f"\n  ✓ All 4 MACF models trained in {time.time()-macf_start:.1f}s")
    
    # ================================================================
    # STEP 4: Train Risk Predictor
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 4/7: Training LightGBM Risk Predictor")
    print("─" * 64)
    
    from src.models.risk_predictor import MCIRiskPredictor
    
    risk_predictor = MCIRiskPredictor(seed=42)
    risk_predictor.fit(
        X_train.values, Y_train,
        eval_X=X_val.values, eval_Y=Y_val,
        verbose=True
    )
    
    # ================================================================
    # STEP 5: Evaluate
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 5/7: Evaluation — PEHE, AUROC, Coverage")
    print("─" * 64)
    
    from src.pipeline.evaluate import evaluate_cate, evaluate_risk_predictor
    
    print("\n  CATE Evaluation (Test Set):")
    print(f"  {'Treatment':<25} {'PEHE':>8} {'Coverage':>10} {'ATE Err':>10}")
    print("  " + "─" * 55)
    
    all_pass = True
    for t_name in treatment_names:
        tau_hat, ci_lo, ci_hi = macf_models[t_name].predict(X_test.values)
        tau_true = tau_test[t_name]
        metrics = evaluate_cate(tau_hat, tau_true, ci_lo, ci_hi)
        
        status = "✓" if metrics['pehe'] < 0.08 else "⚠"
        if metrics['pehe'] >= 0.08:
            all_pass = False
        
        print(f"  {status} {t_name:<23} {metrics['pehe']:>8.4f} "
              f"{metrics['coverage']:>9.1%} {metrics['ate_error']:>10.4f}")
    
    risk_metrics = evaluate_risk_predictor(risk_predictor, X_test.values, Y_test)
    print(f"\n  Risk Predictor AUROC: {risk_metrics['auroc']:.4f} (target: >0.78)")
    
    # ================================================================
    # STEP 6: E-value Sensitivity Analysis
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 6/7: E-value Sensitivity Analysis")
    print("─" * 64)
    
    from src.pipeline.e_value import compute_e_values
    
    baseline_risk = Y_test.mean()
    for t_name in treatment_names:
        tau_hat, _, _ = macf_models[t_name].predict(X_test.values)
        ate = tau_hat.mean()
        e_result = compute_e_values(ate, baseline_risk)
        
        status = "✓" if e_result['e_value'] >= 2.0 else "⚠"
        print(f"  {status} {t_name:<23} E-value={e_result['e_value']:.2f}")
    
    # ================================================================
    # STEP 7: Export Models
    # ================================================================
    print("\n" + "─" * 64)
    print("  STEP 7/7: Exporting Models for Edge Deployment")
    print("─" * 64)
    
    from src.deployment.model_export import (
        export_macf_to_onnx, CausalForestExporter, FastTreeInference,
        load_inference_engine,
    )
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/quantized", exist_ok=True)
    
    # Try ONNX export first, fall back to JSON
    try:
        print("\n  ONNX export (TreeEnsembleRegressor):")
        for t_name, model in macf_models.items():
            export_macf_to_onnx(model, t_name, output_dir="models",
                               n_features=X_train.shape[1])
    except Exception as e:
        print(f"  ⚠ ONNX export failed ({e}). Using JSON fallback.")
    
    # Always export JSON as fallback
    exporter = CausalForestExporter()
    
    print("\n  JSON export (full precision):")
    paths = exporter.export_all(macf_models, output_dir="models")
    
    print("\n  JSON export (int8-quantized):")
    q_paths = exporter.export_all(macf_models, output_dir="models/quantized")
    
    # Save risk predictor
    risk_predictor.save("models/risk_predictor.joblib")
    print(f"  Risk predictor saved: models/risk_predictor.joblib")
    
    # Benchmark inference
    print("\n  Inference Benchmark:")
    for t_name, path in q_paths.items():
        engine = FastTreeInference(path)
        bench = engine.benchmark(X_test.values, n_runs=50)
        print(f"    {t_name}: mean={bench['mean_ms']:.2f}ms, p95={bench['p95_ms']:.2f}ms")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    total_time = time.time() - total_start
    
    print("\n" + "=" * 64)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 64)
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Models:        4 MACF + 1 Risk Predictor")
    print(f"  Exported to:   models/ and models/quantized/")
    print(f"")
    print(f"  To launch the clinician dashboard:")
    print(f"    python demo.py --dashboard")
    print(f"    Then open http://127.0.0.1:5000")
    print("=" * 64 + "\n")
    
    return {
        'macf_models': macf_models,
        'risk_predictor': risk_predictor,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Causal Digital Twin — Demo Pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer samples/trees')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard only')
    args = parser.parse_args()
    
    result = run_demo(quick=args.quick, dashboard_only=args.dashboard)
    
    # If training completed, offer to launch dashboard
    if result and not args.dashboard:
        print("Would you like to launch the dashboard? Run:")
        print("  python demo.py --dashboard\n")


if __name__ == '__main__':
    main()
