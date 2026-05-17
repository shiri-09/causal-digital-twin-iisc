"""
MindBridge Inference Engine

Loads trained models and provides real-time counterfactual predictions
for individual patients. Designed for <10ms per prediction.
"""

import numpy as np
import json
import os
import time
from typing import Dict, Optional, List
from pathlib import Path


# Feature order matching synthetic_mci.py
FEATURE_ORDER = [
    'age', 'sex', 'education_years', 'hba1c', 'systolic_bp',
    'diastolic_bp', 'ldl', 'physical_activity_min_week',
    'gait_speed', 'oct_rnfl_thickness', 'mmse_score', 'hrv_sdnn'
]

TREATMENT_INFO = {
    'hba1c_reduced': {
        'display_name': 'Lower HbA1c by 1%',
        'icon': '🩸',
        'category': 'Metabolic',
        'monitoring': 'Standard blood glucose test',
    },
    'bp_managed': {
        'display_name': 'Manage BP to <130/80',
        'icon': '💊',
        'category': 'Cardiovascular',
        'monitoring': '₹800 digital BP cuff',
    },
    'activity_increased': {
        'display_name': 'Increase activity to 300 min/wk',
        'icon': '🏃',
        'category': 'Lifestyle',
        'monitoring': 'Smartphone pose estimation for gait',
    },
    'ldl_reduced': {
        'display_name': 'Reduce LDL by 20 mg/dL',
        'icon': '🫀',
        'category': 'Cardiovascular',
        'monitoring': 'Standard lipid panel',
    },
}


class InferenceEngine:
    """
    Loads models and provides fast patient-level inference.
    
    Supports two modes:
        1. Live models (MACF objects in memory) — for development
        2. Exported models (JSON files) — for edge deployment
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        live_models: Optional[Dict] = None,
        live_risk_predictor=None,
    ):
        self.model_dir = model_dir
        self.macf_models = {}
        self.risk_predictor = live_risk_predictor
        self._use_live = live_models is not None
        
        if live_models:
            self.macf_models = live_models
        else:
            self._load_exported_models()
    
    def _load_exported_models(self):
        """Load exported JSON models from disk."""
        from src.deployment.model_export import FastTreeInference, load_inference_engine
        
        model_dir = Path(self.model_dir)
        
        # Try ONNX first, then quantized JSON, then regular JSON
        for t_name in TREATMENT_INFO.keys():
            engine = load_inference_engine(str(model_dir), t_name)
            if engine is not None:
                self.macf_models[t_name] = engine
        
        # If no models found, auto-train a quick demo MACF
        if not self.macf_models:
            self._auto_train_demo()
    
    def _auto_train_demo(self):
        """Auto-train a small MACF on synthetic data for live demo."""
        try:
            from src.data.synthetic_mci import generate_synthetic_mci_data
            from src.models.macf import MissingnessAwareCausalForest
            from src.deployment.model_export import CausalForestExporter, FastTreeInference
            
            print("  ⚡ No trained models found. Auto-training demo MACF...")
            X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=300)
            
            for t_name in TREATMENT_INFO.keys():
                if t_name in T.columns:
                    macf = MissingnessAwareCausalForest(
                        n_trees=30, max_depth=4, seed=42, n_jobs=1
                    )
                    macf.fit(X.values, Y, T[t_name].values, verbose=False)
                    
                    # Export to JSON and load inference engine
                    os.makedirs("models", exist_ok=True)
                    exporter = CausalForestExporter()
                    path = exporter.export_macf(macf, t_name, "models")
                    self.macf_models[t_name] = FastTreeInference(path)
            
            self._use_live = False
            print(f"  ✓ Demo models trained: {list(self.macf_models.keys())}")
        except Exception as e:
            print(f"  ⚠ Auto-train failed: {e}. Using feature-based estimation.")
    
    def _estimate_tau_from_features(self, x: np.ndarray, t_name: str) -> float:
        """
        Estimate treatment effect from patient features when no model
        is available. Uses domain knowledge from proposal literature.
        """
        age = x[0] if not np.isnan(x[0]) else 62
        hba1c = x[3] if not np.isnan(x[3]) else 6.2
        sbp = x[4] if not np.isnan(x[4]) else 138
        activity = x[7] if not np.isnan(x[7]) else 120
        
        # Higher baseline → larger treatment effect (domain knowledge)
        if t_name == 'hba1c_reduced':
            return -0.02 * max(0, hba1c - 5.7)  # effect scales with HbA1c
        elif t_name == 'bp_managed':
            return -0.001 * max(0, sbp - 120)  # effect scales with BP
        elif t_name == 'activity_increased':
            return -0.0002 * max(0, 300 - activity)  # effect scales with inactivity
        elif t_name == 'ldl_reduced':
            return -0.03 - 0.001 * max(0, age - 60)  # slight age interaction
        return -0.04
    
    def _patient_to_array(self, patient_data: Dict) -> np.ndarray:
        """Convert patient dict to feature array."""
        x = np.zeros(len(FEATURE_ORDER))
        for i, feat in enumerate(FEATURE_ORDER):
            val = patient_data.get(feat)
            if val is None or val == '' or val == 'null':
                x[i] = np.nan
            else:
                x[i] = float(val)
        return x
    
    def _compute_baseline_risk(self, x: np.ndarray) -> float:
        """
        Compute baseline MCI risk.
        
        If risk predictor is loaded, use it.
        Otherwise, use a simple logistic model as fallback.
        """
        if self.risk_predictor is not None:
            try:
                return float(self.risk_predictor.predict_proba(x.reshape(1, -1))[0])
            except Exception:
                pass
        
        # Fallback: simplified logistic risk model
        age = x[0] if not np.isnan(x[0]) else 62
        edu = x[2] if not np.isnan(x[2]) else 7.5
        hba1c = x[3] if not np.isnan(x[3]) else 6.2
        sbp = x[4] if not np.isnan(x[4]) else 138
        gait = x[8] if not np.isnan(x[8]) else 0.95
        mmse = x[10] if not np.isnan(x[10]) else 26.5
        
        logit = (
            -2.5
            + 0.05 * (age - 62)
            - 0.03 * edu
            + 0.15 * (hba1c - 6.0)
            + 0.008 * (sbp - 130)
            - 0.4 * (gait - 0.95)
            - 0.05 * (mmse - 26)
        )
        
        return float(1 / (1 + np.exp(-logit)))
    
    def predict_patient(self, patient_data: Dict) -> Dict:
        """
        Full patient workup: baseline risk + 4 counterfactual interventions.
        
        This is the core function shown to clinicians.
        """
        start = time.perf_counter()
        
        x = self._patient_to_array(patient_data)
        
        # Baseline risk
        baseline_risk = self._compute_baseline_risk(x)
        
        # Counterfactual interventions
        interventions = []
        combined_reduction = 0.0
        
        for t_name, info in TREATMENT_INFO.items():
            if t_name in self.macf_models:
                model = self.macf_models[t_name]
                
                if self._use_live:
                    result = model.predict_single(x.reshape(1, -1))
                    if isinstance(result, dict):
                        tau = result['tau_hat']
                    else:
                        tau_arr, _, _ = model.predict(x.reshape(1, -1))
                        tau = float(tau_arr[0])
                else:
                    tau = float(model.predict_single(x))
            else:
                # No model available — estimate from patient features
                tau = self._estimate_tau_from_features(x, t_name)
            
            new_risk = max(0.01, min(0.99, baseline_risk + tau))
            risk_reduction = baseline_risk - new_risk
            combined_reduction += risk_reduction * 0.7  # overlap discount
            
            # Simple E-value approximation
            if risk_reduction > 0.01:
                rr = new_risk / baseline_risk if baseline_risk > 0 else 1.0
                if rr < 1:
                    rr = 1 / rr
                e_val = rr + np.sqrt(rr * max(0, rr - 1))
            else:
                e_val = 1.0
            
            confidence = 'high' if e_val >= 3.0 else ('moderate' if e_val >= 2.0 else 'low')
            
            interventions.append({
                'name': info['display_name'],
                'treatment': t_name,
                'icon': info['icon'],
                'category': info['category'],
                'monitoring': info['monitoring'],
                'tau': round(float(tau), 4),
                'new_risk': round(float(new_risk), 4),
                'new_risk_pct': round(float(new_risk * 100), 1),
                'risk_reduction': round(float(risk_reduction), 4),
                'risk_reduction_pct': round(float(risk_reduction * 100), 1),
                'risk_reduction_pp': round(float(risk_reduction * 100), 1),
                'e_value': round(float(e_val), 2),
                'confidence': confidence,
            })
        
        # Sort by risk reduction (largest first)
        interventions.sort(key=lambda x: x['risk_reduction'], reverse=True)
        
        # Combined intervention estimate
        combined_risk = max(0.01, baseline_risk - combined_reduction)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return {
            'baseline_risk': round(float(baseline_risk), 4),
            'baseline_risk_pct': round(float(baseline_risk * 100), 1),
            'interventions': interventions,
            'combined_risk': round(float(combined_risk), 4),
            'combined_risk_pct': round(float(combined_risk * 100), 1),
            'combined_reduction_pct': round(float((baseline_risk - combined_risk) * 100), 1),
            'inference_time_ms': round(elapsed, 2),
            'patient_summary': {
                'age': patient_data.get('age', 'N/A'),
                'sex': 'Female' if patient_data.get('sex', 0) == 1 else 'Male',
                'missing_features': [
                    feat for feat in FEATURE_ORDER
                    if patient_data.get(feat) is None
                ],
            }
        }
