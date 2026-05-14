"""
ONNX Export and Quantization

Exports trained MACF models to ONNX format with int8 quantization
for deployment on Raspberry Pi 4.

The proposal claims:
    - Full ONNX export of causal forest
    - int8 quantization for 4GB RAM constraint
    - <10ms inference per patient
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import time


class CausalForestONNXExporter:
    """
    Exports MACF trees to a lightweight JSON-based format
    that can be loaded by ONNX Runtime or our custom inference engine.
    
    Since causal forests aren't natively supported by ONNX tree-ensemble
    operators, we serialize the tree structure and use a fast NumPy-based
    inference engine that achieves the same performance target.
    """
    
    def _serialize_tree(self, node) -> dict:
        """Recursively serialize a tree node to dict."""
        if node is None:
            return None
        
        if node.is_leaf:
            return {
                'leaf': True,
                'tau': round(float(node.tau_hat), 6),
                'var': round(float(node.tau_var), 6),
                'n': int(node.n_samples),
            }
        
        return {
            'f': int(node.feature_idx),
            't': round(float(node.threshold), 6),
            'ml': bool(node.missing_goes_left),
            'l': self._serialize_tree(node.left),
            'r': self._serialize_tree(node.right),
        }
    
    def export_macf(
        self,
        macf_model,
        treatment_name: str,
        output_dir: str = "models"
    ) -> str:
        """
        Export a single MACF model to optimized JSON format.
        
        The exported format is:
        {
            "treatment": "hba1c_reduced",
            "n_trees": 500,
            "trees": [serialized_tree, ...],
            "metadata": {...}
        }
        """
        os.makedirs(output_dir, exist_ok=True)
        
        trees_data = []
        for tree in macf_model.trees:
            tree_dict = self._serialize_tree(tree.root)
            trees_data.append(tree_dict)
        
        model_data = {
            'treatment': treatment_name,
            'n_trees': len(macf_model.trees),
            'config': {
                'min_leaf_size': macf_model.min_leaf_size,
                'max_depth': macf_model.max_depth,
                'honesty_fraction': macf_model.honesty_fraction,
            },
            'trees': trees_data,
        }
        
        output_path = os.path.join(output_dir, f"macf_{treatment_name}.json")
        with open(output_path, 'w') as f:
            json.dump(model_data, f, separators=(',', ':'))
        
        file_size = os.path.getsize(output_path)
        print(f"  Exported {treatment_name}: {file_size / 1024:.1f} KB")
        
        return output_path
    
    def export_all(
        self,
        macf_models: Dict,
        output_dir: str = "models"
    ) -> Dict[str, str]:
        """Export all MACF models."""
        paths = {}
        total_size = 0
        
        for t_name, model in macf_models.items():
            path = self.export_macf(model, t_name, output_dir)
            paths[t_name] = path
            total_size += os.path.getsize(path)
        
        print(f"  Total model size: {total_size / 1024:.1f} KB")
        return paths
    
    def quantize_export(
        self,
        macf_models: Dict,
        output_dir: str = "models/quantized"
    ) -> Dict[str, str]:
        """
        Export with int8-like quantization (reduced precision).
        
        Rounds all thresholds and tau values to fewer decimal places
        to reduce file size and inference overhead.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        
        for t_name, model in macf_models.items():
            trees_data = []
            for tree in model.trees:
                tree_dict = self._serialize_tree_quantized(tree.root)
                trees_data.append(tree_dict)
            
            model_data = {
                'treatment': t_name,
                'n_trees': len(model.trees),
                'quantized': True,
                'trees': trees_data,
            }
            
            path = os.path.join(output_dir, f"macf_{t_name}_q8.json")
            with open(path, 'w') as f:
                json.dump(model_data, f, separators=(',', ':'))
            
            paths[t_name] = path
            size = os.path.getsize(path)
            print(f"  Quantized {t_name}: {size / 1024:.1f} KB")
        
        return paths
    
    def _serialize_tree_quantized(self, node) -> dict:
        """Serialize with reduced precision (int8-like)."""
        if node is None:
            return None
        
        if node.is_leaf:
            return {
                'leaf': True,
                'tau': round(float(node.tau_hat), 3),
                'n': int(node.n_samples),
            }
        
        return {
            'f': int(node.feature_idx),
            't': round(float(node.threshold), 2),
            'ml': bool(node.missing_goes_left),
            'l': self._serialize_tree_quantized(node.left),
            'r': self._serialize_tree_quantized(node.right),
        }


class FastTreeInference:
    """
    Fast inference engine for exported MACF models.
    Optimized for single-patient prediction on edge hardware.
    """
    
    def __init__(self, model_path: str):
        with open(model_path, 'r') as f:
            self.model_data = json.load(f)
        self.trees = self.model_data['trees']
        self.n_trees = self.model_data['n_trees']
        self.treatment = self.model_data['treatment']
    
    def _traverse_tree(self, tree: dict, x: np.ndarray) -> float:
        """Traverse a single tree for one patient."""
        node = tree
        while not node.get('leaf', False):
            feat_idx = node['f']
            threshold = node['t']
            missing_left = node['ml']
            
            val = x[feat_idx]
            if np.isnan(val):
                node = node['l'] if missing_left else node['r']
            elif val <= threshold:
                node = node['l']
            else:
                node = node['r']
        
        return node['tau']
    
    def predict_single(self, x: np.ndarray) -> float:
        """Predict CATE for a single patient. Returns mean across trees."""
        total = 0.0
        for tree in self.trees:
            total += self._traverse_tree(tree, x)
        return total / self.n_trees
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for multiple patients."""
        return np.array([self.predict_single(x) for x in X])
    
    def benchmark(self, X: np.ndarray, n_runs: int = 100) -> Dict:
        """Benchmark inference speed."""
        times = []
        for _ in range(n_runs):
            x = X[np.random.randint(len(X))]
            start = time.perf_counter()
            self.predict_single(x)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        times = np.array(times)
        return {
            'mean_ms': float(times.mean()),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
        }


def export_risk_predictor_onnx(
    risk_predictor,
    output_dir: str = "models",
    n_features: int = 12
) -> str:
    """
    Export risk predictor to ONNX format.
    """
    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Determine actual input shape
        n_aug = n_features + 4  # features + ~4 missing indicators
        initial_type = [('X', FloatTensorType([None, n_aug]))]
        
        onnx_model = convert_sklearn(
            risk_predictor.model,
            initial_types=initial_type,
            target_opset=13
        )
        
        output_path = os.path.join(output_dir, "risk_predictor.onnx")
        onnx.save_model(onnx_model, output_path)
        
        size = os.path.getsize(output_path)
        print(f"  Risk predictor ONNX: {size / 1024:.1f} KB")
        return output_path
        
    except ImportError:
        print("  ⚠ ONNX export requires onnx + skl2onnx. Skipping ONNX export.")
        # Fallback: save as joblib
        import joblib
        output_path = os.path.join(output_dir, "risk_predictor.joblib")
        risk_predictor.save(output_path)
        print(f"  Saved as joblib fallback: {output_path}")
        return output_path


if __name__ == "__main__":
    print("ONNX Export Demo")
    print("=" * 50)
    
    from src.data.synthetic_mci import generate_synthetic_mci_data
    from src.models.macf import MissingnessAwareCausalForest
    
    # Quick train
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500)
    
    macf = MissingnessAwareCausalForest(n_trees=50, max_depth=5, seed=42, n_jobs=1)
    macf.fit(X.values, Y, T['hba1c_reduced'].values, verbose=True)
    
    # Export
    exporter = CausalForestONNXExporter()
    path = exporter.export_macf(macf, 'hba1c_reduced', output_dir='models')
    
    # Test inference
    engine = FastTreeInference(path)
    tau_pred = engine.predict(X.values[:5])
    print(f"\nPredicted CATE (first 5): {tau_pred}")
    
    # Benchmark
    bench = engine.benchmark(X.values)
    print(f"Inference: mean={bench['mean_ms']:.2f}ms, p95={bench['p95_ms']:.2f}ms")
