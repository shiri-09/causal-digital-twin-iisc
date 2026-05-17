"""
Model Export — ONNX and JSON Serialization

Exports trained MACF models for deployment on Raspberry Pi 4.

Two export paths:
    1. ONNX (preferred): Uses TreeEnsembleRegressor operator with
       nodes_missing_value_tracks_true for native NaN handling.
       ONNX Runtime provides C++ inference at <1ms per patient.
    
    2. JSON (fallback): Compact JSON serialization with Python-based
       FastTreeInference engine. Used when ONNX is unavailable.

Key insight:
    At INFERENCE time, MACF is a standard tree ensemble — the custom
    missingness-aware splitting criterion is only used during TRAINING.
    Therefore, standard ONNX TreeEnsembleRegressor operators with
    nodes_missing_value_tracks_true map 1:1 to MACF's inference logic.

References:
    - ONNX TreeEnsembleRegressor spec: https://onnx.ai/onnx/operators/onnx_ml_TreeEnsembleRegressor.html
    - nodes_missing_value_tracks_true: per-node flag for NaN routing
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX Export (preferred path)
# ---------------------------------------------------------------------------

def _flatten_tree_for_onnx(node, node_id=0, nodes=None):
    """
    Flatten a recursive TreeNode into parallel arrays for ONNX.
    
    ONNX TreeEnsembleRegressor requires flat arrays:
        nodes_featureids, nodes_values, nodes_modes,
        nodes_truenodeids, nodes_falsenodeids,
        nodes_missing_value_tracks_true
    
    Returns:
        (next_free_id, nodes_list) where each node entry is a dict.
    """
    if nodes is None:
        nodes = {}
    
    if node is None or node.is_leaf:
        nodes[node_id] = {
            'feature_id': 0,
            'value': 0.0,
            'mode': 'LEAF',
            'true_child': 0,
            'false_child': 0,
            'missing_tracks_true': 0,
            'is_leaf': True,
            'tau': float(node.tau_hat) if node else 0.0,
            'n_samples': int(node.n_samples) if node else 0,
        }
        return node_id + 1, nodes
    
    # Internal node
    left_id = node_id + 1
    next_free, nodes = _flatten_tree_for_onnx(node.left, left_id, nodes)
    
    right_id = next_free
    next_free, nodes = _flatten_tree_for_onnx(node.right, right_id, nodes)
    
    nodes[node_id] = {
        'feature_id': int(node.feature_idx),
        'value': float(node.threshold),
        'mode': 'BRANCH_LEQ',  # val <= threshold → true (left) child
        'true_child': left_id,
        'false_child': right_id,
        'missing_tracks_true': 1 if node.missing_goes_left else 0,
        'is_leaf': False,
        'tau': 0.0,
        'n_samples': 0,
    }
    
    return next_free, nodes


def export_macf_to_onnx(
    macf_model,
    treatment_name: str,
    output_dir: str = "models",
    n_features: int = 12,
    opset_version: int = 18,
) -> str:
    """
    Export a MACF model to ONNX format using TreeEnsembleRegressor.
    
    The ONNX TreeEnsembleRegressor operator natively supports:
    - nodes_missing_value_tracks_true: maps to MACF's missing_goes_left
    - BRANCH_LEQ mode: maps to MACF's threshold comparison
    - Multiple trees: maps to MACF's ensemble averaging
    
    Args:
        macf_model: Trained MissingnessAwareCausalForest
        treatment_name: Name of the treatment (e.g., 'hba1c_reduced')
        output_dir: Output directory for .onnx file
        n_features: Number of input features
        opset_version: ONNX opset version
    
    Returns:
        Path to the exported .onnx file
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        logger.warning("onnx package not installed. Falling back to JSON export.")
        exporter = CausalForestExporter()
        return exporter.export_macf(macf_model, treatment_name, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten all trees into ONNX parallel arrays
    all_nodes_featureids = []
    all_nodes_values = []
    all_nodes_hitrates = []
    all_nodes_modes = []
    all_nodes_treeids = []
    all_nodes_nodeids = []
    all_nodes_truenodeids = []
    all_nodes_falsenodeids = []
    all_nodes_missing_tracks_true = []
    
    all_target_ids = []
    all_target_nodeids = []
    all_target_treeids = []
    all_target_weights = []
    
    global_offset = 0
    
    for tree_id, tree in enumerate(macf_model.trees):
        _, flat_nodes = _flatten_tree_for_onnx(tree.root, node_id=0)
        
        n_nodes = len(flat_nodes)
        
        for local_id in range(n_nodes):
            node = flat_nodes[local_id]
            
            all_nodes_treeids.append(tree_id)
            all_nodes_nodeids.append(local_id)
            all_nodes_featureids.append(node['feature_id'])
            all_nodes_values.append(node['value'])
            all_nodes_hitrates.append(1.0)
            all_nodes_modes.append(node['mode'])
            all_nodes_missing_tracks_true.append(node['missing_tracks_true'])
            
            if node['is_leaf']:
                all_nodes_truenodeids.append(0)
                all_nodes_falsenodeids.append(0)
                
                # Register leaf value (tau / n_trees for averaging)
                all_target_ids.append(0)  # single output
                all_target_nodeids.append(local_id)
                all_target_treeids.append(tree_id)
                all_target_weights.append(
                    node['tau'] / len(macf_model.trees)
                )
            else:
                all_nodes_truenodeids.append(node['true_child'])
                all_nodes_falsenodeids.append(node['false_child'])
    
    # Build ONNX TreeEnsembleRegressor node
    tree_node = helper.make_node(
        'TreeEnsembleRegressor',
        inputs=['X'],
        outputs=['tau_hat'],
        domain='ai.onnx.ml',
        name=f'macf_{treatment_name}',
        n_targets=1,
        aggregate_function='SUM',
        post_transform='NONE',
        nodes_featureids=all_nodes_featureids,
        nodes_values=all_nodes_values,
        nodes_hitrates=all_nodes_hitrates,
        nodes_modes=all_nodes_modes,
        nodes_treeids=all_nodes_treeids,
        nodes_nodeids=all_nodes_nodeids,
        nodes_truenodeids=all_nodes_truenodeids,
        nodes_falsenodeids=all_nodes_falsenodeids,
        nodes_missing_value_tracks_true=all_nodes_missing_tracks_true,
        target_ids=all_target_ids,
        target_nodeids=all_target_nodeids,
        target_treeids=all_target_treeids,
        target_weights=all_target_weights,
    )
    
    # Input/output specs
    X_input = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, n_features])
    tau_output = helper.make_tensor_value_info('tau_hat', TensorProto.FLOAT, [None, 1])
    
    # Build graph
    graph = helper.make_graph(
        [tree_node],
        f'macf_{treatment_name}',
        [X_input],
        [tau_output],
    )
    
    # Build model
    ml_opset = helper.make_opsetid('ai.onnx.ml', 3)
    default_opset = helper.make_opsetid('', opset_version)
    
    model = helper.make_model(
        graph,
        opset_imports=[default_opset, ml_opset],
        producer_name='MindBridge-MACF',
        producer_version='1.0',
        doc_string=f'Missingness-Aware Causal Forest for {treatment_name}',
    )
    
    # Validate
    onnx.checker.check_model(model)
    
    # Save
    output_path = os.path.join(output_dir, f"macf_{treatment_name}.onnx")
    onnx.save_model(model, output_path)
    
    file_size = os.path.getsize(output_path)
    n_trees = len(macf_model.trees)
    logger.info(
        f"Exported {treatment_name} to ONNX: {file_size / 1024:.1f} KB "
        f"({n_trees} trees, {len(all_nodes_nodeids)} total nodes)"
    )
    print(f"  ✓ ONNX export {treatment_name}: {file_size / 1024:.1f} KB ({n_trees} trees)")
    
    return output_path


def export_all_macf_to_onnx(
    macf_models: Dict,
    output_dir: str = "models",
    n_features: int = 12,
) -> Dict[str, str]:
    """Export all MACF models to ONNX."""
    paths = {}
    for t_name, model in macf_models.items():
        paths[t_name] = export_macf_to_onnx(
            model, t_name, output_dir, n_features
        )
    return paths


# ---------------------------------------------------------------------------
# JSON Export (fallback — no ONNX dependency)
# ---------------------------------------------------------------------------

class CausalForestExporter:
    """
    Exports MACF trees to a lightweight JSON-based format.
    Used as fallback when ONNX Runtime is unavailable.
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
        """Export a single MACF model to optimized JSON format."""
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
        print(f"  ✓ JSON export {treatment_name}: {file_size / 1024:.1f} KB")
        
        return output_path
    
    def export_all(self, macf_models: Dict, output_dir: str = "models") -> Dict[str, str]:
        """Export all MACF models to JSON."""
        paths = {}
        for t_name, model in macf_models.items():
            paths[t_name] = self.export_macf(model, t_name, output_dir)
        return paths


# Backward compatibility alias
CausalForestONNXExporter = CausalForestExporter


# ---------------------------------------------------------------------------
# Inference Engines
# ---------------------------------------------------------------------------

class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for deployed MACF models.
    Uses C++ runtime for maximum performance on ARM hardware.
    """
    
    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )
        
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_path = model_path
        
        # Extract treatment name from filename
        stem = Path(model_path).stem
        self.treatment = stem.replace('macf_', '')
    
    def predict_single(self, x: np.ndarray) -> float:
        """Predict CATE for a single patient."""
        x_input = x.reshape(1, -1).astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: x_input})
        return float(result[0][0, 0])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for multiple patients."""
        X_input = X.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: X_input})
        return result[0].flatten()
    
    def benchmark(self, X: np.ndarray, n_runs: int = 100) -> Dict:
        """Benchmark inference speed."""
        times = []
        for _ in range(n_runs):
            x = X[np.random.randint(len(X))]
            start = time.perf_counter()
            self.predict_single(x)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        times_arr = np.array(times)
        return {
            'mean_ms': float(times_arr.mean()),
            'p95_ms': float(np.percentile(times_arr, 95)),
            'p99_ms': float(np.percentile(times_arr, 99)),
            'engine': 'onnxruntime',
        }


class FastTreeInference:
    """
    Fast JSON-based inference engine for exported MACF models.
    Pure Python fallback when ONNX Runtime is unavailable.
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
        
        times_arr = np.array(times)
        return {
            'mean_ms': float(times_arr.mean()),
            'p95_ms': float(np.percentile(times_arr, 95)),
            'p99_ms': float(np.percentile(times_arr, 99)),
            'engine': 'json_python',
        }


def load_inference_engine(model_dir: str, treatment_name: str):
    """
    Auto-detect and load the best available inference engine.
    
    Priority: ONNX > JSON
    """
    model_dir = Path(model_dir)
    
    # Try ONNX first (quantized, then regular)
    for suffix in ['_q8.onnx', '.onnx']:
        onnx_path = model_dir / f"macf_{treatment_name}{suffix}"
        if onnx_path.exists():
            try:
                return ONNXInferenceEngine(str(onnx_path))
            except ImportError:
                logger.warning("ONNX model found but onnxruntime not installed")
    
    # Fallback to JSON (quantized, then regular)
    for subdir in ['quantized', '.']:
        for suffix in ['_q8.json', '.json']:
            json_path = model_dir / subdir / f"macf_{treatment_name}{suffix}"
            if json_path.exists():
                return FastTreeInference(str(json_path))
    
    return None


if __name__ == "__main__":
    print("MindBridge Model Export Demo")
    print("=" * 50)
    
    from src.data.synthetic_mci import generate_synthetic_mci_data
    from src.models.macf import MissingnessAwareCausalForest
    
    # Quick train
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500)
    
    macf = MissingnessAwareCausalForest(n_trees=50, max_depth=5, seed=42, n_jobs=1)
    macf.fit(X.values, Y, T['hba1c_reduced'].values, verbose=True)
    
    # Export to ONNX (preferred)
    onnx_path = export_macf_to_onnx(macf, 'hba1c_reduced', output_dir='models')
    
    # Export to JSON (fallback)
    json_exporter = CausalForestExporter()
    json_path = json_exporter.export_macf(macf, 'hba1c_reduced', output_dir='models')
    
    # Compare inference engines
    print("\n--- Inference Comparison ---")
    x_test = X.values[:5]
    
    json_engine = FastTreeInference(json_path)
    json_pred = json_engine.predict(x_test)
    json_bench = json_engine.benchmark(X.values)
    print(f"JSON:  mean={json_bench['mean_ms']:.2f}ms, p95={json_bench['p95_ms']:.2f}ms")
    
    try:
        onnx_engine = ONNXInferenceEngine(onnx_path)
        onnx_pred = onnx_engine.predict(x_test)
        onnx_bench = onnx_engine.benchmark(X.values)
        print(f"ONNX:  mean={onnx_bench['mean_ms']:.2f}ms, p95={onnx_bench['p95_ms']:.2f}ms")
        print(f"Max prediction diff: {np.max(np.abs(json_pred - onnx_pred)):.6f}")
    except ImportError:
        print("ONNX Runtime not available — JSON-only mode")
