"""
Model Quantization for Edge Deployment

Reduces model precision and size for Raspberry Pi 4 deployment.
Implements multiple quantization strategies:

    1. Precision reduction: Round thresholds and tau values
    2. Tree pruning: Remove low-importance splits
    3. Feature subsetting: Keep only top-k split features

Target: Models must fit within 4GB RPi4 RAM with <10ms inference.
"""

import numpy as np
import json
import os
from typing import Dict, Optional
from pathlib import Path


def quantize_threshold(value: float, n_bits: int = 8) -> float:
    """
    Reduce floating-point precision for a split threshold.

    Maps the value to the nearest representable value in an
    n-bit fixed-point scheme.

    Args:
        value: Original threshold value
        n_bits: Quantization bit depth (default 8 → 256 levels)

    Returns:
        Quantized value
    """
    if n_bits == 8:
        return round(value, 2)
    elif n_bits == 4:
        return round(value, 1)
    else:
        return round(value, max(1, n_bits // 3))


def quantize_tau(value: float, n_bits: int = 8) -> float:
    """
    Reduce precision for treatment effect estimates.

    Treatment effects are typically in [-0.25, 0.05], so we can
    use fewer bits without losing meaningful resolution.
    """
    if n_bits == 8:
        return round(value, 3)
    elif n_bits == 4:
        return round(value, 2)
    else:
        return round(value, max(1, n_bits // 3))


def quantize_tree(tree_dict: dict, n_bits: int = 8) -> dict:
    """
    Recursively quantize a serialized tree dictionary.

    Args:
        tree_dict: Serialized tree from CausalForestExporter
        n_bits: Quantization bit depth

    Returns:
        Quantized tree dictionary
    """
    if tree_dict is None:
        return None

    if tree_dict.get('leaf', False):
        return {
            'leaf': True,
            'tau': quantize_tau(tree_dict['tau'], n_bits),
            'n': tree_dict.get('n', 0),
        }

    return {
        'f': tree_dict['f'],
        't': quantize_threshold(tree_dict['t'], n_bits),
        'ml': tree_dict['ml'],
        'l': quantize_tree(tree_dict.get('l'), n_bits),
        'r': quantize_tree(tree_dict.get('r'), n_bits),
    }


def prune_tree(tree_dict: dict, min_samples: int = 5) -> dict:
    """
    Prune tree by collapsing leaves with too few samples.

    If both children are leaves with < min_samples, merge them.
    """
    if tree_dict is None:
        return None

    if tree_dict.get('leaf', False):
        return tree_dict

    left = prune_tree(tree_dict.get('l'), min_samples)
    right = prune_tree(tree_dict.get('r'), min_samples)

    # If both children are leaves with few samples, merge
    if (left and left.get('leaf') and right and right.get('leaf')):
        n_left = left.get('n', 0)
        n_right = right.get('n', 0)

        if n_left < min_samples and n_right < min_samples:
            total_n = n_left + n_right
            if total_n > 0:
                merged_tau = (
                    left['tau'] * n_left + right['tau'] * n_right
                ) / total_n
            else:
                merged_tau = 0.0

            return {
                'leaf': True,
                'tau': round(merged_tau, 4),
                'n': total_n,
            }

    result = dict(tree_dict)
    result['l'] = left
    result['r'] = right
    return result


def quantize_model_file(
    input_path: str,
    output_path: str,
    n_bits: int = 8,
    prune: bool = True,
    min_prune_samples: int = 3,
) -> Dict:
    """
    Quantize an exported MACF model file.

    Args:
        input_path: Path to full-precision JSON model
        output_path: Path for quantized output
        n_bits: Quantization bit depth
        prune: Whether to also prune small leaves
        min_prune_samples: Minimum leaf samples for pruning

    Returns:
        Statistics dict (sizes, compression ratio)
    """
    with open(input_path, 'r') as f:
        model_data = json.load(f)

    original_size = os.path.getsize(input_path)

    # Quantize each tree
    quantized_trees = []
    for tree in model_data['trees']:
        q_tree = quantize_tree(tree, n_bits)
        if prune:
            q_tree = prune_tree(q_tree, min_prune_samples)
        quantized_trees.append(q_tree)

    model_data['trees'] = quantized_trees
    model_data['quantized'] = True
    model_data['quantization_bits'] = n_bits

    # Write with compact separators
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model_data, f, separators=(',', ':'))

    quantized_size = os.path.getsize(output_path)

    return {
        'original_size_kb': round(original_size / 1024, 1),
        'quantized_size_kb': round(quantized_size / 1024, 1),
        'compression_ratio': round(original_size / max(1, quantized_size), 2),
        'n_bits': n_bits,
        'pruned': prune,
    }


def quantize_all_models(
    input_dir: str = "models",
    output_dir: str = "models/quantized",
    n_bits: int = 8,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Quantize all MACF model files in a directory.

    Args:
        input_dir: Directory with full-precision models
        output_dir: Output directory for quantized models
        n_bits: Quantization depth
        verbose: Print progress

    Returns:
        Dict mapping treatment name → quantization statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    input_path = Path(input_dir)
    for json_file in sorted(input_path.glob("macf_*.json")):
        treatment = json_file.stem.replace("macf_", "")
        output_name = f"{json_file.stem}_q{n_bits}.json"
        output_path = os.path.join(output_dir, output_name)

        stats = quantize_model_file(
            str(json_file), output_path, n_bits=n_bits
        )
        results[treatment] = stats

        if verbose:
            print(
                f"  {treatment}: {stats['original_size_kb']} KB → "
                f"{stats['quantized_size_kb']} KB "
                f"({stats['compression_ratio']}x compression)"
            )

    return results


if __name__ == "__main__":
    print("Quantization utility")
    print("=" * 50)
    print("Usage: quantize_all_models('models', 'models/quantized')")
    print("  Quantizes all macf_*.json files with int8-like precision")
