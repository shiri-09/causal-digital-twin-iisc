"""
Data package — data generation, loading, and preprocessing.

Exports:
    generate_synthetic_mci_data: SANSCOG-like synthetic data generator
    generate_train_val_test_split: Stratified train/val/test split
    load_ihdp_data: IHDP benchmark loader
    create_missing_indicators: Binary missingness indicator creation
    preprocess_pipeline: Full preprocessing pipeline
"""

from src.data.synthetic_mci import (
    generate_synthetic_mci_data,
    generate_train_val_test_split,
    get_feature_names,
    get_treatment_names,
)

__all__ = [
    'generate_synthetic_mci_data',
    'generate_train_val_test_split',
    'get_feature_names',
    'get_treatment_names',
]
