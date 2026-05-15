"""
Unit Tests for Data Generation and Preprocessing
"""

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic_mci import (
    generate_synthetic_mci_data,
    generate_train_val_test_split,
    get_treatment_names,
    get_feature_names,
)


class TestSyntheticDataGeneration:
    """Tests for synthetic_mci.py"""

    def test_basic_generation(self):
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=100, seed=42)
        assert len(X) == 100
        assert len(Y) == 100

    def test_feature_count(self):
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=100)
        assert X.shape[1] == 12  # 12 clinical features

    def test_treatment_count(self):
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=100)
        assert T.shape[1] == 4  # 4 treatments

    def test_outcome_binary(self):
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500)
        assert set(np.unique(Y)).issubset({0, 1})

    def test_treatment_binary(self):
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500)
        for col in T.columns:
            assert set(np.unique(T[col])).issubset({0, 1})

    def test_missingness_exists(self):
        """OCT and gait should have missing values (MNAR)."""
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=1000)
        assert X['oct_rnfl_thickness'].isna().sum() > 0
        assert X['gait_speed'].isna().sum() > 0

    def test_missingness_rates(self):
        """Missing rates should approximately match proposal targets."""
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=5000, seed=42)
        oct_miss = X['oct_rnfl_thickness'].isna().mean()
        gait_miss = X['gait_speed'].isna().mean()
        assert 0.25 < oct_miss < 0.55, f"OCT missing rate {oct_miss:.2f} outside [0.25, 0.55]"
        assert 0.15 < gait_miss < 0.40, f"Gait missing rate {gait_miss:.2f} outside [0.15, 0.40]"

    def test_tau_has_variation(self):
        """True CATE should vary across patients (heterogeneity)."""
        X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=500)
        for name in tau:
            assert tau[name].std() > 0.001, f"No CATE variation in {name}"

    def test_reproducibility(self):
        """Same seed should produce same data."""
        X1, T1, Y1, _, _ = generate_synthetic_mci_data(n_samples=100, seed=99)
        X2, T2, Y2, _, _ = generate_synthetic_mci_data(n_samples=100, seed=99)
        pd.testing.assert_frame_equal(X1, X2)

    def test_train_val_test_split(self):
        splits = generate_train_val_test_split(n_samples=1000)
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        # Approximate size checks
        X_train, _, _, _ = splits['train']
        X_val, _, _, _ = splits['val']
        X_test, _, _, _ = splits['test']
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 1000

    def test_feature_names(self):
        names = get_feature_names()
        assert 'age' in names
        assert 'hba1c' in names
        assert len(names) == 12

    def test_treatment_names(self):
        names = get_treatment_names()
        assert 'hba1c_reduced' in names
        assert len(names) == 4


class TestIHDPLoader:
    """Tests for ihdp_loader.py"""

    def test_ihdp_generation(self):
        from src.data.ihdp_loader import generate_ihdp_synthetic
        X, T, Y, tau = generate_ihdp_synthetic()
        assert X.shape == (747, 25)
        assert len(T) == 747
        assert len(Y) == 747
        assert len(tau) == 747

    def test_ihdp_treatment_binary(self):
        from src.data.ihdp_loader import generate_ihdp_synthetic
        X, T, Y, tau = generate_ihdp_synthetic()
        assert set(np.unique(T)).issubset({0, 1})

    def test_ihdp_split(self):
        from src.data.ihdp_loader import load_ihdp_data
        data = load_ihdp_data()
        n_train = len(data['train']['X'])
        n_test = len(data['test']['X'])
        assert n_train + n_test == 747


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
