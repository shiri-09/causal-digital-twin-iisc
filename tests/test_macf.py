"""
Unit Tests for MACF Algorithm
"""

import pytest
import numpy as np
from src.models.macf import MACFTree, MissingnessAwareCausalForest


class TestMACFTree:
    """Tests for individual causal tree."""

    def test_tree_fits(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        T = rng.binomial(1, 0.5, 200).astype(float)
        Y = X[:, 0] * 0.3 + T * 0.1 + rng.normal(0, 0.1, 200)

        tree = MACFTree(min_leaf_size=10, max_depth=4)
        tree.fit(X, Y, T, rng=rng)
        assert tree.root is not None

    def test_tree_predicts(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        T = rng.binomial(1, 0.5, 200).astype(float)
        Y = X[:, 0] * 0.3 + T * 0.1 + rng.normal(0, 0.1, 200)

        tree = MACFTree(min_leaf_size=10, max_depth=4)
        tree.fit(X, Y, T, rng=rng)
        tau_hat, tau_var = tree.predict(X[:5])
        assert len(tau_hat) == 5
        assert all(np.isfinite(tau_hat))

    def test_tree_handles_nan(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        X[rng.random(200) < 0.2, 0] = np.nan  # 20% missing in col 0

        T = rng.binomial(1, 0.5, 200).astype(float)
        Y = T * 0.1 + rng.normal(0, 0.1, 200)

        tree = MACFTree(min_leaf_size=10, max_depth=4)
        tree.fit(X, Y, T, rng=rng)

        # Predict with missing values
        x_test = np.array([np.nan, 0.5, -0.3, 1.0, 0.2])
        tau, var = tree.predict_single(x_test)
        assert np.isfinite(tau)


class TestMACF:
    """Tests for the full MACF ensemble."""

    def test_macf_fits_and_predicts(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 5))
        T = rng.binomial(1, 0.5, n).astype(float)
        tau_true = 0.1 + 0.05 * X[:, 0]
        Y = 0.5 * X[:, 0] + T * tau_true + rng.normal(0, 0.1, n)

        macf = MissingnessAwareCausalForest(
            n_trees=20, min_leaf_size=10, max_depth=4, seed=42, n_jobs=1
        )
        macf.fit(X, Y, T, verbose=False)

        tau_hat, ci_lo, ci_hi = macf.predict(X)
        assert len(tau_hat) == n
        assert all(ci_lo <= tau_hat)
        assert all(tau_hat <= ci_hi)

    def test_macf_with_missing_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 5))
        X[rng.random(n) < 0.3, 2] = np.nan

        T = rng.binomial(1, 0.5, n).astype(float)
        Y = T * 0.1 + rng.normal(0, 0.1, n)

        macf = MissingnessAwareCausalForest(
            n_trees=20, min_leaf_size=10, max_depth=4, seed=42, n_jobs=1
        )
        macf.fit(X, Y, T, verbose=False)
        tau_hat, ci_lo, ci_hi = macf.predict(X)
        assert all(np.isfinite(tau_hat))

    def test_macf_feature_importance(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 5))
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = X[:, 0] + T * (0.1 + 0.2 * X[:, 0]) + rng.normal(0, 0.05, n)

        macf = MissingnessAwareCausalForest(
            n_trees=20, min_leaf_size=10, max_depth=4, seed=42, n_jobs=1
        )
        macf.fit(X, Y, T, verbose=False)
        importance = macf.feature_importance(X)
        assert len(importance) == 5
        assert abs(importance.sum() - 1.0) < 0.01

    def test_predict_single(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        T = rng.binomial(1, 0.5, 200).astype(float)
        Y = T * 0.1 + rng.normal(0, 0.1, 200)

        macf = MissingnessAwareCausalForest(
            n_trees=10, min_leaf_size=10, max_depth=3, seed=42, n_jobs=1
        )
        macf.fit(X, Y, T, verbose=False)

        result = macf.predict_single(X[0])
        assert 'tau_hat' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['ci_lower'] <= result['tau_hat'] <= result['ci_upper']

    def test_not_fitted_raises(self):
        macf = MissingnessAwareCausalForest(n_trees=10)
        with pytest.raises(RuntimeError):
            macf.predict(np.zeros((5, 3)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
