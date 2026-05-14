"""
Missingness-Aware Causal Forest (MACF)

The core novel contribution of this project. Extends Generalized Random Forests
(Athey, Tibshirani & Wager 2019) with missingness-aware splitting:

Key Innovation:
    At each tree node, for any feature with missing values, the algorithm tries
    BOTH "missing → left child" AND "missing → right child" and selects the split
    that maximizes treatment effect heterogeneity: (τ̂_left - τ̂_right)²

This avoids:
    - Dropping patients with missing data (sample size loss)
    - Imputation (which introduces bias under MNAR)
    - Multiple imputation (computationally expensive + wrong uncertainty)

Instead, missingness becomes an INFORMATIVE splitting criterion.

References:
    - Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment
      Effects using Random Forests." JASA.
    - Athey, Tibshirani & Wager (2019). "Generalized Random Forests." Annals of Stats.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from joblib import Parallel, delayed
import warnings


@dataclass
class TreeNode:
    """A node in an honest causal tree."""
    feature_idx: int = -1          # Feature index for splitting
    threshold: float = 0.0         # Split threshold
    missing_goes_left: bool = True # Where missing values go at this split
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    is_leaf: bool = False
    tau_hat: float = 0.0           # Estimated treatment effect (leaf only)
    n_samples: int = 0             # Number of estimation samples in leaf
    tau_var: float = 0.0           # Variance of tau estimate (for CIs)


class MACFTree:
    """
    A single Missingness-Aware Causal Tree with honest splitting.
    
    Honest splitting means:
        - 50% of the subsample is used to determine tree structure
        - 50% is used to estimate treatment effects in leaves
    This prevents overfitting of the CATE estimates.
    """
    
    def __init__(
        self,
        min_leaf_size: int = 20,
        max_depth: int = 10,
        honesty_fraction: float = 0.5,
        min_samples_treatment: int = 5,
    ):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.honesty_fraction = honesty_fraction
        self.min_samples_treatment = min_samples_treatment
        self.root = None
    
    def _compute_tau_in_subset(
        self, Y: np.ndarray, T: np.ndarray, W: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute treatment effect estimate in a subset.
        
        Uses the residualized estimator:
            τ̂ = E[Y_resid | T=1] - E[Y_resid | T=0]
        where Y_resid = Y - μ̂(X) and T is centered by ê(X).
        
        For simplicity in the tree splitting phase, we use the raw 
        difference-in-means as a proxy.
        """
        treated = T == 1
        control = T == 0
        
        n_treated = treated.sum()
        n_control = control.sum()
        
        if n_treated < self.min_samples_treatment or n_control < self.min_samples_treatment:
            return 0.0, float('inf')
        
        # Weighted difference in means
        if W is not None:
            tau = (
                np.average(Y[treated], weights=W[treated]) -
                np.average(Y[control], weights=W[control])
            )
            # Variance estimate
            var_t = np.average((Y[treated] - np.average(Y[treated], weights=W[treated]))**2,
                               weights=W[treated]) / n_treated
            var_c = np.average((Y[control] - np.average(Y[control], weights=W[control]))**2,
                               weights=W[control]) / n_control
        else:
            tau = Y[treated].mean() - Y[control].mean()
            var_t = Y[treated].var() / n_treated if n_treated > 1 else float('inf')
            var_c = Y[control].var() / n_control if n_control > 1 else float('inf')
        
        tau_var = var_t + var_c
        return tau, tau_var
    
    def _find_best_split(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        W: Optional[np.ndarray],
        depth: int
    ) -> Tuple[int, float, bool, float]:
        """
        Find the best split that maximizes treatment effect heterogeneity.
        
        THE KEY INNOVATION: For features with missing values, we try both
        "missing → left" and "missing → right" at each candidate split.
        
        Returns:
            (feature_idx, threshold, missing_goes_left, best_score)
        """
        n_samples, n_features = X.shape
        best_score = -float('inf')
        best_feature = -1
        best_threshold = 0.0
        best_missing_left = True
        
        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            is_missing = np.isnan(feature_values)
            has_missing = is_missing.any()
            
            # Get non-missing values for candidate thresholds
            non_missing_mask = ~is_missing
            non_missing_values = feature_values[non_missing_mask]
            
            if len(non_missing_values) < 2 * self.min_leaf_size:
                continue
            
            # Candidate thresholds: quantiles of non-missing values
            n_candidates = min(20, len(np.unique(non_missing_values)) - 1)
            if n_candidates < 1:
                continue
            
            quantiles = np.linspace(0.1, 0.9, n_candidates)
            thresholds = np.quantile(non_missing_values, quantiles)
            thresholds = np.unique(thresholds)
            
            for threshold in thresholds:
                # Try both missing directions if there are missing values
                missing_directions = [True, False] if has_missing else [True]
                
                for missing_goes_left in missing_directions:
                    # Partition samples
                    if has_missing:
                        if missing_goes_left:
                            left_mask = (feature_values <= threshold) | is_missing
                        else:
                            left_mask = (feature_values <= threshold) & ~is_missing
                    else:
                        left_mask = feature_values <= threshold
                    
                    right_mask = ~left_mask
                    
                    n_left = left_mask.sum()
                    n_right = right_mask.sum()
                    
                    # Check minimum leaf size
                    if n_left < self.min_leaf_size or n_right < self.min_leaf_size:
                        continue
                    
                    # Check minimum treatment/control in each child
                    if (T[left_mask].sum() < self.min_samples_treatment or
                        (1 - T[left_mask]).sum() < self.min_samples_treatment or
                        T[right_mask].sum() < self.min_samples_treatment or
                        (1 - T[right_mask]).sum() < self.min_samples_treatment):
                        continue
                    
                    # Compute treatment effects in each child
                    tau_left, _ = self._compute_tau_in_subset(
                        Y[left_mask], T[left_mask],
                        W[left_mask] if W is not None else None
                    )
                    tau_right, _ = self._compute_tau_in_subset(
                        Y[right_mask], T[right_mask],
                        W[right_mask] if W is not None else None
                    )
                    
                    # Score = treatment effect heterogeneity
                    # (τ̂_left - τ̂_right)² weighted by partition sizes
                    score = (n_left * n_right / n_samples**2) * (tau_left - tau_right)**2
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feat_idx
                        best_threshold = threshold
                        best_missing_left = missing_goes_left
        
        return best_feature, best_threshold, best_missing_left, best_score
    
    def _build_tree(
        self,
        X_struct: np.ndarray,
        Y_struct: np.ndarray,
        T_struct: np.ndarray,
        W_struct: Optional[np.ndarray],
        X_est: np.ndarray,
        Y_est: np.ndarray,
        T_est: np.ndarray,
        W_est: Optional[np.ndarray],
        depth: int = 0
    ) -> TreeNode:
        """
        Recursively build the tree using structure sample, then estimate
        with estimation sample (honest splitting).
        """
        node = TreeNode()
        n_struct = len(X_struct)
        n_est = len(X_est)
        
        # Base cases: create leaf
        if (depth >= self.max_depth or
            n_struct < 2 * self.min_leaf_size or
            n_est < self.min_leaf_size):
            node.is_leaf = True
            if n_est > 0:
                node.tau_hat, node.tau_var = self._compute_tau_in_subset(
                    Y_est, T_est, W_est
                )
            node.n_samples = n_est
            return node
        
        # Find best split using STRUCTURE sample
        feat_idx, threshold, missing_left, score = self._find_best_split(
            X_struct, Y_struct, T_struct, W_struct, depth
        )
        
        if feat_idx == -1 or score <= 0:
            node.is_leaf = True
            if n_est > 0:
                node.tau_hat, node.tau_var = self._compute_tau_in_subset(
                    Y_est, T_est, W_est
                )
            node.n_samples = n_est
            return node
        
        node.feature_idx = feat_idx
        node.threshold = threshold
        node.missing_goes_left = missing_left
        
        # Split BOTH structure and estimation samples
        def _get_mask(X_data, feat_idx, threshold, missing_left):
            feat_vals = X_data[:, feat_idx]
            is_missing = np.isnan(feat_vals)
            if missing_left:
                return (feat_vals <= threshold) | is_missing
            else:
                return (feat_vals <= threshold) & ~is_missing
        
        left_struct = _get_mask(X_struct, feat_idx, threshold, missing_left)
        right_struct = ~left_struct
        left_est = _get_mask(X_est, feat_idx, threshold, missing_left)
        right_est = ~left_est
        
        node.left = self._build_tree(
            X_struct[left_struct], Y_struct[left_struct], T_struct[left_struct],
            W_struct[left_struct] if W_struct is not None else None,
            X_est[left_est], Y_est[left_est], T_est[left_est],
            W_est[left_est] if W_est is not None else None,
            depth + 1
        )
        
        node.right = self._build_tree(
            X_struct[right_struct], Y_struct[right_struct], T_struct[right_struct],
            W_struct[right_struct] if W_struct is not None else None,
            X_est[right_est], Y_est[right_est], T_est[right_est],
            W_est[right_est] if W_est is not None else None,
            depth + 1
        )
        
        return node
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        W: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Fit a single MACF tree with honest splitting.
        
        Args:
            X: Features (n × p), may contain NaN
            Y: Residualized outcome
            T: Treatment indicator (0/1)
            W: Sample weights (optional)
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n = len(X)
        
        # Honest split: separate structure and estimation samples
        indices = rng.permutation(n)
        n_struct = int(n * self.honesty_fraction)
        
        struct_idx = indices[:n_struct]
        est_idx = indices[n_struct:]
        
        self.root = self._build_tree(
            X[struct_idx], Y[struct_idx], T[struct_idx],
            W[struct_idx] if W is not None else None,
            X[est_idx], Y[est_idx], T[est_idx],
            W[est_idx] if W is not None else None
        )
    
    def predict_single(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict CATE for a single sample. Returns (tau_hat, tau_var)."""
        node = self.root
        while not node.is_leaf:
            val = x[node.feature_idx]
            if np.isnan(val):
                node = node.left if node.missing_goes_left else node.right
            elif val <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.tau_hat, node.tau_var
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict CATE for multiple samples."""
        predictions = np.array([self.predict_single(x) for x in X])
        return predictions[:, 0], predictions[:, 1]


class MissingnessAwareCausalForest:
    """
    Missingness-Aware Causal Forest (MACF)
    
    An ensemble of MACFTree instances that estimates per-patient
    Conditional Average Treatment Effects (CATE) for a single treatment.
    
    Key features:
        - Handles missing data natively via missingness-aware splits
        - Honest splitting for unbiased CATE estimation
        - Subsampling for valid inference
        - Confidence intervals via tree-level variance
    
    Usage:
        macf = MissingnessAwareCausalForest(n_trees=2000)
        macf.fit(X, Y, T)
        tau_hat, ci_lower, ci_upper = macf.predict(X_new)
    """
    
    def __init__(
        self,
        n_trees: int = 500,
        min_leaf_size: int = 20,
        max_depth: int = 8,
        honesty_fraction: float = 0.5,
        subsample_fraction: float = 0.5,
        min_samples_treatment: int = 5,
        n_jobs: int = -1,
        seed: int = 42,
    ):
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.honesty_fraction = honesty_fraction
        self.subsample_fraction = subsample_fraction
        self.min_samples_treatment = min_samples_treatment
        self.n_jobs = n_jobs
        self.seed = seed
        self.trees: List[MACFTree] = []
        self._is_fitted = False
    
    def _fit_single_tree(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        W: Optional[np.ndarray],
        tree_seed: int
    ) -> MACFTree:
        """Fit a single tree on a subsample."""
        rng = np.random.default_rng(tree_seed)
        n = len(X)
        
        # Subsample
        n_sub = int(n * self.subsample_fraction)
        sub_idx = rng.choice(n, n_sub, replace=False)
        
        tree = MACFTree(
            min_leaf_size=self.min_leaf_size,
            max_depth=self.max_depth,
            honesty_fraction=self.honesty_fraction,
            min_samples_treatment=self.min_samples_treatment,
        )
        
        tree.fit(
            X[sub_idx], Y[sub_idx], T[sub_idx],
            W[sub_idx] if W is not None else None,
            rng=rng
        )
        
        return tree
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        W: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Fit the MACF ensemble.
        
        Args:
            X: Feature matrix (n × p), may contain NaN
            Y: Outcome vector (n,), ideally residualized Y - μ̂(X)
            T: Binary treatment (n,)
            W: Sample weights (optional)
            verbose: Print progress
        """
        if verbose:
            print(f"Training MACF with {self.n_trees} trees on {len(X)} samples...")
        
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        
        # Generate seeds for each tree
        rng = np.random.default_rng(self.seed)
        tree_seeds = rng.integers(0, 2**31, self.n_trees)
        
        # Parallel tree fitting
        n_jobs = self.n_jobs if self.n_jobs > 0 else 1
        
        if n_jobs == 1 or self.n_trees <= 10:
            self.trees = []
            for i in range(self.n_trees):
                tree = self._fit_single_tree(X, Y, T, W, int(tree_seeds[i]))
                self.trees.append(tree)
                if verbose and (i + 1) % max(1, self.n_trees // 10) == 0:
                    print(f"  Trees fitted: {i+1}/{self.n_trees}")
        else:
            try:
                self.trees = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(self._fit_single_tree)(X, Y, T, W, int(seed))
                    for seed in tree_seeds
                )
            except Exception:
                # Fallback to sequential if parallel fails
                self.trees = [
                    self._fit_single_tree(X, Y, T, W, int(seed))
                    for seed in tree_seeds
                ]
        
        self._is_fitted = True
        if verbose:
            print(f"  Training complete. {len(self.trees)} trees fitted.")
    
    def predict(
        self,
        X: np.ndarray,
        return_ci: bool = True,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, ...]:
        """
        Predict CATE for new samples.
        
        Args:
            X: Feature matrix (n × p)
            return_ci: Whether to return confidence intervals
            alpha: Significance level for CIs (default 0.05 → 95% CI)
        
        Returns:
            tau_hat: Point estimate of CATE
            ci_lower: Lower bound of CI (if return_ci)
            ci_upper: Upper bound of CI (if return_ci)
        """
        if not self._is_fitted:
            raise RuntimeError("MACF not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        
        # Collect predictions from all trees
        all_preds = np.zeros((self.n_trees, n))
        
        for t_idx, tree in enumerate(self.trees):
            preds, _ = tree.predict(X)
            all_preds[t_idx] = preds
        
        # Point estimate: mean across trees
        tau_hat = all_preds.mean(axis=0)
        
        if return_ci:
            from scipy import stats
            
            # Standard error from tree-level variation
            tau_se = all_preds.std(axis=0) / np.sqrt(self.n_trees)
            
            z = stats.norm.ppf(1 - alpha / 2)
            ci_lower = tau_hat - z * tau_se
            ci_upper = tau_hat + z * tau_se
            
            return tau_hat, ci_lower, ci_upper
        
        return (tau_hat,)
    
    def predict_single(self, x: np.ndarray) -> Dict:
        """
        Predict CATE for a single patient with full detail.
        
        Returns dict with tau_hat, ci_lower, ci_upper, se, n_trees.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        tau_hat, ci_lower, ci_upper = self.predict(x, return_ci=True)
        
        return {
            'tau_hat': float(tau_hat[0]),
            'ci_lower': float(ci_lower[0]),
            'ci_upper': float(ci_upper[0]),
            'se': float((ci_upper[0] - ci_lower[0]) / (2 * 1.96)),
            'n_trees': self.n_trees,
        }
    
    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute feature importance based on split frequency across trees.
        """
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        def _count_splits(node):
            if node is None or node.is_leaf:
                return
            importance[node.feature_idx] += 1
            _count_splits(node.left)
            _count_splits(node.right)
        
        for tree in self.trees:
            _count_splits(tree.root)
        
        # Normalize
        total = importance.sum()
        if total > 0:
            importance /= total
        
        return importance


if __name__ == "__main__":
    from src.data.ihdp_loader import generate_ihdp_synthetic
    
    print("=" * 60)
    print("MACF Quick Validation on IHDP Benchmark")
    print("=" * 60)
    
    # Load IHDP data
    X, T, Y, tau_true = generate_ihdp_synthetic(seed=42)
    
    # Inject some missingness to test MACF
    rng = np.random.default_rng(99)
    X_missing = X.copy()
    for col in [0, 2, 5]:
        mask = rng.random(len(X)) < 0.2
        X_missing[mask, col] = np.nan
    
    # Train MACF (small for quick test)
    macf = MissingnessAwareCausalForest(
        n_trees=100,
        min_leaf_size=10,
        max_depth=6,
        seed=42,
        n_jobs=1
    )
    macf.fit(X_missing, Y, T, verbose=True)
    
    # Predict
    tau_hat, ci_lo, ci_hi = macf.predict(X_missing)
    
    # Compute PEHE
    pehe = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    
    # Coverage
    coverage = np.mean((tau_true >= ci_lo) & (tau_true <= ci_hi))
    
    print(f"\nResults:")
    print(f"  PEHE:     {pehe:.4f} (target: <0.08)")
    print(f"  Coverage: {coverage:.2%} (target: >90%)")
    print(f"  Mean τ̂:  {tau_hat.mean():.4f} (true: {tau_true.mean():.4f})")
    print(f"  ATE error: {abs(tau_hat.mean() - tau_true.mean()):.4f}")
