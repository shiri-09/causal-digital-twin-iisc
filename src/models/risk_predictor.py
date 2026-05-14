"""
LightGBM Risk Predictor

Baseline 2-year MCI risk prediction model.
Target: AUROC > 0.78 on held-out test set.

This provides the μ̂(X) = P(MCI=1 | X) estimate that clinicians
see as the "baseline risk" before counterfactual interventions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from typing import Tuple, Optional, Dict
import warnings
import joblib


def _add_missing_indicators(X: np.ndarray) -> np.ndarray:
    """Add binary missing indicators and fill NaN with median."""
    n, p = X.shape
    missing_mask = np.isnan(X)
    miss_rates = missing_mask.mean(axis=0)
    high_miss_cols = np.where(miss_rates > 0.05)[0]
    indicators = missing_mask[:, high_miss_cols].astype(np.float64)
    
    X_filled = X.copy()
    medians = np.nanmedian(X_filled, axis=0)
    for col in range(p):
        nan_mask = np.isnan(X_filled[:, col])
        if nan_mask.any():
            X_filled[nan_mask, col] = medians[col]
    
    return np.hstack([X_filled, indicators]), medians


class MCIRiskPredictor:
    """
    LightGBM-based 2-year MCI risk predictor.
    
    Uses explicit missing indicators (not imputation) to preserve
    the informative missingness signal.
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.medians = None
        self.n_original_features = None
        self._is_fitted = False
    
    def _prepare_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Prepare features with missing indicators."""
        n, p = X.shape
        
        if fit:
            self.n_original_features = p
        
        missing_mask = np.isnan(X)
        miss_rates = missing_mask.mean(axis=0)
        high_miss_cols = np.where(miss_rates > 0.05)[0]
        indicators = missing_mask[:, high_miss_cols].astype(np.float64)
        
        X_filled = X.copy()
        if fit:
            self.medians = np.nanmedian(X_filled, axis=0)
        
        for col in range(p):
            nan_mask = np.isnan(X_filled[:, col])
            if nan_mask.any():
                X_filled[nan_mask, col] = self.medians[col]
        
        return np.hstack([X_filled, indicators])
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        eval_X: Optional[np.ndarray] = None,
        eval_Y: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train the risk predictor with 5-fold CV for AUROC optimization.
        """
        try:
            import lightgbm as lgb
            use_lgb = True
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            use_lgb = False
            warnings.warn("LightGBM not available, using sklearn GBM")
        
        X_aug = self._prepare_features(X, fit=True)
        
        if use_lgb:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=self.seed,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=min(200, self.n_estimators),
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                random_state=self.seed,
            )
        
        self.model.fit(X_aug, Y)
        self._is_fitted = True
        
        # Report training AUROC
        if verbose:
            train_pred = self.model.predict_proba(X_aug)[:, 1]
            train_auroc = roc_auc_score(Y, train_pred)
            print(f"  Training AUROC: {train_auroc:.4f}")
            
            if eval_X is not None and eval_Y is not None:
                eval_aug = self._prepare_features(eval_X)
                eval_pred = self.model.predict_proba(eval_aug)[:, 1]
                eval_auroc = roc_auc_score(eval_Y, eval_pred)
                print(f"  Validation AUROC: {eval_auroc:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict MCI probability for each patient."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_aug = self._prepare_features(X)
        return self.model.predict_proba(X_aug)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary MCI outcome."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def cross_validate(
        self, X: np.ndarray, Y: np.ndarray, n_folds: int = 5
    ) -> Dict:
        """5-fold cross-validation returning AUROC per fold."""
        X_aug = self._prepare_features(X, fit=True)
        
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        aurocs = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_aug, Y)):
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, verbose=-1,
                    random_state=self.seed + fold, n_jobs=-1,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(
                    n_estimators=min(200, self.n_estimators),
                    max_depth=self.max_depth, random_state=self.seed + fold,
                )
            
            model.fit(X_aug[train_idx], Y[train_idx])
            pred = model.predict_proba(X_aug[val_idx])[:, 1]
            auroc = roc_auc_score(Y[val_idx], pred)
            aurocs.append(auroc)
        
        return {
            'aurocs': aurocs,
            'mean_auroc': np.mean(aurocs),
            'std_auroc': np.std(aurocs),
        }
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'medians': self.medians,
            'n_original_features': self.n_original_features,
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.medians = data['medians']
        self.n_original_features = data['n_original_features']
        self._is_fitted = True


if __name__ == "__main__":
    from src.data.synthetic_mci import generate_synthetic_mci_data
    
    print("Testing MCI Risk Predictor...")
    X, T, Y, tau, prob = generate_synthetic_mci_data(n_samples=2000)
    
    predictor = MCIRiskPredictor()
    cv_results = predictor.cross_validate(X.values, Y)
    
    print(f"\n5-Fold CV Results:")
    for i, auroc in enumerate(cv_results['aurocs']):
        print(f"  Fold {i+1}: AUROC = {auroc:.4f}")
    print(f"  Mean AUROC: {cv_results['mean_auroc']:.4f} ± {cv_results['std_auroc']:.4f}")
    print(f"  Target: >0.78")
