# src/ensemble_models.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple

class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    """Advanced ensemble with dynamic weighting and position-aware models."""
    
    def __init__(self, position_aware=True):
        self.position_aware = position_aware
        self.base_models = {}
        self.position_models = {}
        self.meta_model = None
        self.weights = None
        
    def fit(self, X, y, positions=None):
        """Fit ensemble with optional position awareness."""
        
        # Base models
        self.base_models = {
            'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=200, max_depth=6, random_state=42, verbose=-1),
            'gbm': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
            'extra': ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
            'elastic': ElasticNet(alpha=0.1, random_state=42),
            'huber': HuberRegressor()
        }
        
        # Fit base models
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                model.fit(X, y)
                base_predictions[:, i] = model.predict(X)
            except Exception as e:
                print(f"Error fitting {name}: {e}")
                base_predictions[:, i] = np.mean(y)  # Fallback to mean
        
        # Position-specific models if enabled
        if self.position_aware and positions is not None:
            unique_positions = np.unique(positions)
            
            for pos in unique_positions:
                if np.sum(positions == pos) >= 50:  # Minimum samples for position model
                    pos_mask = positions == pos
                    X_pos = X[pos_mask]
                    y_pos = y[pos_mask]
                    
                    pos_model = xgb.XGBRegressor(
                        n_estimators=100, max_depth=4, random_state=42
                    )
                    pos_model.fit(X_pos, y_pos)
                    self.position_models[pos] = pos_model
        
        # Meta-learner for dynamic weighting
        self.meta_model = ElasticNet(alpha=0.01, random_state=42)
        self.meta_model.fit(base_predictions, y)
        
        # Calculate base model weights based on performance
        self.weights = self._calculate_weights(base_predictions, y)
        
        return self
    
    def predict(self, X, positions=None):
        """Make predictions with ensemble."""
        
        # Base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                base_predictions[:, i] = model.predict(X)
            except:
                base_predictions[:, i] = 15.0  # Fallback prediction
        
        # Meta-learner prediction
        meta_pred = self.meta_model.predict(base_predictions)
        
        # Weighted average of base models
        weighted_pred = np.average(base_predictions, weights=self.weights, axis=1)
        
        # Position-specific adjustments
        final_pred = (meta_pred + weighted_pred) / 2
        
        if self.position_aware and positions is not None:
            for i, pos in enumerate(positions):
                if pos in self.position_models:
                    pos_pred = self.position_models[pos].predict(X[i:i+1])[0]
                    # Blend position-specific with ensemble
                    final_pred[i] = 0.7 * final_pred[i] + 0.3 * pos_pred
        
        return np.clip(final_pred, 0, 50)  # Reasonable bounds
    
    def _calculate_weights(self, predictions, y):
        """Calculate dynamic weights based on model performance."""
        weights = []
        
        for i in range(predictions.shape[1]):
            mae = mean_absolute_error(y, predictions[:, i])
            weight = 1 / (mae + 1e-6)  # Inverse MAE weighting
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        return weights / np.sum(weights)
