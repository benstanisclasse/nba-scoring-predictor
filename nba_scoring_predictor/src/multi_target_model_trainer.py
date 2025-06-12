# -*- coding: utf-8 -*-
"""
Multi-target model trainer for points, assists, and rebounds
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTargetModelTrainer:
    """Multi-target model trainer for NBA statistics."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.target_names = ['PTS', 'AST', 'REB']
        logger.info("MultiTargetModelTrainer initialized.")
        
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    feature_names: List[str], target_names: List[str],
                    optimize: bool = True) -> Dict[str, Any]:
        """Train multi-target models."""
        logger.info("Training multi-target models...")
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Features scaled successfully.")
        
        # Split data
        split_point = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        logger.info(f"Data split into training set: {X_train.shape[0]} samples, test set: {X_test.shape[0]} samples.")
        
        results = {}
        
        # XGBoost multi-target
        logger.info("Training XGBoost model...")
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
        )
        xgb_model.fit(X_train, y_train)
        results['xgboost'] = self._evaluate_multi_target_model(
            xgb_model, X_train, X_test, y_train, y_test
        )
        logger.info("XGBoost model trained and evaluated.")
        
        # LightGBM multi-target
        logger.info("Training LightGBM model...")
        lgb_model = MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimators=200, max_depth=6, random_state=42, verbose=-1)
        )
        lgb_model.fit(X_train, y_train)
        results['lightgbm'] = self._evaluate_multi_target_model(
            lgb_model, X_train, X_test, y_train, y_test
        )
        logger.info("LightGBM model trained and evaluated.")
        
        # Random Forest multi-target
        logger.info("Training Random Forest model...")
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        )
        rf_model.fit(X_train, y_train)
        results['random_forest'] = self._evaluate_multi_target_model(
            rf_model, X_train, X_test, y_train, y_test
        )
        logger.info("Random Forest model trained and evaluated.")
        
        # Ensemble (average of all models)
        logger.info("Creating ensemble predictions...")
        ensemble_preds = self._create_ensemble_predictions(
            [xgb_model, lgb_model, rf_model], X_test
        )
        results['ensemble'] = self._evaluate_ensemble(ensemble_preds, y_test)
        logger.info("Ensemble predictions created and evaluated.")
        
        self.models = results
        return results
    
    def _evaluate_multi_target_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate multi-target model."""
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        result = {'model': model}
        
        # Calculate metrics for each target
        for i, target in enumerate(self.target_names):
            result[f'{target}_train_mae'] = mean_absolute_error(y_train[:, i], y_pred_train[:, i])
            result[f'{target}_test_mae'] = mean_absolute_error(y_test[:, i], y_pred_test[:, i])
            result[f'{target}_train_r2'] = r2_score(y_train[:, i], y_pred_train[:, i])
            result[f'{target}_test_r2'] = r2_score(y_test[:, i], y_pred_test[:, i])
            logger.info(f"{target} - Train MAE: {result[f'{target}_train_mae']:.3f}, Test MAE: {result[f'{target}_test_mae']:.3f}, Train R²: {result[f'{target}_train_r2']:.3f}, Test R²: {result[f'{target}_test_r2']:.3f}")
        
        return result
    
    def _create_ensemble_predictions(self, models, X_test):
        """Create ensemble predictions."""
        predictions = []
        for model in models:
            predictions.append(model.predict(X_test))
        
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def _evaluate_ensemble(self, y_pred, y_test):
        """Evaluate ensemble predictions."""
        result = {}
        
        for i, target in enumerate(self.target_names):
            result[f'{target}_test_mae'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
            result[f'{target}_test_r2'] = r2_score(y_test[:, i], y_pred[:, i])
            logger.info(f"Ensemble - {target} - Test MAE: {result[f'{target}_test_mae']:.3f}, Test R²: {result[f'{target}_test_r2']:.3f}")
        
        return result
    
    def predict_multi_target(self, X: np.ndarray, model_name: str = 'ensemble') -> Dict:
        """Make multi-target predictions."""
        if model_name not in self.models:
            model_name = 'xgboost'  # Fallback
        
        X_scaled = self.scaler.transform(X)
        
        if model_name == 'ensemble':
            # Use individual models for ensemble
            models = [self.models['xgboost']['model'], 
                     self.models['lightgbm']['model'],
                     self.models['random_forest']['model']]
            predictions = self._create_ensemble_predictions(models, X_scaled)
        else:
            model = self.models[model_name]['model']
            predictions = model.predict(X_scaled)
        
        # Convert to dictionary format
        result = {}
        for i, target in enumerate(self.target_names):
            result[target] = predictions[:, i]
        
        return result