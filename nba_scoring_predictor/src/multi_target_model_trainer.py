# src/multi_target_model_trainer.py
# -*- coding: utf-8 -*-
"""
Multi-target model training for predicting points, assists, and rebounds
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import optuna
from optuna import Trial

from config.settings import MODEL_CONFIGS, OPTIMIZATION_TRIALS
from utils.logger import main_logger as logger

class MultiTargetModelTrainer:
    """Handles multi-target model training for points, assists, and rebounds."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.target_names = ['PTS', 'AST', 'REB']
        self.training_history = {}
    
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                    target_names: List[str] = None, optimize: bool = True) -> Dict[str, Any]:
        """
        Train multi-target ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target matrix (n_samples, n_targets)
            feature_names: List of feature names
            target_names: List of target names (PTS, AST, REB)
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training multi-target models...")
        
        self.feature_names = feature_names
        self.target_names = target_names or self.target_names
        
        # Validate input dimensions
        if y.ndim == 1:
            raise ValueError("y must be 2D array for multi-target prediction")
        if y.shape[1] != len(self.target_names):
            raise ValueError(f"y must have {len(self.target_names)} columns for targets: {self.target_names}")
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Validate targets
        for i, target in enumerate(self.target_names):
            target_data = y[:, i]
            valid_mask = (target_data >= 0) & (target_data <= self._get_target_max(target))
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                logger.warning(f"Removing {invalid_count} samples with invalid {target} values")
                X_scaled = X_scaled[valid_mask]
                y = y[valid_mask]
        
        # Ensure we have enough data
        if len(y) < 100:
            raise ValueError(f"Insufficient valid training data: {len(y)} samples")
        
        logger.info(f"Training with {len(y)} valid samples on {len(self.target_names)} targets")
        
        # Chronological split (80/20)
        split_point = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        results = {}
        
        # XGBoost Multi-target
        if optimize:
            xgb_params = self._optimize_hyperparameters(X_train, y_train, 'xgboost', 30)
        else:
            xgb_params = {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': 0
            }
        
        # Create multi-target XGBoost
        xgb_models = {}
        for i, target in enumerate(self.target_names):
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_train, y_train[:, i])
            xgb_models[target] = xgb_model
        
        results['xgboost'] = self._evaluate_multi_target_model(
            xgb_models, X_train, X_test, y_train, y_test
        )
        
        # LightGBM Multi-target
        if optimize:
            lgb_params = self._optimize_hyperparameters(X_train, y_train, 'lightgbm', 30)
        else:
            lgb_params = {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'num_leaves': 50, 'random_state': 42, 'verbose': -1
            }
        
        lgb_models = {}
        for i, target in enumerate(self.target_names):
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train, y_train[:, i])
            lgb_models[target] = lgb_model
        
        results['lightgbm'] = self._evaluate_multi_target_model(
            lgb_models, X_train, X_test, y_train, y_test
        )
        
        # Random Forest Multi-target
        if optimize:
            rf_params = self._optimize_hyperparameters(X_train, y_train, 'random_forest', 20)
        else:
            rf_params = {
                'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
                'min_samples_leaf': 2, 'random_state': 42
            }
        
        rf_models = {}
        for i, target in enumerate(self.target_names):
            rf_model = RandomForestRegressor(**rf_params)
            rf_model.fit(X_train, y_train[:, i])
            rf_models[target] = rf_model
        
        results['random_forest'] = self._evaluate_multi_target_model(
            rf_models, X_train, X_test, y_train, y_test
        )
        
        # Neural Network Multi-target
        nn_models = {}
        for i, target in enumerate(self.target_names):
            nn_model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            nn_model.fit(X_train, y_train[:, i])
            nn_models[target] = nn_model
        
        results['neural_network'] = self._evaluate_multi_target_model(
            nn_models, X_train, X_test, y_train, y_test
        )
        
        # Ensemble - average of top models
        ensemble_models = {
            'xgboost': xgb_models,
            'lightgbm': lgb_models,
            'random_forest': rf_models
        }
        results['ensemble'] = self._create_ensemble_model(
            ensemble_models, X_train, X_test, y_train, y_test
        )
        
        self.models = results
        self.training_history = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(feature_names),
            'n_targets': len(self.target_names),
            'optimization_used': optimize
        }
        
        # Log results
        self._log_results(results)
        
        return results
    
    def _get_target_max(self, target: str) -> float:
        """Get reasonable maximum value for each target."""
        maxes = {
            'PTS': 81,    # Kobe's record
            'AST': 30,    # Very high assist game
            'REB': 30     # Very high rebound game
        }
        return maxes.get(target, 100)
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                 model_type: str, n_trials: int = OPTIMIZATION_TRIALS) -> Dict:
        """Optimize hyperparameters for multi-target models."""
        logger.info(f"Optimizing {model_type} hyperparameters for multi-target...")
        
        def objective(trial: Trial) -> float:
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'verbosity': 0
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
            
            # Multi-target cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                target_maes = []
                for i, target in enumerate(self.target_names):
                    if model_type == 'xgboost':
                        model = xgb.XGBRegressor(**params)
                    elif model_type == 'lightgbm':
                        model = lgb.LGBMRegressor(**params)
                    elif model_type == 'random_forest':
                        model = RandomForestRegressor(**params)
                    
                    model.fit(X_train, y_train[:, i])
                    y_pred = model.predict(X_val)
                    mae = mean_absolute_error(y_val[:, i], y_pred)
                    target_maes.append(mae)
                
                # Average MAE across all targets
                scores.append(np.mean(target_maes))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best {model_type} parameters: {study.best_params}")
        logger.info(f"Best {model_type} average MAE: {study.best_value:.3f}")
        
        return study.best_params
    
    def _evaluate_multi_target_model(self, models: Dict, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Evaluate multi-target models."""
        results = {'models': models}
        
        # Predictions for each target
        train_predictions = np.zeros((len(X_train), len(self.target_names)))
        test_predictions = np.zeros((len(X_test), len(self.target_names)))
        
        for i, target in enumerate(self.target_names):
            model = models[target]
            train_predictions[:, i] = model.predict(X_train)
            test_predictions[:, i] = model.predict(X_test)
        
        # Calculate metrics for each target
        for i, target in enumerate(self.target_names):
            results[f'{target}_train_mae'] = mean_absolute_error(y_train[:, i], train_predictions[:, i])
            results[f'{target}_test_mae'] = mean_absolute_error(y_test[:, i], test_predictions[:, i])
            results[f'{target}_train_rmse'] = np.sqrt(mean_squared_error(y_train[:, i], train_predictions[:, i]))
            results[f'{target}_test_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], test_predictions[:, i]))
            results[f'{target}_train_r2'] = r2_score(y_train[:, i], train_predictions[:, i])
            results[f'{target}_test_r2'] = r2_score(y_test[:, i], test_predictions[:, i])
        
        # Overall metrics (average across targets)
        results['train_mae'] = np.mean([results[f'{t}_train_mae'] for t in self.target_names])
        results['test_mae'] = np.mean([results[f'{t}_test_mae'] for t in self.target_names])
        results['train_rmse'] = np.mean([results[f'{t}_train_rmse'] for t in self.target_names])
        results['test_rmse'] = np.mean([results[f'{t}_test_rmse'] for t in self.target_names])
        results['train_r2'] = np.mean([results[f'{t}_train_r2'] for t in self.target_names])
        results['test_r2'] = np.mean([results[f'{t}_test_r2'] for t in self.target_names])
        
        return results
    
    def _create_ensemble_model(self, base_models: Dict, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Create ensemble by averaging predictions from base models."""
        ensemble_train_pred = np.zeros((len(X_train), len(self.target_names)))
        ensemble_test_pred = np.zeros((len(X_test), len(self.target_names)))
        
        # Average predictions across models
        for model_name, models in base_models.items():
            for i, target in enumerate(self.target_names):
                model = models[target]
                ensemble_train_pred[:, i] += model.predict(X_train)
                ensemble_test_pred[:, i] += model.predict(X_test)
        
        # Average across models
        ensemble_train_pred /= len(base_models)
        ensemble_test_pred /= len(base_models)
        
        # Create ensemble results
        results = {'models': base_models, 'ensemble': True}
        
        # Calculate metrics
        for i, target in enumerate(self.target_names):
            results[f'{target}_train_mae'] = mean_absolute_error(y_train[:, i], ensemble_train_pred[:, i])
            results[f'{target}_test_mae'] = mean_absolute_error(y_test[:, i], ensemble_test_pred[:, i])
            results[f'{target}_train_rmse'] = np.sqrt(mean_squared_error(y_train[:, i], ensemble_train_pred[:, i]))
            results[f'{target}_test_rmse'] = np.sqrt(mean_squared_error(y_test[:, i], ensemble_test_pred[:, i]))
            results[f'{target}_train_r2'] = r2_score(y_train[:, i], ensemble_train_pred[:, i])
            results[f'{target}_test_r2'] = r2_score(y_test[:, i], ensemble_test_pred[:, i])
        
        # Overall metrics
        results['train_mae'] = np.mean([results[f'{t}_train_mae'] for t in self.target_names])
        results['test_mae'] = np.mean([results[f'{t}_test_mae'] for t in self.target_names])
        results['train_rmse'] = np.mean([results[f'{t}_train_rmse'] for t in self.target_names])
        results['test_rmse'] = np.mean([results[f'{t}_test_rmse'] for t in self.target_names])
        results['train_r2'] = np.mean([results[f'{t}_train_r2'] for t in self.target_names])
        results['test_r2'] = np.mean([results[f'{t}_test_r2'] for t in self.target_names])
        
        return results
    
    def predict_multi_target(self, X: np.ndarray, model_name: str = 'ensemble') -> Dict[str, np.ndarray]:
        """Make multi-target predictions."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X_scaled = self.scaler.transform(X)
        model_data = self.models[model_name]
        
        predictions = {}
        
        if model_name == 'ensemble' or model_data.get('ensemble', False):
            # Ensemble prediction
            ensemble_pred = np.zeros((len(X), len(self.target_names)))
            base_models = model_data['models']
            
            for model_name, models in base_models.items():
                for i, target in enumerate(self.target_names):
                    model = models[target]
                    ensemble_pred[:, i] += model.predict(X_scaled)
            
            # Average across models
            ensemble_pred /= len(base_models)
            
            for i, target in enumerate(self.target_names):
                predictions[target] = ensemble_pred[:, i]
        else:
            # Individual model prediction
            models = model_data['models']
            for i, target in enumerate(self.target_names):
                model = models[target]
                predictions[target] = model.predict(X_scaled)
        
        return predictions
    
    def _log_results(self, results: Dict[str, Any]):
        """Log multi-target training results."""
        logger.info("=== Multi-Target Model Training Results ===")
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Overall - Test MAE: {metrics['test_mae']:.3f}, Test R²: {metrics['test_r2']:.3f}")
            for target in self.target_names:
                target_mae = metrics[f'{target}_test_mae']
                target_r2 = metrics[f'{target}_test_r2']
                logger.info(f"  {target}: MAE={target_mae:.3f}, R²={target_r2:.3f}")
    
    def save_models(self, filepath: str):
        """Save trained multi-target models."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'training_history': self.training_history,
            'model_type': 'multi_target'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Multi-target models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained multi-target models."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data.get('target_names', ['PTS', 'AST', 'REB'])
        self.training_history = model_data.get('training_history', {})
        
        logger.info(f"Multi-target models loaded from {filepath}")
