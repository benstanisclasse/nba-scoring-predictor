# -*- coding: utf-8 -*-
"""
Model training module with hyperparameter optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import optuna
from optuna import Trial

from config.settings import MODEL_CONFIGS, OPTIMIZATION_TRIALS
from utils.logger import main_logger as logger

class ModelTrainer:
    """Handles model training and hyperparameter optimization."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.training_history = {}
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                model_type: str, n_trials: int = OPTIMIZATION_TRIALS) -> Dict:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target values
            model_type: Type of model to optimize
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Optimizing {model_type} hyperparameters...")
        
        def objective(trial: Trial) -> float:
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                scores.append(mae)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best {model_type} parameters: {study.best_params}")
        logger.info(f"Best {model_type} MAE: {study.best_value:.3f}")
        
        return study.best_params
    
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                    optimize: bool = True) -> Dict[str, Any]:
        """
        Train ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training models...")
        
        self.feature_names = feature_names
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Chronological split (80/20)
        split_point = int(0.8 * len(X))
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        results = {}
        
        # XGBoost
        if optimize:
            xgb_params = self.optimize_hyperparameters(X_train, y_train, 'xgboost', 30)
        else:
            xgb_params = {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': 0
            }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        results['xgboost'] = self._evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
        
        # LightGBM
        if optimize:
            lgb_params = self.optimize_hyperparameters(X_train, y_train, 'lightgbm', 30)
        else:
            lgb_params = {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'num_leaves': 50, 'random_state': 42, 'verbose': -1
            }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train)
        results['lightgbm'] = self._evaluate_model(lgb_model, X_train, X_test, y_train, y_test)
        
        # Random Forest
        if optimize:
            rf_params = self.optimize_hyperparameters(X_train, y_train, 'random_forest', 20)
        else:
            rf_params = {
                'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5,
                'min_samples_leaf': 2, 'random_state': 42
            }
        
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        results['random_forest'] = self._evaluate_model(rf_model, X_train, X_test, y_train, y_test)
        
        # Neural Network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        results['neural_network'] = self._evaluate_model(nn_model, X_train, X_test, y_train, y_test)
        
        # Ensemble (Voting Regressor)
        ensemble_models = [
            ('xgboost', xgb_model),
            ('lightgbm', lgb_model),
            ('random_forest', rf_model)
        ]
        ensemble = VotingRegressor(ensemble_models)
        ensemble.fit(X_train, y_train)
        results['ensemble'] = self._evaluate_model(ensemble, X_train, X_test, y_train, y_test)
        
        self.models = results
        self.training_history = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(feature_names),
            'optimization_used': optimize
        }
        
        # Log results
        self._log_results(results)
        
        return results
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Evaluate a single model and return metrics."""
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        return {
            'model': model,
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
    
    def _log_results(self, results: Dict[str, Any]):
        """Log training results."""
        logger.info("=== Model Training Results ===")
        for model_name, metrics in results.items():
            logger.info(
                f"{model_name.upper()}: "
                f"Test MAE={metrics['test_mae']:.3f}, "
                f"Test RMSE={metrics['test_rmse']:.3f}, "
                f"Test R-squared={metrics['test_r2']:.3f}"
            )
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> pd.DataFrame:
        """Get feature importance from specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()
    
    def save_models(self, filepath: str):
        """Save trained models and preprocessing objects."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models and preprocessing objects."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data.get('training_history', {})
        
        logger.info(f"Models loaded from {filepath}")
