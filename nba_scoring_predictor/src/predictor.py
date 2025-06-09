# -*- coding: utf-8 -*-
"""
Main predictor class that coordinates all components
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from src.data_collector import NBADataCollector
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from utils.database import DatabaseManager
from utils.logger import main_logger as logger
from config.settings import DATABASE_PATH

class NBAPlayerScoringPredictor:
    """Main predictor class that orchestrates the entire pipeline."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize the predictor with all components."""
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_manager = DatabaseManager(db_path)
        self.data_collector = NBADataCollector(self.db_manager)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        self.raw_data = None
        self.processed_data = None
        self.is_trained = False
    
    def collect_data(self, player_names: List[str] = None, 
                    seasons: List[str] = None, 
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Collect player data for training or prediction.
        
        Args:
            player_names: List of player names to collect data for
            seasons: List of seasons to collect
            use_cache: Whether to use cached data
            
        Returns:
            Raw game log data
        """
        logger.info("Starting data collection...")
        
        self.raw_data = self.data_collector.collect_player_data(
            player_names=player_names,
            seasons=seasons,
            use_cache=use_cache
        )
        
        logger.info(f"Collected data for {self.raw_data['PLAYER_NAME'].nunique()} players")
        return self.raw_data
    
    def process_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process raw data through feature engineering pipeline.
        
        Args:
            data: Raw data to process (uses self.raw_data if None)
            
        Returns:
            Processed data with engineered features
        """
        if data is None:
            if self.raw_data is None:
                raise ValueError("No data available. Call collect_data() first.")
            data = self.raw_data
        
        logger.info("Processing data through feature engineering...")
        self.processed_data = self.feature_engineer.engineer_features(data)
        
        return self.processed_data
    
    def train(self, data: pd.DataFrame = None, optimize: bool = True) -> Dict:
        """
        Train the prediction models.
        
        Args:
            data: Processed data for training (uses self.processed_data if None)
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Training results
        """
        if data is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Call process_data() first.")
            data = self.processed_data
        
        logger.info("Preparing training data...")
        X, y, feature_names = self.feature_engineer.prepare_features(data)
        
        logger.info("Training models...")
        results = self.model_trainer.train_models(X, y, feature_names, optimize=optimize)
        
        self.is_trained = True
        logger.info("Training completed successfully!")
        
        return results
    
    def predict_player_points(self, player_name: str, recent_games: int = 10) -> Dict:
        """
        Predict points for a specific player based on recent performance.
        
        Args:
            player_name: Name of the player to predict for
            recent_games: Number of recent games to use for features
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        # Get recent data for the player
        player_data = self._get_player_recent_data(player_name, recent_games)
        
        if player_data.empty:
            raise ValueError(f"No recent data found for player: {player_name}")
        
        # Process the data
        processed_data = self.feature_engineer.engineer_features(player_data)
        
        if len(processed_data) == 0:
            raise ValueError(f"Insufficient data for prediction for player: {player_name}")
        
        # Get the most recent game's features
        latest_game = processed_data.iloc[-1:].copy()
        X, _, _ = self.feature_engineer.prepare_features(latest_game)
        
        # Scale features
        X_scaled = self.model_trainer.scaler.transform(X)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model_data in self.model_trainer.models.items():
            model = model_data['model']
            pred = model.predict(X_scaled)[0]
            
            # Calculate confidence interval (simplified approach)
            test_mae = model_data['test_mae']
            predictions[model_name] = {
                'predicted_points': max(0, pred),
                'confidence_interval': (max(0, pred - test_mae), pred + test_mae),
                'model_mae': test_mae
            }
        
        # Add recent performance context
        recent_avg = player_data['PTS'].tail(recent_games).mean()
        predictions['recent_average'] = recent_avg
        predictions['player_name'] = player_name
        
        return predictions
    
    def _get_player_recent_data(self, player_name: str, n_games: int) -> pd.DataFrame:
        """Get recent game data for a specific player."""
        if self.raw_data is None:
            # Try to collect data for this player
            try:
                data = self.data_collector.collect_player_data([player_name])
                return data.tail(n_games)
            except:
                return pd.DataFrame()
        
        # Filter from existing data
        player_data = self.raw_data[
            self.raw_data['PLAYER_NAME'].str.contains(player_name, case=False)
        ].copy()
        
        return player_data.tail(n_games)
    
    def get_model_performance(self) -> pd.DataFrame:
        """Get performance metrics for all trained models."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        performance_data = []
        for model_name, metrics in self.model_trainer.models.items():
            performance_data.append({
                'Model': model_name.title(),
                'Test MAE': round(metrics['test_mae'], 3),
                'Test RMSE': round(metrics['test_rmse'], 3),
                'Test R-squared': round(metrics['test_r2'], 3)
            })
        
        return pd.DataFrame(performance_data)
    
    def get_feature_importance(self, model_name: str = 'xgboost', top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        importance_df = self.model_trainer.get_feature_importance(model_name)
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save the trained model and all components."""
        if not self.is_trained:
            raise ValueError("No trained model to save. Call train() first.")
        
        self.model_trainer.save_models(filepath)
        logger.info(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model."""
        self.model_trainer.load_models(filepath)
        self.is_trained = True
        logger.info(f"Model loaded successfully from {filepath}")
    
    def get_available_players(self) -> List[str]:
        """Get list of players with data in the database."""
        players_df = self.db_manager.get_all_cached_players()
        return sorted(players_df['player_name'].tolist())