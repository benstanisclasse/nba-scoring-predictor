﻿# -*- coding: utf-8 -*-
"""
Enhanced predictor with position-based training and team predictions
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
from utils.nba_player_fetcher import NBAPlayerFetcher
from utils.player_roles import PlayerRoles
from config.settings import DATABASE_PATH

class EnhancedNBAPredictor:
    """Enhanced predictor with position-based training and team predictions."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize the enhanced predictor with all components."""
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_manager = DatabaseManager(db_path)
        self.data_collector = NBADataCollector(self.db_manager)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # Enhanced components
        self.player_fetcher = NBAPlayerFetcher()
        self.player_roles = PlayerRoles()
        
        # Initialize team_predictor as None - will be created when needed
        self.team_predictor = None
        
        self.raw_data = None
        self.processed_data = None
        self.is_trained = False
        
        # Position-specific models
        self.position_models = {}
        self.general_model = None
    
    def get_team_predictor(self):
        """Get or create team predictor."""
        if self.team_predictor is None:
            self.team_predictor = TeamGamePredictor(self)
        return self.team_predictor
    
    def get_feature_importance(self, model_name: str = 'xgboost', top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            model_name: Name of the model to get importance from ('xgboost', 'lightgbm', 'random_forest')
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            logger.warning("No trained model available for feature importance")
            return pd.DataFrame()
        
        try:
            # Check if we have position-specific models
            if hasattr(self, 'position_models') and self.position_models:
                # Use the first available position model
                first_position = list(self.position_models.keys())[0]
                model_trainer = self.position_models[first_position]['model_trainer']
                logger.info(f"Using {first_position}-specific model for feature importance")
            elif hasattr(self, 'general_model') and self.general_model:
                # Use general model
                model_trainer = self.general_model['model_trainer']
                logger.info("Using general model for feature importance")
            else:
                # Use default model trainer
                model_trainer = self.model_trainer
                logger.info("Using default model for feature importance")
            
            return model_trainer.get_feature_importance(model_name).head(top_n)
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()

    # ENHANCED TRAINING METHODS
    def train_by_category(self, category: str, max_players_per_position: int = None, 
                         seasons: List[str] = None, optimize: bool = True, 
                         use_cache: bool = True) -> Dict:
        """
        Train models on specific player categories.
        
        Args:
            category: 'PG', 'SG', 'SF', 'PF', 'C', 'Guards', 'Forwards', 'All'
            max_players_per_position: Maximum players per position (None = all)
            seasons: Seasons to use for training
            optimize: Whether to optimize hyperparameters
            use_cache: Whether to use cached data
            
        Returns:
            Training results
        """
        logger.info(f"Starting category-based training for: {category}")
        
        # Get players for the specified category
        target_players = self._get_players_by_category(category, max_players_per_position)
        
        if not target_players:
            raise ValueError(f"No players found for category: {category}")
        
        logger.info(f"Training on {len(target_players)} players in category '{category}'")
        
        # Collect data for these players
        data = self.collect_data(
            player_names=target_players,
            seasons=seasons,
            use_cache=use_cache
        )
        
        # Process data
        processed_data = self.process_data(data)
        
        # Train models
        results = self.train(processed_data, optimize=optimize)
        
        # Store the model for this category
        if category in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.position_models[category] = {
                'model_trainer': self.model_trainer,
                'feature_engineer': self.feature_engineer,
                'training_results': results,
                'players_trained': target_players
            }
        else:
            self.general_model = {
                'model_trainer': self.model_trainer,
                'feature_engineer': self.feature_engineer,
                'training_results': results,
                'players_trained': target_players,
                'category': category
            }
        
        self.is_trained = True
        logger.info(f"Category training completed for {category}")
        
        return results

    def collect_data_by_roles(self, roles: List[str], max_per_role: int = 3, seasons: List[str] = None, use_cache: bool = True) -> pd.DataFrame:
        """Collect data for players by their roles/positions."""
        all_players = []

        for role in roles:
            role_players = self._get_players_by_category(role, max_per_role)
            all_players.extend(role_players)

        return self.collect_data(
            player_names=all_players,
            seasons=seasons,
            use_cache=use_cache
        )

    def train_with_roles(self, data: pd.DataFrame, optimize: bool = True) -> Dict:
        """Train models with role-based data."""
        return self.train(data, optimize=optimize)

    def _get_players_by_category(self, category: str, max_per_position: int = None) -> List[str]:
        """Get player names for a specific category."""
        
        # Load NBA players data
        nba_data = self.player_fetcher.load_players_data()
        
        if not nba_data:
            logger.warning("No NBA data found, using fallback player list")
            return self._get_fallback_players_by_category(category)
        
        all_players = nba_data['all_players']
        selected_players = []
        
        if category == 'All':
            # All players regardless of position
            selected_players = [p['name'] for p in all_players]
            
        elif category == 'Guards':
            # All guard positions
            guard_players = [p['name'] for p in all_players if p['position'] in ['PG', 'SG']]
            selected_players = guard_players
            
        elif category == 'Forwards':
            # All forward positions  
            forward_players = [p['name'] for p in all_players if p['position'] in ['SF', 'PF']]
            selected_players = forward_players
            
        elif category == 'Bigs':
            # Centers and Power Forwards
            big_players = [p['name'] for p in all_players if p['position'] in ['PF', 'C']]
            selected_players = big_players
            
        elif category in ['PG', 'SG', 'SF', 'PF', 'C']:
            # Specific position
            position_players = [p['name'] for p in all_players if p['position'] == category]
            selected_players = position_players
            
        else:
            raise ValueError(f"Unknown category: {category}")
        
        # Limit number of players if specified
        if max_per_position and len(selected_players) > max_per_position:
            # Sort by some criteria (you could add team quality, games played, etc.)
            selected_players = sorted(selected_players)[:max_per_position]
        
        logger.info(f"Selected {len(selected_players)} players for category '{category}'")
        return selected_players
    
    def _get_fallback_players_by_category(self, category: str) -> List[str]:
        """Fallback player selection when NBA data unavailable."""
        
        fallback_players = {
            'PG': [
                'Stephen Curry', 'Luka Dončić', 'Damian Lillard', 'Ja Morant',
                'Trae Young', 'Chris Paul', 'Russell Westbrook', 'Kyrie Irving',
                'De\'Aaron Fox', 'Tyler Herro'
            ],
            'SG': [
                'Devin Booker', 'Donovan Mitchell', 'Jaylen Brown', 'Anthony Edwards',
                'DeMar DeRozan', 'CJ McCollum', 'Jordan Poole', 'Tyler Herro',
                'Jalen Green', 'Desmond Bane'
            ],
            'SF': [
                'LeBron James', 'Kevin Durant', 'Jayson Tatum', 'Jimmy Butler',
                'Kawhi Leonard', 'Paul George', 'Scottie Barnes', 'Franz Wagner',
                'Mikal Bridges', 'OG Anunoby'
            ],
            'PF': [
                'Giannis Antetokounmpo', 'Anthony Davis', 'Paolo Banchero', 'Evan Mobley',
                'Julius Randle', 'Tobias Harris', 'John Collins', 'Christian Wood',
                'Bobby Portis', 'PJ Washington'
            ],
            'C': [
                'Nikola Jokić', 'Joel Embiid', 'Victor Wembanyama', 'Bam Adebayo',
                'Rudy Gobert', 'Domantas Sabonis', 'Alperen Şengün', 'Jarrett Allen',
                'Myles Turner', 'Clint Capela'
            ]
        }
        
        if category == 'All':
            return [player for players_list in fallback_players.values() for player in players_list]
        elif category == 'Guards':
            return fallback_players['PG'] + fallback_players['SG']
        elif category == 'Forwards':
            return fallback_players['SF'] + fallback_players['PF']
        elif category == 'Bigs':
            return fallback_players['PF'] + fallback_players['C']
        elif category in fallback_players:
            return fallback_players[category]
        else:
            return fallback_players['PG']  # Default fallback
    
    
    def predict_player_points_enhanced(self, player_name: str, recent_games: int = 10) -> Dict:
        """
        Enhanced prediction that uses position-specific models when available.
        """
        if not self.is_trained:
            raise ValueError("No models trained. Call train_by_category() first.")
    
        logger.info(f"Starting prediction for {player_name} (recent games: {recent_games})")
    
        # Try to determine player's position
        player_position = self._get_player_position(player_name)
        logger.info(f"Player position: {player_position}")
    
        # Use position-specific model if available
        if player_position in self.position_models:
            logger.info(f"Using {player_position}-specific model for {player_name}")
            model_trainer = self.position_models[player_position]['model_trainer']
            feature_engineer = self.position_models[player_position]['feature_engineer']
        elif self.general_model:
            logger.info(f"Using general model for {player_name}")
            model_trainer = self.general_model['model_trainer']
            feature_engineer = self.general_model['feature_engineer']
        else:
            logger.info(f"Using default model for {player_name}")
            model_trainer = self.model_trainer
            feature_engineer = self.feature_engineer
    
        # Get recent data for the player
        logger.info(f"Fetching recent data for {player_name}...")
        player_data = self._get_player_recent_data(player_name, recent_games * 2)  # Get more data to be safe
    
        if player_data.empty:
            logger.error(f"No recent data found for player: {player_name}")
        
            # Try to get ANY data for this player
            all_player_data = self._get_player_recent_data(player_name, 100)  # Try last 100 games
            if all_player_data.empty:
                # Check if player exists in database at all
                available_players = self.get_available_players()
                similar_players = [p for p in available_players if player_name.lower() in p.lower() or p.lower() in player_name.lower()]
            
                if similar_players:
                    raise ValueError(f"No data found for '{player_name}'. Did you mean: {', '.join(similar_players[:3])}?")
                else:
                    raise ValueError(f"Player '{player_name}' not found in database. Available players: {len(available_players)} total.")
            else:
                raise ValueError(f"No recent data found for {player_name}. Last game was {all_player_data['GAME_DATE'].max()}")
    
        logger.info(f"Found {len(player_data)} games for {player_name}")
    
        # Add minimum data validation
        required_columns = ['PTS', 'GAME_DATE', 'PLAYER_NAME']
        missing_cols = [col for col in required_columns if col not in player_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {player_name}: {missing_cols}")
    
        # Ensure we have enough games with valid points
        valid_games = player_data[player_data['PTS'].notna() & (player_data['PTS'] >= 0)]
        if len(valid_games) < 3:
            raise ValueError(f"Insufficient valid games for {player_name}. Found {len(valid_games)} games with valid scoring data.")
    
        try:
            # Process the data using the appropriate feature engineer
            logger.info(f"Engineering features for {player_name}...")
            processed_data = feature_engineer.engineer_features(player_data)
        
            if len(processed_data) == 0:
                raise ValueError(f"Feature engineering produced no valid rows for {player_name}")
        
            logger.info(f"Generated {len(processed_data)} processed games with {processed_data.shape[1]} features")
        
        except Exception as e:
            logger.error(f"Feature engineering failed for {player_name}: {e}")
            raise ValueError(f"Feature engineering failed for {player_name}: {str(e)}")
    
        # Get the most recent game's features
        try:
            latest_game = processed_data.iloc[-1:].copy()
            logger.info(f"Using latest game data from {latest_game.get('GAME_DATE', 'unknown date').iloc[0] if 'GAME_DATE' in latest_game.columns else 'unknown date'}")
        
            # Ensure we have the same features as training - MORE ROBUST VERSION
            if hasattr(model_trainer, 'feature_names') and model_trainer.feature_names:
                logger.info(f"Model expects {len(model_trainer.feature_names)} features")
            
                # Add missing features with safe defaults
                missing_features = set(model_trainer.feature_names) - set(latest_game.columns)
                if missing_features:
                    logger.warning(f"Adding {len(missing_features)} missing features with default values")
                    for feature in missing_features:
                        if 'PCT' in feature.upper() or 'RATE' in feature.upper():
                            latest_game[feature] = 0.5  # Default percentage
                        elif 'ROLL' in feature.upper():
                            latest_game[feature] = player_data['PTS'].tail(5).mean() if not player_data.empty else 15.0
                        else:
                            latest_game[feature] = 0  # Default to 0
            
                # Select only the features used in training, in the correct order
                try:
                    latest_game = latest_game[model_trainer.feature_names]
                except KeyError as e:
                    logger.error(f"Feature mismatch for {player_name}. Model features: {model_trainer.feature_names[:5]}...")
                    logger.error(f"Available features: {list(latest_game.columns)[:5]}...")
                    raise ValueError(f"Feature mismatch: {str(e)}")
        
            # Alternative feature preparation if model doesn't have feature_names
            else:
                logger.warning("Model doesn't have feature_names, using feature_engineer prepare_features")
                X, _, feature_names = feature_engineer.prepare_features(latest_game)
                if X.shape[1] == 0:
                    raise ValueError("Feature engineering produced no valid features")
            
                # Convert back to DataFrame for consistency
                latest_game = pd.DataFrame(X, columns=feature_names)
        
            logger.info(f"Final feature matrix shape: {latest_game.shape}")
        
        except Exception as e:
            logger.error(f"Feature preparation failed for {player_name}: {e}")
            raise ValueError(f"Feature preparation failed: {str(e)}")
    
        # Convert to numpy array for prediction
        try:
            if hasattr(model_trainer, 'feature_names'):
                X = latest_game.values
            else:
                X, _, _ = feature_engineer.prepare_features(latest_game)
        
            if X.shape[1] == 0:
                raise ValueError("No valid features generated for prediction")
        
            logger.info(f"Prediction input shape: {X.shape}")
        
            # Scale features
            X_scaled = model_trainer.scaler.transform(X)
        
        except Exception as e:
            logger.error(f"Feature scaling failed for {player_name}: {e}")
            raise ValueError(f"Feature scaling failed: {str(e)}")
    
        # Make predictions with all models
        predictions = {}
        for model_name, model_data in model_trainer.models.items():
            try:
                model = model_data['model']
                pred = model.predict(X_scaled)[0]
            
                # Validate prediction
                from utils.validation import validate_player_prediction
                pred = validate_player_prediction(pred, player_name)
            
                # Calculate confidence interval
                test_mae = model_data['test_mae']
                predictions[model_name] = {
                    'predicted_points': max(0, pred),
                    'confidence_interval': (max(0, pred - test_mae), pred + test_mae),
                    'model_mae': test_mae
                }
            
                logger.info(f"{model_name} prediction: {pred:.1f} points")
            
            except Exception as e:
                logger.warning(f"Prediction failed for model {model_name}: {e}")
                # Skip this model but continue with others
                continue
    
        if not predictions:
            raise ValueError("All prediction models failed")
    
        # Add context information
        recent_avg = player_data['PTS'].tail(recent_games).mean()
        predictions['recent_average'] = recent_avg
        predictions['player_name'] = player_name
        predictions['player_position'] = player_position
        predictions['model_used'] = f"{player_position}-specific" if player_position in self.position_models else "general"
        predictions['games_analyzed'] = len(player_data)
    
        logger.info(f"Prediction completed successfully for {player_name}")
        return predictions

    def _get_player_position(self, player_name: str) -> str:
        """Get a player's position."""
        try:
            return self.player_roles.get_role(player_name)
        except:
            return "UNKNOWN"
    
    # Keep all existing methods from the original predictor...
    def collect_data(self, player_names: List[str] = None, 
                    seasons: List[str] = None, 
                    use_cache: bool = True) -> pd.DataFrame:
        """Collect player data for training or prediction."""
        logger.info("Starting data collection...")
        
        self.raw_data = self.data_collector.collect_player_data(
            player_names=player_names,
            seasons=seasons,
            use_cache=use_cache
        )
        
        logger.info(f"Collected data for {self.raw_data['PLAYER_NAME'].nunique()} players")
        return self.raw_data
    
    def process_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Process raw data through feature engineering pipeline."""
        if data is None:
            if self.raw_data is None:
                raise ValueError("No data available. Call collect_data() first.")
            data = self.raw_data
        
        logger.info("Processing data through feature engineering...")
        self.processed_data = self.feature_engineer.engineer_features(data)
        
        return self.processed_data
    
    def train(self, data: pd.DataFrame = None, optimize: bool = True) -> Dict:
        """Train the prediction models."""
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
    
    
    def _get_player_recent_data(self, player_name: str, n_games: int) -> pd.DataFrame:
        """Get recent game data for a specific player with enhanced search."""
        logger.info(f"Searching for recent data for '{player_name}' (last {n_games} games)")
    
        if self.raw_data is None:
            # Try to collect data for this player
            logger.info("No raw data available, attempting to collect fresh data...")
            try:
                data = self.data_collector.collect_player_data([player_name], use_cache=True)
                if not data.empty:
                    logger.info(f"Successfully collected {len(data)} games for {player_name}")
                    return data.tail(n_games)
                else:
                    logger.warning(f"No data collected for {player_name}")
            except Exception as e:
                logger.error(f"Failed to collect data for {player_name}: {e}")
        
            return pd.DataFrame()
    
        # Search with multiple strategies
        player_data = pd.DataFrame()
    
        # Strategy 1: Exact match
        exact_match = self.raw_data[self.raw_data['PLAYER_NAME'] == player_name].copy()
        if not exact_match.empty:
            logger.info(f"Found exact match: {len(exact_match)} games")
            player_data = exact_match
    
        # Strategy 2: Case-insensitive search
        if player_data.empty:
            case_insensitive = self.raw_data[
                self.raw_data['PLAYER_NAME'].str.lower() == player_name.lower()
            ].copy()
            if not case_insensitive.empty:
                logger.info(f"Found case-insensitive match: {len(case_insensitive)} games")
                player_data = case_insensitive
    
        # Strategy 3: Partial name match
        if player_data.empty:
            partial_match = self.raw_data[
                self.raw_data['PLAYER_NAME'].str.contains(player_name, case=False, na=False)
            ].copy()
            if not partial_match.empty:
                logger.info(f"Found partial match: {len(partial_match)} games")
                player_data = partial_match
    
        # Strategy 4: Try first/last name individually
        if player_data.empty and ' ' in player_name:
            name_parts = player_name.split()
            for part in name_parts:
                if len(part) > 2:  # Only try meaningful name parts
                    name_part_match = self.raw_data[
                        self.raw_data['PLAYER_NAME'].str.contains(part, case=False, na=False)
                    ].copy()
                    if not name_part_match.empty:
                        logger.info(f"Found match using name part '{part}': {len(name_part_match)} games")
                        player_data = name_part_match
                        break
    
        if player_data.empty:
            logger.error(f"No data found for '{player_name}' using any search strategy")
            # Log available players for debugging
            if 'PLAYER_NAME' in self.raw_data.columns:
                unique_players = self.raw_data['PLAYER_NAME'].unique()
                logger.info(f"Available players in database: {len(unique_players)} total")
                # Show similar names
                similar = [p for p in unique_players if any(part.lower() in p.lower() for part in player_name.split())]
                if similar:
                    logger.info(f"Similar player names found: {similar[:5]}")
        
            return pd.DataFrame()
    
        # Sort by date and return most recent games
        if 'GAME_DATE' in player_data.columns:
            player_data = player_data.sort_values('GAME_DATE')
    
        recent_data = player_data.tail(n_games)
        logger.info(f"Returning {len(recent_data)} most recent games for {player_name}")
    
        return recent_data

    def get_model_performance(self) -> pd.DataFrame:
        """Get performance metrics for all trained models."""
        if not self.is_trained:
            raise ValueError("Models not trained.")
    
        performance_data = []
        for model_name, metrics in self.model_trainer.models.items():
            performance_data.append({
                'Model': model_name.title(),
                'Test MAE': round(metrics['test_mae'], 3),
                'Test RMSE': round(metrics['test_rmse'], 3),
                'Test R': round(metrics['test_r2'], 3)  # Fixed column name
            })
    
        return pd.DataFrame(performance_data)
    
    def get_available_players(self) -> List[str]:
        """Get list of players with data in the database."""
        players_df = self.db_manager.get_all_cached_players()
        return sorted(players_df['player_name'].tolist())
    
    def save_model(self, filepath: str):
        """Save the trained model and all components."""
        if not self.is_trained:
            raise ValueError("No trained model to save. Call train_by_category() first.")
        
        self.model_trainer.save_models(filepath)
        logger.info(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model."""
        self.model_trainer.load_models(filepath)
        self.is_trained = True
        logger.info(f"Model loaded successfully from {filepath}")

    # Backward compatibility method
    def predict_player_points(self, player_name: str, recent_games: int = 10) -> Dict:
        """Backward compatibility wrapper for predict_player_points_enhanced."""
        return self.predict_player_points_enhanced(player_name, recent_games)
    
    # Add this method to EnhancedNBAPredictor class

    def get_prediction_confidence(self, player_name: str, prediction_data: Dict) -> Dict:
        """Calculate sophisticated confidence metrics."""
    
        ensemble_pred = prediction_data.get('ensemble', {})
        model_mae = ensemble_pred.get('model_mae', 5.0)
        recent_avg = prediction_data.get('recent_average', 15.0)
    
        # Confidence factors
        factors = {
            'model_accuracy': max(0, 1 - model_mae / 10),  # Better MAE = higher confidence
            'consistency': self._check_model_consistency(prediction_data),
            'data_quality': self._assess_data_quality(player_name),
            'volatility': self._calculate_player_volatility(player_name)
        }
    
        # Weighted confidence score
        confidence_score = (
            factors['model_accuracy'] * 0.4 +
            factors['consistency'] * 0.3 +
            factors['data_quality'] * 0.2 +
            (1 - factors['volatility']) * 0.1
        )
    
        return {
            'confidence_score': confidence_score,
            'confidence_grade': self._grade_confidence(confidence_score),
            'factors': factors,
            'recommendation': self._get_betting_recommendation(confidence_score, ensemble_pred)
        }

    def _check_model_consistency(self, prediction_data: Dict) -> float:
        """Check consistency across different models."""
        predictions = []
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'ensemble']:
            if model_name in prediction_data:
                predictions.append(prediction_data[model_name]['predicted_points'])
    
        if len(predictions) < 2:
            return 0.5
    
        variance = np.var(predictions)
        consistency = max(0, 1 - variance / 25)  # Lower variance = higher consistency
        return consistency

    def _assess_data_quality(self, player_name: str) -> float:
        """Assess data quality for the player."""
        # This would check recent games, missing data, etc.
        # For now, return a reasonable default
        return 0.8

    def _calculate_player_volatility(self, player_name: str) -> float:
        """Calculate player's scoring volatility."""
        # This would analyze recent game variance
        # For now, return a reasonable default
        return 0.3

    def _grade_confidence(self, score: float) -> str:
        """Convert confidence score to grade."""
        if score >= 0.8:
            return "Very High"
        elif score >= 0.65:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.35:
            return "Low"
        else:
            return "Very Low"

    def _get_betting_recommendation(self, confidence: float, prediction: Dict) -> str:
        """Get betting recommendation based on confidence and prediction."""
        if confidence >= 0.8:
            return "Strong recommendation - High confidence play"
        elif confidence >= 0.65:
            return "Good recommendation - Solid play"
        elif confidence >= 0.5:
            return "Moderate recommendation - Proceed with caution"
        else:
            return "Avoid - Low confidence prediction"
     
        # Add these methods to src/predictor.py

    def train_multi_target(self, data: pd.DataFrame = None, optimize: bool = True) -> Dict:
        """Train multi-target prediction models for points, assists, and rebounds."""
        if data is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Call process_data() first.")
            data = self.processed_data
    
        logger.info("Preparing multi-target training data...")
        X, y, feature_names, target_names = self.feature_engineer.prepare_multi_target_features(data)
    
        # Initialize multi-target trainer
        from src.multi_target_model_trainer import MultiTargetModelTrainer
        self.multi_target_trainer = MultiTargetModelTrainer()
    
        logger.info("Training multi-target models...")
        results = self.multi_target_trainer.train_models(X, y, feature_names, target_names, optimize=optimize)
    
        self.is_trained = True
        self.is_multi_target = True
        logger.info("Multi-target training completed successfully!")
    
        return results

    def predict_player_multi_stats(self, player_name: str, recent_games: int = 10) -> Dict:
        """
        Enhanced prediction for points, assists, and rebounds.
        """
        if not hasattr(self, 'multi_target_trainer') or not self.is_trained:
            raise ValueError("No multi-target models trained. Call train_multi_target() first.")
    
        logger.info(f"Making multi-target prediction for {player_name} (recent games: {recent_games})")
    
        # Get recent data for the player
        player_data = self._get_player_recent_data(player_name, recent_games * 2)
    
        if player_data.empty:
            raise ValueError(f"No recent data found for player: {player_name}")
    
        # Process the data
        processed_data = self.feature_engineer.engineer_features(player_data)
    
        if len(processed_data) == 0:
            raise ValueError(f"Feature engineering produced no valid rows for {player_name}")
    
        # Get the most recent game's features
        latest_game = processed_data.iloc[-1:].copy()
    
        # Ensure we have the same features as training
        if hasattr(self.multi_target_trainer, 'feature_names') and self.multi_target_trainer.feature_names:
            # Add missing features with defaults
            missing_features = set(self.multi_target_trainer.feature_names) - set(latest_game.columns)
            if missing_features:
                for feature in missing_features:
                    if 'PCT' in feature.upper() or 'RATE' in feature.upper():
                        latest_game[feature] = 0.5
                    elif 'ROLL' in feature.upper():
                        latest_game[feature] = player_data['PTS'].tail(5).mean() if not player_data.empty else 15.0
                    else:
                        latest_game[feature] = 0
        
            # Select only the features used in training
            latest_game = latest_game[self.multi_target_trainer.feature_names]
    
        # Convert to numpy array for prediction
        X = latest_game.values
    
        # Make predictions with all models
        predictions = {}
        target_names = self.multi_target_trainer.target_names
    
        for model_name, model_data in self.multi_target_trainer.models.items():
            try:
                model_predictions = self.multi_target_trainer.predict_multi_target(X, model_name)
            
                # Convert to prediction format for each target
                model_results = {}
                for target in target_names:
                    pred = model_predictions[target][0]  # Single prediction
                
                    # Validate prediction
                    from utils.validation import validate_player_prediction
                    if target == 'PTS':
                        pred = validate_player_prediction(pred, player_name)
                    elif target == 'AST':
                        pred = max(0, min(pred, 20))  # Reasonable assist range
                    elif target == 'REB':
                        pred = max(0, min(pred, 25))  # Reasonable rebound range
                
                    # Get model MAE for confidence intervals
                    target_mae_key = f'{target}_test_mae'
                    test_mae = model_data.get(target_mae_key, 2.0)
                
                    model_results[target] = {
                        'predicted_value': max(0, pred),
                        'confidence_interval': (max(0, pred - test_mae), pred + test_mae),
                        'model_mae': test_mae
                    }
            
                predictions[model_name] = model_results
            
            except Exception as e:
                logger.warning(f"Multi-target prediction failed for model {model_name}: {e}")
                continue
    
        if not predictions:
            raise ValueError("All multi-target prediction models failed")
    
        # Add context information
        recent_stats = {}
        for target in target_names:
            if target in player_data.columns:
                recent_stats[f'{target.lower()}_recent_average'] = player_data[target].tail(recent_games).mean()
            else:
               recent_stats[f'{target.lower()}_recent_average'] = 0.0

            predictions.update(recent_stats)
            predictions['player_name'] = player_name
            predictions['games_analyzed'] = len(player_data)
   
            logger.info(f"Multi-target prediction completed successfully for {player_name}")
            return predictions
    

# TEAM PREDICTION SYSTEM (keep the existing team prediction classes...)
class TeamGamePredictor:
    """Predict team vs team game outcomes using individual player predictions."""
    
    def __init__(self, player_predictor: EnhancedNBAPredictor):
        self.player_predictor = player_predictor
        self.team_features = TeamFeatureEngineer()
        self.matchup_analyzer = MatchupAnalyzer(player_predictor)
        
    def predict_game(self, team_a: str, team_b: str, game_context: Dict = None) -> Dict:
        """
        Predict game outcome between two teams.
        
        Args:
            team_a: First team name
            team_b: Second team name
            game_context: Optional game context (home team, date, etc.)
            
        Returns:
            Comprehensive game prediction
        """
        logger.info(f"Predicting game: {team_a} vs {team_b}")
        
        # Get team rosters and predict individual players
        team_a_predictions = self._get_team_predictions(team_a)
        team_b_predictions = self._get_team_predictions(team_b)
        
        # Calculate team totals
        team_a_total = sum(p['predicted_points'] for p in team_a_predictions.values())
        team_b_total = sum(p['predicted_points'] for p in team_b_predictions.values())
        
        # Calculate win probability based on point differential
        point_diff = team_a_total - team_b_total
        win_prob_a = self._calculate_win_probability(point_diff)
        
        # Add context adjustments
        if game_context:
            team_a_total, team_b_total, win_prob_a = self._adjust_for_context(
                team_a_total, team_b_total, win_prob_a, game_context
            )
        
        return {
            'team_a': team_a,
            'team_b': team_b,
            'winner_probability': {
                'team_a': win_prob_a,
                'team_b': 1 - win_prob_a
            },
            'predicted_score': {
                'team_a': round(team_a_total, 1),
                'team_b': round(team_b_total, 1)
            },
            'spread': round(team_a_total - team_b_total, 1),
            'total_points': round(team_a_total + team_b_total, 1),
            'confidence': self._calculate_confidence(team_a_predictions, team_b_predictions),
            'key_factors': self._get_key_factors(team_a_predictions, team_b_predictions),
            'team_breakdowns': {
                'team_a': team_a_predictions,
                'team_b': team_b_predictions
            }
        }
    
    def _get_team_predictions(self, team_name: str) -> Dict:
        """Get scoring predictions for all players on a team."""
        
        # Get team roster
        team_players = self._get_team_roster(team_name)
        
        team_predictions = {}
        
        for player_name in team_players:
            try:
                prediction = self.player_predictor.predict_player_points_enhanced(player_name)
                team_predictions[player_name] = {
                    'predicted_points': prediction['ensemble']['predicted_points'],
                    'position': prediction.get('player_position', 'UNKNOWN'),
                    'confidence': prediction['ensemble']['model_mae']
                }
            except Exception as e:
                logger.warning(f"Could not predict for {player_name}: {e}")
                # Use position average as fallback
                team_predictions[player_name] = {
                    'predicted_points': 10.0,  # League average fallback
                    'position': 'UNKNOWN',
                    'confidence': 5.0
                }
        
        return team_predictions
    
    def _get_team_roster(self, team_name: str) -> List[str]:
        """Get current roster for a team."""
        
        # Try to get from NBA data
        nba_data = self.player_predictor.player_fetcher.load_players_data()
        
        if nba_data:
            # Find team abbreviation
            team_abbrev = self._get_team_abbreviation(team_name)
            
            if team_abbrev and team_abbrev in nba_data['players_by_team']:
                return [p['name'] for p in nba_data['players_by_team'][team_abbrev]['players']]
        
        # Fallback to common rosters
        return self._get_fallback_roster(team_name)
    
    def _get_team_abbreviation(self, team_name: str) -> str:
        """Convert team name to abbreviation."""
        team_mapping = {
            'Los Angeles Lakers': 'LAL',
            'Golden State Warriors': 'GSW',
            'Boston Celtics': 'BOS',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Phoenix Suns': 'PHX',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Philadelphia 76ers': 'PHI',
            'Brooklyn Nets': 'BKN'
            # Add more mappings as needed
        }
        
        return team_mapping.get(team_name, team_name[:3].upper())
    
    def _get_fallback_roster(self, team_name: str) -> List[str]:
        """Fallback roster when live data unavailable."""
        
        fallback_rosters = {
            'Los Angeles Lakers': ['LeBron James', 'Anthony Davis', 'D\'Angelo Russell', 'Austin Reaves', 'Rui Hachimura'],
            'Golden State Warriors': ['Stephen Curry', 'Klay Thompson', 'Draymond Green', 'Andrew Wiggins', 'Jonathan Kuminga'],
            'Boston Celtics': ['Jayson Tatum', 'Jaylen Brown', 'Kristaps Porziņģis', 'Derrick White', 'Al Horford'],
            # Add more teams...
        }
        
        return fallback_rosters.get(team_name, ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5'])
    
    def _calculate_win_probability(self, point_diff: float) -> float:
        """Calculate win probability based on predicted point differential."""
        # Use logistic function - teams separated by ~15 points have 90% win probability
        import math
        return 1 / (1 + math.exp(-point_diff / 5))
    
    def _adjust_for_context(self, team_a_total: float, team_b_total: float, 
                           win_prob_a: float, context: Dict) -> Tuple[float, float, float]:
        """Adjust predictions based on game context."""
        
        # Home court advantage (~3 points)
        if context.get('home_team') == 'team_a':
            team_a_total += 3
        elif context.get('home_team') == 'team_b':
            team_b_total += 3
        
        # Rest advantage
        rest_diff = context.get('rest_differential', 0)
        if rest_diff > 0:  # team_a more rested
            team_a_total += min(rest_diff * 0.5, 2)
        else:
            team_b_total += min(abs(rest_diff) * 0.5, 2)
        
        # Recalculate win probability
        new_diff = team_a_total - team_b_total
        win_prob_a = self._calculate_win_probability(new_diff)
        
        return team_a_total, team_b_total, win_prob_a
    
    def _calculate_confidence(self, team_a_preds: Dict, team_b_preds: Dict) -> float:
        """Calculate overall prediction confidence."""
        
        all_confidences = []
        
        for pred in team_a_preds.values():
            all_confidences.append(1 / pred['confidence'])  # Lower MAE = higher confidence
        
        for pred in team_b_preds.values():
            all_confidences.append(1 / pred['confidence'])
        
        avg_confidence = sum(all_confidences) / len(all_confidences)
        return min(avg_confidence, 1.0)  # Cap at 1.0
    
    def _get_key_factors(self, team_a_preds: Dict, team_b_preds: Dict) -> List[str]:
        """Identify key factors in the matchup."""
        
        factors = []
        
        # Find best players on each team
        team_a_best = max(team_a_preds.items(), key=lambda x: x[1]['predicted_points'])
        team_b_best = max(team_b_preds.items(), key=lambda x: x[1]['predicted_points'])
        
        factors.append(f"Top scorers: {team_a_best[0]} ({team_a_best[1]['predicted_points']:.1f}) vs {team_b_best[0]} ({team_b_best[1]['predicted_points']:.1f})")
        
        # Team totals comparison
        team_a_total = sum(p['predicted_points'] for p in team_a_preds.values())
        team_b_total = sum(p['predicted_points'] for p in team_b_preds.values())
        
        if team_a_total > team_b_total:
            factors.append(f"Team A projected for {team_a_total - team_b_total:.1f} point advantage")
        else:
            factors.append(f"Team B projected for {team_b_total - team_a_total:.1f} point advantage")
        
        return factors


class TeamFeatureEngineer:
    """Feature engineering for team-level analysis."""
    
    def __init__(self):
        pass
    
    def engineer_team_features(self, team: str, season: str) -> Dict:
        """Engineer features for a specific team."""
        # This would collect team-level statistics
        # For now, placeholder implementation
        return {
            'team_name': team,
            'season': season,
            'avg_points': 110.0,  # Would be calculated from actual data
            'avg_allowed': 105.0,
            'pace': 98.5,
            'net_rating': 5.0
        }


class MatchupAnalyzer:
    """Analyze team matchups."""
    
    def __init__(self, player_predictor: EnhancedNBAPredictor):
        self.player_predictor = player_predictor
    
    def analyze_matchup(self, team_a: str, team_b: str) -> Dict:
        """Analyze specific matchup between two teams."""
        
        # This would include head-to-head analysis, style matchups, etc.
        return {
            'pace_differential': 0.0,
            'style_matchup': 'neutral',
            'historical_advantage': team_a
        }