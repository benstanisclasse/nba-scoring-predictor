# Add this to src/predictor.py

# -*- coding: utf-8 -*-
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
        
        # Try to determine player's position
        player_position = self._get_player_position(player_name)
        
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
        player_data = self._get_player_recent_data(player_name, recent_games)
        
        if player_data.empty:
            raise ValueError(f"No recent data found for player: {player_name}")
        
        # Process the data using the appropriate feature engineer
        processed_data = feature_engineer.engineer_features(player_data)
        
        if len(processed_data) == 0:
            raise ValueError(f"Insufficient data for prediction for player: {player_name}")
        
        # Get the most recent game's features
        latest_game = processed_data.iloc[-1:].copy()
        
        # Ensure we have the same features as training
        if hasattr(model_trainer, 'feature_names') and model_trainer.feature_names:
            missing_features = set(model_trainer.feature_names) - set(latest_game.columns)
            for feature in missing_features:
                latest_game[feature] = 0
            
            # Select only the features used in training
            latest_game = latest_game[model_trainer.feature_names]
        
        X, _, _ = feature_engineer.prepare_features(latest_game)
        
        if X.shape[1] == 0:
            raise ValueError("No valid features generated for prediction")
        
        # Scale features
        X_scaled = model_trainer.scaler.transform(X)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model_data in model_trainer.models.items():
            model = model_data['model']
            pred = model.predict(X_scaled)[0]
            
            # Calculate confidence interval
            test_mae = model_data['test_mae']
            predictions[model_name] = {
                'predicted_points': max(0, pred),
                'confidence_interval': (max(0, pred - test_mae), pred + test_mae),
                'model_mae': test_mae
            }
        
        # Add context information
        recent_avg = player_data['PTS'].tail(recent_games).mean()
        predictions['recent_average'] = recent_avg
        predictions['player_name'] = player_name
        predictions['player_position'] = player_position
        predictions['model_used'] = f"{player_position}-specific" if player_position in self.position_models else "general"
        
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
    
    # Additional utility methods...
    def get_model_performance(self) -> pd.DataFrame:
        """Get performance metrics for all trained models."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_by_category() first.")
        
        performance_data = []
        for model_name, metrics in self.model_trainer.models.items():
            performance_data.append({
                'Model': model_name.title(),
                'Test MAE': round(metrics['test_mae'], 3),
                'Test RMSE': round(metrics['test_rmse'], 3),
                'Test R-squared': round(metrics['test_r2'], 3)
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


# TEAM PREDICTION SYSTEM
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