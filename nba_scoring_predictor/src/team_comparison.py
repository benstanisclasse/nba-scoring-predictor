# -*- coding: utf-8 -*-
"""
Enhanced Team Comparison Engine for NBA Game Predictions - FIXED VERSION
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import math
from datetime import datetime, timedelta

from utils.logger import main_logger as logger
from utils.nba_player_fetcher import NBAPlayerFetcher
from utils.player_roles import PlayerRoles

class EnhancedTeamComparison:
    """Enhanced team comparison with advanced metrics and analysis."""
    
    def __init__(self, player_predictor):
        self.player_predictor = player_predictor
        self.player_fetcher = NBAPlayerFetcher()
        self.player_roles = PlayerRoles()
        
        # Advanced team metrics calculator
        self.team_analyzer = AdvancedTeamAnalyzer(player_predictor)
        self.matchup_engine = MatchupAnalysisEngine()
        self.monte_carlo = MonteCarloSimulator()
        
    def compare_teams_comprehensive(self, team_a: str, team_b: str, 
                                  game_context: Dict = None) -> Dict:
        """
        Comprehensive team comparison with multiple methodologies.
        
        Args:
            team_a: First team name
            team_b: Second team name  
            game_context: Game context (home/away, rest, injuries, etc.)
            
        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Running comprehensive team comparison: {team_a} vs {team_b}")
        
        # 1. Get team rosters and individual predictions
        team_a_data = self._get_enhanced_team_data(team_a)
        team_b_data = self._get_enhanced_team_data(team_b)
        
        # 2. Calculate advanced team metrics
        team_a_metrics = self.team_analyzer.calculate_team_metrics(team_a_data)
        team_b_metrics = self.team_analyzer.calculate_team_metrics(team_b_data)
        
        # 3. Perform matchup analysis
        matchup_analysis = self.matchup_engine.analyze_matchup(
            team_a_data, team_b_data, team_a_metrics, team_b_metrics
        )
        
        # 4. Multiple prediction methods
        predictions = self._generate_ensemble_predictions(
            team_a_data, team_b_data, team_a_metrics, team_b_metrics, 
            matchup_analysis, game_context
        )
        
        # 5. Monte Carlo simulation
        monte_carlo_results = self.monte_carlo.simulate_game(
            team_a_data, team_b_data, predictions, n_simulations=10000
        )
        
        # 6. Confidence and uncertainty analysis
        confidence_analysis = self._analyze_prediction_confidence(
            team_a_data, team_b_data, predictions, monte_carlo_results
        )
        
        return {
            'teams': {
                'team_a': team_a,
                'team_b': team_b
            },
            'team_metrics': {
                'team_a': team_a_metrics,
                'team_b': team_b_metrics
            },
            'matchup_analysis': matchup_analysis,
            'predictions': predictions,
            'monte_carlo': monte_carlo_results,
            'confidence_analysis': confidence_analysis,
            'game_context': game_context or {},
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_enhanced_team_data(self, team_name: str) -> Dict:
        """Get comprehensive team data including player predictions."""
        team_roster = self._get_team_roster(team_name)
        
        team_data = {
            'team_name': team_name,
            'players': {},
            'roster_stats': {},
            'depth_analysis': {}
        }
        
        # Get individual player predictions and analysis
        for player_name in team_roster:
            try:
                prediction = self.player_predictor.predict_player_points_enhanced(player_name)
                player_position = self.player_roles.get_role(player_name)
                
                # Enhanced player data
                player_data = {
                    'name': player_name,
                    'position': player_position,
                    'predicted_points': prediction['ensemble']['predicted_points'],
                    'confidence_interval': prediction['ensemble']['confidence_interval'],
                    'model_mae': prediction['ensemble']['model_mae'],
                    'recent_average': prediction.get('recent_average', 0),
                    'usage_tier': self._determine_usage_tier(prediction['ensemble']['predicted_points']),
                    'playing_time_estimate': self._estimate_playing_time(player_position, prediction['ensemble']['predicted_points'])
                }
                
                team_data['players'][player_name] = player_data
                
            except Exception as e:
                logger.warning(f"Could not get prediction for {player_name}: {e}")
                # Fallback data
                team_data['players'][player_name] = {
                    'name': player_name,
                    'position': 'UNKNOWN',
                    'predicted_points': 8.0,  # Conservative estimate
                    'confidence_interval': (4.0, 12.0),
                    'model_mae': 5.0,
                    'recent_average': 8.0,
                    'usage_tier': 'bench',
                    'playing_time_estimate': 15.0
                }
        
        # Calculate roster statistics
        team_data['roster_stats'] = self._calculate_roster_stats(team_data['players'])
        team_data['depth_analysis'] = self._analyze_team_depth(team_data['players'])
        
        return team_data
    
    def _generate_ensemble_predictions(self, team_a_data: Dict, team_b_data: Dict,
                                     team_a_metrics: Dict, team_b_metrics: Dict,
                                     matchup_analysis: Dict, context: Dict) -> Dict:
        """Generate predictions using multiple methodologies."""
        
        predictions = {}
        
        # Method 1: Direct aggregation
        predictions['direct_aggregation'] = self._predict_direct_aggregation(
            team_a_data, team_b_data
        )
        
        # Method 2: Possession-based model
        predictions['possession_based'] = self._predict_possession_based(
            team_a_data, team_b_data, team_a_metrics, team_b_metrics
        )
        
        # Method 3: Matchup-adjusted model
        predictions['matchup_adjusted'] = self._predict_matchup_adjusted(
            team_a_data, team_b_data, matchup_analysis
        )
        
        # Method 4: Context-adjusted model
        predictions['context_adjusted'] = self._predict_context_adjusted(
            predictions['direct_aggregation'], context
        )
        
        # Ensemble prediction
        predictions['ensemble'] = self._create_ensemble_prediction(predictions)
        
        return predictions
    
    def _predict_direct_aggregation(self, team_a_data: Dict, team_b_data: Dict) -> Dict:
        """Simple aggregation of individual player predictions."""
        
        team_a_total = sum(
            player['predicted_points'] * (player['playing_time_estimate'] / 48)
            for player in team_a_data['players'].values()
        )
        
        team_b_total = sum(
            player['predicted_points'] * (player['playing_time_estimate'] / 48)
            for player in team_b_data['players'].values()
        )
        
        # Calculate win probability
        point_diff = team_a_total - team_b_total
        win_prob_a = self._calculate_win_probability(point_diff)
        
        return {
            'team_a_score': round(team_a_total, 1),
            'team_b_score': round(team_b_total, 1),
            'win_probability_a': win_prob_a,
            'win_probability_b': 1 - win_prob_a,
            'spread': round(point_diff, 1),
            'total': round(team_a_total + team_b_total, 1),
            'method': 'direct_aggregation'
        }
    
    def _predict_possession_based(self, team_a_data: Dict, team_b_data: Dict,
                                team_a_metrics: Dict, team_b_metrics: Dict) -> Dict:
        """Possession-based prediction model."""
        
        # Estimate game pace
        team_a_pace = team_a_metrics.get('estimated_pace', 98.0)
        team_b_pace = team_b_metrics.get('estimated_pace', 98.0)
        game_pace = (team_a_pace + team_b_pace) / 2
        
        # Estimate possessions per game
        possessions = game_pace * 48 / 48  # Normalized to 48 minutes
        
        # Calculate offensive and defensive ratings
        team_a_off_rating = team_a_metrics.get('offensive_rating', 110.0)
        team_b_off_rating = team_b_metrics.get('offensive_rating', 110.0)
        team_a_def_rating = team_a_metrics.get('defensive_rating', 110.0)
        team_b_def_rating = team_b_metrics.get('defensive_rating', 110.0)
        
        # Adjust ratings based on opponent
        team_a_adj_off = (team_a_off_rating + team_b_def_rating) / 2
        team_b_adj_off = (team_b_off_rating + team_a_def_rating) / 2
        
        # Calculate expected scores
        team_a_score = (possessions * team_a_adj_off) / 100
        team_b_score = (possessions * team_b_adj_off) / 100
        
        point_diff = team_a_score - team_b_score
        win_prob_a = self._calculate_win_probability(point_diff)
        
        return {
            'team_a_score': round(team_a_score, 1),
            'team_b_score': round(team_b_score, 1),
            'win_probability_a': win_prob_a,
            'win_probability_b': 1 - win_prob_a,
            'spread': round(point_diff, 1),
            'total': round(team_a_score + team_b_score, 1),
            'estimated_pace': round(game_pace, 1),
            'possessions': round(possessions, 1),
            'method': 'possession_based'
        }
    
    def _calculate_win_probability(self, point_diff: float) -> float:
        """Calculate win probability from point differential."""
        # Logistic function calibrated to NBA data
        # Teams with 15-point advantage have ~90% win probability
        return 1 / (1 + math.exp(-point_diff / 5.0))
    
    def _determine_usage_tier(self, predicted_points: float) -> str:
        """Determine player usage tier based on predicted points."""
        if predicted_points >= 20:
            return 'star'
        elif predicted_points >= 15:
            return 'starter'
        elif predicted_points >= 10:
            return 'rotation'
        else:
            return 'bench'
    
    def _estimate_playing_time(self, position: str, predicted_points: float) -> float:
        """Estimate playing time based on position and scoring."""
        base_minutes = {
            'PG': 32, 'SG': 30, 'SF': 32, 'PF': 28, 'C': 26, 'UNKNOWN': 25
        }
        
        base = base_minutes.get(position, 25)
        
        # Adjust based on predicted scoring
        if predicted_points >= 20:
            return min(base + 6, 42)
        elif predicted_points >= 15:
            return min(base + 2, 38)
        elif predicted_points >= 10:
            return base
        else:
            return max(base - 10, 12)

    def _predict_matchup_adjusted(self, team_a_data: Dict, team_b_data: Dict, 
                           matchup_analysis: Dict) -> Dict:
       """Prediction adjusted for specific matchups."""
       # Start with direct aggregation
       base_prediction = self._predict_direct_aggregation(team_a_data, team_b_data)
   
       team_a_score = base_prediction['team_a_score']
       team_b_score = base_prediction['team_b_score']
   
       # Apply matchup adjustments
       positional_advantages = matchup_analysis['positional_advantages']
   
       # Adjust scores based on positional advantages
       for pos, matchup in positional_advantages.items():
           advantage_size = abs(matchup['point_differential'])
       
           if matchup['advantage'] == 'team_a' and advantage_size > 3:
               team_a_score += advantage_size * 0.15  # 15% of advantage
           elif matchup['advantage'] == 'team_b' and advantage_size > 3:
               team_b_score += advantage_size * 0.15
   
       # Pace adjustments
       pace_matchup = matchup_analysis['pace_matchup']
       if pace_matchup['pace_advantage'] == 'team_a':
           team_a_score += 2.0  # Faster team gets slight advantage
       elif pace_matchup['pace_advantage'] == 'team_b':
           team_b_score += 2.0
   
       # Recalculate win probability
       point_diff = team_a_score - team_b_score
       win_prob_a = self._calculate_win_probability(point_diff)
   
       return {
           'team_a_score': round(team_a_score, 1),
           'team_b_score': round(team_b_score, 1),
           'win_probability_a': win_prob_a,
           'win_probability_b': 1 - win_prob_a,
           'spread': round(point_diff, 1),
           'total': round(team_a_score + team_b_score, 1),
           'method': 'matchup_adjusted'
       }

    def _predict_context_adjusted(self, base_prediction: Dict, context: Dict) -> Dict:
       """Apply context adjustments to predictions."""
       team_a_score = base_prediction['team_a_score']
       team_b_score = base_prediction['team_b_score']
   
       # Home court advantage
       if context.get('home_team') == 'team_a':
           team_a_score += context.get('home_court_advantage', 3.0)
       elif context.get('home_team') == 'team_b':
           team_b_score += context.get('home_court_advantage', 3.0)
   
       # Rest differential
       rest_diff = context.get('rest_differential', 0)
       if rest_diff > 0:  # Team A more rested
           team_a_score += min(rest_diff * 1.5, 3.0)
       elif rest_diff < 0:  # Team B more rested
           team_b_score += min(abs(rest_diff) * 1.5, 3.0)
   
       # Recalculate win probability
       point_diff = team_a_score - team_b_score
       win_prob_a = self._calculate_win_probability(point_diff)
   
       return {
           'team_a_score': round(team_a_score, 1),
           'team_b_score': round(team_b_score, 1),
           'win_probability_a': win_prob_a,
           'win_probability_b': 1 - win_prob_a,
           'spread': round(point_diff, 1),
           'total': round(team_a_score + team_b_score, 1),
           'method': 'context_adjusted'
       }

    def _create_ensemble_prediction(self, predictions: Dict) -> Dict:
       """Create ensemble prediction from multiple methods."""
       methods = ['direct_aggregation', 'possession_based', 'matchup_adjusted', 'context_adjusted']
   
       # Weight the different methods
       weights = {
           'direct_aggregation': 0.3,
           'possession_based': 0.3,
           'matchup_adjusted': 0.2,
           'context_adjusted': 0.2
       }
   
       team_a_score = 0
       team_b_score = 0
   
       for method in methods:
           if method in predictions:
               weight = weights[method]
               team_a_score += predictions[method]['team_a_score'] * weight
               team_b_score += predictions[method]['team_b_score'] * weight
   
       # Calculate ensemble win probability
       point_diff = team_a_score - team_b_score
       win_prob_a = self._calculate_win_probability(point_diff)
   
       return {
           'team_a_score': round(team_a_score, 1),
           'team_b_score': round(team_b_score, 1),
           'win_probability_a': win_prob_a,
           'win_probability_b': 1 - win_prob_a,
           'spread': round(point_diff, 1),
           'total': round(team_a_score + team_b_score, 1),
           'method': 'ensemble',
           'methodology': 'Weighted average of multiple prediction methods'
       }

    def _calculate_roster_stats(self, players: Dict) -> Dict:
       """Calculate overall roster statistics."""
       if not players:
           return {}
   
       point_values = [p['predicted_points'] for p in players.values()]
       confidence_values = [p['model_mae'] for p in players.values()]
   
       return {
           'total_players': len(players),
           'avg_predicted_points': round(np.mean(point_values), 1),
           'std_predicted_points': round(np.std(point_values), 1),
           'max_predicted_points': round(max(point_values), 1),
           'min_predicted_points': round(min(point_values), 1),
           'avg_confidence': round(np.mean(confidence_values), 2),
           'total_predicted_points': round(sum(point_values), 1)
       }

    def _analyze_team_depth(self, players: Dict) -> Dict:
       """Analyze team depth and rotation strength."""
       if not players:
           return {}
   
       # Sort players by predicted points
       sorted_players = sorted(players.values(), key=lambda x: x['predicted_points'], reverse=True)
   
       # Define tiers
       starters = sorted_players[:5]
       rotation = sorted_players[5:10] if len(sorted_players) > 5 else []
       bench = sorted_players[10:] if len(sorted_players) > 10 else []
   
       starter_points = sum(p['predicted_points'] for p in starters)
       rotation_points = sum(p['predicted_points'] for p in rotation)
       bench_points = sum(p['predicted_points'] for p in bench)
   
       return {
           'starter_strength': round(starter_points, 1),
           'rotation_strength': round(rotation_points, 1),
           'bench_strength': round(bench_points, 1),
           'depth_ratio': round(rotation_points / starter_points, 3) if starter_points > 0 else 0,
           'total_depth': round((rotation_points + bench_points) / starter_points, 3) if starter_points > 0 else 0,
           'star_players': len([p for p in sorted_players if p['predicted_points'] >= 20]),
           'quality_depth': len([p for p in sorted_players[5:] if p['predicted_points'] >= 10])
       }

    def _analyze_prediction_confidence(self, team_a_data: Dict, team_b_data: Dict,
                                    predictions: Dict, monte_carlo_results: Dict) -> Dict:
       """Analyze overall prediction confidence and uncertainty."""
   
       # Calculate model uncertainty from individual player predictions
       team_a_uncertainties = [p['model_mae'] for p in team_a_data['players'].values()]
       team_b_uncertainties = [p['model_mae'] for p in team_b_data['players'].values()]
   
       avg_uncertainty = np.mean(team_a_uncertainties + team_b_uncertainties)
   
       # Calculate prediction consistency across methods
       ensemble_score = predictions['ensemble']['team_a_score']
       method_scores = [pred['team_a_score'] for method, pred in predictions.items() if method != 'ensemble']
       score_variance = np.var(method_scores) if method_scores else 0
   
       # Overall confidence score (0-1)
       uncertainty_factor = 1 / (1 + avg_uncertainty / 4)  # Lower MAE = higher confidence
       consistency_factor = 1 / (1 + score_variance / 25)  # Lower variance = higher confidence
   
       overall_confidence = (uncertainty_factor + consistency_factor) / 2
   
       # Identify uncertainty factors
       uncertainty_factors = []
   
       if avg_uncertainty > 5:
           uncertainty_factors.append("High individual prediction uncertainty")
   
       if score_variance > 16:
           uncertainty_factors.append("Inconsistent predictions across methods")
   
       if monte_carlo_results['margin_analysis']['margin_std'] > 10:
           uncertainty_factors.append("High game outcome variance")
   
       # Check for close game
       win_prob_diff = abs(predictions['ensemble']['win_probability_a'] - 0.5)
       if win_prob_diff < 0.1:
           uncertainty_factors.append("Very close game - small factors could decide outcome")
   
       if not uncertainty_factors:
           uncertainty_factors.append("No major uncertainty factors identified")
   
       return {
           'overall_confidence': round(overall_confidence, 3),
           'model_uncertainty': round(avg_uncertainty, 2),
           'prediction_consistency': round(1 - score_variance / 25, 3),
           'uncertainty_factors': uncertainty_factors,
           'confidence_grade': 'High' if overall_confidence > 0.8 else 
                             'Medium' if overall_confidence > 0.6 else 'Low'
       }

    def _get_team_roster(self, team_name: str) -> List[str]:
       """Get current roster for a team."""
       # Try to get from NBA data
       nba_data = self.player_fetcher.load_players_data()
   
       if nba_data:
           # Find team abbreviation
           team_abbrev = self._get_team_abbreviation(team_name)
       
           if team_abbrev and team_abbrev in nba_data.get('players_by_team', {}):
               return [p['name'] for p in nba_data['players_by_team'][team_abbrev]['players']]
   
       # Fallback to common rosters
       return self._get_fallback_roster(team_name)

    def _get_team_abbreviation(self, team_name: str) -> str:
       """Convert team name to abbreviation."""
       team_mapping = {
           'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
           'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
           'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
           'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
           'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
           'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
           'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
           'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
           'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
           'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
       }
   
       return team_mapping.get(team_name, team_name[:3].upper())

    def _get_fallback_roster(self, team_name: str) -> List[str]:
       """Fallback roster when live data unavailable."""
       fallback_rosters = {
           'Los Angeles Lakers': [
               'LeBron James', 'Anthony Davis', "D'Angelo Russell", 'Austin Reaves', 
               'Rui Hachimura', 'Taurean Prince', 'Christian Wood', 'Gabe Vincent',
               'Jarred Vanderbilt', 'Cam Reddish', 'Max Christie', 'Jalen Hood-Schifino'
           ],
           'Golden State Warriors': [
               'Stephen Curry', 'Klay Thompson', 'Draymond Green', 'Andrew Wiggins',
               'Jonathan Kuminga', 'Moses Moody', 'Kevon Looney', 'Chris Paul',
               'Gary Payton II', 'Brandin Podziemski', 'Trayce Jackson-Davis', 'Cory Joseph'
           ],
           'Boston Celtics': [
               'Jayson Tatum', 'Jaylen Brown', 'Kristaps Porziņģis', 'Derrick White',
               'Al Horford', 'Malcolm Brogdon', 'Robert Williams III', 'Grant Williams',
               'Payton Pritchard', 'Sam Hauser', 'Luke Kornet', 'Dalano Banton'
           ],
           'Miami Heat': [
               'Jimmy Butler', 'Bam Adebayo', 'Tyler Herro', 'Kyle Lowry',
               'Duncan Robinson', 'Terry Rozier', 'Kevin Love', 'Caleb Martin',
               'Nikola Jović', 'Jaime Jaquez Jr.', 'Thomas Bryant', 'Josh Richardson'
           ],
           'Milwaukee Bucks': [
               'Giannis Antetokounmpo', 'Damian Lillard', 'Khris Middleton', 'Brook Lopez',
               'Bobby Portis', 'Malik Beasley', 'Pat Connaughton', 'Jae Crowder',
               'MarJon Beauchamp', 'AJ Green', 'Andre Jackson Jr.', 'Robin Lopez'
           ],
           'Phoenix Suns': [
               'Kevin Durant', 'Devin Booker', 'Bradley Beal', 'Jusuf Nurkić',
               'Grayson Allen', 'Eric Gordon', 'Drew Eubanks', 'Keita Bates-Diop',
               'Yuta Watanabe', 'Josh Okogie', 'Damion Lee', 'Bol Bol'
           ],
           'Dallas Mavericks': [
               'Luka Dončić', 'Kyrie Irving', 'Christian Wood', 'Tim Hardaway Jr.',
               'Maxi Kleber', 'Derrick Jones Jr.', 'Josh Green', 'Dwight Powell',
               'Jaden Hardy', 'Markieff Morris', 'Richaun Holmes', 'Dante Exum'
           ],
           'Denver Nuggets': [
               'Nikola Jokić', 'Jamal Murray', 'Michael Porter Jr.', 'Aaron Gordon',
               'Kentavious Caldwell-Pope', 'Christian Braun', 'Reggie Jackson',
               'DeAndre Jordan', 'Peyton Watson', 'Julian Strawther', 'Zeke Nnaji', 'Justin Holiday'
           ]
       }
   
       return fallback_rosters.get(team_name, [
           'Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5',
           'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'
       ])


class AdvancedTeamAnalyzer:
    """Calculate advanced team-level metrics."""
    
    def __init__(self, player_predictor):
        self.player_predictor = player_predictor
    
    def calculate_team_metrics(self, team_data: Dict) -> Dict:
        """Calculate comprehensive team metrics."""
        players = team_data['players']
        
        # Basic aggregations
        total_predicted_points = sum(p['predicted_points'] for p in players.values())
        avg_confidence = np.mean([p['model_mae'] for p in players.values()])
        
        # Position analysis
        position_breakdown = {}
        for player in players.values():
            pos = player['position']
            if pos not in position_breakdown:
                position_breakdown[pos] = []
            position_breakdown[pos].append(player['predicted_points'])
        
        # Calculate depth score
        starters = sorted(players.values(), key=lambda x: x['predicted_points'], reverse=True)[:5]
        bench = sorted(players.values(), key=lambda x: x['predicted_points'], reverse=True)[5:12]
        
        starter_points = sum(p['predicted_points'] for p in starters)
        bench_points = sum(p['predicted_points'] for p in bench) if bench else 0
        depth_score = bench_points / starter_points if starter_points > 0 else 0
        
        # Estimate team pace and ratings (simplified)
        estimated_pace = self._estimate_team_pace(players)
        offensive_rating = self._estimate_offensive_rating(players, total_predicted_points)
        defensive_rating = self._estimate_defensive_rating(players)
        
        return {
            'total_predicted_points': round(total_predicted_points, 1),
            'avg_prediction_confidence': round(avg_confidence, 2),
            'position_breakdown': position_breakdown,
            'starter_strength': round(starter_points, 1),
            'bench_strength': round(bench_points, 1),
            'depth_score': round(depth_score, 3),
            'estimated_pace': round(estimated_pace, 1),
            'offensive_rating': round(offensive_rating, 1),
            'defensive_rating': round(defensive_rating, 1),
            'net_rating': round(offensive_rating - defensive_rating, 1)
        }
    
    def _estimate_team_pace(self, players: Dict) -> float:
        """Estimate team pace based on player types."""
        # Simplified pace estimation
        pg_players = [p for p in players.values() if p['position'] == 'PG']
        avg_pg_score = np.mean([p['predicted_points'] for p in pg_players]) if pg_players else 15
       
        # Higher-scoring PGs often indicate faster pace
        base_pace = 98.0
        pace_adjustment = (avg_pg_score - 15) * 0.5
       
        return max(92, min(105, base_pace + pace_adjustment))
   
    def _estimate_offensive_rating(self, players: Dict, total_points: float) -> float:
        """Estimate offensive rating (points per 100 possessions)."""
        # Convert total points to per-100-possession metric
        # Average NBA team has ~98 possessions per game
        return (total_points / 98) * 100
   
    def _estimate_defensive_rating(self, players: Dict) -> float:
        """Estimate defensive rating based on player composition."""
        # Simplified defensive rating estimation
        # Better defensive players typically score less but contribute more defensively
       
        total_players = len(players)
        if total_players == 0:
            return 110.0  # League average
       
        # Rough estimation: teams with more balanced scoring have better defense
        point_values = [p['predicted_points'] for p in players.values()]
        scoring_variance = np.var(point_values) if point_values else 50
       
        # Lower variance (more balanced) = better defense
        base_def_rating = 110.0
        variance_adjustment = (scoring_variance - 50) * 0.1
       
        return max(100, min(120, base_def_rating + variance_adjustment))


class MatchupAnalysisEngine:
    """Analyze specific team matchups - FIXED VERSION."""
   
    def analyze_matchup(self, team_a_data: Dict, team_b_data: Dict,
                        team_a_metrics: Dict, team_b_metrics: Dict) -> Dict:
        """Comprehensive matchup analysis."""
       
        return {
            'pace_matchup': self._analyze_pace_matchup(team_a_metrics, team_b_metrics),
            'positional_advantages': self._analyze_positional_matchups(team_a_data, team_b_data),
            'depth_comparison': self._compare_team_depth(team_a_metrics, team_b_metrics),
            'style_matchup': self._analyze_style_matchup(team_a_data, team_b_data),
            'key_factors': self._identify_key_factors(team_a_data, team_b_data, team_a_metrics, team_b_metrics)
        }
   
    def _analyze_pace_matchup(self, team_a_metrics: Dict, team_b_metrics: Dict) -> Dict:
        """Analyze pace matchup between teams."""
        pace_a = team_a_metrics['estimated_pace']
        pace_b = team_b_metrics['estimated_pace']
       
        pace_diff = abs(pace_a - pace_b)
       
        if pace_diff < 2:
            pace_matchup = "Similar pace - expect typical game flow"
        elif pace_a > pace_b + 2:
            pace_matchup = "Team A prefers faster pace - may dictate tempo"
        else:
            pace_matchup = "Team B prefers faster pace - may dictate tempo"
       
        expected_pace = (pace_a + pace_b) / 2
       
        return {
            'team_a_pace': pace_a,
            'team_b_pace': pace_b,
            'expected_game_pace': round(expected_pace, 1),
            'pace_advantage': 'team_a' if pace_a > pace_b + 2 else 'team_b' if pace_b > pace_a + 2 else 'neutral',
            'analysis': pace_matchup
        }
   
    def _analyze_positional_matchups(self, team_a_data: Dict, team_b_data: Dict) -> Dict:
        """Analyze position-by-position matchups."""
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        matchups = {}
       
        for pos in positions:
            team_a_players = [p for p in team_a_data['players'].values() if p['position'] == pos]
            team_b_players = [p for p in team_b_data['players'].values() if p['position'] == pos]
           
            team_a_score = sum(p['predicted_points'] for p in team_a_players)
            team_b_score = sum(p['predicted_points'] for p in team_b_players)
           
            advantage = 'team_a' if team_a_score > team_b_score + 2 else 'team_b' if team_b_score > team_a_score + 2 else 'neutral'
           
            matchups[pos] = {
                'team_a_total': round(team_a_score, 1),
                'team_b_total': round(team_b_score, 1),
                'advantage': advantage,
                'point_differential': round(team_a_score - team_b_score, 1)
            }
       
        return matchups
   
    def _compare_team_depth(self, team_a_metrics: Dict, team_b_metrics: Dict) -> Dict:
        """Compare depth between two teams - FIXED METHOD."""
        depth_a = team_a_metrics.get('depth_score', 0)
        depth_b = team_b_metrics.get('depth_score', 0)
       
        advantage = 'team_a' if depth_a > depth_b + 0.1 else 'team_b' if depth_b > depth_a + 0.1 else 'neutral'
       
        return {
            'team_a_depth': depth_a,
            'team_b_depth': depth_b,
            'advantage': advantage,
            'analysis': f"Team {'A' if advantage == 'team_a' else 'B' if advantage == 'team_b' else 'depth is relatively'} has {'superior' if advantage != 'neutral' else 'even'} bench depth"
        }
   
    def _analyze_style_matchup(self, team_a_data: Dict, team_b_data: Dict) -> Dict:
        """Analyze playing style matchup."""
        # Analyze team composition for style indicators
        team_a_players = team_a_data['players']
        team_b_players = team_b_data['players']
       
        # Calculate style indicators
        team_a_guard_heavy = len([p for p in team_a_players.values() if p['position'] in ['PG', 'SG']]) / len(team_a_players)
        team_b_guard_heavy = len([p for p in team_b_players.values() if p['position'] in ['PG', 'SG']]) / len(team_b_players)
       
        team_a_big_heavy = len([p for p in team_a_players.values() if p['position'] in ['PF', 'C']]) / len(team_a_players)
        team_b_big_heavy = len([p for p in team_b_players.values() if p['position'] in ['PF', 'C']]) / len(team_b_players)
       
        # Determine styles
        team_a_style = 'Guard-Heavy' if team_a_guard_heavy > 0.4 else 'Big-Heavy' if team_a_big_heavy > 0.35 else 'Balanced'
        team_b_style = 'Guard-Heavy' if team_b_guard_heavy > 0.4 else 'Big-Heavy' if team_b_big_heavy > 0.35 else 'Balanced'
       
        # Style matchup analysis
        if team_a_style == team_b_style:
            matchup_type = 'Similar Styles'
            analysis = f"Both teams prefer {team_a_style.lower()} approach - expect similar game flow"
        elif (team_a_style == 'Guard-Heavy' and team_b_style == 'Big-Heavy') or (team_a_style == 'Big-Heavy' and team_b_style == 'Guard-Heavy'):
            matchup_type = 'Contrasting Styles'
            analysis = f"Style clash: {team_a_style} vs {team_b_style} - pace and spacing will be key"
        else:
            matchup_type = 'Complementary Styles'
            analysis = f"One balanced team vs one specialized - adaptability will matter"
       
        return {
            'team_a_style': team_a_style,
            'team_b_style': team_b_style,
            'matchup_type': matchup_type,
            'analysis': analysis
        }
   
    def _identify_key_factors(self, team_a_data: Dict, team_b_data: Dict,
                            team_a_metrics: Dict, team_b_metrics: Dict) -> List[str]:
        """Identify key factors that could determine the game."""
        factors = []
       
        # Scoring advantage
        total_a = team_a_metrics['total_predicted_points']
        total_b = team_b_metrics['total_predicted_points']
       
        if abs(total_a - total_b) > 5:
            leader = "Team A" if total_a > total_b else "Team B"
            factors.append(f"{leader} has significant scoring advantage ({abs(total_a - total_b):.1f} points)")
       
        # Depth advantage
        depth_a = team_a_metrics['depth_score']
        depth_b = team_b_metrics['depth_score']
       
        if abs(depth_a - depth_b) > 0.2:
            deeper_team = "Team A" if depth_a > depth_b else "Team B"
            factors.append(f"{deeper_team} has superior bench depth")
       
        # Pace factor
        pace_diff = abs(team_a_metrics['estimated_pace'] - team_b_metrics['estimated_pace'])
        if pace_diff > 3:
            factors.append(f"Significant pace differential ({pace_diff:.1f}) could affect game flow")
       
        # Star power
        team_a_stars = [p for p in team_a_data['players'].values() if p['predicted_points'] >= 20]
        team_b_stars = [p for p in team_b_data['players'].values() if p['predicted_points'] >= 20]
       
        if len(team_a_stars) != len(team_b_stars):
            star_advantage = "Team A" if len(team_a_stars) > len(team_b_stars) else "Team B"
            factors.append(f"{star_advantage} has more star-level scorers")
       
        return factors


class MonteCarloSimulator:
    """Monte Carlo simulation for game outcomes."""
   
    def simulate_game(self, team_a_data: Dict, team_b_data: Dict,
                    predictions: Dict, n_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulation of the game."""
       
        wins_a = 0
        scores_a = []
        scores_b = []
        margins = []
       
        # Base predictions
        base_a = predictions['ensemble']['team_a_score']
        base_b = predictions['ensemble']['team_b_score']
       
        for _ in range(n_simulations):
            # Add random variance
            score_a = np.random.normal(base_a, 8.0)  # ~8 point standard deviation
            score_b = np.random.normal(base_b, 8.0)
           
            scores_a.append(score_a)
            scores_b.append(score_b)
            margins.append(score_a - score_b)
           
            if score_a > score_b:
                wins_a += 1
       
        win_prob_a = wins_a / n_simulations
       
        return {
            'simulations_run': n_simulations,
            'win_probability_a': round(win_prob_a, 4),
            'win_probability_b': round(1 - win_prob_a, 4),
            'score_distributions': {
                'team_a': {
                    'mean': round(np.mean(scores_a), 1),
                    'std': round(np.std(scores_a), 1),
                    'percentiles': {
                        '10th': round(np.percentile(scores_a, 10), 1),
                        '25th': round(np.percentile(scores_a, 25), 1),
                        '50th': round(np.percentile(scores_a, 50), 1),
                        '75th': round(np.percentile(scores_a, 75), 1),
                        '90th': round(np.percentile(scores_a, 90), 1)
                    }
                },
                'team_b': {
                    'mean': round(np.mean(scores_b), 1),
                    'std': round(np.std(scores_b), 1),
                    'percentiles': {
                        '10th': round(np.percentile(scores_b, 10), 1),
                        '25th': round(np.percentile(scores_b, 25), 1),
                        '50th': round(np.percentile(scores_b, 50), 1),
                        '75th': round(np.percentile(scores_b, 75), 1),
                        '90th': round(np.percentile(scores_b, 90), 1)
                    }
                }
            },
            'margin_analysis': {
                'avg_margin': round(np.mean(margins), 1),
                'margin_std': round(np.std(margins), 1),
                'close_game_probability': round(np.mean(np.abs(margins) <= 5), 3),
                'blowout_probability_a': round(np.mean(np.array(margins) >= 15), 3),
                'blowout_probability_b': round(np.mean(np.array(margins) <= -15), 3)
            }
        }