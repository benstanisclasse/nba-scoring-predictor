# -*- coding: utf-8 -*-
"""
Betting simulation and strategy testing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import main_logger as logger

class BettingSimulator:
    """Simulates various betting strategies."""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.strategies = self._define_betting_strategies()
    
    def run_simulations(self, seasons: List[str]) -> Dict:
        """Run all betting simulations."""
        results = {}
        
        for strategy_name, strategy_config in self.strategies.items():
            logger.info(f"Running betting simulation: {strategy_name}")
            
            strategy_results = self._simulate_strategy(
                seasons, strategy_config
            )
            results[strategy_name] = strategy_results
            
        return results
    
    def _define_betting_strategies(self) -> Dict:
        """Define different betting strategies to test."""
        return {
            'conservative': {
                'type': 'over_under',
                'confidence_threshold': 0.8,
                'edge_threshold': 2.0,  # Only bet if 2+ point edge
                'max_bet_pct': 0.01    # 1% of bankroll max
            },
            'aggressive': {
                'type': 'over_under', 
                'confidence_threshold': 0.6,
                'edge_threshold': 0.5,
                'max_bet_pct': 0.03
            },
            'kelly_criterion': {
                'type': 'kelly',
                'confidence_threshold': 0.7,
                'edge_threshold': 1.0,
                'max_bet_pct': 0.05
            },
            'high_volume': {
                'type': 'over_under',
                'confidence_threshold': 0.5,
                'edge_threshold': 0.0,
                'max_bet_pct': 0.01
            }
        }
    
    def _simulate_strategy(self, seasons: List[str], strategy: Dict) -> Dict:
        """Simulate a specific betting strategy."""
        # Simulate betting results
        np.random.seed(42)
        
        initial_bankroll = 1000
        current_bankroll = initial_bankroll
        bet_history = []
        
        # Simulate 1000 betting opportunities
        n_bets = 1000
        confidence_threshold = strategy['confidence_threshold']
        edge_threshold = strategy['edge_threshold']
        max_bet_pct = strategy['max_bet_pct']
        
        wins = 0
        total_bets = 0
        
        for i in range(n_bets):
            # Simulate prediction confidence and edge
            confidence = np.random.beta(2, 2)
            edge = np.random.normal(0, 2)
            
            # Check if bet meets criteria
            if confidence >= confidence_threshold and abs(edge) >= edge_threshold:
                # Calculate bet size
                if strategy['type'] == 'kelly':
                    # Kelly criterion
                    win_prob = confidence
                    odds = 1.91  # -110 odds
                    kelly_fraction = (win_prob * odds - 1) / (odds - 1)
                    bet_size = min(current_bankroll * kelly_fraction * 0.25, 
                                 current_bankroll * max_bet_pct)
                else:
                    bet_size = current_bankroll * max_bet_pct
                
                bet_size = max(10, min(bet_size, current_bankroll * max_bet_pct))
                
                # Simulate bet outcome
                win_prob = min(0.52 + abs(edge) * 0.02, 0.65)  # Better edge = higher win prob
                bet_wins = np.random.random() < win_prob
                
                if bet_wins:
                    profit = bet_size * 0.91  # -110 odds
                    current_bankroll += profit
                    wins += 1
                else:
                    current_bankroll -= bet_size
                
                total_bets += 1
                
                bet_history.append({
                    'bet_number': total_bets,
                    'bet_size': bet_size,
                    'confidence': confidence,
                    'edge': edge,
                    'won': bet_wins,
                    'bankroll': current_bankroll
                })
        
        # Calculate results
        total_profit = current_bankroll - initial_bankroll
        roi = (total_profit / initial_bankroll) * 100
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate max drawdown
        bankroll_series = [bet['bankroll'] for bet in bet_history]
        running_max = np.maximum.accumulate([initial_bankroll] + bankroll_series)
        drawdowns = [(bankroll - max_so_far) / max_so_far for bankroll, max_so_far in 
                    zip([initial_bankroll] + bankroll_series, running_max)]
        max_drawdown = min(drawdowns) * 100
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': current_bankroll,
            'total_profit': total_profit,
            'roi_percent': roi,
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': win_rate,
            'max_drawdown_percent': max_drawdown,
            'strategy_config': strategy,
            'bet_history': bet_history[:100]  # Store first 100 bets for analysis
        }