# -*- coding: utf-8 -*-
"""
Statistical significance testing
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class StatisticalTester:
    """Handles all statistical significance testing."""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def run_all_tests(self, seasons: List[str]) -> Dict:
        """Run all statistical tests."""
        results = {}
        
        # Test against baselines
        results['baseline_comparison'] = self._test_against_baselines(seasons)
        
        # Test prediction accuracy
        results['accuracy_tests'] = self._test_prediction_accuracy(seasons)
        
        # Test consistency across time
        results['temporal_consistency'] = self._test_temporal_consistency(seasons)
        
        # Test by player types
        results['player_type_analysis'] = self._test_by_player_types(seasons)
        
        return results
    
    def _test_against_baselines(self, seasons: List[str]) -> Dict:
        """Test model performance against various baselines."""
        # Implementation for baseline comparisons
        pass