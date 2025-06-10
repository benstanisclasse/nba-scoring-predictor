# -*- coding: utf-8 -*-
"""
Main backtesting engine that coordinates all validation components
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os

from .betting_simulator import BettingSimulator
from .statistical_tests import StatisticalTester
from .calibration_tester import CalibrationTester
from .performance_analyzer import PerformanceAnalyzer
from .report_generator import ValidationReportGenerator
from utils.logger import main_logger as logger

class NBAPredictionValidator:
    """Main backtesting coordinator that runs all validation tests."""
    
    def __init__(self, predictor, config: Dict = None):
        self.predictor = predictor
        self.config = config or self._get_default_config()
        
        # Initialize sub-components
        self.betting_simulator = BettingSimulator(predictor)
        self.statistical_tester = StatisticalTester(predictor)
        self.calibration_tester = CalibrationTester(predictor)
        self.performance_analyzer = PerformanceAnalyzer(predictor)
        self.report_generator = ValidationReportGenerator()
        
        # Results storage
        self.validation_results = {}
    
    def run_comprehensive_validation(self, seasons: List[str] = None) -> Dict:
        """Run complete validation suite."""
        seasons = seasons or ['2022-23', '2023-24']
        
        logger.info("Starting comprehensive validation suite...")
        
        # 1. Performance Analysis
        performance_results = self.performance_analyzer.analyze_performance(seasons)
        
        # 2. Statistical Tests
        statistical_results = self.statistical_tester.run_all_tests(seasons)
        
        # 3. Betting Simulations
        betting_results = self.betting_simulator.run_simulations(seasons)
        
        # 4. Calibration Testing
        calibration_results = self.calibration_tester.test_calibration(seasons)
        
        # 5. Compile results
        self.validation_results = {
            'performance': performance_results,
            'statistical': statistical_results,
            'betting': betting_results,
            'calibration': calibration_results,
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'seasons_tested': seasons,
                'model_config': self.config
            }
        }
        
        # 6. Generate report
        report_path = self.report_generator.generate_comprehensive_report(
            self.validation_results
        )
        
        logger.info(f"Validation complete. Report saved to: {report_path}")
        return self.validation_results
    
    def _get_default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            'min_predictions': 100,
            'significance_level': 0.05,
            'betting_bankroll': 1000,
            'bet_size_pct': 0.02,
            'confidence_threshold': 0.7
        }