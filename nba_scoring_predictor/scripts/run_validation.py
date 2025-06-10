#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run comprehensive validation
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predictor import EnhancedNBAPredictor
from validation.backtester import NBAPredictionValidator
from utils.logger import main_logger as logger

def main():
    """Run validation suite."""
    logger.info("Starting NBA Prediction Validation")
    
    # Initialize predictor
    predictor = EnhancedNBAPredictor()
    
    # Load or train model
    try:
        predictor.load_model('models/nba_model.pkl')
        logger.info("Loaded existing model")
    except:
        logger.info("Training new model for validation...")
        # Add training logic here
        pass
    
    # Run validation
    validator = NBAPredictionValidator(predictor)
    results = validator.run_comprehensive_validation(['2022-23', '2023-24'])
    
    logger.info("Validation completed successfully!")
    return results

if __name__ == "__main__":
    main()
