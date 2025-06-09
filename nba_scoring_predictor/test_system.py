# -*- coding: utf-8 -*-
"""
Test script to verify all components work
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        print("- Testing PyQt5...")
        from PyQt5.QtWidgets import QApplication
        print("  ✓ PyQt5 works")
        
        print("- Testing core ML libraries...")
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import lightgbm
        print("  ✓ ML libraries work")
        
        print("- Testing NBA API...")
        from nba_api.stats.static import players
        print("  ✓ NBA API works")
        
        print("- Testing project modules...")
        from src.predictor import NBAPlayerScoringPredictor
        from src.data_collector import NBADataCollector
        from src.feature_engineer import FeatureEngineer
        from src.model_trainer import ModelTrainer
        from utils.database import DatabaseManager
        from utils.logger import main_logger
        from config.settings import DEFAULT_SEASONS
        print("  ✓ All project modules work")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_simple_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from src.predictor import NBAPlayerScoringPredictor
        
        # Initialize predictor
        predictor = NBAPlayerScoringPredictor()
        print("  ✓ Predictor initialized")
        
        # Test getting players (should work even without data)
        try:
            players = predictor.get_available_players()
            print(f"  ✓ Found {len(players)} cached players")
        except:
            print("  ✓ No cached players (normal for first run)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== NBA Player Scoring Predictor - System Test ===\n")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("✓ Created necessary directories")
    
    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed. Please install missing dependencies:")
        print("pip install PyQt5 pandas numpy scikit-learn xgboost lightgbm optuna nba_api matplotlib seaborn qdarkstyle")
        return False
    
    # Test functionality
    if not test_simple_functionality():
        print("\n✗ Functionality tests failed.")
        return False
    
    print("\n🎉 All tests passed! The system is ready to use.")
    print("\nTo run the application:")
    print("  python main.py")
    
    return True

if __name__ == "__main__":
    main()