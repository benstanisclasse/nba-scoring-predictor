# -*- coding: utf-8 -*-
"""
Main entry point for NBA Player Scoring Predictor
"""
import os
import sys
import argparse
from typing import List, Optional
from src.predictor import EnhancedNBAPredictor as NBAPlayerScoringPredictor

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_gui():
    """Run the PyQt5 GUI application."""
    try:
        from src.gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying to install PyQt5...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5", "qdarkstyle"])
            print("PyQt5 installed successfully! Please run the program again.")
        except subprocess.CalledProcessError:
            print("Failed to install PyQt5. Please install manually:")
            print("pip install PyQt5 qdarkstyle")
        sys.exit(1)

def train_model_cli(player_names: Optional[List[str]] = None, 
                   seasons: Optional[List[str]] = None,
                   optimize: bool = False,
                   save_path: str = "models/nba_model.pkl"):
    """Train model via command line interface."""
    try:
        from src.predictor import NBAPlayerScoringPredictor
        from utils.logger import main_logger as logger
        from config.settings import DEFAULT_SEASONS
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        return
    
    logger.info("Starting model training via CLI...")
    
    # Initialize predictor
    predictor = NBAPlayerScoringPredictor()
    
    try:
        # Collect data
        logger.info("Collecting player data...")
        data = predictor.collect_data(
            player_names=player_names,
            seasons=seasons or DEFAULT_SEASONS
        )
        
        # Process data
        logger.info("Processing data...")
        processed_data = predictor.process_data(data)
        
        # Train models
        logger.info("Training models...")
        results = predictor.train(processed_data, optimize=optimize)
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        predictor.save_model(save_path)
        
        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        for model_name, metrics in results.items():
            print(f"{model_name.upper()}:")
            print(f"  Test MAE: {metrics['test_mae']:.3f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.3f}")
            print(f"  Test R-squared: {metrics['test_r2']:.3f}")
            print()
        
        print(f"Model saved to: {save_path}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def predict_cli(player_name: str, model_path: str = "models/nba_model.pkl"):
    """Make prediction via command line interface."""
    try:
        from src.predictor import NBAPlayerScoringPredictor
        from utils.logger import main_logger as logger
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        return
    
    logger.info(f"Making prediction for {player_name}...")
    
    # Initialize and load model
    predictor = NBAPlayerScoringPredictor()
    
    try:
        predictor.load_model(model_path)
        
        # Make prediction
        predictions = predictor.predict_player_points(player_name)
        
        # Print results
        print("\n" + "="*50)
        print(f"PREDICTIONS FOR {player_name.upper()}")
        print("="*50)
        
        recent_avg = predictions.get('recent_average', 0)
        print(f"Recent Average: {recent_avg:.1f} points\n")
        
        for model_name, pred_data in predictions.items():
            if model_name not in ['recent_average', 'player_name']:
                pred_points = pred_data['predicted_points']
                ci_low, ci_high = pred_data['confidence_interval']
                print(f"{model_name.title()}: {pred_points:.1f} points (Range: {ci_low:.1f}-{ci_high:.1f})")
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="NBA Player Scoring Predictor")
    parser.add_argument("--mode", choices=["gui", "train", "predict"], default="gui",
                       help="Mode to run the application in")
    parser.add_argument("--players", nargs="+", help="Player names for training")
    parser.add_argument("--seasons", nargs="+", help="Seasons to use for training")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--model-path", default="models/nba_model.pkl", help="Path to save/load model")
    parser.add_argument("--player-name", help="Player name for prediction")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == "gui":
        run_gui()
    elif args.mode == "train":
        train_model_cli(
            player_names=args.players,
            seasons=args.seasons,
            optimize=args.optimize,
            save_path=args.model_path
        )
    elif args.mode == "predict":
        if not args.player_name:
            print("Error: --player-name required for predict mode")
            sys.exit(1)
        predict_cli(args.player_name, args.model_path)

if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-
"""
Main entry point for NBA Player Scoring Predictor
"""
import os
import sys
import argparse
from typing import List, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_gui():
    """Run the PyQt5 GUI application."""
    try:
        from src.gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying to install PyQt5...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5", "qdarkstyle"])
            print("PyQt5 installed successfully! Please run the program again.")
        except subprocess.CalledProcessError:
            print("Failed to install PyQt5. Please install manually:")
            print("pip install PyQt5 qdarkstyle")
        sys.exit(1)

def train_model_cli(player_names: Optional[List[str]] = None, 
                   seasons: Optional[List[str]] = None,
                   optimize: bool = False,
                   save_path: str = "models/nba_model.pkl"):
    """Train model via command line interface."""
    try:
        from src.predictor import NBAPlayerScoringPredictor
        from utils.logger import main_logger as logger
        from config.settings import DEFAULT_SEASONS
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        return
    
    logger.info("Starting model training via CLI...")
    
    # Initialize predictor
    predictor = NBAPlayerScoringPredictor()
    
    try:
        # Collect data
        logger.info("Collecting player data...")
        data = predictor.collect_data(
            player_names=player_names,
            seasons=seasons or DEFAULT_SEASONS
        )
        
        # Process data
        logger.info("Processing data...")
        processed_data = predictor.process_data(data)
        
        # Train models
        logger.info("Training models...")
        results = predictor.train(processed_data, optimize=optimize)
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        predictor.save_model(save_path)
        
        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        for model_name, metrics in results.items():
            print(f"{model_name.upper()}:")
            print(f"  Test MAE: {metrics['test_mae']:.3f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.3f}")
            print(f"  Test R-squared: {metrics['test_r2']:.3f}")
            print()
        
        print(f"Model saved to: {save_path}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def predict_cli(player_name: str, model_path: str = "models/nba_model.pkl"):
    """Make prediction via command line interface."""
    try:
        from src.predictor import NBAPlayerScoringPredictor
        from utils.logger import main_logger as logger
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        return
    
    logger.info(f"Making prediction for {player_name}...")
    
    # Initialize and load model
    predictor = NBAPlayerScoringPredictor()
    
    try:
        predictor.load_model(model_path)
        
        # Make prediction
        predictions = predictor.predict_player_points(player_name)
        
        # Print results
        print("\n" + "="*50)
        print(f"PREDICTIONS FOR {player_name.upper()}")
        print("="*50)
        
        recent_avg = predictions.get('recent_average', 0)
        print(f"Recent Average: {recent_avg:.1f} points\n")
        
        for model_name, pred_data in predictions.items():
            if model_name not in ['recent_average', 'player_name']:
                pred_points = pred_data['predicted_points']
                ci_low, ci_high = pred_data['confidence_interval']
                print(f"{model_name.title()}: {pred_points:.1f} points (Range: {ci_low:.1f}-{ci_high:.1f})")
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="NBA Player Scoring Predictor")
    parser.add_argument("--mode", choices=["gui", "train", "predict"], default="gui",
                       help="Mode to run the application in")
    parser.add_argument("--players", nargs="+", help="Player names for training")
    parser.add_argument("--seasons", nargs="+", help="Seasons to use for training")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--model-path", default="models/nba_model.pkl", help="Path to save/load model")
    parser.add_argument("--player-name", help="Player name for prediction")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == "gui":
        run_gui()
    elif args.mode == "train":
        train_model_cli(
            player_names=args.players,
            seasons=args.seasons,
            optimize=args.optimize,
            save_path=args.model_path
        )
    elif args.mode == "predict":
        if not args.player_name:
            print("Error: --player-name required for predict mode")
            sys.exit(1)
        predict_cli(args.player_name, args.model_path)

if __name__ == "__main__":
    main()