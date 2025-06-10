# In scripts/run_validation.py - CORRECTED VERSION
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
    
    # FIXED: Ensure model is actually trained before validation
    try:
        predictor.load_model('models/Rolebased9playerOH.pkl')
        logger.info("Loaded existing model")
    except:
        logger.info("Training new model for validation...")
        
        # Train a basic model first
        try:
            # Use fallback players if NBA data unavailable
            fallback_players = [
                'LeBron James', 'Stephen Curry', 'Luka Dončić', 
                'Giannis Antetokounmpo', 'Jayson Tatum'
            ]
            
            # Collect data
            data = predictor.collect_data(
                player_names=fallback_players,
                seasons=['2022-23', '2023-24'],
                use_cache=True
            )
            
            # Process and train
            processed_data = predictor.process_data(data)
            training_results = predictor.train(processed_data, optimize=False)
            
            # Save the model
            os.makedirs('models', exist_ok=True)
            predictor.save_model('models/nba_model.pkl')
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
    
    # Run validation with trained model
    validator = NBAPredictionValidator(predictor)
    results = validator.run_comprehensive_validation(['2022-23', '2023-24'])
    
    logger.info("Validation completed successfully!")
    return results

if __name__ == "__main__":
    main()