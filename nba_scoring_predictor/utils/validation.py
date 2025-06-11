# utils/validation.py
from utils.logger import main_logger as logger

def validate_player_prediction(prediction: float, player_name: str = "Unknown") -> float:
    """Validate and cap individual player predictions."""
    if prediction > 50:  # Even superstars rarely score 50+
        logger.warning(f"Unrealistic prediction for {player_name}: {prediction:.1f}, capping at 35")
        return 35.0
    elif prediction < 0:
        logger.warning(f"Negative prediction for {player_name}: {prediction:.1f}, setting to 5")
        return 5.0
    return prediction

def validate_team_total(team_total: float, team_name: str = "Unknown") -> float:
    """More aggressive validation for team totals."""
    if team_total > 130:  # Very high-scoring games
        logger.warning(f"Unrealistic team total for {team_name}: {team_total:.1f}, capping at 125")
        return 125.0
    elif team_total < 90:
        logger.warning(f"Unrealistic low total for {team_name}: {team_total:.1f}, setting to 100")
        return 100.0
    return team_total

def validate_game_prediction(team_a_score: float, team_b_score: float, 
                           team_a_name: str = "Team A", team_b_name: str = "Team B") -> tuple:
    """Validate complete game prediction."""
    # Validate individual team scores
    team_a_score = validate_team_total(team_a_score, team_a_name)
    team_b_score = validate_team_total(team_b_score, team_b_name)
    
    # Check total points (NBA games rarely exceed 260 total points)
    total_points = team_a_score + team_b_score
    if total_points > 260:
        # Scale both teams down proportionally
        scale_factor = 240 / total_points  # Target 240 total points
        team_a_score = team_a_score * scale_factor
        team_b_score = team_b_score * scale_factor
        logger.warning(f"Game total too high ({total_points:.1f}), scaled down to {team_a_score + team_b_score:.1f}")
    
    return team_a_score, team_b_score
