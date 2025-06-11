# utils/data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import main_logger as logger

class NBADataValidator:
    """Validates NBA data throughout the pipeline."""
    
    def __init__(self):
        # NBA statistical bounds (based on real NBA data)
        self.STAT_BOUNDS = {
            'PTS': (0, 81),      # Kobe's 81 is the modern record
            'MIN': (0, 53),      # 53 minutes is OT record
            'FGA': (0, 50),      # Extreme high-volume shooter
            'FGM': (0, 30),      # Made field goals
            'FG3A': (0, 25),     # 3-point attempts
            'FG3M': (0, 15),     # 3-point makes
            'FTA': (0, 30),      # Free throw attempts
            'FTM': (0, 30),      # Free throw makes
            'REB': (0, 30),      # Total rebounds
            'AST': (0, 30),      # Assists
            'STL': (0, 11),      # Steals record
            'BLK': (0, 17),      # Blocks record
            'TOV': (0, 15),      # Turnovers
            'PF': (0, 6),        # Personal fouls (6 = ejection)
        }
        
        # Percentage bounds (0-1)
        self.PCT_BOUNDS = {
            'FG_PCT': (0.0, 1.0),
            'FG3_PCT': (0.0, 1.0),
            'FT_PCT': (0.0, 1.0),
        }
    
    def validate_game_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean individual game log data."""
        if df.empty:
            return df
        
        logger.info(f"Validating {len(df)} game records...")
        initial_count = len(df)
        
        df = df.copy()
        
        # 1. Validate basic stats
        df = self._validate_counting_stats(df)
        
        # 2. Validate percentages
        df = self._validate_percentages(df)
        
        # 3. Validate logical relationships
        df = self._validate_stat_relationships(df)
        
        # 4. Remove impossible games
        df = self._remove_impossible_games(df)
        
        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"Removed {initial_count - final_count} invalid games ({((initial_count - final_count) / initial_count) * 100:.1f}%)")
        
        return df
    
    def _validate_counting_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate counting statistics."""
        for stat, (min_val, max_val) in self.STAT_BOUNDS.items():
            if stat in df.columns:
                # Log extreme values before fixing
                extreme_mask = (df[stat] < min_val) | (df[stat] > max_val)
                if extreme_mask.any():
                    logger.warning(f"Found {extreme_mask.sum()} extreme {stat} values")
                    logger.warning(f"Range: {df[stat].min():.1f} to {df[stat].max():.1f}")
                
                # Cap values at reasonable bounds
                df[stat] = df[stat].clip(min_val, max_val)
        
        return df
    
    def _validate_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate percentage statistics."""
        for stat, (min_val, max_val) in self.PCT_BOUNDS.items():
            if stat in df.columns:
                # Convert percentages > 1 to decimal (common data error)
                over_one_mask = df[stat] > 1
                if over_one_mask.any():
                    logger.warning(f"Converting {over_one_mask.sum()} {stat} values from percentage to decimal")
                    df.loc[over_one_mask, stat] = df.loc[over_one_mask, stat] / 100
                
                # Clip to valid range
                df[stat] = df[stat].clip(min_val, max_val)
        
        return df
    
    def _validate_stat_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate logical relationships between stats."""
        
        # FGM cannot exceed FGA
        if 'FGM' in df.columns and 'FGA' in df.columns:
            invalid_fg = df['FGM'] > df['FGA']
            if invalid_fg.any():
                logger.warning(f"Fixing {invalid_fg.sum()} games where FGM > FGA")
                df.loc[invalid_fg, 'FGM'] = df.loc[invalid_fg, 'FGA']
        
        # FTM cannot exceed FTA
        if 'FTM' in df.columns and 'FTA' in df.columns:
            invalid_ft = df['FTM'] > df['FTA']
            if invalid_ft.any():
                logger.warning(f"Fixing {invalid_ft.sum()} games where FTM > FTA")
                df.loc[invalid_ft, 'FTM'] = df.loc[invalid_ft, 'FTA']
        
        # 3PM cannot exceed 3PA
        if 'FG3M' in df.columns and 'FG3A' in df.columns:
            invalid_3p = df['FG3M'] > df['FG3A']
            if invalid_3p.any():
                logger.warning(f"Fixing {invalid_3p.sum()} games where FG3M > FG3A")
                df.loc[invalid_3p, 'FG3M'] = df.loc[invalid_3p, 'FG3A']
        
        # Recalculate percentages if we fixed the makes/attempts
        if all(col in df.columns for col in ['FGM', 'FGA']):
            df['FG_PCT'] = np.where(df['FGA'] > 0, df['FGM'] / df['FGA'], 0)
        
        if all(col in df.columns for col in ['FTM', 'FTA']):
            df['FT_PCT'] = np.where(df['FTA'] > 0, df['FTM'] / df['FTA'], 0)
        
        if all(col in df.columns for col in ['FG3M', 'FG3A']):
            df['FG3_PCT'] = np.where(df['FG3A'] > 0, df['FG3M'] / df['FG3A'], 0)
        
        return df
    
    def _remove_impossible_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove games with impossible stat combinations."""
        initial_count = len(df)
        
        # Remove games with 0 minutes but stats
        if 'MIN' in df.columns and 'PTS' in df.columns:
            impossible_games = (df['MIN'] == 0) & (df['PTS'] > 0)
            if impossible_games.any():
                logger.warning(f"Removing {impossible_games.sum()} games with 0 minutes but positive stats")
                df = df[~impossible_games]
        
        # Remove games with impossible scoring (more points than possible from FG + FT)
        if all(col in df.columns for col in ['PTS', 'FGM', 'FG3M', 'FTM']):
            max_possible_pts = df['FGM'] * 2 + df['FG3M'] + df['FTM']  # 2pts per FG + 1 extra per 3PM + FTM
            impossible_scoring = df['PTS'] > max_possible_pts
            if impossible_scoring.any():
                logger.warning(f"Removing {impossible_scoring.sum()} games with impossible scoring")
                df = df[~impossible_scoring]
        
        return df
    
    def validate_feature_matrix(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Validate feature matrix before training."""
        logger.info(f"Validating feature matrix: {X.shape}")
        
        # Check for infinite values
        inf_mask = np.isinf(X)
        if inf_mask.any():
            logger.warning(f"Found {inf_mask.sum()} infinite values, replacing with 0")
            X[inf_mask] = 0
        
        # Check for extreme outliers (beyond 5 standard deviations)
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            if len(np.unique(feature_data)) > 1:  # Skip constant features
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                if std_val > 0:
                    outlier_mask = np.abs(feature_data - mean_val) > 5 * std_val
                    if outlier_mask.any():
                        logger.warning(f"Capping {outlier_mask.sum()} outliers in {feature_name}")
                        # Cap at 3 standard deviations
                        X[outlier_mask, i] = np.clip(
                            feature_data[outlier_mask], 
                            mean_val - 3 * std_val, 
                            mean_val + 3 * std_val
                        )
        
        # Remove features with zero variance
        variances = np.var(X, axis=0)
        valid_features = variances > 1e-6
        
        if not valid_features.all():
            n_removed = (~valid_features).sum()
            logger.info(f"Removing {n_removed} zero-variance features")
            X = X[:, valid_features]
            feature_names = [name for i, name in enumerate(feature_names) if valid_features[i]]
        
        logger.info(f"Final feature matrix: {X.shape}")
        return X, feature_names
