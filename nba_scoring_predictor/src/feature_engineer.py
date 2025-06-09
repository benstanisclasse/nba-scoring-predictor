# -*- coding: utf-8 -*-
"""
Feature engineering module for NBA player scoring prediction - FIXED VERSION
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

from config.settings import ROLLING_WINDOWS, EWM_ALPHAS, MAX_DAYS_REST
from utils.logger import main_logger as logger

class FeatureEngineer:
    """Handles feature engineering for NBA player performance prediction."""
    
    def __init__(self):
        self.feature_columns = []
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to raw game data.
        
        Args:
            data: Raw game log data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        try:
            # Make a copy and prepare data
            df = data.copy()
            
            # Ensure we have the required columns
            required_cols = ['PLAYER_ID', 'GAME_DATE', 'PTS']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Reset index to avoid duplicates
            df = df.reset_index(drop=True)
            
            # Convert date and sort
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
            
            # Set target variable
            df['TARGET_PTS'] = df['PTS']
            
            # Add features step by step
            logger.info("Adding basic features...")
            df = self._add_basic_features(df)
            
            logger.info("Adding rolling features...")
            df = self._add_rolling_features(df)
            
            logger.info("Adding efficiency metrics...")
            df = self._add_efficiency_metrics(df)
            
            logger.info("Adding context features...")
            df = self._add_context_features(df)
            
            logger.info("Adding temporal features...")
            df = self._add_temporal_features(df)
            
            logger.info("Adding trend features...")
            df = self._add_trend_features(df)
            
            # Remove early season games with insufficient history
            df = df[df.groupby(['PLAYER_ID', 'SEASON']).cumcount() >= 4].copy()
            
            # Final cleanup - remove any duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            logger.info(f"Feature engineering complete. Dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic statistical features with proper data type handling."""
        
        # Define columns that should be numeric
        numeric_cols = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 
                       'PTS', 'PLUS_MINUS']
        
        # Convert to numeric, forcing errors to NaN, then fill with 0
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle percentage columns specially
        pct_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Ensure percentages are in decimal format (0-1) not (0-100)
                df[col] = df[col].clip(0, 1)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics with data type verification."""
        key_stats = ['PTS', 'MIN', 'FGA', 'FG_PCT', 'REB', 'AST']
        
        for window in [3, 5, 10]:
            for stat in key_stats:
                if stat in df.columns:
                    try:
                        # CRITICAL: Ensure the column is numeric before rolling calculation
                        if not pd.api.types.is_numeric_dtype(df[stat]):
                            logger.warning(f"Converting non-numeric column {stat} to numeric")
                            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
                        
                        # Rolling mean
                        col_name = f'{stat}_ROLL_{window}'
                        df[col_name] = (
                            df.groupby('PLAYER_ID')[stat]
                            .rolling(window=window, min_periods=1)
                            .mean()
                            .values  # Use .values to avoid index issues
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create rolling feature {stat}_{window}: {e}")
                        # Print debug info
                        logger.warning(f"Column {stat} dtype: {df[stat].dtype}")
                        logger.warning(f"Sample values: {df[stat].head().tolist()}")
                        continue
        
        return df
    
    def _add_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced efficiency metrics."""
        try:
            # Ensure all required columns are numeric
            required_cols = ['PTS', 'FGA', 'FTA', 'FGM', 'FG3M', 'TOV', 'MIN']
            for col in required_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # True Shooting Percentage
            df['TRUE_SHOOTING_PCT'] = np.where(
                (2 * (df.get('FGA', 0) + 0.44 * df.get('FTA', 0))) > 0,
                df.get('PTS', 0) / (2 * (df.get('FGA', 0) + 0.44 * df.get('FTA', 0))),
                0
            )
            
            # Effective Field Goal Percentage
            df['EFFECTIVE_FG_PCT'] = np.where(
                df.get('FGA', 0) > 0,
                (df.get('FGM', 0) + 0.5 * df.get('FG3M', 0)) / df.get('FGA', 0),
                0
            )
            
            # Usage Rate (approximation)
            df['USAGE_RATE_APPROX'] = np.where(
                df.get('MIN', 0) > 0,
                ((df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 0)) * 40) / (df.get('MIN', 0) * 5),
                0
            )
            
            # Points per shot attempt
            df['PTS_PER_SHOT'] = np.where(
                df.get('FGA', 0) > 0, 
                df.get('PTS', 0) / df.get('FGA', 0), 
                0
            )
            
        except Exception as e:
            logger.warning(f"Failed to create some efficiency metrics: {e}")
        
        return df
    
    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game context features."""
        try:
            # Home/Away (only if MATCHUP column exists)
            if 'MATCHUP' in df.columns:
                df['IS_HOME'] = (~df['MATCHUP'].str.contains('@', na=False)).astype(int)
            else:
                df['IS_HOME'] = 0
            
            # Days rest
            df['PREV_GAME_DATE'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
            df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
            df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(0, 10)
            
            # Back-to-back games
            df['IS_B2B'] = (df['DAYS_REST'] <= 1).astype(int)
            
            # Game result (only if WL column exists)
            if 'WL' in df.columns:
                df['IS_WIN'] = (df['WL'] == 'W').astype(int)
            else:
                df['IS_WIN'] = 0
                
        except Exception as e:
            logger.warning(f"Failed to create some context features: {e}")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/seasonal features."""
        try:
            # Season progression
            df['SEASON_GAME_NUM'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount() + 1
            
            # Month of season
            df['SEASON_MONTH'] = df['GAME_DATE'].dt.month
            
            # Day of week
            df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
            
            # Weekend games
            df['IS_WEEKEND'] = (df['DAY_OF_WEEK'].isin([5, 6])).astype(int)
            
        except Exception as e:
            logger.warning(f"Failed to create some temporal features: {e}")
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance trend features."""
        try:
            # Ensure PTS is numeric
            df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce').fillna(0)
            
            # Performance vs season average
            season_avg = df.groupby(['PLAYER_ID', 'SEASON'])['PTS'].expanding().mean()
            df['PTS_VS_SEASON_AVG'] = df['PTS'] - season_avg.values
            
            # Hot/cold streaks
            df['SCORING_ABOVE_AVG'] = (df['PTS_VS_SEASON_AVG'] > 0).astype(int)
            
        except Exception as e:
            logger.warning(f"Failed to create some trend features: {e}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for model training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing features for training...")
        
        try:
            # Define columns to exclude
            exclude_cols = {
                'TARGET_PTS', 'PTS', 'PLAYER_ID', 'PLAYER_NAME', 'GAME_ID', 
                'GAME_DATE', 'SEASON', 'MATCHUP', 'WL', 'PREV_GAME_DATE',
                'Video_Available', 'Game_ID'
            }
            
            # Select feature columns
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Handle missing values and infinite values
            X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y = df['TARGET_PTS'].values
            
            # Ensure all feature columns are numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Remove columns with zero variance
            X_clean = X.loc[:, X.var() > 0.001]
            final_features = list(X_clean.columns)
            
            logger.info(f"Selected {len(final_features)} features from {len(feature_cols)} original")
            
            self.feature_columns = final_features
            return X_clean.values, y, final_features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise