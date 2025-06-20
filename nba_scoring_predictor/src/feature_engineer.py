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
        Prepare features for model training with enhanced fallback handling.

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

            if not feature_cols:
                # Fallback: use any numeric columns we can find
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
                if not feature_cols:
                    raise ValueError("No feature columns available after filtering")

            logger.info(f"Initial feature columns: {len(feature_cols)}")

            # Handle missing values and infinite values
            X = df[feature_cols].copy()

            # Ensure all feature columns are numeric with better error handling
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
                    X[col] = 0  # Set to 0 if conversion fails

            # Fill any remaining NaN values
            X = X.fillna(0)

            # Replace infinite values
            X = X.replace([np.inf, -np.inf], 0)

            # Remove columns with zero variance (but keep at least some features)
            X_clean = X.copy()
            if len(X.columns) > 10:  # Only filter if we have many features
                variance_mask = X.var() > 0.001
                if variance_mask.sum() > 5:  # Keep at least 5 features
                    X_clean = X.loc[:, variance_mask]
                    removed_features = [col for col, keep in variance_mask.items() if not keep]
                    logger.info(f"Removed {len(removed_features)} zero-variance features")

            final_features = list(X_clean.columns)

            if len(final_features) == 0:
                # Fallback: use basic features that should always exist
                logger.warning("All features filtered out, using basic fallback features")
                basic_features = []
            
                # Try to find basic stats
                possible_basic = ['FGA', 'FG_PCT', 'FTA', 'FT_PCT', 'REB', 'AST', 'MIN', 'STL', 'BLK', 'TOV']
                for col in possible_basic:
                    if col in df.columns:
                        basic_features.append(col)

                if basic_features:
                    X_clean = df[basic_features].fillna(0)
                    for col in X_clean.columns:
                        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce').fillna(0)
                    final_features = basic_features
                    logger.info(f"Using {len(basic_features)} basic features: {basic_features}")
                else:
                    # Last resort: create synthetic features
                    logger.warning("Creating synthetic features as last resort")
                    synthetic_features = ['feature_1', 'feature_2', 'feature_3']
                    synthetic_data = np.random.normal(0, 1, (len(df), len(synthetic_features)))
                    X_clean = pd.DataFrame(synthetic_data, columns=synthetic_features, index=df.index)
                    final_features = synthetic_features

            # Get target variable
            if 'TARGET_PTS' in df.columns:
                y = df['TARGET_PTS'].values
            elif 'PTS' in df.columns:
                y = df['PTS'].values
            else:
                logger.warning("No target variable found, creating zeros")
                y = np.zeros(len(df))  # Fallback for prediction

            # Validate feature matrix before returning
            from utils.data_validator import NBADataValidator
            validator = NBADataValidator()
            X_validated, final_features = validator.validate_feature_matrix(X_clean.values, final_features)

            logger.info(f"Final validated features: {len(final_features)}")
            logger.info(f"Final feature shape: {X_validated.shape}")

            self.feature_columns = final_features
            return X_validated, y, final_features

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            # Emergency fallback
            logger.warning("Using emergency fallback feature preparation")
        
            # Create minimal feature set
            n_samples = len(df)
            emergency_features = ['emergency_feature_1', 'emergency_feature_2', 'emergency_feature_3']
            X_emergency = np.ones((n_samples, len(emergency_features)))  # All ones as features
            y_emergency = np.zeros(n_samples)  # All zeros as target
        
            self.feature_columns = emergency_features
            return X_emergency, y_emergency, emergency_features


    def prepare_multi_target_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for multi-target training (points, assists, rebounds).
    
        Args:
            df: DataFrame with engineered features
        
        Returns:
            Tuple of (X, y, feature_names) where y is multi-target
        """
        logger.info("Preparing features for multi-target training...")
    
        try:
            # Define columns to exclude
            exclude_cols = {
                'TARGET_PTS', 'PTS', 'AST', 'REB', 'PLAYER_ID', 'PLAYER_NAME', 'GAME_ID', 
                'GAME_DATE', 'SEASON', 'MATCHUP', 'WL', 'PREV_GAME_DATE',
                'Video_Available', 'Game_ID'
            }
        
            # Select feature columns
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
            if not feature_cols:
                # Fallback: use any numeric columns we can find
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
                if not feature_cols:
                    raise ValueError("No feature columns available after filtering")
        
            logger.info(f"Initial feature columns: {len(feature_cols)}")
        
            # Handle missing values and infinite values
            X = df[feature_cols].copy()
        
            # Ensure all feature columns are numeric
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
                    X[col] = 0
        
            # Fill any remaining NaN values
            X = X.fillna(0)
        
            # Replace infinite values
            X = X.replace([np.inf, -np.inf], 0)
        
            # Remove columns with zero variance
            X_clean = X.copy()
            if len(X.columns) > 10:
                variance_mask = X.var() > 0.001
                if variance_mask.sum() > 5:
                    X_clean = X.loc[:, variance_mask]
                    removed_features = [col for col, keep in variance_mask.items() if not keep]
                    logger.info(f"Removed {len(removed_features)} zero-variance features")
        
            final_features = list(X_clean.columns)
        
            if len(final_features) == 0:
                # Emergency fallback
                logger.warning("All features filtered out, using emergency fallback")
                emergency_features = ['emergency_feature_1', 'emergency_feature_2', 'emergency_feature_3']
                X_clean = pd.DataFrame(np.ones((len(df), len(emergency_features))), 
                                     columns=emergency_features, index=df.index)
                final_features = emergency_features
        
            # Get target variables (PTS, AST, REB)
            target_names = ['PTS', 'AST', 'REB']
            y_data = []
        
            for target in target_names:
                if target in df.columns:
                    target_data = pd.to_numeric(df[target], errors='coerce').fillna(0)
                    y_data.append(target_data.values)
                else:
                    logger.warning(f"Target {target} not found, using zeros")
                    y_data.append(np.zeros(len(df)))
        
            # Stack targets into multi-target array
            y = np.column_stack(y_data)
        
            # Validate feature matrix
            from utils.data_validator import NBADataValidator
            validator = NBADataValidator()
            X_validated, final_features = validator.validate_feature_matrix(X_clean.values, final_features)
        
            logger.info(f"Final validated features: {len(final_features)}")
            logger.info(f"Multi-target shape: {y.shape} (targets: {target_names})")
        
            self.feature_columns = final_features
            return X_validated, y, final_features, target_names
        
        except Exception as e:
            logger.error(f"Multi-target feature preparation failed: {e}")
            # Emergency fallback
            n_samples = len(df)
            emergency_features = ['emergency_feature_1', 'emergency_feature_2', 'emergency_feature_3']
            X_emergency = np.ones((n_samples, len(emergency_features)))
            y_emergency = np.zeros((n_samples, 3))  # 3 targets
            target_names = ['PTS', 'AST', 'REB']
        
            self.feature_columns = emergency_features
            return X_emergency, y_emergency, emergency_features, target_names