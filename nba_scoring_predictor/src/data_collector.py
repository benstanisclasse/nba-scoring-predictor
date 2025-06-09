# -*- coding: utf-8 -*-
"""
Data collection module for NBA player statistics
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import time
from datetime import datetime

from nba_api.stats.endpoints import playergamelog, teamgamelog
from nba_api.stats.static import players, teams

from config.settings import DEFAULT_SEASONS, MIN_GAMES_PLAYED
from utils.logger import main_logger as logger
from utils.database import DatabaseManager

class NBADataCollector:
    """Collects NBA player and team data using the NBA API."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.all_players = players.get_players()
        self.all_teams = teams.get_teams()
    
    def find_players(self, player_names: List[str] = None) -> List[dict]:
        """
        Find players by name or return popular players.
    
        Args:
            player_names: List of player names to search for
        
        Returns:
            List of player dictionaries
        """
        if player_names:
            found_players = []
            for name in player_names:
                matches = [p for p in self.all_players 
                          if name.lower() in p['full_name'].lower()]
                if matches:
                    found_players.extend(matches)
                else:
                    logger.warning(f"No match found for player: {name}")
            return found_players
        else:
            # Import here to avoid circular imports
            from utils.player_storage import PlayerStorage
            storage = PlayerStorage()
            popular_players = storage.get_popular_players()
        
            # Find matching player objects
            found_players = []
            for player_name in popular_players:
                matches = [p for p in self.all_players 
                          if player_name.lower() == p['full_name'].lower()]
                found_players.extend(matches)
        
            return found_players
    
    def collect_player_data(self, player_names: List[str] = None, 
                          seasons: List[str] = None, 
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Collect comprehensive player game log data with robust data cleaning.
        
        Args:
            player_names: Specific players to collect data for
            seasons: Seasons to collect data for
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with cleaned player game logs
        """
        seasons = seasons or DEFAULT_SEASONS
        players_to_process = self.find_players(player_names)
        
        logger.info(f"Collecting data for {len(players_to_process)} players across {len(seasons)} seasons")
        
        all_data = []
        
        for i, player in enumerate(players_to_process):
            player_id = str(player['id'])
            player_name = player['full_name']
            
            logger.info(f"Processing {i+1}/{len(players_to_process)}: {player_name}")
            
            # Cache player info
            try:
                self.db_manager.cache_player_info(player_id, player_name, player)
            except Exception as e:
                logger.warning(f"Failed to cache player info for {player_name}: {e}")
            
            for season in seasons:
                try:
                    # Check cache first
                    if use_cache:
                        cached_data = self.db_manager.get_cached_game_logs(player_id, season)
                        if cached_data is not None and len(cached_data) >= MIN_GAMES_PLAYED:
                            # Clean cached data
                            cached_data = self._clean_and_validate_data(cached_data)
                            cached_data['PLAYER_ID'] = player_id
                            cached_data['PLAYER_NAME'] = player_name
                            cached_data['SEASON'] = season
                            all_data.append(cached_data)
                            logger.info(f"Used cached data for {player_name} - {season}")
                            continue
                    
                    # Fetch from API
                    logger.info(f"Fetching API data for {player_name} - {season}")
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        timeout=60
                    )
                    
                    games_df = gamelog.get_data_frames()[0]
                    
                    if len(games_df) < MIN_GAMES_PLAYED:
                        logger.warning(f"Insufficient games for {player_name} in {season}: {len(games_df)}")
                        continue
                    
                    # Clean and validate the data
                    games_df = self._clean_and_validate_data(games_df)
                    
                    # Add metadata
                    games_df['PLAYER_ID'] = player_id
                    games_df['PLAYER_NAME'] = player_name
                    games_df['SEASON'] = season
                    
                    # Cache the cleaned data with better error handling
                    for _, game in games_df.iterrows():
                        try:
                            # Convert the row to a dictionary and handle any datetime objects
                            game_dict = game.to_dict()
                            
                            # Ensure GAME_DATE is properly formatted for caching
                            if 'GAME_DATE' in game_dict and pd.notna(game_dict['GAME_DATE']):
                                if isinstance(game_dict['GAME_DATE'], pd.Timestamp):
                                    game_date_str = game_dict['GAME_DATE'].strftime('%Y-%m-%d')
                                else:
                                    game_date_str = str(game_dict['GAME_DATE'])
                            else:
                                game_date_str = str(game.get('GAME_DATE', ''))
                            
                            self.db_manager.cache_game_log(
                                player_id, 
                                str(game.get('Game_ID', '')),
                                game_date_str,
                                season,
                                game_dict
                            )
                        except Exception as cache_error:
                            logger.warning(f"Failed to cache game data: {cache_error}")
                            # Continue without caching this game
                            continue
                    
                    all_data.append(games_df)
                    
                    # Rate limiting
                    time.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {player_name} - {season}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data collected for any players")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Final data validation and cleaning
        combined_data = self._final_data_validation(combined_data)
        
        logger.info(f"Collected {len(combined_data)} game records for {combined_data['PLAYER_NAME'].nunique()} players")
        
        return combined_data

    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate NBA game log data to ensure proper data types.
        
        Args:
            df: Raw game log DataFrame
            
        Returns:
            Cleaned DataFrame with proper data types
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Define expected numeric columns from NBA API
        counting_stats = [
            'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
        ]
        
        percentage_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        
        other_numeric = ['PLUS_MINUS', 'Game_ID']
        
        # Clean counting statistics
        for col in counting_stats:
            if col in df.columns:
                # Convert to numeric, replace non-numeric with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN with 0 for counting stats
                df[col] = df[col].fillna(0).astype(int)
        
        # Clean percentage statistics
        for col in percentage_stats:
            if col in df.columns:
                # Convert to numeric, replace non-numeric with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN with 0 for percentages
                df[col] = df[col].fillna(0).astype(float)
                # Ensure percentages are between 0 and 1
                df[col] = df[col].clip(0, 1)
        
        # Clean other numeric columns
        for col in other_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'PLUS_MINUS':
                    df[col] = df[col].fillna(0).astype(float)
                else:
                    df[col] = df[col].fillna(0).astype(int)
        
        # Handle minutes - convert from "MM:SS" format if needed
        if 'MIN' in df.columns:
            df['MIN'] = self._convert_minutes_to_decimal(df['MIN'])
        
        # Clean string columns
        string_cols = ['MATCHUP', 'WL']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
        
        # Validate game date
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            # Remove rows with invalid dates
            df = df.dropna(subset=['GAME_DATE'])
        
        return df

    def _convert_minutes_to_decimal(self, minutes_series: pd.Series) -> pd.Series:
        """
        Convert minutes from MM:SS format to decimal minutes.
        
        Args:
            minutes_series: Series containing minutes data
            
        Returns:
            Series with decimal minutes
        """
        def convert_single_minute(minute_val):
            try:
                if pd.isna(minute_val):
                    return 0.0
                
                minute_str = str(minute_val)
                
                # If already a number, return it
                try:
                    return float(minute_str)
                except ValueError:
                    pass
                
                # If in MM:SS format
                if ':' in minute_str:
                    parts = minute_str.split(':')
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        return minutes + (seconds / 60.0)
                
                # If just minutes
                return float(minute_str)
                
            except (ValueError, TypeError, AttributeError):
                return 0.0
        
        return minutes_series.apply(convert_single_minute)

    def _final_data_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform final validation and cleaning on combined dataset.
        
        Args:
            df: Combined DataFrame from all players
            
        Returns:
            Final cleaned DataFrame
        """
        if df.empty:
            return df
        
        logger.info("Performing final data validation...")
        
        # Remove duplicate games
        initial_count = len(df)
        df = df.drop_duplicates(subset=['PLAYER_ID', 'Game_ID'], keep='first')
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate games")
        
        # Ensure we have essential columns
        required_cols = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'PTS', 'SEASON']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing essential data
        initial_count = len(df)
        df = df.dropna(subset=['PLAYER_ID', 'GAME_DATE', 'PTS'])
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with missing essential data")
        
        # Validate points are reasonable (0-100 range)
        unreasonable_points = df[(df['PTS'] < 0) | (df['PTS'] > 100)]
        if len(unreasonable_points) > 0:
            logger.warning(f"Found {len(unreasonable_points)} games with unreasonable point totals")
            df = df[(df['PTS'] >= 0) & (df['PTS'] <= 100)]
        
        # Validate minutes are reasonable (0-60 range)
        if 'MIN' in df.columns:
            unreasonable_minutes = df[(df['MIN'] < 0) | (df['MIN'] > 60)]
            if len(unreasonable_minutes) > 0:
                logger.warning(f"Found {len(unreasonable_minutes)} games with unreasonable minutes")
                df.loc[df['MIN'] > 60, 'MIN'] = 60  # Cap at 60 minutes
                df.loc[df['MIN'] < 0, 'MIN'] = 0    # Floor at 0 minutes
        
        # Sort by player and date
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Log data quality summary
        self._log_data_quality_summary(df)
        
        return df

    def _log_data_quality_summary(self, df: pd.DataFrame):
        """Log summary of data quality metrics."""
        logger.info("=== DATA QUALITY SUMMARY ===")
        logger.info(f"Total games: {len(df)}")
        logger.info(f"Players: {df['PLAYER_NAME'].nunique()}")
        logger.info(f"Seasons: {df['SEASON'].nunique()}")
        logger.info(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        
        # Check for missing values in key columns
        key_columns = ['PTS', 'MIN', 'FGA', 'FG_PCT', 'REB', 'AST']
        for col in key_columns:
            if col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                logger.info(f"{col} missing: {missing_pct:.1f}%")
        
        # Points distribution
        logger.info(f"Points - Mean: {df['PTS'].mean():.1f}, Median: {df['PTS'].median():.1f}, Std: {df['PTS'].std():.1f}")
    
    def get_team_data(self, team_id: str, season: str) -> pd.DataFrame:
        """
        Get team statistics for a specific season.
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., '2023-24')
            
        Returns:
            DataFrame with team game logs
        """
        try:
            teamlog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            return teamlog.get_data_frames()[0]
        except Exception as e:
            logger.error(f"Error getting team data for {team_id} - {season}: {e}")
            return pd.DataFrame()