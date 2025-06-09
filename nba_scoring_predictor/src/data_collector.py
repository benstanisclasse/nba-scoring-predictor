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
        Find players by name or return active players.
        
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
                found_players.extend(matches)
            return found_players
        else:
            # Return a subset of players for demo (in production, implement better filtering)
            return self.all_players[:100]
    
    def collect_player_data(self, player_names: List[str] = None, 
                          seasons: List[str] = None, 
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Collect comprehensive player game log data.
        
        Args:
            player_names: Specific players to collect data for
            seasons: Seasons to collect data for
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with player game logs
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
            self.db_manager.cache_player_info(player_id, player_name, player)
            
            for season in seasons:
                try:
                    # Check cache first
                    if use_cache:
                        cached_data = self.db_manager.get_cached_game_logs(player_id, season)
                        if cached_data is not None and len(cached_data) >= MIN_GAMES_PLAYED:
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
                        timeout=30
                    )
                    
                    games_df = gamelog.get_data_frames()[0]
                    
                    if len(games_df) < MIN_GAMES_PLAYED:
                        logger.warning(f"Insufficient games for {player_name} in {season}: {len(games_df)}")
                        continue
                    
                    # Add metadata
                    games_df['PLAYER_ID'] = player_id
                    games_df['PLAYER_NAME'] = player_name
                    games_df['SEASON'] = season
                    
                    # Cache the data
                    for _, game in games_df.iterrows():
                        self.db_manager.cache_game_log(
                            player_id, 
                            str(game['Game_ID']),
                            str(game['GAME_DATE']),
                            season,
                            game.to_dict()
                        )
                    
                    all_data.append(games_df)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {player_name} - {season}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data collected for any players")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected {len(combined_data)} game records for {combined_data['PLAYER_NAME'].nunique()} players")
        
        return combined_data
    
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
            return pd.DataFrame()# -*- coding: utf-8 -*-
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
        Find players by name or return active players.
        
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
                found_players.extend(matches)
            return found_players
        else:
            # Return a subset of players for demo (in production, implement better filtering)
            return self.all_players[:100]
    
    def collect_player_data(self, player_names: List[str] = None, 
                          seasons: List[str] = None, 
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Collect comprehensive player game log data.
        
        Args:
            player_names: Specific players to collect data for
            seasons: Seasons to collect data for
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with player game logs
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
            self.db_manager.cache_player_info(player_id, player_name, player)
            
            for season in seasons:
                try:
                    # Check cache first
                    if use_cache:
                        cached_data = self.db_manager.get_cached_game_logs(player_id, season)
                        if cached_data is not None and len(cached_data) >= MIN_GAMES_PLAYED:
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
                        timeout=30
                    )
                    
                    games_df = gamelog.get_data_frames()[0]
                    
                    if len(games_df) < MIN_GAMES_PLAYED:
                        logger.warning(f"Insufficient games for {player_name} in {season}: {len(games_df)}")
                        continue
                    
                    # Add metadata
                    games_df['PLAYER_ID'] = player_id
                    games_df['PLAYER_NAME'] = player_name
                    games_df['SEASON'] = season
                    
                    # Cache the data
                    for _, game in games_df.iterrows():
                        self.db_manager.cache_game_log(
                            player_id, 
                            str(game['Game_ID']),
                            str(game['GAME_DATE']),
                            season,
                            game.to_dict()
                        )
                    
                    all_data.append(games_df)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {player_name} - {season}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data collected for any players")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected {len(combined_data)} game records for {combined_data['PLAYER_NAME'].nunique()} players")
        
        return combined_data
    
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
            return pd.DataFrame()# -*- coding: utf-8 -*-
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
        Find players by name or return active players.
        
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
                found_players.extend(matches)
            return found_players
        else:
            # Return a subset of players for demo (in production, implement better filtering)
            return self.all_players[:100]
    
    def collect_player_data(self, player_names: List[str] = None, 
                          seasons: List[str] = None, 
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Collect comprehensive player game log data.
        
        Args:
            player_names: Specific players to collect data for
            seasons: Seasons to collect data for
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with player game logs
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
            self.db_manager.cache_player_info(player_id, player_name, player)
            
            for season in seasons:
                try:
                    # Check cache first
                    if use_cache:
                        cached_data = self.db_manager.get_cached_game_logs(player_id, season)
                        if cached_data is not None and len(cached_data) >= MIN_GAMES_PLAYED:
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
                        timeout=30
                    )
                    
                    games_df = gamelog.get_data_frames()[0]
                    
                    if len(games_df) < MIN_GAMES_PLAYED:
                        logger.warning(f"Insufficient games for {player_name} in {season}: {len(games_df)}")
                        continue
                    
                    # Add metadata
                    games_df['PLAYER_ID'] = player_id
                    games_df['PLAYER_NAME'] = player_name
                    games_df['SEASON'] = season
                    
                    # Cache the data
                    for _, game in games_df.iterrows():
                        self.db_manager.cache_game_log(
                            player_id, 
                            str(game['Game_ID']),
                            str(game['GAME_DATE']),
                            season,
                            game.to_dict()
                        )
                    
                    all_data.append(games_df)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {player_name} - {season}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data collected for any players")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected {len(combined_data)} game records for {combined_data['PLAYER_NAME'].nunique()} players")
        
        return combined_data
    
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