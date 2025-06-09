# -*- coding: utf-8 -*-
"""
NBA Player Fetcher - Automatically get all active players and categorize them by position
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import pandas as pd

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonteamroster, commonplayerinfo, leaguegamefinder
from utils.logger import main_logger as logger

class NBAPlayerFetcher:
    """Fetches and categorizes all active NBA players."""
    
    def __init__(self, storage_path: str = "data/nba_players_categorized.json"):
        self.storage_path = storage_path
        self.all_players = players.get_players()
        self.all_teams = teams.get_teams()
        
        # Position mapping for consistency
        self.position_mapping = {
            'Point Guard': 'PG',
            'Shooting Guard': 'SG', 
            'Small Forward': 'SF',
            'Power Forward': 'PF',
            'Center': 'C',
            'Guard': 'SG',  # Generic guard -> SG
            'Forward': 'SF',  # Generic forward -> SF
            'Guard-Forward': 'SG',
            'Forward-Guard': 'SF',
            'Forward-Center': 'PF',
            'Center-Forward': 'C'
        }
    
    def fetch_all_active_players(self, current_season: str = "2024-25") -> Dict:
        """
        Fetch all active NBA players and categorize them by position.
        
        Args:
            current_season: Current NBA season (e.g., '2024-25')
            
        Returns:
            Dictionary with categorized players
        """
        logger.info(f"Fetching all active NBA players for {current_season} season...")
        
        active_players = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'season': current_season,
                'total_players': 0,
                'total_teams': len(self.all_teams)
            },
            'players_by_position': {
                'PG': [],
                'SG': [],
                'SF': [], 
                'PF': [],
                'C': [],
                'UNKNOWN': []
            },
            'players_by_team': {},
            'all_players': []
        }
        
        processed_players = 0
        
        for team in self.all_teams:
            team_id = team['id']
            team_name = team['full_name']
            team_abbrev = team['abbreviation']
            
            logger.info(f"Processing team: {team_name}")
            
            try:
                # Get team roster
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=current_season
                )
                
                roster_df = roster.get_data_frames()[0]
                team_players = []
                
                for _, player_row in roster_df.iterrows():
                    try:
                        player_data = self._process_player(player_row, team_name, team_abbrev)
                        if player_data:
                            team_players.append(player_data)
                            active_players['all_players'].append(player_data)
                            
                            # Categorize by position
                            position = player_data['position']
                            active_players['players_by_position'][position].append(player_data)
                            
                            processed_players += 1
                            
                        # Rate limiting
                        time.sleep(0.3)
                        
                    except Exception as e:
                        logger.warning(f"Error processing player {player_row.get('PLAYER', 'Unknown')}: {e}")
                        continue
                
                active_players['players_by_team'][team_abbrev] = {
                    'team_name': team_name,
                    'players': team_players
                }
                
                # Rate limiting between teams
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error processing team {team_name}: {e}")
                continue
        
        active_players['metadata']['total_players'] = processed_players
        
        # Save to file
        self._save_players_data(active_players)
        
        # Log summary
        self._log_summary(active_players)
        
        return active_players
    
    def _process_player(self, player_row: pd.Series, team_name: str, team_abbrev: str) -> Optional[Dict]:
        """Process individual player data."""
        try:
            player_id = str(player_row['PLAYER_ID'])
            player_name = player_row['PLAYER']
            
            # Get additional player info
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if info_df.empty:
                return None
            
            player_details = info_df.iloc[0]
            
            # Extract position and normalize it
            raw_position = player_details.get('POSITION', 'Unknown')
            normalized_position = self._normalize_position(raw_position)
            
            player_data = {
                'player_id': player_id,
                'name': player_name,
                'position': normalized_position,
                'raw_position': raw_position,
                'team': team_name,
                'team_abbrev': team_abbrev,
                'jersey_number': player_row.get('NUM', ''),
                'height': player_details.get('HEIGHT', ''),
                'weight': player_details.get('WEIGHT', ''),
                'age': self._calculate_age(player_details.get('BIRTHDATE', '')),
                'experience': player_details.get('SEASON_EXP', 0),
                'school': player_details.get('SCHOOL', ''),
                'country': player_details.get('COUNTRY', 'USA')
            }
            
            return player_data
            
        except Exception as e:
            logger.warning(f"Error processing player details: {e}")
            return None
    
    def _normalize_position(self, raw_position: str) -> str:
        """Normalize position to standard abbreviations."""
        if not raw_position or raw_position == 'Unknown':
            return 'UNKNOWN'
        
        # Clean the position string
        position = raw_position.strip()
        
        # Direct mapping
        if position in self.position_mapping:
            return self.position_mapping[position]
        
        # Keyword-based mapping
        position_lower = position.lower()
        
        if 'point' in position_lower or position_lower in ['pg', 'point guard']:
            return 'PG'
        elif 'shooting' in position_lower or position_lower in ['sg', 'shooting guard']:
            return 'SG'
        elif 'small' in position_lower or position_lower in ['sf', 'small forward']:
            return 'SF'
        elif 'power' in position_lower or position_lower in ['pf', 'power forward']:
            return 'PF'
        elif 'center' in position_lower or position_lower in ['c']:
            return 'C'
        elif 'guard' in position_lower:
            return 'SG'  # Default guard to SG
        elif 'forward' in position_lower:
            return 'SF'  # Default forward to SF
        
        return 'UNKNOWN'
    
    def _calculate_age(self, birthdate_str: str) -> Optional[int]:
        """Calculate player age from birthdate."""
        if not birthdate_str:
            return None
        
        try:
            if 'T' in birthdate_str:
                birthdate = datetime.fromisoformat(birthdate_str.split('T')[0])
            else:
                birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d')
            
            today = datetime.now()
            age = today.year - birthdate.year
            
            if today.month < birthdate.month or (today.month == birthdate.month and today.day < birthdate.day):
                age -= 1
                
            return age
        except:
            return None
    
    def _save_players_data(self, data: Dict):
        """Save players data to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Players data saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving players data: {e}")
    
    def _log_summary(self, data: Dict):
        """Log summary of fetched data."""
        logger.info("=== NBA PLAYERS FETCH SUMMARY ===")
        logger.info(f"Total players: {data['metadata']['total_players']}")
        logger.info(f"Season: {data['metadata']['season']}")
        
        for position, players in data['players_by_position'].items():
            logger.info(f"{position}: {len(players)} players")
        
        logger.info(f"Teams processed: {len(data['players_by_team'])}")
    
    def load_players_data(self) -> Optional[Dict]:
        """Load players data from JSON file."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading players data: {e}")
            return None
    
    def update_players_data(self, force_refresh: bool = False) -> Dict:
        """
        Update players data if needed.
        
        Args:
            force_refresh: Force refresh even if data is recent
            
        Returns:
            Updated players data
        """
        existing_data = self.load_players_data()
        
        # Check if we need to update
        if not force_refresh and existing_data:
            last_updated = datetime.fromisoformat(existing_data['metadata']['last_updated'])
            days_since_update = (datetime.now() - last_updated).days
            
            if days_since_update < 7:  # Don't update if less than a week old
                logger.info(f"Players data is recent ({days_since_update} days old). Use force_refresh=True to update anyway.")
                return existing_data
        
        logger.info("Updating NBA players data...")
        return self.fetch_all_active_players()
    
    def get_players_by_position(self, position: str) -> List[Dict]:
        """Get all players of a specific position."""
        data = self.load_players_data()
        if data:
            return data['players_by_position'].get(position, [])
        return []
    
    def get_players_by_team(self, team_abbrev: str) -> List[Dict]:
        """Get all players from a specific team."""
        data = self.load_players_data()
        if data and team_abbrev in data['players_by_team']:
            return data['players_by_team'][team_abbrev]['players']
        return []
    
    def search_player(self, player_name: str) -> Optional[Dict]:
        """Search for a specific player."""
        data = self.load_players_data()
        if not data:
            return None
        
        for player in data['all_players']:
            if player_name.lower() in player['name'].lower():
                return player
        
        return None
    
    def get_position_distribution(self) -> Dict[str, int]:
        """Get distribution of players by position."""
        data = self.load_players_data()
        if not data:
            return {}
        
        return {pos: len(players) for pos, players in data['players_by_position'].items()}
