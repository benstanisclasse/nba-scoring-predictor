"""
NBA Player Fetcher - Enhanced with better error handling and rate limiting
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonteamroster, commonplayerinfo
from utils.logger import main_logger as logger

class NBAPlayerFetcher:
    """Enhanced NBA Player Fetcher with better error handling."""
    
    def __init__(self, storage_path: str = "data/nba_players_categorized.json"):
        self.storage_path = storage_path
        self.all_players = players.get_players()
        self.all_teams = teams.get_teams()
        
        # Enhanced position mapping
        self.position_mapping = {
            'Point Guard': 'PG', 'Shooting Guard': 'SG', 'Small Forward': 'SF',
            'Power Forward': 'PF', 'Center': 'C', 'Guard': 'SG', 'Forward': 'SF',
            'Guard-Forward': 'SG', 'Forward-Guard': 'SF', 'Forward-Center': 'PF',
            'Center-Forward': 'C', 'G': 'SG', 'F': 'SF', 'C': 'C'
        }
        
        # Setup retry strategy
        self.setup_session()
    
    def setup_session(self):
        """Setup requests session with retry strategy."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def fetch_all_active_players(self, current_season: str = "2024-25") -> Dict:
        """
        Enhanced fetch with better error handling and fallback strategies.
        """
        logger.info(f"Fetching NBA players for {current_season} season...")
        
        active_players = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'season': current_season,
                'total_players': 0,
                'total_teams': len(self.all_teams),
                'fetch_method': 'hybrid'
            },
            'players_by_position': {
                'PG': [], 'SG': [], 'SF': [], 'PF': [], 'C': [], 'UNKNOWN': []
            },
            'players_by_team': {},
            'all_players': []
        }
        
        # Try multiple approaches
        success = False
        
        # Method 1: Try current season rosters
        try:
            success = self._fetch_by_team_rosters(active_players, current_season)
            if success:
                logger.info("Successfully fetched using team rosters method")
        except Exception as e:
            logger.warning(f"Team rosters method failed: {e}")
        
        # Method 2: Fallback to previous season if current fails
        if not success and current_season == "2024-25":
            try:
                logger.info("Trying previous season as fallback...")
                success = self._fetch_by_team_rosters(active_players, "2023-24")
                if success:
                    active_players['metadata']['season'] = "2023-24 (fallback)"
                    active_players['metadata']['fetch_method'] = 'fallback_season'
            except Exception as e:
                logger.warning(f"Previous season fallback failed: {e}")
        
        # Method 3: Use static player list with position estimation
        if not success:
            logger.info("Using static player list with position estimation...")
            success = self._fetch_static_with_estimation(active_players)
            if success:
                active_players['metadata']['fetch_method'] = 'static_estimation'
        
        if success:
            self._save_players_data(active_players)
            self._log_summary(active_players)
            return active_players
        else:
            raise Exception("All fetch methods failed. Check NBA API connectivity.")
    
    def _fetch_by_team_rosters(self, active_players: Dict, season: str) -> bool:
        """Fetch using team rosters with enhanced error handling."""
        processed_players = 0
        failed_teams = []
        
        for i, team in enumerate(self.all_teams):
            team_id = team['id']
            team_name = team['full_name']
            team_abbrev = team['abbreviation']
            
            logger.info(f"Processing team {i+1}/{len(self.all_teams)}: {team_name}")
            
            try:
                # Longer delay between teams to avoid rate limiting
                if i > 0:
                    time.sleep(2.0)
                
                # Get team roster with timeout
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=season,
                    timeout=60
                )
                
                roster_df = roster.get_data_frames()[0]
                
                if roster_df.empty:
                    logger.warning(f"Empty roster for {team_name}")
                    failed_teams.append(team_name)
                    continue
                
                team_players = []
                
                # Process each player with minimal API calls
                for _, player_row in roster_df.iterrows():
                    try:
                        player_data = self._process_player_simple(
                            player_row, team_name, team_abbrev
                        )
                        if player_data:
                            team_players.append(player_data)
                            active_players['all_players'].append(player_data)
                            
                            # Categorize by position
                            position = player_data['position']
                            active_players['players_by_position'][position].append(player_data)
                            processed_players += 1
                        
                        # Small delay between players
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Error processing player {player_row.get('PLAYER', 'Unknown')}: {e}")
                        continue
                
                active_players['players_by_team'][team_abbrev] = {
                    'team_name': team_name,
                    'players': team_players
                }
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i+1}/{len(self.all_teams)} teams, {processed_players} players")
                
            except Exception as e:
                logger.error(f"Failed to process team {team_name}: {e}")
                failed_teams.append(team_name)
                continue
        
        active_players['metadata']['total_players'] = processed_players
        active_players['metadata']['failed_teams'] = failed_teams
        
        # Consider success if we got data for most teams
        success_rate = (len(self.all_teams) - len(failed_teams)) / len(self.all_teams)
        return success_rate > 0.7 and processed_players > 100
    
    def _process_player_simple(self, player_row: pd.Series, team_name: str, team_abbrev: str) -> Optional[Dict]:
        """Process player with minimal API calls and fallback data."""
        try:
            player_id = str(player_row['PLAYER_ID'])
            player_name = player_row['PLAYER']
            
            # Use roster data directly when possible
            player_data = {
                'player_id': player_id,
                'name': player_name,
                'position': self._normalize_position(player_row.get('POSITION', 'Unknown')),
                'raw_position': player_row.get('POSITION', 'Unknown'),
                'team': team_name,
                'team_abbrev': team_abbrev,
                'jersey_number': str(player_row.get('NUM', '')),
                'height': player_row.get('HEIGHT', ''),
                'weight': str(player_row.get('WEIGHT', '')),
                'age': self._calculate_age_from_birthdate(player_row.get('BIRTH_DATE', '')),
                'experience': int(player_row.get('EXP', 0)) if pd.notna(player_row.get('EXP', 0)) else 0,
                'school': player_row.get('SCHOOL', ''),
                'country': 'USA'  # Default, could be enhanced
            }
            
            return player_data
            
        except Exception as e:
            logger.warning(f"Error in simple player processing: {e}")
            return None
    
    def _fetch_static_with_estimation(self, active_players: Dict) -> bool:
        """Fallback method using static player data with position estimation."""
        try:
            logger.info("Using static player list as fallback...")
            
            # Get common/popular players with estimated positions
            estimated_players = self._get_estimated_current_players()
            
            for player_info in estimated_players:
                active_players['all_players'].append(player_info)
                position = player_info['position']
                active_players['players_by_position'][position].append(player_info)
            
            active_players['metadata']['total_players'] = len(estimated_players)
            active_players['metadata']['warning'] = "Using estimated data due to API issues"
            
            return len(estimated_players) > 50
            
        except Exception as e:
            logger.error(f"Static estimation method failed: {e}")
            return False
    
    def _get_estimated_current_players(self) -> List[Dict]:
        """Get estimated current players with positions."""
        # This is a fallback list of known current players
        estimated_players = [
            {'player_id': '2544', 'name': 'LeBron James', 'position': 'SF', 'team': 'Los Angeles Lakers', 'team_abbrev': 'LAL'},
            {'player_id': '201939', 'name': 'Stephen Curry', 'position': 'PG', 'team': 'Golden State Warriors', 'team_abbrev': 'GSW'},
            {'player_id': '1629029', 'name': 'Luka Dončić', 'position': 'PG', 'team': 'Dallas Mavericks', 'team_abbrev': 'DAL'},
            {'player_id': '203507', 'name': 'Giannis Antetokounmpo', 'position': 'PF', 'team': 'Milwaukee Bucks', 'team_abbrev': 'MIL'},
            {'player_id': '1628369', 'name': 'Jayson Tatum', 'position': 'SF', 'team': 'Boston Celtics', 'team_abbrev': 'BOS'},
            {'player_id': '201142', 'name': 'Kevin Durant', 'position': 'SF', 'team': 'Phoenix Suns', 'team_abbrev': 'PHX'},
            {'player_id': '203999', 'name': 'Nikola Jokić', 'position': 'C', 'team': 'Denver Nuggets', 'team_abbrev': 'DEN'},
            {'player_id': '203954', 'name': 'Joel Embiid', 'position': 'C', 'team': 'Philadelphia 76ers', 'team_abbrev': 'PHI'},
            {'player_id': '203081', 'name': 'Damian Lillard', 'position': 'PG', 'team': 'Milwaukee Bucks', 'team_abbrev': 'MIL'},
            {'player_id': '203076', 'name': 'Anthony Davis', 'position': 'PF', 'team': 'Los Angeles Lakers', 'team_abbrev': 'LAL'},
            # Add more players as needed
        ]
        
        # Add default fields
        for player in estimated_players:
            player.update({
                'raw_position': player['position'],
                'jersey_number': '',
                'height': '',
                'weight': '',
                'age': None,
                'experience': 5,  # Estimated
                'school': '',
                'country': 'USA'
            })
        
        return estimated_players
    
    def _calculate_age_from_birthdate(self, birthdate_str: str) -> Optional[int]:
        """Calculate age from birthdate string."""
        if not birthdate_str or pd.isna(birthdate_str):
            return None
        
        try:
            # Handle different date formats
            if 'T' in str(birthdate_str):
                birthdate = datetime.fromisoformat(str(birthdate_str).split('T')[0])
            else:
                birthdate = datetime.strptime(str(birthdate_str), '%Y-%m-%d')
            
            today = datetime.now()
            age = today.year - birthdate.year
            
            if today.month < birthdate.month or (today.month == birthdate.month and today.day < birthdate.day):
                age -= 1
                
            return age
        except:
            return None
    
    def _normalize_position(self, raw_position: str) -> str:
        """Enhanced position normalization."""
        if not raw_position or pd.isna(raw_position) or raw_position == 'Unknown':
            return 'UNKNOWN'
        
        position = str(raw_position).strip()
        
        # Direct mapping
        if position in self.position_mapping:
            return self.position_mapping[position]
        
        # Keyword-based mapping
        position_lower = position.lower()
        
        if any(keyword in position_lower for keyword in ['point', 'pg']):
            return 'PG'
        elif any(keyword in position_lower for keyword in ['shooting', 'sg']):
            return 'SG'
        elif any(keyword in position_lower for keyword in ['small', 'sf']):
            return 'SF'
        elif any(keyword in position_lower for keyword in ['power', 'pf']):
            return 'PF'
        elif any(keyword in position_lower for keyword in ['center', 'c']):
            return 'C'
        elif 'guard' in position_lower:
            return 'SG'  # Default guard to SG
        elif 'forward' in position_lower:
            return 'SF'  # Default forward to SF
        
        return 'UNKNOWN'
    
    def _save_players_data(self, data: Dict):
        """Enhanced save with backup."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Create backup of existing file
            if os.path.exists(self.storage_path):
                backup_path = self.storage_path.replace('.json', '_backup.json')
                os.rename(self.storage_path, backup_path)
            
            # Save new data
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Players data saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving players data: {e}")
            # Restore backup if save failed
            backup_path = self.storage_path.replace('.json', '_backup.json')
            if os.path.exists(backup_path):
                os.rename(backup_path, self.storage_path)
                logger.info("Restored backup file due to save failure")
    
    def _log_summary(self, data: Dict):
        """Enhanced logging with diagnostic info."""
        logger.info("=== NBA PLAYERS FETCH SUMMARY ===")
        logger.info(f"Total players: {data['metadata']['total_players']}")
        logger.info(f"Season: {data['metadata']['season']}")
        logger.info(f"Fetch method: {data['metadata']['fetch_method']}")
        
        for position, players in data['players_by_position'].items():
            if players:  # Only show positions with players
                logger.info(f"{position}: {len(players)} players")
        
        logger.info(f"Teams processed: {len(data['players_by_team'])}")
        
        if 'failed_teams' in data['metadata'] and data['metadata']['failed_teams']:
            logger.warning(f"Failed teams: {data['metadata']['failed_teams']}")
        
        if 'warning' in data['metadata']:
            logger.warning(f"Warning: {data['metadata']['warning']}")

    # Keep all other existing methods unchanged...