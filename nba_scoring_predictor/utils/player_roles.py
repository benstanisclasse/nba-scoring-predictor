# -*- coding: utf-8 -*-
"""
Player role classification and management for NBA players - Enhanced with live data
"""
import pandas as pd
from typing import Dict, List, Optional
from utils.logger import main_logger as logger
from utils.nba_player_fetcher import NBAPlayerFetcher

class PlayerRoles:
    """Player role classification and management with live NBA data."""
    
    def __init__(self):
        self.fetcher = NBAPlayerFetcher()
        self._cached_role_mapping = None
    
    def _get_role_mapping(self) -> Dict[str, str]:
        """Get role mapping from live NBA data."""
        if self._cached_role_mapping is not None:
            return self._cached_role_mapping
        
        data = self.fetcher.load_players_data()
        if not data:
            logger.warning("No NBA players data found. Fetching...")
            data = self.fetcher.fetch_all_active_players()
        
        # Create role mapping from live data
        role_mapping = {}
        for player in data['all_players']:
            role_mapping[player['name']] = player['position']
        
        self._cached_role_mapping = role_mapping
        return role_mapping
    
    @property
    def ROLE_MAPPING(self) -> Dict[str, str]:
        """Dynamic role mapping from NBA data."""
        return self._get_role_mapping()
    
    def get_role(self, player_name: str) -> str:
        """Get player role from live data."""
        mapping = self.ROLE_MAPPING
        return mapping.get(player_name, "UNKNOWN")
    
    def get_players_by_role(self, role: str) -> List[str]:
        """Get all players of a specific role from live data."""
        data = self.fetcher.load_players_data()
        if not data:
            return []
        
        return [player['name'] for player in data['players_by_position'].get(role, [])]
    
    def refresh_data(self, force: bool = False) -> bool:
        """Refresh NBA players data."""
        try:
            self.fetcher.update_players_data(force_refresh=force)
            self._cached_role_mapping = None  # Clear cache
            return True
        except Exception as e:
            logger.error(f"Error refreshing NBA data: {e}")
            return False
    
    def get_all_active_players(self) -> List[Dict]:
        """Get all active NBA players with their info."""
        data = self.fetcher.load_players_data()
        return data['all_players'] if data else []
    
    def get_position_stats(self) -> Dict[str, int]:
        """Get statistics about position distribution."""
        return self.fetcher.get_position_distribution()
    
    # Keep existing methods for backward compatibility
    ROLE_CHARACTERISTICS = {
        "PG": {
            "primary_stats": ["AST", "PTS", "STL", "TOV"],
            "efficiency_focus": ["AST_TO_RATIO", "TRUE_SHOOTING_PCT"],
            "typical_usage": (20, 35),
            "scoring_range": (12, 30)
        },
        "SG": {
            "primary_stats": ["PTS", "FG3M", "FG_PCT", "STL"],
            "efficiency_focus": ["TRUE_SHOOTING_PCT", "EFFECTIVE_FG_PCT"],
            "typical_usage": (18, 30),
            "scoring_range": (15, 35)
        },
        "SF": {
            "primary_stats": ["PTS", "REB", "AST", "STL"],
            "efficiency_focus": ["TRUE_SHOOTING_PCT", "PTS_PER_SHOT"],
            "typical_usage": (20, 32),
            "scoring_range": (15, 30)
        },
        "PF": {
            "primary_stats": ["PTS", "REB", "BLK", "FG_PCT"],
            "efficiency_focus": ["TRUE_SHOOTING_PCT", "REB_RATE"],
            "typical_usage": (18, 28),
            "scoring_range": (12, 28)
        },
        "C": {
            "primary_stats": ["PTS", "REB", "BLK", "FG_PCT"],
            "efficiency_focus": ["FG_PCT", "REB_RATE", "BLK_RATE"],
            "typical_usage": (15, 25),
            "scoring_range": (10, 25)
        }
    }
    
    @classmethod
    def get_similar_roles(cls, role: str) -> List[str]:
        """Get roles that have similar playing styles."""
        similar_groups = {
            "PG": ["PG"],
            "SG": ["SG", "SF"],
            "SF": ["SF", "SG", "PF"],
            "PF": ["PF", "SF", "C"],
            "C": ["C", "PF"]
        }
        return similar_groups.get(role, [role])
    
    @classmethod
    def get_role_weights(cls, roles: List[str]) -> Dict[str, float]:
        """Calculate weights for different roles in training data."""
        role_counts = pd.Series(roles).value_counts()
        total_players = len(roles)
        
        weights = {}
        for role in role_counts.index:
            weight = total_players / (len(role_counts) * role_counts[role])
            weights[role] = weight
        
        return weights
