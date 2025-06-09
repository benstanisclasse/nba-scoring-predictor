# -*- coding: utf-8 -*-
"""
Player storage utilities for NBA Scoring Predictor
"""
import json
import os
from typing import List, Set
from datetime import datetime
from utils.logger import main_logger as logger

class PlayerStorage:
    """Manages storage and retrieval of trained players."""
    
    def __init__(self, storage_path: str = "data/trained_players.json"):
        self.storage_path = storage_path
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self):
        """Ensure the storage file exists."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            self._save_data({
                "trained_players": [],
                "popular_players": [
                    "LeBron James", "Stephen Curry", "Luka Dončić", 
                    "Giannis Antetokounmpo", "Jayson Tatum", "Kevin Durant",
                    "Nikola Jokić", "Joel Embiid", "Damian Lillard",
                    "Anthony Davis", "Kawhi Leonard", "Jimmy Butler",
                    "Devin Booker", "Ja Morant", "Zion Williamson"
                ],
                "last_updated": datetime.now().isoformat()
            })
    
    def _load_data(self) -> dict:
        """Load data from storage file."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading player data: {e}")
            return {"trained_players": [], "popular_players": []}
    
    def _save_data(self, data: dict):
        """Save data to storage file."""
        try:
            data["last_updated"] = datetime.now().isoformat()
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving player data: {e}")
    
    def get_all_players(self) -> List[str]:
        """Get all available players (trained + popular)."""
        data = self._load_data()
        trained = data.get("trained_players", [])
        popular = data.get("popular_players", [])
        
        # Combine and remove duplicates while preserving order
        all_players = []
        seen = set()
        
        # Add trained players first
        for player in trained:
            if player not in seen:
                all_players.append(player)
                seen.add(player)
        
        # Add popular players
        for player in popular:
            if player not in seen:
                all_players.append(player)
                seen.add(player)
        
        return sorted(all_players)
    
    def get_trained_players(self) -> List[str]:
        """Get only previously trained players."""
        data = self._load_data()
        return sorted(data.get("trained_players", []))
    
    def get_popular_players(self) -> List[str]:
        """Get popular/recommended players."""
        data = self._load_data()
        return data.get("popular_players", [])
    
    def add_trained_players(self, players: List[str]):
        """Add players to the trained players list."""
        if not players:
            return
        
        data = self._load_data()
        trained_set = set(data.get("trained_players", []))
        
        # Add new players
        for player in players:
            player = player.strip()
            if player:
                trained_set.add(player)
        
        data["trained_players"] = sorted(list(trained_set))
        self._save_data(data)
        logger.info(f"Added {len(players)} players to trained players list")
    
    def remove_trained_player(self, player_name: str):
        """Remove a player from trained players list."""
        data = self._load_data()
        trained = data.get("trained_players", [])
        
        if player_name in trained:
            trained.remove(player_name)
            data["trained_players"] = trained
            self._save_data(data)
            logger.info(f"Removed {player_name} from trained players")
    
    def add_popular_players(self, players: List[str]):
        """Add players to the popular players list."""
        data = self._load_data()
        popular_set = set(data.get("popular_players", []))
        
        for player in players:
            player = player.strip()
            if player:
                popular_set.add(player)
        
        data["popular_players"] = sorted(list(popular_set))
        self._save_data(data)
