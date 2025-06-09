# -*- coding: utf-8 -*-
"""
Database utilities for NBA Scoring Predictor
"""
import sqlite3
import pandas as pd
import json
from typing import Optional, Dict, Any
from utils.logger import main_logger as logger

class DatabaseManager:
    """Manages SQLite database operations for caching NBA data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Game logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_logs (
                player_id TEXT,
                game_id TEXT,
                game_date TEXT,
                season TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, game_id)
            )
        ''')
        
        # Team stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                team_id TEXT,
                game_id TEXT,
                game_date TEXT,
                season TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, game_id)
            )
        ''')
        
        # Player info cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_info (
                player_id TEXT PRIMARY KEY,
                player_name TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                model_id TEXT PRIMARY KEY,
                model_type TEXT,
                features TEXT,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def cache_game_log(self, player_id: str, game_id: str, game_date: str, 
                      season: str, data: Dict[str, Any]):
        """Cache game log data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO game_logs 
            (player_id, game_id, game_date, season, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (player_id, game_id, game_date, season, json.dumps(data)))
        
        conn.commit()
        conn.close()
    
    def get_cached_game_logs(self, player_id: str, season: str) -> Optional[pd.DataFrame]:
        """Retrieve cached game logs for a player and season."""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT data FROM game_logs 
            WHERE player_id = ? AND season = ?
            ORDER BY game_date
        '''
        
        try:
            result = pd.read_sql_query(query, conn, params=(player_id, season))
            if not result.empty:
                # Convert JSON data back to DataFrame
                data_list = [json.loads(row) for row in result['data']]
                return pd.DataFrame(data_list)
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
        finally:
            conn.close()
        
        return None
    
    def cache_player_info(self, player_id: str, player_name: str, data: Dict[str, Any]):
        """Cache player information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO player_info 
            (player_id, player_name, data)
            VALUES (?, ?, ?)
        ''', (player_id, player_name, json.dumps(data)))
        
        conn.commit()
        conn.close()
    
    def get_all_cached_players(self) -> pd.DataFrame:
        """Get all cached players."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = 'SELECT player_id, player_name FROM player_info ORDER BY player_name'
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error retrieving players: {e}")
            return pd.DataFrame()
        finally:
            conn.close()