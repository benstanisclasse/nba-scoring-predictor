# -*- coding: utf-8 -*-
"""
Database utilities for NBA Scoring Predictor
"""
import sqlite3
import pandas as pd
import json
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
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
    
    def _serialize_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert pandas/numpy objects to JSON-serializable format."""
        serialized = {}
        for key, value in data.items():
            if pd.isna(value):
                serialized[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                serialized[key] = value.strftime('%Y-%m-%d') if not pd.isna(value) else None
            elif isinstance(value, np.integer):
                serialized[key] = int(value)
            elif isinstance(value, np.floating):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                serialized[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            else:
                serialized[key] = value
        return serialized
    
    def cache_game_log(self, player_id: str, game_id: str, game_date: str, 
                      season: str, data: Dict[str, Any]):
        """Cache game log data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize the data to handle pandas/numpy objects
            serialized_data = self._serialize_for_json(data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO game_logs 
                (player_id, game_id, game_date, season, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (player_id, game_id, game_date, season, json.dumps(serialized_data)))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error caching game log: {e}")
        finally:
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
                data_list = []
                for row in result['data']:
                    try:
                        parsed_data = json.loads(row)
                        data_list.append(parsed_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing cached data: {e}")
                        continue
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    
                    # Convert date strings back to datetime
                    if 'GAME_DATE' in df.columns:
                        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
                    
                    return df
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
        finally:
            conn.close()
        
        return None
    
    def cache_player_info(self, player_id: str, player_name: str, data: Dict[str, Any]):
        """Cache player information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize the data to handle pandas/numpy objects
            serialized_data = self._serialize_for_json(data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO player_info 
                (player_id, player_name, data)
                VALUES (?, ?, ?)
            ''', (player_id, player_name, json.dumps(serialized_data)))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error caching player info: {e}")
        finally:
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