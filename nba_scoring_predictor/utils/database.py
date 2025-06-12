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
        """Convert pandas/numpy objects to JSON-serializable format with robust encoding handling."""
        serialized = {}
    
        for key, value in data.items():
            try:
                # Handle None/NaN values
                if value is None or pd.isna(value):
                    serialized[key] = None
                    continue
            
                # Handle datetime objects
                elif isinstance(value, (pd.Timestamp, datetime)):
                    try:
                        serialized[key] = value.strftime('%Y-%m-%d') if not pd.isna(value) else None
                    except (ValueError, AttributeError):
                        serialized[key] = str(value) if value else None
                    continue
            
                # Handle string values with encoding issues
                elif isinstance(value, str):
                    try:
                        # First, try to encode/decode to clean up any encoding issues
                        clean_value = value.encode('utf-8', errors='replace').decode('utf-8')
                        # Remove any null bytes that might cause issues
                        clean_value = clean_value.replace('\x00', '')
                        serialized[key] = clean_value
                    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
                        # If encoding fails, convert to string and clean
                        try:
                            serialized[key] = str(value).replace('\x00', '')
                        except:
                            serialized[key] = "encoding_error"
                    continue
            
                # Handle bytes objects
                elif isinstance(value, bytes):
                    try:
                        # Try to decode as UTF-8
                        serialized[key] = value.decode('utf-8', errors='replace')
                    except (UnicodeDecodeError, AttributeError):
                        # If that fails, convert to base64 string
                        import base64
                        serialized[key] = base64.b64encode(value).decode('ascii')
                    continue
            
                # Handle numpy integers
                elif isinstance(value, np.integer):
                    serialized[key] = int(value)
                    continue
            
                # Handle numpy floats
                elif isinstance(value, np.floating):
                    if np.isnan(value) or np.isinf(value):
                        serialized[key] = None
                    else:
                        serialized[key] = float(value)
                    continue
            
                # Handle numpy arrays
                elif isinstance(value, np.ndarray):
                    try:
                        # Convert to list, handling NaN/inf values
                        array_list = value.tolist()
                        # Clean the list recursively
                        serialized[key] = self._clean_array_for_json(array_list)
                    except (ValueError, TypeError):
                        serialized[key] = str(value)
                    continue
            
                # Handle pandas Series
                elif isinstance(value, pd.Series):
                    try:
                        # Convert to dict, then serialize recursively
                        series_dict = value.to_dict()
                        serialized[key] = self._serialize_for_json(series_dict)
                    except (ValueError, TypeError):
                        serialized[key] = str(value)
                    continue
            
                # Handle pandas DataFrame
                elif isinstance(value, pd.DataFrame):
                    try:
                        # Convert to dict format
                        df_dict = value.to_dict('records')
                        serialized[key] = self._clean_array_for_json(df_dict)
                    except (ValueError, TypeError):
                        serialized[key] = f"DataFrame_shape_{value.shape}"
                    continue
            
                # Handle boolean values
                elif isinstance(value, (bool, np.bool_)):
                    serialized[key] = bool(value)
                    continue
            
                # Handle numeric types (int, float)
                elif isinstance(value, (int, float)):
                    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        serialized[key] = None
                    else:
                        serialized[key] = value
                    continue
            
                # Handle lists and tuples
                elif isinstance(value, (list, tuple)):
                    try:
                        serialized[key] = self._clean_array_for_json(list(value))
                    except (ValueError, TypeError):
                        serialized[key] = str(value)
                    continue
            
                # Handle dictionaries
                elif isinstance(value, dict):
                    try:
                        serialized[key] = self._serialize_for_json(value)
                    except (ValueError, TypeError):
                        serialized[key] = str(value)
                    continue
            
                # Handle complex numbers
                elif isinstance(value, complex):
                    serialized[key] = {"real": value.real, "imag": value.imag}
                    continue
            
                # Handle any other object types
                else:
                    try:
                        # Try JSON serialization test
                        import json
                        json.dumps(value)
                        serialized[key] = value
                    except (TypeError, ValueError):
                        # If not JSON serializable, convert to string
                        try:
                            str_value = str(value)
                            # Clean string of problematic characters
                            clean_str = str_value.encode('utf-8', errors='replace').decode('utf-8')
                            serialized[key] = clean_str.replace('\x00', '')
                        except:
                            serialized[key] = f"object_type_{type(value).__name__}"
        
            except Exception as e:
                # Ultimate fallback for any serialization error
                logger.warning(f"Serialization error for key '{key}' (type: {type(value).__name__}): {e}")
                try:
                    # Try one more time with string conversion
                    fallback_value = str(value) if value is not None else None
                    if fallback_value:
                        # Clean the string
                        clean_fallback = fallback_value.encode('utf-8', errors='replace').decode('utf-8')
                        serialized[key] = clean_fallback.replace('\x00', '')
                    else:
                        serialized[key] = None
                except:
                    # Absolute last resort
                    serialized[key] = f"serialization_failed_{type(value).__name__}"
    
        return serialized

    def _clean_array_for_json(self, array_data):
        """Recursively clean array data for JSON serialization."""
        if isinstance(array_data, list):
            cleaned = []
            for item in array_data:
                if isinstance(item, (list, tuple)):
                    cleaned.append(self._clean_array_for_json(list(item)))
                elif isinstance(item, dict):
                    cleaned.append(self._serialize_for_json(item))
                elif isinstance(item, (np.integer, np.floating)):
                    if isinstance(item, np.floating) and (np.isnan(item) or np.isinf(item)):
                        cleaned.append(None)
                    else:
                        cleaned.append(float(item) if isinstance(item, np.floating) else int(item))
                elif isinstance(item, str):
                    # Clean string encoding
                    try:
                        clean_item = item.encode('utf-8', errors='replace').decode('utf-8')
                        cleaned.append(clean_item.replace('\x00', ''))
                    except:
                        cleaned.append(str(item))
                elif pd.isna(item):
                    cleaned.append(None)
                else:
                    try:
                        # Test JSON serializability
                        import json
                        json.dumps(item)
                        cleaned.append(item)
                    except:
                        cleaned.append(str(item))
            return cleaned
    
        elif isinstance(array_data, dict):
            return self._serialize_for_json(array_data)
    
        else:
            # Single item
            if isinstance(array_data, (np.integer, np.floating)):
                if isinstance(array_data, np.floating) and (np.isnan(array_data) or np.isinf(array_data)):
                    return None
                else:
                    return float(array_data) if isinstance(array_data, np.floating) else int(array_data)
            elif isinstance(array_data, str):
                try:
                    clean_item = array_data.encode('utf-8', errors='replace').decode('utf-8')
                    return clean_item.replace('\x00', '')
                except:
                    return str(array_data)
            elif pd.isna(array_data):
                return None
            else:
                try:
                    import json
                    json.dumps(array_data)
                    return array_data
                except:
                    return str(array_data)
    
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