import sqlite3
import json
import time
from typing import Dict, Optional
import threading

class MarketDataCache:
    def __init__(self, db_path: str = "market_cache.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            ''')
            conn.commit()
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT data FROM market_data WHERE symbol = ? AND expires_at > ?',
                    (symbol, time.time())
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        return None
    
    def set_market_data(self, symbol: str, data: Dict, ttl_seconds: int = 30):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                expires_at = time.time() + ttl_seconds
                conn.execute(
                    'INSERT OR REPLACE INTO market_data (symbol, data, timestamp, expires_at) VALUES (?, ?, ?, ?)',
                    (symbol, json.dumps(data), time.time(), expires_at)
                )
                conn.commit()
    
    def get_agent_data(self, key: str) -> Optional[Dict]:
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT data FROM agent_cache WHERE key = ? AND expires_at > ?',
                    (key, time.time())
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        return None
    
    def set_agent_data(self, key: str, data: Dict, ttl_seconds: int = 60):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                expires_at = time.time() + ttl_seconds
                conn.execute(
                    'INSERT OR REPLACE INTO agent_cache (key, data, timestamp, expires_at) VALUES (?, ?, ?, ?)',
                    (key, json.dumps(data), time.time(), expires_at)
                )
                conn.commit()

cache = MarketDataCache()