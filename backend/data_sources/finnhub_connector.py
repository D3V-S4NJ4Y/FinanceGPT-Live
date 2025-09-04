"""
📊 Finnhub Data Connector
=========================

High-performance connector for real-time market data from Finnhub API.
Primary data source when Yahoo Finance fails.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time

from core.config import settings

logger = logging.getLogger(__name__)

class FinnhubConnector:
    """
    🚀 Finnhub API Data Connector
    
    Features:
    - Real-time market quotes
    - Company profiles
    - Market news
    - Financial metrics
    - High reliability fallback
    """
    
    def __init__(self):
        self.base_url = "https://finnhub.io/api/v1"
        self.api_key = settings.finnhub_key
        self.session = None
        self.last_update = {}
        self.data_cache = {}
        
        if not self.api_key:
            logger.warning("⚠️ Finnhub API key not configured")
        else:
            logger.info("✅ FinnhubConnector initialized")
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        if not self.api_key:
            raise ValueError("Finnhub API key not configured")
        
        try:
            session = await self.get_session()
            url = f"{self.base_url}/quote"
            params = {
                'symbol': symbol.upper(),
                'token': self.api_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert Finnhub response to our standard format
                    current_price = data.get('c', 0)
                    previous_close = data.get('pc', current_price)
                    change = current_price - previous_close
                    change_percent = (change / previous_close * 100) if previous_close > 0 else 0
                    
                    return {
                        'symbol': symbol.upper(),
                        'price': current_price,
                        'change': change,
                        'changePercent': change_percent,
                        'open': data.get('o', current_price),
                        'high': data.get('h', current_price),
                        'low': data.get('l', current_price),
                        'previousClose': previous_close,
                        'volume': 0,  # Volume not available in this endpoint
                        'timestamp': datetime.now().isoformat(),
                        'source': 'finnhub'
                    }
                else:
                    logger.error(f"Finnhub API error for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Finnhub quote for {symbol}: {e}")
            return None
    
    async def get_multiple_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get quotes for multiple symbols"""
        if not self.api_key:
            logger.warning("Finnhub API key not configured, using fallback data")
            return self._generate_fallback_data(symbols)
        
        quotes = []
        
        # Process symbols in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_quotes = await asyncio.gather(
                *[self.get_quote(symbol) for symbol in batch],
                return_exceptions=True
            )
            
            for quote in batch_quotes:
                if isinstance(quote, dict) and quote:
                    quotes.append(quote)
            
            # Rate limiting - wait between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.2)
        
        # If no quotes received, return fallback data
        if not quotes:
            logger.warning("No quotes from Finnhub, using fallback data")
            return self._generate_fallback_data(symbols)
        
        return quotes
    
    def _generate_fallback_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate realistic fallback data when APIs fail"""
        fallback_data = []
        
        # Realistic stock prices and movements
        stock_data = {
            'AAPL': {'base_price': 175.00, 'volatility': 0.02},
            'GOOGL': {'base_price': 135.00, 'volatility': 0.025},
            'MSFT': {'base_price': 415.00, 'volatility': 0.018},
            'AMZN': {'base_price': 145.00, 'volatility': 0.03},
            'TSLA': {'base_price': 250.00, 'volatility': 0.045},
            'NVDA': {'base_price': 480.00, 'volatility': 0.035},
            'META': {'base_price': 510.00, 'volatility': 0.028},
            'NFLX': {'base_price': 445.00, 'volatility': 0.025},
            'SPY': {'base_price': 445.00, 'volatility': 0.015},
            'QQQ': {'base_price': 385.00, 'volatility': 0.02}
        }
        
        for symbol in symbols:
            if symbol in stock_data:
                base = stock_data[symbol]
                
                # Generate realistic price movement
                import random
                price_change_pct = random.uniform(-base['volatility'], base['volatility'])
                current_price = base['base_price'] * (1 + price_change_pct)
                change = current_price - base['base_price']
                
                fallback_data.append({
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'changePercent': round(price_change_pct * 100, 2),
                    'open': round(base['base_price'] * (1 + random.uniform(-0.01, 0.01)), 2),
                    'high': round(current_price * (1 + random.uniform(0, 0.02)), 2),
                    'low': round(current_price * (1 - random.uniform(0, 0.02)), 2),
                    'previousClose': base['base_price'],
                    'volume': random.randint(10000000, 100000000),
                    'marketCap': round(current_price * random.randint(1000000000, 10000000000), 2),
                    'high24h': round(current_price * 1.05, 2),
                    'low24h': round(current_price * 0.95, 2),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fallback'
                })
        
        return fallback_data
    
    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile data"""
        if not self.api_key:
            return {}
        
        try:
            session = await self.get_session()
            url = f"{self.base_url}/stock/profile2"
            params = {
                'symbol': symbol.upper(),
                'token': self.api_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {}

# Global instance
finnhub_connector = FinnhubConnector()
