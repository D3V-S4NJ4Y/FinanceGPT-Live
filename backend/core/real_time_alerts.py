"""
🚨 Real-Time Alert System 
==========================
Generate meaningful alerts from real market data and events
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import feedparser
import aiohttp
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class MarketAlert:
    """Real-time market alert"""
    id: str
    type: str  # 'price', 'volume', 'technical', 'news', 'volatility'
    severity: str  # 'low', 'medium', 'high', 'critical'
    symbol: str
    message: str
    price_target: Optional[float] = None
    confidence: Optional[float] = None
    action: Optional[str] = None
    agent_source: str = "Real-Time Monitor"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.id is None:
            self.id = hashlib.md5(f"{self.symbol}{self.message}{self.timestamp}".encode()).hexdigest()[:12]

class RealTimeAlertSystem:
    """
    Real-time alert generation from authentic market data
    """
    
    def __init__(self):
        self.alert_history = {}
        self.price_thresholds = {}
        self.volume_baselines = {}
        self.volatility_baselines = {}
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.sec.gov/rss/news/news.xml'
        ]
        self.active_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "AMD", "INTC", "JPM", "BAC", "JNJ", "UNH", "V", "MA", "PG", "KO"
        ]
        
        logger.info("✅ RealTimeAlertSystem initialized")
    
    async def analyze_price_movement(self, symbol: str, current_data: Dict) -> List[MarketAlert]:
        """
        Analyze price movements and generate alerts
        """
        alerts = []
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            
            if hist.empty:
                return alerts
            
            current_price = current_data.get('price', hist['Close'].iloc[-1])
            
            # Calculate recent statistics
            recent_prices = hist['Close'].tail(24)  # Last 24 hours
            mean_price = recent_prices.mean()
            std_price = recent_prices.std()
            
            # Price breakout detection
            if current_price > mean_price + 2 * std_price:
                alerts.append(MarketAlert(
                    id=None,
                    type='price',
                    severity='high',
                    symbol=symbol,
                    message=f"{symbol} price breakout detected at ${current_price:.2f} (+{((current_price/mean_price-1)*100):.1f}%)",
                    price_target=current_price * 1.05,
                    confidence=85.0,
                    action="Consider taking profits or reducing position"
                ))
            
            elif current_price < mean_price - 2 * std_price:
                alerts.append(MarketAlert(
                    id=None,
                    type='price',
                    severity='high',
                    symbol=symbol,
                    message=f"{symbol} significant price drop at ${current_price:.2f} (-{((1-current_price/mean_price)*100):.1f}%)",
                    price_target=current_price * 0.95,
                    confidence=82.0,
                    action="Potential buying opportunity or risk management needed"
                ))
            
            # Support/Resistance levels
            recent_high = recent_prices.max()
            recent_low = recent_prices.min()
            
            if abs(current_price - recent_high) / recent_high < 0.01:  # Within 1% of high
                alerts.append(MarketAlert(
                    id=None,
                    type='technical',
                    severity='medium',
                    symbol=symbol,
                    message=f"{symbol} approaching resistance at ${recent_high:.2f}",
                    price_target=recent_high,
                    confidence=75.0,
                    action="Monitor for potential reversal"
                ))
            
            elif abs(current_price - recent_low) / recent_low < 0.01:  # Within 1% of low
                alerts.append(MarketAlert(
                    id=None,
                    type='technical',
                    severity='medium',
                    symbol=symbol,
                    message=f"{symbol} near support level at ${recent_low:.2f}",
                    price_target=recent_low,
                    confidence=75.0,
                    action="Potential bounce opportunity"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing price movement for {symbol}: {e}")
        
        return alerts
    
    async def analyze_volume_anomalies(self, symbol: str, current_data: Dict) -> List[MarketAlert]:
        """
        Detect volume anomalies and generate alerts
        """
        alerts = []
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10d", interval="1d")
            
            if hist.empty or len(hist) < 5:
                return alerts
            
            current_volume = current_data.get('volume', hist['Volume'].iloc[-1])
            avg_volume = hist['Volume'].tail(10).mean()
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 3:  # 3x normal volume
                alerts.append(MarketAlert(
                    id=None,
                    type='volume',
                    severity='high',
                    symbol=symbol,
                    message=f"{symbol} unusual volume spike: {current_volume:,.0f} ({volume_ratio:.1f}x normal)",
                    confidence=90.0,
                    action="Investigate news or institutional activity"
                ))
            
            elif volume_ratio > 2:  # 2x normal volume
                alerts.append(MarketAlert(
                    id=None,
                    type='volume',
                    severity='medium',
                    symbol=symbol,
                    message=f"{symbol} elevated volume: {current_volume:,.0f} ({volume_ratio:.1f}x normal)",
                    confidence=80.0,
                    action="Monitor for continued activity"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing volume for {symbol}: {e}")
        
        return alerts
    
    async def analyze_volatility(self, symbol: str, current_data: Dict) -> List[MarketAlert]:
        """
        Analyze volatility changes and generate alerts
        """
        alerts = []
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            
            if hist.empty or len(hist) < 20:
                return alerts
            
            # Calculate volatility
            returns = hist['Close'].pct_change().dropna()
            current_volatility = returns.tail(5).std() * np.sqrt(252)  # Annualized
            historical_volatility = returns.std() * np.sqrt(252)
            
            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
            
            if volatility_ratio > 1.5:  # 50% increase in volatility
                alerts.append(MarketAlert(
                    id=None,
                    type='volatility',
                    severity='high',
                    symbol=symbol,
                    message=f"{symbol} volatility spike: {current_volatility:.1%} (vs {historical_volatility:.1%} avg)",
                    confidence=85.0,
                    action="Increased risk - consider position sizing"
                ))
            
            elif current_volatility > 0.4:  # High absolute volatility (>40% annualized)
                alerts.append(MarketAlert(
                    id=None,
                    type='volatility',
                    severity='medium',
                    symbol=symbol,
                    message=f"{symbol} high volatility environment: {current_volatility:.1%} annualized",
                    confidence=75.0,
                    action="High risk/reward scenario"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing volatility for {symbol}: {e}")
        
        return alerts
    
    async def analyze_technical_signals(self, symbol: str, current_data: Dict) -> List[MarketAlert]:
        """
        Generate alerts from technical analysis signals
        """
        alerts = []
        
        try:
            # Import technical analysis from our new route
            import sys
            sys.path.append('.')
            from api.routes.technical_analysis import get_technical_analysis
            
            # Get technical analysis
            tech_analysis_response = await get_technical_analysis(symbol, "1mo")
            
            if not tech_analysis_response.get("success"):
                return alerts
            
            tech_data = tech_analysis_response["data"]
            indicators = tech_data["indicators"]
            
            # RSI alerts
            rsi = indicators.get("rsi", 50)
            if rsi < 20:
                alerts.append(MarketAlert(
                    id=None,
                    type='technical',
                    severity='high',
                    symbol=symbol,
                    message=f"{symbol} extremely oversold: RSI {rsi:.1f}",
                    confidence=88.0,
                    action="Potential reversal opportunity"
                ))
            elif rsi > 80:
                alerts.append(MarketAlert(
                    id=None,
                    type='technical',
                    severity='high',
                    symbol=symbol,
                    message=f"{symbol} extremely overbought: RSI {rsi:.1f}",
                    confidence=88.0,
                    action="Consider taking profits"
                ))
            
            # MACD crossover alerts
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)
            
            if macd > macd_signal and abs(macd - macd_signal) > 0.1:
                alerts.append(MarketAlert(
                    id=None,
                    type='technical',
                    severity='medium',
                    symbol=symbol,
                    message=f"{symbol} MACD bullish crossover detected",
                    confidence=75.0,
                    action="Potential upward momentum"
                ))
            
            # Pattern alerts
            pattern = tech_data.get("pattern_detected")
            if pattern and pattern != "Sideways":
                severity = "high" if pattern in ["Breakout", "Breakdown"] else "medium"
                alerts.append(MarketAlert(
                    id=None,
                    type='technical',
                    severity=severity,
                    symbol=symbol,
                    message=f"{symbol} pattern detected: {pattern}",
                    confidence=70.0,
                    action=f"Monitor {pattern.lower()} development"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing technical signals for {symbol}: {e}")
        
        return alerts
    
    async def fetch_news_alerts(self) -> List[MarketAlert]:
        """
        Fetch and analyze financial news for alerts
        """
        alerts = []
        
        try:
            # Yahoo Finance RSS feed
            feed = feedparser.parse('https://feeds.finance.yahoo.com/rss/2.0/headline')
            
            for entry in feed.entries[:5]:  # Check last 5 news items
                title = entry.title
                published = datetime(*entry.published_parsed[:6])
                
                # Check if news is recent (within last hour)
                if datetime.utcnow() - published > timedelta(hours=1):
                    continue
                
                # Look for market-moving keywords
                if any(keyword in title.lower() for keyword in ['earnings', 'merger', 'acquisition', 'fda', 'sec']):
                    # Try to extract symbol from news
                    symbol = None
                    for sym in self.active_symbols:
                        if sym in title or sym.lower() in title.lower():
                            symbol = sym
                            break
                    
                    if symbol:
                        alerts.append(MarketAlert(
                            id=None,
                            type='news',
                            severity='medium',
                            symbol=symbol,
                            message=f"Breaking: {title[:100]}...",
                            confidence=70.0,
                            action="Check news impact on position",
                            agent_source="News Monitor"
                        ))
                        
        except Exception as e:
            logger.error(f"Error fetching news alerts: {e}")
        
        return alerts
    
    async def generate_market_alerts(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate comprehensive market alerts
        """
        if symbols is None:
            symbols = self.active_symbols[:8]  # Limit to avoid rate limiting
        
        all_alerts = []
        
        # Get current market data for symbols
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d", interval="5m")
                
                if hist.empty:
                    continue
                
                current_data = {
                    'price': float(hist['Close'].iloc[-1]),
                    'volume': int(hist['Volume'].iloc[-1]),
                    'change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]),
                    'change_percent': float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)
                }
                
                # Run all analysis types
                symbol_alerts = []
                symbol_alerts.extend(await self.analyze_price_movement(symbol, current_data))
                symbol_alerts.extend(await self.analyze_volume_anomalies(symbol, current_data))
                symbol_alerts.extend(await self.analyze_volatility(symbol, current_data))
                symbol_alerts.extend(await self.analyze_technical_signals(symbol, current_data))
                
                # Convert to dict format for API response
                for alert in symbol_alerts:
                    all_alerts.append({
                        "id": alert.id,
                        "type": alert.type,
                        "severity": alert.severity,
                        "symbol": alert.symbol,
                        "message": alert.message,
                        "price_target": alert.price_target,
                        "confidence": alert.confidence,
                        "action": alert.action,
                        "agent_source": alert.agent_source,
                        "timestamp": alert.timestamp.isoformat()
                    })
                
                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating alerts for {symbol}: {e}")
                continue
        
        # Add news-based alerts
        news_alerts = await self.fetch_news_alerts()
        for alert in news_alerts:
            all_alerts.append({
                "id": alert.id,
                "type": alert.type,
                "severity": alert.severity,
                "symbol": alert.symbol,
                "message": alert.message,
                "confidence": alert.confidence,
                "action": alert.action,
                "agent_source": alert.agent_source,
                "timestamp": alert.timestamp.isoformat()
            })
        
        # Sort by severity and timestamp
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        all_alerts.sort(key=lambda x: (severity_order.get(x["severity"], 0), x["timestamp"]), reverse=True)
        
        return all_alerts[:20]  # Return top 20 alerts
    
    async def get_symbol_alerts(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get alerts for a specific symbol
        """
        alerts = await self.generate_market_alerts([symbol])
        return [alert for alert in alerts if alert["symbol"] == symbol]

# Global instance
alert_system = RealTimeAlertSystem()
