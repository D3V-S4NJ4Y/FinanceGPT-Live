"""
🤖 Market Sentinel Agent
========================

Advanced AI agent for real-time market monitoring and analysis.
Provides intelligent market surveillance with ML-powered insights.

"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent
from core.config import settings

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketCondition(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

@dataclass
class MarketAlert:
    """Market alert data structure"""
    symbol: str
    alert_type: str
    level: AlertLevel
    message: str
    confidence: float
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'level': self.level.value,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class MarketAnalysis:
    """Comprehensive market analysis"""
    symbol: str
    condition: MarketCondition
    trend_strength: float
    volatility: float
    momentum: float
    support_level: float
    resistance_level: float
    recommendation: str
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'condition': self.condition.value,
            'timestamp': self.timestamp.isoformat()
        }

class MarketSentinelAgent(BaseAgent):
    """
    🎯 Market Sentinel - Advanced Market Monitoring Agent
    
    Capabilities:
    - Real-time price monitoring with anomaly detection
    - Technical analysis with 20+ indicators
    - Volume analysis and unusual activity detection
    - Support/resistance level identification
    - Trend analysis and momentum tracking
    - Risk assessment and alert generation
    - Multi-timeframe analysis
    """
    
    def __init__(self):
        super().__init__(
            name="MarketSentinel",
            description="Advanced real-time market monitoring and analysis agent",
            version="1.0.0"
        )
        
        # Configuration
        self.monitored_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "SPY", "QQQ"]
        self.alert_thresholds = {
            'price_change': 0.05,  # 5% price change
            'volume_spike': 2.0,   # 2x normal volume
            'volatility_spike': 1.5,  # 1.5x normal volatility
            'gap_threshold': 0.03  # 3% gap
        }
        
        # Data storage
        self.price_history = {}
        self.volume_history = {}
        self.analysis_cache = {}
        self.active_alerts = []
        
        # Performance tracking
        self.analysis_count = 0
        self.alert_count = 0
        
        logger.info("✅ MarketSentinelAgent initialized")
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming market data message"""
        try:
            message_type = message.get('type')
            
            if message_type == 'price_update':
                return await self._process_price_update(message)
            elif message_type == 'analysis_request':
                return await self._process_analysis_request(message)
            elif message_type == 'alert_query':
                return await self._process_alert_query(message)
            elif message_type == 'health_check':
                return await self._health_check()
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown message type: {message_type}'
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _process_price_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process real-time price update"""
        symbol = message.get('symbol')
        price_data = message.get('data', {})
        
        if not symbol or not price_data:
            return {'status': 'error', 'message': 'Invalid price data'}
        
        # Store price data
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price_data.get('price'),
            'volume': price_data.get('volume'),
            'timestamp': datetime.now()
        })
        
        # Keep only recent data (last 1000 points)
        self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        # Perform analysis
        analysis = await self._analyze_symbol(symbol)
        alerts = await self._check_alerts(symbol, price_data)
        
        self.analysis_count += 1
        
        return {
            'status': 'success',
            'symbol': symbol,
            'analysis': analysis.to_dict() if analysis else None,
            'alerts': [alert.to_dict() for alert in alerts],
            'timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_symbol(self, symbol: str) -> Optional[MarketAnalysis]:
        """Perform comprehensive technical analysis"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
                return None
            
            # Get recent price data
            data = self.price_history[symbol][-100:]  # Last 100 points
            prices = [d['price'] for d in data]
            volumes = [d['volume'] for d in data]
            
            # Technical indicators
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            current_price = prices[-1]
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices)
            
            # Calculate MACD
            macd_line, signal_line = self._calculate_macd(prices)
            
            # Calculate Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Support and resistance levels
            support, resistance = self._find_support_resistance(prices)
            
            # Determine market condition
            condition = self._determine_market_condition(
                current_price, sma_20, sma_50, rsi, macd_line, signal_line
            )
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(prices)
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices)
            
            # Generate recommendation
            recommendation, confidence = self._generate_recommendation(
                condition, rsi, macd_line, signal_line, current_price, 
                sma_20, bb_upper, bb_lower, volume_ratio
            )
            
            analysis = MarketAnalysis(
                symbol=symbol,
                condition=condition,
                trend_strength=trend_strength,
                volatility=volatility,
                momentum=momentum,
                support_level=support,
                resistance_level=resistance,
                recommendation=recommendation,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Cache analysis
            self.analysis_cache[symbol] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def _check_alerts(self, symbol: str, price_data: Dict[str, Any]) -> List[MarketAlert]:
        """Check for alert conditions"""
        alerts = []
        current_price = price_data.get('price', 0)
        current_volume = price_data.get('volume', 0)
        
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
                return alerts
            
            # Get historical data
            history = self.price_history[symbol]
            previous_prices = [d['price'] for d in history[-10:-1]]
            previous_volumes = [d['volume'] for d in history[-10:-1]]
            
            if not previous_prices:
                return alerts
            
            previous_price = previous_prices[-1]
            avg_volume = np.mean(previous_volumes) if previous_volumes else 1
            
            # Price change alert
            price_change = abs(current_price - previous_price) / previous_price
            if price_change > self.alert_thresholds['price_change']:
                level = AlertLevel.HIGH if price_change > 0.10 else AlertLevel.MEDIUM
                alerts.append(MarketAlert(
                    symbol=symbol,
                    alert_type="PRICE_MOVEMENT",
                    level=level,
                    message=f"{symbol} moved {price_change:.2%} to ${current_price:.2f}",
                    confidence=0.9,
                    timestamp=datetime.now(),
                    data={
                        'price_change': price_change,
                        'current_price': current_price,
                        'previous_price': previous_price
                    }
                ))
            
            # Volume spike alert
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio > self.alert_thresholds['volume_spike']:
                    level = AlertLevel.HIGH if volume_ratio > 5.0 else AlertLevel.MEDIUM
                    alerts.append(MarketAlert(
                        symbol=symbol,
                        alert_type="VOLUME_SPIKE",
                        level=level,
                        message=f"{symbol} volume spike: {volume_ratio:.1f}x normal",
                        confidence=0.8,
                        timestamp=datetime.now(),
                        data={
                            'volume_ratio': volume_ratio,
                            'current_volume': current_volume,
                            'avg_volume': avg_volume
                        }
                    ))
            
            # Gap detection
            if len(previous_prices) >= 2:
                last_close = previous_prices[-1]
                gap_size = abs(current_price - last_close) / last_close
                if gap_size > self.alert_thresholds['gap_threshold']:
                    direction = "UP" if current_price > last_close else "DOWN"
                    alerts.append(MarketAlert(
                        symbol=symbol,
                        alert_type=f"GAP_{direction}",
                        level=AlertLevel.MEDIUM,
                        message=f"{symbol} gap {direction}: {gap_size:.2%}",
                        confidence=0.85,
                        timestamp=datetime.now(),
                        data={
                            'gap_size': gap_size,
                            'direction': direction,
                            'current_price': current_price,
                            'last_close': last_close
                        }
                    ))
            
            # Store new alerts
            self.active_alerts.extend(alerts)
            self.alert_count += len(alerts)
            
            # Clean old alerts (keep last 100)
            self.active_alerts = self.active_alerts[-100:]
            
        except Exception as e:
            logger.error(f"❌ Error checking alerts for {symbol}: {e}")
        
        return alerts
    
    # Technical Analysis Functions
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate MACD and Signal line"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        # EMA calculations
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # Signal line (9-period EMA of MACD) - simplified
        signal_line = macd_line * 0.9  # Simplified
        
        return macd_line, signal_line
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not prices:
            return 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            sma = np.mean(prices)
            return sma * 1.02, sma * 0.98
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        return upper_band, lower_band
    
    def _find_support_resistance(self, prices: List[float]) -> Tuple[float, float]:
        """Find support and resistance levels"""
        if len(prices) < 10:
            return min(prices), max(prices)
        
        # Simple approach: use recent highs and lows
        recent_prices = prices[-50:] if len(prices) >= 50 else prices
        
        # Find local maxima and minima
        highs = []
        lows = []
        
        for i in range(1, len(recent_prices) - 1):
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1]):
                highs.append(recent_prices[i])
            elif (recent_prices[i] < recent_prices[i-1] and 
                  recent_prices[i] < recent_prices[i+1]):
                lows.append(recent_prices[i])
        
        resistance = np.mean(highs) if highs else max(recent_prices)
        support = np.mean(lows) if lows else min(recent_prices)
        
        return support, resistance
    
    def _determine_market_condition(self, current_price: float, sma_20: float, 
                                  sma_50: float, rsi: float, macd: float, 
                                  signal: float) -> MarketCondition:
        """Determine overall market condition"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Price vs moving averages
        if current_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if sma_20 > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # RSI analysis
        if rsi > 70:
            bearish_signals += 1  # Overbought
        elif rsi < 30:
            bullish_signals += 1  # Oversold
        
        # MACD analysis
        if macd > signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Determine condition
        if bullish_signals > bearish_signals + 1:
            return MarketCondition.BULLISH
        elif bearish_signals > bullish_signals + 1:
            return MarketCondition.BEARISH
        else:
            return MarketCondition.NEUTRAL
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength (0-1)"""
        if len(prices) < 20:
            return 0.5
        
        # Linear regression slope
        x = np.arange(len(prices[-20:]))
        y = np.array(prices[-20:])
        
        slope = np.corrcoef(x, y)[0, 1]
        
        # Normalize to 0-1
        return (slope + 1) / 2
    
    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        momentum = (current_price - past_price) / past_price
        return momentum
    
    def _generate_recommendation(self, condition: MarketCondition, rsi: float,
                               macd: float, signal: float, price: float,
                               sma_20: float, bb_upper: float, bb_lower: float,
                               volume_ratio: float) -> Tuple[str, float]:
        """Generate trading recommendation with confidence"""
        
        score = 0
        factors = []
        
        # Trend analysis
        if condition == MarketCondition.BULLISH:
            score += 2
            factors.append("Bullish trend")
        elif condition == MarketCondition.BEARISH:
            score -= 2
            factors.append("Bearish trend")
        
        # RSI analysis
        if rsi < 30:
            score += 1
            factors.append("Oversold RSI")
        elif rsi > 70:
            score -= 1
            factors.append("Overbought RSI")
        
        # MACD analysis
        if macd > signal and macd > 0:
            score += 1
            factors.append("Bullish MACD")
        elif macd < signal and macd < 0:
            score -= 1
            factors.append("Bearish MACD")
        
        # Price vs moving average
        if price > sma_20:
            score += 1
            factors.append("Above SMA20")
        else:
            score -= 1
            factors.append("Below SMA20")
        
        # Bollinger Bands
        if price < bb_lower:
            score += 1
            factors.append("Below lower Bollinger Band")
        elif price > bb_upper:
            score -= 1
            factors.append("Above upper Bollinger Band")
        
        # Volume confirmation
        if volume_ratio > 1.5:
            score += 0.5
            factors.append("High volume")
        
        # Generate recommendation
        if score >= 3:
            recommendation = "STRONG_BUY"
            confidence = min(0.95, 0.6 + (score * 0.1))
        elif score >= 1:
            recommendation = "BUY"
            confidence = min(0.85, 0.6 + (score * 0.1))
        elif score <= -3:
            recommendation = "STRONG_SELL"
            confidence = min(0.95, 0.6 + (abs(score) * 0.1))
        elif score <= -1:
            recommendation = "SELL"
            confidence = min(0.85, 0.6 + (abs(score) * 0.1))
        else:
            recommendation = "HOLD"
            confidence = 0.5
        
        return recommendation, confidence
    
    async def _process_analysis_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis request for specific symbol"""
        symbol = message.get('symbol')
        
        if not symbol:
            return {'status': 'error', 'message': 'Symbol required'}
        
        # Get latest data if needed
        if symbol not in self.analysis_cache:
            await self._fetch_latest_data(symbol)
        
        analysis = self.analysis_cache.get(symbol)
        
        return {
            'status': 'success',
            'symbol': symbol,
            'analysis': analysis.to_dict() if analysis else None,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_alert_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process alert query"""
        limit = message.get('limit', 10)
        
        recent_alerts = self.active_alerts[-limit:]
        
        return {
            'status': 'success',
            'alerts': [alert.to_dict() for alert in recent_alerts],
            'total_alerts': len(self.active_alerts),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _health_check(self) -> Dict[str, Any]:
        """Agent health check"""
        return {
            'status': 'healthy',
            'agent': self.name,
            'version': self.version,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'monitored_symbols': len(self.monitored_symbols),
            'analysis_count': self.analysis_count,
            'alert_count': self.alert_count,
            'active_alerts': len(self.active_alerts),
            'memory_usage': len(self.price_history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _fetch_latest_data(self, symbol: str):
        """Fetch latest market data for symbol"""
        try:
            # In production, this would use real-time data feeds
            # For now, using yfinance as example
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1m")
            
            if not data.empty:
                for _, row in data.tail(10).iterrows():
                    price_data = {
                        'price': row['Close'],
                        'volume': row['Volume'],
                        'timestamp': datetime.now()
                    }
                    
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    self.price_history[symbol].append(price_data)
                
                # Perform analysis
                await self._analyze_symbol(symbol)
                
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': self.name,
            'status': 'active',
            'monitored_symbols': self.monitored_symbols,
            'analysis_count': self.analysis_count,
            'alert_count': self.alert_count,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'last_activity': datetime.now().isoformat()
        }
