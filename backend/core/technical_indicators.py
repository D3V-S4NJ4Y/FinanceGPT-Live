"""
📊 Technical Indicators Engine
=============================
High-performance technical analysis calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Fast technical indicator calculations"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def calculate_all_indicators(ohlcv_data: pd.DataFrame) -> Dict[str, any]:
        """Calculate comprehensive technical indicators"""
        try:
            if ohlcv_data.empty or len(ohlcv_data) < 20:
                return {}
            
            high = ohlcv_data['High']
            low = ohlcv_data['Low']
            close = ohlcv_data['Close']
            volume = ohlcv_data['Volume']
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = TechnicalIndicators.sma(close, 20).iloc[-1] if len(close) >= 20 else None
            indicators['sma_50'] = TechnicalIndicators.sma(close, 50).iloc[-1] if len(close) >= 50 else None
            indicators['ema_12'] = TechnicalIndicators.ema(close, 12).iloc[-1] if len(close) >= 12 else None
            indicators['ema_26'] = TechnicalIndicators.ema(close, 26).iloc[-1] if len(close) >= 26 else None
            
            # Momentum Indicators
            rsi_series = TechnicalIndicators.rsi(close)
            indicators['rsi'] = rsi_series.iloc[-1] if not rsi_series.empty else None
            
            macd_data = TechnicalIndicators.macd(close)
            indicators['macd'] = macd_data['macd'].iloc[-1] if not macd_data['macd'].empty else None
            indicators['macd_signal'] = macd_data['signal'].iloc[-1] if not macd_data['signal'].empty else None
            indicators['macd_histogram'] = macd_data['histogram'].iloc[-1] if not macd_data['histogram'].empty else None
            
            # Volatility Indicators
            bb_data = TechnicalIndicators.bollinger_bands(close)
            indicators['bb_upper'] = bb_data['upper'].iloc[-1] if not bb_data['upper'].empty else None
            indicators['bb_middle'] = bb_data['middle'].iloc[-1] if not bb_data['middle'].empty else None
            indicators['bb_lower'] = bb_data['lower'].iloc[-1] if not bb_data['lower'].empty else None
            
            atr_series = TechnicalIndicators.atr(high, low, close)
            indicators['atr'] = atr_series.iloc[-1] if not atr_series.empty else None
            
            # Stochastic
            stoch_data = TechnicalIndicators.stochastic(high, low, close)
            indicators['stoch_k'] = stoch_data['k'].iloc[-1] if not stoch_data['k'].empty else None
            indicators['stoch_d'] = stoch_data['d'].iloc[-1] if not stoch_data['d'].empty else None
            
            # Williams %R
            williams_series = TechnicalIndicators.williams_r(high, low, close)
            indicators['williams_r'] = williams_series.iloc[-1] if not williams_series.empty else None
            
            # Volume indicators
            indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] else None
            
            # Price action
            indicators['price_change'] = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else None
            indicators['volatility'] = close.pct_change().std() * np.sqrt(252) if len(close) >= 10 else None
            
            # Support and Resistance levels
            if len(high) >= 20:
                indicators['resistance'] = high.rolling(20).max().iloc[-1]
                indicators['support'] = low.rolling(20).min().iloc[-1]
            
            # Clean up None values and convert to float
            cleaned_indicators = {}
            for key, value in indicators.items():
                if value is not None and not pd.isna(value):
                    cleaned_indicators[key] = float(value)
            
            return cleaned_indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    @staticmethod
    def get_trading_signals(indicators: Dict[str, float]) -> Dict[str, any]:
        """Generate trading signals from technical indicators"""
        try:
            signals = {
                'overall_signal': 'HOLD',
                'signal_strength': 0,
                'signals': []
            }
            
            signal_count = 0
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI Signals
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi < 30:
                    signals['signals'].append({'indicator': 'RSI', 'signal': 'BUY', 'reason': f'Oversold at {rsi:.1f}'})
                    bullish_signals += 1
                elif rsi > 70:
                    signals['signals'].append({'indicator': 'RSI', 'signal': 'SELL', 'reason': f'Overbought at {rsi:.1f}'})
                    bearish_signals += 1
                signal_count += 1
            
            # MACD Signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                if macd > macd_signal:
                    signals['signals'].append({'indicator': 'MACD', 'signal': 'BUY', 'reason': 'MACD above signal line'})
                    bullish_signals += 1
                else:
                    signals['signals'].append({'indicator': 'MACD', 'signal': 'SELL', 'reason': 'MACD below signal line'})
                    bearish_signals += 1
                signal_count += 1
            
            # Moving Average Signals
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20 = indicators['sma_20']
                sma_50 = indicators['sma_50']
                if sma_20 > sma_50:
                    signals['signals'].append({'indicator': 'MA', 'signal': 'BUY', 'reason': 'SMA20 above SMA50'})
                    bullish_signals += 1
                else:
                    signals['signals'].append({'indicator': 'MA', 'signal': 'SELL', 'reason': 'SMA20 below SMA50'})
                    bearish_signals += 1
                signal_count += 1
            
            # Bollinger Bands Signals
            if all(key in indicators for key in ['bb_upper', 'bb_lower']) and 'price_change' in indicators:
                current_price = indicators.get('bb_middle', 0)  # Use middle as proxy for current price
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                
                if current_price <= bb_lower:
                    signals['signals'].append({'indicator': 'BB', 'signal': 'BUY', 'reason': 'Price at lower Bollinger Band'})
                    bullish_signals += 1
                elif current_price >= bb_upper:
                    signals['signals'].append({'indicator': 'BB', 'signal': 'SELL', 'reason': 'Price at upper Bollinger Band'})
                    bearish_signals += 1
                signal_count += 1
            
            # Stochastic Signals
            if 'stoch_k' in indicators and 'stoch_d' in indicators:
                stoch_k = indicators['stoch_k']
                stoch_d = indicators['stoch_d']
                if stoch_k < 20 and stoch_k > stoch_d:
                    signals['signals'].append({'indicator': 'Stochastic', 'signal': 'BUY', 'reason': 'Oversold with bullish crossover'})
                    bullish_signals += 1
                elif stoch_k > 80 and stoch_k < stoch_d:
                    signals['signals'].append({'indicator': 'Stochastic', 'signal': 'SELL', 'reason': 'Overbought with bearish crossover'})
                    bearish_signals += 1
                signal_count += 1
            
            # Determine overall signal
            if signal_count > 0:
                bullish_ratio = bullish_signals / signal_count
                bearish_ratio = bearish_signals / signal_count
                
                if bullish_ratio >= 0.6:
                    signals['overall_signal'] = 'BUY'
                    signals['signal_strength'] = bullish_ratio
                elif bearish_ratio >= 0.6:
                    signals['overall_signal'] = 'SELL'
                    signals['signal_strength'] = bearish_ratio
                else:
                    signals['overall_signal'] = 'HOLD'
                    signals['signal_strength'] = 0.5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {'overall_signal': 'HOLD', 'signal_strength': 0, 'signals': []}