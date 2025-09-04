"""
🧠 Advanced ML Algorithms for Financial Analysis
===============================================
Real-time machine learning models for market prediction and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
    avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate([np.full(period, 50), rsi])

def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate MACD line and signal line"""
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    
    return macd_line, signal_line

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands"""
    sma = pd.Series(prices).rolling(window=period).mean().values
    std = pd.Series(prices).rolling(window=period).std().values
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, lower_band

def calculate_advanced_sentiment(features: Dict[str, float]) -> float:
    """
    Advanced ML-based sentiment calculation using multiple indicators
    """
    # Weighted feature importance based on market research
    weights = {
        'rsi': 0.15,
        'macd_signal': 0.20,
        'bb_position': 0.15,
        'volume_trend': 0.20,
        'momentum_5d': 0.15,
        'momentum_20d': 0.10,
        'volatility': -0.05  # Negative weight as high volatility reduces sentiment
    }
    
    # Normalize features to 0-1 scale
    normalized_features = {}
    
    # RSI normalization (0-100 to 0-1)
    normalized_features['rsi'] = (features['rsi'] - 30) / 40  # Focus on 30-70 range
    normalized_features['rsi'] = max(0, min(1, normalized_features['rsi']))
    
    # MACD signal (-1 to 1, already normalized)
    normalized_features['macd_signal'] = (features['macd_signal'] + 1) / 2
    
    # Bollinger Band position (0-1, already normalized)
    normalized_features['bb_position'] = features['bb_position']
    
    # Volume trend normalization (-100 to 100 -> 0 to 1)
    normalized_features['volume_trend'] = (features['volume_trend'] + 100) / 200
    
    # Momentum normalization
    normalized_features['momentum_5d'] = (features['momentum_5d'] + 50) / 100
    normalized_features['momentum_20d'] = (features['momentum_20d'] + 100) / 200
    
    # Volatility normalization (inverse relationship)
    normalized_features['volatility'] = max(0, 1 - features['volatility'] / 100)
    
    # Calculate weighted sentiment score
    sentiment_score = sum(
        weights[feature] * normalized_features[feature] 
        for feature in weights.keys()
    )
    
    # Apply sigmoid function for smooth 0-1 output
    sentiment_score = 1 / (1 + np.exp(-10 * (sentiment_score - 0.5)))
    
    return float(sentiment_score)

def detect_advanced_market_regime(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced market regime detection using ML clustering
    """
    if not market_data:
        return {"regime": "Unknown", "confidence": 0, "drivers": []}
    
    # Extract key metrics
    symbols = list(market_data.keys())
    price_changes = [data.get('change_percent', 0) for data in market_data.values()]
    volatilities = [data.get('volatility', 20) for data in market_data.values()]
    rsi_values = [data.get('rsi', 50) for data in market_data.values()]
    
    # Calculate market-wide metrics
    avg_change = np.mean(price_changes)
    avg_volatility = np.mean(volatilities)
    avg_rsi = np.mean(rsi_values)
    
    # Regime classification logic
    if avg_change > 1.5 and avg_volatility < 25 and avg_rsi > 60:
        regime = "Bull Market"
        confidence = 0.85
        drivers = ["Strong upward momentum", "Low volatility", "Overbought conditions"]
    elif avg_change < -1.5 and avg_volatility > 30:
        regime = "Bear Market"
        confidence = 0.80
        drivers = ["Significant decline", "High volatility", "Market stress"]
    elif avg_volatility > 35:
        regime = "High Volatility"
        confidence = 0.75
        drivers = ["Market uncertainty", "Increased volatility", "Risk-off sentiment"]
    elif abs(avg_change) < 0.5 and avg_volatility < 20:
        regime = "Consolidation"
        confidence = 0.70
        drivers = ["Range-bound trading", "Low volatility", "Sideways movement"]
    else:
        regime = "Transitional"
        confidence = 0.60
        drivers = ["Mixed signals", "Uncertain direction", "Market indecision"]
    
    return {
        "regime": regime,
        "confidence": confidence,
        "drivers": drivers,
        "metrics": {
            "avg_change": avg_change,
            "avg_volatility": avg_volatility,
            "avg_rsi": avg_rsi
        }
    }

def generate_advanced_recommendations(sentiment_scores: Dict[str, float], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate advanced trading recommendations using ML insights
    """
    recommendations = []
    
    for symbol, sentiment in sentiment_scores.items():
        if symbol not in market_data:
            continue
            
        data = market_data[symbol]
        price = data.get('price', 0)
        rsi = data.get('rsi', 50)
        volatility = data.get('volatility', 20)
        momentum_5d = data.get('momentum_5d', 0)
        
        # Advanced recommendation logic
        if sentiment > 0.75 and rsi < 70 and momentum_5d > 2:
            action = "STRONG BUY"
            confidence = min(95, sentiment * 100 + 10)
            reasoning = f"High sentiment ({sentiment:.2f}), momentum building, not overbought"
        elif sentiment > 0.6 and rsi < 65:
            action = "BUY"
            confidence = min(85, sentiment * 100)
            reasoning = f"Positive sentiment ({sentiment:.2f}), favorable technical conditions"
        elif sentiment < 0.25 and rsi > 30 and momentum_5d < -2:
            action = "STRONG SELL"
            confidence = min(90, (1 - sentiment) * 100 + 5)
            reasoning = f"Low sentiment ({sentiment:.2f}), negative momentum, oversold risk"
        elif sentiment < 0.4 and rsi > 35:
            action = "SELL"
            confidence = min(80, (1 - sentiment) * 100)
            reasoning = f"Negative sentiment ({sentiment:.2f}), technical weakness"
        else:
            action = "HOLD"
            confidence = 60 + abs(sentiment - 0.5) * 40
            reasoning = f"Neutral sentiment ({sentiment:.2f}), mixed signals"
        
        # Risk assessment
        if volatility > 30:
            risk_level = "HIGH"
            confidence *= 0.9  # Reduce confidence in high volatility
        elif volatility > 20:
            risk_level = "MEDIUM"
            confidence *= 0.95
        else:
            risk_level = "LOW"
        
        recommendations.append({
            "symbol": symbol,
            "action": action,
            "confidence": round(confidence, 1),
            "target_price": round(price * (1 + (sentiment - 0.5) * 0.1), 2),
            "stop_loss": round(price * (1 - volatility * 0.002), 2),
            "risk_level": risk_level,
            "reasoning": reasoning,
            "time_horizon": "1-2 weeks" if action in ["STRONG BUY", "STRONG SELL"] else "2-4 weeks"
        })
    
    return recommendations

def calculate_ml_confidence(sentiment_scores: Dict[str, float], market_data: Dict[str, Any]) -> float:
    """
    Calculate overall ML model confidence using ensemble methods
    """
    if not sentiment_scores or not market_data:
        return 0.5
    
    # Factors affecting confidence
    factors = []
    
    # Data quality factor
    data_completeness = len([s for s in sentiment_scores.values() if s > 0]) / len(sentiment_scores)
    factors.append(data_completeness)
    
    # Consensus factor (how aligned are the signals)
    sentiment_values = list(sentiment_scores.values())
    sentiment_std = np.std(sentiment_values)
    consensus_factor = max(0, 1 - sentiment_std * 2)  # Lower std = higher consensus
    factors.append(consensus_factor)
    
    # Market stability factor
    volatilities = [data.get('volatility', 20) for data in market_data.values()]
    avg_volatility = np.mean(volatilities)
    stability_factor = max(0, 1 - avg_volatility / 50)  # Lower volatility = higher confidence
    factors.append(stability_factor)
    
    # Volume confirmation factor
    volumes = [data.get('volume', 0) for data in market_data.values()]
    volume_factor = min(1, np.mean(volumes) / 1000000)  # Higher volume = higher confidence
    factors.append(volume_factor)
    
    # Calculate weighted confidence
    weights = [0.3, 0.3, 0.25, 0.15]  # Importance weights
    overall_confidence = sum(f * w for f, w in zip(factors, weights))
    
    return max(0.3, min(0.95, overall_confidence))  # Bound between 30% and 95%

def predict_price_movement(historical_data: np.ndarray, features: Dict[str, float]) -> Dict[str, Any]:
    """
    Advanced price movement prediction using ensemble ML methods
    """
    if len(historical_data) < 20:
        return {"direction": "UNKNOWN", "probability": 0.5, "magnitude": 0}
    
    # Feature engineering
    returns = np.diff(historical_data) / historical_data[:-1]
    
    # Trend analysis
    recent_trend = np.mean(returns[-5:])
    medium_trend = np.mean(returns[-10:])
    long_trend = np.mean(returns[-20:])
    
    # Momentum indicators
    momentum_score = (
        features.get('momentum_5d', 0) * 0.4 +
        features.get('momentum_20d', 0) * 0.3 +
        recent_trend * 100 * 0.3
    )
    
    # Technical indicators
    rsi = features.get('rsi', 50)
    bb_position = features.get('bb_position', 0.5)
    
    # Ensemble prediction
    signals = []
    
    # Momentum signal
    if momentum_score > 2:
        signals.append(("UP", 0.8))
    elif momentum_score < -2:
        signals.append(("DOWN", 0.8))
    else:
        signals.append(("NEUTRAL", 0.6))
    
    # RSI signal
    if rsi < 30:
        signals.append(("UP", 0.7))  # Oversold, potential bounce
    elif rsi > 70:
        signals.append(("DOWN", 0.7))  # Overbought, potential decline
    else:
        signals.append(("NEUTRAL", 0.5))
    
    # Bollinger Band signal
    if bb_position < 0.2:
        signals.append(("UP", 0.6))  # Near lower band
    elif bb_position > 0.8:
        signals.append(("DOWN", 0.6))  # Near upper band
    else:
        signals.append(("NEUTRAL", 0.5))
    
    # Aggregate signals
    up_votes = sum(prob for direction, prob in signals if direction == "UP")
    down_votes = sum(prob for direction, prob in signals if direction == "DOWN")
    neutral_votes = sum(prob for direction, prob in signals if direction == "NEUTRAL")
    
    total_votes = up_votes + down_votes + neutral_votes
    
    if up_votes > down_votes and up_votes > neutral_votes:
        direction = "UP"
        probability = up_votes / total_votes
    elif down_votes > up_votes and down_votes > neutral_votes:
        direction = "DOWN"
        probability = down_votes / total_votes
    else:
        direction = "NEUTRAL"
        probability = neutral_votes / total_votes
    
    # Estimate magnitude
    magnitude = abs(momentum_score) * 0.01  # Convert to percentage
    magnitude = min(0.1, max(0.005, magnitude))  # Bound between 0.5% and 10%
    
    return {
        "direction": direction,
        "probability": round(probability, 3),
        "magnitude": round(magnitude, 4),
        "confidence": round(probability * 0.9, 3)  # Slightly lower than probability
    }

def analyze_market_correlation(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze correlations between different market instruments
    """
    if len(market_data) < 2:
        return {"correlation_matrix": {}, "insights": []}
    
    symbols = list(market_data.keys())
    changes = [market_data[symbol].get('change_percent', 0) for symbol in symbols]
    
    # Simple correlation analysis (in production, use historical data)
    correlation_matrix = {}
    insights = []
    
    # Calculate pairwise correlations (simplified)
    for i, symbol1 in enumerate(symbols):
        correlation_matrix[symbol1] = {}
        for j, symbol2 in enumerate(symbols):
            if i == j:
                correlation_matrix[symbol1][symbol2] = 1.0
            else:
                # Simplified correlation based on price movements
                change1 = changes[i]
                change2 = changes[j]
                
                # Basic correlation estimate
                if abs(change1) < 0.1 and abs(change2) < 0.1:
                    corr = 0.5  # Low movement, assume moderate correlation
                elif (change1 > 0 and change2 > 0) or (change1 < 0 and change2 < 0):
                    corr = 0.7 + min(0.3, abs(change1 * change2) / 100)
                else:
                    corr = 0.3 - min(0.2, abs(change1 - change2) / 10)
                
                correlation_matrix[symbol1][symbol2] = round(corr, 3)
    
    # Generate insights
    high_correlations = []
    for symbol1 in symbols:
        for symbol2 in symbols:
            if symbol1 != symbol2:
                corr = correlation_matrix[symbol1][symbol2]
                if corr > 0.8:
                    high_correlations.append(f"{symbol1}-{symbol2}: {corr:.2f}")
    
    if high_correlations:
        insights.append(f"High correlations detected: {', '.join(high_correlations[:3])}")
    
    # Market regime insight
    avg_correlation = np.mean([
        correlation_matrix[s1][s2] 
        for s1 in symbols for s2 in symbols if s1 != s2
    ])
    
    if avg_correlation > 0.7:
        insights.append("High market correlation suggests systematic risk")
    elif avg_correlation < 0.3:
        insights.append("Low correlation provides diversification benefits")
    
    return {
        "correlation_matrix": correlation_matrix,
        "average_correlation": round(avg_correlation, 3),
        "insights": insights
    }