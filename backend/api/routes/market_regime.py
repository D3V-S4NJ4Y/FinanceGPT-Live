from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# Models
class MarketRegimeResponse(BaseModel):
    regime: str
    confidence: float
    volatility_regime: str
    trend_strength: float
    market_stress: float
    characteristics: List[str]
    recommendations: List[str]
    risk_level: str
    next_update: str

router = APIRouter(tags=["Market Analysis"])

@router.get("/market-regime")
async def get_market_regime() -> MarketRegimeResponse:
    try:
        # Get real market data for key indices
        market_indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow, Nasdaq, Russell 2000
        
        # Fetch real-time data for analysis
        index_data = {}
        volatility_values = []
        trend_values = []
        
        for index in market_indices:
            try:
                # Get historical data to analyze trend and volatility
                ticker = yf.Ticker(index)
                hist = ticker.history(period="1mo")
                
                if not hist.empty:
                    # Calculate volatility (standard deviation of returns)
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    
                    # Calculate trend (ratio of price to moving average)
                    sma_50 = hist['Close'].rolling(window=20).mean()
                    current_price = hist['Close'].iloc[-1]
                    trend_ratio = current_price / sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else 1.0
                    
                    # Store the computed metrics
                    volatility_values.append(volatility)
                    trend_values.append(trend_ratio)
                    
                    # Store the data for this index
                    index_data[index] = {
                        "price": float(current_price),
                        "change": float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]),
                        "change_percent": float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100),
                        "volume": float(hist['Volume'].iloc[-1]),
                        "volatility": float(volatility),
                        "trend_ratio": float(trend_ratio)
                    }
            except Exception as e:
                logger.warning(f"Failed to get data for {index}: {e}")
                continue
        
        # Calculate aggregate market metrics
        avg_volatility = np.mean(volatility_values) if volatility_values else 0.2  # Default if data is missing
        avg_trend = np.mean(trend_values) if trend_values else 1.0  # Default if data is missing
        
        # Determine market regime based on real metrics
        # Bullish: trend > 1.02, Bearish: trend < 0.98, Neutral: in between
        # High volatility: vol > 0.25, Low volatility: vol < 0.15, Medium: in between
        
        # Determine market regime
        if avg_trend > 1.02:
            if avg_volatility < 0.15:
                regime = "Bullish Trend"
                confidence = 0.85
                characteristics = [
                    "Upward price momentum",
                    "Decreasing volatility",
                    "High trading volume",
                    "Positive sentiment"
                ]
                recommendations = [
                    "Focus on high-growth sectors",
                    "Maintain diversified exposure",
                    "Consider momentum strategies",
                    "Monitor for signs of regime transition"
                ]
                volatility_regime = "Low"
                risk_level = "Medium"
                market_stress = 0.35
                trend_strength = 0.82
            else:
                regime = "Volatile Bullish"
                confidence = 0.75
                characteristics = [
                    "Upward price momentum with pullbacks",
                    "Increasing volatility",
                    "Irregular trading volume",
                    "Mixed sentiment with bullish bias"
                ]
                recommendations = [
                    "Use stop losses for protection",
                    "Consider partial profit taking",
                    "Reduce position sizes",
                    "Look for high-quality names with stability"
                ]
                volatility_regime = "High"
                risk_level = "High"
                market_stress = 0.65
                trend_strength = 0.70
        elif avg_trend < 0.98:
            if avg_volatility > 0.25:
                regime = "Volatile Bearish"
                confidence = 0.80
                characteristics = [
                    "Strong downward momentum",
                    "High volatility",
                    "Panic selling",
                    "Extreme negative sentiment"
                ]
                recommendations = [
                    "Defensive positioning",
                    "Consider hedging strategies",
                    "Focus on cash preservation",
                    "Watch for capitulation signals"
                ]
                volatility_regime = "High"
                risk_level = "High"
                market_stress = 0.85
                trend_strength = 0.75
            else:
                regime = "Bearish Trend"
                confidence = 0.70
                characteristics = [
                    "Downward price momentum",
                    "Moderate volatility",
                    "Decreasing trading volume",
                    "Negative sentiment"
                ]
                recommendations = [
                    "Reduce equity exposure",
                    "Focus on quality and defensive sectors",
                    "Consider short-term tactical opportunities",
                    "Monitor for bottoming signals"
                ]
                volatility_regime = "Medium"
                risk_level = "High"
                market_stress = 0.70
                trend_strength = 0.65
        else:
            regime = "Neutral"
            confidence = 0.60
            characteristics = [
                "Sideways price action",
                "Moderate volatility",
                "Average trading volume",
                "Mixed sentiment indicators"
            ]
            recommendations = [
                "Balanced portfolio approach",
                "Stock selection over market timing",
                "Consider range-bound trading strategies",
                "Monitor for breakout signals"
            ]
            volatility_regime = "Medium"
            risk_level = "Medium"
            market_stress = 0.50
            trend_strength = 0.45
        
        # Create response
        return MarketRegimeResponse(
            regime=regime,
            confidence=confidence,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            market_stress=market_stress,
            characteristics=characteristics,
            recommendations=recommendations,
            risk_level=risk_level,
            next_update=(datetime.utcnow() + timedelta(hours=1)).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing market regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-regime/status")
async def get_market_regime_status():
    """
    Get a simplified market regime status update
    """
    try:
        # Get the full regime analysis
        regime_data = await get_market_regime()
        
        # Return a simplified version for status checks
        return {
            "regime": regime_data.regime,
            "confidence": regime_data.confidence,
            "volatility_regime": regime_data.volatility_regime,
            "trend_strength": regime_data.trend_strength,
            "market_stress": regime_data.market_stress,
            "risk_level": regime_data.risk_level
        }
    except Exception as e:
        logger.error(f"Error getting market regime status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
