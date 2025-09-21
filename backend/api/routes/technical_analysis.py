from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/technical-analysis", tags=["Technical Analysis"])

@router.get("/{symbol}")
async def get_technical_analysis(
    symbol: str,
    period: str = Query("3mo", description="Data period for analysis")
):
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        
        # Get historical data for technical analysis
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Calculate technical indicators using real data
        close_prices = hist['Close']
        high_prices = hist['High']
        low_prices = hist['Low']
        volume = hist['Volume']
        
        # RSI (Relative Strength Index)
        rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi()
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0
        
        # MACD
        macd_indicator = ta.trend.MACD(close_prices)
        macd_line = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_histogram = macd_indicator.macd_diff()
        
        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0.0
        current_signal = macd_signal.iloc[-1] if not macd_signal.empty else 0.0
        current_histogram = macd_histogram.iloc[-1] if not macd_histogram.empty else 0.0
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close_prices, window=20, window_dev=2)
        bb_upper = bb_indicator.bollinger_hband()
        bb_lower = bb_indicator.bollinger_lband()
        bb_middle = bb_indicator.bollinger_mavg()
        
        current_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else close_prices.iloc[-1] * 1.02
        current_bb_lower = bb_lower.iloc[-1] if not bb_lower.empty else close_prices.iloc[-1] * 0.98
        current_bb_middle = bb_middle.iloc[-1] if not bb_middle.empty else close_prices.iloc[-1]
        
        # Moving Averages
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        
        current_sma_20 = sma_20.iloc[-1] if not sma_20.empty else close_prices.iloc[-1]
        current_sma_50 = sma_50.iloc[-1] if not sma_50.empty else close_prices.iloc[-1]
        current_ema_12 = ema_12.iloc[-1] if not ema_12.empty else close_prices.iloc[-1]
        current_ema_26 = ema_26.iloc[-1] if not ema_26.empty else close_prices.iloc[-1]
        
        # Volume analysis
        volume_sma = volume.rolling(window=20).mean()
        current_volume_ratio = (volume.iloc[-1] / volume_sma.iloc[-1]) if not volume_sma.empty else 1.0
        
        if current_volume_ratio > 1.5:
            volume_profile = "High"
        elif current_volume_ratio > 0.8:
            volume_profile = "Medium"
        else:
            volume_profile = "Low"
        
        # Support and Resistance levels (using pivot points and recent highs/lows)
        recent_data = hist.tail(20)
        support_levels = []
        resistance_levels = []
        
        # Calculate pivot points
        if len(recent_data) >= 3:
            pivot = (recent_data['High'].max() + recent_data['Low'].min() + close_prices.iloc[-1]) / 3
            r1 = 2 * pivot - recent_data['Low'].min()
            r2 = pivot + (recent_data['High'].max() - recent_data['Low'].min())
            s1 = 2 * pivot - recent_data['High'].max()
            s2 = pivot - (recent_data['High'].max() - recent_data['Low'].min())
            
            support_levels = [float(s2), float(s1), float(pivot)]
            resistance_levels = [float(pivot), float(r1), float(r2)]
        else:
            # Fallback to simple support/resistance
            current_price = close_prices.iloc[-1]
            support_levels = [float(current_price * 0.95), float(current_price * 0.97), float(current_price * 0.99)]
            resistance_levels = [float(current_price * 1.01), float(current_price * 1.03), float(current_price * 1.05)]
        
        # Generate signals based on technical indicators
        signals = []
        
        # RSI signals
        if current_rsi < 30:
            signals.append("Oversold - Potential Buy")
        elif current_rsi > 70:
            signals.append("Overbought - Potential Sell")
        else:
            signals.append("RSI Neutral")
        
        # MACD signals
        if current_macd > current_signal:
            signals.append("MACD Bullish")
        else:
            signals.append("MACD Bearish")
        
        # Moving Average signals
        current_price = close_prices.iloc[-1]
        if current_price > current_sma_20 > current_sma_50:
            signals.append("MA Golden Cross")
        elif current_price < current_sma_20 < current_sma_50:
            signals.append("MA Death Cross")
        else:
            signals.append("MA Mixed")
        
        # Bollinger Band signals
        if current_price > current_bb_upper:
            signals.append("BB Overbought")
        elif current_price < current_bb_lower:
            signals.append("BB Oversold")
        else:
            signals.append("BB Neutral")
        
        # Pattern detection (basic implementation)
        pattern_detected = None
        if len(hist) >= 10:
            recent_highs = high_prices.tail(10)
            recent_lows = low_prices.tail(10)
            
            # Simple pattern detection
            if recent_highs.iloc[-1] == recent_highs.max() and recent_highs.iloc[-2] < recent_highs.iloc[-1]:
                pattern_detected = "Breakout"
            elif recent_lows.iloc[-1] == recent_lows.min() and recent_lows.iloc[-2] > recent_lows.iloc[-1]:
                pattern_detected = "Breakdown"
            elif current_price > current_sma_20 and current_sma_20 > current_sma_50:
                pattern_detected = "Uptrend"
            elif current_price < current_sma_20 and current_sma_20 < current_sma_50:
                pattern_detected = "Downtrend"
            else:
                pattern_detected = "Sideways"
        
        # Compile the complete technical analysis
        technical_analysis = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "current_price": float(current_price),
            "indicators": {
                "rsi": float(current_rsi),
                "macd": float(current_macd),
                "macd_signal": float(current_signal),
                "macd_histogram": float(current_histogram),
                "bollinger_upper": float(current_bb_upper),
                "bollinger_lower": float(current_bb_lower),
                "bollinger_middle": float(current_bb_middle),
                "sma_20": float(current_sma_20),
                "sma_50": float(current_sma_50),
                "ema_12": float(current_ema_12),
                "ema_26": float(current_ema_26),
                "volume_profile": volume_profile,
                "volume_ratio": float(current_volume_ratio)
            },
            "signals": signals,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "pattern_detected": pattern_detected,
            "trend_analysis": {
                "short_term": "Bullish" if current_price > current_sma_20 else "Bearish",
                "medium_term": "Bullish" if current_sma_20 > current_sma_50 else "Bearish",
                "long_term": "Bullish" if current_price > current_sma_50 else "Bearish"
            },
            "volatility": float(close_prices.pct_change().std() * np.sqrt(252)) if len(close_prices) > 1 else 0.0
        }
        
        return {
            "success": True,
            "data": technical_analysis
        }
        
    except Exception as e:
        logger.error(f"❌ Technical analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze {symbol}: {str(e)}")

@router.get("/batch")
async def get_batch_technical_analysis(
    symbols: str = Query(..., description="Comma-separated symbols"),
    period: str = Query("3mo", description="Data period for analysis")
):
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        results = {}
        
        for symbol in symbol_list:
            try:
                # Use the single symbol endpoint
                analysis_response = await get_technical_analysis(symbol, period)
                if analysis_response.get("success"):
                    results[symbol] = analysis_response["data"]
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                # Don't add failed symbols to results
                continue
        
        return {
            "success": True,
            "data": results,
            "symbols_analyzed": list(results.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/screener")
async def technical_screener(
    criteria: str = Query("oversold", description="Screening criteria: oversold, overbought, breakout, breakdown"),
    limit: int = Query(20, description="Maximum results to return")
):
    try:
        # Popular symbols to screen
        screening_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "AMD", "INTC", "CRM", "ORCL", "UBER", "LYFT", "ZOOM", "ROKU",
            "SQ", "PYPL", "V", "MA", "JPM", "BAC", "WFC", "GS",
            "JNJ", "PFE", "MRNA", "ABT", "TMO", "UNH", "CVS", "CI"
        ]
        
        screened_results = []
        
        for symbol in screening_symbols[:limit * 2]:  # Screen more than needed
            try:
                analysis_response = await get_technical_analysis(symbol, "1mo")
                if not analysis_response.get("success"):
                    continue
                
                analysis = analysis_response["data"]
                indicators = analysis["indicators"]
                
                # Apply screening criteria
                meets_criteria = False
                
                if criteria.lower() == "oversold":
                    meets_criteria = (indicators["rsi"] < 30 or 
                    analysis["current_price"] < indicators["bollinger_lower"])
                
                elif criteria.lower() == "overbought":
                    meets_criteria = (indicators["rsi"] > 70 or 
                    analysis["current_price"] > indicators["bollinger_upper"])
                
                elif criteria.lower() == "breakout":
                    meets_criteria = (analysis["pattern_detected"] == "Breakout" or
                    (analysis["current_price"] > indicators["sma_20"] and
                    indicators["volume_ratio"] > 1.5))
                
                elif criteria.lower() == "breakdown":
                    meets_criteria = (analysis["pattern_detected"] == "Breakdown" or
                    (analysis["current_price"] < indicators["sma_20"] and
                    indicators["volume_ratio"] > 1.5))
                
                if meets_criteria:
                    screened_results.append({
                        "symbol": symbol,
                        "current_price": analysis["current_price"],
                        "rsi": indicators["rsi"],
                        "pattern": analysis["pattern_detected"],
                        "signals": analysis["signals"][:3],  # Top 3 signals
                        "volume_profile": indicators["volume_profile"]
                    })
                
                if len(screened_results) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Screening error for {symbol}: {e}")
                continue
        
        return {
            "success": True,
            "data": {
                "criteria": criteria,
                "results": screened_results,
                "count": len(screened_results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Technical screener error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
