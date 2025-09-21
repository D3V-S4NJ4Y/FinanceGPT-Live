from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from database.cache_manager import cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio Analytics"])

class PortfolioHolding(BaseModel):
    symbol: str
    shares: float
    avg_cost: float

class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]
    timeframe: Optional[str] = '1D'

@router.post("/analytics")
async def get_portfolio_analytics(request: PortfolioRequest):
    try:
        if not request.holdings:
            raise HTTPException(status_code=400, detail="No holdings provided")
        
        # Get real-time market data from our corrected endpoint
        from .market_data import get_latest_market_data
        
        logger.info(f" Analyzing portfolio with {len(request.holdings)} holdings")
        
        # Get current market data for all symbols
        market_stocks = await get_latest_market_data()
        market_data = {stock['symbol']: stock for stock in market_stocks}
        
        logger.info(f" Retrieved market data for: {list(market_data.keys())}")
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'total_value': 0.0,
            'total_cost': 0.0,
            'total_return': 0.0,
            'total_return_percent': 0.0,
            'day_change': 0.0,
            'day_change_percent': 0.0,
            'sharpe_ratio': 0.0,
            'beta': 1.0,
            'volatility': 15.0,
            'max_drawdown': 8.5
        }
        
        holdings_data = []
        
        for holding in request.holdings:
            symbol = holding.symbol
            shares = holding.shares
            avg_cost = holding.avg_cost
            
            # Get current market data
            stock_data = market_data.get(symbol, {})
            current_price = stock_data.get('price', avg_cost)  # fallback to avg cost
            day_change = stock_data.get('change', 0.0)
            day_change_percent = stock_data.get('changePercent', 0.0)
            
            # Calculate position metrics
            market_value = shares * current_price
            cost_basis = shares * avg_cost
            total_return = market_value - cost_basis
            total_return_percent = (total_return / cost_basis * 100) if cost_basis > 0 else 0.0
            position_day_change = shares * day_change
            
            logger.info(f" {symbol}: {shares} shares @ ${current_price} = ${market_value:.2f}")
            
            # Add to portfolio totals
            portfolio_metrics['total_value'] += market_value
            portfolio_metrics['total_cost'] += cost_basis
            portfolio_metrics['day_change'] += position_day_change
            
            holdings_data.append({
                'symbol': symbol,
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'market_value': market_value,
                'total_return': total_return,
                'total_return_percent': total_return_percent,
                'day_change': position_day_change,
                'day_change_percent': day_change_percent,
                'weight': 0.0  # Will calculate after totals
            })
        
        # Calculate portfolio-level metrics
        portfolio_metrics['total_return'] = portfolio_metrics['total_value'] - portfolio_metrics['total_cost']
        if portfolio_metrics['total_cost'] > 0:
            portfolio_metrics['total_return_percent'] = (portfolio_metrics['total_return'] / portfolio_metrics['total_cost']) * 100
            
        if portfolio_metrics['total_value'] > 0:
            portfolio_metrics['day_change_percent'] = (portfolio_metrics['day_change'] / portfolio_metrics['total_value']) * 100
            
            # Calculate position weights
            for holding in holdings_data:
                holding['weight'] = (holding['market_value'] / portfolio_metrics['total_value']) * 100
        # Calculate risk metrics
        weights_array = np.array([h['weight']/100 for h in holdings_data])
        returns_array = np.array([h['total_return_percent'] for h in holdings_data])
        
        # Portfolio-level risk calculations
        if len(weights_array) > 1:
            # Concentration risk (Herfindahl index)
            concentration_risk = np.sum(weights_array**2) * 100
            diversification_score = max(0, 100 - concentration_risk)
            
            # Correlation risk (simplified)
            correlation_risk = concentration_risk * 0.8  # Approximation
        else:
            concentration_risk = 100.0
            diversification_score = 0.0
            correlation_risk = 100.0
            
        # Sector exposure (simplified mapping)
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Consumer Discretionary',
            'AMZN': 'Consumer Discretionary', 'NFLX': 'Communication Services'
        }
        
        sector_exposure = {}
        for holding in holdings_data:
            sector = sector_map.get(holding['symbol'], 'Other')
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += holding['weight']
            
        risk_metrics = {
            'portfolio_risk': min(100, concentration_risk + correlation_risk * 0.3),
            'diversification_score': diversification_score,
            'concentration_risk': concentration_risk,
            'sector_exposure': sector_exposure,
            'correlation_risk': correlation_risk
        }
        
        logger.info(f"✅ Portfolio analysis complete: ${portfolio_metrics['total_value']:.2f} total value")
        
        return {
            "success": True,
            "data": {
                "portfolio_metrics": portfolio_metrics,
                "holdings": holdings_data,
                "risk_metrics": risk_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")


def calculate_risk_metrics(holdings: List[Dict], total_value: float) -> Dict[str, Any]:
    """Calculate portfolio risk metrics"""
    if not holdings or total_value <= 0:
        return {
            "portfolio_risk": 0,
            "diversification_score": 0,
            "concentration_risk": 0,
            "sector_exposure": {},
            "correlation_risk": 0
        }
    
    # Concentration risk (Herfindahl Index)
    weights = [h["weight"] / 100 for h in holdings]
    concentration_risk = sum(w ** 2 for w in weights) * 100
    
    # Diversification score
    diversification_score = max(0, 100 - concentration_risk * 2)
    
    # Sector exposure
    sector_exposure = {}
    for holding in holdings:
        sector = holding.get("sector", "Other")
        if sector not in sector_exposure:
            sector_exposure[sector] = 0
        sector_exposure[sector] += holding["weight"]
    
    # Portfolio risk (weighted volatility estimate)
    portfolio_risk = sum(abs(h["total_return_percent"]) * (h["weight"] / 100) for h in holdings)
    
    return {
        "portfolio_risk": round(portfolio_risk, 2),
        "diversification_score": round(diversification_score, 2),
        "concentration_risk": round(concentration_risk, 2),
        "sector_exposure": sector_exposure,
        "correlation_risk": round(max(0, 50 - diversification_score), 2)
    }


@router.get("/performance/{symbol}")
async def get_symbol_performance(
    symbol: str,
    period: str = Query("1y", description="Period: 1d,5d,1mo,3mo,6mo,1y,2y,5y")
):
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate performance metrics
        returns = hist['Close'].pct_change().dropna()
        
        # Basic metrics
        total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - (0.02 / 252)  # Daily risk-free rate
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        return {
            "success": True,
            "data": {
                "symbol": symbol.upper(),
                "period": period,
                "performance_metrics": {
                    "total_return": total_return,
                    "annualized_volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "downside_volatility": downside_volatility,
                    "var_95": var_95,
                    "current_price": float(hist['Close'].iloc[-1]),
                    "period_high": float(hist['High'].max()),
                    "period_low": float(hist['Low'].min())
                },
                "price_data": {
                    "dates": [date.isoformat() for date in hist.index],
                    "prices": hist['Close'].tolist(),
                    "volumes": hist['Volume'].tolist()
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Symbol performance error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark-comparison")
async def get_benchmark_comparison(
    symbols: str = Query(..., description="Comma-separated symbols"),
    benchmark: str = Query("SPY", description="Benchmark symbol")
):
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        all_symbols = symbol_list + [benchmark.upper()]
        
        # Fetch data for all symbols
        data = {}
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                    volatility = returns.std() * np.sqrt(252) * 100
                    
                    data[symbol] = {
                        "total_return": total_return,
                        "volatility": volatility,
                        "current_price": float(hist['Close'].iloc[-1])
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch benchmark data for {symbol}: {e}")
        
        # Calculate comparisons
        benchmark_data = data.get(benchmark.upper(), {})
        benchmark_return = benchmark_data.get("total_return", 0)
        
        comparisons = []
        for symbol in symbol_list:
            if symbol in data:
                symbol_data = data[symbol]
                alpha = symbol_data["total_return"] - benchmark_return
                
                comparisons.append({
                    "symbol": symbol,
                    "total_return": symbol_data["total_return"],
                    "benchmark_return": benchmark_return,
                    "alpha": alpha,
                    "volatility": symbol_data["volatility"],
                    "outperforming": alpha > 0
                })
        
        return {
            "success": True,
            "data": {
                "benchmark": benchmark.upper(),
                "benchmark_return": benchmark_return,
                "comparisons": comparisons,
                "summary": {
                    "outperforming_count": sum(1 for c in comparisons if c["outperforming"]),
                    "total_symbols": len(comparisons),
                    "avg_alpha": sum(c["alpha"] for c in comparisons) / len(comparisons) if comparisons else 0
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Benchmark comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))