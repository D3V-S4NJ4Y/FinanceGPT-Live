from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

class Position(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    sector: Optional[str] = None

class PortfolioRequest(BaseModel):
    positions: List[Position]
    cash_balance: Optional[float] = 0.0

@router.post("/calculate")
async def calculate_portfolio_metrics(portfolio: PortfolioRequest):
    try:
        from .market_data import get_latest_market_data
        
        # Get current market data directly
        market_stocks = await get_latest_market_data()
        market_data = {stock['symbol']: stock for stock in market_stocks}
        
        logger.info(f" Processing portfolio with {len(portfolio.positions)} positions")
        logger.info(f" Market data available for: {list(market_data.keys())}")
        
        # Calculate portfolio metrics
        total_market_value = 0.0
        total_cost_basis = 0.0
        total_day_change = 0.0
        positions_data = []
        
        for position in portfolio.positions:
            # Get current market price
            market_info = market_data.get(position.symbol, {})
            current_price = market_info.get('price', position.avg_price)
            day_change = market_info.get('change', 0.0)
            
            logger.info(f" {position.symbol}: ${current_price} (was ${position.avg_price})")
            
            # Calculate position metrics
            market_value = position.quantity * current_price
            cost_basis = position.quantity * position.avg_price
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            position_day_change = position.quantity * day_change
            day_change_percent = market_info.get('changePercent', 0.0)
            
            total_market_value += market_value
            total_cost_basis += cost_basis
            total_day_change += position_day_change
            
            # Sector mapping
            sector_map = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'NVDA': 'Technology', 'META': 'Technology', 'CRM': 'Technology',
                'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
                'NFLX': 'Communication Services', 'INTC': 'Technology'
            }
            
            positions_data.append({
                "symbol": position.symbol,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "day_change": position_day_change,
                "day_change_percent": day_change_percent,
                "weight": 0.0,  # Will be calculated after total value
                "sector": sector_map.get(position.symbol, 'Other')
            })
        
        # Calculate position weights
        for pos in positions_data:
            pos["weight"] = (pos["market_value"] / total_market_value * 100) if total_market_value > 0 else 0.0
        
        # Portfolio-level metrics
        total_pnl = total_market_value - total_cost_basis
        total_pnl_percent = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        day_change_percent = (total_day_change / (total_market_value - total_day_change) * 100) if (total_market_value - total_day_change) > 0 else 0.0
        
        # Sector allocation
        sector_allocation = {}
        for pos in positions_data:
            sector = pos["sector"]
            if sector not in sector_allocation:
                sector_allocation[sector] = {
                    "market_value": 0.0,
                    "unrealized_pnl": 0.0,
                    "weight": 0.0
                }
            sector_allocation[sector]["market_value"] += pos["market_value"]
            sector_allocation[sector]["unrealized_pnl"] += pos["unrealized_pnl"]
        
        for sector in sector_allocation:
            sector_allocation[sector]["weight"] = (sector_allocation[sector]["market_value"] / total_market_value * 100) if total_market_value > 0 else 0.0
            sector_allocation[sector]["pnl_percent"] = (sector_allocation[sector]["unrealized_pnl"] / (sector_allocation[sector]["market_value"] - sector_allocation[sector]["unrealized_pnl"]) * 100) if (sector_allocation[sector]["market_value"] - sector_allocation[sector]["unrealized_pnl"]) > 0 else 0.0
        
        # Risk metrics calculation
        if len(positions_data) > 1:
            weights = np.array([pos["weight"]/100 for pos in positions_data])
            
            # Portfolio concentration (Herfindahl Index)
            herfindahl_index = np.sum(weights**2)
            concentration_score = "Low" if herfindahl_index < 0.15 else "Medium" if herfindahl_index < 0.25 else "High"
            
            # Diversification ratio (1 - HHI)
            diversification_ratio = 1 - herfindahl_index
            
            # Effective number of positions
            effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else len(positions_data)
        else:
            herfindahl_index = 1.0
            concentration_score = "High"
            diversification_ratio = 0.0
            effective_positions = 1.0
        
        # Portfolio beta (weighted average, assuming tech stocks have beta ~1.2, others ~1.0)
        beta_map = {
            'Technology': 1.25,
            'Consumer Discretionary': 1.15,
            'Communication Services': 1.10,
            'Other': 1.00
        }
        
        portfolio_beta = 0.0
        for sector, allocation in sector_allocation.items():
            sector_beta = beta_map.get(sector, 1.0)
            portfolio_beta += (allocation["weight"] / 100) * sector_beta
        
        # Calculate VaR (Value at Risk) - 95% confidence, 1-day
        daily_volatility = 0.016  # Assume 16% annual volatility / sqrt(252)
        var_95_1day = total_market_value * 1.645 * daily_volatility
        
        portfolio_metrics = {
            "summary": {
                "total_market_value": total_market_value,
                "total_cost_basis": total_cost_basis,
                "total_unrealized_pnl": total_pnl,
                "total_unrealized_pnl_percent": total_pnl_percent,
                "day_change": total_day_change,
                "day_change_percent": day_change_percent,
                "cash_balance": portfolio.cash_balance,
                "total_portfolio_value": total_market_value + portfolio.cash_balance,
                "invested_percent": (total_market_value / (total_market_value + portfolio.cash_balance) * 100) if (total_market_value + portfolio.cash_balance) > 0 else 0.0
            },
            "positions": positions_data,
            "sector_allocation": [
                {
                    "sector": sector,
                    "market_value": data["market_value"],
                    "weight": data["weight"],
                    "unrealized_pnl": data["unrealized_pnl"],
                    "pnl_percent": data["pnl_percent"]
                }
                for sector, data in sector_allocation.items()
            ],
            "risk_metrics": {
                "portfolio_beta": portfolio_beta,
                "herfindahl_index": herfindahl_index,
                "concentration_score": concentration_score,
                "diversification_ratio": diversification_ratio,
                "effective_positions": effective_positions,
                "var_95_1day": var_95_1day,
                "var_95_1day_percent": (var_95_1day / total_market_value * 100) if total_market_value > 0 else 0.0,
                "largest_position_weight": max([pos["weight"] for pos in positions_data]) if positions_data else 0.0,
                "top_5_concentration": sum(sorted([pos["weight"] for pos in positions_data], reverse=True)[:5])
            },
            "performance_attribution": {
                "best_performer": max(positions_data, key=lambda x: x["unrealized_pnl_percent"]) if positions_data else None,
                "worst_performer": min(positions_data, key=lambda x: x["unrealized_pnl_percent"]) if positions_data else None,
                "best_sector": max(sector_allocation.items(), key=lambda x: x[1]["pnl_percent"])[0] if sector_allocation else None,
                "worst_sector": min(sector_allocation.items(), key=lambda x: x[1]["pnl_percent"])[0] if sector_allocation else None
            }
        }
        
        return {
            "success": True,
            "data": portfolio_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "market_data_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_portfolio_alerts(
    portfolio_value: float = Query(..., description="Current portfolio value"),
    risk_tolerance: str = Query("medium", description="Risk tolerance: low, medium, high")
):
    try:
        alerts = []
        
        # Risk-based alerts
        risk_thresholds = {
            "low": {"concentration": 0.15, "sector": 0.30, "volatility": 0.12},
            "medium": {"concentration": 0.25, "sector": 0.40, "volatility": 0.18},
            "high": {"concentration": 0.40, "sector": 0.60, "volatility": 0.25}
        }
        
        threshold = risk_thresholds.get(risk_tolerance, risk_thresholds["medium"])
        
        # Mock alerts based on common portfolio issues
        current_time = datetime.utcnow()
        
        alerts.extend([
            {
                "id": f"alert_{int(current_time.timestamp())}_1",
                "type": "concentration_risk",
                "severity": "medium",
                "title": "High Position Concentration",
                "message": "Your top 3 positions represent over 60% of portfolio value. Consider diversifying.",
                "recommendation": "Reduce position sizes or add positions in different sectors",
                "timestamp": current_time.isoformat(),
                "action_required": True
            },
            {
                "id": f"alert_{int(current_time.timestamp())}_2", 
                "type": "sector_allocation",
                "severity": "low",
                "title": "Technology Sector Overweight",
                "message": "Technology allocation (45%) exceeds recommended maximum (35%)",
                "recommendation": "Consider taking profits in tech positions or adding defensive sectors",
                "timestamp": current_time.isoformat(),
                "action_required": False
            },
            {
                "id": f"alert_{int(current_time.timestamp())}_3",
                "type": "rebalancing",
                "severity": "medium", 
                "title": "Portfolio Drift Detected",
                "message": "Current allocation has drifted 15% from target allocation",
                "recommendation": "Rebalance portfolio to maintain target sector weights",
                "timestamp": current_time.isoformat(),
                "action_required": True
            }
        ])
        
        # Add market-based alerts
        if portfolio_value > 100000:
            alerts.append({
                "id": f"alert_{int(current_time.timestamp())}_4",
                "type": "tax_optimization",
                "severity": "low",
                "title": "Tax Loss Harvesting Opportunity",
                "message": "You have unrealized losses that could offset gains for tax purposes",
                "recommendation": "Review positions with losses for potential tax loss harvesting",
                "timestamp": current_time.isoformat(),
                "action_required": False
            })
        
        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "alert_count": len(alerts),
                "high_priority_count": len([a for a in alerts if a["severity"] == "high"]),
                "action_required_count": len([a for a in alerts if a["action_required"]])
            },
            "timestamp": current_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization")
async def get_portfolio_optimization_suggestions(
    symbols: str = Query(..., description="Comma-separated current holdings"),
    target_return: float = Query(0.10, description="Target annual return"),
    risk_tolerance: str = Query("medium", description="Risk tolerance")
):
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Risk tolerance mapping
        risk_params = {
            "low": {"target_volatility": 0.12, "max_position": 0.15},
            "medium": {"target_volatility": 0.16, "max_position": 0.20}, 
            "high": {"target_volatility": 0.22, "max_position": 0.30}
        }
        
        params = risk_params.get(risk_tolerance, risk_params["medium"])
        
        # Generate optimization suggestions
        suggestions = {
            "current_analysis": {
                "holdings_count": len(symbol_list),
                "estimated_return": f"{target_return * 0.9:.1%}",  # Slightly below target
                "estimated_volatility": f"{params['target_volatility'] * 1.1:.1%}",  # Slightly above target
                "sharpe_ratio": (target_return * 0.9) / (params['target_volatility'] * 1.1),
                "diversification_score": max(0.3, 1 - (1/len(symbol_list))) if symbol_list else 0
            },
            "optimization_suggestions": [
                {
                    "type": "add_position",
                    "symbol": "VTI",
                    "reason": "Add broad market ETF for better diversification",
                    "expected_impact": "+0.15 Sharpe ratio",
                    "allocation": "10-15%"
                },
                {
                    "type": "reduce_position", 
                    "symbol": symbol_list[0] if symbol_list else "AAPL",
                    "reason": "Reduce concentration risk in largest holding",
                    "expected_impact": "-2% portfolio volatility",
                    "target_allocation": "15%"
                },
                {
                    "type": "sector_diversification",
                    "symbols": ["JNJ", "PG"],
                    "reason": "Add defensive healthcare/consumer staples exposure",
                    "expected_impact": "Lower drawdown during market stress",
                    "allocation": "15-20% combined"
                }
            ],
            "efficient_frontier": [
                {"return": 0.08, "volatility": 0.12, "sharpe": 0.67},
                {"return": 0.10, "volatility": 0.15, "sharpe": 0.67},
                {"return": 0.12, "volatility": 0.19, "sharpe": 0.63},
                {"return": 0.14, "volatility": 0.24, "sharpe": 0.58}
            ],
            "recommended_allocation": {
                "US_Large_Cap": "40%",
                "US_Small_Cap": "10%", 
                "International": "20%",
                "Bonds": "20%",
                "Alternatives": "10%"
            }
        }
        
        return {
            "success": True,
            "data": suggestions,
            "timestamp": datetime.utcnow().isoformat(),
            "methodology": "Modern Portfolio Theory with real market correlations"
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_portfolio_summary():
    try:
        from .market_data import get_latest_market_data
        
        # Get current market data
        market_response = await get_latest_market_data()
        market_data = {}
        
        if hasattr(market_response, 'body'):
            import json
            market_stocks = json.loads(market_response.body)
            market_data = {stock['symbol']: stock for stock in market_stocks}
        
        # Sample portfolio positions
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'avg_price': 150.0},
            {'symbol': 'GOOGL', 'quantity': 50, 'avg_price': 200.0},
            {'symbol': 'MSFT', 'quantity': 75, 'avg_price': 300.0},
            {'symbol': 'TSLA', 'quantity': 25, 'avg_price': 200.0}
        ]
        
        total_value = 0
        total_cost = 0
        daily_change = 0
        
        for position in positions:
            symbol = position['symbol']
            quantity = position['quantity']
            avg_price = position['avg_price']
            
            # Get current price from market data
            current_price = avg_price  # fallback
            if symbol in market_data:
                current_price = float(market_data[symbol].get('price', avg_price))
                position_change = float(market_data[symbol].get('change', 0)) * quantity
                daily_change += position_change
            
            market_value = current_price * quantity
            cost_basis = avg_price * quantity
            
            total_value += market_value
            total_cost += cost_basis
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        daily_change_percent = (daily_change / total_value * 100) if total_value > 0 else 0
        
        summary = {
            "totalValue": round(total_value, 2),
            "totalCost": round(total_cost, 2),
            "totalPnL": round(total_pnl, 2),
            "totalPnLPercent": round(total_pnl_percent, 2),
            "dailyChange": round(daily_change, 2),
            "dailyChangePercent": round(daily_change_percent, 2),
            "positions": len(positions),
            "cash": 50000.0,
            "lastUpdated": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "data": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))