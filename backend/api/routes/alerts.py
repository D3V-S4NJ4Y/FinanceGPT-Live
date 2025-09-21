from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

# Alert models
class Alert(BaseModel):
    type: str  # 'technical', 'market', 'risk', 'signal'
    severity: str  # 'low', 'medium', 'high'
    message: str
    symbol: Optional[str] = None
    source: str
    timestamp: Optional[str] = None

class AlertsResponse(BaseModel):
    alerts: List[Alert]
    total_count: int
    timestamp: str

# Import needed for signal-based alerts
def get_finance_system():
    """Get the global finance system instance"""
    import main
    return main.finance_system

@router.get("/latest")
async def get_latest_alerts():
    try:
        finance_system = get_finance_system()
        all_alerts = []
        
        # Get market alerts from market sentinel
        try:
            market_agent = finance_system.agents.get("market_sentinel")
            if market_agent and hasattr(market_agent, 'get_latest_alerts'):
                market_alerts = market_agent.get_latest_alerts()
                if market_alerts:
                    all_alerts.extend([
                        Alert(
                            type="market",
                            severity=alert.get("severity", "medium"),
                            message=alert.get("message", "Market condition alert"),
                            symbol=alert.get("symbol"),
                            source="Market Sentinel",
                            timestamp=alert.get("timestamp", datetime.now().isoformat())
                        ) for alert in market_alerts
                    ])
        except Exception as e:
            logger.warning(f"Failed to get market alerts: {e}")
        
        # Get risk alerts from risk assessor
        try:
            risk_agent = finance_system.agents.get("risk_assessor")
            if risk_agent and hasattr(risk_agent, 'get_latest_alerts'):
                risk_alerts = risk_agent.get_latest_alerts()
                if risk_alerts:
                    all_alerts.extend([
                        Alert(
                            type="risk",
                            severity=alert.get("severity", "medium"),
                            message=alert.get("message", "Risk threshold alert"),
                            symbol=alert.get("symbol"),
                            source="Risk Assessor",
                            timestamp=alert.get("timestamp", datetime.now().isoformat())
                        ) for alert in risk_alerts
                    ])
        except Exception as e:
            logger.warning(f"Failed to get risk alerts: {e}")
            
        # Get technical alerts from signal generator
        try:
            signal_agent = finance_system.agents.get("signal_generator")
            if signal_agent and hasattr(signal_agent, 'get_latest_alerts'):
                tech_alerts = signal_agent.get_latest_alerts()
                if tech_alerts:
                    all_alerts.extend([
                        Alert(
                            type="technical", 
                            severity=alert.get("severity", "medium"),
                            message=alert.get("message", "Technical analysis alert"),
                            symbol=alert.get("symbol"),
                            source="Signal Generator",
                            timestamp=alert.get("timestamp", datetime.now().isoformat())
                        ) for alert in tech_alerts
                    ])
        except Exception as e:
            logger.warning(f"Failed to get technical alerts: {e}")
            
        # Get AI signal alerts
        try:
            signal_agent = finance_system.agents.get("signal_generator")
            if signal_agent and hasattr(signal_agent, 'get_latest_signals'):
                signal_alerts = signal_agent.get_latest_signals()
                if signal_alerts:
                    all_alerts.extend([
                        Alert(
                            type="signal",
                            severity="medium" if signal.get("confidence", 0) > 0.7 else "low",
                            message=f"{signal.get('direction', 'NEUTRAL')} signal: {signal.get('description', 'No details')}",
                            symbol=signal.get("symbol"),
                            source="AI Signal Generator",
                            timestamp=signal.get("timestamp", datetime.now().isoformat())
                        ) for signal in signal_alerts
                    ])
        except Exception as e:
            logger.warning(f"Failed to get signal alerts: {e}")
        
        # If no alerts found, provide some fallback alerts based on market conditions
        if not all_alerts:
            # Current timestamp for consistency
            now = datetime.now().isoformat()
            
            # Check market data for potential alerts
            from data_sources.yahoo_finance import YahooFinanceConnector
            yahoo = YahooFinanceConnector()
            
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
            for symbol in symbols:
                try:
                    ticker_data = yahoo.get_ticker_data(symbol)
                    if ticker_data:
                        # Generate alert if significant price movement
                        change_pct = ticker_data.get("change_percent", 0)
                        if abs(change_pct) > 1.5:
                            direction = "up" if change_pct > 0 else "down"
                            severity = "medium" if abs(change_pct) > 2.5 else "low"
                            
                            all_alerts.append(Alert(
                                type="market",
                                severity=severity,
                                message=f"Price moved {direction} by {abs(change_pct):.2f}%",
                                symbol=symbol,
                                source="Market Monitor",
                                timestamp=now
                            ))
                except Exception as e:
                    logger.error(f"Error generating alert for {symbol}: {e}")
        
        # Sort alerts by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        all_alerts.sort(key=lambda x: severity_order.get(x.severity, 3))
        
        return AlertsResponse(
            alerts=all_alerts[:20],  # Limit to 20 most important alerts
            total_count=len(all_alerts),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in alerts API: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")

@router.get("/historical")
async def get_historical_alerts(days: int = 7):
    # Implementation similar to /latest but with historical data
    # This would typically pull from a database
    pass

@router.get("/recent")
async def get_recent_alerts():
    return await get_latest_alerts()
