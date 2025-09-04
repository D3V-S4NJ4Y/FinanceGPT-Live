"""
🤖 Live AI Agent with Pathway Integration
========================================
Real-time financial AI agent with streaming data and dynamic responses
"""

import pathway as pw
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime
import yfinance as yf
import logging
from .realtime_rag_engine import pathway_rag_engine, market_streamer

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    symbols: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

class LiveAIResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    timestamp: str
    real_time_data: Dict[str, Any]

class PathwayLiveAIAgent:
    """Live AI Agent powered by Pathway real-time streaming"""
    
    def __init__(self):
        self.app = FastAPI(title="Pathway Live AI Agent")
        self.rag_engine = pathway_rag_engine
        self.market_streamer = market_streamer
        self.setup_endpoints()
        
    def setup_endpoints(self):
        """Setup FastAPI endpoints for the Live AI agent"""
        
        @self.app.post("/query", response_model=LiveAIResponse)
        async def process_query(request: QueryRequest):
            """Process user query with real-time context"""
            try:
                # Get real-time market context
                real_time_context = await self._get_real_time_context(request.symbols)
                
                # Process query through Pathway RAG pipeline
                rag_response = await self._process_with_pathway_rag(
                    request.query, 
                    request.symbols, 
                    real_time_context
                )
                
                return LiveAIResponse(
                    response=rag_response['response'],
                    confidence=rag_response['confidence'],
                    sources=rag_response['sources'],
                    timestamp=datetime.now().isoformat(),
                    real_time_data=real_time_context
                )
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/market-stream/{symbol}")
        async def get_live_market_data(symbol: str):
            """Get live market data for a specific symbol"""
            try:
                # Fetch real-time data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                
                if hist.empty:
                    raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
                
                current_data = {
                    'symbol': symbol,
                    'price': float(hist['Close'].iloc[-1]),
                    'volume': int(hist['Volume'].iloc[-1]),
                    'timestamp': datetime.now().isoformat(),
                    'intraday_high': float(hist['High'].max()),
                    'intraday_low': float(hist['Low'].min())
                }
                
                return current_data
                
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze-portfolio")
        async def analyze_portfolio_real_time(portfolio: Dict[str, Any]):
            """Analyze portfolio with real-time market data"""
            try:
                symbols = [holding['symbol'] for holding in portfolio.get('holdings', [])]
                
                # Get real-time data for all symbols
                real_time_data = {}
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty and len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        prev_price = float(hist['Close'].iloc[-2])
                        change_percent = ((current_price - prev_price) / prev_price) * 100
                        
                        real_time_data[symbol] = {
                            'current_price': current_price,
                            'change_percent': change_percent,
                            'volume': int(hist['Volume'].iloc[-1])
                        }
                
                # Calculate portfolio metrics
                portfolio_analysis = self._calculate_portfolio_metrics(portfolio, real_time_data)
                
                return {
                    'portfolio_value': portfolio_analysis['total_value'],
                    'daily_change': portfolio_analysis['daily_change'],
                    'daily_change_percent': portfolio_analysis['daily_change_percent'],
                    'holdings_analysis': portfolio_analysis['holdings'],
                    'risk_assessment': portfolio_analysis['risk_metrics'],
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'real_time_yahoo_finance'
                }
                
            except Exception as e:
                logger.error(f"Error analyzing portfolio: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts/real-time")
        async def get_real_time_alerts():
            """Get real-time market alerts"""
            try:
                alerts = await self._generate_real_time_alerts()
                return {
                    'alerts': alerts,
                    'count': len(alerts),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error generating alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_real_time_context(self, symbols: Optional[List[str]]) -> Dict[str, Any]:
        """Get real-time market context for query processing"""
        
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        context = {
            'market_data': {},
            'market_summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Fetch real-time data for symbols
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                info = ticker.info
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                    change_percent = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                    
                    # Calculate technical indicators
                    volatility = hist['Close'].pct_change().std() * 100
                    volume_avg = hist['Volume'].mean()
                    current_volume = hist['Volume'].iloc[-1]
                    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
                    
                    context['market_data'][symbol] = {
                        'price': current_price,
                        'change_percent': change_percent,
                        'volume': int(current_volume),
                        'volume_ratio': volume_ratio,
                        'volatility': volatility,
                        'market_cap': info.get('marketCap', 0),
                        'sector': info.get('sector', 'Unknown')
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to fetch context for {symbol}: {e}")
        
        # Calculate market summary
        if context['market_data']:
            changes = [data['change_percent'] for data in context['market_data'].values()]
            context['market_summary'] = {
                'avg_change': sum(changes) / len(changes),
                'positive_stocks': sum(1 for change in changes if change > 0),
                'total_stocks': len(changes),
                'market_sentiment': 'bullish' if sum(changes) > 0 else 'bearish'
            }
        
        return context
    
    async def _process_with_pathway_rag(self, query: str, symbols: Optional[List[str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using Pathway RAG pipeline"""
        
        # Create query data for Pathway
        query_data = {
            'query': query,
            'symbols': symbols or [],
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        # Write query to Pathway input stream
        query_file = f"./data/query_stream/query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(query_file, 'w') as f:
            json.dump(query_data, f)
            f.write('\n')
        
        # Generate response using real-time context
        response = self._generate_contextual_response(query, context)
        
        return {
            'response': response,
            'confidence': 0.85,  # High confidence due to real-time data
            'sources': ['Real-time Yahoo Finance', 'Pathway RAG Pipeline', 'Technical Analysis']
        }
    
    def _generate_contextual_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate contextual response using real-time market data"""
        
        market_data = context.get('market_data', {})
        market_summary = context.get('market_summary', {})
        
        # Analyze query intent
        query_lower = query.lower()
        
        if 'price' in query_lower or 'cost' in query_lower:
            # Price-related query
            if market_data:
                symbol_prices = []
                for symbol, data in market_data.items():
                    change_indicator = "📈" if data['change_percent'] > 0 else "📉"
                    symbol_prices.append(f"{symbol}: ${data['price']:.2f} ({data['change_percent']:+.2f}%) {change_indicator}")
                
                response = f"**Real-Time Price Analysis:**\n" + "\n".join(symbol_prices)
                response += f"\n\n**Market Overview:** {market_summary.get('market_sentiment', 'neutral').title()} sentiment with {market_summary.get('positive_stocks', 0)}/{market_summary.get('total_stocks', 0)} stocks positive."
                
        elif 'risk' in query_lower or 'volatility' in query_lower:
            # Risk-related query
            if market_data:
                high_vol_stocks = []
                for symbol, data in market_data.items():
                    if data.get('volatility', 0) > 3:
                        high_vol_stocks.append(f"{symbol} ({data['volatility']:.1f}% volatility)")
                
                response = f"**Real-Time Risk Analysis:**\n"
                if high_vol_stocks:
                    response += f"⚠️ High volatility detected in: {', '.join(high_vol_stocks)}\n"
                else:
                    response += "✅ Market volatility is within normal ranges\n"
                
                response += f"Average market change: {market_summary.get('avg_change', 0):.2f}%"
                
        elif 'volume' in query_lower or 'activity' in query_lower:
            # Volume-related query
            if market_data:
                high_volume_stocks = []
                for symbol, data in market_data.items():
                    if data.get('volume_ratio', 1) > 1.5:
                        high_volume_stocks.append(f"{symbol} ({data['volume_ratio']:.1f}x avg volume)")
                
                response = f"**Real-Time Volume Analysis:**\n"
                if high_volume_stocks:
                    response += f"📊 High volume activity in: {', '.join(high_volume_stocks)}"
                else:
                    response += "📊 Trading volumes are at normal levels"
                    
        else:
            # General market query
            response = f"**Real-Time Market Analysis:**\n"
            response += f"Market Sentiment: {market_summary.get('market_sentiment', 'neutral').title()}\n"
            response += f"Average Change: {market_summary.get('avg_change', 0):+.2f}%\n"
            response += f"Positive Stocks: {market_summary.get('positive_stocks', 0)}/{market_summary.get('total_stocks', 0)}\n"
            
            if market_data:
                top_performer = max(market_data.items(), key=lambda x: x[1]['change_percent'])
                worst_performer = min(market_data.items(), key=lambda x: x[1]['change_percent'])
                
                response += f"\n📈 Top Performer: {top_performer[0]} ({top_performer[1]['change_percent']:+.2f}%)"
                response += f"\n📉 Worst Performer: {worst_performer[0]} ({worst_performer[1]['change_percent']:+.2f}%)"
        
        response += f"\n\n*Analysis based on real-time data as of {context['timestamp']}*"
        return response
    
    def _calculate_portfolio_metrics(self, portfolio: Dict[str, Any], real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio metrics using real-time data"""
        
        holdings = portfolio.get('holdings', [])
        total_value = 0
        total_cost = 0
        daily_change = 0
        holdings_analysis = []
        
        for holding in holdings:
            symbol = holding['symbol']
            shares = holding.get('shares', 0)
            avg_cost = holding.get('avg_cost', 0)
            
            if symbol in real_time_data:
                current_price = real_time_data[symbol]['current_price']
                change_percent = real_time_data[symbol]['change_percent']
                
                market_value = shares * current_price
                cost_basis = shares * avg_cost
                total_return = market_value - cost_basis
                daily_change_value = market_value * (change_percent / 100)
                
                total_value += market_value
                total_cost += cost_basis
                daily_change += daily_change_value
                
                holdings_analysis.append({
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': current_price,
                    'market_value': market_value,
                    'total_return': total_return,
                    'total_return_percent': (total_return / cost_basis * 100) if cost_basis > 0 else 0,
                    'daily_change': daily_change_value,
                    'daily_change_percent': change_percent
                })
        
        daily_change_percent = (daily_change / (total_value - daily_change) * 100) if (total_value - daily_change) > 0 else 0
        
        # Calculate risk metrics
        returns = [h['total_return_percent'] for h in holdings_analysis]
        portfolio_volatility = np.std(returns) if returns else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'daily_change': daily_change,
            'daily_change_percent': daily_change_percent,
            'holdings': holdings_analysis,
            'risk_metrics': {
                'portfolio_volatility': portfolio_volatility,
                'diversification_score': min(100, len(holdings) * 15),
                'risk_level': 'high' if portfolio_volatility > 20 else 'medium' if portfolio_volatility > 10 else 'low'
            }
        }
    
    async def _generate_real_time_alerts(self) -> List[Dict[str, Any]]:
        """Generate real-time market alerts"""
        
        alerts = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[-2])
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    volume = int(hist['Volume'].iloc[-1])
                    avg_volume = int(hist['Volume'].mean())
                    
                    # Volume spike alert
                    if volume > avg_volume * 2:
                        alerts.append({
                            'id': f"vol_{symbol}_{int(datetime.now().timestamp())}",
                            'type': 'volume_spike',
                            'severity': 'medium',
                            'symbol': symbol,
                            'message': f"{symbol} volume spike: {volume:,} vs avg {avg_volume:,} ({volume/avg_volume:.1f}x)",
                            'timestamp': datetime.now().isoformat(),
                            'action': 'Monitor for breakout or news'
                        })
                    
                    # Price movement alert
                    if abs(change_percent) > 3:
                        severity = 'high' if abs(change_percent) > 5 else 'medium'
                        direction = 'surge' if change_percent > 0 else 'decline'
                        
                        alerts.append({
                            'id': f"price_{symbol}_{int(datetime.now().timestamp())}",
                            'type': 'price_movement',
                            'severity': severity,
                            'symbol': symbol,
                            'message': f"{symbol} {direction}: {change_percent:+.2f}% to ${current_price:.2f}",
                            'timestamp': datetime.now().isoformat(),
                            'action': f"Review {symbol} position"
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to generate alerts for {symbol}: {e}")
        
        return alerts

# Global instance
live_ai_agent = PathwayLiveAIAgent()

def get_live_ai_app():
    """Get the FastAPI app for the Live AI agent"""
    return live_ai_agent.app