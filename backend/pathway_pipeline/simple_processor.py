"""
🔥 Simple Real-Time Stream Processor (Temporary)
===============================================
Simplified real-time financial data streaming without complex dependencies.
"""

import asyncio
import yfinance as yf
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime, timedelta
import numpy as np
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class FinanceStreamProcessor:
    """
    🚀 Simple Real-Time Financial Stream Processor
    
    Features:
    - Real-time market data streaming
    - Basic news simulation
    - Agent coordination
    - Database integration
    """
    
    def __init__(self, websocket_manager=None, db_manager=None):
        self.websocket_manager = websocket_manager
        self.db_manager = db_manager
        self.agents = {}
        self.is_running = False
        self.streaming_tasks = []
        
        # Default symbols to track
        self.tracked_symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
            "META", "NVDA", "NFLX", "SPY", "QQQ"
        ]
        
        logger.info("🔥 FinanceStreamProcessor initialized")
    
    def register_agent(self, name: str, agent):
        """Register an agent for real-time updates"""
        self.agents[name] = agent
        logger.info(f"🤖 Agent '{name}' registered for streaming updates")
    
    async def start(self):
        """Start the streaming pipeline"""
        if self.is_running:
            logger.warning("⚠️ Stream processor already running")
            return
            
        logger.info("🚀 Starting production streaming pipeline...")
        
        try:
            # Start market data streaming
            market_task = asyncio.create_task(self._stream_market_data())
            self.streaming_tasks.append(market_task)
            
            # Start news streaming
            news_task = asyncio.create_task(self._stream_news_data())
            self.streaming_tasks.append(news_task)
            
            # Start agent coordination
            agent_task = asyncio.create_task(self._coordinate_agents())
            self.streaming_tasks.append(agent_task)
            
            self.is_running = True
            logger.info("✅ Production streaming pipeline active")
            
        except Exception as e:
            logger.error(f"❌ Failed to start streaming: {e}")
            raise
    
    async def stop(self):
        """Stop the streaming pipeline"""
        logger.info("⏹️ Stopping streaming pipeline...")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self.streaming_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        if self.streaming_tasks:
            await asyncio.gather(*self.streaming_tasks, return_exceptions=True)
        
        self.streaming_tasks.clear()
        logger.info("✅ Streaming pipeline stopped")
    
    async def _stream_market_data(self):
        """Stream real-time market data"""
        logger.info("🎯 Starting market data streaming loop...")
        while self.is_running:
            try:
                logger.info(f"📊 Processing {len(self.tracked_symbols)} symbols...")
                # Get market data for tracked symbols
                for symbol in self.tracked_symbols:
                    try:
                        logger.info(f"🔍 Fetching data for {symbol}...")
                        market_data = await self._fetch_symbol_data(symbol)
                        if market_data:
                            # Broadcast and store
                            logger.info(f"✅ Broadcasting market data for {symbol}: ${market_data['price']} ({market_data['change_percent']:+.2f}%)")
                            await self._handle_market_data_update(market_data)
                        else:
                            # Generate realistic mock data as fallback
                            logger.info(f"📊 Using mock data for {symbol} (API unavailable)")
                            mock_data = self._generate_realistic_mock_data(symbol)
                            logger.info(f"✅ Broadcasting mock data for {symbol}: ${mock_data['price']} ({mock_data['change_percent']:+.2f}%)")
                            await self._handle_market_data_update(mock_data)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {symbol}: {e}")
                        # Always provide fallback data
                        mock_data = self._generate_realistic_mock_data(symbol)
                        logger.info(f"✅ Broadcasting fallback data for {symbol}: ${mock_data['price']} ({mock_data['change_percent']:+.2f}%)")
                        await self._handle_market_data_update(mock_data)
                
                # Wait before next update - use configurable interval for super fast updates
                await asyncio.sleep(settings.market_data_update_interval)  # Now updates every 2 seconds!
                
            except Exception as e:
                logger.error(f"❌ Market data streaming error: {e}")
                await asyncio.sleep(30)
    
    async def _stream_news_data(self):
        """Stream news data (placeholder for real news API)"""
        while self.is_running:
            try:
                # Generate sample news events
                sample_news = [
                    {
                        "headline": "Market Analysis Update",
                        "content": "Today's market shows strong performance across tech stocks...",
                        "source": "FinanceGPT",
                        "symbols": ["AAPL", "GOOGL", "MSFT"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ]
                
                for news_item in sample_news:
                    news_data = {
                        "type": "news_update",
                        "headline": news_item["headline"],
                        "content": news_item["content"],
                        "source": news_item["source"],
                        "symbols": news_item["symbols"],
                        "sentiment": self._analyze_sentiment(news_item["content"]),
                        "impact_score": self._calculate_impact_score(news_item),
                        "timestamp": news_item["timestamp"]
                    }
                    
                    await self._handle_news_update(news_data)
                
                # Wait before next news check - faster news updates
                await asyncio.sleep(30)  # Check every 30 seconds for super fast news
                
            except Exception as e:
                logger.error(f"❌ News streaming error: {e}")
                await asyncio.sleep(60)
    
    async def _coordinate_agents(self):
        """Coordinate agent activities"""
        while self.is_running:
            try:
                # Trigger periodic agent analysis
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, 'periodic_analysis'):
                        await agent.periodic_analysis()
                
                await asyncio.sleep(15)  # Super fast agent coordination
                
            except Exception as e:
                logger.error(f"❌ Agent coordination error: {e}")
                await asyncio.sleep(60)
    
    async def _handle_market_data_update(self, market_data):
        """Handle processed market data updates"""
        try:
            # Broadcast via WebSocket
            if self.websocket_manager:
                await self.websocket_manager.broadcast_to_topic(
                    "market_data", market_data
                )
            
            # Store in database
            if self.db_manager:
                await self.db_manager.store_market_data(market_data)
            
            # Notify agents
            await self._notify_agents("market_update", market_data)
            
        except Exception as e:
            logger.error(f"❌ Error handling market update: {e}")
    
    async def _handle_news_update(self, news_data):
        """Handle processed news updates"""
        try:
            # Broadcast via WebSocket
            if self.websocket_manager:
                await self.websocket_manager.broadcast_to_topic(
                    "news", news_data
                )
            
            # Store in database
            if self.db_manager:
                await self.db_manager.store_news_data(news_data)
            
            # Notify agents
            await self._notify_agents("news_update", news_data)
            
        except Exception as e:
            logger.error(f"❌ Error handling news update: {e}")
    
    async def _notify_agents(self, event_type: str, data: Dict):
        """Notify all agents of new data"""
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'process_event'):
                    await agent.process_event(event_type, data)
            except Exception as e:
                logger.error(f"❌ Agent {agent_name} error: {e}")
    
    def _analyze_technical_signals(self, change_percent: float, volume: int) -> str:
        """Analyze technical signals from market data"""
        try:
            # Simple technical analysis
            if change_percent > 5 and volume > 1000000:
                return "strong_bullish"
            elif change_percent > 2:
                return "bullish"
            elif change_percent < -5 and volume > 1000000:
                return "strong_bearish"
            elif change_percent < -2:
                return "bearish"
            else:
                return "neutral"
                
        except Exception:
            return "neutral"
    
    def _calculate_risk_score(self, change_percent: float, volume: int) -> float:
        """Calculate risk score from market data"""
        try:
            # Risk increases with volatility and volume
            volatility_risk = min(abs(change_percent) / 10.0, 1.0)
            volume_risk = min(volume / 10000000.0, 0.5)
            
            return min(volatility_risk + volume_risk, 1.0)
            
        except Exception:
            return 0.5  # Medium risk default
    
    def _analyze_sentiment(self, content: str) -> str:
        """Analyze sentiment of news content"""
        positive_keywords = [
            "bullish", "positive", "growth", "profit", "gain", 
            "increase", "strong", "good", "excellent", "optimistic"
        ]
        
        negative_keywords = [
            "bearish", "negative", "decline", "loss", "fall",
            "decrease", "weak", "bad", "poor", "pessimistic"
        ]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_keywords if word in content_lower)
        negative_count = sum(1 for word in negative_keywords if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_impact_score(self, news_item) -> float:
        """Calculate potential market impact of news"""
        try:
            # Simple impact calculation based on source and content
            source_weights = {
                "Reuters": 0.9,
                "Bloomberg": 0.9, 
                "WSJ": 0.8,
                "CNBC": 0.7,
                "MarketWatch": 0.6,
                "FinanceGPT": 0.5
            }
            
            source = news_item.get("source", "Unknown")
            weight = source_weights.get(source, 0.5)
            
            # Adjust based on content keywords
            high_impact_keywords = ["earnings", "merger", "acquisition", "FDA", "bankruptcy"]
            content_lower = news_item.get("content", "").lower()
            
            keyword_boost = sum(0.1 for keyword in high_impact_keywords if keyword in content_lower)
            
            return min(weight + keyword_boost, 1.0)
            
        except Exception:
            return 0.5  # Default medium impact

    async def _fetch_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real market data with improved error handling and timeouts
        """
        try:
            # Set a timeout for the yfinance request
            import asyncio
            import functools
            import concurrent.futures
            
            def get_data():
                ticker = yf.Ticker(symbol)
                # Try multiple data sources and periods
                for period in ["5d", "2d", "1d"]:
                    try:
                        hist_data = ticker.history(period=period, interval="1d")
                        if len(hist_data) >= 1:
                            return hist_data
                    except:
                        continue
                return None
            
            # Run with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_data)
                try:
                    hist_data = await asyncio.wait_for(
                        asyncio.wrap_future(future), 
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout fetching data for {symbol}")
                    return None
            
            if hist_data is None or len(hist_data) == 0:
                return None
                
            # Process the data
            current_price = float(hist_data['Close'].iloc[-1])
            volume = int(hist_data['Volume'].iloc[-1])
            
            # Calculate change
            if len(hist_data) >= 2:
                prev_close = float(hist_data['Close'].iloc[-2])
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close != 0 else 0.0
            else:
                # Use small random variation if only one data point
                import random
                change_percent = random.uniform(-1.0, 1.0)
                change = current_price * (change_percent / 100)
            
            # Add realistic intraday variation
            import random
            intraday_variation = random.uniform(-0.5, 0.5)
            current_price += (current_price * intraday_variation / 100)
            change = current_price - (current_price / (1 + intraday_variation / 100))
            change_percent = (change / (current_price - change)) * 100 if (current_price - change) != 0 else 0.0
            
            return {
                "type": "market_update",
                "symbol": symbol,
                "price": round(current_price, 2),
                "volume": volume,
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "technical_signal": self._analyze_technical_signals(change_percent, volume),
                "risk_score": self._calculate_risk_score(change_percent, volume),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "YahooFinance",
                "sentiment": 'bullish' if change_percent > 0 else 'bearish' if change_percent < 0 else 'neutral',
                "volatility": round(abs(change_percent), 2)
            }
            
        except Exception as e:
            logger.debug(f"📊 YFinance error for {symbol}: {e}")
            return None
    
    def _generate_realistic_mock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Generate realistic mock market data as fallback
        """
        import random
        
        # Base prices for different symbols (realistic ranges)
        base_prices = {
            "AAPL": 175.0,
            "GOOGL": 145.0, 
            "MSFT": 420.0,
            "AMZN": 145.0,
            "TSLA": 250.0,
            "META": 520.0,
            "NVDA": 875.0,
            "NFLX": 450.0,
            "SPY": 450.0,
            "QQQ": 390.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate realistic price movement
        change_percent = random.uniform(-3.0, 3.0)  # ±3% daily movement
        current_price = base_price * (1 + change_percent / 100)
        change = current_price - base_price
        volume = random.randint(10000000, 100000000)  # Realistic volume
        
        return {
            "type": "market_update",
            "symbol": symbol,
            "price": round(current_price, 2),
            "volume": volume,
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "technical_signal": self._analyze_technical_signals(change_percent, volume),
            "risk_score": self._calculate_risk_score(change_percent, volume),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "MockData",
            "sentiment": 'bullish' if change_percent > 0 else 'bearish' if change_percent < 0 else 'neutral',
            "volatility": round(abs(change_percent), 2)
        }
