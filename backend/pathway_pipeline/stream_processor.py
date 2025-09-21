import asyncio
import pathway as pw
from typing import Dict, List, Any, Optional, Callable
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from core.config import settings
from data_sources.yahoo_finance import YahooFinanceConnector
from data_sources.news_apis import NewsAPIConnector
from data_sources.sentiment_feeds import SentimentAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class StreamMessage:
    """Data structure for stream messages"""
    source: str
    message_type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 0

class FinanceStreamProcessor:
    def __init__(self, websocket_manager=None, agents=None):
        self.websocket_manager = websocket_manager
        self.agents = agents or {}
        self.is_running = False
        self.tables = {}
        self.connectors = {}
        self.subscribers = {}
        
        # Performance metrics
        self.processed_messages = 0
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # Initialize data sources
        self._initialize_connectors()
        
        # Initialize Pathway tables
        self._initialize_pathway_tables()
        
        logger.info("✅ FinanceStreamProcessor initialized")
    
    def _initialize_connectors(self):
        """Initialize all data source connectors"""
        logger.info("🔌 Initializing data connectors...")
        
        try:
            self.connectors = {
                'yahoo_finance': YahooFinanceConnector(),
                'news_api': NewsAPIConnector(),
                'sentiment': SentimentAnalyzer()
            }
            logger.info(f"✅ Initialized {len(self.connectors)} connectors")
        except Exception as e:
            logger.error(f"❌ Error initializing connectors: {e}")
    
    def _initialize_pathway_tables(self):
        """Initialize Pathway streaming tables"""
        logger.info(" Setting up Pathway streaming tables...")
        
        try:
            # Market data stream
            self.tables['market_data'] = pw.io.jsonlines.read(
                "./data/market_stream",
                schema=pw.schema_from_types(
                    symbol=str,
                    price=float,
                    volume=int,
                    timestamp=str,
                    change=float,
                    change_percent=float
                ),
                mode="streaming"
            )
            
            # News data stream  
            self.tables['news_data'] = pw.io.jsonlines.read(
                "./data/news_stream",
                schema=pw.schema_from_types(
                    headline=str,
                    content=str,
                    source=str,
                    symbols=str,
                    sentiment_score=float,
                    timestamp=str
                ),
                mode="streaming"
            )
            
            # Trading signals stream
            self.tables['trading_signals'] = pw.io.jsonlines.read(
                "./data/signals_stream",
                schema=pw.schema_from_types(
                    symbol=str,
                    signal_type=str,
                    strength=float,
                    confidence=float,
                    agent_source=str,
                    timestamp=str
                ),
                mode="streaming"
            )
            
            logger.info("✅ Pathway tables initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Pathway tables: {e}")
    
    def setup_market_data_pipeline(self):
        """Setup real-time market data processing pipeline"""
        
        # Market data enrichment
        enriched_market_data = self.tables['market_data'].select(
            pw.this.symbol,
            pw.this.price,
            pw.this.volume,
            pw.this.timestamp,
            pw.this.change,
            pw.this.change_percent,
            # Add moving averages
            ma_5=pw.apply(self._calculate_moving_average, pw.this.price, period=5),
            ma_20=pw.apply(self._calculate_moving_average, pw.this.price, period=20),
            # Add volatility
            volatility=pw.apply(self._calculate_volatility, pw.this.price),
            # Add momentum indicators
            rsi=pw.apply(self._calculate_rsi, pw.this.price),
            macd=pw.apply(self._calculate_macd, pw.this.price),
        )
        
        # Alert generation
        alerts = enriched_market_data.filter(
            (pw.this.change_percent > 5.0) | 
            (pw.this.change_percent < -5.0) |
            (pw.this.volume > pw.apply(self._get_avg_volume, pw.this.symbol) * 2)
        ).select(
            pw.this.symbol,
            alert_type=pw.apply(self._determine_alert_type, pw.this),
            severity=pw.apply(self._calculate_alert_severity, pw.this),
            timestamp=pw.this.timestamp
        )
        
        # Output streams
        pw.io.jsonlines.write(enriched_market_data, "./output/enriched_market_data")
        pw.io.jsonlines.write(alerts, "./output/market_alerts")
        
        return enriched_market_data, alerts
    
    def setup_news_analysis_pipeline(self):
        """Setup news analysis and sentiment processing"""
        
        # News sentiment analysis
        analyzed_news = self.tables['news_data'].select(
            pw.this.headline,
            pw.this.content,
            pw.this.source,
            pw.this.symbols,
            pw.this.timestamp,
            # Enhanced sentiment analysis
            sentiment_score=pw.apply(self._analyze_sentiment, pw.this.content),
            sentiment_label=pw.apply(self._get_sentiment_label, pw.this.sentiment_score),
            # Extract key entities and topics
            entities=pw.apply(self._extract_entities, pw.this.content),
            topics=pw.apply(self._extract_topics, pw.this.content),
            # Calculate news impact score
            impact_score=pw.apply(self._calculate_news_impact, pw.this),
        )
        
        # High-impact news alerts
        news_alerts = analyzed_news.filter(
            (pw.this.impact_score > 0.7) | 
            (pw.this.sentiment_score > 0.8) | 
            (pw.this.sentiment_score < -0.8)
        ).select(
            pw.this.headline,
            pw.this.symbols,
            pw.this.sentiment_label,
            pw.this.impact_score,
            pw.this.timestamp
        )
        
        # Output streams
        pw.io.jsonlines.write(analyzed_news, "./output/analyzed_news")
        pw.io.jsonlines.write(news_alerts, "./output/news_alerts")
        
        return analyzed_news, news_alerts
    
    def setup_signal_aggregation_pipeline(self):
        """Aggregate and rank trading signals from multiple agents"""
        
        # Signal aggregation and ranking
        aggregated_signals = self.tables['trading_signals'].groupby(
            pw.this.symbol,
            pw.this.signal_type
        ).reduce(
            symbol=pw.this.symbol,
            signal_type=pw.this.signal_type,
            avg_strength=pw.reducers.avg(pw.this.strength),
            avg_confidence=pw.reducers.avg(pw.this.confidence),
            signal_count=pw.reducers.count(),
            agents=pw.reducers.tuple(pw.this.agent_source),
            latest_timestamp=pw.reducers.max(pw.this.timestamp)
        ).select(
            pw.this.symbol,
            pw.this.signal_type,
            pw.this.avg_strength,
            pw.this.avg_confidence,
            pw.this.signal_count,
            # Calculate consensus score
            consensus_score=pw.apply(self._calculate_consensus_score, pw.this),
            # Rank signals
            rank=pw.apply(self._rank_signals, pw.this),
            agents=pw.this.agents,
            timestamp=pw.this.latest_timestamp
        )
        
        # Top signals for execution
        top_signals = aggregated_signals.filter(
            (pw.this.consensus_score > 0.75) & (pw.this.signal_count >= 2)
        ).sort(key=pw.this.consensus_score, direction="desc")
        
        # Output streams
        pw.io.jsonlines.write(aggregated_signals, "./output/aggregated_signals")
        pw.io.jsonlines.write(top_signals, "./output/top_signals")
        
        return aggregated_signals, top_signals
    
    async def start(self):
        """Start the streaming pipeline"""
        logger.info(" Starting FinanceStreamProcessor...")
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            # Setup all pipelines
            market_data, market_alerts = self.setup_market_data_pipeline()
            news_data, news_alerts = self.setup_news_analysis_pipeline()
            signals, top_signals = self.setup_signal_aggregation_pipeline()
            
            # Start data connectors
            await self._start_data_connectors()
            
            # Start Pathway computation
            pw.run(
                monitoring_level=pw.MonitoringLevel.ALL,
                persistence_config=pw.PersistenceConfig.simple_config(
                    pw.persistence.Backend.filesystem(settings.pathway_persistence_dir)
                )
            )
            
            logger.info("✅ FinanceStreamProcessor started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting stream processor: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the streaming pipeline"""
        logger.info(" Stopping FinanceStreamProcessor...")
        
        self.is_running = False
        
        # Stop data connectors
        for connector in self.connectors.values():
            if hasattr(connector, 'stop'):
                await connector.stop()
        
        logger.info("✅ FinanceStreamProcessor stopped")
    
    async def _start_data_connectors(self):
        """Start all data source connectors"""
        logger.info("🔌 Starting data connectors...")
        
        for name, connector in self.connectors.items():
            try:
                if hasattr(connector, 'start'):
                    await connector.start()
                logger.info(f"✅ Started {name} connector")
            except Exception as e:
                logger.error(f"❌ Error starting {name} connector: {e}")
    
    # Technical Analysis Functions
    def _calculate_moving_average(self, prices: List[float], period: int = 20) -> float:
        """Calculate moving average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        return np.mean(prices[-period:])
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
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
    
    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd_line = ema12 - ema26
        
        # Signal line (9-period EMA of MACD)
        signal_line = 0.0  # Simplified for brevity
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not prices:
            return 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    # News Analysis Functions
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of news text"""
        # Placeholder for advanced sentiment analysis
        # In production, use transformer models or APIs
        positive_words = ["growth", "profit", "increase", "positive", "bull", "gain"]
        negative_words = ["loss", "decline", "negative", "bear", "fall", "drop"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment * 10))  # Scale to [-1, 1]
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.3:
            return "POSITIVE"
        elif score < -0.3:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Placeholder for NER
        return ["ENTITY1", "ENTITY2"]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Placeholder for topic modeling
        return ["TOPIC1", "TOPIC2"]
    
    def _calculate_news_impact(self, news_row) -> float:
        """Calculate potential market impact of news"""
        base_impact = abs(news_row.sentiment_score)
        
        # Boost impact based on source credibility
        credible_sources = ["Reuters", "Bloomberg", "WSJ", "Financial Times"]
        if news_row.source in credible_sources:
            base_impact *= 1.5
        
        # Boost impact based on number of symbols mentioned
        symbols_count = len(news_row.symbols.split(",")) if news_row.symbols else 1
        base_impact *= min(2.0, 1 + symbols_count * 0.1)
        
        return min(1.0, base_impact)
    
    # Signal Processing Functions
    def _calculate_consensus_score(self, signal_row) -> float:
        """Calculate consensus score for aggregated signals"""
        strength_weight = 0.4
        confidence_weight = 0.4
        count_weight = 0.2
        
        strength_score = signal_row.avg_strength
        confidence_score = signal_row.avg_confidence
        count_score = min(1.0, signal_row.signal_count / 5.0)  # Normalize to max 5 agents
        
        consensus = (
            strength_score * strength_weight +
            confidence_score * confidence_weight +
            count_score * count_weight
        )
        
        return consensus
    
    def _rank_signals(self, signal_row) -> int:
        """Rank signals based on consensus score"""
        # This would be implemented with proper ranking logic
        return int(signal_row.consensus_score * 100)
    
    # Alert Functions
    def _determine_alert_type(self, market_row) -> str:
        """Determine type of market alert"""
        if abs(market_row.change_percent) > 10:
            return "EXTREME_MOVE"
        elif abs(market_row.change_percent) > 5:
            return "SIGNIFICANT_MOVE"
        elif market_row.volume > self._get_avg_volume(market_row.symbol) * 3:
            return "HIGH_VOLUME"
        else:
            return "REGULAR"
    
    def _calculate_alert_severity(self, market_row) -> str:
        """Calculate alert severity"""
        if abs(market_row.change_percent) > 15:
            return "CRITICAL"
        elif abs(market_row.change_percent) > 10:
            return "HIGH"
        elif abs(market_row.change_percent) > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_avg_volume(self, symbol: str) -> int:
        """Get average volume for a symbol (placeholder)"""
        # In production, this would query historical data
        return 1000000
    
    # Performance Metrics
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get stream processor performance metrics"""
        uptime = datetime.now() - self.start_time
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "processed_messages": self.processed_messages,
            "messages_per_second": self.processed_messages / uptime.total_seconds() if uptime.total_seconds() > 0 else 0,
            "is_running": self.is_running,
            "active_connectors": len([c for c in self.connectors.values() if hasattr(c, 'is_active') and c.is_active]),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "memory_usage": "N/A",  # Would implement proper memory monitoring
        }
