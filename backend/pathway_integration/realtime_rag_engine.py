"""
🚀 Pathway Real-Time RAG Engine for FinanceGPT
==============================================
Live AI with real-time data streaming and dynamic knowledge updates
"""

import pathway as pw
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PathwayFinanceRAG:
    """Real-time RAG engine using Pathway for live financial data"""
    
    def __init__(self):
        self.market_stream = None
        self.news_stream = None
        self.vector_store = None
        self.knowledge_base = {}
        
    def setup_market_data_stream(self):
        """Setup real-time market data ingestion using Pathway"""
        
        # Define market data schema
        market_schema = pw.Schema.from_types(
            symbol=str,
            price=float,
            change=float,
            change_percent=float,
            volume=int,
            timestamp=str,
            market_cap=Optional[float],
            pe_ratio=Optional[float]
        )
        
        # Create streaming connector for market data
        self.market_stream = pw.io.jsonlines.read(
            "./data/market_stream",
            schema=market_schema,
            mode="streaming"
        )
        
        # Transform and enrich market data in real-time
        enriched_market = self.market_stream.select(
            symbol=pw.this.symbol,
            price=pw.this.price,
            change=pw.this.change,
            change_percent=pw.this.change_percent,
            volume=pw.this.volume,
            timestamp=pw.this.timestamp,
            # Add computed fields
            price_momentum=pw.apply(self._calculate_momentum, pw.this.symbol, pw.this.change_percent),
            risk_level=pw.apply(self._assess_risk, pw.this.change_percent, pw.this.volume),
            market_sentiment=pw.apply(self._calculate_sentiment, pw.this.change_percent, pw.this.volume)
        )
        
        return enriched_market
    
    def setup_news_stream(self):
        """Setup real-time news ingestion and sentiment analysis"""
        
        news_schema = pw.Schema.from_types(
            title=str,
            content=str,
            source=str,
            timestamp=str,
            symbols=List[str],
            sentiment_score=Optional[float]
        )
        
        self.news_stream = pw.io.jsonlines.read(
            "./data/news_stream",
            schema=news_schema,
            mode="streaming"
        )
        
        # Process news with sentiment analysis
        processed_news = self.news_stream.select(
            title=pw.this.title,
            content=pw.this.content,
            source=pw.this.source,
            timestamp=pw.this.timestamp,
            symbols=pw.this.symbols,
            # Real-time sentiment analysis
            sentiment=pw.apply(self._analyze_news_sentiment, pw.this.content),
            market_impact=pw.apply(self._assess_market_impact, pw.this.content, pw.this.symbols),
            urgency_score=pw.apply(self._calculate_urgency, pw.this.title, pw.this.content)
        )
        
        return processed_news
    
    def setup_vector_store(self):
        """Setup real-time vector indexing for RAG"""
        
        # Combine market and news data for vector indexing
        combined_data = self.market_stream + self.news_stream
        
        # Create embeddings in real-time
        embedded_data = combined_data.select(
            content=pw.apply(self._create_content_for_embedding, pw.this),
            embedding=pw.apply(self._generate_embedding, pw.this),
            metadata=pw.apply(self._extract_metadata, pw.this),
            timestamp=pw.this.timestamp
        )
        
        # Setup vector store with real-time indexing
        self.vector_store = pw.stdlib.ml.index(
            embedded_data,
            embedded_data.embedding,
            n_dimensions=1536,  # OpenAI embedding dimension
            approximate=True
        )
        
        return self.vector_store
    
    def create_live_rag_pipeline(self):
        """Create complete real-time RAG pipeline"""
        
        # Setup all streams
        market_data = self.setup_market_data_stream()
        news_data = self.setup_news_stream()
        vector_store = self.setup_vector_store()
        
        # Create query processing pipeline
        query_schema = pw.Schema.from_types(
            query=str,
            symbols=Optional[List[str]],
            timestamp=str
        )
        
        query_stream = pw.io.jsonlines.read(
            "./data/query_stream",
            schema=query_schema,
            mode="streaming"
        )
        
        # Process queries with real-time context
        query_results = query_stream.select(
            query=pw.this.query,
            symbols=pw.this.symbols,
            timestamp=pw.this.timestamp,
            # Retrieve relevant context
            context=pw.apply(self._retrieve_context, pw.this.query, pw.this.symbols),
            # Generate response with LLM
            response=pw.apply(self._generate_llm_response, pw.this.query, pw.this.context),
            # Add confidence and sources
            confidence=pw.apply(self._calculate_response_confidence, pw.this.context),
            sources=pw.apply(self._extract_sources, pw.this.context)
        )
        
        return query_results
    
    def _calculate_momentum(self, symbol: str, change_percent: float) -> str:
        """Calculate price momentum indicator"""
        if abs(change_percent) > 5:
            return "strong"
        elif abs(change_percent) > 2:
            return "moderate"
        else:
            return "weak"
    
    def _assess_risk(self, change_percent: float, volume: int) -> str:
        """Assess risk level based on price movement and volume"""
        volatility = abs(change_percent)
        volume_factor = 1 if volume > 1000000 else 0.5
        
        risk_score = volatility * volume_factor
        
        if risk_score > 5:
            return "high"
        elif risk_score > 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_sentiment(self, change_percent: float, volume: int) -> float:
        """Calculate market sentiment score"""
        # Positive change with high volume = bullish
        # Negative change with high volume = bearish
        volume_weight = min(volume / 1000000, 2.0)  # Cap at 2x weight
        sentiment = change_percent * volume_weight / 100
        return max(-1.0, min(1.0, sentiment))  # Normalize to [-1, 1]
    
    def _analyze_news_sentiment(self, content: str) -> float:
        """Analyze news sentiment using NLP"""
        # Simplified sentiment analysis - in production use proper NLP models
        positive_words = ['growth', 'profit', 'gain', 'rise', 'increase', 'bullish', 'positive']
        negative_words = ['loss', 'decline', 'fall', 'decrease', 'bearish', 'negative', 'risk']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return sentiment
    
    def _assess_market_impact(self, content: str, symbols: List[str]) -> str:
        """Assess potential market impact of news"""
        impact_keywords = ['earnings', 'merger', 'acquisition', 'regulation', 'lawsuit', 'breakthrough']
        content_lower = content.lower()
        
        impact_count = sum(1 for keyword in impact_keywords if keyword in content_lower)
        symbol_count = len(symbols) if symbols else 0
        
        if impact_count >= 2 or symbol_count >= 3:
            return "high"
        elif impact_count >= 1 or symbol_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _calculate_urgency(self, title: str, content: str) -> float:
        """Calculate urgency score for news"""
        urgent_words = ['breaking', 'urgent', 'alert', 'immediate', 'emergency']
        text = (title + " " + content).lower()
        
        urgency_count = sum(1 for word in urgent_words if word in text)
        return min(urgency_count / 2.0, 1.0)  # Normalize to [0, 1]
    
    def _create_content_for_embedding(self, data: Dict[str, Any]) -> str:
        """Create text content for embedding generation"""
        if 'symbol' in data:  # Market data
            return f"Stock {data['symbol']} price ${data['price']:.2f} change {data['change_percent']:.2f}% volume {data['volume']} sentiment {data.get('market_sentiment', 0):.2f}"
        elif 'title' in data:  # News data
            return f"News: {data['title']} Content: {data['content'][:500]} Sentiment: {data.get('sentiment', 0):.2f}"
        else:
            return str(data)
    
    def _generate_embedding(self, data: Dict[str, Any]) -> List[float]:
        """Generate embeddings for vector search"""
        # Simplified embedding - in production use OpenAI or other embedding models
        content = self._create_content_for_embedding(data)
        # Return dummy embedding for now
        return [0.1] * 1536
    
    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for search results"""
        return {
            'type': 'market' if 'symbol' in data else 'news',
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'symbols': data.get('symbols', [data.get('symbol')] if 'symbol' in data else [])
        }
    
    def _retrieve_context(self, query: str, symbols: Optional[List[str]]) -> Dict[str, Any]:
        """Retrieve relevant context for query"""
        # Simplified context retrieval - in production use vector search
        return {
            'market_data': f"Latest market data for {symbols if symbols else 'general market'}",
            'news_data': f"Recent news related to query: {query}",
            'technical_analysis': "Technical indicators and patterns",
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_llm_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate LLM response using retrieved context"""
        # Simplified response generation - in production use actual LLM
        return f"Based on real-time analysis: {query}. Context includes {context['market_data']} and {context['news_data']}. Analysis timestamp: {context['timestamp']}"
    
    def _calculate_response_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence score for response"""
        # Simplified confidence calculation
        data_freshness = 0.9  # Assume fresh data
        context_relevance = 0.8  # Assume relevant context
        return (data_freshness + context_relevance) / 2
    
    def _extract_sources(self, context: Dict[str, Any]) -> List[str]:
        """Extract data sources used in response"""
        return ['Real-time market data', 'Live news feeds', 'Technical analysis', 'Vector search']

class PathwayMarketDataStreamer:
    """Real-time market data streamer using Pathway"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        self.data_dir = Path("./data/market_stream")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def start_streaming(self):
        """Start streaming real market data"""
        while True:
            try:
                # Fetch real market data
                market_data = await self._fetch_real_market_data()
                
                # Write to Pathway input stream
                for data in market_data:
                    timestamp = datetime.now().isoformat()
                    filename = f"market_{timestamp.replace(':', '-')}.jsonl"
                    filepath = self.data_dir / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(data, f)
                        f.write('\n')
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in market data streaming: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _fetch_real_market_data(self) -> List[Dict[str, Any]]:
        """Fetch real market data from Yahoo Finance"""
        market_data = []
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                info = ticker.info
                
                if not hist.empty and len(hist) >= 2:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[-2])
                    change = current_price - prev_price
                    change_percent = (change / prev_price) * 100
                    volume = int(hist['Volume'].iloc[-1])
                    
                    market_data.append({
                        'symbol': symbol,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': volume,
                        'timestamp': datetime.now().isoformat(),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0)
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
        
        return market_data

# Global instances
pathway_rag_engine = PathwayFinanceRAG()
market_streamer = PathwayMarketDataStreamer()

async def initialize_pathway_system():
    """Initialize the complete Pathway-based system"""
    
    # Setup RAG pipeline
    rag_pipeline = pathway_rag_engine.create_live_rag_pipeline()
    
    # Start market data streaming
    streaming_task = asyncio.create_task(market_streamer.start_streaming())
    
    # Run Pathway computation
    pw.run()
    
    return rag_pipeline, streaming_task