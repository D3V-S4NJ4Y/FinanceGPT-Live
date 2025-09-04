"""
🚀 LiveAI Engine - Pathway-Inspired Real-Time RAG System
=====================================================
Hackathon-Ready Real-Time Financial Intelligence System
Inspired by Pathway's LiveAI concepts but working on Windows
"""

import asyncio
import aiohttp
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import pandas as pd
import feedparser
import re
from collections import deque
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LiveDataPoint:
    """Real-time data point for streaming"""
    id: str
    timestamp: datetime
    source: str
    data_type: str
    content: Dict[str, Any]
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class FinancialStream:
    """Financial data stream configuration"""
    name: str
    symbols: List[str]
    refresh_rate: float
    data_source: str
    active: bool = True

class LiveAIVectorStore:
    """In-memory vector store for real-time embeddings"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.vectors = deque(maxlen=max_size)
        self.index = {}
        self.last_updated = datetime.now()
        
    def add_vector(self, data_point: LiveDataPoint):
        """Add vector to store"""
        if data_point.embedding:
            self.vectors.append(data_point)
            self.index[data_point.id] = len(self.vectors) - 1
            self.last_updated = datetime.now()
            
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[LiveDataPoint]:
        """Find similar vectors using cosine similarity"""
        if not query_embedding or not self.vectors:
            return []
            
        similarities = []
        query_vec = np.array(query_embedding)
        
        for i, point in enumerate(self.vectors):
            if point.embedding:
                vec = np.array(point.embedding)
                similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
                similarities.append((similarity, point))
                
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [point for _, point in similarities[:top_k]]

class RealTimeRAGProcessor:
    """Real-time RAG processing engine"""
    
    def __init__(self, vector_store: LiveAIVectorStore):
        self.vector_store = vector_store
        self.context_window = 5000  # tokens
        self.embedding_cache = {}
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding (placeholder for real model)"""
        # Simple hash-based embedding for demo
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hex to float array
        embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, len(text_hash), 2)]
        # Pad to standard size
        while len(embedding) < 384:
            embedding.append(0.0)
        return embedding[:384]
    
    async def process_query(self, query: str, context_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Process RAG query with real-time context"""
        query_embedding = await self.generate_embedding(query)
        
        # Find relevant context
        relevant_docs = self.vector_store.similarity_search(query_embedding, top_k=10)
        
        # Build context
        context_texts = []
        for doc in relevant_docs:
            if context_filter:
                # Apply filters
                if context_filter.get('data_type') and doc.data_type != context_filter['data_type']:
                    continue
                if context_filter.get('max_age_minutes'):
                    age = (datetime.now() - doc.timestamp).total_seconds() / 60
                    if age > context_filter['max_age_minutes']:
                        continue
            
            context_texts.append(f"[{doc.timestamp.strftime('%H:%M:%S')}] {doc.source}: {json.dumps(doc.content)}")
        
        context = "\n".join(context_texts[:10])  # Limit context size
        
        return {
            "query": query,
            "context": context,
            "relevant_docs": len(relevant_docs),
            "timestamp": datetime.now().isoformat(),
            "generated_response": f"Based on real-time data: {context[:200]}..." if context else "No relevant real-time data found"
        }

class LiveAIStreamProcessor:
    """Main LiveAI stream processing engine"""
    
    def __init__(self):
        self.vector_store = LiveAIVectorStore(max_size=50000)
        self.rag_processor = RealTimeRAGProcessor(self.vector_store)
        self.streams = {}
        self.is_running = False
        self.stream_tasks = []
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("🚀 LiveAI Stream Processor initialized")
        
    async def start(self):
        """Start the LiveAI streaming system"""
        if self.is_running:
            return
            
        self.is_running = True
        self.session = aiohttp.ClientSession()
        
        # Start default financial streams
        await self._start_default_streams()
        
        logger.info("✅ LiveAI Stream Processor started")
        
    async def stop(self):
        """Stop the streaming system"""
        self.is_running = False
        
        # Cancel all stream tasks
        for task in self.stream_tasks:
            task.cancel()
        await asyncio.gather(*self.stream_tasks, return_exceptions=True)
        self.stream_tasks.clear()
        
        if self.session:
            await self.session.close()
            
        logger.info("⏹️ LiveAI Stream Processor stopped")
        
    async def _start_default_streams(self):
        """Start default financial data streams"""
        # Market data stream
        market_stream = FinancialStream(
            name="market_data",
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"],
            refresh_rate=5.0,  # seconds
            data_source="yahoo_finance"
        )
        
        # News stream
        news_stream = FinancialStream(
            name="financial_news", 
            symbols=[],
            refresh_rate=30.0,  # seconds
            data_source="rss_feeds"
        )
        
        await self.add_stream(market_stream)
        await self.add_stream(news_stream)
        
    async def add_stream(self, stream: FinancialStream):
        """Add a new data stream"""
        if stream.name in self.streams:
            return
            
        self.streams[stream.name] = stream
        
        # Start streaming task
        if stream.data_source == "yahoo_finance":
            task = asyncio.create_task(self._yahoo_finance_stream(stream))
        elif stream.data_source == "rss_feeds":
            task = asyncio.create_task(self._news_rss_stream(stream))
        else:
            logger.warning(f"Unknown data source: {stream.data_source}")
            return
            
        self.stream_tasks.append(task)
        logger.info(f"📡 Started stream: {stream.name}")
        
    async def _yahoo_finance_stream(self, stream: FinancialStream):
        """Yahoo Finance data streaming"""
        while self.is_running and stream.active:
            try:
                for symbol in stream.symbols:
                    # Fetch data in executor to avoid blocking
                    data = await asyncio.get_event_loop().run_in_executor(
                        self.executor, self._fetch_yahoo_data, symbol
                    )
                    
                    if data:
                        # Create data point
                        data_point = LiveDataPoint(
                            id=f"market_{symbol}_{int(datetime.now().timestamp())}",
                            timestamp=datetime.now(),
                            source=f"YahooFinance_{symbol}",
                            data_type="market_data",
                            content=data,
                            metadata={"symbol": symbol, "stream": stream.name}
                        )
                        
                        # Generate embedding
                        text_content = f"{symbol} price {data.get('price', 0)} change {data.get('change', 0)}%"
                        data_point.embedding = await self.rag_processor.generate_embedding(text_content)
                        
                        # Add to vector store
                        self.vector_store.add_vector(data_point)
                        
                        logger.debug(f"📊 Market data updated: {symbol} = ${data.get('price', 0)}")
                        
                await asyncio.sleep(stream.refresh_rate)
                
            except Exception as e:
                logger.error(f"Error in Yahoo Finance stream: {e}")
                await asyncio.sleep(5.0)
                
    def _fetch_yahoo_data(self, symbol: str) -> Optional[Dict]:
        """Fetch Yahoo Finance data synchronously"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
                
            latest = hist.iloc[-1]
            
            return {
                "symbol": symbol,
                "price": float(latest['Close']),
                "volume": int(latest['Volume']),
                "change": float((latest['Close'] - hist.iloc[0]['Open']) if len(hist) > 1 else 0),
                "change_percent": float(((latest['Close'] - hist.iloc[0]['Open']) / hist.iloc[0]['Open'] * 100) if len(hist) > 1 and hist.iloc[0]['Open'] > 0 else 0),
                "high": float(latest['High']),
                "low": float(latest['Low']),
                "timestamp": latest.name.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
            
    async def _news_rss_stream(self, stream: FinancialStream):
        """Financial news RSS streaming"""
        rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.sec.gov/news/pressreleases.xml",
        ]
        
        seen_articles = set()
        
        while self.is_running and stream.active:
            try:
                for feed_url in rss_feeds:
                    # Fetch RSS in executor
                    articles = await asyncio.get_event_loop().run_in_executor(
                        self.executor, self._fetch_rss_news, feed_url
                    )
                    
                    for article in articles:
                        article_id = hashlib.md5(article['title'].encode()).hexdigest()
                        
                        if article_id not in seen_articles:
                            seen_articles.add(article_id)
                            
                            # Create data point
                            data_point = LiveDataPoint(
                                id=f"news_{article_id}",
                                timestamp=datetime.now(),
                                source=f"RSS_{feed_url.split('/')[2]}",
                                data_type="news",
                                content=article,
                                metadata={"feed_url": feed_url}
                            )
                            
                            # Generate embedding
                            text_content = f"{article['title']} {article.get('summary', '')}"
                            data_point.embedding = await self.rag_processor.generate_embedding(text_content)
                            
                            # Add to vector store
                            self.vector_store.add_vector(data_point)
                            
                            logger.debug(f"📰 News added: {article['title'][:50]}...")
                            
                await asyncio.sleep(stream.refresh_rate)
                
            except Exception as e:
                logger.error(f"Error in news stream: {e}")
                await asyncio.sleep(10.0)
                
    def _fetch_rss_news(self, feed_url: str) -> List[Dict]:
        """Fetch RSS news synchronously"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:10]:  # Limit to recent articles
                article = {
                    "title": entry.get('title', ''),
                    "link": entry.get('link', ''),
                    "summary": entry.get('summary', ''),
                    "published": entry.get('published', ''),
                    "source": feed.feed.get('title', 'Unknown')
                }
                articles.append(article)
                
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS {feed_url}: {e}")
            return []
            
    async def query(self, question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the LiveAI system"""
        return await self.rag_processor.process_query(question, filters)
        
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            "active_streams": len([s for s in self.streams.values() if s.active]),
            "total_vectors": len(self.vector_store.vectors),
            "vector_store_size": len(self.vector_store.vectors),
            "last_updated": self.vector_store.last_updated.isoformat(),
            "is_running": self.is_running,
            "streams": {name: {"active": stream.active, "symbols": len(stream.symbols)} 
                      for name, stream in self.streams.items()}
        }

# Global LiveAI instance
liveai_engine = None

async def get_liveai_engine() -> LiveAIStreamProcessor:
    """Get or create the global LiveAI engine"""
    global liveai_engine
    if liveai_engine is None:
        liveai_engine = LiveAIStreamProcessor()
        await liveai_engine.start()
    return liveai_engine

async def shutdown_liveai_engine():
    """Shutdown the global LiveAI engine"""
    global liveai_engine
    if liveai_engine:
        await liveai_engine.stop()
        liveai_engine = None

# Example usage for hackathon demo
async def hackathon_demo():
    """Demo function showing LiveAI capabilities"""
    engine = await get_liveai_engine()
    
    # Wait for some data to accumulate
    await asyncio.sleep(10)
    
    # Query examples
    queries = [
        "What is the current price of AAPL?",
        "Show me the latest financial news",
        "Which stocks are performing well today?",
        "What are the market trends right now?"
    ]
    
    print("🚀 LiveAI Hackathon Demo - Real-Time Financial Intelligence")
    print("=" * 60)
    
    for query in queries:
        result = await engine.query(query, {"max_age_minutes": 5})
        print(f"\n🔍 Query: {query}")
        print(f"📊 Response: {result['generated_response']}")
        print(f"📈 Relevant docs: {result['relevant_docs']}")
        
    stats = engine.get_stream_stats()
    print(f"\n📊 System Stats:")
    print(f"   • Active streams: {stats['active_streams']}")
    print(f"   • Total vectors: {stats['total_vectors']}")  
    print(f"   • Last updated: {stats['last_updated']}")
    
if __name__ == "__main__":
    asyncio.run(hackathon_demo())
