# Import Pathway with fallback to mock for demonstration
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
    print("✅ Real Pathway package loaded")
except ImportError:
    try:
        from .mock_pathway import pw
        PATHWAY_AVAILABLE = False
        print("⚠️  Using Mock Pathway for demonstration (real Pathway not available on Windows)")
    except ImportError:
        print("❌ Neither real nor mock Pathway available")
        pw = None
        PATHWAY_AVAILABLE = False

import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
import json
import logging
from dataclasses import dataclass
import feedparser
import os
from concurrent.futures import ThreadPoolExecutor
import time
import threading

logger = logging.getLogger(__name__)

@dataclass
class RealTimeMarketData:
    """Real-time market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    change: float
    change_percent: float
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

@dataclass
class FinancialNewsItem:
    """Financial news item structure"""
    headline: str
    summary: str
    timestamp: datetime
    source: str
    sentiment: str
    impact_score: float
    related_symbols: List[str]
    url: Optional[str] = None

@dataclass
class SECFiling:
    """SEC filing structure"""
    company: str
    filing_type: str
    date: datetime
    url: str
    size_kb: int
    summary: Optional[str] = None

@dataclass
class EconomicIndicator:
    """Economic indicator structure"""
    name: str
    value: float
    timestamp: datetime
    previous_value: Optional[float] = None
    change: Optional[float] = None
    unit: str = ""

class RealTimeFinancialDataStreams:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self.update_threads = []
        
        # Data storage for streaming
        self.market_data_cache = {}
        self.news_data_cache = []
        self.sec_filings_cache = []
        self.economic_indicators_cache = {}
        
        # Pathway tables
        self.tables = {}
        
        # Major stock symbols for tracking
        self.tracked_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'BRK-B',
            'AVGO', 'LLY', 'V', 'JPM', 'UNH', 'XOM', 'ORCL', 'MA', 'HD', 'PG',
            'COST', 'JNJ'
        ]
        
        # Financial news RSS feeds (REAL sources)
        self.news_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://rss.cnn.com/rss/money_latest.rss',
            'https://feeds.feedburner.com/cnbc-technology',
        ]
        
        self.logger.info(" RealTimeFinancialDataStreams initialized")
        self.logger.info(f" Tracking {len(self.tracked_symbols)} symbols")
        self.logger.info(f" Monitoring {len(self.news_feeds)} news sources")
        self.logger.info(f" Pathway Available: {PATHWAY_AVAILABLE}")
    
    def create_market_data_stream(self):
        """Create Pathway table for real-time market data"""
        self.logger.info(" Creating real-time market data stream...")
        
        if pw is None:
            return None
        
        try:
            # Create a mock table that will be updated with real data
            market_table = pw.Table(source_type="market_data")
            
            # Start background thread for real-time updates
            if not hasattr(self, '_market_update_thread'):
                self._market_update_thread = threading.Thread(
                    target=self._update_market_data_loop, 
                    daemon=True
                )
                self._market_update_thread.start()
                self.logger.info(" Market data update thread started")
            
            self.tables['market_data'] = market_table
            return market_table
            
        except Exception as e:
            self.logger.error(f"❌ Error creating market data stream: {e}")
            return None
    
    def create_financial_news_stream(self):
        """Create Pathway table for real-time financial news"""
        self.logger.info(" Creating real-time financial news stream...")
        
        if pw is None:
            return None
        
        try:
            news_table = pw.Table(source_type="financial_news")
            
            # Start background thread for news updates
            if not hasattr(self, '_news_update_thread'):
                self._news_update_thread = threading.Thread(
                    target=self._update_news_data_loop,
                    daemon=True
                )
                self._news_update_thread.start()
                self.logger.info(" News data update thread started")
            
            self.tables['financial_news'] = news_table
            return news_table
            
        except Exception as e:
            self.logger.error(f"❌ Error creating news stream: {e}")
            return None
    
    def create_sec_filings_stream(self):
        """Create Pathway table for SEC filings monitoring"""
        self.logger.info(" Creating SEC filings monitoring stream...")
        
        if pw is None:
            return None
        
        try:
            sec_table = pw.Table(source_type="sec_filings")
            
            # Start background monitoring for SEC filings
            if not hasattr(self, '_sec_update_thread'):
                self._sec_update_thread = threading.Thread(
                    target=self._update_sec_filings_loop,
                    daemon=True
                )
                self._sec_update_thread.start()
                self.logger.info(" SEC filings monitoring thread started")
            
            self.tables['sec_filings'] = sec_table
            return sec_table
            
        except Exception as e:
            self.logger.error(f"❌ Error creating SEC filings stream: {e}")
            return None
    
    def create_economic_indicators_stream(self):
        """Create Pathway table for economic indicators"""
        self.logger.info("📈 Creating economic indicators stream...")
        
        if pw is None:
            return None
        
        try:
            econ_table = pw.Table(source_type="economic_indicators")
            
            # Start background monitoring for economic data
            if not hasattr(self, '_econ_update_thread'):
                self._econ_update_thread = threading.Thread(
                    target=self._update_economic_indicators_loop,
                    daemon=True
                )
                self._econ_update_thread.start()
                self.logger.info(" Economic indicators monitoring thread started")
            
            self.tables['economic_indicators'] = econ_table
            return econ_table
            
        except Exception as e:
            self.logger.error(f"❌ Error creating economic indicators stream: {e}")
            return None
    
    def _update_market_data_loop(self):
        """Background loop to fetch real market data"""
        self.logger.info(" Starting market data update loop...")
        
        while True:
            try:
                # Fetch real market data using yfinance
                market_data = self._fetch_real_market_data()
                
                # Update the Pathway table
                if 'market_data' in self.tables and market_data:
                    self.tables['market_data'].update_data(market_data)
                    self.logger.info(f" Updated market data: {len(market_data)} symbols")
                
                # Wait 30 seconds before next update
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"❌ Error in market data loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_news_data_loop(self):
        """Background loop to fetch real financial news"""
        self.logger.info(" Starting news data update loop...")
        
        while True:
            try:
                # Fetch real news data from RSS feeds
                news_data = self._fetch_real_news_data()
                
                # Update the Pathway table
                if 'financial_news' in self.tables and news_data:
                    self.tables['financial_news'].update_data(news_data)
                    self.logger.info(f" Updated news data: {len(news_data)} articles")
                
                # Wait 3 minutes before next update
                time.sleep(180)
                
            except Exception as e:
                self.logger.error(f"❌ Error in news data loop: {e}")
                time.sleep(300)  # Wait longer on error
    
    def _update_sec_filings_loop(self):
        """Background loop to monitor SEC filings"""
        self.logger.info(" Starting SEC filings monitoring loop...")
        
        while True:
            try:
                # Check for new SEC filings
                filings_data = self._fetch_sec_filings_data()
                
                # Update the Pathway table
                if 'sec_filings' in self.tables and filings_data:
                    self.tables['sec_filings'].update_data(filings_data)
                    self.logger.info(f" Updated SEC filings: {len(filings_data)} new filings")
                
                # Wait 10 minutes before next check
                time.sleep(600)
                
            except Exception as e:
                self.logger.error(f"❌ Error in SEC filings loop: {e}")
                time.sleep(900)  # Wait longer on error
    
    def _update_economic_indicators_loop(self):
        """Background loop to fetch economic indicators"""
        self.logger.info("📈 Starting economic indicators update loop...")
        
        while True:
            try:
                # Fetch economic indicators data
                econ_data = self._fetch_economic_indicators()
                
                # Update the Pathway table
                if 'economic_indicators' in self.tables and econ_data:
                    self.tables['economic_indicators'].update_data(econ_data)
                    self.logger.info(f"📈 Updated economic indicators: {len(econ_data)} indicators")
                
                # Wait 1 hour before next update
                time.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"❌ Error in economic indicators loop: {e}")
                time.sleep(3600)  # Wait longer on error
    
    def _fetch_real_market_data(self) -> List[Dict]:
        """Fetch REAL market data from Yahoo Finance - NO MOCK DATA"""
        try:
            market_data = []
            
            # Use yfinance to get real market data
            for symbol in self.tracked_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        
                        # Calculate change from previous day
                        prev_close = info.get('previousClose', latest['Close'])
                        change = latest['Close'] - prev_close
                        change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        market_data.append({
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'price': round(float(latest['Close']), 2),
                            'volume': int(latest['Volume']),
                            'change': round(float(change), 2),
                            'change_percent': round(float(change_percent), 2),
                            'market_cap': info.get('marketCap'),
                            'pe_ratio': info.get('trailingPE'),
                            'dividend_yield': info.get('dividendYield')
                        })
                        
                except Exception as ticker_error:
                    self.logger.warning(f"⚠️  Error fetching data for {symbol}: {ticker_error}")
                    continue
            
            self.logger.info(f" Fetched real market data for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching market data: {e}")
            return []
    
    def _fetch_real_news_data(self) -> List[Dict]:
        """Fetch REAL financial news from RSS feeds - NO MOCK DATA"""
        try:
            news_data = []
            
            for feed_url in self.news_feeds:
                try:
                    # Parse RSS feed
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:5]:  # Get latest 5 articles
                        # Simple sentiment analysis (could be enhanced)
                        sentiment = self._analyze_sentiment(entry.title + " " + entry.get('summary', ''))
                        
                        # Extract related symbols (basic keyword matching)
                        related_symbols = self._extract_symbols(entry.title + " " + entry.get('summary', ''))
                        
                        news_data.append({
                            'headline': entry.title,
                            'summary': entry.get('summary', '')[:500],  # Truncate to 500 chars
                            'timestamp': datetime.now().isoformat(),
                            'source': feed_url.split('/')[2],  # Extract domain
                            'sentiment': sentiment,
                            'impact_score': self._calculate_impact_score(sentiment, related_symbols),
                            'related_symbols': related_symbols,
                            'url': entry.get('link', '')
                        })
                        
                except Exception as feed_error:
                    self.logger.warning(f"⚠️  Error parsing feed {feed_url}: {feed_error}")
                    continue
            
            self.logger.info(f" Fetched {len(news_data)} real news articles")
            return news_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching news data: {e}")
            return []
    
    def _fetch_sec_filings_data(self) -> List[Dict]:
        """Monitor for new SEC filings - Simulated for demo"""
        try:
            # In a real implementation, this would connect to SEC EDGAR database
            # For hackathon demo, we'll simulate realistic filings
            import random
            
            filings_data = []
            companies = ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Tesla Inc.', 'Amazon.com Inc.']
            filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A', 'S-1']
            
            # Simulate 1-2 new filings per check
            num_filings = random.randint(0, 2)
            
            for _ in range(num_filings):
                company = random.choice(companies)
                filing_type = random.choice(filing_types)
                
                filings_data.append({
                    'company': company,
                    'filing_type': filing_type,
                    'date': datetime.now().isoformat(),
                    'url': f"https://www.sec.gov/edgar/browse/?CIK={random.randint(1000, 9999)}",
                    'size_kb': random.randint(100, 5000),
                    'summary': f"{company} filed {filing_type} - {random.choice(['quarterly earnings', 'annual report', 'material agreement', 'proxy statement'])}"
                })
            
            if filings_data:
                self.logger.info(f" Found {len(filings_data)} new SEC filings")
            
            return filings_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching SEC filings: {e}")
            return []
    
    def _fetch_economic_indicators(self) -> List[Dict]:
        """Fetch economic indicators - Simulated for demo"""
        try:
            import random
            
            indicators_data = []
            
            # Common economic indicators
            indicators = {
                'VIX': {'current': random.uniform(12, 35), 'unit': 'index'},
                'Federal_Funds_Rate': {'current': random.uniform(4.0, 6.0), 'unit': '%'},
                '10Y_Treasury_Yield': {'current': random.uniform(3.5, 5.0), 'unit': '%'},
                'USD_Index': {'current': random.uniform(100, 108), 'unit': 'index'},
                'Gold_Price': {'current': random.uniform(1800, 2200), 'unit': 'USD/oz'},
                'Oil_Price_WTI': {'current': random.uniform(60, 90), 'unit': 'USD/barrel'}
            }
            
            for name, data in indicators.items():
                # Simulate slight variations from previous values
                prev_value = data['current'] * random.uniform(0.98, 1.02)
                change = data['current'] - prev_value
                
                indicators_data.append({
                    'name': name.replace('_', ' '),
                    'value': round(data['current'], 2),
                    'timestamp': datetime.now().isoformat(),
                    'previous_value': round(prev_value, 2),
                    'change': round(change, 2),
                    'unit': data['unit']
                })
            
            self.logger.info(f" Fetched {len(indicators_data)} economic indicators")
            return indicators_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching economic indicators: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis for news"""
        # Simple keyword-based sentiment analysis
        positive_words = ['bullish', 'gains', 'up', 'rise', 'growth', 'positive', 'beat', 'strong', 'rally', 'surge']
        negative_words = ['bearish', 'losses', 'down', 'fall', 'decline', 'negative', 'miss', 'weak', 'crash', 'drop']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []
        text_upper = text.upper()
        
        for symbol in self.tracked_symbols:
            if symbol in text_upper or symbol.replace('-', '.') in text_upper:
                symbols.append(symbol)
        
        return symbols
    
    def _calculate_impact_score(self, sentiment: str, related_symbols: List[str]) -> float:
        """Calculate news impact score"""
        base_score = 0.5
        
        if sentiment == 'positive':
            base_score = 0.7
        elif sentiment == 'negative':
            base_score = 0.3
        
        # Boost score if multiple symbols are mentioned
        symbol_boost = min(len(related_symbols) * 0.1, 0.3)
        
        return min(base_score + symbol_boost, 1.0)
    
    def get_all_tables(self) -> Dict[str, Any]:
        """Get all Pathway tables"""
        return self.tables
    
    def stop_all_streams(self):
        """Stop all data streams"""
        self.is_running = False
        self.logger.info(" Stopping all real-time data streams")
        
        # Stop all tables
        for table_name, table in self.tables.items():
            if hasattr(table, 'stop_streaming'):
                table.stop_streaming()

# Create global instance
real_time_streams = RealTimeFinancialDataStreams()
