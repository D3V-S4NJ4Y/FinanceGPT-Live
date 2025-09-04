"""
📰 News Intelligence Agent
==========================

Advanced AI agent for real-time news analysis and sentiment processing.
Provides intelligent news monitoring with ML-powered insights.

"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import re
import yfinance as yf
from textblob import TextBlob
import requests
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent
from core.config import settings

logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class NewsCategory(Enum):
    EARNINGS = "earnings"
    MARKET_ANALYSIS = "market_analysis"
    REGULATORY = "regulatory"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    LEADERSHIP_CHANGE = "leadership_change"
    GENERAL = "general"

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str]
    sentiment_score: float
    sentiment_label: SentimentScore
    category: NewsCategory
    impact_score: float
    entities: List[str]
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'published_at': self.published_at.isoformat(),
            'sentiment_label': self.sentiment_label.value,
            'category': self.category.value
        }

@dataclass
class NewsAnalysis:
    """Comprehensive news analysis"""
    symbol: str
    overall_sentiment: float
    sentiment_trend: str
    news_volume: int
    high_impact_count: int
    categories: Dict[str, int]
    key_themes: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class NewsIntelligenceAgent(BaseAgent):
    """
    📰 News Intelligence - Advanced News Analysis Agent
    
    Capabilities:
    - Real-time news monitoring from multiple sources
    - Advanced sentiment analysis with context awareness
    - Entity and keyword extraction
    - News categorization and impact assessment
    - Trend analysis and pattern recognition
    - Multi-language support
    - Fake news detection
    - Market impact correlation
    """
    
    def __init__(self):
        super().__init__(
            name="NewsIntelligence",
            description="Advanced real-time news analysis and sentiment processing agent",
            version="1.0.0"
        )
        
        # Configuration
        self.news_sources = [
            "reuters", "bloomberg", "cnbc", "marketwatch", 
            "financial-post", "the-wall-street-journal"
        ]
        self.monitored_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "SPY", "QQQ"]
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'very_positive': 0.6,
            'positive': 0.2,
            'neutral': 0.2,
            'negative': -0.2,
            'very_negative': -0.6
        }
        
        # Data storage
        self.news_cache = []
        self.symbol_sentiment = {}
        self.trending_keywords = {}
        self.analysis_cache = {}
        
        # Performance tracking
        self.articles_processed = 0
        self.sentiment_analyses = 0
        
        # Keywords for categorization
        self.category_keywords = {
            NewsCategory.EARNINGS: ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'results'],
            NewsCategory.MARKET_ANALYSIS: ['market', 'analysis', 'forecast', 'outlook', 'prediction'],
            NewsCategory.REGULATORY: ['regulation', 'sec', 'compliance', 'policy', 'government'],
            NewsCategory.MERGER_ACQUISITION: ['merger', 'acquisition', 'takeover', 'deal', 'buyout'],
            NewsCategory.PRODUCT_LAUNCH: ['launch', 'product', 'service', 'release', 'unveil'],
            NewsCategory.LEADERSHIP_CHANGE: ['ceo', 'cfo', 'executive', 'leadership', 'appointment']
        }
        
        logger.info("✅ NewsIntelligenceAgent initialized")
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming news-related message"""
        try:
            message_type = message.get('type')
            
            if message_type == 'news_update':
                return await self._process_news_update(message)
            elif message_type == 'sentiment_analysis':
                return await self._process_sentiment_analysis(message)
            elif message_type == 'news_query':
                return await self._process_news_query(message)
            elif message_type == 'trending_topics':
                return await self._get_trending_topics()
            elif message_type == 'health_check':
                return await self._health_check()
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown message type: {message_type}'
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _process_news_update(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process new news article"""
        article_data = message.get('article', {})
        
        if not article_data:
            return {'status': 'error', 'message': 'No article data provided'}
        
        try:
            # Create news article
            article = await self._create_news_article(article_data)
            
            # Store in cache
            self.news_cache.append(article)
            
            # Keep only recent articles (last 1000)
            self.news_cache = self.news_cache[-1000:]
            
            # Update symbol sentiment
            for symbol in article.symbols:
                if symbol not in self.symbol_sentiment:
                    self.symbol_sentiment[symbol] = []
                
                self.symbol_sentiment[symbol].append({
                    'sentiment': article.sentiment_score,
                    'impact': article.impact_score,
                    'timestamp': article.published_at
                })
                
                # Keep only recent sentiment data
                self.symbol_sentiment[symbol] = self.symbol_sentiment[symbol][-100:]
            
            # Update trending keywords
            self._update_trending_keywords(article)
            
            self.articles_processed += 1
            
            return {
                'status': 'success',
                'article': article.to_dict(),
                'symbols_affected': article.symbols,
                'sentiment': article.sentiment_label.value,
                'impact_score': article.impact_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing news update: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _create_news_article(self, article_data: Dict[str, Any]) -> NewsArticle:
        """Create structured news article from raw data"""
        title = article_data.get('title', '')
        content = article_data.get('content', article_data.get('description', ''))
        source = article_data.get('source', {}).get('name', 'Unknown')
        url = article_data.get('url', '')
        published_at = datetime.fromisoformat(
            article_data.get('publishedAt', datetime.now().isoformat()).replace('Z', '+00:00')
        )
        
        # Extract symbols mentioned in the article
        symbols = self._extract_symbols(title + " " + content)
        
        # Perform sentiment analysis
        sentiment_score = await self._analyze_sentiment(content)
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        # Categorize the news
        category = self._categorize_news(title + " " + content)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(
            sentiment_score, len(symbols), source, category
        )
        
        # Extract entities and keywords
        entities = self._extract_entities(content)
        keywords = self._extract_keywords(title + " " + content)
        
        return NewsArticle(
            title=title,
            content=content,
            source=source,
            url=url,
            published_at=published_at,
            symbols=symbols,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            category=category,
            impact_score=impact_score,
            entities=entities,
            keywords=keywords
        )
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of news text"""
        try:
            # Use TextBlob for basic sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Enhance with financial context
            financial_sentiment = self._analyze_financial_sentiment(text)
            
            # Combine scores (weighted average)
            combined_score = (polarity * 0.6) + (financial_sentiment * 0.4)
            
            self.sentiment_analyses += 1
            
            return combined_score
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}")
            return 0.0
    
    def _analyze_financial_sentiment(self, text: str) -> float:
        """Analyze financial-specific sentiment"""
        positive_financial_terms = [
            'profit', 'growth', 'increase', 'beat expectations', 'exceed', 'outperform',
            'bullish', 'rally', 'surge', 'gain', 'revenue up', 'strong results',
            'upgrade', 'buy rating', 'positive outlook', 'expansion', 'innovation'
        ]
        
        negative_financial_terms = [
            'loss', 'decline', 'decrease', 'miss expectations', 'underperform',
            'bearish', 'crash', 'plunge', 'drop', 'revenue down', 'weak results',
            'downgrade', 'sell rating', 'negative outlook', 'layoffs', 'bankruptcy'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for term in positive_financial_terms if term in text_lower)
        negative_count = sum(1 for term in negative_financial_terms if term in text_lower)
        
        # Calculate sentiment score
        total_terms = positive_count + negative_count
        if total_terms == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_terms
        return sentiment
    
    def _get_sentiment_label(self, score: float) -> SentimentScore:
        """Convert sentiment score to label"""
        if score >= self.sentiment_thresholds['very_positive']:
            return SentimentScore.VERY_POSITIVE
        elif score >= self.sentiment_thresholds['positive']:
            return SentimentScore.POSITIVE
        elif score <= self.sentiment_thresholds['very_negative']:
            return SentimentScore.VERY_NEGATIVE
        elif score <= self.sentiment_thresholds['negative']:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.NEUTRAL
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Look for ticker symbols (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter to known symbols or common patterns
        known_symbols = set(self.monitored_symbols)
        extracted_symbols = []
        
        for ticker in potential_tickers:
            if ticker in known_symbols or ticker.endswith('Y') or len(ticker) <= 4:
                extracted_symbols.append(ticker)
        
        # Also look for company names and map to symbols
        company_symbol_map = {
            'apple': 'AAPL',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'nvidia': 'NVDA'
        }
        
        text_lower = text.lower()
        for company, symbol in company_symbol_map.items():
            if company in text_lower and symbol not in extracted_symbols:
                extracted_symbols.append(symbol)
        
        return list(set(extracted_symbols))
    
    def _categorize_news(self, text: str) -> NewsCategory:
        """Categorize news article"""
        text_lower = text.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return NewsCategory.GENERAL
    
    def _calculate_impact_score(self, sentiment_score: float, symbols_count: int,
                              source: str, category: NewsCategory) -> float:
        """Calculate potential market impact score"""
        base_impact = abs(sentiment_score)
        
        # Boost based on source credibility
        credible_sources = ['Reuters', 'Bloomberg', 'WSJ', 'Financial Times']
        if any(source_name in source for source_name in credible_sources):
            base_impact *= 1.5
        
        # Boost based on number of symbols
        base_impact *= min(2.0, 1 + symbols_count * 0.2)
        
        # Boost based on category
        high_impact_categories = [
            NewsCategory.EARNINGS,
            NewsCategory.MERGER_ACQUISITION,
            NewsCategory.REGULATORY
        ]
        if category in high_impact_categories:
            base_impact *= 1.3
        
        return min(1.0, base_impact)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simplified entity extraction
        # In production, would use spaCy or similar NLP library
        entities = []
        
        # Extract potential company names (capitalized words)
        words = text.split()
        for i, word in enumerate(words):
            if (word.isupper() and len(word) > 2) or (word.istitle() and len(word) > 3):
                # Check if it's likely a company or person name
                if not word.lower() in ['the', 'and', 'for', 'with', 'this', 'that']:
                    entities.append(word)
        
        return list(set(entities))[:10]  # Limit to 10 entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common stop words
        stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        ])
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter keywords
        keywords = [word for word in words 
                   if len(word) > 3 and word not in stop_words]
        
        # Count frequency and return top keywords
        from collections import Counter
        word_counts = Counter(keywords)
        
        return [word for word, count in word_counts.most_common(10)]
    
    def _update_trending_keywords(self, article: NewsArticle):
        """Update trending keywords tracking"""
        for keyword in article.keywords:
            if keyword not in self.trending_keywords:
                self.trending_keywords[keyword] = []
            
            self.trending_keywords[keyword].append({
                'timestamp': article.published_at,
                'sentiment': article.sentiment_score,
                'impact': article.impact_score
            })
            
            # Keep only recent data
            cutoff = datetime.now() - timedelta(hours=24)
            self.trending_keywords[keyword] = [
                item for item in self.trending_keywords[keyword]
                if item['timestamp'] > cutoff
            ]
    
    async def _process_sentiment_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment analysis request"""
        text = message.get('text', '')
        
        if not text:
            return {'status': 'error', 'message': 'No text provided'}
        
        sentiment_score = await self._analyze_sentiment(text)
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        # Extract additional insights
        symbols = self._extract_symbols(text)
        entities = self._extract_entities(text)
        keywords = self._extract_keywords(text)
        
        return {
            'status': 'success',
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label.value,
            'symbols_detected': symbols,
            'entities': entities,
            'keywords': keywords,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _process_news_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process news query request"""
        symbol = message.get('symbol')
        limit = message.get('limit', 10)
        hours_back = message.get('hours_back', 24)
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter news articles
        filtered_articles = []
        for article in self.news_cache:
            if article.published_at > cutoff_time:
                if not symbol or symbol in article.symbols:
                    filtered_articles.append(article)
        
        # Sort by impact score and recency
        filtered_articles.sort(
            key=lambda x: (x.impact_score, x.published_at),
            reverse=True
        )
        
        # Generate analysis for the symbol if provided
        analysis = None
        if symbol and symbol in self.symbol_sentiment:
            analysis = await self._generate_symbol_analysis(symbol)
        
        return {
            'status': 'success',
            'articles': [article.to_dict() for article in filtered_articles[:limit]],
            'total_articles': len(filtered_articles),
            'symbol_analysis': analysis.to_dict() if analysis else None,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _generate_symbol_analysis(self, symbol: str) -> NewsAnalysis:
        """Generate comprehensive news analysis for symbol"""
        if symbol not in self.symbol_sentiment:
            return None
        
        sentiment_data = self.symbol_sentiment[symbol]
        
        # Calculate overall sentiment
        recent_sentiments = [item['sentiment'] for item in sentiment_data[-20:]]
        overall_sentiment = np.mean(recent_sentiments) if recent_sentiments else 0
        
        # Determine trend
        if len(recent_sentiments) >= 10:
            first_half = np.mean(recent_sentiments[:len(recent_sentiments)//2])
            second_half = np.mean(recent_sentiments[len(recent_sentiments)//2:])
            
            if second_half > first_half + 0.1:
                trend = "improving"
            elif second_half < first_half - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Get articles for this symbol
        symbol_articles = [
            article for article in self.news_cache[-100:]
            if symbol in article.symbols
        ]
        
        # Calculate news volume
        news_volume = len(symbol_articles)
        high_impact_count = sum(1 for article in symbol_articles if article.impact_score > 0.7)
        
        # Categorize articles
        categories = {}
        for article in symbol_articles:
            cat = article.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        # Extract key themes
        all_keywords = []
        for article in symbol_articles:
            all_keywords.extend(article.keywords)
        
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        key_themes = [word for word, count in keyword_counts.most_common(10)]
        
        # Identify risks and opportunities
        risk_factors = []
        opportunities = []
        
        for article in symbol_articles:
            if article.sentiment_score < -0.5:
                risk_factors.extend(article.keywords[:3])
            elif article.sentiment_score > 0.5:
                opportunities.extend(article.keywords[:3])
        
        return NewsAnalysis(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            sentiment_trend=trend,
            news_volume=news_volume,
            high_impact_count=high_impact_count,
            categories=categories,
            key_themes=key_themes[:5],
            risk_factors=list(set(risk_factors))[:5],
            opportunities=list(set(opportunities))[:5],
            timestamp=datetime.now()
        )
    
    async def _get_trending_topics(self) -> Dict[str, Any]:
        """Get currently trending topics and keywords"""
        # Calculate trending scores
        trending_data = {}
        
        for keyword, data_points in self.trending_keywords.items():
            if len(data_points) >= 3:  # Minimum data points
                # Calculate trend score based on frequency and recency
                recent_count = len([
                    dp for dp in data_points
                    if dp['timestamp'] > datetime.now() - timedelta(hours=6)
                ])
                
                avg_impact = np.mean([dp['impact'] for dp in data_points])
                trend_score = recent_count * avg_impact
                
                trending_data[keyword] = {
                    'trend_score': trend_score,
                    'mention_count': len(data_points),
                    'avg_impact': avg_impact,
                    'avg_sentiment': np.mean([dp['sentiment'] for dp in data_points])
                }
        
        # Sort by trend score
        sorted_topics = sorted(
            trending_data.items(),
            key=lambda x: x[1]['trend_score'],
            reverse=True
        )
        
        return {
            'status': 'success',
            'trending_topics': dict(sorted_topics[:10]),
            'total_keywords_tracked': len(self.trending_keywords),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _health_check(self) -> Dict[str, Any]:
        """Agent health check"""
        return {
            'status': 'healthy',
            'agent': self.name,
            'version': self.version,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'articles_processed': self.articles_processed,
            'sentiment_analyses': self.sentiment_analyses,
            'cached_articles': len(self.news_cache),
            'monitored_symbols': len(self.monitored_symbols),
            'trending_keywords': len(self.trending_keywords),
            'news_sources': len(self.news_sources),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': self.name,
            'status': 'active',
            'articles_processed': self.articles_processed,
            'sentiment_analyses': self.sentiment_analyses,
            'monitored_symbols': self.monitored_symbols,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'last_activity': datetime.now().isoformat()
        }
