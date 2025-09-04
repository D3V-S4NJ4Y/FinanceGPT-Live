import React, { useState, useEffect } from 'react';
import { 
  Globe, 
  Newspaper,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Clock,
  ExternalLink,
  Filter,
  Search,
  Eye,
  Loader
} from 'lucide-react';

interface NewsItem {
  id: string;
  headline: string;
  summary: string;
  source: string;
  timestamp: Date;
  sentiment: 'positive' | 'negative' | 'neutral';
  impact: 'high' | 'medium' | 'low';
  symbols: string[];
  category: string;
  url: string;
  imageUrl?: string;
  sentimentScore?: number;
  impactScore?: number;
}

interface SentimentAnalysis {
  symbol: string;
  averageSentiment: number;
  sentimentLabel: string;
  articleCount: number;
  distribution: {
    positive: number;
    negative: number;
    neutral: number;
  };
  confidence: number;
}

interface MarketEvent {
  type: 'earnings' | 'economic' | 'announcement' | 'alert';
  title: string;
  description: string;
  time: Date;
  severity: 'high' | 'medium' | 'low';
  symbols?: string[];
}

export default function RealTimeNewsCenter() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [events, setEvents] = useState<MarketEvent[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [sentimentAnalysis, setSentimentAnalysis] = useState<SentimentAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const categories = ['all', 'markets', 'earnings', 'economic', 'technology', 'policy'];
  const sentiments = ['all', 'positive', 'negative', 'neutral'];

  useEffect(() => {
    fetchRealNews();
    fetchRealEvents();
    
    // Set up real-time updates every 2 minutes for news, 10 minutes for events
    const newsInterval = setInterval(fetchRealNews, 120000); // 2 minutes
    const eventsInterval = setInterval(fetchRealEvents, 600000); // 10 minutes

    return () => {
      clearInterval(newsInterval);
      clearInterval(eventsInterval);
    };
  }, []);
  
  // Refetch when filters change
  useEffect(() => {
    fetchRealNews();
  }, [selectedCategory, sentimentFilter]);

  const fetchRealNews = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        symbols: 'AAPL,GOOGL,MSFT,TSLA,AMZN,NVDA,META,NFLX,JPM,BAC,WFC,GS,XOM,CVX,JNJ,PFE,ORCL,CRM',
        limit: '25'
      });
      
      if (selectedCategory !== 'all') {
        params.append('category', selectedCategory);
      }
      
      if (sentimentFilter !== 'all') {
        params.append('sentiment', sentimentFilter);
      }
      
      console.log('Fetching news from:', `http://127.0.0.1:8001/api/news/latest?${params}`);
      
      const [newsResponse, sentimentResponse] = await Promise.all([
        fetch(`http://127.0.0.1:8001/api/news/latest?${params}`).catch(e => {
          console.error('News API error:', e);
          return null;
        }),
        fetch(`http://127.0.0.1:8001/api/news/sentiment-analysis?symbols=AAPL,GOOGL,MSFT,TSLA,AMZN,NVDA&hours=24`).catch(e => {
          console.error('Sentiment API error:', e);
          return null;
        })
      ]);
      
      if (newsResponse && newsResponse.ok) {
        const newsData = await newsResponse.json();
        console.log('News API response:', newsData);
        
        if (newsData?.data?.articles && Array.isArray(newsData.data.articles)) {
          const realNews: NewsItem[] = newsData.data.articles.map((article: any, index: number) => ({
            id: `news-${Date.now()}-${index}`,
            headline: article.title || 'Market Update',
            summary: article.summary || 'Real-time financial news update',
            timestamp: new Date(article.published),
            source: article.source || 'Financial News',
            category: article.category || 'markets',
            sentiment: article.sentiment?.label || 'neutral',
            impact: article.impact_score > 0.7 ? 'high' : article.impact_score > 0.4 ? 'medium' : 'low',
            symbols: article.symbols || [],
            url: article.url || '#',
            sentimentScore: article.sentiment?.score || 0,
            impactScore: article.impact_score || 0
          }));
          
          console.log('Processed news items:', realNews.length);
          setNews(realNews);
        } else {
          console.log('No articles in response, API error:', newsData?.error);
          setNews([]);
        }
      } else {
        console.error('News API request failed');
        setNews([]);
      }
      
      if (sentimentResponse && sentimentResponse.ok) {
        const sentimentData = await sentimentResponse.json();
        
        if (sentimentData?.success && sentimentData.data?.symbol_sentiment) {
          const analysis: SentimentAnalysis[] = Object.entries(sentimentData.data.symbol_sentiment).map(([symbol, data]: [string, any]) => ({
            symbol,
            averageSentiment: data.average_sentiment,
            sentimentLabel: data.sentiment_label,
            articleCount: data.article_count,
            distribution: data.distribution,
            confidence: data.confidence
          }));
          
          setSentimentAnalysis(analysis);
        }
      }
      
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch news data:', error);
      // Try direct API call as fallback
      try {
        const directResponse = await fetch('http://127.0.0.1:8001/api/news/latest?symbols=AAPL,GOOGL,MSFT&limit=10');
        if (directResponse.ok) {
          const directData = await directResponse.json();
          console.log('Direct API response:', directData);
          if (directData?.data?.articles) {
            const fallbackNews = directData.data.articles.map((article: any, index: number) => ({
              id: `fallback-${Date.now()}-${index}`,
              headline: article.title || 'Market Update',
              summary: article.summary || 'Financial news update',
              timestamp: new Date(article.published || Date.now()),
              source: article.source || 'Market News',
              category: article.category || 'markets',
              sentiment: article.sentiment?.label || 'neutral',
              impact: 'medium',
              symbols: article.symbols || [],
              url: article.url || '#',
              sentimentScore: article.sentiment?.score || 0,
              impactScore: article.impact_score || 0.5
            }));
            setNews(fallbackNews);
          }
        }
      } catch (fallbackError) {
        console.error('Fallback API also failed:', fallbackError);
        setNews([]);
      }
    }
    setLoading(false);
  };

  const fetchRealEvents = async () => {
    try {
      // Fetch real market events from comprehensive events API
      const response = await fetch('http://127.0.0.1:8001/api/news/market-events?symbols=AAPL,GOOGL,MSFT,TSLA,AMZN,NVDA&hours=168');
      
      if (response.ok) {
        const data = await response.json();
        
        if (data?.success && data.data?.events && Array.isArray(data.data.events)) {
          const realEvents: MarketEvent[] = data.data.events.map((event: any) => ({
            type: event.type as 'earnings' | 'economic' | 'announcement' | 'alert',
            title: event.title || 'Market Event',
            time: new Date(event.date),
            description: event.description || 'Market event',
            severity: event.importance === 'high' ? 'high' : event.importance === 'low' ? 'low' : 'medium',
            symbols: event.symbol ? [event.symbol] : []
          }));
          
          // Sort by time (upcoming first)
          realEvents.sort((a, b) => a.time.getTime() - b.time.getTime());
          
          setEvents(realEvents);
        } else {
          setEvents([
            {
              type: 'announcement',
              title: 'Market Events System Active',
              time: new Date(),
              description: 'Monitoring earnings calendars, economic indicators, and market events',
              severity: 'low'
            }
          ]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch real events:', error);
      setEvents([
        {
          type: 'alert',
          title: 'Events Service Connecting...',
          time: new Date(),
          description: 'Connecting to earnings calendars and economic event feeds...',
          severity: 'low'
        }
      ]);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-400 bg-green-400/20';
      case 'negative': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'border-red-500 bg-red-500/10';
      case 'medium': return 'border-yellow-500 bg-yellow-500/10';
      default: return 'border-gray-500 bg-gray-500/10';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-400';
      case 'medium': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return `${Math.floor(diffInSeconds / 86400)}d ago`;
  };

  const filteredNews = news.filter(item => {
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    const matchesSentiment = sentimentFilter === 'all' || item.sentiment === sentimentFilter;
    const matchesSearch = searchTerm === '' || 
      item.headline.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.symbols.some(symbol => symbol.toLowerCase().includes(searchTerm.toLowerCase()));
    
    return matchesCategory && matchesSentiment && matchesSearch;
  });

  return (
    <div className="min-h-screen bg-gray-900 p-2 sm:p-4 lg:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 lg:mb-6 space-y-2 sm:space-y-0">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white mb-1 sm:mb-2">Real-Time News Center</h1>
          <div className="text-gray-400 text-sm sm:text-base">Live market news, events, and sentiment analysis</div>
        </div>
        
        <div className="flex items-center space-x-2 text-xs sm:text-sm text-gray-400">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span>Live Updates</span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 lg:gap-6 h-full">
        {/* Filters and Search */}
        <div className="xl:col-span-1 space-y-3 lg:space-y-4 order-2 xl:order-1">
          {/* Search */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 border border-gray-700">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search news, symbols..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-gray-800 text-white rounded-lg pl-10 pr-4 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none text-sm sm:text-base"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center text-sm sm:text-base">
              <Filter className="w-4 h-4 mr-2" />
              Categories
            </h3>
            <div className="space-y-1 sm:space-y-2">
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`w-full text-left px-2 sm:px-3 py-1 sm:py-2 rounded-lg text-xs sm:text-sm transition-colors ${
                    selectedCategory === category
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Sentiment Filter */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 text-sm sm:text-base">Sentiment</h3>
            <div className="space-y-1 sm:space-y-2">
              {sentiments.map(sentiment => (
                <button
                  key={sentiment}
                  onClick={() => setSentimentFilter(sentiment)}
                  className={`w-full text-left px-2 sm:px-3 py-1 sm:py-2 rounded-lg text-xs sm:text-sm transition-colors ${
                    sentimentFilter === sentiment
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Sentiment Analysis */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center text-sm sm:text-base">
              <TrendingUp className="w-4 h-4 mr-2" />
              Sentiment Analysis
            </h3>
            <div className="space-y-2">
              {sentimentAnalysis.slice(0, 6).map((analysis) => (
                <div key={analysis.symbol} className="p-2 bg-gray-800/50 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-white">{analysis.symbol}</span>
                    <span className={`text-xs px-2 py-1 rounded ${getSentimentColor(analysis.sentimentLabel)}`}>
                      {analysis.sentimentLabel}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    {analysis.articleCount} articles • {(analysis.confidence * 100).toFixed(0)}% confidence
                  </div>
                  <div className="flex gap-1 mt-1">
                    <div className="flex-1 bg-gray-700 rounded-full h-1">
                      <div 
                        className="bg-green-500 h-1 rounded-full" 
                        style={{ width: `${(analysis.distribution.positive / analysis.articleCount) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Market Events */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center text-sm sm:text-base">
              <Clock className="w-4 h-4 mr-2" />
              Market Events
            </h3>
            <div className="space-y-2 sm:space-y-3">
              {events.slice(0, 4).map((event, index) => (
                <div key={index} className="p-2 sm:p-3 bg-gray-800/50 rounded-lg">
                  <div className={`text-xs sm:text-sm font-medium ${getSeverityColor(event.severity)}`}>
                    {event.title}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {event.description}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {formatTimeAgo(event.time)}
                  </div>
                  {event.symbols && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {event.symbols.map(symbol => (
                        <span key={symbol} className="text-xs bg-blue-600 text-white px-1 sm:px-2 py-1 rounded">
                          {symbol}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* News Feed */}
        <div className="xl:col-span-3 order-1 xl:order-2">
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 lg:p-6 border border-gray-700 h-full">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 lg:mb-6 space-y-2 sm:space-y-0">
              <h2 className="text-lg sm:text-xl font-bold text-white flex items-center">
                <Newspaper className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                Latest News ({filteredNews.length})
                {loading && <Loader className="w-4 h-4 ml-2 animate-spin" />}
              </h2>
              <div className="text-xs text-gray-400">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </div>
              <button className="text-blue-400 hover:text-blue-300 text-xs sm:text-sm flex items-center">
                <Eye className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                Mark all as read
              </button>
            </div>

            <div className="space-y-3 sm:space-y-4 max-h-[calc(100vh-200px)] sm:max-h-[calc(100vh-300px)] overflow-y-auto">
              {filteredNews.length > 0 ? (
                filteredNews.map((item) => (
                  <div key={item.id} className={`p-3 sm:p-4 rounded-lg border-l-4 ${getImpactColor(item.impact)}`}>
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-white font-semibold leading-tight pr-2 sm:pr-4 text-sm sm:text-base">
                        {item.headline}
                      </h3>
                      <div className="flex items-center space-x-1 sm:space-x-2 flex-shrink-0">
                        <span className={`px-1 sm:px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(item.sentiment)}`}>
                          {item.sentiment === 'positive' ? <TrendingUp className="w-3 h-3" /> :
                           item.sentiment === 'negative' ? <TrendingDown className="w-3 h-3" /> :
                           <AlertCircle className="w-3 h-3" />}
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3 leading-relaxed">
                      {item.summary}
                    </p>
                    
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between space-y-2 sm:space-y-0">
                      <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-xs sm:text-sm">
                        <span className="text-gray-400">{item.source}</span>
                        <span className="text-gray-500">
                          {formatTimeAgo(item.timestamp)}
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {item.symbols.map(symbol => (
                            <span key={symbol} className="text-xs bg-blue-600 text-white px-1 sm:px-2 py-1 rounded">
                              {symbol}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <button className="text-blue-400 hover:text-blue-300 text-xs sm:text-sm flex items-center whitespace-nowrap">
                        <ExternalLink className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />
                        Read more
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <Newspaper className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">No Real News Available</h3>
                  <p className="text-gray-400 mb-4">Real-time financial news feeds are currently unavailable</p>
                  <button 
                    onClick={fetchRealNews}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm"
                  >
                    Retry Real News
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}