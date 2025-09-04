import React from 'react';

interface NewsItem {
  id: string;
  headline: string;
  content: string;
  source: string;
  timestamp: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  impactScore: number;
  symbols?: string[];
}

interface NewsFlashProps {
  className?: string;
}

export const NewsFlash: React.FC<NewsFlashProps> = ({ className = '' }) => {
  // Sample news items
  const newsItems: NewsItem[] = [
    {
      id: 'news-1',
      headline: 'Fed Signals Potential Rate Cut in Coming Months',
      content: 'Federal Reserve officials indicated they could begin lowering interest rates in the coming months if inflation continues to cool.',
      source: 'Financial Times',
      timestamp: new Date(Date.now() - 1800000).toISOString(), // 30 minutes ago
      sentiment: 'positive',
      impactScore: 85,
      symbols: ['SPY', 'QQQ', 'IWM'],
    },
    {
      id: 'news-2',
      headline: 'Tech Giant Announces New AI Initiative',
      content: 'A major technology company revealed plans for significant investment in artificial intelligence research and development.',
      source: 'Tech Insider',
      timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
      sentiment: 'positive',
      impactScore: 78,
      symbols: ['MSFT', 'GOOGL', 'NVDA'],
    },
    {
      id: 'news-3',
      headline: 'Oil Prices Surge Amid Supply Concerns',
      content: 'Crude oil prices jumped 3% following reports of production disruptions in key oil-producing regions.',
      source: 'Energy News',
      timestamp: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
      sentiment: 'negative',
      impactScore: 72,
      symbols: ['XOM', 'CVX', 'USO'],
    },
    {
      id: 'news-4',
      headline: 'Retail Sales Data Shows Mixed Consumer Sentiment',
      content: 'Latest economic data reveals retail sales grew modestly, though below economists\' expectations.',
      source: 'Market Watch',
      timestamp: new Date(Date.now() - 10800000).toISOString(), // 3 hours ago
      sentiment: 'neutral',
      impactScore: 65,
      symbols: ['WMT', 'TGT', 'XRT'],
    },
  ];

  // Format timestamp to relative time
  const getRelativeTime = (timestamp: string) => {
    const now = new Date();
    const newsTime = new Date(timestamp);
    const diffSeconds = Math.floor((now.getTime() - newsTime.getTime()) / 1000);
    
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
    return `${Math.floor(diffSeconds / 86400)}d ago`;
  };

  return (
    <div className={`news-flash ${className}`}>
      {newsItems.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <p className="text-slate-400">No recent news</p>
        </div>
      ) : (
        <div className="space-y-3">
          {newsItems.map((newsItem) => (
            <div
              key={newsItem.id}
              className="bg-slate-700 rounded-lg p-3 border border-slate-600"
            >
              <h4 className="font-medium text-white mb-1">{newsItem.headline}</h4>
              <p className="text-sm text-slate-300 mb-2">{newsItem.content}</p>
              <div className="flex justify-between text-xs">
                <div className="text-slate-400">
                  {newsItem.source} • {getRelativeTime(newsItem.timestamp)}
                </div>
                <div className="flex items-center space-x-1">
                  <span
                    className={`px-1.5 py-0.5 rounded ${
                      newsItem.sentiment === 'positive'
                        ? 'bg-green-500/20 text-green-400'
                        : newsItem.sentiment === 'negative'
                        ? 'bg-red-500/20 text-red-400'
                        : 'bg-blue-500/20 text-blue-400'
                    }`}
                  >
                    {newsItem.sentiment}
                  </span>
                  {newsItem.symbols && (
                    <span className="text-slate-400">
                      {newsItem.symbols.join(', ')}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
