import React from 'react';

interface MarketTick {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
}

export interface MarketData {
  stocks?: MarketTick[];
  indices?: MarketTick[];
  crypto?: MarketTick[];
  sectors?: {
    name: string;
    performance: number;
    tickers: string[];
  }[];
}

interface MarketMapProps {
  data: MarketData;
  className?: string;
}

export const MarketMap: React.FC<MarketMapProps> = ({ data, className = '' }) => {
  // Generate a treemap-like visualization for market data
  const allItems = [
    ...(data.stocks || []).map(stock => ({
      ...stock,
      type: 'stock',
      size: stock.volume / 1000000, // Size by volume in millions
    })),
    ...(data.indices || []).map(index => ({
      ...index,
      type: 'index',
      size: 50, // Fixed size for indices
    })),
    ...(data.crypto || []).map(crypto => ({
      ...crypto,
      type: 'crypto',
      size: crypto.volume / 10000, // Size by volume adjusted for crypto
    })),
  ];

  return (
    <div className={`market-map ${className}`}>
      {allItems.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <p className="text-slate-400">Loading market data...</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {allItems.map((item) => (
            <div
              key={item.symbol}
              className={`rounded p-3 flex flex-col justify-between ${
                item.change_percent > 0
                  ? 'bg-green-500/10 border border-green-500/20'
                  : item.change_percent < 0
                  ? 'bg-red-500/10 border border-red-500/20'
                  : 'bg-slate-700 border border-slate-600'
              }`}
              style={{
                height: `${Math.max(80, Math.min(140, 80 + item.size / 2))}px`,
              }}
            >
              <div className="flex justify-between items-start">
                <div className="font-bold text-white">{item.symbol}</div>
                <div
                  className={`text-xs px-2 py-1 rounded ${
                    item.type === 'stock'
                      ? 'bg-blue-500/20 text-blue-300'
                      : item.type === 'index'
                      ? 'bg-purple-500/20 text-purple-300'
                      : 'bg-yellow-500/20 text-yellow-300'
                  }`}
                >
                  {item.type}
                </div>
              </div>

              <div>
                <div className="font-medium text-white">${item.price.toFixed(2)}</div>
                <div
                  className={`text-sm ${
                    item.change_percent > 0
                      ? 'text-green-400'
                      : item.change_percent < 0
                      ? 'text-red-400'
                      : 'text-slate-400'
                  }`}
                >
                  {item.change_percent > 0 ? '+' : ''}
                  {item.change_percent.toFixed(2)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
