import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Target, Brain, BarChart3, 
  RefreshCw, Eye, Play, Pause, AlertTriangle, DollarSign, Clock
} from 'lucide-react';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  timestamp: string;
}

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
}

interface TradingSignal {
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  timestamp: string;
}

const ProfessionalTradingTerminal: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1h');
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>('');

  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX'];
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  const fetchMarketData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:8001/api/market/latest?symbols=${symbols.join(',')}&timeframe=${timeframe}`, {
        signal: AbortSignal.timeout(3000)
      });
      
      if (response.ok) {
        const data = await response.json();
        const stocks = Array.isArray(data) ? data : data.data || [];
        
        const newMarketData: Record<string, MarketData> = {};
        
        stocks.forEach((stock: any) => {
          if (symbols.includes(stock.symbol)) {
            const price = parseFloat(stock.price || stock.current_price || stock.last_price);
            if (price > 0) {
              newMarketData[stock.symbol] = {
                symbol: stock.symbol,
                price,
                change: parseFloat(stock.change || stock.price_change) || 0,
                changePercent: parseFloat(stock.changePercent || stock.change_percent) || 0,
                volume: parseInt(stock.volume || stock.total_volume) || 0,
                high: parseFloat(stock.high || stock.day_high) || price,
                low: parseFloat(stock.low || stock.day_low) || price,
                open: parseFloat(stock.open || stock.day_open) || price,
                timestamp: new Date().toISOString()
              };
            }
          }
        });
        
        if (Object.keys(newMarketData).length > 0) {
          setMarketData(newMarketData);
          setLastUpdate(new Date().toLocaleTimeString());
        }
      }
    } catch (err) {
      setError('Market data unavailable');
    } finally {
      setIsLoading(false);
    }
  }, [symbols, timeframe]);

  const fetchIndicators = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8001/api/agents/signal-generator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: [selectedSymbol] }),
        signal: AbortSignal.timeout(3000)
      });
      
      if (response.ok) {
        const data = await response.json();
        const signalData = data?.data?.individual_signals?.[selectedSymbol];
        
        if (signalData) {
          const newIndicators: TechnicalIndicator[] = [];
          
          if (signalData.rsi !== undefined) {
            const rsi = parseFloat(signalData.rsi);
            newIndicators.push({
              name: 'RSI (14)',
              value: rsi,
              signal: rsi > 70 ? 'SELL' : rsi < 30 ? 'BUY' : 'HOLD',
              confidence: Math.abs(rsi - 50) / 50
            });
          }
          
          if (signalData.macd !== undefined) {
            const macd = parseFloat(signalData.macd);
            newIndicators.push({
              name: 'MACD',
              value: macd,
              signal: macd > 0 ? 'BUY' : macd < 0 ? 'SELL' : 'HOLD',
              confidence: Math.min(Math.abs(macd) / 2, 1)
            });
          }
          
          if (signalData.signal_type) {
            newIndicators.push({
              name: 'AI Signal',
              value: parseFloat(signalData.composite_score) || 0,
              signal: signalData.signal_type as 'BUY' | 'SELL' | 'HOLD',
              confidence: parseFloat(signalData.confidence) / 100 || 0.5
            });
          }
          
          setIndicators(newIndicators);
        }
      }
    } catch (err) {
      setIndicators([]);
    }
  }, [selectedSymbol]);

  const fetchSignals = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8001/api/agents/signal-generator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: [selectedSymbol] }),
        signal: AbortSignal.timeout(3000)
      });
      
      if (response.ok) {
        const data = await response.json();
        const signalData = data?.data?.individual_signals?.[selectedSymbol];
        
        if (signalData) {
          const signal: TradingSignal = {
            symbol: selectedSymbol,
            type: signalData.signal_type as 'BUY' | 'SELL' | 'HOLD' || 'HOLD',
            confidence: parseFloat(signalData.confidence) || 50,
            reasoning: signalData.reasoning || 'AI market analysis',
            timestamp: new Date().toISOString()
          };
          
          setSignals([signal]);
        }
      }
    } catch (err) {
      setSignals([]);
    }
  }, [selectedSymbol]);

  const refreshAll = useCallback(() => {
    fetchMarketData();
    fetchIndicators();
    fetchSignals();
  }, [fetchMarketData, fetchIndicators, fetchSignals]);

  useEffect(() => {
    refreshAll();
    
    if (isAutoRefresh) {
      const interval = setInterval(refreshAll, 5000);
      return () => clearInterval(interval);
    }
  }, [refreshAll, isAutoRefresh]);

  useEffect(() => {
    fetchIndicators();
    fetchSignals();
  }, [selectedSymbol, fetchIndicators, fetchSignals]);

  const currentData = marketData[selectedSymbol];
  
  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-400';
      case 'SELL': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <TrendingUp className="w-4 h-4" />;
      case 'SELL': return <TrendingDown className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white p-2 sm:p-4 lg:p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 sm:mb-6 lg:mb-8 gap-4 sm:gap-0">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-white">Professional Trading Terminal</h1>
          <div className="flex items-center justify-between sm:justify-end space-x-3 sm:space-x-4">
            <div className="text-xs sm:text-sm text-gray-400">
              <span className="hidden sm:inline">Last Update: </span>{lastUpdate}
            </div>
            <button
              onClick={refreshAll}
              disabled={isLoading}
              title="Refresh all data"
              aria-label="Refresh all data"
              className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white transition-colors"
            >
              <RefreshCw className={`w-4 h-4 sm:w-5 sm:h-5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
              title={`Auto-refresh: ${isAutoRefresh ? 'ON' : 'OFF'}`}
              aria-label={`Auto-refresh: ${isAutoRefresh ? 'ON' : 'OFF'}`}
              className={`p-2 rounded-lg transition-colors ${
                isAutoRefresh ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
              }`}
            >
              {isAutoRefresh ? <Play className="w-4 h-4 sm:w-5 sm:h-5" /> : <Pause className="w-4 h-4 sm:w-5 sm:h-5" />}
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-6 mb-4 sm:mb-6 lg:mb-8">
          <select 
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            title="Select trading symbol"
            aria-label="Select trading symbol"
            className="bg-gray-800 text-white rounded-lg px-3 sm:px-4 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none text-sm sm:text-base"
          >
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          
          <div className="flex space-x-1 sm:space-x-2 overflow-x-auto pb-2 sm:pb-0">
            {timeframes.map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 sm:px-3 py-1 rounded-md text-xs sm:text-sm font-medium transition-colors flex-shrink-0 ${
                  timeframe === tf 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500/50 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <span className="text-red-300">{error}</span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          
          {/* Main Price Display */}
          <div className="xl:col-span-2 bg-black/40 rounded-xl p-6 backdrop-blur-sm border border-gray-700">
            {error ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <AlertTriangle className="w-12 h-12 mx-auto mb-2 text-red-500" />
                  <p className="text-red-400 text-lg font-semibold">Backend Offline</p>
                  <p className="text-gray-400">No market data available</p>
                  <p className="text-sm text-gray-500 mt-2">Start backend server to see real data</p>
                </div>
              </div>
            ) : currentData ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-white">{selectedSymbol}</h2>
                  <div className="text-right">
                    <div className="text-4xl font-bold text-white">
                      ${currentData.price.toFixed(2)}
                    </div>
                    <div className={`flex items-center space-x-2 ${
                      currentData.change >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {currentData.change >= 0 ? 
                        <TrendingUp className="w-6 h-6" /> : 
                        <TrendingDown className="w-6 h-6" />
                      }
                      <span className="text-xl font-semibold">
                        {currentData.changePercent >= 0 ? '+' : ''}{currentData.changePercent.toFixed(2)}%
                      </span>
                      <span className="text-lg">
                        ({currentData.change >= 0 ? '+' : ''}${currentData.change.toFixed(2)})
                      </span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <div className="text-gray-400 text-sm">Open</div>
                    <div className="text-white font-semibold">${currentData.open.toFixed(2)}</div>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <div className="text-gray-400 text-sm">High</div>
                    <div className="text-green-400 font-semibold">${currentData.high.toFixed(2)}</div>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <div className="text-gray-400 text-sm">Low</div>
                    <div className="text-red-400 font-semibold">${currentData.low.toFixed(2)}</div>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <div className="text-gray-400 text-sm">Volume</div>
                    <div className="text-white font-semibold">{(currentData.volume / 1000000).toFixed(1)}M</div>
                  </div>
                </div>

                {/* Chart Placeholder */}
                <div className="h-64 bg-gray-800/30 rounded-lg border border-gray-600 flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-2 text-gray-500" />
                    <p className="text-gray-400">Real-time chart for {selectedSymbol}</p>
                    <p className="text-sm text-gray-500">Price: ${currentData.price.toFixed(2)}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <Clock className="w-12 h-12 mx-auto mb-2 text-gray-500" />
                  <p className="text-gray-400">Connecting to backend...</p>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel */}
          <div className="space-y-6">
            
            {/* AI Trading Signals */}
            <div className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 rounded-xl p-4 backdrop-blur-sm border border-purple-500/30">
              <div className="flex items-center space-x-2 mb-4">
                <Brain className="w-5 h-5 text-purple-400" />
                <h3 className="text-white font-semibold">AI Trading Signals</h3>
              </div>
              
              {error ? (
                <div className="text-center py-4 text-red-400">
                  <AlertTriangle className="w-6 h-6 mx-auto mb-2" />
                  <p className="text-sm">Backend offline</p>
                  <p className="text-xs text-gray-500">No AI signals available</p>
                </div>
              ) : signals.length > 0 ? (
                signals.map((signal, index) => (
                  <div key={index} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm">Signal:</span>
                      <span className={`font-semibold ${getSignalColor(signal.type)} flex items-center space-x-1`}>
                        {getSignalIcon(signal.type)}
                        <span>{signal.type}</span>
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm">Confidence:</span>
                      <span className="text-white font-semibold">{signal.confidence}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          signal.type === 'BUY' ? 'bg-green-400' :
                          signal.type === 'SELL' ? 'bg-red-400' : 'bg-yellow-400'
                        }`}
                        style={{ width: `${signal.confidence}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-400">{signal.reasoning}</p>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-gray-400">
                  <Target className="w-6 h-6 mx-auto mb-2" />
                  <p className="text-sm">Connecting to AI...</p>
                </div>
              )}
            </div>
            
            {/* Technical Indicators */}
            <div className="bg-black/40 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
              <div className="flex items-center space-x-2 mb-4">
                <Activity className="w-5 h-5 text-blue-400" />
                <h3 className="text-white font-semibold">Technical Indicators</h3>
              </div>
              
              <div className="space-y-3">
                {error ? (
                  <div className="text-center py-4 text-red-400">
                    <AlertTriangle className="w-6 h-6 mx-auto mb-2" />
                    <p className="text-sm">Backend offline</p>
                    <p className="text-xs text-gray-500">No indicators available</p>
                  </div>
                ) : indicators.length > 0 ? (
                  indicators.map((indicator, index) => (
                    <div key={index} className="p-3 bg-gray-800/50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white font-medium text-sm">{indicator.name}</span>
                        <div className={`flex items-center space-x-1 ${getSignalColor(indicator.signal)}`}>
                          {getSignalIcon(indicator.signal)}
                          <span className="text-xs font-semibold">{indicator.signal}</span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-400">Value: {indicator.value.toFixed(2)}</span>
                        <span className="text-gray-400">Confidence: {(indicator.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-gray-400">
                    <Activity className="w-6 h-6 mx-auto mb-2" />
                    <p className="text-sm">Connecting to backend...</p>
                  </div>
                )}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-black/40 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
              <h3 className="text-white font-semibold mb-4 flex items-center">
                <DollarSign className="w-5 h-5 mr-2 text-green-400" />
                Quick Actions
              </h3>
              
              <div className="space-y-2">
                <button 
                  onClick={() => currentData && alert(`Buy ${selectedSymbol} at $${currentData.price.toFixed(2)}`)}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2"
                >
                  <TrendingUp className="w-4 h-4" />
                  <span>Buy {selectedSymbol}</span>
                </button>
                <button 
                  onClick={() => currentData && alert(`Sell ${selectedSymbol} at $${currentData.price.toFixed(2)}`)}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2"
                >
                  <TrendingDown className="w-4 h-4" />
                  <span>Sell {selectedSymbol}</span>
                </button>
                <button 
                  onClick={() => currentData && alert(`Alert set for ${selectedSymbol} at $${currentData.price.toFixed(2)}`)}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2"
                >
                  <Target className="w-4 h-4" />
                  <span>Set Alert</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalTradingTerminal;