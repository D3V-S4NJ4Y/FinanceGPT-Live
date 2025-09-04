import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Target, Zap, Brain, BarChart3, 
  RefreshCw, Settings, Eye, Filter, Maximize2, Play, Pause, Volume2,
  AlertTriangle, CheckCircle, Clock, DollarSign, Percent, PieChart
} from 'lucide-react';

interface RealTimeData {
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
  description: string;
}

interface TradingSignal {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  target: number;
  stopLoss: number;
  reasoning: string;
  timeframe: string;
  timestamp: string;
}

interface OrderBookEntry {
  price: number;
  size: number;
  side: 'bid' | 'ask';
}

interface CandlestickData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const useRealTimeMarketData = (symbols: string[], timeframe: string) => {
  const [marketData, setMarketData] = useState<Record<string, RealTimeData>>({});
  const [candlestickData, setCandlestickData] = useState<Record<string, CandlestickData[]>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const fetchMarketData = useCallback(async () => {
    // Use the working endpoint from logs
    const endpoint = `http://localhost:8001/api/market/latest?symbols=${symbols.join(',')}&timeframe=${timeframe}`;
    
    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(2000)
      });
      
      if (response.ok) {
        const data = await response.json();
        const stocks = Array.isArray(data) ? data : data.data || data.stocks || data.results || [];
        
        if (stocks.length > 0) {
          const newMarketData: Record<string, RealTimeData> = {};
          const newCandlestickData: Record<string, CandlestickData[]> = {};
          
          for (const stock of stocks) {
            if (!symbols.includes(stock.symbol)) continue;
            
            const price = parseFloat(stock.price || stock.current_price || stock.last_price || stock.regularMarketPrice);
            if (!price || price <= 0) continue;
            
            newMarketData[stock.symbol] = {
              symbol: stock.symbol,
              price,
              change: parseFloat(stock.change || stock.price_change || stock.regularMarketChange) || 0,
              changePercent: parseFloat(stock.changePercent || stock.change_percent || stock.regularMarketChangePercent) || 0,
              volume: parseInt(stock.volume || stock.total_volume || stock.regularMarketVolume) || 0,
              high: parseFloat(stock.high || stock.day_high || stock.regularMarketDayHigh) || price,
              low: parseFloat(stock.low || stock.day_low || stock.regularMarketDayLow) || price,
              open: parseFloat(stock.open || stock.day_open || stock.regularMarketOpen) || price,
              timestamp: new Date().toISOString()
            };
            
            newCandlestickData[stock.symbol] = [{
              timestamp: Date.now(),
              open: price,
              high: price * 1.001,
              low: price * 0.999,
              close: price,
              volume: parseInt(stock.volume) || 1000000
            }];
          }
          
          if (Object.keys(newMarketData).length > 0) {
            setMarketData(newMarketData);
            setCandlestickData(newCandlestickData);
            setError(null);
            setIsLoading(false);
            return;
          }
        }
      }
    } catch (err) {
      console.warn('Market data fetch failed:', err);
    }
    
    setError('Market data temporarily unavailable');
    setIsLoading(false);
  }, [symbols, timeframe]);

  const connectWebSocket = useCallback(() => {
    const wsEndpoints = [
      `ws://localhost:8001/ws/trading-${Date.now()}`,
      `ws://localhost:8001/ws/market`,
      `ws://localhost:8001/api/ws`
    ];
    
    let currentEndpoint = 0;
    
    const tryConnect = () => {
      if (currentEndpoint >= wsEndpoints.length) {
        console.log('All WebSocket endpoints failed, will retry in 10 seconds');
        setTimeout(() => {
          currentEndpoint = 0;
          tryConnect();
        }, 10000);
        return;
      }
      
      try {
        const ws = new WebSocket(wsEndpoints[currentEndpoint]);
        wsRef.current = ws;
        
        const connectionTimeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            console.log(`WebSocket timeout for ${wsEndpoints[currentEndpoint]}`);
            ws.close();
            currentEndpoint++;
            tryConnect();
          }
        }, 2000);
        
        ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('✅ WebSocket connected to:', wsEndpoints[currentEndpoint]);
          
          ws.send(JSON.stringify({
            type: 'subscribe',
            channels: ['market_data', 'price_updates'],
            symbols: symbols,
            timeframe: timeframe
          }));
        };
        
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            console.log('WebSocket message:', message);
            
            if (message.type === 'market_update' || message.type === 'price_update') {
              const updates = Array.isArray(message.data) ? message.data : [message.data];
              
              setMarketData(prev => {
                const updated = { ...prev };
                updates.forEach((update: any) => {
                  if (symbols.includes(update.symbol)) {
                    updated[update.symbol] = {
                      symbol: update.symbol,
                      price: parseFloat(update.price) || prev[update.symbol]?.price || 0,
                      change: parseFloat(update.change) || prev[update.symbol]?.change || 0,
                      changePercent: parseFloat(update.changePercent) || prev[update.symbol]?.changePercent || 0,
                      volume: parseInt(update.volume) || prev[update.symbol]?.volume || 0,
                      high: parseFloat(update.high) || prev[update.symbol]?.high || 0,
                      low: parseFloat(update.low) || prev[update.symbol]?.low || 0,
                      open: parseFloat(update.open) || prev[update.symbol]?.open || 0,
                      timestamp: new Date().toISOString()
                    };
                  }
                });
                return updated;
              });
            }
          } catch (e) {
            console.error('WebSocket message parse error:', e);
          }
        };
        
        ws.onerror = (error) => {
          console.warn(`WebSocket error for ${wsEndpoints[currentEndpoint]}:`, error);
          clearTimeout(connectionTimeout);
          currentEndpoint++;
          tryConnect();
        };
        
        ws.onclose = () => {
          console.log(`WebSocket closed for ${wsEndpoints[currentEndpoint]}`);
          clearTimeout(connectionTimeout);
          setTimeout(() => {
            currentEndpoint = 0;
            tryConnect();
          }, 5000);
        };
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        currentEndpoint++;
        tryConnect();
      }
    };
    
    tryConnect();
  }, [symbols, timeframe]);

  useEffect(() => {
    setIsLoading(true);
    fetchMarketData();
    connectWebSocket();
    
    const marketInterval = setInterval(fetchMarketData, 5000);
    
    return () => {
      clearInterval(marketInterval);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [fetchMarketData, connectWebSocket]);

  return { marketData, candlestickData, isLoading, error, refresh: fetchMarketData };
};

const useTechnicalIndicators = (symbol: string) => {
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchIndicators = useCallback(async () => {
    const endpoints = [
      'http://localhost:8001/api/agents/signal-generator',
      'http://localhost:8001/agents/signal-generator',
      'http://localhost:8001/signal-generator'
    ];
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: [symbol] }),
          signal: AbortSignal.timeout(2000)
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Indicators response:', data);
          
          const signalData = data?.data?.individual_signals?.[symbol] || data?.signals?.[symbol] || data;
          
          if (signalData) {
            const indicators: TechnicalIndicator[] = [];
            
            if (signalData.rsi !== undefined) {
              indicators.push({
                name: 'RSI (14)',
                value: parseFloat(signalData.rsi),
                signal: signalData.signal_type as 'BUY' | 'SELL' | 'HOLD' || 'HOLD',
                confidence: parseFloat(signalData.confidence) / 100 || 0.7,
                description: 'Relative Strength Index'
              });
            }
            
            if (signalData.macd !== undefined) {
              indicators.push({
                name: 'MACD',
                value: parseFloat(signalData.macd),
                signal: signalData.signal_type as 'BUY' | 'SELL' | 'HOLD' || 'HOLD',
                confidence: parseFloat(signalData.confidence) / 100 || 0.6,
                description: 'Moving Average Convergence Divergence'
              });
            }
            
            setIndicators(indicators);
            return;
          }
        }
      } catch (error) {
        console.warn(`Indicators endpoint ${endpoint} failed:`, error);
        continue;
      }
    }
    
    setIndicators([]);
  }, [symbol]);

  useEffect(() => {
    const timeout = setTimeout(fetchIndicators, 500);
    const interval = setInterval(fetchIndicators, 12000);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [fetchIndicators]);

  return { indicators, isLoading, refresh: fetchIndicators };
};

const useTradingSignals = (symbol: string) => {
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchSignals = useCallback(async () => {
    const endpoints = [
      'http://localhost:8001/api/agents/signal-generator',
      'http://localhost:8001/api/agents/executive-summary'
    ];
    
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: [symbol] }),
          signal: AbortSignal.timeout(2000)
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Signals response:', data);
          
          const signalData = data?.data?.individual_signals?.[symbol] || data?.data || data;
          
          if (signalData && typeof signalData === 'object') {
            const signal: TradingSignal = {
              id: `signal-${symbol}-${Date.now()}`,
              symbol,
              type: signalData.signal_type as 'BUY' | 'SELL' | 'HOLD' || 'HOLD',
              confidence: parseFloat(signalData.confidence) || 65,
              price: parseFloat(signalData.current_price || signalData.price) || 0,
              target: parseFloat(signalData.target_price || signalData.target) || 0,
              stopLoss: parseFloat(signalData.stop_loss || signalData.stopLoss) || 0,
              reasoning: signalData.reasoning || signalData.executive_summary || 'AI market analysis',
              timeframe: signalData.timeframe || signalData.time_horizon || '1-3 days',
              timestamp: new Date().toISOString()
            };
            
            setSignals([signal]);
            return;
          }
        }
      } catch (error) {
        console.warn(`Signals endpoint ${endpoint} failed:`, error);
        continue;
      }
    }
    
    // No default signal - show empty state when backend is offline
    setSignals([]);
  }, [symbol]);

  useEffect(() => {
    const timeout = setTimeout(fetchSignals, 200);
    const interval = setInterval(fetchSignals, 15000);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [fetchSignals]);

  return { signals, isLoading, refresh: fetchSignals };
};

const AdvancedChart = ({ data, symbol }: { data: CandlestickData[], symbol: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCandle, setHoveredCandle] = useState<CandlestickData | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    setMousePos({ x, y });
    
    const margin = 40;
    const chartWidth = canvas.width - margin * 2;
    const candleIndex = Math.floor(((x - margin) / chartWidth) * data.length);
    
    if (candleIndex >= 0 && candleIndex < data.length) {
      setHoveredCandle(data[candleIndex]);
    } else {
      setHoveredCandle(null);
    }
  }, [data]);

  useEffect(() => {
    if (!canvasRef.current || !data.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    
    // Clear canvas with gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, '#0a0a0a');
    gradient.addColorStop(1, '#1a1a1a');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Calculate price range with padding
    const prices = data.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.1;
    const adjustedMin = minPrice - padding;
    const adjustedMax = maxPrice + padding;
    const adjustedRange = adjustedMax - adjustedMin;

    const margin = 60;
    const chartWidth = width - margin * 2;
    const chartHeight = height - margin * 2;

    // Draw horizontal grid lines
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 0.5;
    
    for (let i = 0; i <= 8; i++) {
      const y = margin + (chartHeight / 8) * i;
      ctx.beginPath();
      ctx.moveTo(margin, y);
      ctx.lineTo(width - margin, y);
      ctx.stroke();
    }

    // Draw vertical grid lines
    const timeStep = Math.max(1, Math.floor(data.length / 10));
    for (let i = 0; i < data.length; i += timeStep) {
      const x = margin + (chartWidth / data.length) * i;
      ctx.beginPath();
      ctx.moveTo(x, margin);
      ctx.lineTo(x, height - margin);
      ctx.stroke();
    }

    // Calculate moving averages
    const sma20: number[] = [];
    const sma50: number[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (i >= 19) {
        const sum20 = data.slice(i - 19, i + 1).reduce((sum, candle) => sum + candle.close, 0);
        sma20.push(sum20 / 20);
      } else {
        sma20.push(data[i].close);
      }
      
      if (i >= 49) {
        const sum50 = data.slice(i - 49, i + 1).reduce((sum, candle) => sum + candle.close, 0);
        sma50.push(sum50 / 50);
      } else {
        sma50.push(data[i].close);
      }
    }

    // Draw moving averages
    const drawMA = (values: number[], color: string, lineWidth: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      
      values.forEach((value, index) => {
        const x = margin + (chartWidth / data.length) * (index + 0.5);
        const y = margin + ((adjustedMax - value) / adjustedRange) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    };

    drawMA(sma20, '#4ade80', 1.5); // Green for SMA20
    drawMA(sma50, '#f59e0b', 1.5); // Orange for SMA50

    // Draw candlesticks
    const candleWidth = Math.max(2, (chartWidth / data.length) * 0.7);
    
    data.forEach((candle, index) => {
      const x = margin + (chartWidth / data.length) * index;
      const openY = margin + ((adjustedMax - candle.open) / adjustedRange) * chartHeight;
      const closeY = margin + ((adjustedMax - candle.close) / adjustedRange) * chartHeight;
      const highY = margin + ((adjustedMax - candle.high) / adjustedRange) * chartHeight;
      const lowY = margin + ((adjustedMax - candle.low) / adjustedRange) * chartHeight;
      
      const isGreen = candle.close > candle.open;
      const wickColor = isGreen ? '#10b981' : '#ef4444';
      const bodyColor = isGreen ? '#10b981' : '#ef4444';
      
      // Draw wick
      ctx.strokeStyle = wickColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();
      
      // Draw body
      if (isGreen) {
        ctx.strokeStyle = bodyColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + candleWidth * 0.15, closeY, candleWidth * 0.7, Math.abs(openY - closeY) || 1);
      } else {
        ctx.fillStyle = bodyColor;
        ctx.fillRect(x + candleWidth * 0.15, closeY, candleWidth * 0.7, Math.abs(openY - closeY) || 1);
      }
    });

    // Draw price labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 8; i++) {
      const price = adjustedMax - (adjustedRange / 8) * i;
      const y = margin + (chartHeight / 8) * i;
      ctx.fillText(`$${price.toFixed(2)}`, margin - 10, y + 4);
    }

    // Draw time labels
    ctx.textAlign = 'center';
    for (let i = 0; i < data.length; i += timeStep) {
      const x = margin + (chartWidth / data.length) * i;
      const time = new Date(data[i].timestamp).toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      ctx.fillText(time, x, height - 10);
    }

    // Draw legend
    ctx.textAlign = 'left';
    ctx.fillStyle = '#4ade80';
    ctx.fillText('SMA20', margin + 10, margin - 25);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('SMA50', margin + 70, margin - 25);
    
  }, [data]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        className="w-full h-full border border-gray-700 rounded-lg bg-black cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredCandle(null)}
      />
      
      {/* Tooltip */}
      {hoveredCandle && (
        <div 
          className="absolute bg-gray-900 border border-gray-600 rounded-lg p-3 text-xs text-white pointer-events-none z-10"
          style={{
            left: Math.min(mousePos.x + 10, 600),
            top: Math.max(mousePos.y - 80, 10)
          }}
        >
          <div className="space-y-1">
            <div className="font-semibold text-blue-400">
              {new Date(hoveredCandle.timestamp).toLocaleString()}
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>Open: <span className="text-gray-300">${hoveredCandle.open.toFixed(2)}</span></div>
              <div>High: <span className="text-green-400">${hoveredCandle.high.toFixed(2)}</span></div>
              <div>Low: <span className="text-red-400">${hoveredCandle.low.toFixed(2)}</span></div>
              <div>Close: <span className="text-white">${hoveredCandle.close.toFixed(2)}</span></div>
            </div>
            <div className="text-gray-400">
              Volume: {(hoveredCandle.volume / 1000000).toFixed(1)}M
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const usePortfolioData = () => {
  const [portfolio, setPortfolio] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPortfolio = useCallback(async () => {
    // Portfolio endpoint not available (404), skip for now
    setIsLoading(false);
  }, []);

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 30000);
    return () => clearInterval(interval);
  }, [fetchPortfolio]);

  return { portfolio, isLoading, refresh: fetchPortfolio };
};

export default function EnhancedTradingTerminal() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1h');
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [showOrderBook, setShowOrderBook] = useState(false);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [showPortfolio, setShowPortfolio] = useState(false);

  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'CRM', 'INTC'];
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  const { marketData, candlestickData, isLoading: marketLoading, error: marketError, refresh: refreshMarket } = useRealTimeMarketData([selectedSymbol], timeframe);
  const { indicators, isLoading: indicatorsLoading, refresh: refreshIndicators } = useTechnicalIndicators(selectedSymbol);
  const { signals, isLoading: signalsLoading, refresh: refreshSignals } = useTradingSignals(selectedSymbol);
  const { portfolio, isLoading: portfolioLoading, refresh: refreshPortfolio } = usePortfolioData();

  const currentData = marketData[selectedSymbol];
  const currentCandles = candlestickData[selectedSymbol] || [];

  const handleRefreshAll = useCallback(() => {
    refreshMarket();
    refreshIndicators();
    refreshSignals();
    refreshPortfolio();
  }, [refreshMarket, refreshIndicators, refreshSignals, refreshPortfolio]);

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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white p-4">
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 h-full">
        
        {/* Main Chart Area */}
        <div className="xl:col-span-3 bg-black/40 rounded-xl p-6 backdrop-blur-sm border border-gray-700">
          
          {/* Header Controls */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <select 
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                title="Select trading symbol"
                aria-label="Select trading symbol"
                className="bg-gray-800 text-white rounded-lg px-4 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                {symbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
              
              <div className="flex space-x-1">
                {timeframes.map(tf => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
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
            
            <div className="flex items-center space-x-4">
              {currentData && (
                <div className="text-right">
                  <div className="text-2xl font-bold text-white">
                    ${currentData.price.toFixed(2)}
                  </div>
                  <div className={`flex items-center space-x-1 ${
                    currentData.change >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {currentData.change >= 0 ? 
                      <TrendingUp className="w-5 h-5" /> : 
                      <TrendingDown className="w-5 h-5" />
                    }
                    <span className="font-semibold">
                      {currentData.changePercent >= 0 ? '+' : ''}{currentData.changePercent.toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
              
              <button
                onClick={handleRefreshAll}
                disabled={marketLoading}
                title="Refresh all data"
                aria-label="Refresh all data"
                className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white transition-colors"
              >
                <RefreshCw className={`w-5 h-5 ${marketLoading ? 'animate-spin' : ''}`} />
              </button>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                  className={`p-2 rounded-lg transition-colors ${
                    isAutoRefresh ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                  title={`Auto-refresh: ${isAutoRefresh ? 'ON' : 'OFF'}`}
                  aria-label={`Auto-refresh: ${isAutoRefresh ? 'ON' : 'OFF'}`}
                >
                  {isAutoRefresh ? <Play className="w-5 h-5" /> : <Pause className="w-5 h-5" />}
                </button>
                
                <button
                  onClick={() => setShowOrderBook(!showOrderBook)}
                  className={`p-2 rounded-lg transition-colors ${
                    showOrderBook ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                  title="Toggle Order Book"
                  aria-label="Toggle Order Book"
                >
                  <Eye className="w-5 h-5" />
                </button>
                
                <button
                  onClick={() => setAlertsEnabled(!alertsEnabled)}
                  className={`p-2 rounded-lg transition-colors ${
                    alertsEnabled ? 'bg-yellow-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                  title={`Alerts: ${alertsEnabled ? 'ON' : 'OFF'}`}
                  aria-label={`Alerts: ${alertsEnabled ? 'ON' : 'OFF'}`}
                >
                  <AlertTriangle className="w-5 h-5" />
                </button>
                
                <button
                  onClick={() => setShowPortfolio(!showPortfolio)}
                  className={`p-2 rounded-lg transition-colors ${
                    showPortfolio ? 'bg-purple-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                  title="Toggle Portfolio"
                  aria-label="Toggle Portfolio"
                >
                  <PieChart className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {marketError && (
            <div className="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <span className="text-red-300">{marketError}</span>
              </div>
            </div>
          )}
          
          {/* Advanced Chart */}
          <div className="relative h-96">
            {marketLoading ? (
              <div className="flex items-center justify-center h-full bg-black/20 rounded-lg border border-gray-700">
                <div className="text-center">
                  <RefreshCw className="w-8 h-8 mx-auto mb-2 text-blue-400 animate-spin" />
                  <p className="text-gray-400">Loading real-time data...</p>
                </div>
              </div>
            ) : currentCandles.length > 0 ? (
              <AdvancedChart data={currentCandles} symbol={selectedSymbol} />
            ) : marketError ? (
              <div className="flex items-center justify-center h-full bg-black/20 rounded-lg border border-gray-700">
                <div className="text-center">
                  <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-red-400" />
                  <p className="text-gray-400 mb-2">Failed to load chart data</p>
                  <p className="text-xs text-gray-500 mb-3">{marketError}</p>
                  <button 
                    onClick={handleRefreshAll}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
                  >
                    Retry Connection
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full bg-black/20 rounded-lg border border-gray-700">
                <div className="text-center">
                  <Clock className="w-8 h-8 mx-auto mb-2 text-blue-400" />
                  <p className="text-gray-400">Connecting to market data...</p>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Right Panel */}
        <div className="space-y-6">
          
          {/* Portfolio Overview */}
          {showPortfolio && (
            <div className="bg-gradient-to-br from-green-900/40 to-blue-900/40 rounded-xl p-4 backdrop-blur-sm border border-green-500/30">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <PieChart className="w-5 h-5 text-green-400" />
                  <h3 className="text-white font-semibold">Portfolio</h3>
                </div>
                {portfolioLoading && <RefreshCw className="w-4 h-4 animate-spin text-green-400" />}
              </div>
              
              {portfolio ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Total Value:</span>
                      <div className="text-white font-semibold">
                        ${(portfolio.total_value || 100000).toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400">Day P&L:</span>
                      <div className={`font-semibold ${
                        (portfolio.day_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        ${(portfolio.day_pnl || 0).toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400">Total P&L:</span>
                      <div className={`font-semibold ${
                        (portfolio.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        ${(portfolio.total_pnl || 0).toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400">Cash:</span>
                      <div className="text-white font-semibold">
                        ${(portfolio.cash || 50000).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  {portfolio.positions && portfolio.positions.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-gray-300 text-sm font-medium mb-2">Top Holdings:</h4>
                      <div className="space-y-2">
                        {portfolio.positions.slice(0, 3).map((position: any, index: number) => (
                          <div key={index} className="flex justify-between items-center text-xs">
                            <span className="text-white font-medium">{position.symbol}</span>
                            <div className="text-right">
                              <div className="text-gray-300">{position.quantity} shares</div>
                              <div className={`${
                                position.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                ${position.unrealized_pnl?.toFixed(2) || '0.00'}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-4 text-gray-400">
                  <DollarSign className="w-6 h-6 mx-auto mb-2" />
                  <p className="text-sm">Loading portfolio data...</p>
                </div>
              )}
            </div>
          )}
          
          {/* AI Trading Signals */}
          <div className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 rounded-xl p-4 backdrop-blur-sm border border-purple-500/30">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Brain className="w-5 h-5 text-purple-400" />
                <h3 className="text-white font-semibold">AI Signals</h3>
              </div>
              {signalsLoading && <RefreshCw className="w-4 h-4 animate-spin text-purple-400" />}
            </div>
            
            {signals.length > 0 ? (
              signals.map((signal) => (
                <div key={signal.id} className="space-y-3">
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
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300 text-sm">Timeframe:</span>
                    <span className="text-white font-semibold">{signal.timeframe}</span>
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
                  <p className="text-xs text-gray-400 mt-2">{signal.reasoning}</p>
                </div>
              ))
            ) : (
              <div className="text-center py-4 text-red-400">
                <AlertTriangle className="w-6 h-6 mx-auto mb-2" />
                <p className="text-sm">Backend offline</p>
                <p className="text-xs text-gray-500">No AI signals available</p>
              </div>
            )}
          </div>
          
          {/* Technical Indicators */}
          <div className="bg-black/40 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-blue-400" />
                <h3 className="text-white font-semibold">Technical Analysis</h3>
              </div>
              {indicatorsLoading && <RefreshCw className="w-4 h-4 animate-spin text-blue-400" />}
            </div>
            
            <div className="space-y-3">
              {indicators.length > 0 ? (
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
                    <p className="text-xs text-gray-500 mt-1">{indicator.description}</p>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-red-400">
                  <Activity className="w-6 h-6 mx-auto mb-2" />
                  <p className="text-sm">Backend offline</p>
                  <p className="text-xs text-gray-500">No indicators available</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Market Data */}
          {currentData && (
            <div className="bg-black/40 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
              <h3 className="text-white font-semibold mb-4 flex items-center">
                <DollarSign className="w-5 h-5 mr-2 text-green-400" />
                Market Data
              </h3>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Open:</span>
                  <span className="text-white">${currentData.open.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">High:</span>
                  <span className="text-green-400">${currentData.high.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Low:</span>
                  <span className="text-red-400">${currentData.low.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Volume:</span>
                  <span className="text-white">{(currentData.volume / 1000000).toFixed(1)}M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Change:</span>
                  <span className={currentData.change >= 0 ? 'text-green-400' : 'text-red-400'}>
                    ${currentData.change.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          {/* Order Book */}
          {showOrderBook && (
            <div className="bg-black/40 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
              <h3 className="text-white font-semibold mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                Order Book
              </h3>
              
              <div className="space-y-2">
                <div className="text-xs text-gray-400 grid grid-cols-3 gap-2">
                  <span>Price</span>
                  <span>Size</span>
                  <span>Total</span>
                </div>
                
                <div className="text-center py-8 text-gray-400">
                  <BarChart3 className="w-8 h-8 mx-auto mb-2" />
                  <p className="text-sm">Real order book data not available</p>
                  <p className="text-xs">Connect to trading platform for live data</p>
                </div>
              </div>
            </div>
          )}
          
          {/* Quick Actions */}
          <div className="bg-black/40 rounded-xl p-4 backdrop-blur-sm border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-semibold flex items-center">
                <Zap className="w-5 h-5 mr-2 text-yellow-400" />
                Quick Actions
              </h3>
              <button
                onClick={() => setShowOrderBook(!showOrderBook)}
                title="Toggle Order Book"
                aria-label="Toggle Order Book"
                className={`p-1 rounded transition-colors ${
                  showOrderBook ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
                }`}
              >
                <Eye className="w-4 h-4" />
              </button>
            </div>
            
            <div className="space-y-2">
              <button 
                onClick={() => alert(`Buy order for ${selectedSymbol} at $${currentData?.price.toFixed(2)}`)}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2"
              >
                <TrendingUp className="w-4 h-4" />
                <span>Buy {selectedSymbol}</span>
              </button>
              <button 
                onClick={() => alert(`Sell order for ${selectedSymbol} at $${currentData?.price.toFixed(2)}`)}
                className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2"
              >
                <TrendingDown className="w-4 h-4" />
                <span>Sell {selectedSymbol}</span>
              </button>
              <button 
                onClick={() => {
                  if (alertsEnabled) {
                    alert(`Price alert set for ${selectedSymbol} at $${currentData?.price.toFixed(2)}`);
                  }
                }}
                className={`w-full font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2 ${
                  alertsEnabled ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-gray-600 text-gray-400'
                }`}
              >
                <Target className="w-4 h-4" />
                <span>Set Alert</span>
              </button>
              <button className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm flex items-center justify-center space-x-2">
                <Activity className="w-4 h-4" />
                <span>Add to Watchlist</span>
              </button>
            </div>
            
            <div className="mt-4 pt-4 border-t border-gray-600">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Auto Refresh:</span>
                <button
                  onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    isAutoRefresh ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                >
                  {isAutoRefresh ? 'ON' : 'OFF'}
                </button>
              </div>
              <div className="flex items-center justify-between text-sm mt-2">
                <span className="text-gray-400">Alerts:</span>
                <button
                  onClick={() => setAlertsEnabled(!alertsEnabled)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    alertsEnabled ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                >
                  {alertsEnabled ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}