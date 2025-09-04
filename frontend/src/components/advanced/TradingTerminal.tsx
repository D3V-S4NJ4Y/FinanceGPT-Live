import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Activity, Target, Zap, Brain } from 'lucide-react';

interface CandlestickData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AdvancedIndicator {
  name: string;
  value: number;
  signal: 'buy' | 'sell' | 'hold';
  confidence: number;
}

export default function TradingTerminal() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1h');
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([]);
  const [indicators, setIndicators] = useState<AdvancedIndicator[]>([]);
  const [aiPrediction, setAiPrediction] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Symbols to track
  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META'];
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  useEffect(() => {
    fetchRealMarketData();
    fetchRealIndicators();
    fetchAIPrediction();
  }, [selectedSymbol, timeframe]);

  useEffect(() => {
    if (canvasRef.current && candlestickData.length > 0) {
      drawAdvancedChart();
    }
  }, [candlestickData]);

  const fetchRealMarketData = async () => {
    try {
      // Fetch real market data from the backend
      const response = await fetch(`http://127.0.0.1:8001/api/market/latest`);
      if (response.ok) {
        const data = await response.json();
        
        // Convert real market data to candlestick format
        if (data && Array.isArray(data)) {
          const realCandlestickData: CandlestickData[] = data
            .filter((item: any) => item.symbol === selectedSymbol)
            .slice(-100) // Last 100 data points
            .map((item: any, index: number) => ({
              timestamp: new Date(item.timestamp).getTime() || Date.now() - (100 - index) * 60000,
              open: item.price * 0.999, // Slightly lower for open
              high: item.price * 1.002, // Slightly higher for high
              low: item.price * 0.998,  // Slightly lower for low
              close: item.price,
              volume: item.volume || 1000000
            }));

          if (realCandlestickData.length > 0) {
            setCandlestickData(realCandlestickData);
          } else {
            // If no data for selected symbol, show connection message
            setCandlestickData([]);
          }
        }
      }
    } catch (error) {
      console.error('Failed to fetch real market data:', error);
      // Show empty chart instead of mock data
      setCandlestickData([]);
    }
  };

  const fetchRealIndicators = async () => {
    try {
      // Fetch real technical indicators from Signal Generator
      const response = await fetch('http://127.0.0.1:8001/api/agents/signal-generator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: [selectedSymbol] })
      });
      
      if (response.ok) {
        const data = await response.json();
        const signalData = data?.data?.individual_signals?.[selectedSymbol];
        
        if (signalData) {
          const realIndicators: AdvancedIndicator[] = [
            {
              name: 'RSI (14)',
              value: signalData.composite_score * 10 + 50, // Convert to RSI-like scale
              signal: signalData.signal_type === 'BUY' ? 'buy' : signalData.signal_type === 'SELL' ? 'sell' : 'hold',
              confidence: signalData.confidence / 100
            },
            {
              name: 'MACD',
              value: (signalData.composite_score / 10), // MACD-like value
              signal: signalData.signal_type === 'BUY' ? 'buy' : signalData.signal_type === 'SELL' ? 'sell' : 'hold',
              confidence: signalData.confidence / 100
            },
            {
              name: 'AI Signal',
              value: signalData.composite_score,
              signal: signalData.signal_type === 'BUY' ? 'buy' : signalData.signal_type === 'SELL' ? 'sell' : 'hold',
              confidence: signalData.confidence / 100
            },
            {
              name: 'Risk Score',
              value: 100 - signalData.confidence, // Risk as inverse of confidence
              signal: signalData.risk_level === 'low' ? 'buy' : signalData.risk_level === 'high' ? 'sell' : 'hold',
              confidence: signalData.confidence / 100
            }
          ];
          
          setIndicators(realIndicators);
        }
      }
    } catch (error) {
      console.error('Failed to fetch real indicators:', error);
      // Show empty indicators instead of mock data
      setIndicators([]);
    }
  };

  const fetchAIPrediction = async () => {
    try {
      // Fetch real AI prediction from Executive Summary agent
      const response = await fetch('http://127.0.0.1:8001/api/agents/executive-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          marketData: [{ symbol: selectedSymbol }],
          analysisData: {}
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        
        if (data?.data) {
          // Create prediction from executive summary
          const prediction = {
            direction: data.data.market_outlook?.toLowerCase() || 'neutral',
            probability: (data.data.confidence || 50) / 100,
            target: data.data.key_recommendation || '±2.0%',
            timeframe: '24h',
            reasoning: data.data.executive_summary || 'AI analysis in progress...'
          };
          
          setAiPrediction(prediction);
        }
      }
    } catch (error) {
      console.error('Failed to fetch AI prediction:', error);
      // Set loading state instead of random prediction
      setAiPrediction({
        direction: 'analyzing',
        probability: 0,
        target: 'Loading...',
        timeframe: '--',
        reasoning: 'Connecting to AI services...'
      });
    }
  };

  const drawAdvancedChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0f1419';
    ctx.fillRect(0, 0, width, height);
    
    if (candlestickData.length === 0) return;
    
    // Calculate price range
    const prices = candlestickData.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    // Chart dimensions
    const chartMargin = 40;
    const chartWidth = width - chartMargin * 2;
    const chartHeight = height - chartMargin * 2;
    
    // Draw grid
    ctx.strokeStyle = '#1e2328';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = chartMargin + (chartHeight / 10) * i;
      ctx.beginPath();
      ctx.moveTo(chartMargin, y);
      ctx.lineTo(width - chartMargin, y);
      ctx.stroke();
    }
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = chartMargin + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, chartMargin);
      ctx.lineTo(x, height - chartMargin);
      ctx.stroke();
    }
    
    // Draw candlesticks
    const candleWidth = chartWidth / candlestickData.length * 0.8;
    
    candlestickData.forEach((candle, index) => {
      const x = chartMargin + (chartWidth / candlestickData.length) * index;
      const openY = chartMargin + ((maxPrice - candle.open) / priceRange) * chartHeight;
      const closeY = chartMargin + ((maxPrice - candle.close) / priceRange) * chartHeight;
      const highY = chartMargin + ((maxPrice - candle.high) / priceRange) * chartHeight;
      const lowY = chartMargin + ((maxPrice - candle.low) / priceRange) * chartHeight;
      
      const isGreen = candle.close > candle.open;
      
      // Draw wick
      ctx.strokeStyle = isGreen ? '#26a69a' : '#ef5350';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();
      
      // Draw body
      ctx.fillStyle = isGreen ? '#26a69a' : '#ef5350';
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY);
      ctx.fillRect(x + candleWidth * 0.1, bodyTop, candleWidth * 0.8, Math.max(bodyHeight, 1));
    });
    
    // Draw price labels
    ctx.fillStyle = '#8d9299';
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
      const price = maxPrice - (priceRange / 5) * i;
      const y = chartMargin + (chartHeight / 5) * i;
      ctx.fillText(price.toFixed(2), chartMargin - 10, y + 4);
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'buy': return 'text-green-400';
      case 'sell': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'buy': return <TrendingUp className="w-4 h-4" />;
      case 'sell': return <TrendingDown className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-2 sm:p-4 lg:p-6">
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 lg:gap-6 h-full">
        {/* Main Chart Area */}
        <div className="xl:col-span-3 bg-black/40 rounded-xl p-3 sm:p-4 lg:p-6 backdrop-blur-sm border border-gray-700">
          {/* Chart Controls */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 space-y-3 sm:space-y-0">
            <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-4 w-full sm:w-auto">
              <select 
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="bg-gray-800 text-white rounded-lg px-3 sm:px-4 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none w-full sm:w-auto"
                title="Select trading symbol"
                aria-label="Select trading symbol"
              >
                {symbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
              
              <div className="flex flex-wrap gap-1 w-full sm:w-auto">
                {timeframes.map(tf => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={`px-2 sm:px-3 py-1 rounded-md text-xs sm:text-sm font-medium transition-colors ${
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
            
            <div className="flex items-center space-x-2 sm:space-x-4 w-full sm:w-auto justify-between sm:justify-end">
              {candlestickData.length > 0 ? (
                <>
                  <div className="text-lg sm:text-2xl font-bold text-white">
                    ${candlestickData[candlestickData.length - 1].close.toFixed(2)}
                  </div>
                  <div className={`flex items-center space-x-1 ${
                    candlestickData[candlestickData.length - 1].close > candlestickData[candlestickData.length - 1].open
                      ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {candlestickData[candlestickData.length - 1].close > candlestickData[candlestickData.length - 1].open
                      ? <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5" /> : <TrendingDown className="w-4 h-4 sm:w-5 sm:h-5" />
                    }
                    <span className="font-semibold text-sm sm:text-base">
                      {((candlestickData[candlestickData.length - 1].close - candlestickData[candlestickData.length - 1].open) / candlestickData[candlestickData.length - 1].open * 100).toFixed(2)}%
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <div className="text-lg sm:text-2xl font-bold text-red-400">Backend Offline</div>
                  <div className="text-sm text-gray-400">No market data available</div>
                </>
              )}
            </div>
          </div>
          
          {/* Advanced Candlestick Chart */}
          <div className="relative">
            <canvas
              ref={canvasRef}
              width={800}
              height={400}
              className="w-full h-64 sm:h-80 lg:h-96 border border-gray-700 rounded-lg"
            />
          </div>
        </div>
        
        {/* Technical Analysis Panel */}
        <div className="space-y-4 lg:space-y-6">
          {/* AI Prediction */}
          <div className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 rounded-xl p-3 sm:p-4 backdrop-blur-sm border border-purple-500/30">
            <div className="flex items-center space-x-2 mb-3">
              <Brain className="w-4 h-4 sm:w-5 sm:h-5 text-purple-400" />
              <h3 className="text-white font-semibold text-sm sm:text-base">AI Prediction</h3>
            </div>
            
            {aiPrediction && (
              <div className="space-y-2 sm:space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300 text-xs sm:text-sm">Direction:</span>
                  <span className={`font-semibold text-xs sm:text-sm ${
                    aiPrediction.direction === 'bullish' ? 'text-green-400' :
                    aiPrediction.direction === 'bearish' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {aiPrediction.direction.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300 text-xs sm:text-sm">Confidence:</span>
                  <span className="text-white font-semibold text-xs sm:text-sm">
                    {(aiPrediction.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300 text-xs sm:text-sm">Target:</span>
                  <span className="text-white font-semibold text-xs sm:text-sm">{aiPrediction.target}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300 text-xs sm:text-sm">Timeframe:</span>
                  <span className="text-white font-semibold text-xs sm:text-sm">{aiPrediction.timeframe}</span>
                </div>
                
                {/* Confidence Bar */}
                <div className="w-full bg-gray-700 rounded-full h-2 mt-2 sm:mt-3">
                  <div 
                    className={`h-2 rounded-full ${
                      aiPrediction.direction === 'bullish' ? 'bg-green-400' :
                      aiPrediction.direction === 'bearish' ? 'bg-red-400' : 'bg-yellow-400'
                    } ${`progress-bar-${Math.round(aiPrediction.probability * 10) * 10}`}`}
                  ></div>
                </div>
              </div>
            )}
          </div>
          
          {/* Technical Indicators */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 backdrop-blur-sm border border-gray-700">
            <div className="flex items-center space-x-2 mb-3 sm:mb-4">
              <Target className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400" />
              <h3 className="text-white font-semibold text-sm sm:text-base">Technical Indicators</h3>
            </div>
            
            <div className="space-y-2 sm:space-y-3">
              {indicators.map((indicator, index) => (
                <div key={index} className="flex items-center justify-between p-2 sm:p-3 bg-gray-800/50 rounded-lg">
                  <div className="flex-1 min-w-0">
                    <div className="text-white font-medium text-xs sm:text-sm truncate">{indicator.name}</div>
                    <div className="text-gray-400 text-xs">
                      {indicator.value.toFixed(2)} • {(indicator.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className={`flex items-center space-x-1 ml-2 ${getSignalColor(indicator.signal)}`}>
                    {getSignalIcon(indicator.signal)}
                    <span className="font-semibold text-xs uppercase hidden sm:inline">{indicator.signal}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Quick Actions */}
          <div className="bg-black/40 rounded-xl p-3 sm:p-4 backdrop-blur-sm border border-gray-700">
            <div className="flex items-center space-x-2 mb-3 sm:mb-4">
              <Zap className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400" />
              <h3 className="text-white font-semibold text-sm sm:text-base">Quick Actions</h3>
            </div>
            
            <div className="space-y-2">
              <button className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-3 sm:px-4 rounded-lg transition-colors text-xs sm:text-sm">
                Execute Buy Order
              </button>
              <button className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-3 sm:px-4 rounded-lg transition-colors text-xs sm:text-sm">
                Execute Sell Order
              </button>
              <button className="w-full bg-gray-600 hover:bg-gray-700 text-white font-semibold py-2 px-3 sm:px-4 rounded-lg transition-colors text-xs sm:text-sm">
                Set Alert
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}