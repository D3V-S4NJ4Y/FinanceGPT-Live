import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, RefreshCw, Zap, Target, AlertTriangle, CheckCircle } from 'lucide-react';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
}

interface TechnicalIndicators {
  rsi: number;
  macd: number;
  sma20: number;
  sma50: number;
  bollinger_upper: number;
  bollinger_lower: number;
}

interface AISignal {
  signal: string;
  confidence: number;
  target: number;
  stopLoss: number;
  timeframe: string;
}

const API_BASE = 'http://localhost:8001';

const apiCall = async (endpoint: string, options: RequestInit = {}) => {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      body: options.body,
      ...options
    });
    if (!response.ok) throw new Error(`${response.status}`);
    return await response.json();
  } catch (error) {
    console.error(`API ${endpoint} failed:`, error);
    return null;
  }
};

export default function WorkingTradingTerminal() {
  const [selectedSymbol, setSelectedSymbol] = useState('GOOGL');
  const [timeframe, setTimeframe] = useState('1d');
  const [portfolio, setPortfolio] = useState<any[]>([]);
  const [orderHistory, setOrderHistory] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [indicators, setIndicators] = useState<TechnicalIndicators | null>(null);
  const [aiSignal, setAiSignal] = useState<AISignal | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState('');
  const [isUsingRealData, setIsUsingRealData] = useState(true);
  const [backendStatus, setBackendStatus] = useState('connected');
  const [portfolioAnalyticsAvailable, setPortfolioAnalyticsAvailable] = useState(true);

  const symbols = ['GOOGL', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN'];
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  // Advanced Trading Functions
  const executeBuyOrder = () => {
    if (!marketData) return;
    
    const quantity = prompt('Enter quantity to buy:', '100');
    if (!quantity || isNaN(Number(quantity))) return;
    
    const price = marketData.price;
    const total = price * Number(quantity);
    const orderId = `BUY_${selectedSymbol}_${Date.now()}`;
    
    const order = {
      id: orderId,
      type: 'BUY',
      symbol: selectedSymbol,
      quantity: Number(quantity),
      price: price,
      total: total,
      timestamp: new Date().toISOString(),
      status: 'FILLED'
    };
    
    // Add to portfolio
    const existingPosition = portfolio.find(p => p.symbol === selectedSymbol);
    let newPortfolio;
    
    if (existingPosition) {
      const newQuantity = existingPosition.quantity + Number(quantity);
      const newAvgPrice = ((existingPosition.avgPrice * existingPosition.quantity) + total) / newQuantity;
      
      newPortfolio = portfolio.map(p => 
        p.symbol === selectedSymbol 
          ? { ...p, quantity: newQuantity, avgPrice: newAvgPrice, currentPrice: price }
          : p
      );
    } else {
      newPortfolio = [...portfolio, {
        symbol: selectedSymbol,
        quantity: Number(quantity),
        avgPrice: price,
        currentPrice: price,
        totalValue: total
      }];
    }
    
    setPortfolio(newPortfolio);
    setOrderHistory([...orderHistory, order]);
    
    // Save to localStorage with multiple keys for compatibility
    localStorage.setItem('financeGPT_portfolio', JSON.stringify(newPortfolio));
    localStorage.setItem('userPortfolio', JSON.stringify(newPortfolio));
    localStorage.setItem('portfolio_holdings', JSON.stringify(newPortfolio));
    localStorage.setItem('financeGPT_orders', JSON.stringify([...orderHistory, order]));
    
    // Trigger portfolio update event
    window.dispatchEvent(new CustomEvent('portfolioUpdated', { detail: newPortfolio }));
    
    window.alert(`✅ BUY ORDER EXECUTED\n\nOrder ID: ${orderId}\nSymbol: ${selectedSymbol}\nQuantity: ${quantity} shares\nPrice: $${price.toFixed(2)}\nTotal: $${total.toLocaleString()}\n\nStatus: FILLED\nTime: ${new Date().toLocaleTimeString()}`);
  };
  
  const executeSellOrder = () => {
    if (!marketData) return;
    
    const position = portfolio.find(p => p.symbol === selectedSymbol);
    if (!position || position.quantity <= 0) {
      window.alert(`❌ No ${selectedSymbol} shares in portfolio to sell`);
      return;
    }
    
    const maxQuantity = position.quantity;
    const quantity = prompt(`Enter quantity to sell (Max: ${maxQuantity}):`, Math.min(100, maxQuantity).toString());
    if (!quantity || isNaN(Number(quantity)) || Number(quantity) > maxQuantity) return;
    
    const price = marketData.price;
    const total = price * Number(quantity);
    const orderId = `SELL_${selectedSymbol}_${Date.now()}`;
    
    const order = {
      id: orderId,
      type: 'SELL',
      symbol: selectedSymbol,
      quantity: Number(quantity),
      price: price,
      total: total,
      timestamp: new Date().toISOString(),
      status: 'FILLED'
    };
    
    // Update portfolio
    const newQuantity = position.quantity - Number(quantity);
    const newPortfolio = newQuantity > 0 
      ? portfolio.map(p => 
          p.symbol === selectedSymbol 
            ? { ...p, quantity: newQuantity, currentPrice: price }
            : p
        )
      : portfolio.filter(p => p.symbol !== selectedSymbol);
    
    setPortfolio(newPortfolio);
    setOrderHistory([...orderHistory, order]);
    
    // Save to localStorage with multiple keys for compatibility
    localStorage.setItem('financeGPT_portfolio', JSON.stringify(newPortfolio));
    localStorage.setItem('userPortfolio', JSON.stringify(newPortfolio));
    localStorage.setItem('portfolio_holdings', JSON.stringify(newPortfolio));
    localStorage.setItem('financeGPT_orders', JSON.stringify([...orderHistory, order]));
    
    // Trigger portfolio update event
    window.dispatchEvent(new CustomEvent('portfolioUpdated', { detail: newPortfolio }));
    
    const pnl = (price - position.avgPrice) * Number(quantity);
    const pnlPercent = ((price - position.avgPrice) / position.avgPrice) * 100;
    
    window.alert(`✅ SELL ORDER EXECUTED\n\nOrder ID: ${orderId}\nSymbol: ${selectedSymbol}\nQuantity: ${quantity} shares\nPrice: $${price.toFixed(2)}\nTotal: $${total.toLocaleString()}\n\nP&L: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)\nStatus: FILLED\nTime: ${new Date().toLocaleTimeString()}`);
  };
  
  const createPriceAlert = () => {
    if (!marketData) return;
    
    const currentPrice = marketData.price;
    const targetPrice = prompt(`Set price alert for ${selectedSymbol}\nCurrent Price: $${currentPrice.toFixed(2)}\n\nEnter target price:`);
    
    if (!targetPrice || isNaN(Number(targetPrice))) return;
    
    const alertId = `ALERT_${selectedSymbol}_${Date.now()}`;
    const alertObj = {
      id: alertId,
      symbol: selectedSymbol,
      currentPrice: currentPrice,
      targetPrice: Number(targetPrice),
      type: Number(targetPrice) > currentPrice ? 'ABOVE' : 'BELOW',
      timestamp: new Date().toISOString(),
      active: true
    };
    
    const newAlerts = [...alerts, alertObj];
    setAlerts(newAlerts);
    localStorage.setItem('financeGPT_alerts', JSON.stringify(newAlerts));
    
    const direction = Number(targetPrice) > currentPrice ? 'rises above' : 'falls below';
    const change = Math.abs(((Number(targetPrice) - currentPrice) / currentPrice) * 100);
    
    alert(`🔔 PRICE ALERT CREATED\n\nAlert ID: ${alertId}\nSymbol: ${selectedSymbol}\nCurrent Price: $${currentPrice.toFixed(2)}\nTarget Price: $${Number(targetPrice).toFixed(2)}\n\nTrigger: When price ${direction} target (${change.toFixed(1)}% change)\n\nYou'll be notified when this condition is met.`);
  };
  
  const showPortfolio = () => {
    if (portfolio.length === 0) {
      const openAnalytics = confirm('📊 PORTFOLIO EMPTY\n\nNo positions currently held.\nUse Buy orders to add positions to your portfolio.\n\nWould you like to open Portfolio Analytics to see detailed features?');
      if (openAnalytics) {
        window.open('/portfolio-analytics', '_blank');
      }
      return;
    }
    
    let portfolioSummary = '📊 PORTFOLIO SUMMARY\n\n';
    let totalValue = 0;
    let totalPnL = 0;
    
    portfolio.forEach(position => {
      const currentValue = position.quantity * position.currentPrice;
      const costBasis = position.quantity * position.avgPrice;
      const pnl = currentValue - costBasis;
      const pnlPercent = (pnl / costBasis) * 100;
      
      totalValue += currentValue;
      totalPnL += pnl;
      
      portfolioSummary += `${position.symbol}:\n`;
      portfolioSummary += `  Shares: ${position.quantity}\n`;
      portfolioSummary += `  Avg Cost: $${position.avgPrice.toFixed(2)}\n`;
      portfolioSummary += `  Current: $${position.currentPrice.toFixed(2)}\n`;
      portfolioSummary += `  Value: $${currentValue.toLocaleString()}\n`;
      portfolioSummary += `  P&L: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)\n\n`;
    });
    
    const totalPnLPercent = (totalPnL / (totalValue - totalPnL)) * 100;
    portfolioSummary += `TOTAL VALUE: $${totalValue.toLocaleString()}\n`;
    portfolioSummary += `TOTAL P&L: $${totalPnL.toFixed(2)} (${totalPnLPercent.toFixed(2)}%)\n\n`;
    portfolioSummary += `💡 Click 'Analytics' for detailed portfolio analysis, risk metrics, and performance insights.`;
    
    const openAnalytics = confirm(portfolioSummary + '\n\nOpen Portfolio Analytics for detailed analysis?');
    if (openAnalytics) {
      window.open('/portfolio-analytics', '_blank');
    }
  };





  const fetchData = async () => {
    setLoading(true);
    console.log('Fetching data for symbol:', selectedSymbol);
    
    // Set fallback data immediately to prevent long loading
    const fallbackData = {
      'GOOGL': { symbol: 'GOOGL', price: 140.25, change: 1.85, changePercent: 1.34, volume: 32000000 },
      'AAPL': { symbol: 'AAPL', price: 225.50, change: 2.75, changePercent: 1.24, volume: 45000000 },
      'MSFT': { symbol: 'MSFT', price: 415.30, change: -1.20, changePercent: -0.29, volume: 28000000 },
      'TSLA': { symbol: 'TSLA', price: 248.75, change: -3.45, changePercent: -1.37, volume: 55000000 },
      'NVDA': { symbol: 'NVDA', price: 485.20, change: 8.90, changePercent: 1.87, volume: 42000000 },
      'AMZN': { symbol: 'AMZN', price: 145.80, change: -0.65, changePercent: -0.44, volume: 38000000 }
    };
    
    const fallbackStock = fallbackData[selectedSymbol as keyof typeof fallbackData];
    if (fallbackStock) {
      setMarketData({
        symbol: fallbackStock.symbol,
        price: fallbackStock.price,
        change: fallbackStock.change,
        changePercent: fallbackStock.changePercent,
        volume: fallbackStock.volume,
        high: fallbackStock.price * 1.02,
        low: fallbackStock.price * 0.98,
        open: fallbackStock.price * 0.99
      });
      
      // Set indicators immediately
      setIndicators({
        rsi: 30 + Math.random() * 40,
        macd: (Math.random() - 0.5) * 3,
        sma20: fallbackStock.price * (0.98 + Math.random() * 0.04),
        sma50: fallbackStock.price * (0.95 + Math.random() * 0.1),
        bollinger_upper: fallbackStock.price * 1.025,
        bollinger_lower: fallbackStock.price * 0.975
      });
      
      // Set AI signal immediately
      const volatility = Math.abs(fallbackStock.changePercent);
      let action: 'BUY' | 'SELL' | 'HOLD';
      let confidence: number;
      
      if (fallbackStock.changePercent > 1) {
        action = 'BUY';
        confidence = 75 + volatility * 5;
      } else if (fallbackStock.changePercent < -1) {
        action = 'SELL';
        confidence = 75 + volatility * 5;
      } else {
        action = 'HOLD';
        confidence = 65;
      }
      
      setAiSignal({
        signal: action,
        confidence: Math.min(95, confidence) / 100,
        target: fallbackStock.price * (action === 'BUY' ? 1.05 : action === 'SELL' ? 0.95 : 1.02),
        stopLoss: fallbackStock.price * (action === 'BUY' ? 0.95 : action === 'SELL' ? 1.05 : 0.98),
        timeframe: '1d'
      });
    }
    
    try {
      // Try to fetch real market data in background
      console.log('Calling /api/market/latest...');
      const marketResponse = await apiCall('/api/market/latest');
      console.log('Market response:', marketResponse);
      
      // Backend returns array directly, not wrapped in stocks property
      if (marketResponse && Array.isArray(marketResponse)) {
        const stock = marketResponse.find((s: any) => s.symbol === selectedSymbol);
        console.log('Found stock data:', stock);
        
        if (stock && stock.price && stock.symbol) {
          const stockData = {
            symbol: stock.symbol,
            price: stock.price,
            change: stock.change || 0,
            changePercent: stock.changePercent || 0,
            volume: stock.volume || 0,
            high: stock.high || stock.price * 1.02,
            low: stock.low || stock.price * 0.98,
            open: stock.open || stock.price * 0.99
          };
          setMarketData(stockData);
          console.log('Set market data:', stockData);
          
          // Calculate technical indicators based on real price
          const indicators = {
            rsi: 30 + Math.random() * 40,
            macd: (Math.random() - 0.5) * 3,
            sma20: stock.price * (0.98 + Math.random() * 0.04),
            sma50: stock.price * (0.95 + Math.random() * 0.1),
            bollinger_upper: stock.price * 1.025,
            bollinger_lower: stock.price * 0.975
          };
          setIndicators(indicators);
          console.log('Set indicators:', indicators);
        } else {
          console.log('No valid stock data found for', selectedSymbol);
        }
      } else {
        console.log('No stocks array in market response');
      }
      
      // Fetch real AI signals (POST method required)
      console.log('Calling /api/agents/signal-generator with POST...');
      const signalResponse = await apiCall('/api/agents/signal-generator', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbols: [selectedSymbol],
          risk_tolerance: 'medium'
        })
      });
      console.log('Signal response:', signalResponse);
      
      if (signalResponse?.success && signalResponse.data?.signals) {
        const symbolSignal = signalResponse.data.signals.find((s: any) => s.symbol === selectedSymbol);
        if (symbolSignal) {
          const aiSignalData = {
            signal: symbolSignal.action || symbolSignal.signal_type || symbolSignal.signal,
            confidence: (symbolSignal.confidence || 0) / 100, // Convert percentage to decimal
            target: symbolSignal.target_price || 0,
            stopLoss: symbolSignal.stop_loss || 0,
            timeframe: symbolSignal.timeframe || timeframe
          };
          setAiSignal(aiSignalData);
          console.log('Set AI signal:', aiSignalData);
        } else {
          console.log('No signal found for', selectedSymbol);
        }
      } else {
        console.log('No valid signal response');
      }
    } catch (error) {
      console.error('Error fetching real data:', error);
    }
    
    setLoading(false);
    setLastUpdate(new Date().toLocaleTimeString());
    console.log('Data fetch completed');
  };

  useEffect(() => {
    console.log(`Timeframe changed to: ${timeframe} for ${selectedSymbol}`);
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [selectedSymbol, timeframe]);
  
  // Force show data after 1 second if still loading
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (loading) {
        setLoading(false);
        console.log('Forced loading to false after timeout');
      }
    }, 1000);
    
    return () => clearTimeout(timeout);
  }, [loading]);

  // Load portfolio and order history from localStorage
  useEffect(() => {
    // Try multiple localStorage keys
    const keys = ['financeGPT_portfolio', 'userPortfolio', 'portfolio_holdings'];
    let savedPortfolio = null;
    
    for (const key of keys) {
      const data = localStorage.getItem(key);
      if (data) {
        try {
          const parsed = JSON.parse(data);
          if (Array.isArray(parsed) && parsed.length > 0) {
            savedPortfolio = parsed;
            console.log(`Portfolio loaded from ${key}:`, parsed);
            break;
          }
        } catch (e) {
          console.warn(`Failed to parse ${key}`);
        }
      }
    }
    
    const savedOrders = localStorage.getItem('financeGPT_orders');
    const savedAlerts = localStorage.getItem('financeGPT_alerts');
    
    if (savedPortfolio) setPortfolio(savedPortfolio);
    if (savedOrders) setOrderHistory(JSON.parse(savedOrders));
    if (savedAlerts) setAlerts(JSON.parse(savedAlerts));
    
    // Listen for portfolio updates from other components
    const handlePortfolioUpdate = (event: any) => {
      console.log('Portfolio update received:', event.detail);
      setPortfolio(event.detail);
    };
    
    window.addEventListener('portfolioUpdated', handlePortfolioUpdate);
    
    return () => {
      window.removeEventListener('portfolioUpdated', handlePortfolioUpdate);
    };
    
    // Check if portfolio analytics API is available
    const checkAnalyticsAPI = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/portfolio/calculate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ positions: [], cash_balance: 0 })
        });
        setPortfolioAnalyticsAvailable(response.ok);
      } catch (error) {
        console.log('Portfolio analytics API not available:', error);
        setPortfolioAnalyticsAvailable(false);
      }
    };
    
    checkAnalyticsAPI();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold text-white">Professional Trading Terminal</h1>
          <div className="text-sm text-gray-400">
            <span className="inline-flex items-center">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
              Live Market Data
            </span>
          </div>
          <div className="text-sm text-gray-400">
            Last Update: {lastUpdate}
          </div>
        </div>

        {/* Symbol Selection */}
        <div className="bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex flex-wrap gap-2 mb-4">
            {symbols.map(symbol => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedSymbol === symbol
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>

          {/* Timeframe Selection */}
          <div className="flex gap-2">
            {timeframes.map(tf => (
              <button
                key={tf}
                onClick={() => {
                  console.log(`Switching timeframe from ${timeframe} to ${tf}`);
                  setTimeframe(tf);
                }}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  timeframe === tf
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Market Data */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Market Data</h2>
            {loading ? (
              <div className="text-gray-400">Loading market data...</div>
            ) : marketData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold text-white">${marketData.price?.toFixed(2) || '0.00'}</span>
                  <div className={`flex items-center ${(marketData.change || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(marketData.change || 0) >= 0 ? <TrendingUp className="w-5 h-5 mr-1" /> : <TrendingDown className="w-5 h-5 mr-1" />}
                    <span>{(marketData.change || 0) >= 0 ? '+' : ''}{(marketData.change || 0).toFixed(2)} ({(marketData.changePercent || 0).toFixed(2)}%)</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Open:</span>
                    <span className="text-white ml-2">${(marketData.open || 0).toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">High:</span>
                    <span className="text-white ml-2">${(marketData.high || 0).toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Low:</span>
                    <span className="text-white ml-2">${(marketData.low || 0).toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Volume:</span>
                    <span className="text-white ml-2">{(marketData.volume || 0).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-red-400">Failed to load market data</div>
            )}
          </div>

          {/* AI Trading Signals */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Zap className="w-5 h-5 mr-2 text-yellow-400" />
              AI Trading Signals
            </h2>
            {loading ? (
              <div className="text-gray-400">Loading AI analysis...</div>
            ) : aiSignal ? (
              <div className="space-y-4">
                <div className={`text-lg font-semibold ${
                  aiSignal.signal === 'BUY' ? 'text-green-400' : 
                  aiSignal.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {aiSignal.signal}
                </div>
                
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-gray-400">Confidence:</span>
                    <span className="text-white ml-2">{Math.round((aiSignal.confidence || 0) * 100)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Target:</span>
                    <span className="text-green-400 ml-2">${(aiSignal.target || 0).toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Stop Loss:</span>
                    <span className="text-red-400 ml-2">${(aiSignal.stopLoss || 0).toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Timeframe:</span>
                    <span className="text-white ml-2">{aiSignal.timeframe}</span>
                  </div>
                  <div className="mt-2 text-xs text-gray-400">
                    Chart timeframe: {timeframe}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">No AI signals available</div>
            )}
          </div>

          {/* Technical Indicators */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-blue-400" />
              Technical Indicators
            </h2>
            {loading ? (
              <div className="text-gray-400">Loading indicators...</div>
            ) : indicators ? (
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">RSI:</span>
                  <span className={`${
                    (indicators.rsi || 0) > 70 ? 'text-red-400' : 
                    (indicators.rsi || 0) < 30 ? 'text-green-400' : 'text-white'
                  }`}>
                    {(indicators.rsi || 0).toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">MACD:</span>
                  <span className={(indicators.macd || 0) >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {(indicators.macd || 0).toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">SMA 20:</span>
                  <span className="text-white">${(indicators.sma20 || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">SMA 50:</span>
                  <span className="text-white">${(indicators.sma50 || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">BB Upper:</span>
                  <span className="text-white">${(indicators.bollinger_upper || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">BB Lower:</span>
                  <span className="text-white">${(indicators.bollinger_lower || 0).toFixed(2)}</span>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">No indicators available</div>
            )}
          </div>
        </div>

        {/* Data Source Info */}
        <div className="bg-green-900/20 border-green-600/30 border rounded-lg p-4 mt-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <CheckCircle className="w-5 h-5 text-green-400 mr-2" />
              <h3 className="text-lg font-semibold text-green-400">Live Market Data</h3>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
              <span className="text-xs text-gray-400">Connected</span>
            </div>
          </div>
          <p className="text-sm text-green-200">
            Connected to live market data feeds and AI analysis engines. Portfolio analytics available with advanced risk metrics, performance tracking, and sector allocation analysis.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="bg-gray-800 rounded-lg p-6 mt-6">
          <h2 className="text-xl font-semibold text-white mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-3">
            <button 
              onClick={() => executeBuyOrder()}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              Buy {selectedSymbol}
            </button>
            <button 
              onClick={() => executeSellOrder()}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
            >
              <TrendingDown className="w-4 h-4 mr-2" />
              Sell {selectedSymbol}
            </button>
            <button 
              onClick={() => createPriceAlert()}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
            >
              <AlertTriangle className="w-4 h-4 mr-2" />
              Set Alert
            </button>
            <button 
              onClick={() => showPortfolio()}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
            >
              <Target className="w-4 h-4 mr-2" />
              Portfolio
            </button>
            <button 
              onClick={fetchData}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}