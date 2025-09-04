import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { globalStore } from '../../store/globalStore';
import {
  Brain, Bell, BarChart3, CheckCircle, Loader, TrendingUp, TrendingDown,
  Activity, Target, Zap, Shield, AlertTriangle, RefreshCw, Play, Pause,
  DollarSign, Percent, Volume2, Clock, Eye, Settings, Filter, Search,
  ArrowUp, ArrowDown, Minus, ChevronRight, Star, Bookmark, Globe
} from 'lucide-react';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
  high_24h?: number;
  low_24h?: number;
  market_cap?: number;
  pe_ratio?: number;
}

interface MLPrediction {
  symbol: string;
  target_price: number;
  confidence: number;
  direction: string;
  risk_score?: number;
  time_horizon?: string;
  probability?: number;
  stop_loss?: number;
}

interface RealTimeAlert {
  id: string;
  type: 'prediction' | 'risk' | 'opportunity' | 'technical' | 'news';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  symbol?: string;
  timestamp: Date;
  action?: string;
}

interface SectorData {
  [sector: string]: {
    price: number;
    change_1d: number;
    change_5d: number;
    symbol: string;
  };
}

interface TechnicalIndicators {
  sma_20?: number;
  sma_50?: number;
  volatility?: number;
  rsi?: number;
  macd?: number;
}

const API_BASE_URL = 'http://localhost:8001';

const apiCall = async (endpoint: string, options: RequestInit = {}) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);
  
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      signal: controller.signal,
      ...options,
    });
    clearTimeout(timeoutId);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    console.error(`API ${endpoint} failed:`, error);
    return null;
  }
};

export default function SuperAdvancedDashboard() {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [predictions, setPredictions] = useState<Record<string, MLPrediction>>({});
  const [alerts, setAlerts] = useState<RealTimeAlert[]>([]);
  const [sectorData, setSectorData] = useState<SectorData>({});
  const [technicalData, setTechnicalData] = useState<Record<string, TechnicalIndicators>>({});
  const [loading, setLoading] = useState(true);
  const [isLive, setIsLive] = useState(true);
  const [lastUpdate, setLastUpdate] = useState('');
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']);
  const [filterType, setFilterType] = useState<'all' | 'bullish' | 'bearish' | 'high-confidence'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [performanceMetrics, setPerformanceMetrics] = useState({
    totalValue: 0,
    totalGain: 0,
    totalGainPercent: 0,
    winRate: 0,
    avgConfidence: 0,
    riskScore: 0
  });

  const symbols = useMemo(() => [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'ZOOM'
  ], []);

  const fetchMarketData = useCallback(async () => {
    try {
      const response = await apiCall('/api/market/latest');
      if (response && Array.isArray(response)) {
        setMarketData(response);
        return response;
      }
    } catch (error) {
      console.error('Market data fetch failed:', error);
    }
    return [];
  }, []);

  const fetchPredictions = useCallback(async () => {
    try {
      const response = await apiCall('/api/agents/signal-generator', {
        method: 'POST',
        body: JSON.stringify({ symbols: selectedSymbols, risk_tolerance: 'medium' })
      });
      
      if (response?.success && response.data?.signals) {
        const predictionMap: Record<string, MLPrediction> = {};
        response.data.signals.forEach((signal: any) => {
          predictionMap[signal.symbol] = {
            symbol: signal.symbol,
            target_price: signal.target_price || 0,
            confidence: signal.confidence || 0,
            direction: signal.action || 'HOLD',
            risk_score: signal.risk_score || 0,
            time_horizon: signal.time_horizon || '1D',
            probability: signal.probability || 0,
            stop_loss: signal.stop_loss || 0
          };
        });
        setPredictions(predictionMap);
        return predictionMap;
      }
    } catch (error) {
      console.error('Predictions fetch failed:', error);
    }
    return {};
  }, [selectedSymbols]);

  const fetchAlerts = useCallback(async () => {
    try {
      const response = await apiCall('/api/agents/alerts?limit=10');
      if (response?.success && response.data?.alerts) {
        const alertsData = response.data.alerts.map((alert: any) => ({
          id: alert.id || `alert-${Date.now()}-${Math.random()}`,
          type: alert.type || 'technical',
          severity: alert.severity || 'medium',
          message: alert.message || 'Market update',
          symbol: alert.symbol,
          timestamp: new Date(alert.timestamp || Date.now()),
          action: alert.action
        }));
        setAlerts(alertsData);
      }
    } catch (error) {
      console.error('Alerts fetch failed:', error);
    }
  }, []);

  const fetchSectorData = useCallback(async () => {
    try {
      const response = await apiCall('/api/market/sectors');
      if (response?.success && response.data) {
        setSectorData(response.data);
      }
    } catch (error) {
      console.error('Sector data fetch failed:', error);
    }
  }, []);

  const fetchTechnicalData = useCallback(async () => {
    try {
      const technicalMap: Record<string, TechnicalIndicators> = {};
      const promises = selectedSymbols.slice(0, 5).map(async (symbol) => {
        const response = await apiCall(`/api/market/history/${symbol}?period=1mo&interval=1d`);
        if (response?.success && response.data?.technical_indicators) {
          technicalMap[symbol] = response.data.technical_indicators;
        }
      });
      
      await Promise.all(promises);
      setTechnicalData(technicalMap);
    } catch (error) {
      console.error('Technical data fetch failed:', error);
    }
  }, [selectedSymbols]);

  const calculatePerformanceMetrics = useCallback((marketData: MarketData[], predictions: Record<string, MLPrediction>) => {
    const totalValue = marketData.reduce((sum, stock) => sum + stock.price, 0);
    const totalGain = marketData.reduce((sum, stock) => sum + stock.change, 0);
    const totalGainPercent = marketData.length > 0 ? 
      marketData.reduce((sum, stock) => sum + stock.changePercent, 0) / marketData.length : 0;
    
    const predictionValues = Object.values(predictions);
    const avgConfidence = predictionValues.length > 0 ?
      predictionValues.reduce((sum, p) => sum + p.confidence, 0) / predictionValues.length : 0;
    
    const winRate = predictionValues.length > 0 ?
      predictionValues.filter(p => p.direction === 'BUY' && p.confidence > 0.7).length / predictionValues.length : 0;
    
    const riskScore = predictionValues.length > 0 ?
      predictionValues.reduce((sum, p) => sum + (p.risk_score || 0), 0) / predictionValues.length : 0;

    setPerformanceMetrics({
      totalValue,
      totalGain,
      totalGainPercent,
      winRate,
      avgConfidence,
      riskScore
    });
  }, []);

  const fetchAllData = useCallback(async () => {
    if (!isLive) return;
    
    // Check if we have fresh data in global store
    const state = globalStore.getState();
    if (state.marketData.length > 0 && globalStore.isDataFresh()) {
      setMarketData(state.marketData);
      setLoading(false);
      setLastUpdate(new Date(state.lastUpdate).toLocaleTimeString());
      return;
    }
    
    setLoading(true);
    try {
      const [marketResponse, predictionsResponse] = await Promise.all([
        fetchMarketData(),
        fetchPredictions()
      ]);
      
      // Save to global store
      globalStore.setState({
        marketData: marketResponse,
        lastUpdate: new Date().toISOString(),
        isLoading: false
      });
      
      // Fetch additional data in parallel
      Promise.all([
        fetchAlerts(),
        fetchSectorData(),
        fetchTechnicalData()
      ]);
      
      calculatePerformanceMetrics(marketResponse, predictionsResponse);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Data fetch error:', error);
    } finally {
      setLoading(false);
    }
  }, [isLive, fetchMarketData, fetchPredictions, fetchAlerts, fetchSectorData, fetchTechnicalData, calculatePerformanceMetrics]);

  useEffect(() => {
    // Load persisted data first
    const state = globalStore.getState();
    if (state.marketData.length > 0) {
      setMarketData(state.marketData);
      setLoading(false);
      if (state.lastUpdate) {
        setLastUpdate(new Date(state.lastUpdate).toLocaleTimeString());
      }
    }
    
    fetchAllData();
    
    if (isLive) {
      const interval = setInterval(fetchAllData, 30000);
      return () => clearInterval(interval);
    }
  }, [fetchAllData, isLive]);

  const filteredPredictions = useMemo(() => {
    let filtered = Object.values(predictions);
    
    if (filterType === 'bullish') filtered = filtered.filter(p => p.direction === 'BUY');
    else if (filterType === 'bearish') filtered = filtered.filter(p => p.direction === 'SELL');
    else if (filterType === 'high-confidence') filtered = filtered.filter(p => p.confidence > 0.7);
    
    if (searchTerm) {
      filtered = filtered.filter(p => 
        p.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    return filtered.sort((a, b) => b.confidence - a.confidence);
  }, [predictions, filterType, searchTerm]);

  const getDirectionIcon = (direction: string, changePercent?: number) => {
    if (direction === 'BUY' || (changePercent && changePercent > 0)) {
      return <ArrowUp className="w-4 h-4 text-green-400" />;
    } else if (direction === 'SELL' || (changePercent && changePercent < 0)) {
      return <ArrowDown className="w-4 h-4 text-red-400" />;
    }
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-600 text-white border-red-500';
      case 'high': return 'bg-orange-600 text-white border-orange-500';
      case 'medium': return 'bg-yellow-600 text-white border-yellow-500';
      default: return 'bg-blue-600 text-white border-blue-500';
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (loading && marketData.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center">
        <div className="text-white text-xl flex items-center">
          <Loader className="w-8 h-8 mr-3 animate-spin text-blue-400" />
          <div>
            <div className="font-semibold">Loading AI Intelligence Dashboard</div>
            <div className="text-sm text-gray-400 mt-1">Fetching real-time market data...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between mb-6 space-y-4 lg:space-y-0">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
              <Brain className="w-8 h-8 mr-3 text-purple-400" />
              AI Intelligence Dashboard
            </h1>
            <div className="text-gray-400 text-sm">Real-time market analysis with advanced ML predictions</div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <input
                type="text"
                placeholder="Search symbols..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="px-3 py-2 bg-gray-800 text-white rounded-lg text-sm border border-gray-600 focus:border-blue-500 focus:outline-none"
              />
              <Search className="w-4 h-4 text-gray-400" />
            </div>
            
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="px-3 py-2 bg-gray-800 text-white rounded-lg text-sm border border-gray-600 focus:border-blue-500 focus:outline-none"
              title="Filter predictions by type"
            >
              <option value="all">All Signals</option>
              <option value="bullish">Bullish Only</option>
              <option value="bearish">Bearish Only</option>
              <option value="high-confidence">High Confidence</option>
            </select>
            
            <button
              onClick={() => setIsLive(!isLive)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                isLive ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isLive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isLive ? 'Pause' : 'Start'}</span>
            </button>
            
            <button
              onClick={fetchAllData}
              disabled={loading}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
            
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span>{isLive ? 'Live' : 'Paused'}</span>
              <span className="text-xs">• {lastUpdate}</span>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Market Value</p>
                <p className="text-lg font-bold text-white">{formatCurrency(performanceMetrics.totalValue)}</p>
              </div>
              <DollarSign className="w-6 h-6 text-green-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Total Gain</p>
                <p className={`text-lg font-bold ${performanceMetrics.totalGain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(performanceMetrics.totalGainPercent)}
                </p>
              </div>
              <Percent className="w-6 h-6 text-blue-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Win Rate</p>
                <p className="text-lg font-bold text-white">{(performanceMetrics.winRate * 100).toFixed(1)}%</p>
              </div>
              <Target className="w-6 h-6 text-purple-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Avg Confidence</p>
                <p className="text-lg font-bold text-white">{(performanceMetrics.avgConfidence * 100).toFixed(1)}%</p>
              </div>
              <Brain className="w-6 h-6 text-blue-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Risk Score</p>
                <p className={`text-lg font-bold ${
                  performanceMetrics.riskScore > 0.6 ? 'text-red-400' :
                  performanceMetrics.riskScore > 0.3 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {(performanceMetrics.riskScore * 100).toFixed(0)}%
                </p>
              </div>
              <Shield className="w-6 h-6 text-orange-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Active Signals</p>
                <p className="text-lg font-bold text-white">{filteredPredictions.length}</p>
              </div>
              <Activity className="w-6 h-6 text-cyan-400" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Market Data */}
          <div className="xl:col-span-1">
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mb-6">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-green-400" />
                Live Market Data ({marketData.length})
              </h3>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {marketData.slice(0, 10).map((stock) => (
                  <div key={stock.symbol} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg hover:bg-gray-700/50 transition-colors">
                    <div className="flex items-center space-x-3">
                      {getDirectionIcon('', stock.changePercent)}
                      <div>
                        <div className="font-medium text-white">{stock.symbol}</div>
                        <div className="text-xs text-gray-400">
                          Vol: {(stock.volume / 1000000).toFixed(1)}M
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-white font-semibold">{formatCurrency(stock.price)}</div>
                      <div className={`text-sm font-medium ${
                        stock.changePercent >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatPercent(stock.changePercent)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Sector Performance */}
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <Globe className="w-5 h-5 mr-2 text-blue-400" />
                Sector Performance
              </h3>
              
              <div className="space-y-2">
                {Object.entries(sectorData).slice(0, 6).map(([sector, data]) => (
                  <div key={sector} className="flex items-center justify-between p-2 bg-gray-800/30 rounded">
                    <div className="text-sm text-white truncate">{sector}</div>
                    <div className={`text-sm font-medium ${
                      data.change_1d >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPercent(data.change_1d)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* AI Predictions */}
          <div className="xl:col-span-1">
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-purple-400" />
                AI Predictions ({filteredPredictions.length})
              </h3>
              
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {filteredPredictions.slice(0, 8).map((prediction) => (
                  <div key={prediction.symbol} className="p-4 bg-gray-800/50 rounded-lg border border-gray-600 hover:border-gray-500 transition-colors">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <span className="text-white font-bold">{prediction.symbol}</span>
                        {getDirectionIcon(prediction.direction)}
                        <span className={`text-xs px-2 py-1 rounded ${
                          prediction.direction === 'BUY' ? 'bg-green-600 text-white' :
                          prediction.direction === 'SELL' ? 'bg-red-600 text-white' :
                          'bg-gray-600 text-white'
                        }`}>
                          {prediction.direction}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-white font-semibold">{formatCurrency(prediction.target_price)}</div>
                        <div className="text-xs text-gray-400">{prediction.time_horizon}</div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-300">Confidence:</span>
                        <span className="text-green-400 font-semibold">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-gray-300">Risk:</span>
                        <span className={`font-semibold ${
                          (prediction.risk_score || 0) > 0.6 ? 'text-red-400' :
                          (prediction.risk_score || 0) > 0.3 ? 'text-yellow-400' : 'text-green-400'
                        }`}>
                          {((prediction.risk_score ?? 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                      
                      {prediction.stop_loss && prediction.stop_loss > 0 && (
                        <div className="flex justify-between col-span-2">
                          <span className="text-gray-300">Stop Loss:</span>
                          <span className="text-red-400 font-semibold">{formatCurrency(prediction.stop_loss)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Alerts & Technical */}
          <div className="xl:col-span-1">
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mb-6">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-yellow-400" />
                Live Alerts ({alerts.length})
              </h3>
              
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {alerts.slice(0, 6).map((alert) => (
                  <div key={alert.id} className={`p-3 rounded-lg border ${getSeverityColor(alert.severity)}`}>
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {alert.type === 'prediction' && <Brain className="w-4 h-4" />}
                        {alert.type === 'risk' && <Shield className="w-4 h-4" />}
                        {alert.type === 'opportunity' && <Zap className="w-4 h-4" />}
                        {alert.type === 'technical' && <BarChart3 className="w-4 h-4" />}
                        {alert.type === 'news' && <Bell className="w-4 h-4" />}
                        <span className="font-semibold text-xs">{alert.type.toUpperCase()}</span>
                      </div>
                      <span className="text-xs opacity-70">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    
                    <div className="text-sm mb-2">{alert.message}</div>
                    
                    {alert.symbol && (
                      <div className="text-xs font-semibold mb-1">Symbol: {alert.symbol}</div>
                    )}
                    
                    {alert.action && (
                      <div className="text-xs italic opacity-80">Action: {alert.action}</div>
                    )}
                  </div>
                ))}
                
                {alerts.length === 0 && (
                  <div className="text-center text-gray-500 py-8">
                    <Eye className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <div>No alerts</div>
                    <div className="text-sm">Monitoring markets...</div>
                  </div>
                )}
              </div>
            </div>

            {/* Technical Indicators */}
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-cyan-400" />
                Technical Indicators
              </h3>
              
              <div className="space-y-3">
                {Object.entries(technicalData).slice(0, 4).map(([symbol, indicators]) => (
                  <div key={symbol} className="p-3 bg-gray-800/30 rounded-lg">
                    <div className="font-medium text-white mb-2">{symbol}</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {indicators.sma_20 && (
                        <div className="flex justify-between">
                          <span className="text-gray-300">SMA 20:</span>
                          <span className="text-blue-400">{formatCurrency(indicators.sma_20)}</span>
                        </div>
                      )}
                      {indicators.sma_50 && (
                        <div className="flex justify-between">
                          <span className="text-gray-300">SMA 50:</span>
                          <span className="text-purple-400">{formatCurrency(indicators.sma_50)}</span>
                        </div>
                      )}
                      {indicators.volatility && (
                        <div className="flex justify-between col-span-2">
                          <span className="text-gray-300">Volatility:</span>
                          <span className="text-yellow-400">{(indicators.volatility * 100).toFixed(2)}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}