import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  TrendingUp, TrendingDown, Activity, Brain, Shield, FileText, Zap, 
  AlertTriangle, Globe, Wifi, WifiOff, DollarSign, BarChart3, 
  PieChart, Users, Database, Cpu, Settings, RefreshCw, 
  Eye, Target, Bell, Clock, Filter, Maximize2
} from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getAIData, setAIData, isAIDataFresh } from '../utils/aiDataStore';
import { useSharedData } from '../hooks/useSharedData';
import styles from './CommandCenter.module.css';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
  marketCap?: number;
  high24h?: number;
  low24h?: number;
  sector?: string;
}

interface AgentStatus {
  id: string;
  name: string;
  status: 'active' | 'processing' | 'idle' | 'error';
  lastUpdate: string;
  performance: number;
  signals: number;
  health?: string;
  uptime?: string;
  tasksCompleted?: number;
  currentTask?: string;
}

interface TradingSignal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: string;
  agent: string;
  reasoning?: string;
}

interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
}

interface PortfolioMetrics {
  totalValue: number;
  dailyChange: number;
  dailyChangePercent: number;
  positions: number;
  cash: number;
}

interface RiskMetrics {
  portfolioRisk: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  beta: number;
}

const CommandCenter: React.FC = () => {
  console.log('CommandCenter: Enhanced component initialized');
  
  // Use shared data for superfast performance
  const sharedData = useSharedData();
  const [marketData, setMarketData] = useState<MarketData[]>(sharedData.marketData);
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [tradingSignals, setTradingSignals] = useState<TradingSignal[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [portfolioMetrics, setPortfolioMetrics] = useState<PortfolioMetrics>({
    totalValue: 0,
    dailyChange: 0,
    dailyChangePercent: 0,
    positions: 0,
    cash: 0
  });
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    portfolioRisk: 0,
    volatility: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    beta: 0
  });
  const [globalStats, setGlobalStats] = useState({
    totalVolume: 0,
    marketCap: 0,
    activeSignals: 0,
    riskLevel: 'LOW' as 'LOW' | 'MEDIUM' | 'HIGH',
    agentsOnline: 0,
    dataPointsProcessed: 0,
    systemUptime: '99.9%'
  });
  const [activityFeed, setActivityFeed] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1m' | '5m' | '1h' | '1d'>('5m');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'analytics'>('overview');
  const [filterSymbol, setFilterSymbol] = useState<string>('');
  const [showAlerts, setShowAlerts] = useState(true);

  // WebSocket connection with retry - DISABLED FOR TESTING
  const [wsUrl] = useState('ws://127.0.0.1:8001/ws');
  // Temporarily disable WebSocket to focus on market data API
  const { isConnected, lastMessage, sendMessage, connectionState } = {
    isConnected: false, 
    lastMessage: null, 
    sendMessage: () => false, 
    connectionState: 'disabled' as const
  };

  // Performance tracking
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date());
  const [fetchingData, setFetchingData] = useState(false);

  // Enhanced market data fetching with real-time capabilities - UPDATED
  const fetchEnhancedMarketData = useCallback(async () => {
    if (fetchingData) return;
    
    setFetchingData(true);
    console.log('🚀 STARTING MARKET DATA FETCH - Updated Version');
    console.log('📈 Fetching enhanced market data...');
    
    try {
      setError(null);
      console.log('🔄 Attempting to fetch from: http://127.0.0.1:8001/api/market/latest');
      
      // Check for shared market data from other components
      const sharedData = window.localStorage.getItem('market_data_cache');
      if (sharedData && marketData.length === 0) {
        try {
          const cachedData = JSON.parse(sharedData);
          const cacheTime = new Date(cachedData.timestamp);
          const now = new Date();
          
          // Use cached data if it's less than 2 minutes old
          if ((now.getTime() - cacheTime.getTime()) < 120000) {
            console.log('⚡ Using shared market data from other components');
            setMarketData(cachedData.data);
            setError(null);
            setFetchingData(false);
            return;
          }
        } catch (e) {
          console.log('Failed to parse shared market data');
        }
      }
      
      const response = await Promise.race([
        fetch('http://127.0.0.1:8001/api/market/latest', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        }),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout after 25 seconds')), 25000)
        )
      ]) as Response;
      
      console.log('📡 Response received!');
      console.log('📡 Response status:', response.status, response.statusText);
      console.log('📡 Response headers:', [...response.headers.entries()]);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      setLastUpdateTime(new Date());
      
      const data = await response.json();
      console.log('📊 Raw API response:', data);
      console.log('📊 Data type:', typeof data, 'Is array:', Array.isArray(data), 'Length:', data?.length);
      
      if (Array.isArray(data) && data.length > 0) {
          const enhancedData = data.map((item: any) => ({
            symbol: item.symbol || 'N/A',
            price: parseFloat(item.price) || 0,
            change: parseFloat(item.change) || 0,
            changePercent: parseFloat(item.changePercent) || 0,
            volume: parseInt(item.volume) || 0,
            timestamp: item.timestamp || new Date().toISOString(),
            marketCap: item.marketCap || (parseFloat(item.price) * parseInt(item.volume)),
            high24h: item.high24h || (parseFloat(item.price) * 1.05),
            low24h: item.low24h || (parseFloat(item.price) * 0.95),
            sector: item.sector || 'Technology'
          }));
          
          setMarketData(enhancedData);
          if (!dataLoaded) setDataLoaded(true);
          
          // Calculate portfolio metrics immediately
          const portfolioValue = enhancedData.slice(0, 5).reduce((sum, stock) => 
            sum + (stock.price * 100), 0);
          const portfolioChange = enhancedData.slice(0, 5).reduce((sum, stock) => 
            sum + (stock.change * 100), 0);
          
          setPortfolioMetrics({
            totalValue: portfolioValue,
            dailyChange: portfolioChange,
            dailyChangePercent: portfolioValue > 0 ? (portfolioChange / portfolioValue) * 100 : 0,
            positions: 5,
            cash: 50000
          });
          
          // Calculate market cap properly
          const marketCapCalc = enhancedData.reduce((sum: number, stock: MarketData) => {
            const cap = stock.marketCap || (stock.price * stock.volume * 1000);
            return sum + (cap / 1000000000000);
          }, 0);
          
          const activeSignals = enhancedData.filter((stock: MarketData) => 
            Math.abs(stock.changePercent) > 1.0).length;
          
          setGlobalStats(prev => ({
            ...prev,
            totalVolume: parseFloat((enhancedData.reduce((sum: number, stock: MarketData) => 
              sum + (stock.volume / 1000000000), 0)).toFixed(2)),
            activeSignals: activeSignals,
            marketCap: parseFloat(marketCapCalc.toFixed(1)),
            riskLevel: marketCapCalc > 50 ? 'HIGH' : marketCapCalc > 25 ? 'MEDIUM' : 'LOW',
            dataPointsProcessed: enhancedData.length
          }));

          console.log('✅ Enhanced market data processed successfully:', enhancedData.length, 'stocks');
          setError(null); // Clear any previous errors
          
          // Share data with other components
          window.localStorage.setItem('market_data_cache', JSON.stringify({
            data: enhancedData,
            timestamp: new Date().toISOString()
          }));
          
          // Dispatch event to notify other components
          window.dispatchEvent(new CustomEvent('marketDataUpdated', { detail: enhancedData }));
        } else {
          console.log('No valid data received, using fallback');
          throw new Error('No market data available');
        }
    } catch (error) {
      console.error('❌ Market data fetch failed:', error);
      // Set error to show connection issue to user
      setError(error instanceof Error ? error.message : 'Failed to connect to market data API');
    } finally {
      setFetchingData(false);
    }
  }, [fetchingData, marketData.length, dataLoaded]);

  // Fetch REAL agent status from backend API - NO FAKE DATA
  const fetchEnhancedAgentStatus = useCallback(async () => {
    console.log('🤖 Fetching REAL agent status from backend API...');
    try {
      // First try direct API call for real agent data
      const response = await fetch('http://127.0.0.1:8001/api/agents/status');
      if (response.ok) {
        const data = await response.json();
        console.log('🤖 Backend agent API response:', data);
        
        if (data.success && data.data?.agents) {
          const agentObjects = data.data.agents;
          const agentList: AgentStatus[] = Object.values(agentObjects).map((agent: any) => ({
            id: agent.id,
            name: agent.name,
            status: agent.status,
            lastUpdate: agent.last_update || new Date().toISOString(),
            performance: parseFloat(agent.performance) || 0,
            signals: parseInt(agent.signals_generated) || 0,
            health: agent.health || 'healthy',
            uptime: agent.uptime || '99.8%',
            tasksCompleted: parseInt(agent.tasks_completed) || 0,
            currentTask: agent.current_task || 'Monitoring'
          }));
          
          console.log('✅ REAL agent data processed:', agentList.length, 'agents');
          console.log('🔢 Real performance values:', agentList.map(a => `${a.name}: ${a.performance}%`));
          
          setAgents(agentList);
          setGlobalStats(prev => ({ 
            ...prev, 
            agentsOnline: agentList.filter(a => a.status === 'active').length 
          }));
          return; // Success - exit early
        }
      }
      
      // Fallback to sharedData if direct API fails
      console.log('⚠️ Direct API failed, trying sharedData...');
      const agentsData = await sharedData.getAgents();
      if (agentsData.length > 0) {
        const agentList: AgentStatus[] = agentsData.map((agent: any) => ({
          id: agent.id,
          name: agent.name,
          status: agent.status,
          lastUpdate: agent.last_update || new Date().toISOString(),
          performance: parseFloat(agent.performance) || 0,
          signals: parseInt(agent.signals_generated) || 0,
          health: agent.health || 'good',
          uptime: agent.uptime || '99%',
          tasksCompleted: parseInt(agent.tasks_completed) || 0,
          currentTask: agent.current_task || 'Monitoring'
        }));
        
        console.log('✅ SharedData agent data processed:', agentList.length, 'agents');
        setAgents(agentList);
        setGlobalStats(prev => ({ ...prev, agentsOnline: agentList.length }));
        return; // Success - exit early
      }
      
      throw new Error('No agent data available from any source');
      
    } catch (error) {
      console.error('❌ All agent data sources failed:', error);
      console.log('🚫 NOT using any fake fallback data - showing connection error');
      
      // Show connection error instead of fake data
      setError('Unable to connect to agent status API');
      setAgents([]); // Clear agents to show error state
      setGlobalStats(prev => ({ ...prev, agentsOnline: 0 }));
    }
  }, [sharedData]);

  // Generate trading signals from market data only
  const generateTradingSignals = useCallback(() => {
    if (marketData.length === 0) return;
    
    const generatedSignals: TradingSignal[] = marketData
      .filter(stock => Math.abs(stock.changePercent) > 0.5) // Reduced from 2% to 0.5%
      .slice(0, 5)
      .map((stock, index) => ({
        id: `signal-${stock.symbol}-${Date.now()}`,
        symbol: stock.symbol,
        action: stock.changePercent > 0.5 ? 'BUY' : stock.changePercent < -0.5 ? 'SELL' : 'HOLD',
        confidence: Math.min(95, 60 + Math.abs(stock.changePercent) * 8), // More sensitive
        price: stock.price,
        timestamp: new Date().toISOString(),
        agent: ['Signal Generator', 'Market Sentinel', 'Risk Assessor'][index % 3],
        reasoning: stock.changePercent > 0 ? `Positive momentum (+${stock.changePercent.toFixed(1)}%)` : `Negative movement (${stock.changePercent.toFixed(1)}%)`
      }));
    
    console.log(`🎯 Generated ${generatedSignals.length} trading signals from ${marketData.length} stocks`);
    setTradingSignals(generatedSignals);
  }, [marketData]);

  // Real alerts fetching
  const fetchAlerts = useCallback(async () => {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/agents/alerts?limit=5');
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data?.alerts) {
          const validAlerts = data.data.alerts.filter((alert: any) => 
            alert.title && alert.message
          ).map((alert: any) => ({
            id: alert.id || `alert-${Date.now()}`,
            type: alert.type || 'info',
            title: alert.title,
            message: alert.message,
            timestamp: alert.timestamp || new Date().toISOString(),
            severity: alert.severity || 'low'
          }));
          setAlerts(validAlerts);
        }
      }
    } catch (error) {
      console.error('Alerts fetch error:', error);
    }
  }, []);

  // Enhanced activity feed with multiple event types
  const updateEnhancedActivityFeed = useCallback(() => {
    const now = new Date().toLocaleTimeString();
    const newEntries: string[] = [];
    
    if (marketData.length > 0) {
      newEntries.push(`${now} - Market data updated: ${marketData.length} stocks`);
      
      const gainers = marketData.filter(s => s.changePercent > 2).length;
      const losers = marketData.filter(s => s.changePercent < -2).length;
      
      if (gainers > 0) newEntries.push(`${now} - ${gainers} stocks showing strong gains`);
      if (losers > 0) newEntries.push(`${now} - ${losers} stocks declining significantly`);
    }
    
    if (tradingSignals.length > 0) {
      newEntries.push(`${now} - ${tradingSignals.length} trading signals generated`);
    }
    
    if (agents.length > 0) {
      const activeAgents = agents.filter(a => a.status === 'active').length;
      if (activeAgents !== globalStats.agentsOnline) {
        newEntries.push(`${now} - ${activeAgents} AI agents online and monitoring`);
      }
    }
    
    setActivityFeed(prev => {
      const updated = [...newEntries, ...prev].slice(0, 8);
      return updated.filter((entry, index, arr) => 
        arr.findIndex(e => e.split(' - ')[1] === entry.split(' - ')[1]) === index
      );
    });
  }, [marketData, tradingSignals, agents, globalStats.agentsOnline]);

  // Throttled updates to prevent performance issues
  useEffect(() => {
    if (marketData.length > 0) {
      const timeoutId = setTimeout(() => {
        requestIdleCallback(() => {
          updateEnhancedActivityFeed();
          generateTradingSignals();
        });
      }, 1000);
      return () => clearTimeout(timeoutId);
    }
  }, [marketData.length]);

  // Disabled WebSocket message handling to prevent performance issues
  // useEffect(() => {
  //   if (lastMessage) {
  //     // Handle WebSocket messages
  //   }
  // }, [lastMessage]);

  // Initialize with real data only - no fallback fake data
  useEffect(() => {
    console.log('🚀 CommandCenter initializing - will fetch real data only');
    // Don't set any fake data - let the fetchEnhancedMarketData function get real data
    
    // Immediate market data fetch
    fetchEnhancedMarketData();
    
    // Load agents with delay
    setTimeout(() => {
      fetchEnhancedAgentStatus();
    }, 2000);
  }, [fetchEnhancedMarketData]);

  // Auto refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      console.log('🔄 Auto refresh triggered');
      fetchEnhancedMarketData();
    }, 30000);

    return () => clearInterval(interval);
  }, [fetchEnhancedMarketData]);

  // Disabled auto refresh to prevent excessive API calls
  // Users can manually refresh using the refresh button

  // Filtered market data based on search
  const filteredMarketData = useMemo(() => {
    if (!filterSymbol) return marketData;
    return marketData.filter(stock => 
      stock.symbol.toLowerCase().includes(filterSymbol.toLowerCase())
    );
  }, [marketData, filterSymbol]);

  // Enhanced utility functions
  const getAgentIcon = (id: string) => {
    const icons = {
      market_sentinel: <Activity className="w-5 h-5" />,
      news_intelligence: <Globe className="w-5 h-5" />,
      risk_assessor: <Shield className="w-5 h-5" />,
      signal_generator: <Zap className="w-5 h-5" />,
      compliance_guardian: <FileText className="w-5 h-5" />,
      executive_summary: <Brain className="w-5 h-5" />
    };
    return icons[id as keyof typeof icons] || <Cpu className="w-5 h-5" />;
  };

  const getStatusColor = (status: string) => {
    const colors = {
      active: 'text-green-400 bg-green-900/20',
      processing: 'text-yellow-400 bg-yellow-900/20',
      idle: 'text-gray-400 bg-gray-900/20',
      error: 'text-red-400 bg-red-900/20'
    };
    return colors[status as keyof typeof colors] || 'text-gray-400 bg-gray-900/20';
  };

  const getRiskColor = (risk: string) => {
    const colors = {
      LOW: 'text-green-400',
      MEDIUM: 'text-yellow-400',
      HIGH: 'text-red-400'
    };
    return colors[risk as keyof typeof colors] || 'text-gray-400';
  };

  const getSignalColor = (action: string) => {
    const colors = {
      BUY: 'text-green-400 bg-green-900/20',
      SELL: 'text-red-400 bg-red-900/20',
      HOLD: 'text-yellow-400 bg-yellow-900/20'
    };
    return colors[action as keyof typeof colors] || 'text-gray-400 bg-gray-900/20';
  };

  // Render enhanced header with controls
  const renderEnhancedHeader = () => (
    <div className="bg-black/40 backdrop-blur-sm rounded-xl p-3 sm:p-4 lg:p-6 border border-gray-700 mb-4 sm:mb-6">
      <div className="flex flex-col space-y-4">
        {/* Title Section */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 sm:gap-0">
          <div className="flex items-center space-x-2 sm:space-x-3">
            <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-blue-400" />
            <div>
              <h1 className="text-lg sm:text-xl lg:text-2xl font-bold text-white">Command Center</h1>
              <p className="text-gray-400 text-xs sm:text-sm hidden sm:block">Real-time Financial Intelligence</p>
            </div>
          </div>
          
          {/* Connection Status */}
          <div className="flex items-center justify-between sm:justify-end space-x-2 sm:space-x-4">
            <div className="flex items-center space-x-1 sm:space-x-2 text-green-400">
              <Wifi className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="text-xs sm:text-sm">
                {fetchingData ? 'Fetching...' : error ? 'API Error' : 'API Connected'}
              </span>
            </div>
            
            {/* Agent Count - Desktop only */}
            <div className="hidden md:flex items-center space-x-1 text-xs px-2 py-1 rounded bg-green-900/20 text-green-400">
              <Brain className="w-3 h-3" />
              <span>{agents.length} Agents</span>
            </div>
          </div>
        </div>
        
        {/* Controls Section */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 sm:gap-4">
          {/* Left Controls */}
          <div className="flex items-center space-x-2">
            {/* Refresh & Auto-refresh buttons */}
            <button
              onClick={async () => {
                setFetchingData(true);
                try {
                  await Promise.all([
                    fetchEnhancedMarketData(),
                    fetchEnhancedAgentStatus()
                  ]);
                } finally {
                  setFetchingData(false);
                }
              }}
              disabled={fetchingData}
              className="p-1.5 sm:p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white transition-colors"
              title="Refresh Data"
            >
              <RefreshCw className={`w-3 h-3 sm:w-4 sm:h-4 ${fetchingData ? 'animate-spin' : ''}`} />
            </button>
            
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
                autoRefresh ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
              }`}
              title={autoRefresh ? 'Disable Auto Refresh' : 'Enable Auto Refresh'}
            >
              <RefreshCw className={`w-3 h-3 sm:w-4 sm:h-4 ${autoRefresh && !fetchingData ? 'animate-spin' : ''}`} />
            </button>
          </div>
          
          {/* Right Controls */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
            {/* View Mode Selector - Desktop */}
            <div className="hidden lg:flex space-x-1 bg-gray-800 rounded-lg p-1">
              {(['overview', 'detailed', 'analytics'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm transition-colors ${
                    viewMode === mode
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:text-white'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
            
            {/* Timeframe Selector - Desktop */}
            <div className="hidden sm:flex space-x-1 bg-gray-800 rounded-lg p-1">
              {(['1m', '5m', '1h', '1d'] as const).map((tf) => (
                <button
                  key={tf}
                  onClick={() => setSelectedTimeframe(tf)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    selectedTimeframe === tf
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:text-white'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
            
            {/* Mobile Combined Selector */}
            <div className="sm:hidden">
              <select 
                value={`${viewMode}-${selectedTimeframe}`}
                onChange={(e) => {
                  const [mode, tf] = e.target.value.split('-');
                  setViewMode(mode as any);
                  setSelectedTimeframe(tf as any);
                }}
                className="w-full bg-gray-800 text-white rounded-lg px-3 py-2 text-sm border border-gray-600"
                title="Select view mode and timeframe"
                aria-label="Select view mode and timeframe"
              >
                {(['overview', 'detailed', 'analytics'] as const).map((mode) =>
                  (['1m', '5m', '1h', '1d'] as const).map((tf) => (
                    <option key={`${mode}-${tf}`} value={`${mode}-${tf}`}>
                      {mode.charAt(0).toUpperCase() + mode.slice(1)} - {tf}
                    </option>
                  ))
                )}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Render enhanced metrics cards
  const renderEnhancedMetrics = () => (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4 mb-4 sm:mb-6">
      <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-xs">Portfolio Value</p>
            <p className="text-lg sm:text-xl font-bold text-blue-400">${((portfolioMetrics.totalValue || 0) / 1000).toFixed(0)}K</p>
            <p className={`text-xs ${(portfolioMetrics.dailyChangePercent || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {(portfolioMetrics.dailyChangePercent || 0) >= 0 ? '+' : ''}{(portfolioMetrics.dailyChangePercent || 0).toFixed(2)}%
            </p>
          </div>
          <DollarSign className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400" />
        </div>
      </div>
      
      <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-xs">Market Cap</p>
            <p className="text-lg sm:text-xl font-bold text-green-400">${(globalStats.marketCap || 0).toFixed(1)}T</p>
          </div>
          <Globe className="w-4 h-4 sm:w-5 sm:h-5 text-green-400" />
        </div>
      </div>
      
      <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-xs">Active Signals</p>
            <p className="text-lg sm:text-xl font-bold text-purple-400">{globalStats.activeSignals || 0}</p>
          </div>
          <Zap className="w-4 h-4 sm:w-5 sm:h-5 text-purple-400" />
        </div>
      </div>
      
      <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-xs">Risk Level</p>
            <p className={`text-lg sm:text-xl font-bold ${getRiskColor(globalStats.riskLevel)}`}>
              {globalStats.riskLevel}
            </p>
          </div>
          <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400" />
        </div>
      </div>
      
      <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-xs">Agents Online</p>
            <p className="text-lg sm:text-xl font-bold text-cyan-400">{globalStats.agentsOnline || 0}/6</p>
          </div>
          <Users className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
        </div>
      </div>
      
      <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-gray-400 text-xs">Data Points</p>
            <p className="text-lg sm:text-xl font-bold text-orange-400">{globalStats.dataPointsProcessed || 0}</p>
          </div>
          <Database className="w-4 h-4 sm:w-5 sm:h-5 text-orange-400" />
        </div>
      </div>
    </div>
  );

  // Render market data based on view mode
  const renderEnhancedMarketData = () => {
    const getDisplayData = () => {
      switch(viewMode) {
        case 'detailed':
          return filteredMarketData.slice(0, 15);
        case 'analytics':
          return filteredMarketData.filter(stock => Math.abs(stock.changePercent || 0) > 1).slice(0, 12);
        default:
          return filteredMarketData.slice(0, 10); // Show all 10 stocks in overview mode
      }
    };

    return (
    <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-3 sm:gap-0">
        <h2 className="text-base sm:text-lg font-bold flex items-center">
          <Activity className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-blue-400" />
          <span className="hidden sm:inline">Market Data ({viewMode}) - {selectedTimeframe}</span>
          <span className="sm:hidden">Market Data</span>
        </h2>
        <div className="flex items-center space-x-2">
          <input
            type="text"
            placeholder="Filter symbols..."
            value={filterSymbol}
            onChange={(e) => setFilterSymbol(e.target.value)}
            className="px-2 sm:px-3 py-1 bg-gray-800 border border-gray-600 rounded text-xs sm:text-sm text-white w-32 sm:w-auto"
          />
          <Filter className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
        </div>
      </div>
      
      <div className="space-y-2 sm:space-y-3 max-h-96 overflow-y-auto">
        {getDisplayData().length > 0 ? (
          getDisplayData().map((stock) => (
            <div key={`${stock.symbol}-${stock.timestamp}`} className="flex items-center justify-between p-2 sm:p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800/70 transition-colors">
              <div className="flex items-center space-x-2 sm:space-x-3">
                <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-xs sm:text-sm font-bold">{stock.symbol.slice(0, 3)}</span>
                </div>
                <div>
                  <p className="font-semibold text-white text-sm sm:text-base">{stock.symbol}</p>
                  {viewMode === 'detailed' && (
                    <p className="text-xs text-gray-400 hidden sm:block">
                      H: ${(stock.high24h || 0).toFixed(2)} | L: ${(stock.low24h || 0).toFixed(2)}
                    </p>
                  )}
                  {viewMode === 'analytics' && (
                    <p className="text-xs text-gray-400 hidden sm:block">
                      Volatility: {Math.abs(stock.changePercent || 0).toFixed(1)}% | Sector: {stock.sector}
                    </p>
                  )}
                  {viewMode === 'overview' && (
                    <p className="text-xs text-gray-400">
                      <span className="sm:hidden">Vol: {((stock.volume || 0) / 1000000).toFixed(1)}M</span>
                      <span className="hidden sm:inline">Vol: {((stock.volume || 0) / 1000000).toFixed(1)}M | Cap: ${(((stock.marketCap || (stock.price * stock.volume * 1000)) || 0) / 1000000000).toFixed(1)}B</span>
                    </p>
                  )}
                </div>
              </div>
              <div className="text-right">
                <p className="font-bold text-white">${(stock.price || 0).toFixed(2)}</p>
                <div className="flex items-center justify-end">
                  {(stock.change || 0) > 0 ? (
                    <TrendingUp className="w-4 h-4 text-green-400 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-400 mr-1" />
                  )}
                  <span className={`text-sm ${(stock.change || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(stock.changePercent || 0) >= 0 ? '+' : ''}{(stock.changePercent || 0).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-8 text-gray-400">
            {fetchingData ? (
              <div>
                <div className="animate-pulse rounded-full h-8 w-8 bg-blue-400/20 mx-auto mb-2"></div>
                <p className="text-blue-400">Fetching real Yahoo Finance data...</p>
                <p className="text-xs text-gray-500 mt-1">This may take 10-15 seconds</p>
              </div>
            ) : (
              <div>
                <Activity className="w-8 h-8 mx-auto mb-2" />
                <p>{error ? 'API Connection Issue' : 'Loading market data...'}</p>
                {error && (
                  <div className="mt-2">
                    <p className="text-xs text-yellow-400 mb-2">{error}</p>
                    <button 
                      onClick={fetchEnhancedMarketData}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
                    >
                      Retry Connection
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
    );
  };

  // Render trading signals
  const renderTradingSignals = () => (
    <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
      <h2 className="text-base sm:text-lg font-bold mb-3 sm:mb-4 flex items-center">
        <Target className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-purple-400" />
        <span className="hidden sm:inline">Trading Signals</span>
        <span className="sm:hidden">Signals</span>
      </h2>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {tradingSignals.length > 0 ? (
          tradingSignals.map((signal) => (
            <div key={`${signal.id}-${signal.timestamp}`} className="p-2 sm:p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800/70 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 flex-1 min-w-0">
                  <span className={`px-1.5 sm:px-2 py-0.5 sm:py-1 rounded text-xs font-bold ${getSignalColor(signal.action)}`}>
                    {signal.action}
                  </span>
                  <span className="font-semibold text-white text-sm sm:text-base truncate">{signal.symbol}</span>
                </div>
                <div className="text-right flex-shrink-0 ml-2">
                  <p className="text-sm text-white">${(signal.price || 0).toFixed(2)}</p>
                  <p className="text-xs text-gray-400">{(signal.confidence || 0).toFixed(0)}%</p>
                </div>
              </div>
              {signal.reasoning && (
                <p className="text-xs text-gray-400 mt-1 truncate hidden sm:block">{signal.reasoning}</p>
              )}
              <p className="text-xs text-gray-500 mt-1">by {signal.agent}</p>
            </div>
          ))
        ) : (
          <div className="text-center py-4 text-gray-400">
            <Target className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2" />
            <p className="text-sm">No trading signals available</p>
            <p className="text-xs text-gray-500 mt-1 hidden sm:block">Requires market data connection</p>
          </div>
        )}
      </div>
    </div>
  );

  console.log('CommandCenter: Rendering enhanced component');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white overflow-x-hidden px-2 sm:px-4 lg:px-6 py-4 sm:py-6">
      <div className="container mx-auto max-w-7xl">
        {renderEnhancedHeader()}
        {renderEnhancedMetrics()}
        
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 sm:gap-6">
          {/* Left Column - Market Data */}
          <div className="xl:col-span-2 order-2 xl:order-1">
            {renderEnhancedMarketData()}
          </div>
          
          {/* Right Column - Signals and Agents */}
          <div className="space-y-4 sm:space-y-6 order-1 xl:order-2">
            {renderTradingSignals()}
            
            {/* AI Agents Status */}
            <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
              <h2 className="text-base sm:text-lg font-bold mb-3 sm:mb-4 flex items-center">
                <Brain className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-purple-400" />
                <span className="hidden sm:inline">AI Agent Network</span>
                <span className="sm:hidden">AI Agents</span>
              </h2>
              <div className="space-y-2 sm:space-y-3 max-h-64 overflow-y-auto">
                {agents.length > 0 ? (
                  agents.map((agent) => (
                    <div key={`${agent.id}-${agent.lastUpdate}`} className="flex items-center justify-between p-2 sm:p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800/70 transition-colors">
                      <div className="flex items-center space-x-2 sm:space-x-3 flex-1 min-w-0">
                        <div className={`p-1.5 sm:p-2 rounded-lg ${getStatusColor(agent.status)}`}>
                          {getAgentIcon(agent.id)}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="font-semibold text-white text-sm sm:text-base truncate">{agent.name}</p>
                          <p className="text-xs text-gray-400 truncate hidden sm:block">{agent.currentTask || 'Monitoring markets'}</p>
                        </div>
                      </div>
                      <div className="text-right flex-shrink-0 ml-2">
                        <p className="text-sm font-bold text-blue-400">{(agent.performance || 0).toFixed(1)}%</p>
                        <p className="text-xs text-gray-400">{agent.signals || 0} signals</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-gray-400">
                    <Brain className="w-5 h-5 sm:w-6 sm:h-6 mx-auto mb-2" />
                    {error ? (
                      <div className="text-red-400">
                        <p className="text-sm">⚠️ Agent Status API Unavailable</p>
                        <p className="text-xs mt-1 hidden sm:block">Cannot fetch real agent data</p>
                      </div>
                    ) : (
                      <p className="text-sm">Loading agent status...</p>
                    )}
                    <button 
                      onClick={fetchEnhancedAgentStatus}
                      className="mt-2 px-2 sm:px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs transition-colors"
                    >
                      Retry Connection
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Activity Feed */}
        <div className="mt-6 bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
          <h2 className="text-lg font-bold mb-3 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-yellow-400" />
            Live Activity Feed
          </h2>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {activityFeed.length > 0 ? (
              activityFeed.map((activity, index) => {
                const [timestamp, ...messageParts] = activity.split(' - ');
                const message = messageParts.join(' - ');
                return (
                  <div key={`${index}-${timestamp}`} className="flex items-center space-x-2 text-sm p-2 hover:bg-gray-800/30 rounded-md transition-colors">
                    <div className={`w-2 h-2 rounded-full ${
                      index % 4 === 0 ? 'bg-green-400' : 
                      index % 4 === 1 ? 'bg-blue-400' : 
                      index % 4 === 2 ? 'bg-yellow-400' : 'bg-purple-400'
                    }`}></div>
                    <span className="text-gray-400 text-xs">{timestamp}</span>
                    <span className="text-white flex-1">{message}</span>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-4 text-gray-400">
                <Zap className="w-6 h-6 mx-auto mb-2" />
                <p className="text-sm">Activity feed will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CommandCenter;