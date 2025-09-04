import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Brain, TrendingUp, TrendingDown, Shield, Zap, Target, Activity, BarChart3,
  AlertTriangle, Eye, Settings, RefreshCw, Bot, LineChart, PieChart, Gauge,
  Filter, Search, Bell, Clock, CheckCircle, XCircle, Loader, DollarSign,
  Percent, Calendar, Users, Database, Cpu, Globe, WifiOff, Wifi, Play, Pause
} from 'lucide-react';
import { useAIIntelligence } from '../hooks/useAIIntelligence';

interface AIAgent {
  id: string;
  name: string;
  status: 'active' | 'processing' | 'idle' | 'error';
  performance: number;
  accuracy: number;
  signals_generated: number;
  last_prediction: string;
  specialization: string;
  model_version: string;
}

const AIIntelligence: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [isRunning, setIsRunning] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'predictions' | 'agents' | 'alerts'>('overview');
  const [timeframe, setTimeframe] = useState<'1h' | '4h' | '1d' | '1w'>('1d');
  const [alertFilter, setAlertFilter] = useState<string>('all');
  const [agents, setAgents] = useState<AIAgent[]>([]);

  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'];
  
  const {
    analyses,
    marketIntel,
    alerts,
    isLoading: loading,
    error,
    lastUpdate,
    refresh,
    clearAlerts,
    getAnalysis,
    getPerformanceMetrics,
    isConnected
  } = useAIIntelligence(symbols, timeframe);

  // Fetch AI agents data
  const fetchAIAgents = useCallback(async () => {
    try {
      console.log('🤖 AI Intelligence fetching agent status...');
      const response = await fetch('http://127.0.0.1:8001/api/agents/status');
      if (response.ok) {
        const data = await response.json();
        console.log('🤖 AI Intelligence agent API response:', data);
        if (data.success && data.data?.agents) {
          const agentList = Object.entries(data.data.agents).map(([id, agent]: [string, any]) => ({
            id,
            name: agent.name || id.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
            status: agent.status || 'idle',
            performance: parseFloat(agent.performance) || Math.random() * 40 + 60,
            accuracy: parseFloat(agent.accuracy) || Math.random() * 20 + 75,
            signals_generated: parseInt(agent.signals_generated) || Math.floor(Math.random() * 50),
            last_prediction: agent.last_prediction || new Date().toISOString(),
            specialization: agent.specialization || 'Market Analysis',
            model_version: agent.model_version || 'v2.1'
          }));
          console.log('✅ AI Intelligence agents loaded:', agentList.length);
          setAgents(agentList);
        }
      } else {
        console.error('❌ AI Intelligence agent API failed:', response.status);
      }
    } catch (error) {
      console.error('❌ AI Intelligence agent fetch error:', error);
    }
  }, []);

  // Initialize agents data
  useEffect(() => {
    fetchAIAgents();
    
    if (isRunning) {
      const interval = setInterval(fetchAIAgents, 15000);
      return () => clearInterval(interval);
    }
  }, [isRunning, fetchAIAgents]);

  // Filtered alerts
  const filteredAlerts = useMemo(() => {
    return alertFilter === 'all' 
      ? alerts 
      : alerts.filter(alert => alert.type === alertFilter);
  }, [alerts, alertFilter]);

  // Performance metrics
  const performanceMetrics = useMemo(() => {
    const aiMetrics = getPerformanceMetrics();
    const activeAgents = agents.filter(a => a.status === 'active').length;
    const avgPerformance = agents.length > 0 
      ? agents.reduce((sum, a) => sum + a.performance, 0) / agents.length 
      : 0;
    
    return {
      avgConfidence: aiMetrics?.avgConfidence || 0,
      activeAgents,
      totalAgents: agents.length,
      avgPerformance,
      totalSignals: agents.reduce((sum, a) => sum + a.signals_generated, 0),
      criticalAlerts: alerts.filter(a => a.severity === 'critical').length,
      bullishSignals: aiMetrics?.bullishSignals || 0,
      bearishSignals: aiMetrics?.bearishSignals || 0
    };
  }, [getPerformanceMetrics, agents, alerts]);



  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-2 sm:p-4 lg:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 sm:mb-6 gap-4 sm:gap-0">
        <div className="flex items-center space-x-3 sm:space-x-4">
          <div className="p-2 sm:p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg sm:rounded-xl">
            <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              AI Intelligence Center
            </h1>
            <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-4 text-xs sm:text-sm text-gray-400 gap-1 sm:gap-0">
              <span className={`flex items-center ${
                isConnected ? 'text-green-400' : 'text-red-400'
              }`}>
                {isConnected ? <Wifi className="w-3 h-3 sm:w-4 sm:h-4 mr-1" /> : <WifiOff className="w-3 h-3 sm:w-4 sm:h-4 mr-1" />}
                {isConnected ? 'Online' : 'Offline'}
              </span>
              <span className="hidden sm:inline">Updated: {lastUpdate.toLocaleTimeString()}</span>
              <span className="sm:hidden">{lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value as any)}
            title="Select timeframe"
            aria-label="Select analysis timeframe"
            className="px-2 sm:px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm flex-1 sm:flex-initial"
          >
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
            <option value="1w">1 Week</option>
          </select>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex items-center justify-center space-x-1 sm:space-x-2 px-3 sm:px-4 py-2 rounded-lg text-sm font-medium ${
                isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
              } text-white transition-colors flex-1 sm:flex-initial`}
            >
              {isRunning ? <Pause className="w-3 h-3 sm:w-4 sm:h-4" /> : <Play className="w-3 h-3 sm:w-4 sm:h-4" />}
              <span className="hidden sm:inline">{isRunning ? 'Pause' : 'Resume'}</span>
              <span className="sm:hidden">{isRunning ? '⏸' : '▶'}</span>
            </button>
            
            <button
              onClick={() => {
                refresh();
                fetchAIAgents();
              }}
              title="Refresh data"
              aria-label="Refresh all AI data"
              className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white transition-colors flex-shrink-0"
            >
              <RefreshCw className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Performance Dashboard */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4 mb-4 sm:mb-6">
        <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs">AI Confidence</p>
              <p className="text-lg sm:text-xl lg:text-2xl font-bold text-blue-400">{performanceMetrics.avgConfidence.toFixed(1)}%</p>
            </div>
            <Target className="w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-blue-400" />
          </div>
        </div>

        <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs">Active Agents</p>
              <p className="text-lg sm:text-xl lg:text-2xl font-bold text-green-400">{performanceMetrics.activeAgents}/{performanceMetrics.totalAgents}</p>
            </div>
            <Bot className="w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-green-400" />
          </div>
        </div>

        <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs">Performance</p>
              <p className="text-lg sm:text-xl lg:text-2xl font-bold text-purple-400">{performanceMetrics.avgPerformance.toFixed(1)}%</p>
            </div>
            <Gauge className="w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-purple-400" />
          </div>
        </div>

        <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs">Total Signals</p>
              <p className="text-lg sm:text-xl lg:text-2xl font-bold text-yellow-400">{performanceMetrics.totalSignals}</p>
            </div>
            <Zap className="w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-yellow-400" />
          </div>
        </div>

        <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs">Critical Alerts</p>
              <p className="text-2xl font-bold text-red-400">{performanceMetrics.criticalAlerts}</p>
            </div>
            <AlertTriangle className="w-6 h-6 text-red-400" />
          </div>
        </div>

        <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-xs">Market Regime</p>
              <p className="text-sm font-bold text-cyan-400">{marketIntel?.regime || 'Loading...'}</p>
            </div>
            <Activity className="w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-cyan-400" />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6">
        {/* AI Analysis Cards */}
        <div className="lg:col-span-3 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 sm:gap-4 order-2 lg:order-1">
          {symbols.map(symbol => {
            const analysis = analyses[symbol];
            if (!analysis) return null;

            return (
              <div
                key={symbol}
                className={`bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border cursor-pointer transition-all ${
                  selectedSymbol === symbol ? 'border-blue-500 bg-blue-500/10' : 'border-gray-700 hover:border-gray-600'
                }`}
                onClick={() => setSelectedSymbol(symbol)}
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-base sm:text-lg font-bold text-white">{symbol}</h3>
                  <div className={`flex items-center space-x-1 ${
                    analysis?.prediction?.direction === 'bullish' ? 'text-green-400' :
                    analysis?.prediction?.direction === 'bearish' ? 'text-red-400' : 'text-gray-400'
                  }`}>
                    {analysis?.prediction?.direction === 'bullish' ? <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" /> :
                     analysis?.prediction?.direction === 'bearish' ? <TrendingDown className="w-3 h-3 sm:w-4 sm:h-4" /> :
                     <Activity className="w-3 h-3 sm:w-4 sm:h-4" />}
                    <span className="text-xs sm:text-sm font-medium hidden sm:inline">
                      {analysis?.prediction?.direction || 'Loading...'}
                    </span>
                  </div>
                </div>

                <div className="space-y-2 sm:space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400 text-xs sm:text-sm">Price Target</span>
                    <span className="text-white font-semibold text-sm sm:text-base">
                      ${analysis?.prediction?.priceTarget?.toFixed(2) || 'Loading...'}
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-gray-400 text-xs sm:text-sm">Confidence</span>
                    <span className={`font-semibold ${
                      (analysis?.prediction?.confidence || 0) > 80 ? 'text-green-400' :
                      (analysis?.prediction?.confidence || 0) > 60 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {analysis?.prediction?.confidence?.toFixed(1) || '0.0'}%
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">RSI</span>
                    <span className={`font-semibold ${
                      (analysis?.technical?.rsi || 0) > 70 ? 'text-red-400' :
                      (analysis?.technical?.rsi || 0) < 30 ? 'text-green-400' : 'text-gray-300'
                    }`}>
                      {analysis?.technical?.rsi?.toFixed(1) || '0.0'}
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">Sentiment</span>
                    <span className={`font-semibold ${
                      (analysis?.sentiment?.score || 0) > 60 ? 'text-green-400' :
                      (analysis?.sentiment?.score || 0) < 40 ? 'text-red-400' : 'text-gray-300'
                    }`}>
                      {analysis?.sentiment?.score?.toFixed(0) || '0'}
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-gray-400 text-sm">Risk Level</span>
                    <span className={`font-semibold ${
                      analysis?.risk?.riskLevel === 'high' ? 'text-red-400' :
                      analysis?.risk?.riskLevel === 'medium' ? 'text-yellow-400' : 'text-green-400'
                    }`}>
                      {analysis?.risk?.riskLevel ? 
                        analysis.risk.riskLevel.charAt(0).toUpperCase() + analysis.risk.riskLevel.slice(1) : 
                        'Low'
                      }
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right Sidebar */}
        <div className="space-y-4 sm:space-y-6 order-1 lg:order-2">
          {/* Market Intelligence */}
          {marketIntel && (
            <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
              <h3 className="text-base sm:text-lg font-bold text-white mb-3 sm:mb-4 flex items-center">
                <Globe className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-blue-400" />
                Market Intelligence
              </h3>
              
              <div className="space-y-2 sm:space-y-3">
                <div>
                  <span className="text-gray-400 text-xs sm:text-sm">Current Regime</span>
                  <p className="text-white font-semibold text-sm sm:text-base">{marketIntel.regime}</p>
                </div>
                
                <div>
                  <span className="text-gray-400 text-xs sm:text-sm">Confidence</span>
                  <p className="text-green-400 font-semibold text-sm sm:text-base">{(marketIntel.confidence * 100).toFixed(1)}%</p>
                </div>
                
                <div>
                  <span className="text-gray-400 text-xs sm:text-sm">Volatility State</span>
                  <p className="text-yellow-400 font-semibold text-sm sm:text-base">{marketIntel.volatilityState}</p>
                </div>
                
                <div>
                  <span className="text-gray-400 text-xs sm:text-sm">Key Drivers</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {marketIntel.keyDrivers.map((driver, index) => (
                      <span key={index} className="text-xs px-2 py-1 bg-blue-500/20 text-blue-300 rounded">
                        {driver}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Smart Alerts */}
          <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <h3 className="text-base sm:text-lg font-bold text-white flex items-center">
                <Bell className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-red-400" />
                Smart Alerts
              </h3>
              <select
                value={alertFilter}
                onChange={(e) => setAlertFilter(e.target.value)}
                title="Filter alerts"
                aria-label="Filter alerts by type"
                className="text-xs px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white"
              >
                <option value="all">All</option>
                <option value="opportunity">Opportunities</option>
                <option value="risk">Risks</option>
                <option value="technical">Technical</option>
              </select>
            </div>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {filteredAlerts.length > 0 ? filteredAlerts.slice(0, 10).map(alert => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border ${
                    alert.severity === 'critical' ? 'bg-red-900/20 border-red-500/50' :
                    alert.severity === 'high' ? 'bg-orange-900/20 border-orange-500/50' :
                    alert.severity === 'medium' ? 'bg-yellow-900/20 border-yellow-500/50' :
                    'bg-gray-800/50 border-gray-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-white bg-gray-700 px-2 py-1 rounded">
                      {alert.symbol}
                    </span>
                    <span className="text-xs text-gray-400">
                      {alert.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-200 mb-2">{alert.message}</p>
                  
                  <div className="flex items-center justify-between text-xs">
                    <span className={`px-2 py-1 rounded ${
                      alert.type === 'opportunity' ? 'bg-green-500/20 text-green-300' :
                      alert.type === 'risk' ? 'bg-red-500/20 text-red-300' :
                      'bg-blue-500/20 text-blue-300'
                    }`}>
                      {alert.type}
                    </span>
                    <span className="text-gray-400">
                      {(alert.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
              )) : (
                <div className="text-center py-8 text-gray-400">
                  <Bell className="w-8 h-8 mx-auto mb-2" />
                  <p>No alerts available</p>
                  <p className="text-xs mt-1">AI is monitoring markets...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIIntelligence;