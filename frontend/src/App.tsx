import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, Zap, MessageSquare, Grid, Settings, PieChart, Newspaper, Activity, Brain, Menu, X, Mic, Eye, Target } from 'lucide-react';
import CommandCenter from './components/CommandCenter';
import EnhancedAIAssistant from './components/advanced/EnhancedAIAssistant';
import AdvancedAnalytics from './components/AdvancedAnalytics';
import WorkingTradingTerminal from './components/advanced/WorkingTradingTerminal';
import AdvancedPortfolioAnalytics from './components/advanced/AdvancedPortfolioAnalytics';
import PortfolioAnalytics from './components/advanced/PortfolioAnalytics';
import AdvancedAnalyticsDashboard from './components/advanced/AdvancedAnalyticsDashboard';
import RealTimeNewsCenter from './components/advanced/RealTimeNewsCenter';
import AIIntelligence from './components/AIIntelligence';
import Enhanced3DMarketVisualization from './components/advanced/Enhanced3DMarketVisualization';
import EnhancedVoiceAI from './components/advanced/EnhancedVoiceAI';
import ErrorBoundary from './components/ErrorBoundary';
import ConnectionManager from './utils/connectionManager';
import './App.css';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  timestamp: string;
}

interface AgentData {
  market_sentiment: any;
  news_intelligence: any;
  risk_assessor: any;
  signal_generator: any;
  compliance_guardian: any;
  executive_summary: any;
}

function App() {
  const [currentView, setCurrentView] = useState<'dashboard' | 'ai-center' | 'trading' | 'portfolio' | 'advanced-analytics' | 'news' | 'chat' | '3d-market' | 'voice-ai'>('dashboard');
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [agentData, setAgentData] = useState<AgentData>({
    market_sentiment: null,
    news_intelligence: null,
    risk_assessor: null,
    signal_generator: null,
    compliance_guardian: null,
    executive_summary: null
  });
  const [isConnected, setIsConnected] = useState(true); // Start as connected since backend is running
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  // Keep all components mounted to prevent reloading
  const componentRefs = useRef<{[key: string]: React.ReactElement}>({});

  // Optimized connection check using singleton pattern
  useEffect(() => {
    const connectionManager = ConnectionManager.getInstance();
    
    const checkConnection = async () => {
      const connected = await connectionManager.checkConnection();
      setIsConnected(connected);
      setLastUpdate(new Date());
    };
    
    // Initial check after component loads
    const initialTimeout = setTimeout(checkConnection, 1000);
    
    // Check every 2 minutes to reduce load
    const interval = setInterval(checkConnection, 120000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, []);

  // AI Agent polling for real-time updates
  useEffect(() => {
    const fetchAgentData = async () => {
      try {
        const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
        
        // Fetch all agent data in parallel
        const [
          marketSentiment,
          newsIntelligence,
          riskAssessment,
          signalGeneration,
          complianceCheck,
          executiveSummary
        ] = await Promise.all([
          fetch('http://127.0.0.1:8001/api/agents/market-sentinel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols, timeframe: '1d' })
          }).then(res => res.json()).catch(err => ({ error: err.message })),
          
          fetch('http://127.0.0.1:8001/api/agents/news-intelligence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols })
          }).then(res => res.json()).catch(err => ({ error: err.message })),
          
          fetch('http://127.0.0.1:8001/api/agents/risk-assessor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              portfolio: [
                { symbol: 'AAPL', quantity: 100, value: 17523 },
                { symbol: 'MSFT', quantity: 50, value: 16944 },
                { symbol: 'GOOGL', quantity: 25, value: 3364 }
              ]
            })
          }).then(res => res.json()).catch(err => ({ error: err.message })),
          
          fetch('http://localhost:8001/api/agents/signal-generator', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols, risk_tolerance: 'medium' })
          }).then(res => res.json()).catch(err => ({ error: err.message })),
          
          fetch('http://localhost:8001/api/agents/compliance-guardian', {
            method: 'GET'
          }).then(res => res.json()).catch(err => ({ error: err.message })),
          
          fetch('http://localhost:8001/api/agents/executive-summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              marketData: marketData,
              analysisData: {} 
            })
          }).then(res => res.json()).catch(err => ({ error: err.message }))
        ]);

        setAgentData({
          market_sentiment: marketSentiment,
          news_intelligence: newsIntelligence,
          risk_assessor: riskAssessment,
          signal_generator: signalGeneration,
          compliance_guardian: complianceCheck,
          executive_summary: executiveSummary
        });
        
      } catch (error) {
        console.error('Error fetching agent data:', error);
      }
    };

    // Disabled aggressive polling to prevent page freezing
    // fetchAgentData();
    
    // No automatic polling to prevent responsiveness issues
    // const interval = setInterval(fetchAgentData, 60000);
    
    // return () => clearInterval(interval);
  }, [marketData]); // Re-run when market data changes

  const navigationItems = [
    { id: 'dashboard', label: 'Command Center', icon: <Grid className="w-5 h-5" /> },
    { id: 'ai-center', label: 'AI Intelligence', icon: <Brain className="w-5 h-5" /> },
    { id: '3d-market', label: '3D Market Visualization', icon: <Eye className="w-5 h-5" /> },
    { id: 'voice-ai', label: 'Voice AI Assistant', icon: <Mic className="w-5 h-5" /> },
    { id: 'trading', label: 'Trading Terminal', icon: <TrendingUp className="w-5 h-5" /> },
    { id: 'portfolio', label: 'Portfolio Analytics', icon: <PieChart className="w-5 h-5" /> },
    { id: 'advanced-analytics', label: 'Advanced Analytics', icon: <BarChart3 className="w-5 h-5" /> },
    { id: 'news', label: 'News Center', icon: <Newspaper className="w-5 h-5" /> },
    { id: 'chat', label: 'AI Assistant', icon: <MessageSquare className="w-5 h-5" /> }
  ];

  // Memoized components to prevent re-renders
  const components = useMemo(() => ({
    'dashboard': (
      <ErrorBoundary key="dashboard">
        <CommandCenter />
      </ErrorBoundary>
    ),
    'ai-center': (
      <ErrorBoundary key="ai-center">
        <AIIntelligence />
      </ErrorBoundary>
    ),
    'trading': (
      <ErrorBoundary key="trading">
        <WorkingTradingTerminal />
      </ErrorBoundary>
    ),
    'portfolio': (
      <ErrorBoundary key="portfolio">
        <PortfolioAnalytics />
      </ErrorBoundary>
    ),
    'advanced-analytics': (
      <ErrorBoundary key="advanced-analytics">
        <AdvancedAnalyticsDashboard />
      </ErrorBoundary>
    ),
    'news': (
      <ErrorBoundary key="news">
        <RealTimeNewsCenter />
      </ErrorBoundary>
    ),
    'chat': (
      <ErrorBoundary key="chat">
        <EnhancedAIAssistant />
      </ErrorBoundary>
    ),
    '3d-market': (
      <ErrorBoundary key="3d-market">
        <div className="h-screen">
          <Enhanced3DMarketVisualization 
            onSymbolSelect={(symbol) => console.log('Selected:', symbol)}
          />
        </div>
      </ErrorBoundary>
    ),
    'voice-ai': (
      <ErrorBoundary key="voice-ai">
        <EnhancedVoiceAI />
      </ErrorBoundary>
    )
  }), []);

  // Fast tab switching handler
  const handleTabSwitch = useCallback((viewId: string) => {
    // Throttle tab switching to prevent excessive re-renders
    requestAnimationFrame(() => {
      setCurrentView(viewId as any);
      setIsMobileMenuOpen(false);
    });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Mobile Navigation Header */}
      <nav className="bg-black/30 backdrop-blur-sm border-b border-gray-700">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Settings className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg sm:text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                FinanceGPT Live
              </span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden lg:flex space-x-1 xl:space-x-2 overflow-x-auto scrollbar-thin max-w-[calc(100vw-300px)] md:max-w-[calc(100vw-320px)] xl:max-w-[calc(100vw-350px)]">
              {navigationItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleTabSwitch(item.id)}
                  className={`flex items-center space-x-1 xl:space-x-2 px-2 xl:px-3 py-2 rounded-lg text-xs xl:text-sm font-medium transition-colors whitespace-nowrap flex-shrink-0 ${
                    currentView === item.id
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  {item.icon}
                  <span className="hidden xl:block">{item.label}</span>
                </button>
              ))}
            </div>

            {/* Mobile Menu Button + Status */}
            <div className="flex items-center space-x-3">
              {/* Connection Status - Always visible but responsive */}
              <div className="hidden sm:flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                  <span className="text-xs text-gray-300">
                    {isConnected ? 'API Online' : 'API Check...'}
                  </span>
                </div>
                <div className="text-xs text-gray-400 hidden md:block">
                  {lastUpdate.toLocaleTimeString()}
                </div>
              </div>

              {/* Mobile Status Indicator */}
              <div className="sm:hidden flex items-center">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
              </div>

              {/* Mobile Menu Button */}
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="lg:hidden p-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              >
                {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Mobile Navigation Menu */}
          {isMobileMenuOpen && (
            <div className="lg:hidden border-t border-gray-700 py-4 overflow-y-auto max-h-[70vh]">
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                {navigationItems.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => handleTabSwitch(item.id)}
                    className={`flex flex-col items-center space-y-1 p-3 rounded-lg text-xs font-medium transition-colors ${
                      currentView === item.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    {item.icon}
                    <span className="text-center leading-tight">{item.label}</span>
                  </button>
                ))}
              </div>
              
              {/* Mobile Status Information */}
              <div className="mt-4 pt-4 border-t border-gray-700 sm:hidden">
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>Status: {isConnected ? 'API Online' : 'API Check...'}</span>
                  <span>Updated: {lastUpdate.toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Main Content - Fast switching with display:none/block */}
      <main className="relative">
        {Object.entries(components).map(([viewId, component]) => (
          <div
            key={viewId}
            className={currentView === viewId ? 'block' : 'hidden'}
          >
            {component}
          </div>
        ))}
      </main>
    </div>
  );
}

export default App;
