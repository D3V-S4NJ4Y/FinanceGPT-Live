import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  LineChart, 
  Activity,
  AlertTriangle,
  Target,
  DollarSign,
  Percent,
  Calendar
} from 'lucide-react';

interface PortfolioMetrics {
  totalValue: number;
  dailyChange: number;
  dailyChangePercent: number;
  ytdReturn: number;
  sharpeRatio: number;
  volatility: number;
  maxDrawdown: number;
  beta: number;
}

interface AssetAllocation {
  symbol: string;
  name: string;
  percentage: number;
  value: number;
  sector: string;
  color: string;
}

interface RiskMetrics {
  var95: number;
  cvar95: number;
  correlation: { [key: string]: number };
  concentration: number;
}

const AdvancedAnalytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'portfolio' | 'risk' | 'performance' | 'allocation'>('portfolio');
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | '3M' | '1Y'>('1M');
  
  // Helper function to get CSS class for colors
  const getColorClass = (color: string): string => {
    const colorMap: { [key: string]: string } = {
      '#3B82F6': 'asset-color-blue',
      '#8B5CF6': 'asset-color-purple',
      '#10B981': 'asset-color-green',
      '#F59E0B': 'asset-color-yellow',
      '#EF4444': 'asset-color-red',
      '#EC4899': 'asset-color-pink',
      '#6366F1': 'asset-color-indigo',
      '#6B7280': 'asset-color-gray'
    };
    return colorMap[color] || 'asset-color-gray';
  };

  const portfolioMetrics: PortfolioMetrics = {
    totalValue: 1250000,
    dailyChange: 15420,
    dailyChangePercent: 1.25,
    ytdReturn: 18.7,
    sharpeRatio: 1.34,
    volatility: 16.2,
    maxDrawdown: -8.5,
    beta: 1.12
  };

  const assetAllocation: AssetAllocation[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', percentage: 22.5, value: 281250, sector: 'Technology', color: '#3B82F6' },
    { symbol: 'MSFT', name: 'Microsoft', percentage: 18.3, value: 228750, sector: 'Technology', color: '#8B5CF6' },
    { symbol: 'GOOGL', name: 'Alphabet', percentage: 15.2, value: 190000, sector: 'Technology', color: '#06B6D4' },
    { symbol: 'TSLA', name: 'Tesla', percentage: 12.1, value: 151250, sector: 'Consumer Cyclical', color: '#10B981' },
    { symbol: 'AMZN', name: 'Amazon', percentage: 10.8, value: 135000, sector: 'Consumer Cyclical', color: '#F59E0B' },
    { symbol: 'NVDA', name: 'NVIDIA', percentage: 21.1, value: 263750, sector: 'Technology', color: '#EF4444' }
  ];

  const riskMetrics: RiskMetrics = {
    var95: -45000,
    cvar95: -62000,
    correlation: {
      'AAPL-MSFT': 0.72,
      'AAPL-GOOGL': 0.68,
      'TSLA-BTC': 0.45,
      'TECH-SECTOR': 0.85
    },
    concentration: 76.1
  };

  const tabs = [
    { id: 'portfolio', label: 'Portfolio Overview', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'risk', label: 'Risk Analysis', icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 'performance', label: 'Performance', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'allocation', label: 'Asset Allocation', icon: <PieChart className="w-4 h-4" /> }
  ];

  const timeframes = ['1D', '1W', '1M', '3M', '1Y'];

  const MetricCard: React.FC<{ title: string; value: string; change?: string; positive?: boolean; icon: React.ReactNode }> = 
    ({ title, value, change, positive, icon }) => (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">{title}</span>
        <div className="text-blue-400">{icon}</div>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      {change && (
        <div className={`text-sm ${positive ? 'text-green-400' : 'text-red-400'}`}>
          {positive ? '+' : ''}{change}
        </div>
      )}
    </div>
  );

  const renderPortfolioOverview = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Value"
          value={`$${(portfolioMetrics.totalValue / 1000000).toFixed(2)}M`}
          change={`$${(portfolioMetrics.dailyChange / 1000).toFixed(1)}K`}
          positive={portfolioMetrics.dailyChange > 0}
          icon={<DollarSign className="w-5 h-5" />}
        />
        <MetricCard
          title="Daily Return"
          value={`${portfolioMetrics.dailyChangePercent.toFixed(2)}%`}
          change="vs. S&P 500: +0.85%"
          positive={portfolioMetrics.dailyChangePercent > 0}
          icon={<Percent className="w-5 h-5" />}
        />
        <MetricCard
          title="YTD Return"
          value={`${portfolioMetrics.ytdReturn.toFixed(1)}%`}
          change="Benchmark: +12.3%"
          positive={portfolioMetrics.ytdReturn > 12.3}
          icon={<Calendar className="w-5 h-5" />}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={portfolioMetrics.sharpeRatio.toFixed(2)}
          change="Risk-adj. return"
          positive={portfolioMetrics.sharpeRatio > 1}
          icon={<Target className="w-5 h-5" />}
        />
      </div>

      {/* Performance Chart Placeholder */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Portfolio Performance</h3>
          <div className="flex space-x-2">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf as any)}
                className={`px-3 py-1 rounded text-sm ${
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
        
        {/* Mock Chart */}
        <div className="h-64 bg-gray-900/50 rounded-lg flex items-center justify-center border border-gray-600">
          <div className="text-center">
            <LineChart className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">Interactive performance chart</p>
            <p className="text-sm text-gray-500">Real-time data visualization coming soon</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderRiskAnalysis = () => (
    <div className="space-y-6">
      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Portfolio VaR (95%)"
          value={`$${Math.abs(riskMetrics.var95 / 1000).toFixed(0)}K`}
          change="Daily risk exposure"
          positive={false}
          icon={<AlertTriangle className="w-5 h-5" />}
        />
        <MetricCard
          title="Expected Shortfall"
          value={`$${Math.abs(riskMetrics.cvar95 / 1000).toFixed(0)}K`}
          change="Tail risk (CVaR)"
          positive={false}
          icon={<Activity className="w-5 h-5" />}
        />
        <MetricCard
          title="Volatility"
          value={`${portfolioMetrics.volatility.toFixed(1)}%`}
          change="Annualized"
          positive={portfolioMetrics.volatility < 20}
          icon={<BarChart3 className="w-5 h-5" />}
        />
        <MetricCard
          title="Max Drawdown"
          value={`${portfolioMetrics.maxDrawdown.toFixed(1)}%`}
          change="Peak to trough"
          positive={portfolioMetrics.maxDrawdown > -10}
          icon={<TrendingUp className="w-5 h-5" />}
        />
      </div>

      {/* Risk Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Correlation Matrix</h3>
          <div className="space-y-3">
            {Object.entries(riskMetrics.correlation).map(([pair, corr]) => (
              <div key={pair} className="flex items-center justify-between">
                <span className="text-gray-300">{pair}</span>
                <div className="flex items-center">
                  <div className={`w-20 h-2 rounded mr-2 ${
                    corr > 0.7 ? 'bg-red-500' : corr > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}>
                    <div 
                      className={`h-full bg-white rounded progress-bar-${Math.round(Math.abs(corr) * 10) * 10}`}
                    ></div>
                  </div>
                  <span className="text-white font-mono">{corr.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Factors</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-300">Tech Concentration</span>
                <span className="text-white">{riskMetrics.concentration}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={`bg-red-500 h-2 rounded-full progress-bar-${Math.round(riskMetrics.concentration / 10) * 10}`}
                ></div>
              </div>
              <p className="text-xs text-gray-400 mt-1">High sector concentration risk</p>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-300">Market Beta</span>
                <span className="text-white">{portfolioMetrics.beta}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={`bg-blue-500 h-2 rounded-full progress-bar-${Math.round((portfolioMetrics.beta / 2) * 10) * 10}`}
                ></div>
              </div>
              <p className="text-xs text-gray-400 mt-1">Slightly more volatile than market</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAssetAllocation = () => (
    <div className="space-y-6">
      {/* Allocation Chart */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Current Allocation</h3>
        
        {/* Holdings List */}
        <div className="space-y-3">
          {assetAllocation.map((asset, index) => (
            <div key={asset.symbol} className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div 
                  className={`w-4 h-4 rounded-full ${getColorClass(asset.color)}`}
                ></div>
                <div>
                  <div className="text-white font-medium">{asset.symbol}</div>
                  <div className="text-gray-400 text-sm">{asset.name}</div>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-white font-semibold">{asset.percentage}%</div>
                <div className="text-gray-400 text-sm">${(asset.value / 1000).toFixed(0)}K</div>
              </div>
              
              <div className="text-right">
                <div className="text-gray-300 text-sm">{asset.sector}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sector Breakdown */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Sector Allocation</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-300">Technology</span>
              <span className="text-white font-semibold">76.1%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full progress-bar-70"></div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-300">Consumer Cyclical</span>
              <span className="text-white font-semibold">22.9%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-green-500 h-2 rounded-full progress-bar-20"></div>
            </div>
          </div>
        </div>
        
        <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
          <div className="flex items-center">
            <AlertTriangle className="w-4 h-4 text-yellow-400 mr-2" />
            <span className="text-yellow-200 text-sm">
              High concentration in Technology sector (76.1%). Consider diversification.
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPerformance = () => (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <h4 className="text-gray-400 text-sm mb-2">Alpha</h4>
          <div className="text-2xl font-bold text-green-400">+4.2%</div>
          <p className="text-xs text-gray-500">Excess return vs benchmark</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <h4 className="text-gray-400 text-sm mb-2">Information Ratio</h4>
          <div className="text-2xl font-bold text-blue-400">1.8</div>
          <p className="text-xs text-gray-500">Risk-adjusted active return</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <h4 className="text-gray-400 text-sm mb-2">Tracking Error</h4>
          <div className="text-2xl font-bold text-yellow-400">5.2%</div>
          <p className="text-xs text-gray-500">Deviation from benchmark</p>
        </div>
      </div>

      {/* Attribution Analysis */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Attribution</h3>
        <div className="space-y-3">
          {assetAllocation.slice(0, 5).map((asset, index) => (
            <div key={asset.symbol} className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div 
                  className={`w-3 h-3 rounded-full ${getColorClass(asset.color)}`}
                ></div>
                <span className="text-white">{asset.symbol}</span>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="text-right">
                  <div className="text-white">+2.4%</div>
                  <div className="text-xs text-gray-400">Contribution</div>
                </div>
                
                <div className="text-right">
                  <div className="text-green-400">+15.2%</div>
                  <div className="text-xs text-gray-400">Total Return</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderAllocation = () => (
    <div className="bg-black/40 backdrop-blur-sm rounded-xl border border-gray-700">
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">Advanced Analytics Dashboard</h1>
            <p className="text-gray-400">Comprehensive portfolio analysis and risk management</p>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-300">Live Data</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-500/10'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-700/50'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'portfolio' && renderPortfolioOverview()}
        {activeTab === 'risk' && renderRiskAnalysis()}
        {activeTab === 'performance' && renderPerformance()}
        {activeTab === 'allocation' && renderAssetAllocation()}
      </div>
    </div>
  );

  return renderAllocation();
};

export default AdvancedAnalytics;
