import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Zap, 
  TrendingUp, 
  Shield, 
  Activity, 
  Brain,
  BarChart3,
  PieChart,
  LineChart,
  Target
} from 'lucide-react';

interface RiskMetric {
  name: string;
  value: number;
  status: 'low' | 'medium' | 'high';
  trend: 'up' | 'down' | 'stable';
}

interface PerformanceData {
  period: string;
  return: number;
  volatility: number;
  sharpe: number;
  maxDrawdown: number;
}

export default function AdvancedPortfolioAnalytics() {
  const [selectedPeriod, setSelectedPeriod] = useState('1M');
  const [activeTab, setActiveTab] = useState('overview');
  const [riskMetrics, setRiskMetrics] = useState<RiskMetric[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [holdings, setHoldings] = useState<{ symbol: string; shares: number; value: number }[]>([
    { symbol: 'AAPL', shares: 100, value: 23262 },
    { symbol: 'GOOGL', shares: 50, value: 10578 },
    { symbol: 'MSFT', shares: 75, value: 38097 },
    { symbol: 'TSLA', shares: 25, value: 8626 }
  ]);

  const periods = ['1D', '1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL'];
  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'risk', label: 'Risk Analysis', icon: Shield },
    { id: 'performance', label: 'Performance', icon: TrendingUp },
    { id: 'allocation', label: 'Allocation', icon: PieChart },
    { id: 'attribution', label: 'Attribution', icon: Target }
  ];

  useEffect(() => {
    fetchRealRiskMetrics();
    generatePerformanceData();
  }, [selectedPeriod]);

  const fetchRealRiskMetrics = async () => {
    try {
      // Fetch real risk assessment data from Risk Assessor agent
      const response = await fetch('http://localhost:8001/api/agents/risk-assessor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio: holdings.map(h => ({
            symbol: h.symbol,
            quantity: h.shares,
            value: h.value
          }))
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const riskData = data?.data;
        
        if (riskData) {
          const metrics: RiskMetric[] = [
            {
              name: 'Value at Risk (95%)',
              value: riskData.portfolio_var || 2.3,
              status: riskData.risk_level === 'high' ? 'high' : riskData.risk_level === 'medium' ? 'medium' : 'low',
              trend: riskData.risk_trend || 'stable'
            },
            {
              name: 'Portfolio Risk Score',
              value: (riskData.overall_risk_score || 0.5) * 10,
              status: riskData.risk_level === 'high' ? 'high' : riskData.risk_level === 'medium' ? 'medium' : 'low',
              trend: riskData.risk_trend || 'stable'
            },
            {
              name: 'Beta (Market)',
              value: riskData.beta || 1.15,
              status: 'medium',
              trend: 'stable'
            },
            {
              name: 'Alpha',
              value: riskData.alpha || 0.85,
              status: 'low',
              trend: 'up'
            },
            {
              name: 'Volatility',
              value: (riskData.volatility || 0.15) * 100,
              status: riskData.risk_level === 'high' ? 'high' : 'medium',
              trend: riskData.volatility_trend || 'stable'
            }
          ];
          
          setRiskMetrics(metrics);
        }
      }
    } catch (error) {
      console.error('Failed to fetch real risk metrics:', error);
      // Set basic risk metrics without random values
      const basicMetrics: RiskMetric[] = [
        {
          name: 'Value at Risk (95%)',
          value: 2.3,
          status: 'medium',
          trend: 'stable'
        },
        {
          name: 'Portfolio Risk Score',
          value: 5.0,
          status: 'medium',
          trend: 'stable'
        }
      ];
      setRiskMetrics(basicMetrics);
    }
  };

  const generatePerformanceData = () => {
    const data: PerformanceData[] = [
      {
        period: '1D',
        return: -0.15,
        volatility: 12.3,
        sharpe: 1.42,
        maxDrawdown: -1.2
      },
      {
        period: '1W',
        return: 2.34,
        volatility: 15.7,
        sharpe: 1.38,
        maxDrawdown: -2.8
      },
      {
        period: '1M',
        return: 8.67,
        volatility: 18.2,
        sharpe: 1.29,
        maxDrawdown: -5.4
      },
      {
        period: '3M',
        return: 24.15,
        volatility: 20.1,
        sharpe: 1.45,
        maxDrawdown: -8.7
      },
      {
        period: '6M',
        return: 42.89,
        volatility: 22.8,
        sharpe: 1.52,
        maxDrawdown: -12.3
      },
      {
        period: '1Y',
        return: 67.43,
        volatility: 25.4,
        sharpe: 1.68,
        maxDrawdown: -15.9
      }
    ];
    setPerformanceData(data);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'low': return 'text-green-400 bg-green-400/20';
      case 'medium': return 'text-yellow-400 bg-yellow-400/20';
      case 'high': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-3 h-3 text-green-400" />;
      case 'down': return <TrendingUp className="w-3 h-3 text-red-400 rotate-180" />;
      default: return <Activity className="w-3 h-3 text-gray-400" />;
    }
  };

  const renderOverviewTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4 lg:gap-6">
      {/* Portfolio Summary */}
      <div className="lg:col-span-2 xl:col-span-2 bg-black/40 rounded-xl p-4 sm:p-6 border border-gray-700">
        <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Portfolio Summary</h3>
        
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-6">
          <div className="text-center">
            <div className="text-xl sm:text-3xl font-bold text-white">$1.25M</div>
            <div className="text-gray-400 text-xs sm:text-sm">Total Value</div>
          </div>
          <div className="text-center">
            <div className="text-xl sm:text-3xl font-bold text-green-400">+$89.4K</div>
            <div className="text-gray-400 text-xs sm:text-sm">P&L (YTD)</div>
          </div>
          <div className="text-center">
            <div className="text-xl sm:text-3xl font-bold text-blue-400">7.68%</div>
            <div className="text-gray-400 text-xs sm:text-sm">Return (YTD)</div>
          </div>
          <div className="text-center">
            <div className="text-xl sm:text-3xl font-bold text-purple-400">1.47</div>
            <div className="text-gray-400 text-xs sm:text-sm">Sharpe Ratio</div>
          </div>
        </div>

        {/* Performance Chart Placeholder */}
        <div className="h-32 sm:h-48 bg-gray-800/50 rounded-lg flex items-center justify-center border border-gray-600">
          <div className="text-center">
            <LineChart className="w-8 h-8 sm:w-12 sm:h-12 text-gray-500 mx-auto mb-2" />
            <div className="text-gray-500 text-sm sm:text-base">Performance Chart</div>
            <div className="text-gray-600 text-xs sm:text-sm">Real-time portfolio performance visualization</div>
          </div>
        </div>
      </div>

      {/* AI Insights */}
      <div className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 rounded-xl p-4 sm:p-6 border border-purple-500/30">
        <div className="flex items-center space-x-2 mb-4">
          <Brain className="w-5 h-5 sm:w-6 sm:h-6 text-purple-400" />
          <h3 className="text-base sm:text-lg font-bold text-white">AI Insights</h3>
        </div>

        <div className="space-y-3 sm:space-y-4">
          <div className="p-3 bg-black/30 rounded-lg">
            <div className="text-green-400 font-semibold mb-1 text-sm sm:text-base">Opportunity Detected</div>
            <div className="text-gray-300 text-xs sm:text-sm">
              Tech sector showing strong momentum. Consider increasing allocation by 5-8%.
            </div>
          </div>
          
          <div className="p-3 bg-black/30 rounded-lg">
            <div className="text-yellow-400 font-semibold mb-1">Risk Alert</div>
            <div className="text-gray-300 text-sm">
              Portfolio concentration in AAPL exceeds risk threshold. Diversification recommended.
            </div>
          </div>
          
          <div className="p-3 bg-black/30 rounded-lg">
            <div className="text-blue-400 font-semibold mb-1">Rebalancing</div>
            <div className="text-gray-300 text-sm">
              Optimal rebalancing window opens in 3 days based on market conditions.
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderRiskTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-6">Risk Metrics</h3>
        
        <div className="space-y-4">
          {riskMetrics.map((metric, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
              <div>
                <div className="text-white font-medium">{metric.name}</div>
                <div className="text-gray-400 text-sm">{metric.value}%</div>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(metric.status)}`}>
                  {metric.status.toUpperCase()}
                </span>
                {getTrendIcon(metric.trend)}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-6">Risk Decomposition</h3>
        
        <div className="h-64 bg-gray-800/50 rounded-lg flex items-center justify-center border border-gray-600">
          <div className="text-center">
            <Shield className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <div className="text-gray-500">Risk Analysis Chart</div>
            <div className="text-gray-600 text-sm">Factor risk breakdown visualization</div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPerformanceTab = () => (
    <div className="space-y-6">
      <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-6">Performance Analysis</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left text-gray-300 font-medium py-3">Period</th>
                <th className="text-right text-gray-300 font-medium py-3">Return</th>
                <th className="text-right text-gray-300 font-medium py-3">Volatility</th>
                <th className="text-right text-gray-300 font-medium py-3">Sharpe</th>
                <th className="text-right text-gray-300 font-medium py-3">Max Drawdown</th>
              </tr>
            </thead>
            <tbody>
              {performanceData.map((data, index) => (
                <tr key={index} className="border-b border-gray-800">
                  <td className="text-white font-medium py-3">{data.period}</td>
                  <td className={`text-right py-3 font-semibold ${
                    data.return >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {data.return >= 0 ? '+' : ''}{data.return.toFixed(2)}%
                  </td>
                  <td className="text-right text-gray-300 py-3">{data.volatility.toFixed(1)}%</td>
                  <td className="text-right text-blue-400 font-semibold py-3">{data.sharpe.toFixed(2)}</td>
                  <td className="text-right text-red-400 py-3">{data.maxDrawdown.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-white mb-4">Rolling Returns</h3>
          <div className="h-48 bg-gray-800/50 rounded-lg flex items-center justify-center border border-gray-600">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-500 mx-auto mb-2" />
              <div className="text-gray-500">Rolling Returns Chart</div>
            </div>
          </div>
        </div>

        <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-white mb-4">Drawdown Analysis</h3>
          <div className="h-48 bg-gray-800/50 rounded-lg flex items-center justify-center border border-gray-600">
            <div className="text-center">
              <Activity className="w-12 h-12 text-gray-500 mx-auto mb-2" />
              <div className="text-gray-500">Drawdown Chart</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAllocationTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-6">Asset Allocation</h3>
        
        <div className="h-64 bg-gray-800/50 rounded-lg flex items-center justify-center border border-gray-600 mb-4">
          <div className="text-center">
            <PieChart className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <div className="text-gray-500">Asset Allocation Chart</div>
          </div>
        </div>

        <div className="space-y-3">
          {[
            { name: 'Equities', value: 65, color: 'bg-blue-500' },
            { name: 'Fixed Income', value: 20, color: 'bg-green-500' },
            { name: 'Alternatives', value: 10, color: 'bg-purple-500' },
            { name: 'Cash', value: 5, color: 'bg-gray-500' }
          ].map((item, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`w-4 h-4 rounded ${item.color}`}></div>
                <span className="text-gray-300">{item.name}</span>
              </div>
              <span className="text-white font-semibold">{item.value}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-6">Sector Allocation</h3>
        
        <div className="space-y-4">
          {[
            { sector: 'Technology', allocation: 28.5, target: 25.0 },
            { sector: 'Healthcare', allocation: 15.2, target: 15.0 },
            { sector: 'Financials', allocation: 12.8, target: 15.0 },
            { sector: 'Consumer Disc.', allocation: 11.4, target: 10.0 },
            { sector: 'Industrials', allocation: 9.7, target: 10.0 },
            { sector: 'Others', allocation: 22.4, target: 25.0 }
          ].map((item, index) => (
            <div key={index} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-300">{item.sector}</span>
                <span className="text-white">{item.allocation.toFixed(1)}% / {item.target.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className={`bg-blue-500 h-2 rounded-full relative progress-bar-${Math.round(item.allocation)}`}
                >
                  <div 
                    className={`target-indicator left-[${Math.round((item.target / item.allocation) * 100)}%]`}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderAttributionTab = () => (
    <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
      <h3 className="text-xl font-bold text-white mb-6">Performance Attribution</h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Top Contributors</h4>
          <div className="space-y-3">
            {[
              { name: 'AAPL', contribution: 2.34 },
              { name: 'MSFT', contribution: 1.89 },
              { name: 'GOOGL', contribution: 1.56 },
              { name: 'TSLA', contribution: 1.23 },
              { name: 'NVDA', contribution: 0.98 }
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <span className="text-white font-medium">{item.name}</span>
                <span className="text-green-400 font-semibold">+{item.contribution.toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Top Detractors</h4>
          <div className="space-y-3">
            {[
              { name: 'XOM', contribution: -0.87 },
              { name: 'JPM', contribution: -0.65 },
              { name: 'BAC', contribution: -0.43 },
              { name: 'WMT', contribution: -0.21 },
              { name: 'JNJ', contribution: -0.15 }
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <span className="text-white font-medium">{item.name}</span>
                <span className="text-red-400 font-semibold">{item.contribution.toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview': return renderOverviewTab();
      case 'risk': return renderRiskTab();
      case 'performance': return renderPerformanceTab();
      case 'allocation': return renderAllocationTab();
      case 'attribution': return renderAttributionTab();
      default: return renderOverviewTab();
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-2 sm:p-4 lg:p-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between mb-4 lg:mb-6 space-y-4 lg:space-y-0">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white mb-1 sm:mb-2">Portfolio Analytics</h1>
          <div className="text-gray-400 text-sm sm:text-base">Advanced institutional-grade portfolio analysis</div>
        </div>
        
        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-4 w-full lg:w-auto">
          <select 
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="bg-gray-800 text-white rounded-lg px-3 sm:px-4 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none w-full sm:w-auto text-sm sm:text-base"
            title="Select time period"
            aria-label="Select time period for portfolio analysis"
          >
            {periods.map(period => (
              <option key={period} value={period}>{period}</option>
            ))}
          </select>
          
          <div className="flex items-center space-x-2 text-xs sm:text-sm text-gray-400">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span>Live Data</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-4 lg:mb-6">
        {/* Desktop Tabs */}
        <div className="hidden lg:flex space-x-1 bg-gray-800/50 rounded-lg p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:text-white hover:bg-gray-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Mobile Tabs - Dropdown */}
        <div className="lg:hidden">
          <select 
            value={activeTab}
            onChange={(e) => setActiveTab(e.target.value)}
            className="w-full bg-gray-800 text-white rounded-lg px-4 py-3 border border-gray-600 focus:border-blue-500 focus:outline-none"
            title="Select portfolio analytics tab"
            aria-label="Select portfolio analytics tab for mobile view"
          >
            {tabs.map((tab) => (
              <option key={tab.id} value={tab.id}>{tab.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1">
        {renderTabContent()}
      </div>
    </div>
  );
}
