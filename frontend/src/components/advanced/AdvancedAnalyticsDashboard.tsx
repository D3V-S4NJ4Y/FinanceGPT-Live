import React, { useState, useEffect, useCallback } from 'react';
import './AdvancedAnalytics.css';
import {
  TrendingUp, TrendingDown, PieChart, BarChart3, Shield, Target,
  DollarSign, Percent, Activity, AlertTriangle, RefreshCw, Calendar, Settings
} from 'lucide-react';
import PortfolioEditor from '../PortfolioEditor';
import { portfolioService, type RealPortfolioData, hasPortfolioData } from '../../services/portfolioService';

interface PortfolioHolding {
  symbol: string;
  shares: number;
  avgCost: number;
  currentPrice: number;
  marketValue: number;
  totalReturn: number;
  totalReturnPercent: number;
  dayChange: number;
  dayChangePercent: number;
  weight: number;
}

interface PortfolioMetrics {
  totalValue: number;
  totalCost: number;
  totalReturn: number;
  totalReturnPercent: number;
  dayChange: number;
  dayChangePercent: number;
  sharpeRatio: number;
  beta: number;
  volatility: number;
  maxDrawdown: number;
}

interface RiskMetrics {
  portfolioRisk: number;
  diversificationScore: number;
  concentrationRisk: number;
  sectorExposure: Record<string, number>;
  correlationRisk: number;
}

const API_BASE_URL = 'http://127.0.0.1:8001';

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
    console.warn(`API ${endpoint} failed:`, error);
    return null;
  }
};

export default function AdvancedAnalyticsDashboard() {
  const [portfolio, setPortfolio] = useState<PortfolioHolding[]>([]);
  const [metrics, setMetrics] = useState<PortfolioMetrics>({
    totalValue: 0,
    totalCost: 0,
    totalReturn: 0,
    totalReturnPercent: 0,
    dayChange: 0,
    dayChangePercent: 0,
    sharpeRatio: 0,
    beta: 0,
    volatility: 0,
    maxDrawdown: 0
  });
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    portfolioRisk: 0,
    diversificationScore: 0,
    concentrationRisk: 0,
    sectorExposure: {},
    correlationRisk: 0
  });
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState('');
  const [timeframe, setTimeframe] = useState('1D');
  const [showPortfolioEditor, setShowPortfolioEditor] = useState(false);
  const [hasPortfolio, setHasPortfolio] = useState(false);

  const [realPortfolioData, setRealPortfolioData] = useState<RealPortfolioData | null>(null);

  const fetchPortfolioData = useCallback(async (selectedTimeframe?: string) => {
    const currentTimeframe = selectedTimeframe || timeframe;
    try {
      const userPortfolio = portfolioService.getDefaultPortfolio();
      
      // If no portfolio, fetch market data directly
      if (userPortfolio.length === 0) {
        await fetchMarketDataDirectly();
        return;
      }
      
      const realData = await portfolioService.fetchRealPortfolioData(userPortfolio, currentTimeframe);
      
      setRealPortfolioData(realData);
      
      // Update state with real data
      setPortfolio(realData.holdings);
      setMetrics({
        totalValue: realData.totalValue,
        totalCost: realData.totalCost,
        totalReturn: realData.totalReturn,
        totalReturnPercent: realData.totalReturnPercent,
        dayChange: realData.dayChange,
        dayChangePercent: realData.dayChangePercent,
        sharpeRatio: realData.sharpeRatio,
        beta: realData.beta,
        volatility: realData.volatility,
        maxDrawdown: realData.maxDrawdown
      });
      
      setRiskMetrics(realData.riskMetrics);
      
    } catch (error) {
      console.warn('Failed to fetch portfolio data:', error);
      await fetchMarketDataDirectly();
    }
  }, [timeframe]);

  const fetchMarketDataDirectly = useCallback(async () => {
    try {
      const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'];
      const marketResponse = await apiCall(`/api/market/quotes?symbols=${symbols.join(',')}`);
      
      if (marketResponse?.success && marketResponse.data) {
        const holdings: PortfolioHolding[] = [];
        
        Object.entries(marketResponse.data).forEach(([symbol, data]: [string, any]) => {
          if (data && !data.error) {
            holdings.push({
              symbol,
              shares: 0,
              avgCost: data.price,
              currentPrice: data.price,
              marketValue: 0,
              totalReturn: 0,
              totalReturnPercent: 0,
              dayChange: data.change || 0,
              dayChangePercent: data.change_percent || 0,
              weight: 0
            });
          }
        });
        
        setPortfolio(holdings);
        setMetrics({
          totalValue: 0,
          totalCost: 0,
          totalReturn: 0,
          totalReturnPercent: 0,
          dayChange: 0,
          dayChangePercent: 0,
          sharpeRatio: 0,
          beta: 0,
          volatility: 0,
          maxDrawdown: 0
        });
        
        setRiskMetrics({
          portfolioRisk: 0,
          diversificationScore: 0,
          concentrationRisk: 0,
          sectorExposure: {},
          correlationRisk: 0
        });
      }
    } catch (error) {
      console.warn('Market data fetch failed:', error);
    }
  }, []);

  const calculateRiskMetrics = (holdings: PortfolioHolding[], totalValue: number) => {
    // Concentration risk (Herfindahl index)
    const concentrationRisk = holdings.reduce((sum, h) => {
      const weight = h.weight / 100;
      return sum + weight * weight;
    }, 0) * 100;

    // Diversification score (inverse of concentration)
    const diversificationScore = Math.max(0, 100 - concentrationRisk * 2);

    // Portfolio risk (weighted average of individual risks)
    const portfolioRisk = holdings.reduce((sum, h) => {
      const individualRisk = Math.abs(h.totalReturnPercent) > 20 ? 80 : 
                           Math.abs(h.totalReturnPercent) > 10 ? 60 : 40;
      return sum + (h.weight / 100) * individualRisk;
    }, 0);

    // Sector exposure (simplified)
    const sectorExposure: Record<string, number> = {
      'Technology': 0,
      'Consumer Discretionary': 0,
      'Communication Services': 0,
      'Automotive': 0
    };

    holdings.forEach(h => {
      if (['AAPL', 'MSFT', 'GOOGL', 'NVDA'].includes(h.symbol)) {
        sectorExposure['Technology'] += h.weight;
      } else if (['AMZN'].includes(h.symbol)) {
        sectorExposure['Consumer Discretionary'] += h.weight;
      } else if (['TSLA'].includes(h.symbol)) {
        sectorExposure['Automotive'] += h.weight;
      }
    });

    // Correlation risk (simplified - high tech concentration)
    const techWeight = sectorExposure['Technology'];
    const correlationRisk = techWeight > 60 ? 80 : techWeight > 40 ? 60 : 40;

    setRiskMetrics({
      portfolioRisk,
      diversificationScore,
      concentrationRisk,
      sectorExposure,
      correlationRisk
    });
  };

  useEffect(() => {
    const fetchData = async () => {
      setHasPortfolio(hasPortfolioData());
      await fetchPortfolioData();
      setLastUpdate(new Date().toLocaleTimeString());
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [fetchPortfolioData]);

  // Animate progress bars when data updates
  useEffect(() => {
    const animateBars = () => {
      // Animate risk bars
      const riskBars = document.querySelectorAll('.risk-progress-bar');
      const diversificationBars = document.querySelectorAll('.diversification-progress-bar');
      const concentrationBars = document.querySelectorAll('.concentration-progress-bar');
      const sectorBars = document.querySelectorAll('.sector-progress-bar');

      // Reset all bars to 0 width first
      [...riskBars, ...diversificationBars, ...concentrationBars, ...sectorBars].forEach(bar => {
        const barElement = bar as HTMLElement;
        barElement.style.width = '0%';
      });

      // Animate with delay
      setTimeout(() => {
        riskBars.forEach(bar => {
          const barElement = bar as HTMLElement;
          const width = barElement.getAttribute('data-target-width');
          if (width) {
            barElement.style.width = `${width}%`;
          }
        });
        
        diversificationBars.forEach(bar => {
          const barElement = bar as HTMLElement;
          const width = barElement.getAttribute('data-target-width');
          if (width) {
            barElement.style.width = `${width}%`;
          }
        });
        
        concentrationBars.forEach(bar => {
          const barElement = bar as HTMLElement;
          const width = barElement.getAttribute('data-target-width');
          if (width) {
            barElement.style.width = `${width}%`;
          }
        });
        
        sectorBars.forEach(bar => {
          const barElement = bar as HTMLElement;
          const width = barElement.getAttribute('data-target-width');
          if (width) {
            barElement.style.width = `${width}%`;
          }
        });
      }, 300);
    };

    if (riskMetrics.portfolioRisk > 0) {
      animateBars();
    }
  }, [riskMetrics]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatLargeNumber = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(1)}K`;
    }
    return formatCurrency(value);
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-2 sm:p-4 lg:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between mb-4 sm:mb-6 space-y-4 lg:space-y-0">
          <div>
            <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-white mb-2">Advanced Analytics</h1>
            <div className="text-gray-400 text-xs sm:text-sm">
              <span className="hidden sm:inline">Comprehensive portfolio analysis and risk management • </span>
              Timeframe: {timeframe}
            </div>
            {!hasPortfolio && (
              <div className="mt-2 px-2 sm:px-3 py-1 bg-blue-600/20 border border-blue-500/30 rounded-lg text-blue-300 text-xs flex items-center">
                <Target className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
                Add portfolio for analysis
              </div>
            )}
          </div>
          
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 w-full sm:w-auto">
            <div className="flex items-center justify-center sm:justify-start space-x-1 sm:space-x-2 bg-gray-800 rounded-lg p-1 overflow-x-auto">
              {['1D', '1W', '1M', '3M', '1Y'].map((period) => (
                <button
                  key={period}
                  onClick={async () => {
                    setTimeframe(period);
                    setLoading(true);
                    await fetchPortfolioData(period);
                    setLoading(false);
                    setLastUpdate(new Date().toLocaleTimeString());
                  }}
                  className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm font-medium transition-all duration-200 flex-shrink-0 ${
                    timeframe === period
                      ? 'bg-blue-600 text-white shadow-lg transform scale-105'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  {period}
                </button>
              ))}
            </div>
            
            <div className="flex items-center space-x-2 sm:space-x-4">
              <button
                onClick={() => setShowPortfolioEditor(true)}
                className="flex items-center justify-center space-x-1 sm:space-x-2 px-3 sm:px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors flex-1 sm:flex-initial"
              >
                <Settings className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">Edit Portfolio</span>
                <span className="sm:hidden">Edit</span>
              </button>
              
              <button
                onClick={async () => {
                  setLoading(true);
                  await fetchPortfolioData();
                  setLoading(false);
                  setLastUpdate(new Date().toLocaleTimeString());
                }}
                className="flex items-center justify-center space-x-1 sm:space-x-2 px-3 sm:px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors flex-1 sm:flex-initial"
                disabled={loading}
              >
                <RefreshCw className={`w-3 h-3 sm:w-4 sm:h-4 ${loading ? 'animate-spin' : ''}`} />
                <span className="hidden sm:inline">{loading ? 'Loading...' : 'Refresh'}</span>
                <span className="sm:hidden">↻</span>
              </button>
            </div>
            
            <div className="flex items-center justify-center sm:justify-start space-x-2 text-xs sm:text-sm text-gray-400 mt-3 sm:mt-0">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
              <span className="hidden sm:inline">Live Data •</span>
              <span>{lastUpdate}</span>
            </div>
          </div>
        </div>

        {/* Portfolio Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4 mb-4 sm:mb-6">
          <div className="bg-black/40 backdrop-blur-sm rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Total Value</p>
                <p className="text-lg sm:text-xl font-bold text-white">{formatLargeNumber(metrics.totalValue)}</p>
                <p className={`text-xs sm:text-sm font-medium ${metrics.dayChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(metrics.dayChangePercent)}
                </p>
              </div>
              <DollarSign className="w-6 h-6 text-green-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Daily Return</p>
                <p className={`text-xl font-bold ${metrics.dayChangePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(metrics.dayChangePercent)}
                </p>
                <p className="text-xs text-gray-400">vs. S&P 500: {formatPercent(metrics.dayChangePercent - 0.85)}</p>
              </div>
              <Percent className="w-6 h-6 text-blue-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">YTD Return</p>
                <p className={`text-xl font-bold ${metrics.totalReturnPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(metrics.totalReturnPercent)}
                </p>
                <p className="text-xs text-gray-400">vs. Benchmark: {formatPercent(metrics.totalReturnPercent - 12.3)}</p>
              </div>
              <TrendingUp className="w-6 h-6 text-purple-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Sharpe Ratio</p>
                <p className="text-xl font-bold text-white">{metrics.sharpeRatio.toFixed(2)}</p>
                <p className="text-xs text-gray-400">Risk-adj. return</p>
              </div>
              <Target className="w-6 h-6 text-cyan-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Beta</p>
                <p className="text-xl font-bold text-white">{metrics.beta.toFixed(2)}</p>
                <p className="text-xs text-gray-400">Market correlation</p>
              </div>
              <Activity className="w-6 h-6 text-orange-400" />
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs">Max Drawdown</p>
                <p className="text-xl font-bold text-red-400">{metrics.maxDrawdown.toFixed(1)}%</p>
                <p className="text-xs text-gray-400">Worst decline</p>
              </div>
              <TrendingDown className="w-6 h-6 text-red-400" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Portfolio Holdings */}
          <div className="xl:col-span-2">
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <PieChart className="w-5 h-5 mr-2 text-green-400" />
                Portfolio Holdings
              </h3>
              
              <div className="space-y-3">
                {portfolio.map((holding) => (
                  <div key={holding.symbol} className="p-4 bg-gray-800/50 rounded-lg border border-gray-600">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className="text-white font-bold text-lg">{holding.symbol}</span>
                        <span className="text-gray-400 text-sm">{holding.shares} shares</span>
                        <span className="text-xs px-2 py-1 bg-blue-600 text-white rounded">
                          {holding.weight.toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-white font-semibold">{formatCurrency(holding.marketValue)}</div>
                        <div className="text-gray-400 text-sm">${holding.currentPrice.toFixed(2)}</div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-300">Avg Cost:</span>
                        <div className="text-white font-medium">${holding.avgCost.toFixed(2)}</div>
                      </div>
                      
                      <div>
                        <span className="text-gray-300">Total Return:</span>
                        <div className={`font-medium ${
                          holding.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {formatCurrency(holding.totalReturn)}
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-gray-300">Return %:</span>
                        <div className={`font-medium ${
                          holding.totalReturnPercent >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {formatPercent(holding.totalReturnPercent)}
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-gray-300">Day Change:</span>
                        <div className={`font-medium ${
                          holding.dayChange >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {formatPercent(holding.dayChangePercent)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Analysis */}
          <div className="xl:col-span-1">
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mb-6">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <Shield className="w-5 h-5 mr-2 text-red-400" />
                Risk Analysis
              </h3>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-300 text-sm">Portfolio Risk</span>
                    <span className={`font-semibold ${
                      riskMetrics.portfolioRisk > 70 ? 'text-red-400' :
                      riskMetrics.portfolioRisk > 50 ? 'text-yellow-400' : 'text-green-400'
                    }`}>
                      {riskMetrics.portfolioRisk.toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 ease-out risk-progress-bar ${
                        riskMetrics.portfolioRisk > 70 ? 'bg-red-400' :
                        riskMetrics.portfolioRisk > 50 ? 'bg-yellow-400' : 'bg-green-400'
                      }`}
                      data-target-width={riskMetrics.portfolioRisk}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-300 text-sm">Diversification</span>
                    <span className="text-blue-400 font-semibold">{riskMetrics.diversificationScore.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full transition-all duration-1000 ease-out bg-blue-400 diversification-progress-bar"
                      data-target-width={riskMetrics.diversificationScore}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-300 text-sm">Concentration Risk</span>
                    <span className="text-orange-400 font-semibold">{riskMetrics.concentrationRisk.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full transition-all duration-1000 ease-out bg-orange-400 concentration-progress-bar"
                      data-target-width={riskMetrics.concentrationRisk}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-300 text-sm">Volatility</span>
                    <span className="text-purple-400 font-semibold">{metrics.volatility.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Asset Allocation */}
            <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-cyan-400" />
                Asset Allocation
              </h3>
              
              <div className="space-y-3">
                {Object.entries(riskMetrics.sectorExposure).map(([sector, weight]) => (
                  weight > 0 && (
                    <div key={sector} className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm">{sector}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-gray-700 rounded-full h-2">
                          <div 
                            className="h-2 rounded-full transition-all duration-1000 ease-out bg-gradient-to-r from-blue-500 to-purple-500 sector-progress-bar"
                            data-target-width={(weight / Math.max(...Object.values(riskMetrics.sectorExposure))) * 100}
                          ></div>
                        </div>
                        <span className="text-white font-medium text-sm">{weight.toFixed(1)}%</span>
                      </div>
                    </div>
                  )
                ))}
              </div>
              
              <div className="mt-4 p-3 bg-gray-800/30 rounded-lg">
                <div className="text-xs text-gray-400 mb-1">Performance Summary</div>
                <div className="text-sm text-white">
                  {metrics.totalReturnPercent > 12.3 ? 
                    `Portfolio outperforming benchmark by ${(metrics.totalReturnPercent - 12.3).toFixed(1)}% YTD` :
                    `Portfolio underperforming benchmark by ${(12.3 - metrics.totalReturnPercent).toFixed(1)}% YTD`
                  }
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Total Value: {formatLargeNumber(metrics.totalValue)} | Risk Score: {(riskMetrics.portfolioRisk).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <PortfolioEditor
        isOpen={showPortfolioEditor}
        onClose={() => setShowPortfolioEditor(false)}
        onSave={() => {
          setHasPortfolio(true);
          fetchPortfolioData();
        }}
      />
    </div>
  );
}