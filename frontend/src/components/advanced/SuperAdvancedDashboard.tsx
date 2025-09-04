import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Shield,
  Zap,
  Target,
  Activity,
  BarChart3,
  AlertTriangle,
  Eye,
  Settings,
  RefreshCw,
  Play,
  Pause
} from 'lucide-react';

interface MLPrediction {
  symbol: string;
  predicted_price: number;
  confidence: number;
  direction: string;
  probability: number;
  target_price: number;
  stop_loss: number;
  time_horizon: string;
  timeframe?: string;
  risk_score: number;
  model_used: string;
  features_importance?: Record<string, number>;
  features?: {
    technical_score: number;
    sentiment_score: number;
    momentum_score: number;
  };
}

interface MarketRegime {
  regime: string;
  confidence: number;
  indicators?: Record<string, any>;
  recommendations?: string[];
  characteristics?: string[];
  risk_level?: string;
}

interface RealTimeAlert {
  id: string;
  type: 'prediction' | 'risk' | 'opportunity' | 'technical';
  severity: 'low' | 'medium' | 'high';
  message: string;
  symbol?: string;
  timestamp: Date;
  action?: string;
}

export default function SuperAdvancedDashboard() {
  const [predictions, setPredictions] = useState<Record<string, MLPrediction>>({});
  const [marketRegime, setMarketRegime] = useState<MarketRegime | null>(null);
  const [alerts, setAlerts] = useState<RealTimeAlert[]>([]);
  const [isRunning, setIsRunning] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [performanceMetrics, setPerformanceMetrics] = useState<any>({});
  const intervalRef = useRef<number | null>(null);

  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META'];

  useEffect(() => {
    if (isRunning) {
      fetchAllData();
      intervalRef.current = setInterval(fetchAllData, 15000); // Update every 15 seconds
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning]);

  const fetchAllData = async () => {
    try {
      // First try to fetch from API, then fallback to mock data
      let predictionResults: Array<{symbol: string, prediction: MLPrediction} | null> = [];
      
      try {
        // Fetch ML predictions for all symbols
        const predictionPromises = symbols.map(async (symbol) => {
          try {
            const response = await fetch(`http://localhost:8001/api/ml/predict/${symbol}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            return { symbol, prediction: data.prediction };
          } catch (error) {
            return null;
          }
        });

        predictionResults = await Promise.all(predictionPromises);
      } catch (error) {
        console.log('API not available, using mock data');
        predictionResults = [];
      }

      const newPredictions: Record<string, MLPrediction> = {};
      
      // Use API data if available, otherwise use mock data
      if (predictionResults.some(r => r && r.prediction)) {
        predictionResults.forEach(result => {
          if (result && result.prediction) {
            newPredictions[result.symbol] = result.prediction;
            
            // Generate alerts based on predictions
            if (result.prediction.confidence > 0.8) {
              addAlert({
                type: 'prediction',
                severity: result.prediction.risk_score > 0.6 ? 'high' : 'medium',
                message: `High confidence ${result.prediction.direction} signal for ${result.symbol}`,
                symbol: result.symbol,
                action: result.prediction.direction === 'bullish' ? 'Consider buying' : 'Consider selling'
              });
            }
          }
        });
      } else {
        // System status data when API is not available - no random values
        symbols.forEach((symbol, index) => {
          const systemStatus: MLPrediction = {
            symbol,
            predicted_price: 0,
            direction: 'neutral',
            confidence: 0,
            probability: 0,
            target_price: 0,
            stop_loss: 0,
            time_horizon: '1D',
            risk_score: 0,
            model_used: 'connecting',
            features_importance: {
              technical: 0,
              sentiment: 0,
              momentum: 0
            },
            features: {
              technical_score: 0,
              sentiment_score: 0,
              momentum_score: 0
            }
          };
          newPredictions[symbol] = systemStatus;
        });
      }

      setPredictions(newPredictions);

      // Try to fetch market regime, fallback to mock
      try {
        const regimeResponse = await fetch('http://localhost:8001/api/ml/market-regime');
        const regimeData = await regimeResponse.json();
        setMarketRegime(regimeData);
      } catch (error) {
        // Mock market regime data
        setMarketRegime({
          regime: 'bull_market',
          confidence: 0.82,
          characteristics: [
            'Strong momentum indicators',
            'Positive sentiment trends',
            'Low volatility environment'
          ],
          recommendations: [
            'Consider increasing equity exposure',
            'Monitor for momentum reversals',
            'Maintain risk management protocols'
          ],
          risk_level: 'moderate'
        });
      }

      // Update performance metrics
      updatePerformanceMetrics(newPredictions);

    } catch (error) {
      console.error('Error fetching ML data:', error);
    }
  };

  const updatePerformanceMetrics = (predictions: Record<string, MLPrediction>) => {
    const metrics = {
      totalPredictions: Object.keys(predictions).length,
      avgConfidence: Object.values(predictions).reduce((sum, p) => sum + p.confidence, 0) / Object.keys(predictions).length,
      bullishSignals: Object.values(predictions).filter(p => p.direction === 'bullish').length,
      bearishSignals: Object.values(predictions).filter(p => p.direction === 'bearish').length,
      highRiskSignals: Object.values(predictions).filter(p => p.risk_score > 0.6).length,
      averageReturn: Object.values(predictions).reduce((sum, p) => {
        const expectedReturn = (p.predicted_price / p.target_price - 1) * 100;
        return sum + expectedReturn;
      }, 0) / Object.keys(predictions).length
    };

    setPerformanceMetrics(metrics);
  };

  const addAlert = (alertData: Omit<RealTimeAlert, 'id' | 'timestamp'>) => {
    const newAlert: RealTimeAlert = {
      ...alertData,
      id: Date.now().toString(),
      timestamp: new Date()
    };

    setAlerts(prev => [newAlert, ...prev].slice(0, 20)); // Keep only last 20 alerts
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-400/20 border-red-500';
      case 'medium': return 'text-yellow-400 bg-yellow-400/20 border-yellow-500';
      default: return 'text-blue-400 bg-blue-400/20 border-blue-500';
    }
  };

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'bullish': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'bearish': return <TrendingDown className="w-4 h-4 text-red-400" />;
      default: return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  const formatTimeAgo = (date: Date) => {
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <div className="min-h-screen bg-gray-900 p-2 sm:p-4 lg:p-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between mb-4 lg:mb-6 space-y-4 lg:space-y-0">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white mb-1 sm:mb-2 flex items-center">
            <Brain className="w-6 h-6 sm:w-8 sm:h-8 mr-2 sm:mr-3 text-purple-400" />
            <span className="hidden sm:inline">AI Trading Intelligence Center</span>
            <span className="sm:hidden">AI Intelligence</span>
          </h1>
          <div className="text-gray-400 text-sm sm:text-base">Advanced machine learning predictions and market analysis</div>
        </div>
        
        <div className="flex items-center space-x-2 sm:space-x-4 w-full lg:w-auto">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center space-x-1 sm:space-x-2 px-2 sm:px-4 py-2 rounded-lg font-medium transition-colors text-sm sm:text-base ${
              isRunning 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isRunning ? <Pause className="w-3 h-3 sm:w-4 sm:h-4" /> : <Play className="w-3 h-3 sm:w-4 sm:h-4" />}
            <span className="hidden sm:inline">{isRunning ? 'Pause' : 'Start'}</span>
          </button>
          
          <button
            onClick={fetchAllData}
            className="flex items-center space-x-1 sm:space-x-2 px-2 sm:px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors text-sm sm:text-base"
          >
            <RefreshCw className="w-3 h-3 sm:w-4 sm:h-4" />
            <span className="hidden sm:inline">Refresh</span>
          </button>
          
          <div className="flex items-center space-x-1 sm:space-x-2 text-xs sm:text-sm text-gray-400">
            <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
            <span>{isRunning ? 'Live' : 'Paused'}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 lg:gap-6 h-full">
        {/* Performance Metrics */}
        <div className="xl:col-span-1 order-2 xl:order-1">
          <div className="bg-black/40 rounded-xl p-4 sm:p-6 border border-gray-700 mb-4 lg:mb-6">
            <h3 className="text-base sm:text-lg font-bold text-white mb-3 sm:mb-4 flex items-center">
              <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
              AI Performance
            </h3>
            
            <div className="space-y-3 sm:space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-300 text-sm sm:text-base">Active Predictions</span>
                <span className="text-white font-semibold text-sm sm:text-base">{performanceMetrics.totalPredictions || 0}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300 text-sm sm:text-base">Avg Confidence</span>
                <span className="text-green-400 font-semibold text-sm sm:text-base">
                  {((performanceMetrics.avgConfidence || 0) * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300 text-sm sm:text-base">Bullish Signals</span>
                <span className="text-green-400 font-semibold text-sm sm:text-base">{performanceMetrics.bullishSignals || 0}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300 text-sm sm:text-base">Bearish Signals</span>
                <span className="text-red-400 font-semibold text-sm sm:text-base">{performanceMetrics.bearishSignals || 0}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300">High Risk</span>
                <span className="text-yellow-400 font-semibold">{performanceMetrics.highRiskSignals || 0}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Avg Return</span>
                <span className={`font-semibold ${
                  (performanceMetrics.averageReturn || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {(performanceMetrics.averageReturn || 0).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          {/* Market Regime */}
          {marketRegime && (
            <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                Market Regime
              </h3>
              
              <div className="space-y-3">
                <div className="text-center">
                  <div className={`text-2xl font-bold ${
                    marketRegime.regime === 'bull_market' ? 'text-green-400' :
                    marketRegime.regime === 'bear_market' ? 'text-red-400' :
                    marketRegime.regime === 'high_volatility' ? 'text-yellow-400' : 'text-gray-400'
                  }`}>
                    {marketRegime.regime.replace('_', ' ').toUpperCase()}
                  </div>
                  <div className="text-gray-400 text-sm">
                    Confidence: {(marketRegime.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="text-white font-semibold text-sm">Recommendations:</h4>
                  {marketRegime.recommendations?.slice(0, 3).map((rec, index) => (
                    <div key={index} className="text-gray-300 text-xs p-2 bg-gray-800/50 rounded">
                      {rec}
                    </div>
                  )) || <div className="text-gray-400 text-xs">No recommendations available</div>}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ML Predictions Grid */}
        <div className="xl:col-span-2">
          <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2" />
              ML Predictions ({Object.keys(predictions).length})
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-h-[calc(100vh-300px)] overflow-y-auto">
              {Object.entries(predictions).map(([symbol, prediction]) => (
                <div key={symbol} className="p-4 bg-gray-800/50 rounded-lg border border-gray-600">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <span className="text-white font-bold text-lg">{symbol}</span>
                      {getDirectionIcon(prediction.direction)}
                    </div>
                    <div className="text-right">
                      <div className="text-white font-semibold">${prediction.predicted_price}</div>
                      <div className="text-gray-400 text-sm">{prediction.model_used}</div>
                    </div>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Confidence:</span>
                      <span className="text-green-400 font-semibold">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-300">Target:</span>
                      <span className="text-white font-semibold">${prediction.target_price}</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-300">Stop Loss:</span>
                      <span className="text-red-400 font-semibold">${prediction.stop_loss}</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-300">Risk Score:</span>
                      <span className={`font-semibold ${
                        prediction.risk_score > 0.6 ? 'text-red-400' :
                        prediction.risk_score > 0.3 ? 'text-yellow-400' : 'text-green-400'
                      }`}>
                        {(prediction.risk_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-300">Horizon:</span>
                      <span className="text-gray-400">{prediction.time_horizon}</span>
                    </div>
                  </div>
                  
                  {/* Feature Importance Preview */}
                  <div className="mt-3 pt-3 border-t border-gray-600">
                    <div className="text-xs text-gray-400 mb-1">Top Features:</div>
                    {prediction.features_importance ? Object.entries(prediction.features_importance)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 3)
                      .map(([feature, importance]) => (
                        <div key={feature} className="flex justify-between text-xs">
                          <span className="text-gray-300">{feature}:</span>
                          <span className="text-blue-400">{(importance * 100).toFixed(1)}%</span>
                        </div>
                      )) : <div className="text-gray-400 text-xs">Feature data loading...</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Real-Time Alerts */}
        <div className="xl:col-span-1">
          <div className="bg-black/40 rounded-xl p-6 border border-gray-700">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              Live Alerts ({alerts.length})
            </h3>
            
            <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
              {alerts.map(alert => (
                <div key={alert.id} className={`p-3 rounded-lg border ${getSeverityColor(alert.severity)}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {alert.type === 'prediction' && <Brain className="w-4 h-4" />}
                      {alert.type === 'risk' && <Shield className="w-4 h-4" />}
                      {alert.type === 'opportunity' && <Zap className="w-4 h-4" />}
                      {alert.type === 'technical' && <BarChart3 className="w-4 h-4" />}
                      <span className="font-semibold text-sm">
                        {alert.type.toUpperCase()}
                      </span>
                    </div>
                    <span className="text-xs opacity-70">
                      {formatTimeAgo(alert.timestamp)}
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
                  <div>No alerts yet</div>
                  <div className="text-sm">Monitoring for opportunities...</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
