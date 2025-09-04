// AI Analytics Utilities for Real-Time Financial Intelligence

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high24h: number;
  low24h: number;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: number;
  bollingerUpper: number;
  bollingerLower: number;
  sma20: number;
  sma50: number;
  volumeProfile: 'high' | 'normal' | 'low';
}

export interface SentimentData {
  newsScore: number;
  socialScore: number;
  institutionalFlow: number;
  overallSentiment: number;
}

// Calculate RSI (Relative Strength Index)
export const calculateRSI = (prices: number[], period: number = 14): number => {
  if (prices.length < period + 1) return 50;
  
  let gains = 0;
  let losses = 0;
  
  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
};

// Calculate MACD (Moving Average Convergence Divergence)
export const calculateMACD = (prices: number[], fastPeriod: number = 12, slowPeriod: number = 26): number => {
  if (prices.length < slowPeriod) return 0;
  
  const fastEMA = calculateEMA(prices, fastPeriod);
  const slowEMA = calculateEMA(prices, slowPeriod);
  
  return fastEMA - slowEMA;
};

// Calculate Exponential Moving Average
export const calculateEMA = (prices: number[], period: number): number => {
  if (prices.length === 0) return 0;
  
  const multiplier = 2 / (period + 1);
  let ema = prices[0];
  
  for (let i = 1; i < Math.min(prices.length, period); i++) {
    ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
  }
  
  return ema;
};

// Calculate Bollinger Bands
export const calculateBollingerBands = (prices: number[], period: number = 20, stdDev: number = 2) => {
  if (prices.length < period) return { upper: 0, middle: 0, lower: 0 };
  
  const sma = prices.slice(-period).reduce((sum, price) => sum + price, 0) / period;
  const variance = prices.slice(-period).reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
  const standardDeviation = Math.sqrt(variance);
  
  return {
    upper: sma + (standardDeviation * stdDev),
    middle: sma,
    lower: sma - (standardDeviation * stdDev)
  };
};

// Calculate volatility
export const calculateVolatility = (prices: number[]): number => {
  if (prices.length < 2) return 0;
  
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  
  const meanReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) / returns.length;
  
  return Math.sqrt(variance) * Math.sqrt(252) * 100; // Annualized volatility
};

// Generate AI prediction based on technical analysis
export const generateAIPrediction = (marketData: MarketData, technicals: TechnicalIndicators, sentiment: SentimentData) => {
  const { price, changePercent, volume } = marketData;
  const { rsi, macd } = technicals;
  const { overallSentiment } = sentiment;
  
  // Technical score (0-100)
  let technicalScore = 50;
  
  // RSI analysis
  if (rsi > 70) technicalScore -= 20; // Overbought
  else if (rsi < 30) technicalScore += 20; // Oversold
  else if (rsi > 50) technicalScore += 10; // Bullish momentum
  else technicalScore -= 10; // Bearish momentum
  
  // MACD analysis
  if (macd > 0) technicalScore += 15; // Bullish crossover
  else technicalScore -= 15; // Bearish crossover
  
  // Volume analysis
  if (volume > 1000000) {
    if (changePercent > 0) technicalScore += 10; // High volume + positive change
    else technicalScore -= 10; // High volume + negative change
  }
  
  // Sentiment integration
  const sentimentWeight = 0.3;
  const finalScore = (technicalScore * 0.7) + (overallSentiment * sentimentWeight);
  
  // Determine direction and confidence
  let direction: 'bullish' | 'bearish' | 'neutral';
  let confidence: number;
  
  if (finalScore > 65) {
    direction = 'bullish';
    confidence = Math.min(95, finalScore + 10);
  } else if (finalScore < 35) {
    direction = 'bearish';
    confidence = Math.min(95, (100 - finalScore) + 10);
  } else {
    direction = 'neutral';
    confidence = 50 + Math.abs(finalScore - 50);
  }
  
  // Calculate price target
  const volatilityFactor = calculateVolatility([price]) / 100;
  const targetMultiplier = direction === 'bullish' ? 1 + volatilityFactor : 1 - volatilityFactor;
  const priceTarget = price * targetMultiplier;
  
  return {
    direction,
    confidence,
    priceTarget,
    technicalScore: finalScore,
    probability: confidence / 100,
    timeframe: '1d',
    riskLevel: volatilityFactor > 0.3 ? 'high' : volatilityFactor > 0.15 ? 'medium' : 'low'
  };
};

// Market regime detection
export const detectMarketRegime = (marketDataArray: MarketData[]) => {
  if (marketDataArray.length === 0) return null;
  
  const avgChange = marketDataArray.reduce((sum, data) => sum + data.changePercent, 0) / marketDataArray.length;
  const volatility = calculateVolatility(marketDataArray.map(d => d.price));
  
  let regime: string;
  let confidence: number;
  
  if (avgChange > 2 && volatility < 20) {
    regime = 'Bull Market - Low Volatility';
    confidence = 0.85;
  } else if (avgChange > 1 && volatility > 25) {
    regime = 'Bull Market - High Volatility';
    confidence = 0.75;
  } else if (avgChange < -2 && volatility < 20) {
    regime = 'Bear Market - Low Volatility';
    confidence = 0.85;
  } else if (avgChange < -1 && volatility > 25) {
    regime = 'Bear Market - High Volatility';
    confidence = 0.75;
  } else if (Math.abs(avgChange) < 0.5 && volatility < 15) {
    regime = 'Sideways Market - Low Volatility';
    confidence = 0.80;
  } else {
    regime = 'Transitional Market';
    confidence = 0.60;
  }
  
  return {
    regime,
    confidence,
    volatilityState: volatility > 25 ? 'High' : volatility > 15 ? 'Medium' : 'Low',
    trendStrength: Math.abs(avgChange) / 5,
    marketStress: volatility / 50
  };
};

// Risk assessment
export const calculateRiskMetrics = (marketData: MarketData, historicalPrices: number[]) => {
  const volatility = calculateVolatility(historicalPrices);
  const currentPrice = marketData.price;
  
  // Value at Risk (95% confidence)
  const var95 = currentPrice * 0.05 * (volatility / 100);
  
  // Maximum drawdown estimation
  const maxDrawdown = volatility * 0.5;
  
  // Beta calculation (simplified - assumes market beta of 1)
  const beta = 1 + (volatility - 20) / 100;
  
  return {
    volatility,
    var95,
    maxDrawdown,
    beta: Math.max(0.1, Math.min(2.0, beta)),
    riskLevel: volatility > 30 ? 'high' : volatility > 15 ? 'medium' : 'low'
  };
};

// Generate smart alerts based on analysis
export const generateSmartAlerts = (symbol: string, analysis: any) => {
  const alerts = [];
  const timestamp = new Date();
  
  // High confidence prediction alert
  if (analysis.prediction.confidence > 85) {
    alerts.push({
      id: `pred-${symbol}-${Date.now()}`,
      type: 'opportunity',
      severity: 'high',
      symbol,
      message: `High confidence ${analysis.prediction.direction} signal: ${analysis.prediction.confidence.toFixed(1)}%`,
      confidence: analysis.prediction.confidence / 100,
      timestamp,
      actionRequired: true
    });
  }
  
  // RSI extreme levels
  if (analysis.technical.rsi > 80) {
    alerts.push({
      id: `rsi-${symbol}-${Date.now()}`,
      type: 'technical',
      severity: 'medium',
      symbol,
      message: `Overbought condition detected - RSI: ${analysis.technical.rsi.toFixed(1)}`,
      confidence: 0.8,
      timestamp,
      actionRequired: true
    });
  } else if (analysis.technical.rsi < 20) {
    alerts.push({
      id: `rsi-${symbol}-${Date.now()}`,
      type: 'technical',
      severity: 'medium',
      symbol,
      message: `Oversold condition detected - RSI: ${analysis.technical.rsi.toFixed(1)}`,
      confidence: 0.8,
      timestamp,
      actionRequired: true
    });
  }
  
  // High volatility alert
  if (analysis.risk.volatility > 40) {
    alerts.push({
      id: `vol-${symbol}-${Date.now()}`,
      type: 'risk',
      severity: 'high',
      symbol,
      message: `Extreme volatility detected: ${analysis.risk.volatility.toFixed(1)}%`,
      confidence: 0.9,
      timestamp,
      actionRequired: false
    });
  }
  
  return alerts;
};

// Performance tracking for AI models
export const trackModelPerformance = (predictions: any[], actualResults: any[]) => {
  if (predictions.length === 0 || actualResults.length === 0) return null;
  
  let correctPredictions = 0;
  let totalPredictions = Math.min(predictions.length, actualResults.length);
  
  for (let i = 0; i < totalPredictions; i++) {
    const predicted = predictions[i].direction;
    const actual = actualResults[i].direction;
    
    if (predicted === actual) correctPredictions++;
  }
  
  const accuracy = (correctPredictions / totalPredictions) * 100;
  
  return {
    accuracy,
    totalPredictions,
    correctPredictions,
    performance: accuracy > 70 ? 'excellent' : accuracy > 60 ? 'good' : accuracy > 50 ? 'average' : 'poor'
  };
};