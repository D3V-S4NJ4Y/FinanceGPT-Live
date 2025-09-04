/**
 * Market Data Utilities
 */

export interface MarketTick {
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

export interface MarketAnalysis {
  volatilityScore: number;
  trendStrength: number;
  volumeProfile: 'low' | 'normal' | 'high';
  marketSentiment: 'bearish' | 'neutral' | 'bullish';
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
}

/**
 * Calculate advanced market volatility score
 */
export const calculateVolatilityScore = (marketData: MarketTick[]): number => {
  if (marketData.length === 0) return 0;
  
  const volatilities = marketData.map(stock => Math.abs(stock.changePercent));
  const avgVolatility = volatilities.reduce((sum, vol) => sum + vol, 0) / volatilities.length;
  
  // Weight by volume
  const volumeWeightedVolatility = marketData.reduce((score, stock) => {
    const volumeWeight = stock.volume / Math.max(...marketData.map(s => s.volume));
    return score + (Math.abs(stock.changePercent) * volumeWeight);
  }, 0) / marketData.length;
  
  return (avgVolatility + volumeWeightedVolatility) / 2;
};

/**
 * Calculate trend strength (0-1 scale)
 */
export const calculateTrendStrength = (marketData: MarketTick[]): number => {
  if (marketData.length === 0) return 0;
  
  const upTrending = marketData.filter(s => s.changePercent > 0);
  const downTrending = marketData.filter(s => s.changePercent < 0);
  
  // Calculate directional bias
  const directionalBias = Math.abs((upTrending.length / marketData.length) - 0.5) * 2;
  
  // Calculate magnitude of moves
  const avgMagnitude = marketData.reduce((sum, stock) => 
    sum + Math.abs(stock.changePercent), 0) / marketData.length;
  
  // Normalize magnitude (assume 5% is very strong)
  const normalizedMagnitude = Math.min(avgMagnitude / 5, 1);
  
  return (directionalBias + normalizedMagnitude) / 2;
};

/**
 * Analyze volume profile
 */
export const analyzeVolumeProfile = (marketData: MarketTick[]): 'low' | 'normal' | 'high' => {
  if (marketData.length === 0) return 'normal';
  
  const avgVolume = marketData.reduce((sum, stock) => sum + stock.volume, 0) / marketData.length;
  
  // These thresholds would typically be based on historical data
  if (avgVolume > 50000000) return 'high';
  if (avgVolume < 10000000) return 'low';
  return 'normal';
};

/**
 * Determine market sentiment
 */
export const calculateMarketSentiment = (marketData: MarketTick[]): 'bearish' | 'neutral' | 'bullish' => {
  if (marketData.length === 0) return 'neutral';
  
  const upTrending = marketData.filter(s => s.changePercent > 0);
  const upPercentage = upTrending.length / marketData.length;
  
  // Weight by magnitude
  const weightedSentiment = marketData.reduce((sum, stock) => {
    const volumeWeight = stock.volume / Math.max(...marketData.map(s => s.volume));
    return sum + (stock.changePercent * volumeWeight);
  }, 0) / marketData.length;
  
  if (upPercentage > 0.6 && weightedSentiment > 0.5) return 'bullish';
  if (upPercentage < 0.4 && weightedSentiment < -0.5) return 'bearish';
  return 'neutral';
};

/**
 * Calculate risk level based on multiple factors
 */
export const calculateRiskLevel = (marketData: MarketTick[]): 'LOW' | 'MEDIUM' | 'HIGH' => {
  const volatilityScore = calculateVolatilityScore(marketData);
  const trendStrength = calculateTrendStrength(marketData);
  
  // High volatility + strong trend = medium risk
  // High volatility + weak trend = high risk
  // Low volatility = low risk
  
  if (volatilityScore > 3) {
    return trendStrength > 0.7 ? 'MEDIUM' : 'HIGH';
  } else if (volatilityScore > 1.5) {
    return 'MEDIUM';
  } else {
    return 'LOW';
  }
};

/**
 * Comprehensive market analysis
 */
export const analyzeMarket = (marketData: MarketTick[]): MarketAnalysis => {
  return {
    volatilityScore: calculateVolatilityScore(marketData),
    trendStrength: calculateTrendStrength(marketData),
    volumeProfile: analyzeVolumeProfile(marketData),
    marketSentiment: calculateMarketSentiment(marketData),
    riskLevel: calculateRiskLevel(marketData)
  };
};

/**
 * Format large numbers for display
 */
export const formatLargeNumber = (num: number): string => {
  if (num >= 1e12) return `${(num / 1e12).toFixed(1)}T`;
  if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
  return num.toFixed(0);
};

/**
 * Format percentage with proper sign
 */
export const formatPercentage = (percent: number, decimals: number = 2): string => {
  const sign = percent >= 0 ? '+' : '';
  return `${sign}${percent.toFixed(decimals)}%`;
};

/**
 * Calculate portfolio metrics from market data
 */
export const calculatePortfolioMetrics = (
  marketData: MarketTick[], 
  holdings: Record<string, number> = {}
) => {
  let totalValue = 0;
  let totalChange = 0;
  let positions = 0;
  
  marketData.forEach(stock => {
    const shares = holdings[stock.symbol] || 100; // Default 100 shares
    const value = stock.price * shares;
    const change = stock.change * shares;
    
    totalValue += value;
    totalChange += change;
    positions++;
  });
  
  const changePercent = totalValue > 0 ? (totalChange / totalValue) * 100 : 0;
  
  return {
    totalValue,
    dailyChange: totalChange,
    dailyChangePercent: changePercent,
    positions,
    cash: 50000 // Mock cash value
  };
};

/**
 * Generate activity feed messages based on market data
 */
export const generateActivityMessage = (
  marketData: MarketTick[],
  previousData: MarketTick[] = []
): string => {
  if (marketData.length === 0) return 'Market data loading...';
  
  // Find significant changes
  const significantMovers = marketData.filter(stock => Math.abs(stock.changePercent) > 2);
  const highVolumeStocks = marketData.filter(stock => stock.volume > 20000000);
  
  if (significantMovers.length > 0) {
    const topMover = significantMovers.reduce((max, stock) => 
      Math.abs(stock.changePercent) > Math.abs(max.changePercent) ? stock : max);
    
    const direction = topMover.changePercent >= 0 ? '🚀' : '📉';
    return `${direction} ${topMover.symbol} ${formatPercentage(topMover.changePercent)} to $${topMover.price.toFixed(2)}`;
  }
  
  if (highVolumeStocks.length > 0) {
    const topVolume = highVolumeStocks.reduce((max, stock) => 
      stock.volume > max.volume ? stock : max);
    return `📊 HIGH VOLUME: ${topVolume.symbol} ${formatLargeNumber(topVolume.volume)} shares`;
  }
  
  // General market update
  const upCount = marketData.filter(s => s.changePercent > 0).length;
  const totalCount = marketData.length;
  const upPercentage = (upCount / totalCount * 100).toFixed(0);
  
  return `📈 MARKET: ${upCount}/${totalCount} stocks up (${upPercentage}%)`;
};

/**
 * Detect market anomalies
 */
export const detectAnomalies = (marketData: MarketTick[]): string[] => {
  const anomalies: string[] = [];
  
  // Check for unusual volatility
  const avgVolatility = calculateVolatilityScore(marketData);
  if (avgVolatility > 5) {
    anomalies.push('⚠️ Extremely high market volatility detected');
  }
  
  // Check for volume spikes
  const volumes = marketData.map(s => s.volume);
  const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
  const maxVolume = Math.max(...volumes);
  
  if (maxVolume > avgVolume * 5) {
    const highVolumeStock = marketData.find(s => s.volume === maxVolume);
    anomalies.push(`🔥 Volume spike: ${highVolumeStock?.symbol} (${formatLargeNumber(maxVolume)})`);
  }
  
  // Check for coordinated moves
  const strongMovers = marketData.filter(s => Math.abs(s.changePercent) > 3);
  if (strongMovers.length > marketData.length * 0.5) {
    anomalies.push('📊 Coordinated market movement detected');
  }
  
  return anomalies;
};

/**
 * Calculate correlation between stocks
 */
export const calculateCorrelation = (stock1: MarketTick, stock2: MarketTick): number => {
  // Simple correlation based on price movements
  // In a real implementation, this would use historical data
  const change1 = stock1.changePercent;
  const change2 = stock2.changePercent;
  
  // Simplified correlation: same direction = positive, opposite = negative
  if (Math.sign(change1) === Math.sign(change2)) {
    return Math.min(Math.abs(change1), Math.abs(change2)) / Math.max(Math.abs(change1), Math.abs(change2));
  } else {
    return -Math.min(Math.abs(change1), Math.abs(change2)) / Math.max(Math.abs(change1), Math.abs(change2));
  }
};

/**
 * Get sector performance summary
 */
export const getSectorPerformance = (marketData: MarketTick[]): Record<string, number> => {
  const sectorPerformance: Record<string, { total: number; count: number }> = {};
  
  marketData.forEach(stock => {
    const sector = stock.sector || 'Unknown';
    if (!sectorPerformance[sector]) {
      sectorPerformance[sector] = { total: 0, count: 0 };
    }
    sectorPerformance[sector].total += stock.changePercent;
    sectorPerformance[sector].count += 1;
  });
  
  const result: Record<string, number> = {};
  Object.entries(sectorPerformance).forEach(([sector, data]) => {
    result[sector] = data.total / data.count;
  });
  
  return result;
};

export default {
  calculateVolatilityScore,
  calculateTrendStrength,
  analyzeVolumeProfile,
  calculateMarketSentiment,
  calculateRiskLevel,
  analyzeMarket,
  formatLargeNumber,
  formatPercentage,
  calculatePortfolioMetrics,
  generateActivityMessage,
  detectAnomalies,
  calculateCorrelation,
  getSectorPerformance
};