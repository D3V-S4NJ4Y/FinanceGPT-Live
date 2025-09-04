/**
 * Real Portfolio Data Service
 * Fetches actual market data and calculates real portfolio metrics
 */

const API_BASE_URL = 'http://127.0.0.1:8001';

export interface PortfolioHolding {
  symbol: string;
  shares: number;
  avgCost: number;
}

export interface RealPortfolioData {
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
  holdings: Array<{
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
  }>;
  riskMetrics: {
    portfolioRisk: number;
    diversificationScore: number;
    concentrationRisk: number;
    sectorExposure: Record<string, number>;
    correlationRisk: number;
  };
}

class PortfolioService {
  async fetchRealPortfolioData(holdings: PortfolioHolding[], timeframe: string = '1D'): Promise<RealPortfolioData> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/portfolio/analytics`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          holdings: holdings.map(h => ({
            symbol: h.symbol,
            shares: h.shares,
            avg_cost: h.avgCost
          })),
          timeframe: timeframe
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success && result.data) {
        return {
          totalValue: result.data.portfolio_metrics.total_value,
          totalCost: result.data.portfolio_metrics.total_cost,
          totalReturn: result.data.portfolio_metrics.total_return,
          totalReturnPercent: result.data.portfolio_metrics.total_return_percent,
          dayChange: result.data.portfolio_metrics.day_change,
          dayChangePercent: result.data.portfolio_metrics.day_change_percent,
          sharpeRatio: result.data.portfolio_metrics.sharpe_ratio,
          beta: result.data.portfolio_metrics.beta,
          volatility: result.data.portfolio_metrics.volatility,
          maxDrawdown: result.data.portfolio_metrics.max_drawdown,
          holdings: result.data.holdings.map((h: any) => ({
            symbol: h.symbol,
            shares: h.shares,
            avgCost: h.avg_cost,
            currentPrice: h.current_price,
            marketValue: h.market_value,
            totalReturn: h.total_return,
            totalReturnPercent: h.total_return_percent,
            dayChange: h.day_change,
            dayChangePercent: h.day_change_percent,
            weight: h.weight
          })),
          riskMetrics: {
            portfolioRisk: result.data.risk_metrics.portfolio_risk,
            diversificationScore: result.data.risk_metrics.diversification_score,
            concentrationRisk: result.data.risk_metrics.concentration_risk,
            sectorExposure: result.data.risk_metrics.sector_exposure,
            correlationRisk: result.data.risk_metrics.correlation_risk
          }
        };
      }

      throw new Error('Invalid response format');
    } catch (error) {
      console.error('Portfolio service error:', error);
      throw error;
    }
  }

  getDefaultPortfolio(): PortfolioHolding[] {
    // Get real portfolio from localStorage or return empty for real data only
    const savedPortfolio = localStorage.getItem('userPortfolio');
    if (savedPortfolio) {
      try {
        return JSON.parse(savedPortfolio);
      } catch (e) {
        console.warn('Invalid saved portfolio');
      }
    }
    
    // Return empty array - no sample data
    return [];
  }

  savePortfolio(holdings: PortfolioHolding[]): void {
    localStorage.setItem('userPortfolio', JSON.stringify(holdings));
  }

  clearPortfolio(): void {
    localStorage.removeItem('userPortfolio');
  }
}

export const portfolioService = new PortfolioService();

// Helper function to check if portfolio exists
export const hasPortfolioData = (): boolean => {
  const saved = localStorage.getItem('userPortfolio');
  return saved ? JSON.parse(saved).length > 0 : false;
};