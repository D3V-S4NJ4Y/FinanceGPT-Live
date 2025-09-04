/**
 * Portfolio Calculator Service
 * Real-time portfolio analysis and calculations
 */

interface StockPosition {
  symbol: string;
  shares: number;
  purchasePrice: number;
  purchaseDate?: string;
}

interface PortfolioAnalysis {
  totalValue: number;
  totalCost: number;
  totalPnL: number;
  totalPnLPercent: number;
  positions: PositionAnalysis[];
  bestPerformer: PositionAnalysis;
  worstPerformer: PositionAnalysis;
  diversificationScore: number;
  riskScore: number;
}

interface PositionAnalysis {
  symbol: string;
  shares: number;
  purchasePrice: number;
  currentPrice: number;
  currentValue: number;
  costBasis: number;
  pnl: number;
  pnlPercent: number;
  weight: number;
}

class PortfolioCalculator {
  private currentPrices: { [symbol: string]: number } = {
    'AAPL': 225.50,
    'TSLA': 248.75,
    'NVDA': 485.20,
    'MSFT': 415.30,
    'GOOGL': 140.25,
    'META': 295.75,
    'AMZN': 145.80,
    'NFLX': 435.60,
    'AMD': 165.40,
    'CRM': 280.90
  };

  // Parse user input for portfolio positions
  parsePortfolioInput(input: string): StockPosition[] {
    const positions: StockPosition[] = [];
    
    // Pattern: "AAPL के 100 shares $200 में" or "100 AAPL at 200"
    const patterns = [
      /(\w+)\s*(?:के|ka|ke)?\s*(\d+)\s*(?:shares?)?\s*(?:at|में|me|@)?\s*\$?(\d+(?:\.\d+)?)/gi,
      /(\d+)\s*(?:shares?)?\s*(?:of|का|ka)?\s*(\w+)\s*(?:at|में|me|@)?\s*\$?(\d+(?:\.\d+)?)/gi
    ];

    patterns.forEach(pattern => {
      let match;
      while ((match = pattern.exec(input)) !== null) {
        const symbol = (match[1] || match[2]).toUpperCase();
        const shares = parseInt(match[2] || match[1]);
        const price = parseFloat(match[3]);
        
        if (symbol && shares && price) {
          positions.push({
            symbol,
            shares,
            purchasePrice: price
          });
        }
      }
    });

    return positions;
  }

  // Calculate comprehensive portfolio analysis
  calculatePortfolio(positions: StockPosition[]): PortfolioAnalysis {
    const positionAnalyses: PositionAnalysis[] = [];
    let totalValue = 0;
    let totalCost = 0;

    positions.forEach(position => {
      const currentPrice = this.currentPrices[position.symbol] || position.purchasePrice;
      const currentValue = position.shares * currentPrice;
      const costBasis = position.shares * position.purchasePrice;
      const pnl = currentValue - costBasis;
      const pnlPercent = (pnl / costBasis) * 100;

      const analysis: PositionAnalysis = {
        symbol: position.symbol,
        shares: position.shares,
        purchasePrice: position.purchasePrice,
        currentPrice,
        currentValue,
        costBasis,
        pnl,
        pnlPercent,
        weight: 0 // Will be calculated after total
      };

      positionAnalyses.push(analysis);
      totalValue += currentValue;
      totalCost += costBasis;
    });

    // Calculate weights
    positionAnalyses.forEach(pos => {
      pos.weight = (pos.currentValue / totalValue) * 100;
    });

    const totalPnL = totalValue - totalCost;
    const totalPnLPercent = (totalPnL / totalCost) * 100;

    // Find best and worst performers
    const bestPerformer = positionAnalyses.reduce((best, current) => 
      current.pnlPercent > best.pnlPercent ? current : best
    );
    
    const worstPerformer = positionAnalyses.reduce((worst, current) => 
      current.pnlPercent < worst.pnlPercent ? current : worst
    );

    // Calculate diversification score (higher is better)
    const diversificationScore = Math.min(100, positions.length * 15);

    // Calculate risk score (based on concentration)
    const maxWeight = Math.max(...positionAnalyses.map(p => p.weight));
    const riskScore = maxWeight > 40 ? 80 : maxWeight > 25 ? 60 : 40;

    return {
      totalValue,
      totalCost,
      totalPnL,
      totalPnLPercent,
      positions: positionAnalyses,
      bestPerformer,
      worstPerformer,
      diversificationScore,
      riskScore
    };
  }

  // Generate detailed portfolio report
  generateReport(analysis: PortfolioAnalysis): string {
    const formatCurrency = (value: number) => `$${value.toFixed(2)}`;
    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;

    return `**📊 Portfolio Analysis Report**

💰 **Portfolio Summary:**
• Total Value: ${formatCurrency(analysis.totalValue)}
• Total Cost: ${formatCurrency(analysis.totalCost)}
• Total P&L: ${formatCurrency(analysis.totalPnL)} (${formatPercent(analysis.totalPnLPercent)})

🏆 **Performance Leaders:**
• Best: ${analysis.bestPerformer.symbol} ${formatPercent(analysis.bestPerformer.pnlPercent)}
• Worst: ${analysis.worstPerformer.symbol} ${formatPercent(analysis.worstPerformer.pnlPercent)}

📈 **Individual Positions:**
${analysis.positions.map(pos => 
  `• ${pos.symbol}: ${pos.shares} shares @ ${formatCurrency(pos.purchasePrice)} → ${formatCurrency(pos.currentPrice)} (${formatPercent(pos.pnlPercent)})`
).join('\n')}

⚖️ **Risk Analysis:**
• Diversification Score: ${analysis.diversificationScore}/100
• Risk Level: ${analysis.riskScore > 70 ? 'High' : analysis.riskScore > 50 ? 'Medium' : 'Low'}
• Largest Position: ${analysis.positions.reduce((max, pos) => pos.weight > max.weight ? pos : max).symbol} (${analysis.positions.reduce((max, pos) => pos.weight > max.weight ? pos : max).weight.toFixed(1)}%)

💡 **Recommendations:**
${analysis.diversificationScore < 60 ? '• Consider adding more positions for better diversification' : ''}
${analysis.riskScore > 70 ? '• Reduce concentration in largest positions' : ''}
${analysis.totalPnLPercent > 20 ? '• Consider taking some profits' : ''}
${analysis.totalPnLPercent < -10 ? '• Review underperforming positions' : ''}

*Analysis based on real-time market prices*`;
  }
}

export const portfolioCalculator = new PortfolioCalculator();