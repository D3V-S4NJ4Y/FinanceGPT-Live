// Market Data Types
export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

// Agent Types
export interface AgentStatus {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error';
  lastUpdate: string;
  signalsCount: number;
  description?: string;
  messageCount?: number;
  uptime?: number;
  successRate?: number;
}

// Alert Types
export interface Alert {
  id: string;
  type: 'warning' | 'error' | 'info' | 'success';
  message: string;
  title?: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

// Signal Types
export interface Signal {
  id: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  symbol: string;
  confidence: number;
  price: number;
  timestamp: string;
  agent: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

// Risk Types
export interface RiskMetric {
  name: string;
  value: number;
  level: 'low' | 'medium' | 'high';
}

// Sentiment Types
export interface SentimentData {
  score: number;
  label: 'bearish' | 'neutral' | 'bullish';
  sources: string[];
  confidence: number;
}

// News Types
export interface NewsItem {
  headline: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  impact: 'low' | 'medium' | 'high';
  time: string;
}

// Component Props Types
export interface MarketMapProps {
  data: any;
  className?: string;
}

export interface SentimentPulseProps {
  className?: string;
}

// Real-time Data Types
export interface RealTimeData {
  marketData: any[];
  newsData: any[];
  agentSignals: any[];
  isConnected: boolean;
  lastUpdate: string;
}

// Portfolio Types
export interface PortfolioPosition {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  marketValue: number;
  costBasis: number;
}

// Timeframe Types
export type Timeframe = '1m' | '5m' | '1h' | '1d';
