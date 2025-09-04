// Market data types
export interface MarketTick {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
}

export interface MarketData {
  stocks?: MarketTick[];
  indices?: MarketTick[];
  crypto?: MarketTick[];
  sectors?: {
    name: string;
    performance: number;
    tickers: string[];
  }[];
  lastUpdated?: string;
}

// Agent types
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

export interface Signal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  timestamp: string;
  agentName: string;
}

// Alert types
export interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  symbols?: string[];
}

// Component prop types
export interface MarketMapProps {
  data: MarketData;
  className?: string;
}

export interface SentimentPulseProps {
  timeframe: Timeframe;
  className?: string;
}

export interface RiskRadarProps {
  className?: string;
}

export interface SignalStreamProps {
  className?: string;
}

export interface NewsFlashProps {
  className?: string;
}

// Utility types
export type Timeframe = '1m' | '5m' | '1h' | '1d';
