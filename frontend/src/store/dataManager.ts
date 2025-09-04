// Global Data Manager - Real-time data sharing between all components

import React from 'react';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
  high?: number;
  low?: number;
  open?: number;
  marketCap?: number;
}

interface AgentData {
  id: string;
  name: string;
  status: string;
  performance: number;
  signals: number;
}

interface GlobalDataState {
  marketData: MarketData[];
  agents: AgentData[];
  portfolio: any[];
  alerts: any[];
  news: any[];
  lastUpdate: Date;
  isLoading: boolean;
}

class DataManager {
  private data: GlobalDataState = {
    marketData: [],
    agents: [],
    portfolio: [],
    alerts: [],
    news: [],
    lastUpdate: new Date(),
    isLoading: false
  };

  private subscribers: Map<string, (data: any) => void> = new Map();
  private apiCallInProgress: Map<string, Promise<any>> = new Map();

  // Subscribe to data updates
  subscribe(componentId: string, callback: (data: GlobalDataState) => void) {
    this.subscribers.set(componentId, callback);
    // Immediately send current data
    callback(this.data);
  }

  // Unsubscribe from updates
  unsubscribe(componentId: string) {
    this.subscribers.delete(componentId);
  }

  // Notify all subscribers
  private notify() {
    this.subscribers.forEach(callback => callback(this.data));
  }

  // Get market data with smart caching
  async getMarketData(forceRefresh = false): Promise<MarketData[]> {
    const cacheKey = 'marketData';
    
    // Return cached data if fresh (less than 30 seconds old)
    if (!forceRefresh && this.data.marketData.length > 0) {
      const age = Date.now() - this.data.lastUpdate.getTime();
      if (age < 30000) {
        console.log('⚡ Using cached market data');
        return this.data.marketData;
      }
    }

    // Prevent duplicate API calls
    if (this.apiCallInProgress.has(cacheKey)) {
      console.log('⏳ Market data API call in progress, waiting...');
      return await this.apiCallInProgress.get(cacheKey)!;
    }

    // Make API call
    const apiCall = this.fetchMarketDataFromAPI();
    this.apiCallInProgress.set(cacheKey, apiCall);

    try {
      const data = await apiCall;
      this.data.marketData = data;
      this.data.lastUpdate = new Date();
      this.notify();
      return data;
    } finally {
      this.apiCallInProgress.delete(cacheKey);
    }
  }

  // Get agents data with smart caching
  async getAgents(forceRefresh = false): Promise<AgentData[]> {
    const cacheKey = 'agents';
    
    if (!forceRefresh && this.data.agents.length > 0) {
      const age = Date.now() - this.data.lastUpdate.getTime();
      if (age < 15000) {
        console.log('⚡ Using cached agents data');
        return this.data.agents;
      }
    }

    if (this.apiCallInProgress.has(cacheKey)) {
      return await this.apiCallInProgress.get(cacheKey)!;
    }

    const apiCall = this.fetchAgentsFromAPI();
    this.apiCallInProgress.set(cacheKey, apiCall);

    try {
      const data = await apiCall;
      this.data.agents = data;
      this.notify();
      return data;
    } finally {
      this.apiCallInProgress.delete(cacheKey);
    }
  }

  // Private API methods
  private async fetchMarketDataFromAPI(): Promise<MarketData[]> {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/market/latest');
      if (response.ok) {
        const data = await response.json();
        console.log('🔄 Fresh market data fetched from API');
        return Array.isArray(data) ? data : [];
      }
      throw new Error(`API error: ${response.status}`);
    } catch (error) {
      console.error('Market data API failed:', error);
      return this.data.marketData; // Return cached data on error
    }
  }

  private async fetchAgentsFromAPI(): Promise<AgentData[]> {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/agents/status');
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data?.agents) {
          const agents = Object.entries(data.data.agents).map(([id, agent]: [string, any]) => ({
            id,
            name: agent.name || id.replace('_', ' '),
            status: agent.status || 'active',
            performance: parseFloat(agent.performance) || 85,
            signals: parseInt(agent.signals_generated) || 0
          }));
          console.log('🔄 Fresh agents data fetched from API');
          return agents;
        }
      }
      throw new Error(`Agents API error: ${response.status}`);
    } catch (error) {
      console.error('Agents API failed:', error);
      return this.data.agents; // Return cached data on error
    }
  }

  // Update portfolio data
  updatePortfolio(portfolio: any[]) {
    this.data.portfolio = portfolio;
    this.notify();
  }

  // Get current data state
  getCurrentData(): GlobalDataState {
    return { ...this.data };
  }

  // Force refresh all data
  async refreshAll(): Promise<void> {
    this.data.isLoading = true;
    this.notify();

    try {
      await Promise.all([
        this.getMarketData(true),
        this.getAgents(true)
      ]);
    } finally {
      this.data.isLoading = false;
      this.notify();
    }
  }
}

// Global singleton instance
export const dataManager = new DataManager();

// React hook for easy component integration
export const useGlobalData = (componentId: string) => {
  const [data, setData] = React.useState<GlobalDataState>(dataManager.getCurrentData());

  React.useEffect(() => {
    dataManager.subscribe(componentId, setData);
    return () => dataManager.unsubscribe(componentId);
  }, [componentId]);

  return {
    ...data,
    getMarketData: (force?: boolean) => dataManager.getMarketData(force),
    getAgents: (force?: boolean) => dataManager.getAgents(force),
    updatePortfolio: (portfolio: any[]) => dataManager.updatePortfolio(portfolio),
    refreshAll: () => dataManager.refreshAll()
  };
};