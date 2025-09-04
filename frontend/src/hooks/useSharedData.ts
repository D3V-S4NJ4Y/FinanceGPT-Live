// Shared Data Hook - Superfast data sharing between all components
import { useState, useEffect } from 'react';

interface SharedDataState {
  marketData: any[];
  agents: any[];
  portfolio: any[];
  alerts: any[];
  lastUpdate: Date;
}

class SharedDataStore {
  private data: SharedDataState = {
    marketData: [],
    agents: [],
    portfolio: [],
    alerts: [],
    lastUpdate: new Date()
  };

  private listeners: Set<(data: SharedDataState) => void> = new Set();
  private apiCalls: Map<string, Promise<any>> = new Map();

  subscribe(callback: (data: SharedDataState) => void) {
    this.listeners.add(callback);
    callback(this.data); // Send current data immediately
    return () => {
      this.listeners.delete(callback);
    };
  }

  private notify() {
    this.listeners.forEach(callback => callback(this.data));
  }

  async getMarketData(): Promise<any[]> {
    // Return cached if fresh (< 30 seconds)
    const age = Date.now() - this.data.lastUpdate.getTime();
    if (this.data.marketData.length > 0 && age < 30000) {
      return this.data.marketData;
    }

    // Prevent duplicate API calls
    if (this.apiCalls.has('market')) {
      return await this.apiCalls.get('market')!;
    }

    const apiCall = this.fetchMarketData();
    this.apiCalls.set('market', apiCall);

    try {
      const data = await apiCall;
      this.data.marketData = data;
      this.data.lastUpdate = new Date();
      this.notify();
      return data;
    } finally {
      this.apiCalls.delete('market');
    }
  }

  private async fetchMarketDataBackground() {
    if (this.apiCalls.has('market')) return;
    
    const apiCall = this.fetchMarketData();
    this.apiCalls.set('market', apiCall);

    try {
      const data = await apiCall;
      if (data.length > 0) {
        this.data.marketData = data;
        this.data.lastUpdate = new Date();
        this.notify();
      }
    } finally {
      this.apiCalls.delete('market');
    }
  }

  async getAgents(): Promise<any[]> {
    const age = Date.now() - this.data.lastUpdate.getTime();
    if (this.data.agents.length > 0 && age < 15000) {
      return this.data.agents;
    }

    if (this.apiCalls.has('agents')) {
      return await this.apiCalls.get('agents')!;
    }

    const apiCall = this.fetchAgents();
    this.apiCalls.set('agents', apiCall);

    try {
      const data = await apiCall;
      this.data.agents = data;
      this.notify();
      return data;
    } finally {
      this.apiCalls.delete('agents');
    }
  }

  private async fetchAgentsBackground() {
    if (this.apiCalls.has('agents')) return;
    
    const apiCall = this.fetchAgents();
    this.apiCalls.set('agents', apiCall);

    try {
      const data = await apiCall;
      if (data.length > 0) {
        this.data.agents = data;
        this.notify();
      }
    } finally {
      this.apiCalls.delete('agents');
    }
  }

  private async fetchMarketData(): Promise<any[]> {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/market/latest');
      if (response.ok) {
        const data = await response.json();
        console.log('🔄 Fresh market data from API');
        return Array.isArray(data) ? data : [];
      }
    } catch (error) {
      console.error('Market API failed:', error);
    }
    return this.data.marketData;
  }

  private async fetchAgents(): Promise<any[]> {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/agents/status');
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data?.agents) {
          console.log('🔄 Fresh agents data from API');
          return Object.entries(data.data.agents).map(([id, agent]: [string, any]) => ({
            id,
            name: agent.name || id.replace('_', ' '),
            status: agent.status || 'active',
            performance: parseFloat(agent.performance) || 85,
            signals: parseInt(agent.signals_generated) || 0
          }));
        }
      }
    } catch (error) {
      console.error('Agents API failed:', error);
    }
    return this.data.agents;
  }

  updatePortfolio(portfolio: any[]) {
    this.data.portfolio = portfolio;
    this.notify();
  }
}

const sharedStore = new SharedDataStore();

export const useSharedData = () => {
  const [data, setData] = useState<SharedDataState>(sharedStore['data']);

  useEffect(() => {
    const unsubscribe = sharedStore.subscribe(setData);
    return unsubscribe;
  }, []);

  return {
    ...data,
    getMarketData: () => sharedStore.getMarketData(),
    getAgents: () => sharedStore.getAgents(),
    updatePortfolio: (portfolio: any[]) => sharedStore.updatePortfolio(portfolio)
  };
};