/**
 * Global State Management for Data Persistence
 * Prevents data clearing on tab switches
 */

interface GlobalState {
  marketData: any[];
  portfolioData: any[];
  aiMessages: any[];
  agentStatus: any;
  lastUpdate: string;
  isLoading: boolean;
}

class GlobalStore {
  private state: GlobalState = {
    marketData: [],
    portfolioData: [],
    aiMessages: [],
    agentStatus: null,
    lastUpdate: '',
    isLoading: false
  };

  private listeners: Set<() => void> = new Set();

  getState(): GlobalState {
    return { ...this.state };
  }

  setState(updates: Partial<GlobalState>) {
    this.state = { ...this.state, ...updates };
    this.notifyListeners();
    this.saveToStorage();
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener());
  }

  private saveToStorage() {
    try {
      localStorage.setItem('financeGPT_globalState', JSON.stringify(this.state));
    } catch (error) {
      console.warn('Failed to save state:', error);
    }
  }

  loadFromStorage() {
    try {
      const saved = localStorage.getItem('financeGPT_globalState');
      if (saved) {
        this.state = { ...this.state, ...JSON.parse(saved) };
        this.notifyListeners();
      }
    } catch (error) {
      console.warn('Failed to load state:', error);
    }
  }

  isDataFresh(): boolean {
    if (!this.state.lastUpdate) return false;
    const lastUpdate = new Date(this.state.lastUpdate);
    const now = new Date();
    return (now.getTime() - lastUpdate.getTime()) < 5 * 60 * 1000;
  }
}

export const globalStore = new GlobalStore();
globalStore.loadFromStorage();