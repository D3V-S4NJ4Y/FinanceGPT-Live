// Simple data store to prevent re-loading on tab switches
interface AIDataStore {
  analyses: any;
  marketIntel: any;
  alerts: any[];
  lastUpdate: Date;
  isInitialized: boolean;
}

let aiDataStore: AIDataStore = {
  analyses: {},
  marketIntel: null,
  alerts: [],
  lastUpdate: new Date(),
  isInitialized: false
};

export const getAIData = () => aiDataStore;

export const setAIData = (data: Partial<AIDataStore>) => {
  aiDataStore = { ...aiDataStore, ...data };
};

export const isAIDataFresh = () => {
  const now = new Date();
  const timeDiff = now.getTime() - aiDataStore.lastUpdate.getTime();
  return timeDiff < 60000; // 1 minute
};

export const clearAIData = () => {
  aiDataStore = {
    analyses: {},
    marketIntel: null,
    alerts: [],
    lastUpdate: new Date(),
    isInitialized: false
  };
};