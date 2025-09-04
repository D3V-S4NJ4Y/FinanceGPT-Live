// API Configuration for FinanceGPT
export const API_CONFIG = {
  BASE_URL: "http://127.0.0.1:8001",
  WEBSOCKET_URL: "ws://127.0.0.1:8001",
  ENDPOINTS: {
    ML: {
      PREDICT: "/api/ml/predict",
      MARKET_REGIME: "/api/ml/market-regime"
    },
    MARKET_DATA: "/api/v1/market-data",
    AGENTS: "/api/v1/agents",
    ANALYTICS: "/api/v1/analytics",
    PORTFOLIO: "/api/v1/portfolio"
  }
};

export default API_CONFIG;
