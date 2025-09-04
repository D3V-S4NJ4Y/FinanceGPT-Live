import { useState, useEffect } from 'react';
import { useWebSocket } from './useWebSocket';

interface RealTimeData {
  marketData: any[];
  newsData: any[];
  signals: any[];
  isConnected: boolean;
  dataSource: 'real' | 'fallback';
  lastUpdated: string;
}

export const useRealTimeData = () => {
  const [data, setData] = useState<RealTimeData>({
    marketData: [],
    newsData: [],
    signals: [],
    isConnected: false,
    dataSource: 'fallback', // Start with fallback until real data confirmed
    lastUpdated: new Date().toISOString(),
  });

  const { messageHistory, sendMessage, isConnected } = useWebSocket('ws://127.0.0.1:8001/ws');

  useEffect(() => {
    if (messageHistory.length > 0) {
      const latestMessage = messageHistory[messageHistory.length - 1];
      
      // Update connection status
      setData(prev => ({
        ...prev,
        isConnected
      }));
      
      // Process different message types
      switch (latestMessage.type) {
        case 'market_data':
          setData(prev => ({
            ...prev,
            marketData: [...latestMessage.data, ...prev.marketData.slice(0, 99)],
            lastUpdated: new Date().toISOString(),
            // Check if we're getting real data or fallback data
            dataSource: latestMessage.data.some((item: any) => item.agent_status === 'real') ? 'real' : 'fallback'
          }));
          break;
        case 'news_update':
          setData(prev => ({
            ...prev,
            newsData: [...latestMessage.data, ...prev.newsData.slice(0, 49)],
            lastUpdated: new Date().toISOString()
          }));
          break;
        case 'ai_signal':
          setData(prev => ({
            ...prev,
            signals: [...latestMessage.data, ...prev.signals.slice(0, 19)],
            lastUpdated: new Date().toISOString(),
            // Check signal source
            dataSource: latestMessage.data.some((item: any) => item.agent_status === 'real') ? 'real' : 'fallback'
          }));
          break;
        case 'status':
          // Status updates
          setData(prev => ({
            ...prev,
            dataSource: latestMessage.data.source === 'real' ? 'real' : 'fallback',
            lastUpdated: new Date().toISOString()
          }));
          break;
      }
    }
  }, [messageHistory, isConnected]);
  
  // Request data refresh on initial connection
  useEffect(() => {
    if (isConnected) {
      sendMessage({ type: 'subscribe', channels: ['market_data', 'news', 'signals'] });
      sendMessage({ type: 'request_data', timestamp: Date.now() });
    }
  }, [isConnected, sendMessage]);

  return data;
};
