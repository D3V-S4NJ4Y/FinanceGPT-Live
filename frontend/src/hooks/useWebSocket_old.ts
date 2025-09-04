import { useState, useEffect, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface WebSocketState {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
  error: string | null;
}

export const useWebSocket = (url: string = 'ws://localhost:8001/ws') => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    lastMessage: null,
    connectionState: 'disconnected',
    error: null
  });
  
  const [messageHistory, setMessageHistory] = useState<WebSocketMessage[]>([]);

  const connect = useCallback(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      return;
    }

    setState(prev => ({ ...prev, connectionState: 'connecting', error: null }));

    try {
      // Generate unique client ID for proper WebSocket routing
      const clientId = `command-center-${Date.now()}${Math.random().toString(36).substr(2, 9)}`;
      const wsUrl = `${url}/${clientId}`;
      
      console.log('🔌 Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionState: 'connected',
          error: null
        }));
        
        // Send subscription for AI Intelligence channels immediately after connection
        const subscriptionMessage = {
          type: 'subscribe',
          channels: ['ai_intelligence', 'market_data', 'predictions', 'ai_agents'],
          symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
          timestamp: new Date().toISOString()
        };
        
        ws.send(JSON.stringify(subscriptionMessage));
        console.log('✅ WebSocket connected and subscribed to AI Intelligence channels');
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setState(prev => ({
            ...prev,
            lastMessage: message
          }));
          
          setMessageHistory(prev => [...prev.slice(-99), message]); // Keep last 100 messages
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionState: 'disconnected'
        }));
        console.log('WebSocket disconnected');
      };

      ws.onerror = (error) => {
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionState: 'error',
          error: 'WebSocket connection failed'
        }));
        console.error('WebSocket error:', error);
      };

      setSocket(ws);
    } catch (err) {
      setState(prev => ({
        ...prev,
        connectionState: 'error',
        error: err instanceof Error ? err.message : 'Connection failed'
      }));
    }
  }, [url, socket]);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close();
      setSocket(null);
    }
  }, [socket]);

  const sendMessage = useCallback((message: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, [socket]);

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [url]);

  // Auto-reconnect on disconnect
  useEffect(() => {
    if (state.connectionState === 'disconnected' && !state.error) {
      const timeout = setTimeout(connect, 3000);
      return () => clearTimeout(timeout);
    }
  }, [state.connectionState, state.error, connect]);

  return {
    ...state,
    messageHistory,
    connect,
    disconnect,
    sendMessage
  };
};
