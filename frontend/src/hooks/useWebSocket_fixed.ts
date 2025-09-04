import { useState, useEffect, useRef, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
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
  const isMounted = useRef(true);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const sendMessage = useCallback((message: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, [socket]);

  const disconnect = useCallback(() => {
    console.log('🔌 Disconnecting WebSocket');
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      socket.close();
    }
    
    setSocket(null);
    setState({
      isConnected: false,
      lastMessage: null,
      connectionState: 'disconnected',
      error: null
    });
  }, [socket]);

  const connect = useCallback(() => {
    if (!isMounted.current) return;
    if (socket?.readyState === WebSocket.OPEN) return;

    setState(prev => ({ ...prev, connectionState: 'connecting', error: null }));

    try {
      // Generate unique client ID for proper WebSocket routing
      const clientId = `command-center-${Date.now()}${Math.random().toString(36).substr(2, 9)}`;
      const wsUrl = `${url}/${clientId}`;
      
      console.log('🔌 Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        if (!isMounted.current) {
          ws.close();
          return;
        }
        
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
        if (!isMounted.current) return;
        
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

      ws.onclose = (event) => {
        if (!isMounted.current) return;
        
        console.log('WebSocket disconnected, code:', event.code);
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionState: 'disconnected'
        }));
        
        // Only attempt reconnect if it was not intentional (code 1000 is normal closure)
        if (event.code !== 1000 && isMounted.current) {
          console.log('Attempting to reconnect in 5 seconds...');
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMounted.current) {
              connect();
            }
          }, 5000);
        }
      };

      ws.onerror = (error) => {
        if (!isMounted.current) return;
        
        console.error('WebSocket error:', error);
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionState: 'error',
          error: 'WebSocket connection failed'
        }));
      };

      setSocket(ws);
      
    } catch (err) {
      if (!isMounted.current) return;
      
      console.error('WebSocket connection error:', err);
      setState(prev => ({
        ...prev,
        connectionState: 'error',
        error: err instanceof Error ? err.message : 'Connection failed'
      }));
    }
  }, [url, socket]);

  // Initial connection and cleanup
  useEffect(() => {
    isMounted.current = true;
    connect();
    
    return () => {
      console.log('🧹 useWebSocket cleanup');
      isMounted.current = false;
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close(1000, 'Component unmounting');
      }
    };
  }, []);

  return {
    isConnected: state.isConnected,
    lastMessage: state.lastMessage,
    connectionState: state.connectionState,
    error: state.error,
    messageHistory,
    sendMessage,
    connect,
    disconnect
  };
};
