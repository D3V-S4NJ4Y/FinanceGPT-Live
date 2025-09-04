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

export const useWebSocket = (url: string = 'ws://127.0.0.1:8001/ws') => {
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
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

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
      // Simple WebSocket connection without complex client ID
      console.log('🔌 Connecting to WebSocket:', url);
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        if (!isMounted.current) {
          ws.close();
          return;
        }
        
        console.log('✅ WebSocket connected successfully');
        reconnectAttempts.current = 0; // Reset reconnection attempts on successful connection
        
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionState: 'connected',
          error: null
        }));
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
        
        console.log('🔌 WebSocket disconnected, code:', event.code, 'reason:', event.reason);
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionState: 'disconnected'
        }));
        
        // Only reconnect if it wasn't an intentional closure and we haven't exceeded max attempts
        if (event.code !== 1000 && isMounted.current && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000); // Exponential backoff, max 30s
          console.log(`🔄 Attempting to reconnect in ${delay/1000}s... (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          setState(prev => ({ ...prev, connectionState: 'connecting' }));
          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (isMounted.current) {
              connect();
            }
          }, delay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          console.error('❌ Max reconnection attempts reached. WebSocket connection failed permanently.');
          setState(prev => ({
            ...prev,
            connectionState: 'error',
            error: 'Connection failed after multiple attempts'
          }));
        }
      };

      ws.onerror = (error) => {
        if (!isMounted.current) return;
        
        console.error('❌ WebSocket error:', error);
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionState: 'error',
          error: 'WebSocket connection error'
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
    
    // Delay initial connection to ensure backend is ready
    const initialTimeout = setTimeout(() => {
      if (isMounted.current) {
        connect();
      }
    }, 1500);
    
    return () => {
      console.log('🧹 useWebSocket cleanup');
      isMounted.current = false;
      
      clearTimeout(initialTimeout);
      
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
