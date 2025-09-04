import { useState, useEffect } from 'react';

export interface AgentSignal {
  id: string;
  agentName: string;
  signal: 'BUY' | 'SELL' | 'HOLD' | 'WARNING';
  confidence: number;
  description: string;
  timestamp: string;
  symbol?: string;
  metadata?: Record<string, any>;
}

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

export const useAgents = () => {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [signals, setSignals] = useState<AgentSignal[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAgentsStatus = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/agents/status');
      const data = await response.json();
      
      if (data.success && data.data?.agents) {
        const agentList: AgentStatus[] = Object.entries(data.data.agents).map(([id, agent]: [string, any]) => ({
          id: agent.id,
          name: agent.name,
          status: agent.status,
          lastUpdate: agent.last_update,
          signalsCount: agent.signals_generated || 0,
          description: agent.current_task,
          messageCount: agent.tasks_completed || 0,
          uptime: parseFloat(agent.uptime?.replace('%', '') || '99.8'),
          successRate: agent.performance || 90
        }));
        
        setAgents(agentList);
      } else {
        console.warn('Failed to fetch agent status, keeping current state');
      }
    } catch (error) {
      console.error('Error fetching agents status:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSignals = async () => {
    try {
      // Fetch signals from signal generator
      const response = await fetch('http://localhost:8001/api/agents/signal-generator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'] })
      });
      
      const data = await response.json();
      if (data.success && data.data?.signals) {
        const newSignals: AgentSignal[] = data.data.signals.map((signal: any, index: number) => ({
          id: `signal-${Date.now()}-${index}`,
          agentName: 'Signal Generator',
          signal: signal.action as 'BUY' | 'SELL' | 'HOLD',
          confidence: signal.confidence,
          description: signal.reasoning,
          timestamp: new Date().toISOString(),
          symbol: signal.symbol,
          metadata: { agent_status: signal.agent_status }
        }));
        
        setSignals(prev => [...newSignals, ...prev.slice(0, 20)]); // Keep last 20 signals
      }
    } catch (error) {
      console.error('Error fetching signals:', error);
    }
  };

  useEffect(() => {
    fetchAgentsStatus();
    fetchSignals();
    
    // Update every 30 seconds
    const interval = setInterval(() => {
      fetchAgentsStatus();
      fetchSignals();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const refreshAgents = () => {
    setLoading(true);
    fetchAgentsStatus();
    fetchSignals();
  };

  return {
    agents,
    signals,
    loading,
    refreshAgents
  };
};
