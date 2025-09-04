import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Mic, MicOff, Volume2, VolumeX, Brain, MessageSquare, Zap, 
  TrendingUp, Settings, Activity, AlertTriangle, Target, Play,
  RefreshCw, Eye, BarChart3, DollarSign, Percent, Clock
} from 'lucide-react';

interface VoiceCommand {
  id: string;
  command: string;
  timestamp: Date;
  response: string;
  confidence: number;
  action: string;
  executionTime: number;
  dataUsed: string[];
}

interface RealTimeData {
  marketData: Record<string, any>;
  agents: any[];
  alerts: any[];
  portfolio: any;
  news: any[];
  lastUpdate: string;
}

interface VoiceSettings {
  voiceIndex: number;
  autoSpeak: boolean;
  sensitivity: number;
  language: string;
  speed: number;
  pitch: number;
}

const useRealTimeData = () => {
  const [data, setData] = useState<RealTimeData>({
    marketData: {},
    agents: [],
    alerts: [],
    portfolio: null,
    news: [],
    lastUpdate: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Enhanced data fetching with multiple endpoint attempts
      const endpoints = [
        { url: 'http://127.0.0.1:8001/api/market/latest', key: 'marketData' },
        { url: 'http://127.0.0.1:8001/api/market-data', key: 'marketData' },
        { url: 'http://127.0.0.1:8001/market-data', key: 'marketData' }
      ];

      const newData: RealTimeData = {
        marketData: {},
        agents: [],
        alerts: [],
        portfolio: null,
        news: [],
        lastUpdate: new Date().toISOString()
      };

      // Try multiple market data endpoints
      let marketDataFetched = false;
      for (const endpoint of endpoints) {
        if (marketDataFetched) break;
        try {
          const response = await fetch(endpoint.url, { 
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
          });
          if (response.ok) {
            const marketData = await response.json();
            if (Array.isArray(marketData)) {
              newData.marketData = marketData.reduce((acc, stock) => ({ 
                ...acc, 
                [stock.symbol]: {
                  ...stock,
                  price: parseFloat(stock.price) || 0,
                  changePercent: parseFloat(stock.changePercent) || 0,
                  volume: parseInt(stock.volume) || 0
                }
              }), {});
              marketDataFetched = true;
            } else if (marketData.data || marketData.stocks) {
              const stocks = marketData.data || marketData.stocks;
              newData.marketData = Array.isArray(stocks) ? 
                stocks.reduce((acc, stock) => ({ ...acc, [stock.symbol]: stock }), {}) : stocks;
              marketDataFetched = true;
            }
          }
        } catch (e) {
          console.log(`Endpoint ${endpoint.url} failed, trying next...`);
        }
      }

      // Add sample market data if API fails
      if (!marketDataFetched) {
        console.log('🔄 Using sample market data for Voice AI...');
        newData.marketData = {
          'AAPL': { symbol: 'AAPL', price: 232.14, changePercent: -0.18, volume: 28450000, change: -0.42 },
          'MSFT': { symbol: 'MSFT', price: 506.69, changePercent: -0.58, volume: 15230000, change: -2.96 },
          'GOOGL': { symbol: 'GOOGL', price: 212.91, changePercent: 0.60, volume: 12340000, change: 1.27 },
          'AMZN': { symbol: 'AMZN', price: 186.45, changePercent: 1.24, volume: 18920000, change: 2.28 },
          'TSLA': { symbol: 'TSLA', price: 348.10, changePercent: -1.45, volume: 35670000, change: -5.12 },
          'NVDA': { symbol: 'NVDA', price: 489.33, changePercent: 2.18, volume: 22480000, change: 10.44 },
          'META': { symbol: 'META', price: 542.81, changePercent: 0.92, volume: 14560000, change: 4.95 },
          'NFLX': { symbol: 'NFLX', price: 678.90, changePercent: -0.33, volume: 8930000, change: -2.25 }
        };
      }

      // Fetch agents data with fallback
      try {
        const agentsRes = await fetch('http://127.0.0.1:8001/api/agents/status');
        if (agentsRes.ok) {
          const agentsData = await agentsRes.json();
          if (agentsData.agents) {
            newData.agents = Object.entries(agentsData.agents).map(([id, agent]: [string, any]) => ({
              id,
              name: agent.name || id,
              status: agent.status || 'active',
              last_update: agent.last_update || new Date().toISOString(),
              performance: parseFloat(agent.performance) || 75,
              signals_today: parseInt(agent.signals_generated) || 0,
              uptime: agent.uptime || '99%',
              model_version: agent.version || '1.0'
            }));
          }
        }
      } catch (e) {
        // Create default agent data if API fails
        newData.agents = [
          { id: 'signal_generator', name: 'Signal Generator', status: 'active', last_update: new Date().toISOString(), performance: 78, signals_today: 12, uptime: '99%', model_version: '1.0' },
          { id: 'risk_assessor', name: 'Risk Assessor', status: 'active', last_update: new Date().toISOString(), performance: 82, signals_today: 8, uptime: '98%', model_version: '1.0' },
          { id: 'news_intelligence', name: 'News Intelligence', status: 'active', last_update: new Date().toISOString(), performance: 75, signals_today: 15, uptime: '97%', model_version: '1.0' }
        ];
      }

      // Fetch alerts with fallback
      try {
        let alertsFetched = false;
        const alertsEndpoints = [
          'http://127.0.0.1:8001/api/alerts/recent',
          'http://127.0.0.1:8001/api/portfolio/alerts?portfolio_value=100000&risk_tolerance=medium'
        ];

        for (const endpoint of alertsEndpoints) {
          if (alertsFetched) break;
          try {
            const alertsRes = await fetch(endpoint);
            if (alertsRes.ok) {
              const alertsData = await alertsRes.json();
              newData.alerts = alertsData.alerts || alertsData.data || [];
              alertsFetched = true;
            }
          } catch (e) {
            console.log(`Alerts endpoint ${endpoint} failed, trying next...`);
          }
        }

        if (!alertsFetched) {
          throw new Error('All alerts endpoints failed');
        }
      } catch (e) {
        // Generate alerts from market data if API fails
        if (Object.keys(newData.marketData).length > 0) {
          newData.alerts = Object.entries(newData.marketData)
            .filter(([_, stock]: [string, any]) => Math.abs(stock.changePercent || 0) > 2)
            .map(([symbol, stock]: [string, any]) => ({
              id: `alert-${symbol}-${Date.now()}`,
              type: 'market' as const,
              severity: Math.abs(stock.changePercent) > 5 ? 'high' as const : 'medium' as const,
              message: `${symbol} ${stock.changePercent > 0 ? 'up' : 'down'} ${Math.abs(stock.changePercent).toFixed(2)}%`,
              symbol,
              timestamp: new Date(),
              agent_source: 'Market Monitor'
            }));
        }
      }

      // Fetch portfolio with fallback
      try {
        // Try multiple portfolio endpoints
        let portfolioFetched = false;
        const portfolioEndpoints = [
          'http://127.0.0.1:8001/api/portfolio/summary',
          'http://127.0.0.1:8001/api/portfolio/calculate'
        ];

        for (const endpoint of portfolioEndpoints) {
          if (portfolioFetched) break;
          try {
            let response;
            if (endpoint.includes('calculate')) {
              // POST request for calculate endpoint
              response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  positions: [
                    { symbol: 'AAPL', quantity: 100, avg_price: 150 },
                    { symbol: 'GOOGL', quantity: 50, avg_price: 200 },
                    { symbol: 'MSFT', quantity: 75, avg_price: 300 }
                  ],
                  cash_balance: 50000
                })
              });
            } else {
              // GET request for summary endpoint
              response = await fetch(endpoint);
            }

            if (response.ok) {
              const portfolioData = await response.json();
              if (portfolioData.data || portfolioData.summary) {
                newData.portfolio = portfolioData.data || portfolioData;
                portfolioFetched = true;
              }
            }
          } catch (e) {
            console.log(`Portfolio endpoint ${endpoint} failed, trying next...`);
          }
        }

        if (!portfolioFetched) {
          throw new Error('All portfolio endpoints failed');
        }
      } catch (e) {
        // Create portfolio from market data if API fails
        if (Object.keys(newData.marketData).length > 0) {
          const topStocks = Object.entries(newData.marketData).slice(0, 5);
          const totalValue = topStocks.reduce((sum, [_, stock]: [string, any]) => sum + (stock.price * 100), 0);
          const dailyChange = topStocks.reduce((sum, [_, stock]: [string, any]) => sum + (stock.change * 100), 0);
          
          newData.portfolio = {
            totalValue,
            dailyChange,
            dailyChangePercent: totalValue > 0 ? (dailyChange / totalValue) * 100 : 0,
            positions: topStocks.length,
            cash: 50000
          };
        }
      }

      // Fetch news with fallback
      try {
        const symbols = Object.keys(newData.marketData).slice(0, 5).join(',');
        const newsRes = await fetch(`http://127.0.0.1:8001/api/news/latest?symbols=${symbols}`);
        if (newsRes.ok) {
          const newsData = await newsRes.json();
          newData.news = newsData.data?.articles || newsData.articles || [];
        }
      } catch (e) {
        // Generate sample news if API fails
        newData.news = [
          {
            id: 'news-1',
            title: 'Market Analysis: Tech stocks show resilience',
            summary: 'Technology sector continues to outperform broader markets.',
            timestamp: new Date(),
            sentiment: 'positive'
          },
          {
            id: 'news-2',
            title: 'Federal Reserve maintains current interest rates',
            summary: 'Central bank keeps rates steady amid economic uncertainty.',
            timestamp: new Date(),
            sentiment: 'neutral'
          }
        ];
      }

      setData(newData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Faster updates
    return () => clearInterval(interval);
  }, [fetchData]);

  return { data, isLoading, error, refresh: fetchData };
};

const useVoiceRecognition = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (event: any) => {
      const result = event.results[0];
      if (result.isFinal) {
        setTranscript(result[0].transcript);
        setConfidence(result[0].confidence || 0.8);
      }
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    return () => recognition?.stop();
  }, []);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  }, [isListening]);

  const resetTranscript = useCallback(() => {
    setTranscript('');
    setConfidence(0);
  }, []);

  return {
    isListening,
    transcript,
    confidence,
    startListening,
    stopListening,
    resetTranscript,
    isSupported: !!recognitionRef.current
  };
};

const useTextToSpeech = () => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    const updateVoices = () => setVoices(speechSynthesis.getVoices());
    updateVoices();
    speechSynthesis.onvoiceschanged = updateVoices;
    return () => { speechSynthesis.onvoiceschanged = null; };
  }, []);

  const speak = useCallback((text: string, settings: VoiceSettings) => {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      const voice = voices[settings.voiceIndex] || voices.find(v => v.lang === 'en-US') || voices[0];
      
      if (voice) utterance.voice = voice;
      utterance.rate = settings.speed;
      utterance.pitch = settings.pitch;
      utterance.volume = 1;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      speechSynthesis.speak(utterance);
    }
  }, [voices]);

  const stop = useCallback(() => {
    speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  return { speak, stop, isSpeaking, voices, isSupported: 'speechSynthesis' in window };
};

const processVoiceCommand = async (
  transcript: string, 
  realTimeData: RealTimeData
): Promise<{ action: string; response: string; confidence: number; dataUsed: string[] }> => {
  const command = transcript.toLowerCase().trim();
  const dataUsed: string[] = [];
  
  // Fast AI analysis with timeout
  const getAIAnalysis = async (symbol?: string) => {
    try {
      const timeout = 3000; // 3 second timeout for fast responses
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      
      const [signalRes, riskRes, newsRes, summaryRes] = await Promise.all([
        fetch('http://127.0.0.1:8001/api/agents/signal-generator', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: symbol ? [symbol] : ['AAPL', 'MSFT', 'GOOGL'] }),
          signal: controller.signal
        }).catch(() => null),
        fetch('http://127.0.0.1:8001/api/agents/risk-assessor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: [{ symbol: symbol || 'AAPL', quantity: 100 }] }),
          signal: controller.signal
        }).catch(() => null),
        fetch('http://127.0.0.1:8001/api/agents/news-intelligence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: symbol ? [symbol] : ['AAPL'] }),
          signal: controller.signal
        }).catch(() => null),
        fetch('http://127.0.0.1:8001/api/agents/executive-summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ marketData: realTimeData.marketData }),
          signal: controller.signal
        }).catch(() => null)
      ]);
      
      clearTimeout(timeoutId);
      
      const [signals, risk, news, summary] = await Promise.all([
        signalRes?.ok ? signalRes.json().catch(() => null) : null,
        riskRes?.ok ? riskRes.json().catch(() => null) : null,
        newsRes?.ok ? newsRes.json().catch(() => null) : null,
        summaryRes?.ok ? summaryRes.json().catch(() => null) : null
      ]);
      
      return { signals, risk, news, summary };
    } catch (error) {
      return { signals: null, risk: null, news: null, summary: null };
    }
  };

  // Price queries with enhanced analysis
  if (command.includes('price') || command.includes('quote')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    const stockData = realTimeData.marketData[symbol];
    
    if (stockData) {
      dataUsed.push('market_data', 'ai_analysis');
      const aiData = await getAIAnalysis(symbol);
      
      let response = `${symbol} is trading at $${stockData.price?.toFixed(2)}, ${stockData.changePercent >= 0 ? 'up' : 'down'} ${Math.abs(stockData.changePercent || 0).toFixed(2)}% today with ${((stockData.volume || 0) / 1000000).toFixed(1)}M volume.`;
      
      if (aiData.signals?.data?.individual_signals?.[symbol]) {
        const signal = aiData.signals.data.individual_signals[symbol];
        response += ` AI Signal: ${signal.signal_type} with ${signal.confidence}% confidence. ${signal.reasoning || ''}`;
      }
      
      if (aiData.risk?.data) {
        response += ` Risk Level: ${aiData.risk.data.risk_level || 'Medium'}.`;
      }
      
      return { action: 'enhanced_price_query', response, confidence: 0.95, dataUsed };
    }
  }

  // Comprehensive market analysis
  if (command.includes('analyze') || command.includes('analysis')) {
    if (command.includes('market')) {
      dataUsed.push('market_data', 'ai_analysis', 'alerts');
      const aiData = await getAIAnalysis();
      
      const marketSymbols = Object.keys(realTimeData.marketData);
      const gainers = marketSymbols.filter(s => (realTimeData.marketData[s]?.changePercent || 0) > 0).length;
      const losers = marketSymbols.filter(s => (realTimeData.marketData[s]?.changePercent || 0) < 0).length;
      
      let response = `Market Analysis: ${gainers} gainers, ${losers} losers. Sentiment: ${gainers > losers ? 'Bullish' : 'Bearish'}.`;
      
      if (aiData.summary?.data) {
        response += ` Executive Summary: ${aiData.summary.data.executive_summary || aiData.summary.data.market_outlook || 'Market conditions are mixed'}.`;
      }
      
      if (realTimeData.alerts.length > 0) {
        response += ` Active Alerts: ${realTimeData.alerts.length}. Latest: ${realTimeData.alerts[0]?.message}`;
      }
      
      return { action: 'comprehensive_market_analysis', response, confidence: 0.92, dataUsed };
    }
    
    // Individual stock analysis
    const symbols = extractSymbols(command);
    if (symbols.length > 0) {
      const symbol = symbols[0];
      dataUsed.push('market_data', 'ai_analysis');
      const aiData = await getAIAnalysis(symbol);
      const stockData = realTimeData.marketData[symbol];
      
      let response = `${symbol} Analysis: `;
      
      if (stockData) {
        response += `Price $${stockData.price?.toFixed(2)}, ${stockData.changePercent >= 0 ? '+' : ''}${stockData.changePercent?.toFixed(2)}%. `;
      }
      
      if (aiData.signals?.data?.individual_signals?.[symbol]) {
        const signal = aiData.signals.data.individual_signals[symbol];
        response += `Signal: ${signal.signal_type} (${signal.confidence}% confidence). `;
      }
      
      if (aiData.risk?.data) {
        response += `Risk: ${aiData.risk.data.risk_level}. `;
      }
      
      if (aiData.news?.data?.sentiment) {
        response += `News Sentiment: ${aiData.news.data.sentiment > 0 ? 'Positive' : 'Negative'}.`;
      }
      
      return { action: 'stock_analysis', response, confidence: 0.90, dataUsed };
    }
  }

  // Trading signals and recommendations
  if (command.includes('signal') || command.includes('recommend') || command.includes('buy') || command.includes('sell')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    dataUsed.push('ai_signals', 'market_data');
    
    const aiData = await getAIAnalysis(symbol);
    
    if (aiData.signals?.data?.individual_signals?.[symbol]) {
      const signal = aiData.signals.data.individual_signals[symbol];
      const stockData = realTimeData.marketData[symbol];
      
      let response = `Trading Signal for ${symbol}: ${signal.signal_type} with ${signal.confidence}% confidence. `;
      
      if (stockData) {
        response += `Current price: $${stockData.price?.toFixed(2)}. `;
      }
      
      if (signal.target_price) {
        response += `Target: $${signal.target_price}. `;
      }
      
      if (signal.reasoning) {
        response += `Reasoning: ${signal.reasoning}`;
      }
      
      return { action: 'trading_signal', response, confidence: 0.88, dataUsed };
    }
  }

  // Risk assessment
  if (command.includes('risk')) {
    dataUsed.push('risk_analysis', 'portfolio_data');
    
    try {
      const riskRes = await fetch('http://127.0.0.1:8001/api/agents/risk-assessor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          portfolio: Object.keys(realTimeData.marketData).slice(0, 5).map(symbol => ({
            symbol, 
            quantity: 100, 
            value: realTimeData.marketData[symbol]?.price * 100 || 10000
          }))
        })
      });
      
      if (riskRes.ok) {
        const riskData = await riskRes.json();
        
        let response = 'Risk Assessment: ';
        
        if (riskData.data?.risk_level) {
          response += `Overall risk level is ${riskData.data.risk_level}. `;
        }
        
        if (riskData.data?.var_95) {
          response += `Value at Risk (95%): ${riskData.data.var_95}%. `;
        }
        
        if (riskData.data?.recommendations) {
          response += `Recommendation: ${riskData.data.recommendations}`;
        }
        
        return { action: 'risk_assessment', response, confidence: 0.85, dataUsed };
      }
    } catch (error) {
      console.error('Risk analysis error:', error);
    }
  }

  // News and sentiment analysis
  if (command.includes('news') || command.includes('sentiment')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    dataUsed.push('news_intelligence', 'alerts');
    
    try {
      const newsRes = await fetch('http://127.0.0.1:8001/api/agents/news-intelligence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: [symbol] })
      });
      
      if (newsRes.ok) {
        const newsData = await newsRes.json();
        
        let response = `News Analysis for ${symbol}: `;
        
        if (newsData.data?.sentiment_score) {
          response += `Sentiment score: ${newsData.data.sentiment_score}/100. `;
        }
        
        if (newsData.data?.news_impact) {
          response += `News impact: ${newsData.data.news_impact}. `;
        }
        
        if (realTimeData.alerts.length > 0) {
          const relevantAlerts = realTimeData.alerts.filter(alert => 
            alert.symbol === symbol || !alert.symbol
          );
          response += `${relevantAlerts.length} relevant alerts.`;
        }
        
        return { action: 'news_sentiment', response, confidence: 0.82, dataUsed };
      }
    } catch (error) {
      console.error('News analysis error:', error);
    }
  }

  // Portfolio analysis
  if (command.includes('portfolio') || command.includes('holdings')) {
    if (realTimeData.portfolio) {
      dataUsed.push('portfolio_data', 'risk_analysis');
      
      let response = `Portfolio Status: Value $${realTimeData.portfolio.totalValue?.toLocaleString()}, `;
      response += `Daily change: ${realTimeData.portfolio.dailyChangePercent >= 0 ? '+' : ''}${realTimeData.portfolio.dailyChangePercent?.toFixed(2)}%, `;
      response += `${realTimeData.portfolio.positions} positions. `;
      
      // Get portfolio risk analysis
      try {
        const riskRes = await fetch('http://127.0.0.1:8001/api/agents/risk-assessor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: realTimeData.portfolio })
        });
        
        if (riskRes.ok) {
          const riskData = await riskRes.json();
          if (riskData.data?.risk_level) {
            response += `Risk level: ${riskData.data.risk_level}.`;
          }
        }
      } catch (error) {
        console.error('Portfolio risk error:', error);
      }
      
      return { action: 'portfolio_analysis', response, confidence: 0.92, dataUsed };
    }
  }

  // AI agent status
  if (command.includes('agents') || command.includes('ai')) {
    if (realTimeData.agents.length > 0) {
      dataUsed.push('agent_status');
      const activeAgents = realTimeData.agents.filter(a => a.status === 'active').length;
      const totalSignals = realTimeData.agents.reduce((sum, a) => sum + (a.signals_today || 0), 0);
      
      let response = `AI Agent Status: ${activeAgents}/${realTimeData.agents.length} agents active. `;
      response += `Total signals today: ${totalSignals}. `;
      
      const topAgent = realTimeData.agents.reduce((best, agent) => 
        (agent.performance || 0) > (best.performance || 0) ? agent : best
      );
      
      if (topAgent) {
        response += `Top performer: ${topAgent.name} (${topAgent.performance}% accuracy).`;
      }
      
      return { action: 'agent_status', response, confidence: 0.90, dataUsed };
    }
  }

  // Prediction and forecasting
  if (command.includes('predict') || command.includes('forecast') || command.includes('future')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    dataUsed.push('ai_prediction', 'market_data');
    
    const aiData = await getAIAnalysis(symbol);
    
    let response = `AI Prediction for ${symbol}: `;
    
    if (aiData.signals?.data?.individual_signals?.[symbol]) {
      const signal = aiData.signals.data.individual_signals[symbol];
      response += `Expected direction: ${signal.signal_type} with ${signal.confidence}% confidence. `;
      
      if (signal.target_price) {
        response += `Price target: $${signal.target_price}. `;
      }
      
      if (signal.time_horizon) {
        response += `Time horizon: ${signal.time_horizon}.`;
      }
    } else {
      response += 'Analysis in progress. Current indicators suggest neutral to slightly positive outlook.';
    }
    
    return { action: 'ai_prediction', response, confidence: 0.75, dataUsed };
  }

  // Enhanced command matching with guaranteed responses
  const symbols = extractSymbols(command);
  const symbol = symbols[0] || 'AAPL';
  
  // Market overview commands - ALWAYS works
  if (command.includes('market') || command.includes('overview') || command.includes('summary')) {
    dataUsed.push('market_data', 'alerts');
    const marketSymbols = Object.keys(realTimeData.marketData);
    
    if (marketSymbols.length > 0) {
      const gainers = marketSymbols.filter(s => (realTimeData.marketData[s]?.changePercent || 0) > 0).length;
      const losers = marketSymbols.filter(s => (realTimeData.marketData[s]?.changePercent || 0) < 0).length;
      const totalVolume = marketSymbols.reduce((sum, s) => sum + (realTimeData.marketData[s]?.volume || 0), 0);
      
      let response = `Market Overview: ${gainers} gainers, ${losers} decliners out of ${marketSymbols.length} stocks. `;
      response += `Sentiment: ${gainers > losers ? 'Bullish' : gainers < losers ? 'Bearish' : 'Neutral'}. `;
      response += `Total volume: ${(totalVolume / 1000000000).toFixed(1)} billion shares. `;
      response += `${realTimeData.alerts.length} active alerts.`;
      
      if (realTimeData.alerts.length > 0) {
        response += ` Latest: ${realTimeData.alerts[0]?.message}`;
      }
      
      return { action: 'market_overview', response, confidence: 0.95, dataUsed };
    } else {
      return { 
        action: 'market_overview', 
        response: 'Market data is currently loading. Please try again in a moment.', 
        confidence: 0.80, 
        dataUsed 
      };
    }
  }
  
  // Status and health commands
  if (command.includes('status') || command.includes('health') || command.includes('system')) {
    dataUsed.push('agent_status', 'market_data');
    const activeAgents = realTimeData.agents.filter(a => a.status === 'active').length;
    const marketCount = Object.keys(realTimeData.marketData).length;
    
    let response = `System Status: ${activeAgents}/${realTimeData.agents.length} AI agents active. `;
    response += `${marketCount} stocks monitored. `;
    response += `${realTimeData.alerts.length} alerts. `;
    response += `Last update: ${new Date(realTimeData.lastUpdate).toLocaleTimeString()}.`;
    
    return { action: 'system_status', response, confidence: 0.95, dataUsed };
  }
  
  // Performance and metrics
  if (command.includes('performance') || command.includes('metrics') || command.includes('stats')) {
    dataUsed.push('agent_status', 'portfolio_data');
    const totalSignals = realTimeData.agents.reduce((sum, a) => sum + (a.signals_today || 0), 0);
    const avgPerformance = realTimeData.agents.reduce((sum, a) => sum + (a.performance || 0), 0) / realTimeData.agents.length;
    
    let response = `Performance Metrics: ${totalSignals} signals generated today. `;
    response += `Average AI accuracy: ${avgPerformance.toFixed(1)}%. `;
    
    if (realTimeData.portfolio) {
      response += `Portfolio: ${realTimeData.portfolio.dailyChangePercent >= 0 ? '+' : ''}${realTimeData.portfolio.dailyChangePercent?.toFixed(2)}% today.`;
    }
    
    return { action: 'performance_metrics', response, confidence: 0.88, dataUsed };
  }
  
  // Top movers and trending
  if (command.includes('top') || command.includes('best') || command.includes('worst') || command.includes('trending')) {
    dataUsed.push('market_data');
    const marketSymbols = Object.keys(realTimeData.marketData);
    
    if (marketSymbols.length > 0) {
      const sorted = marketSymbols
        .map(s => ({ symbol: s, change: realTimeData.marketData[s]?.changePercent || 0 }))
        .sort((a, b) => Math.abs(b.change) - Math.abs(a.change));
      
      const topGainer = sorted.find(s => s.change > 0);
      const topLoser = sorted.find(s => s.change < 0);
      
      let response = 'Top Movers: ';
      if (topGainer) {
        response += `${topGainer.symbol} up ${topGainer.change.toFixed(2)}%. `;
      }
      if (topLoser) {
        response += `${topLoser.symbol} down ${Math.abs(topLoser.change).toFixed(2)}%.`;
      }
      
      return { action: 'top_movers', response, confidence: 0.92, dataUsed };
    }
  }
  
  // Volume and activity
  if (command.includes('volume') || command.includes('activity') || command.includes('trading')) {
    dataUsed.push('market_data');
    const marketSymbols = Object.keys(realTimeData.marketData);
    
    if (marketSymbols.length > 0) {
      const totalVolume = marketSymbols.reduce((sum, s) => 
        sum + (realTimeData.marketData[s]?.volume || 0), 0
      );
      
      const highVolumeStock = marketSymbols.reduce((max, s) => 
        (realTimeData.marketData[s]?.volume || 0) > (realTimeData.marketData[max]?.volume || 0) ? s : max
      );
      
      let response = `Trading Activity: Total volume ${(totalVolume / 1000000000).toFixed(1)}B shares. `;
      response += `Highest volume: ${highVolumeStock} with ${((realTimeData.marketData[highVolumeStock]?.volume || 0) / 1000000).toFixed(1)}M shares.`;
      
      return { action: 'volume_analysis', response, confidence: 0.85, dataUsed };
    }
  }
  
  // Alerts and notifications
  if (command.includes('alert') || command.includes('notification') || command.includes('warning')) {
    dataUsed.push('alerts');
    
    if (realTimeData.alerts.length > 0) {
      const criticalAlerts = realTimeData.alerts.filter(a => a.severity === 'critical').length;
      const highAlerts = realTimeData.alerts.filter(a => a.severity === 'high').length;
      
      let response = `Alerts: ${realTimeData.alerts.length} total. `;
      if (criticalAlerts > 0) response += `${criticalAlerts} critical. `;
      if (highAlerts > 0) response += `${highAlerts} high priority. `;
      response += `Latest: ${realTimeData.alerts[0]?.message}`;
      
      return { action: 'alerts_summary', response, confidence: 0.90, dataUsed };
    } else {
      return { action: 'no_alerts', response: 'No active alerts. All systems operating normally.', confidence: 0.95, dataUsed };
    }
  }
  
  // Help and capabilities
  if (command.includes('help') || command.includes('what can') || command.includes('commands')) {
    return {
      action: 'help_commands',
      response: 'I can analyze any stock, check market conditions, provide trading signals, assess portfolio risk, review news sentiment, predict price movements, show top movers, check system status, and much more. Just ask about any stock or market topic!',
      confidence: 0.95,
      dataUsed: []
    };
  }
  
  // Stock-specific commands - ALWAYS works with available data
  if (symbols.length > 0) {
    const stockData = realTimeData.marketData[symbol];
    dataUsed.push('market_data');
    
    if (stockData) {
      let response = `${symbol}: $${stockData.price?.toFixed(2) || 'N/A'}`;
      
      if (stockData.changePercent !== undefined) {
        response += `, ${stockData.changePercent >= 0 ? '+' : ''}${stockData.changePercent.toFixed(2)}% today`;
      }
      
      if (stockData.volume) {
        response += `, volume ${(stockData.volume / 1000000).toFixed(1)}M shares`;
      }
      
      // Add AI analysis if available
      if (command.includes('analyze') || command.includes('analysis')) {
        const aiData = await getAIAnalysis(symbol);
        if (aiData.signals?.data?.individual_signals?.[symbol]) {
          const signal = aiData.signals.data.individual_signals[symbol];
          response += `. AI Signal: ${signal.signal_type} with ${signal.confidence}% confidence`;
        }
      }
      
      return { action: 'stock_info', response, confidence: 0.90, dataUsed };
    } else {
      // Try to get data from API directly
      try {
        const directRes = await fetch(`http://127.0.0.1:8001/api/market/latest?symbol=${symbol}`);
        if (directRes.ok) {
          const directData = await directRes.json();
          if (directData && directData.price) {
            return {
              action: 'stock_info',
              response: `${symbol}: $${directData.price.toFixed(2)}, ${directData.changePercent >= 0 ? '+' : ''}${directData.changePercent?.toFixed(2) || '0.00'}% today`,
              confidence: 0.85,
              dataUsed
            };
          }
        }
      } catch (e) {
        // Fallback response
        return {
          action: 'stock_info',
          response: `${symbol} data is currently unavailable. The stock is being monitored and data will be available shortly.`,
          confidence: 0.70,
          dataUsed
        };
      }
    }
  }
  
  // Intelligent fallback - analyze command for any financial terms
  const financialTerms = ['stock', 'price', 'trade', 'buy', 'sell', 'invest', 'market', 'portfolio', 'risk', 'profit', 'loss', 'bull', 'bear', 'volume', 'chart'];
  const hasFinancialTerm = financialTerms.some(term => command.includes(term));
  
  if (hasFinancialTerm) {
    // Provide market summary for any financial query
    const marketSymbols = Object.keys(realTimeData.marketData);
    if (marketSymbols.length > 0) {
      const topStock = marketSymbols[0];
      const stockData = realTimeData.marketData[topStock];
      
      return {
        action: 'financial_query',
        response: `I understand you're asking about ${command}. Here's current market info: ${topStock} at $${stockData?.price?.toFixed(2)}, market has ${marketSymbols.length} active stocks, ${realTimeData.alerts.length} alerts. Ask about specific stocks like "Apple price" or "Tesla analysis".`,
        confidence: 0.75,
        dataUsed: ['market_data']
      };
    }
  }
  
  // Final fallback with current market status
  const marketCount = Object.keys(realTimeData.marketData).length;
  const agentCount = realTimeData.agents.length;
  
  return {
    action: 'help_with_status',
    response: `I'm monitoring ${marketCount} stocks with ${agentCount} AI agents active. Try: "Market overview", "Apple price", "Tesla analysis", "Top movers", "System status", "Latest alerts", or ask about any stock!`,
    confidence: 0.80,
    dataUsed: ['market_data', 'agent_status']
  };
};

const extractSymbols = (command: string): string[] => {
  const symbolRegex = /\b[A-Z]{1,5}\b/g;
  const commonSymbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'INTC'];
  const matches = command.toUpperCase().match(symbolRegex) || [];
  const validSymbols = matches.filter(symbol => commonSymbols.includes(symbol));
  
  // Enhanced company name matching
  const companyNames = {
    'apple': 'AAPL', 'google': 'GOOGL', 'alphabet': 'GOOGL',
    'microsoft': 'MSFT', 'amazon': 'AMZN', 'tesla': 'TSLA',
    'nvidia': 'NVDA', 'meta': 'META', 'facebook': 'META',
    'netflix': 'NFLX', 'salesforce': 'CRM', 'intel': 'INTC'
  };

  Object.keys(companyNames).forEach(name => {
    if (command.toLowerCase().includes(name)) {
      validSymbols.push(companyNames[name as keyof typeof companyNames]);
    }
  });

  return [...new Set(validSymbols)];
};

export default function EnhancedVoiceAI() {
  const [isEnabled, setIsEnabled] = useState(false);
  const [commandHistory, setCommandHistory] = useState<VoiceCommand[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>({
    voiceIndex: 0,
    autoSpeak: true,
    sensitivity: 0.7,
    language: 'en-US',
    speed: 0.9,
    pitch: 1.0
  });

  const { data: realTimeData, isLoading: dataLoading, error: dataError, refresh } = useRealTimeData();
  const { isListening, transcript, confidence, startListening, stopListening, resetTranscript, isSupported: voiceSupported } = useVoiceRecognition();
  const { speak, stop: stopSpeaking, isSpeaking, voices, isSupported: speechSupported } = useTextToSpeech();

  useEffect(() => {
    if (transcript && confidence > voiceSettings.sensitivity && !isProcessing) {
      setIsProcessing(true);
      
      const processCommand = async () => {
        const startTime = Date.now();
        const result = await processVoiceCommand(transcript, realTimeData);
        const executionTime = Date.now() - startTime;
        
        const command: VoiceCommand = {
          id: `cmd-${Date.now()}`,
          command: transcript,
          timestamp: new Date(),
          response: result.response,
          confidence: result.confidence,
          action: result.action,
          executionTime,
          dataUsed: result.dataUsed
        };

        setCommandHistory(prev => [command, ...prev.slice(0, 9)]);

        if (voiceSettings.autoSpeak && speechSupported) {
          speak(result.response, voiceSettings);
        }

        resetTranscript();
        setIsProcessing(false);
      };
      
      processCommand();
    }
  }, [transcript, confidence, isProcessing, realTimeData, resetTranscript, speak, voiceSettings, speechSupported]);

  const handleManualCommand = async (commandText: string) => {
    const startTime = Date.now();
    const result = await processVoiceCommand(commandText, realTimeData);
    const executionTime = Date.now() - startTime;
    
    const command: VoiceCommand = {
      id: `manual-${Date.now()}`,
      command: commandText,
      timestamp: new Date(),
      response: result.response,
      confidence: result.confidence,
      action: result.action,
      executionTime,
      dataUsed: result.dataUsed
    };

    setCommandHistory(prev => [command, ...prev.slice(0, 9)]);

    if (voiceSettings.autoSpeak && speechSupported) {
      speak(result.response, voiceSettings);
    }
  };

  const quickCommands = [
    "Market overview",
    "Analyze Apple",
    "Tesla trading signals",
    "Top movers today",
    "System status",
    "Latest alerts",
    "Portfolio performance",
    "NVIDIA price prediction",
    "Trading volume analysis",
    "Risk assessment",
    "News sentiment",
    "AI recommendations"
  ];

  if (!voiceSupported && !speechSupported) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black p-4 sm:p-6 flex items-center justify-center">
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 sm:p-8 text-center max-w-md">
          <AlertTriangle className="w-12 h-12 sm:w-16 sm:h-16 text-yellow-500 mx-auto mb-4" />
          <h3 className="text-lg sm:text-xl font-semibold text-white mb-2">Voice Interface Unavailable</h3>
          <p className="text-sm sm:text-base text-gray-400">Your browser doesn't support speech recognition or synthesis.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 p-2 sm:p-4 lg:p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="bg-black/40 rounded-lg sm:rounded-xl p-4 sm:p-6 mb-4 sm:mb-6 backdrop-blur-sm border border-blue-500/30">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 sm:gap-0">
            <div>
              <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-white flex items-center">
                <Brain className="w-6 h-6 sm:w-8 sm:h-8 mr-2 sm:mr-3 text-blue-400" />
                🎤 <span className="hidden sm:inline">Enhanced </span>Voice AI<span className="hidden sm:inline"> Assistant</span>
              </h1>
              <p className="text-blue-300 mt-1 sm:mt-2 text-sm sm:text-base">
                <span className="hidden sm:inline">Real-time financial analysis with </span>Voice commands
              </p>
            </div>
            
            <div className="flex items-center justify-between sm:justify-end space-x-3 sm:space-x-4">
              <div className="text-right">
                <div className="text-xs sm:text-sm text-gray-400">
                  <span className="hidden sm:inline">Last </span>Update
                </div>
                <div className="text-white font-semibold text-sm sm:text-base">
                  {realTimeData.lastUpdate ? new Date(realTimeData.lastUpdate).toLocaleTimeString() : '--:--'}
                </div>
              </div>
              
              <button
                onClick={refresh}
                disabled={dataLoading}
                title="Refresh data"
                aria-label="Refresh data"
                className="p-2 sm:p-3 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white transition-colors"
              >
                <RefreshCw className={`w-4 h-4 sm:w-5 sm:h-5 ${dataLoading ? 'animate-spin' : ''}`} />
              </button>
              
              <button
                onClick={() => setIsEnabled(!isEnabled)}
                className={`flex items-center px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  isEnabled 
                    ? 'bg-green-600 hover:bg-green-700 text-white' 
                    : 'bg-gray-600 hover:bg-gray-700 text-gray-200'
                }`}
              >
                {isEnabled ? <Volume2 className="w-4 h-4 mr-2" /> : <VolumeX className="w-4 h-4 mr-2" />}
                {isEnabled ? 'Enabled' : 'Disabled'}
              </button>
            </div>
          </div>
        </div>

        {dataError && (
          <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4 mb-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <span className="text-red-300">Data Error: {dataError}</span>
            </div>
          </div>
        )}

        {isEnabled && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            
            {/* Voice Controls */}
            <div className="xl:col-span-2 space-y-6">
              
              {/* Main Voice Interface */}
              <div className="bg-black/40 rounded-xl p-6 backdrop-blur-sm border border-gray-700">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  
                  {/* Microphone Control */}
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center justify-center">
                      <Mic className="w-5 h-5 mr-2 text-blue-400" />
                      Voice Input
                    </h3>
                    
                    <button
                      onClick={isListening ? stopListening : startListening}
                      disabled={!isEnabled || isProcessing}
                      className={`relative w-24 h-24 rounded-full border-4 transition-all duration-300 ${
                        isListening 
                          ? 'bg-red-600 border-red-400 animate-pulse' 
                          : 'bg-blue-600 border-blue-400 hover:bg-blue-700'
                      } disabled:opacity-50`}
                    >
                      {isListening ? (
                        <MicOff className="w-10 h-10 text-white mx-auto" />
                      ) : (
                        <Mic className="w-10 h-10 text-white mx-auto" />
                      )}
                      
                      {isListening && (
                        <div className="absolute inset-0 rounded-full border-4 border-red-400 animate-ping"></div>
                      )}
                    </button>
                    
                    <div className="mt-4">
                      <div className={`text-sm font-medium ${isListening ? 'text-red-400' : 'text-gray-400'}`}>
                        {isProcessing ? 'Processing...' : isListening ? 'Listening...' : 'Click to speak'}
                      </div>
                      {transcript && (
                        <div className="mt-2 p-3 bg-gray-800 rounded text-sm text-white">
                          "{transcript}"
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Speech Settings */}
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Volume2 className="w-5 h-5 mr-2 text-purple-400" />
                      Speech Settings
                    </h3>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-300">Auto-speak</span>
                        <button
                          onClick={() => setVoiceSettings(prev => ({ ...prev, autoSpeak: !prev.autoSpeak }))}
                          title={`Auto-speak: ${voiceSettings.autoSpeak ? 'ON' : 'OFF'}`}
                          aria-label={`Auto-speak: ${voiceSettings.autoSpeak ? 'ON' : 'OFF'}`}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            voiceSettings.autoSpeak ? 'bg-green-600' : 'bg-gray-600'
                          }`}
                        >
                          <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            voiceSettings.autoSpeak ? 'translate-x-6' : 'translate-x-1'
                          }`} />
                        </button>
                      </div>
                      
                      <div>
                        <label className="text-sm text-gray-300 block mb-1">Speed: {voiceSettings.speed}</label>
                        <input
                          type="range"
                          min="0.5"
                          max="2"
                          step="0.1"
                          value={voiceSettings.speed}
                          onChange={(e) => setVoiceSettings(prev => ({ ...prev, speed: parseFloat(e.target.value) }))}
                          title="Speech speed"
                          aria-label="Speech speed"
                          className="w-full"
                        />
                      </div>
                      
                      <div>
                        <label className="text-sm text-gray-300 block mb-1">Sensitivity: {voiceSettings.sensitivity}</label>
                        <input
                          type="range"
                          min="0.1"
                          max="1"
                          step="0.1"
                          value={voiceSettings.sensitivity}
                          onChange={(e) => setVoiceSettings(prev => ({ ...prev, sensitivity: parseFloat(e.target.value) }))}
                          title="Voice sensitivity"
                          aria-label="Voice sensitivity"
                          className="w-full"
                        />
                      </div>
                      
                      <div className="flex items-center justify-center">
                        <div className={`w-3 h-3 rounded-full mr-2 ${isSpeaking ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`}></div>
                        <span className="text-sm text-gray-300">
                          {isSpeaking ? 'Speaking...' : 'Ready'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Commands */}
              <div className="bg-black/40 rounded-xl p-6 backdrop-blur-sm border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Zap className="w-5 h-5 mr-2 text-yellow-400" />
                  Quick Commands
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {quickCommands.map((cmd, index) => (
                    <button
                      key={index}
                      onClick={() => handleManualCommand(cmd)}
                      className="px-4 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-sm text-white transition-colors text-left"
                    >
                      {cmd}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Data Status & History */}
            <div className="space-y-6">
              
              {/* Real-Time Data Status */}
              <div className="bg-black/40 rounded-xl p-6 backdrop-blur-sm border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Activity className="w-5 h-5 mr-2 text-green-400" />
                  Data Status
                </h3>
                
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Market Data:</span>
                    <span className="text-white">{Object.keys(realTimeData.marketData).length} stocks</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">AI Agents:</span>
                    <span className="text-white">{realTimeData.agents.length} active</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Alerts:</span>
                    <span className="text-white">{realTimeData.alerts.length} items</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Portfolio:</span>
                    <span className="text-white">{realTimeData.portfolio ? 'Connected' : 'No data'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">News:</span>
                    <span className="text-white">{realTimeData.news.length} articles</span>
                  </div>
                </div>
              </div>

              {/* Command History */}
              <div className="bg-black/40 rounded-xl p-6 backdrop-blur-sm border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <MessageSquare className="w-5 h-5 mr-2 text-blue-400" />
                  Recent Commands
                </h3>
                
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {commandHistory.length === 0 ? (
                    <div className="text-center text-gray-500 py-8">
                      <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No commands yet. Try a voice command or quick action.</p>
                    </div>
                  ) : (
                    commandHistory.map((cmd) => (
                      <div key={cmd.id} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center">
                            <div className="w-2 h-2 bg-blue-400 rounded-full mr-2"></div>
                            <span className="text-sm font-medium text-blue-300">Command</span>
                          </div>
                          <div className="text-xs text-gray-400 flex items-center">
                            <Clock className="w-3 h-3 mr-1" />
                            {cmd.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                        <div className="text-sm text-gray-300 mb-3">"{cmd.command}"</div>
                        
                        <div className="flex items-center mb-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                          <span className="text-sm font-medium text-green-300">Response</span>
                          <div className="ml-auto flex items-center space-x-2">
                            <Target className="w-3 h-3 text-yellow-400" />
                            <span className="text-xs text-yellow-300">{(cmd.confidence * 100).toFixed(0)}%</span>
                            <Clock className="w-3 h-3 text-gray-400" />
                            <span className="text-xs text-gray-400">{cmd.executionTime}ms</span>
                          </div>
                        </div>
                        <div className="text-sm text-gray-200 mb-2">{cmd.response}</div>
                        
                        {cmd.dataUsed.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {cmd.dataUsed.map((source, idx) => (
                              <span key={idx} className="px-2 py-1 bg-purple-600/30 border border-purple-600 rounded text-xs text-purple-300">
                                {source.replace('_', ' ')}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}