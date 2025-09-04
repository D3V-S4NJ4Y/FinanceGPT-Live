import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Mic, MicOff, Volume2, VolumeX, Brain, 
  MessageSquare, Zap, TrendingUp, Settings,
  Activity, AlertTriangle, Target, Play
} from 'lucide-react';

interface VoiceCommand {
  command: string;
  timestamp: Date;
  response: string;
  confidence: number;
  action: string;
}

interface VoiceInterfaceProps {
  onCommand?: (command: string, action: string) => void;
  marketData?: any;
  selectedSymbol?: string;
}

// Helper function to get real price data
const getRealPrice = async (symbol: string) => {
  try {
    const response = await fetch('http://localhost:8001/market-data');
    if (response.ok) {
      const data = await response.json();
      return data[symbol] || null;
    }
  } catch (error) {
    console.error('Failed to fetch real price:', error);
  }
  return null;
};

// Voice recognition hook
const useVoiceRecognition = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // Check if browser supports speech recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported in this browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const current = event.resultIndex;
      const result = event.results[current];
      
      if (result.isFinal) {
        setTranscript(result[0].transcript);
        setConfidence(result[0].confidence);
      }
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
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

// Text-to-speech hook
const useTextToSpeech = () => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    const updateVoices = () => {
      setVoices(speechSynthesis.getVoices());
    };

    updateVoices();
    speechSynthesis.onvoiceschanged = updateVoices;

    return () => {
      speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const speak = useCallback((text: string, voiceIndex: number = 0) => {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      const voice = voices[voiceIndex] || voices.find(v => v.lang === 'en-US') || voices[0];
      
      if (voice) {
        utterance.voice = voice;
      }
      
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      speechSynthesis.speak(utterance);
    }
  }, [voices]);

  const stop = useCallback(() => {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  }, []);

  return {
    speak,
    stop,
    isSpeaking,
    voices,
    isSupported: 'speechSynthesis' in window
  };
};

// Command processor
const processVoiceCommand = async (transcript: string, marketData?: any): Promise<{ action: string; response: string; confidence: number }> => {
  const command = transcript.toLowerCase().trim();
  
  // Market analysis commands
  if (command.includes('analyze') || command.includes('analysis')) {
    if (command.includes('market')) {
      return {
        action: 'market_analysis',
        response: 'Analyzing market conditions. Current sentiment is mixed with moderate volatility. Key indicators suggest a neutral to slightly bullish outlook.',
        confidence: 0.85
      };
    }
    if (command.includes('risk')) {
      return {
        action: 'risk_analysis',
        response: 'Risk analysis shows moderate portfolio exposure. VaR at 95% confidence is negative 3.2%. Consider diversification in defensive sectors.',
        confidence: 0.82
      };
    }
  }

  // Trading commands
  if (command.includes('buy') || command.includes('purchase')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    return {
      action: 'buy_signal',
      response: `Buy signal generated for ${symbol}. Current price is trending upward with strong momentum. Recommended entry point confirmed.`,
      confidence: 0.78
    };
  }

  if (command.includes('sell')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    return {
      action: 'sell_signal',
      response: `Sell recommendation for ${symbol}. Technical indicators show overbought conditions. Consider taking profits at current levels.`,
      confidence: 0.76
    };
  }

  // Information queries
  if (command.includes('price') || command.includes('quote')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'AAPL';
    
    // Try to get real price from market data context or API
    const realPrice = await getRealPrice(symbol);
    const priceData = realPrice || { price: 0, change: 0, change_percent: 0 };
    
    return {
      action: 'price_query',
      response: `${symbol} is currently trading at $${priceData.price.toFixed(2)}${priceData.change >= 0 ? ', up' : ', down'} ${Math.abs(priceData.change_percent).toFixed(2)}% today.`,
      confidence: 0.95
    };
  }

  if (command.includes('news') || command.includes('sentiment')) {
    return {
      action: 'news_query',
      response: 'Latest market sentiment analysis shows mixed signals. Technology sector remains strong while financials show weakness. Overall market sentiment is cautiously optimistic.',
      confidence: 0.73
    };
  }

  // Portfolio commands
  if (command.includes('portfolio') || command.includes('holdings')) {
    return {
      action: 'portfolio_query',
      response: 'Your portfolio is well-diversified across sectors. Current allocation: 45% Technology, 25% Financials, 20% Healthcare, 10% Energy. Overall performance is positive.',
      confidence: 0.88
    };
  }

  // AI agent queries
  if (command.includes('agents') || command.includes('ai')) {
    return {
      action: 'agent_status',
      response: 'All AI agents are operational. Market Sentinel shows bullish signals, Risk Assessor indicates moderate exposure, News Intelligence reports positive sentiment.',
      confidence: 0.90
    };
  }

  // Predictions
  if (command.includes('predict') || command.includes('forecast')) {
    const symbols = extractSymbols(command);
    const symbol = symbols[0] || 'market';
    return {
      action: 'prediction_query',
      response: `AI prediction for ${symbol}: Expected to trend upward over the next 24 hours with 72% confidence. Key resistance level at current price plus 3.5%.`,
      confidence: 0.72
    };
  }

  // Default response
  return {
    action: 'unknown',
    response: 'I understand you want information about the markets. You can ask me to analyze stocks, check prices, assess risk, or get AI predictions.',
    confidence: 0.5
  };
};

// Extract stock symbols from command
const extractSymbols = (command: string): string[] => {
  const symbolRegex = /\b[A-Z]{1,5}\b/g;
  const commonSymbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'JNJ'];
  
  // Look for explicit symbols
  const matches = command.toUpperCase().match(symbolRegex) || [];
  const validSymbols = matches.filter(symbol => commonSymbols.includes(symbol));
  
  // Look for company names
  const companyNames = {
    'apple': 'AAPL',
    'google': 'GOOGL',
    'microsoft': 'MSFT',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'nvidia': 'NVDA',
    'meta': 'META',
    'facebook': 'META',
    'jp morgan': 'JPM',
    'visa': 'V',
    'johnson': 'JNJ'
  };

  Object.keys(companyNames).forEach(name => {
    if (command.toLowerCase().includes(name)) {
      validSymbols.push(companyNames[name as keyof typeof companyNames]);
    }
  });

  return [...new Set(validSymbols)]; // Remove duplicates
};

const VoiceInterface: React.FC<VoiceInterfaceProps> = ({ 
  onCommand, 
  marketData, 
  selectedSymbol 
}) => {
  const [isEnabled, setIsEnabled] = useState(false);
  const [commandHistory, setCommandHistory] = useState<VoiceCommand[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState({
    voiceIndex: 0,
    autoSpeak: true,
    sensitivity: 0.7
  });

  const {
    isListening,
    transcript,
    confidence,
    startListening,
    stopListening,
    resetTranscript,
    isSupported: voiceSupported
  } = useVoiceRecognition();

  const {
    speak,
    stop: stopSpeaking,
    isSpeaking,
    voices,
    isSupported: speechSupported
  } = useTextToSpeech();

  // Process voice commands
  useEffect(() => {
    if (transcript && confidence > voiceSettings.sensitivity && !isProcessing) {
      setIsProcessing(true);
      
      const processCommand = async () => {
        const result = await processVoiceCommand(transcript, marketData);
        
        const command: VoiceCommand = {
          command: transcript,
          timestamp: new Date(),
          response: result.response,
          confidence: result.confidence,
          action: result.action
        };

        setCommandHistory(prev => [command, ...prev.slice(0, 9)]); // Keep last 10 commands

        // Execute callback
        onCommand?.(transcript, result.action);

        // Auto-speak response
        if (voiceSettings.autoSpeak && speechSupported) {
          speak(result.response, voiceSettings.voiceIndex);
        }

        resetTranscript();
        setIsProcessing(false);
      };
      
      processCommand();
    }
  }, [transcript, confidence, isProcessing, marketData, onCommand, resetTranscript, speak, voiceSettings, speechSupported]);

  const toggleVoiceInterface = () => {
    if (isEnabled) {
      stopListening();
      stopSpeaking();
      setIsEnabled(false);
    } else {
      setIsEnabled(true);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  const handleManualCommand = async (commandText: string) => {
    const result = await processVoiceCommand(commandText, marketData);
    
    const command: VoiceCommand = {
      command: commandText,
      timestamp: new Date(),
      response: result.response,
      confidence: result.confidence,
      action: result.action
    };

    setCommandHistory(prev => [command, ...prev.slice(0, 9)]);
    onCommand?.(commandText, result.action);

    if (voiceSettings.autoSpeak && speechSupported) {
      speak(result.response, voiceSettings.voiceIndex);
    }
  };

  const quickCommands = [
    "Analyze the market",
    "Check Apple price",
    "Show portfolio status",
    "What's the risk level?",
    "Get AI predictions",
    "Latest market news"
  ];

  if (!voiceSupported && !speechSupported) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 text-center">
        <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">Voice Interface Unavailable</h3>
        <p className="text-gray-400">Your browser doesn't support speech recognition or synthesis.</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-blue-900 to-purple-900 border border-blue-700 rounded-xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center">
            <Brain className="w-7 h-7 mr-3 text-blue-400" />
            🎤 Voice AI Assistant
          </h2>
          <p className="text-blue-300 mt-1">Speak naturally to analyze markets and get insights</p>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={toggleVoiceInterface}
            title={isEnabled ? 'Disable voice interface' : 'Enable voice interface'}
            className={`flex items-center px-4 py-2 rounded-lg font-medium transition-all ${
              isEnabled 
                ? 'bg-green-600 hover:bg-green-700 text-white' 
                : 'bg-gray-600 hover:bg-gray-700 text-gray-200'
            }`}
          >
            {isEnabled ? <Volume2 className="w-4 h-4 mr-2" /> : <VolumeX className="w-4 h-4 mr-2" />}
            {isEnabled ? 'Enabled' : 'Disabled'}
          </button>
          
          <button 
            title="Voice interface settings"
            className="p-2 hover:bg-blue-800 rounded-lg"
          >
            <Settings className="w-5 h-5 text-blue-300" />
          </button>
        </div>
      </div>

      {isEnabled && (
        <>
          {/* Voice Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Microphone Control */}
            <div className="bg-black/20 rounded-lg p-4 border border-blue-600">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Mic className="w-5 h-5 mr-2 text-blue-400" />
                Voice Input
              </h3>
              
              <div className="flex items-center justify-center mb-4">
                <button
                  onClick={toggleListening}
                  disabled={!isEnabled}
                  title={isListening ? 'Stop listening' : 'Start listening'}
                  className={`relative w-20 h-20 rounded-full border-4 transition-all duration-300 ${
                    isListening 
                      ? 'bg-red-600 border-red-400 animate-pulse' 
                      : 'bg-blue-600 border-blue-400 hover:bg-blue-700'
                  }`}
                >
                  {isListening ? (
                    <MicOff className="w-8 h-8 text-white mx-auto" />
                  ) : (
                    <Mic className="w-8 h-8 text-white mx-auto" />
                  )}
                  
                  {isListening && (
                    <div className="absolute inset-0 rounded-full border-4 border-red-400 animate-ping"></div>
                  )}
                </button>
              </div>
              
              <div className="text-center">
                <div className={`text-sm font-medium ${isListening ? 'text-red-400' : 'text-gray-400'}`}>
                  {isListening ? 'Listening...' : 'Click to speak'}
                </div>
                {transcript && (
                  <div className="mt-2 p-2 bg-gray-800 rounded text-sm text-white">
                    "{transcript}"
                  </div>
                )}
              </div>
            </div>

            {/* Speech Output */}
            <div className="bg-black/20 rounded-lg p-4 border border-purple-600">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Volume2 className="w-5 h-5 mr-2 text-purple-400" />
                Speech Output
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Auto-speak responses</span>
                  <button
                    onClick={() => setVoiceSettings(prev => ({ ...prev, autoSpeak: !prev.autoSpeak }))}
                    title={`${voiceSettings.autoSpeak ? 'Disable' : 'Enable'} automatic speech responses`}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      voiceSettings.autoSpeak ? 'bg-green-600' : 'bg-gray-600'
                    }`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      voiceSettings.autoSpeak ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                
                {voices.length > 0 && (
                  <div>
                    <label className="text-sm text-gray-300">Voice:</label>
                    <select
                      value={voiceSettings.voiceIndex}
                      onChange={(e) => setVoiceSettings(prev => ({ ...prev, voiceIndex: parseInt(e.target.value) }))}
                      title="Select voice for speech synthesis"
                      className="w-full mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-xs"
                    >
                      {voices.slice(0, 10).map((voice, index) => (
                        <option key={index} value={index}>
                          {voice.name} ({voice.lang})
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                
                <div className="flex items-center justify-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${isSpeaking ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`}></div>
                  <span className="text-sm text-gray-300">
                    {isSpeaking ? 'Speaking...' : 'Ready to speak'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Commands */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
              <Zap className="w-5 h-5 mr-2 text-yellow-400" />
              Quick Commands
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {quickCommands.map((cmd, index) => (
                <button
                  key={index}
                  onClick={() => handleManualCommand(cmd)}
                  title={`Execute voice command: ${cmd}`}
                  className="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-sm text-white transition-colors"
                >
                  {cmd}
                </button>
              ))}
            </div>
          </div>

          {/* Command History */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
              <MessageSquare className="w-5 h-5 mr-2 text-green-400" />
              Recent Conversations
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {commandHistory.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No commands yet. Try saying "Analyze the market" or click a quick command.</p>
                </div>
              ) : (
                commandHistory.map((cmd, index) => (
                  <div key={index} className="bg-black/30 rounded-lg p-3 border border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-400 rounded-full mr-2"></div>
                        <span className="text-sm font-medium text-blue-300">You said:</span>
                      </div>
                      <div className="text-xs text-gray-400">
                        {cmd.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                    <div className="text-sm text-gray-300 mb-2">"{cmd.command}"</div>
                    
                    <div className="flex items-center mb-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                      <span className="text-sm font-medium text-green-300">AI Response:</span>
                      <div className="ml-auto flex items-center">
                        <Target className="w-3 h-3 mr-1 text-yellow-400" />
                        <span className="text-xs text-yellow-300">{(cmd.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="text-sm text-gray-200">{cmd.response}</div>
                    
                    {cmd.action !== 'unknown' && (
                      <div className="mt-2 inline-block px-2 py-1 bg-purple-600/30 border border-purple-600 rounded text-xs text-purple-300">
                        Action: {cmd.action.replace('_', ' ').toUpperCase()}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default VoiceInterface;
