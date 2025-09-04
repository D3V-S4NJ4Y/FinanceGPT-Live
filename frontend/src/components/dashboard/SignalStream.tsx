import React from 'react';

interface SignalStreamProps {
  className?: string;
}

export const SignalStream: React.FC<SignalStreamProps> = ({ className }) => {
  const signals = [
    { type: 'BUY', symbol: 'AAPL', confidence: 0.85, timestamp: '2 min ago' },
    { type: 'HOLD', symbol: 'MSFT', confidence: 0.72, timestamp: '5 min ago' },
    { type: 'SELL', symbol: 'TSLA', confidence: 0.91, timestamp: '8 min ago' },
  ];

  return (
    <div className={`signal-stream ${className || ''}`}>
      <div className="bg-gray-800 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-3">AI Trading Signals</h3>
        <div className="space-y-3">
          {signals.map((signal, index) => (
            <div key={index} className="flex items-center justify-between p-2 bg-gray-700 rounded">
              <div className="flex items-center space-x-3">
                <div className={`px-2 py-1 rounded text-xs font-bold ${
                  signal.type === 'BUY' ? 'bg-green-600 text-white' :
                  signal.type === 'SELL' ? 'bg-red-600 text-white' :
                  'bg-yellow-600 text-white'
                }`}>
                  {signal.type}
                </div>
                <div className="font-semibold text-white">{signal.symbol}</div>
              </div>
              <div className="text-right">
                <div className="text-sm font-semibold text-white">
                  {(signal.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-400">{signal.timestamp}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-gray-400 text-center">
          Powered by AI Market Intelligence
        </div>
      </div>
    </div>
  );
};
