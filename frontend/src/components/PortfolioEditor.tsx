import React, { useState } from 'react';
import { X, Plus, Save } from 'lucide-react';
import { portfolioService } from '../services/portfolioService';

interface PortfolioHolding {
  symbol: string;
  shares: number;
  avgCost: number;
}

interface PortfolioEditorProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: () => void;
}

export default function PortfolioEditor({ isOpen, onClose, onSave }: PortfolioEditorProps) {
  const [holdings, setHoldings] = useState<PortfolioHolding[]>(() => {
    const existing = portfolioService.getDefaultPortfolio();
    return existing.length > 0 ? existing : [{ symbol: '', shares: 0, avgCost: 0 }];
  });

  const addHolding = () => {
    setHoldings([...holdings, { symbol: '', shares: 0, avgCost: 0 }]);
  };

  const updateHolding = (index: number, field: keyof PortfolioHolding, value: string | number) => {
    const updated = [...holdings];
    updated[index] = { ...updated[index], [field]: value };
    setHoldings(updated);
  };

  const removeHolding = (index: number) => {
    setHoldings(holdings.filter((_, i) => i !== index));
  };

  const savePortfolio = () => {
    const validHoldings = holdings.filter(h => h.symbol.trim() && h.shares > 0 && h.avgCost > 0);
    if (validHoldings.length === 0) {
      alert('Please add at least one valid holding');
      return;
    }
    portfolioService.savePortfolio(validHoldings);
    onSave();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white">Portfolio Editor</h2>
          <button 
            onClick={onClose} 
            title="Close portfolio editor"
            aria-label="Close portfolio editor"
            className="text-gray-400 hover:text-white"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-4">
          {holdings.map((holding, index) => (
            <div key={index} className="grid grid-cols-4 gap-3 p-3 bg-gray-700 rounded-lg">
              <input
                type="text"
                placeholder="Symbol (e.g., AAPL)"
                value={holding.symbol}
                onChange={(e) => updateHolding(index, 'symbol', e.target.value.toUpperCase())}
                className="bg-gray-600 text-white px-3 py-2 rounded border border-gray-500 focus:border-blue-500"
              />
              <input
                type="number"
                placeholder="Shares"
                value={holding.shares === 0 ? '' : holding.shares}
                onChange={(e) => updateHolding(index, 'shares', parseFloat(e.target.value) || 0)}
                className="bg-gray-600 text-white px-3 py-2 rounded border border-gray-500 focus:border-blue-500"
              />
              <input
                type="number"
                step="0.01"
                placeholder="Avg Cost ($)"
                value={holding.avgCost === 0 ? '' : holding.avgCost}
                onChange={(e) => updateHolding(index, 'avgCost', parseFloat(e.target.value) || 0)}
                className="bg-gray-600 text-white px-3 py-2 rounded border border-gray-500 focus:border-blue-500"
              />
              <button
                onClick={() => removeHolding(index)}
                title="Remove holding"
                aria-label="Remove holding"
                className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded"
              >
                Remove
              </button>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between mt-6">
          <button
            onClick={addHolding}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
          >
            <Plus className="w-4 h-4" />
            <span>Add Holding</span>
          </button>

          <button
            onClick={savePortfolio}
            className="flex items-center space-x-2 px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg"
          >
            <Save className="w-4 h-4" />
            <span>Save Portfolio</span>
          </button>
        </div>
      </div>
    </div>
  );
}