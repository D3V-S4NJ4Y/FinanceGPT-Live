import React from 'react';

interface RiskRadarProps {
  className?: string;
}

export const RiskRadar: React.FC<RiskRadarProps> = ({ className }) => {
  const riskLevel = 'Medium';
  const riskScore = 0.45;
  
  // Convert to percentage and round to nearest 5% for consistent styling
  const widthPercentage = Math.round(riskScore * 20) * 5; // 45% -> 45%
  const widthClass = `w-[${widthPercentage}%]`;

  return (
    <div className={`risk-radar ${className || ''}`}>
      <div className="bg-gray-800 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-3">Risk Assessment</h3>
        <div className="text-center">
          <div className="text-3xl font-bold text-yellow-400 mb-2">
            {riskLevel}
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mb-3">
            <div
              className={`bg-yellow-400 h-2 rounded-full transition-all duration-300 ${widthClass}`}
            />
          </div>
          <div className="text-sm text-gray-400">
            Risk Score: {(riskScore * 100).toFixed(0)}/100
          </div>
        </div>
        <div className="mt-4 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Market Volatility</span>
            <span className="text-yellow-400">Moderate</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Liquidity Risk</span>
            <span className="text-green-400">Low</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Credit Risk</span>
            <span className="text-yellow-400">Medium</span>
          </div>
        </div>
      </div>
    </div>
  );
};
