import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import { BarChart3, TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  marketCap?: number;
  sector?: string;
}

interface Market3DVisualizationProps {
  marketData: MarketData[];
  selectedSymbol?: string;
  onSymbolSelect?: (symbol: string) => void;
}

// 3D Market Cube Component
function MarketCube({ 
  data, 
  position, 
  isSelected, 
  onClick 
}: { 
  data: MarketData; 
  position: [number, number, number]; 
  isSelected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Animate the cube
  useFrame((state) => {
    if (meshRef.current) {
      // Subtle rotation based on performance
      meshRef.current.rotation.y += (data.change / 100) * 0.01;
      
      // Pulsing effect for selected cube
      if (isSelected) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
        meshRef.current.scale.setScalar(scale);
      }
      
      // Hover effect
      if (hovered) {
        meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 4) * 0.2;
      } else {
        meshRef.current.position.y = position[1];
      }
    }
  });

  // Color based on performance
  const getColor = () => {
    if (data.change > 5) return '#00ff00';
    if (data.change > 2) return '#7fff00';
    if (data.change > 0) return '#00ff7f';
    if (data.change > -2) return '#ffff00';
    if (data.change > -5) return '#ff7f00';
    return '#ff0000';
  };

  // Size based on market cap or volume
  const getSize = () => {
    const baseSize = 0.5;
    const multiplier = Math.max(0.3, Math.min(2.0, (data.marketCap || data.volume) / 1000000));
    return baseSize * multiplier;
  };

  return (
    <group>
      <mesh
        ref={meshRef}
        position={position}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        scale={getSize()}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial 
          color={getColor()} 
          transparent 
          opacity={isSelected ? 1.0 : 0.8}
          emissive={getColor()}
          emissiveIntensity={isSelected ? 0.3 : 0.1}
        />
      </mesh>
      
      {/* Symbol label */}
      <Text
        position={[position[0], position[1] + 0.8, position[2]]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
        font="/fonts/inter-bold.woff"
      >
        {data.symbol}
      </Text>
      
      {/* Price label */}
      <Text
        position={[position[0], position[1] - 0.8, position[2]]}
        fontSize={0.2}
        color={getColor()}
        anchorX="center"
        anchorY="middle"
        font="/fonts/inter-regular.woff"
      >
        ${data.price.toFixed(2)}
      </Text>
      
      {/* Hover info panel */}
      {hovered && (
        <Html position={[position[0] + 1, position[1], position[2]]} center>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white shadow-lg">
            <div className="text-lg font-bold text-blue-400">{data.symbol}</div>
            <div className="text-sm space-y-1">
              <div>Price: ${data.price.toFixed(2)}</div>
              <div className={`flex items-center ${data.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {data.change >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                <span className="ml-1">{data.change.toFixed(2)}%</span>
              </div>
              <div>Volume: {data.volume.toLocaleString()}</div>
              {data.sector && <div>Sector: {data.sector}</div>}
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

// Sector clustering visualization
function SectorClusters({ marketData }: { marketData: MarketData[] }) {
  const sectorData = marketData.reduce((acc, stock) => {
    const sector = stock.sector || 'Other';
    if (!acc[sector]) {
      acc[sector] = { stocks: [], totalValue: 0, avgChange: 0 };
    }
    acc[sector].stocks.push(stock);
    acc[sector].totalValue += stock.marketCap || stock.volume;
    return acc;
  }, {} as Record<string, { stocks: MarketData[]; totalValue: number; avgChange: number }>);

  // Calculate average change per sector
  Object.keys(sectorData).forEach(sector => {
    const stocks = sectorData[sector].stocks;
    sectorData[sector].avgChange = stocks.reduce((sum, stock) => sum + stock.change, 0) / stocks.length;
  });

  return (
    <>
      {Object.entries(sectorData).map(([sector, data], sectorIndex) => (
        <group key={sector} position={[sectorIndex * 4 - 8, 0, 0]}>
          {/* Sector sphere */}
          <mesh position={[0, 3, 0]}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial 
              color={data.avgChange >= 0 ? '#00ff00' : '#ff0000'}
              transparent
              opacity={0.3}
            />
          </mesh>
          
          {/* Sector label */}
          <Text
            position={[0, 4.5, 0]}
            fontSize={0.4}
            color="white"
            anchorX="center"
            anchorY="middle"
            font="/fonts/inter-bold.woff"
          >
            {sector}
          </Text>
          
          {/* Stocks in sector */}
          {data.stocks.map((stock, stockIndex) => {
            const angle = (stockIndex / data.stocks.length) * Math.PI * 2;
            const radius = 2;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            
            return (
              <mesh key={stock.symbol} position={[x, 1, z]} scale={0.5}>
                <boxGeometry args={[1, 1, 1]} />
                <meshStandardMaterial 
                  color={stock.change >= 0 ? '#00ff7f' : '#ff7f00'}
                  transparent
                  opacity={0.7}
                />
              </mesh>
            );
          })}
        </group>
      ))}
    </>
  );
}

// Market correlation network
function CorrelationNetwork({ marketData }: { marketData: MarketData[] }) {
  const connections = useRef<THREE.Group>(null);

  useFrame(() => {
    if (connections.current) {
      connections.current.rotation.y += 0.005;
    }
  });

  // Generate correlation lines (simplified - in real app would use actual correlation data)
  const generateCorrelations = () => {
    const correlations = [];
    for (let i = 0; i < marketData.length - 1; i++) {
      for (let j = i + 1; j < marketData.length; j++) {
        const stock1 = marketData[i];
        const stock2 = marketData[j];
        
        // Simplified correlation based on price movement similarity
        const correlation = Math.abs(stock1.change - stock2.change) < 2 ? 0.8 : 0.2;
        
        if (correlation > 0.5) {
          correlations.push({
            from: i,
            to: j,
            strength: correlation,
            stock1,
            stock2
          });
        }
      }
    }
    return correlations;
  };

  const correlations = generateCorrelations();

  return (
    <group ref={connections}>
      {correlations.map((corr, index) => {
        const fromPos = [
          (corr.from % 5) * 2 - 4,
          Math.floor(corr.from / 5) * 2 - 2,
          0
        ];
        const toPos = [
          (corr.to % 5) * 2 - 4,
          Math.floor(corr.to / 5) * 2 - 2,
          0
        ];

        return (
          <line key={index}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([...fromPos, ...toPos])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial 
              color={corr.strength > 0.7 ? '#00ff00' : '#ffff00'} 
              transparent
              opacity={corr.strength}
            />
          </line>
        );
      })}
    </group>
  );
}

// Animated background particles
function MarketParticles() {
  const particlesRef = useRef<THREE.Points>(null);
  const particleCount = 1000;
  
  // Create particle positions based on market data pattern
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  
  for (let i = 0; i < particleCount; i++) {
    // Use deterministic positioning based on index and golden ratio
    const phi = (1 + Math.sqrt(5)) / 2; // Golden ratio
    const theta = 2 * Math.PI * i / phi;
    const y = 1 - (i / (particleCount - 1)) * 2; // y goes from 1 to -1
    const radius = Math.sqrt(1 - y * y);
    
    positions[i * 3] = Math.cos(theta) * radius * 25;
    positions[i * 3 + 1] = y * 25;
    positions[i * 3 + 2] = Math.sin(theta) * radius * 25;
    
    // Color based on position and market performance
    const intensity = (i / particleCount);
    colors[i * 3] = 0.2 + intensity * 0.6; // Red channel
    colors[i * 3 + 1] = 0.4 + (1 - intensity) * 0.4; // Green channel  
    colors[i * 3 + 2] = 0.8 - intensity * 0.3; // Blue channel
  }

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.001;
      particlesRef.current.rotation.x += 0.0005;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particleCount}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.05} vertexColors transparent opacity={0.6} />
    </points>
  );
}

// Main 3D Scene
function Market3DScene({ marketData, selectedSymbol, onSymbolSelect }: Market3DVisualizationProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'sectors' | 'network'>('grid');

  return (
    <div className="w-full h-full relative bg-black">
      {/* Control Panel */}
      <div className="absolute top-4 left-4 z-10 bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h3 className="text-white font-bold mb-2">3D Market View</h3>
        <div className="space-y-2">
          <button
            onClick={() => setViewMode('grid')}
            className={`px-3 py-1 rounded text-xs ${
              viewMode === 'grid' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            Grid View
          </button>
          <button
            onClick={() => setViewMode('sectors')}
            className={`px-3 py-1 rounded text-xs ${
              viewMode === 'sectors' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            Sectors
          </button>
          <button
            onClick={() => setViewMode('network')}
            className={`px-3 py-1 rounded text-xs ${
              viewMode === 'network' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            Network
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute top-4 right-4 z-10 bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h4 className="text-white font-bold mb-2">Legend</h4>
        <div className="space-y-1 text-xs text-gray-300">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 mr-2"></div>
            <span>Strong Gain (&gt;5%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-300 mr-2"></div>
            <span>Moderate Gain (2-5%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-500 mr-2"></div>
            <span>Flat (-2% to 2%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 mr-2"></div>
            <span>Loss (&lt;-2%)</span>
          </div>
        </div>
      </div>

      <Canvas
        camera={{ 
          position: [10, 10, 10], 
          fov: 75,
          near: 0.1,
          far: 1000
        }}
        style={{ width: '100%', height: '100%' }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00ffff" />
        <spotLight position={[0, 20, 0]} angle={0.3} penumbra={0.1} intensity={1} />

        {/* Background particles */}
        <MarketParticles />

        {/* Market visualization based on view mode */}
        {viewMode === 'grid' && (
          <>
            {marketData.slice(0, 25).map((stock, index) => (
              <MarketCube
                key={stock.symbol}
                data={stock}
                position={[
                  (index % 5) * 2 - 4,
                  Math.floor(index / 5) * 2 - 2,
                  0
                ]}
                isSelected={selectedSymbol === stock.symbol}
                onClick={() => onSymbolSelect?.(stock.symbol)}
              />
            ))}
          </>
        )}

        {viewMode === 'sectors' && (
          <SectorClusters marketData={marketData} />
        )}

        {viewMode === 'network' && (
          <>
            {marketData.slice(0, 20).map((stock, index) => (
              <MarketCube
                key={stock.symbol}
                data={stock}
                position={[
                  (index % 5) * 2 - 4,
                  Math.floor(index / 5) * 2 - 2,
                  0
                ]}
                isSelected={selectedSymbol === stock.symbol}
                onClick={() => onSymbolSelect?.(stock.symbol)}
              />
            ))}
            <CorrelationNetwork marketData={marketData.slice(0, 20)} />
          </>
        )}

        {/* Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          zoomSpeed={0.6}
          panSpeed={0.8}
          rotateSpeed={0.4}
        />
      </Canvas>

      {/* Performance Stats */}
      <div className="absolute bottom-4 left-4 z-10 bg-gray-900 border border-gray-700 rounded-lg p-4">
        <div className="text-white text-sm space-y-1">
          <div>Stocks: {marketData.length}</div>
          <div className="flex items-center text-green-400">
            <TrendingUp size={16} className="mr-1" />
            Gainers: {marketData.filter(s => s.change > 0).length}
          </div>
          <div className="flex items-center text-red-400">
            <TrendingDown size={16} className="mr-1" />
            Losers: {marketData.filter(s => s.change < 0).length}
          </div>
        </div>
      </div>
    </div>
  );
}

// Export the main component
export default function Market3DVisualization({ marketData, selectedSymbol, onSymbolSelect }: Market3DVisualizationProps) {
  // Add default sample data if none provided
  const defaultData: MarketData[] = [
    { symbol: 'AAPL', price: 175.43, change: 2.34, volume: 12500000, marketCap: 2800000000, sector: 'Technology' },
    { symbol: 'GOOGL', price: 2456.78, change: -1.23, volume: 8900000, marketCap: 1600000000, sector: 'Technology' },
    { symbol: 'MSFT', price: 334.56, change: 3.45, volume: 15600000, marketCap: 2500000000, sector: 'Technology' },
    { symbol: 'AMZN', price: 3187.44, change: -0.87, volume: 4500000, marketCap: 1300000000, sector: 'Consumer Discretionary' },
    { symbol: 'TSLA', price: 789.23, change: 5.67, volume: 25000000, marketCap: 800000000, sector: 'Automotive' },
    { symbol: 'NVDA', price: 456.78, change: 8.90, volume: 18000000, marketCap: 1100000000, sector: 'Technology' },
    { symbol: 'JPM', price: 145.67, change: -2.34, volume: 6700000, marketCap: 400000000, sector: 'Financials' },
    { symbol: 'V', price: 234.56, change: 1.23, volume: 3400000, marketCap: 450000000, sector: 'Financials' },
    { symbol: 'JNJ', price: 167.89, change: -0.56, volume: 5600000, marketCap: 440000000, sector: 'Healthcare' },
    { symbol: 'WMT', price: 142.34, change: 0.78, volume: 4500000, marketCap: 380000000, sector: 'Consumer Staples' },
  ];

  const dataToUse = marketData.length > 0 ? marketData : defaultData;

  return (
    <div className="w-full h-full">
      <Market3DScene 
        marketData={dataToUse}
        selectedSymbol={selectedSymbol}
        onSymbolSelect={onSymbolSelect}
      />
    </div>
  );
}
