import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import { 
  BarChart3, TrendingUp, TrendingDown, Activity, Maximize2, 
  Settings, RefreshCw, Eye, Filter, Layers, Box, Circle, 
  Network, Grid3X3, PieChart, Target, Zap 
} from 'lucide-react';

interface RealMarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high24h: number;
  low24h: number;
  sector: string;
  timestamp: string;
}

interface Enhanced3DVisualizationProps {
  onSymbolSelect?: (symbol: string) => void;
}

import { useSharedData } from '../../hooks/useSharedData';

// Real-time market data hook using shared data system
const useRealMarketData = () => {
  const sharedData = useSharedData();
  const [marketData, setMarketData] = useState<RealMarketData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchMarketData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('🔄 3D Market: Fetching data...');
      // Force fetch from shared data system
      const data = await sharedData.getMarketData();
      console.log('🔄 3D Market: Received data:', data?.length || 0, 'items');
      
      if (data && data.length > 0) {
        const processedData = data.map((stock: any) => ({
          symbol: stock.symbol || 'N/A',
          price: parseFloat(stock.price) || 0,
          change: parseFloat(stock.change) || 0,
          changePercent: parseFloat(stock.changePercent) || 0,
          volume: parseInt(stock.volume) || 0,
          marketCap: stock.marketCap || (parseFloat(stock.price) * parseInt(stock.volume) * 100),
          high24h: stock.high24h || (parseFloat(stock.price) * 1.05),
          low24h: stock.low24h || (parseFloat(stock.price) * 0.95),
          sector: stock.sector || 'Technology',
          timestamp: new Date().toISOString()
        }));

        setMarketData(processedData);
        setLastUpdate(new Date());
        setError(null);
        console.log('✅ 3D Market loaded', processedData.length, 'stocks');
      } else {
        console.log('⚠️ No market data available for 3D visualization');
        // Try direct API call as fallback
        try {
          const response = await fetch('http://localhost:8001/api/market/latest');
          if (response.ok) {
            const directData = await response.json();
            if (directData && directData.length > 0) {
              const processedData = directData.map((stock: any) => ({
                symbol: stock.symbol || 'N/A',
                price: parseFloat(stock.price) || 0,
                change: parseFloat(stock.change) || 0,
                changePercent: parseFloat(stock.changePercent) || 0,
                volume: parseInt(stock.volume) || 0,
                marketCap: stock.marketCap || (parseFloat(stock.price) * parseInt(stock.volume) * 100),
                high24h: stock.high24h || (parseFloat(stock.price) * 1.05),
                low24h: stock.low24h || (parseFloat(stock.price) * 0.95),
                sector: stock.sector || 'Technology',
                timestamp: new Date().toISOString()
              }));
              setMarketData(processedData);
              setLastUpdate(new Date());
              setError(null);
              console.log('✅ 3D Market loaded via direct API:', processedData.length, 'stocks');
            }
          }
        } catch (directErr) {
          console.error('Direct API call failed:', directErr);
        }
      }
    } catch (err) {
      console.error('3D Market data error:', err);
      setError('Failed to load market data');
    } finally {
      setIsLoading(false);
      console.log('🔄 3D Market: Fetch complete, isLoading set to false');
    }
  }, [sharedData]);

  useEffect(() => {
    console.log('🔄 3D Market: Component mounted, starting data fetch');
    // Don't set loading on mount if we already have shared data
    if (sharedData.marketData.length === 0) {
      fetchMarketData();
    }
    const interval = setInterval(fetchMarketData, 30000);
    return () => clearInterval(interval);
  }, [fetchMarketData, sharedData.marketData.length]);

  // Update when shared data changes
  useEffect(() => {
    if (sharedData.marketData && sharedData.marketData.length > 0) {
      const processedData = sharedData.marketData.map((stock: any) => ({
        symbol: stock.symbol || 'N/A',
        price: parseFloat(stock.price) || 0,
        change: parseFloat(stock.change) || 0,
        changePercent: parseFloat(stock.changePercent) || 0,
        volume: parseInt(stock.volume) || 0,
        marketCap: stock.marketCap || (parseFloat(stock.price) * parseInt(stock.volume) * 100),
        high24h: stock.high24h || (parseFloat(stock.price) * 1.05),
        low24h: stock.low24h || (parseFloat(stock.price) * 0.95),
        sector: stock.sector || 'Technology',
        timestamp: new Date().toISOString()
      }));
      setMarketData(processedData);
      setLastUpdate(new Date());
      setError(null);
      setIsLoading(false);
      console.log('✅ 3D Market: Data loaded, isLoading set to false');
    }
  }, [sharedData.marketData]);

  return { marketData, isLoading, error, lastUpdate, refresh: fetchMarketData };
};

// Enhanced 3D Stock Cube
function Enhanced3DStockCube({ 
  data, 
  position, 
  isSelected, 
  onClick,
  viewMode 
}: { 
  data: RealMarketData; 
  position: [number, number, number]; 
  isSelected: boolean;
  onClick: () => void;
  viewMode: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      // Real-time rotation based on actual volatility
      const volatility = Math.abs(data.changePercent);
      meshRef.current.rotation.y += (volatility / 100) * 0.02;
      
      // Pulsing for selected
      if (isSelected) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
        meshRef.current.scale.setScalar(scale);
      } else {
        meshRef.current.scale.setScalar(1);
      }
      
      // Hover animation
      if (hovered) {
        meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 5) * 0.3;
      } else {
        meshRef.current.position.y = position[1];
      }
    }
  });

  // Color based on real performance
  const getPerformanceColor = () => {
    const change = data.changePercent;
    if (change > 5) return '#00ff00';
    if (change > 2) return '#7fff00';
    if (change > 0) return '#00ff7f';
    if (change > -2) return '#ffff00';
    if (change > -5) return '#ff7f00';
    return '#ff0000';
  };

  // Size based on real market cap and volume
  const getSize = () => {
    const baseSize = 0.6;
    const capMultiplier = Math.log(data.marketCap / 1000000) / 10;
    const volMultiplier = Math.log(data.volume / 1000000) / 15;
    return Math.max(0.3, Math.min(1.5, baseSize + capMultiplier + volMultiplier));
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
        {viewMode === 'cubes' && <boxGeometry args={[1, 1, 1]} />}
        {viewMode === 'spheres' && <sphereGeometry args={[0.7, 16, 16]} />}
        {viewMode === 'cylinders' && <cylinderGeometry args={[0.5, 0.5, 1, 8]} />}
        
        <meshStandardMaterial 
          color={getPerformanceColor()} 
          transparent 
          opacity={isSelected ? 1.0 : 0.85}
          emissive={getPerformanceColor()}
          emissiveIntensity={isSelected ? 0.4 : 0.15}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
      
      {/* Symbol label */}
      <Text
        position={[position[0], position[1] + 1, position[2]]}
        fontSize={0.25}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {data.symbol}
      </Text>
      
      {/* Price label */}
      <Text
        position={[position[0], position[1] - 1, position[2]]}
        fontSize={0.18}
        color={getPerformanceColor()}
        anchorX="center"
        anchorY="middle"
      >
        ${data.price.toFixed(2)}
      </Text>
      
      {/* Enhanced hover info */}
      {hovered && (
        <Html position={[position[0] + 1.5, position[1], position[2]]} center>
          <div className="bg-gray-900/95 border border-gray-600 rounded-lg p-4 text-white shadow-xl backdrop-blur-sm">
            <div className="text-lg font-bold text-blue-400 mb-2">{data.symbol}</div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span>Price:</span>
                <span className="font-semibold">${data.price.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Change:</span>
                <span className={`font-semibold ${data.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {data.changePercent >= 0 ? '+' : ''}{data.changePercent.toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Volume:</span>
                <span>{(data.volume / 1000000).toFixed(1)}M</span>
              </div>
              <div className="flex justify-between">
                <span>Market Cap:</span>
                <span>${(data.marketCap / 1000000000).toFixed(1)}B</span>
              </div>
              <div className="flex justify-between">
                <span>High:</span>
                <span className="text-green-400">${data.high24h.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Low:</span>
                <span className="text-red-400">${data.low24h.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Sector:</span>
                <span className="text-blue-300">{data.sector}</span>
              </div>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

// Real-time sector clustering
function RealSectorClusters({ marketData }: { marketData: RealMarketData[] }) {
  const sectorGroups = useMemo(() => {
    const groups = marketData.reduce((acc, stock) => {
      const sector = stock.sector || 'Other';
      if (!acc[sector]) {
        acc[sector] = { stocks: [], totalCap: 0, avgChange: 0 };
      }
      acc[sector].stocks.push(stock);
      acc[sector].totalCap += stock.marketCap;
      return acc;
    }, {} as Record<string, { stocks: RealMarketData[]; totalCap: number; avgChange: number }>);

    // Calculate average change per sector
    Object.keys(groups).forEach(sector => {
      const stocks = groups[sector].stocks;
      groups[sector].avgChange = stocks.reduce((sum, stock) => sum + stock.changePercent, 0) / stocks.length;
    });

    return groups;
  }, [marketData]);

  return (
    <>
      {Object.entries(sectorGroups).map(([sector, data], sectorIndex) => (
        <group key={sector} position={[sectorIndex * 5 - 10, 0, 0]}>
          {/* Sector sphere */}
          <mesh position={[0, 4, 0]}>
            <sphereGeometry args={[1.2, 20, 20]} />
            <meshStandardMaterial 
              color={data.avgChange >= 0 ? '#00ff88' : '#ff4444'}
              transparent
              opacity={0.4}
              emissive={data.avgChange >= 0 ? '#00ff88' : '#ff4444'}
              emissiveIntensity={0.2}
            />
          </mesh>
          
          {/* Sector label */}
          <Text
            position={[0, 5.5, 0]}
            fontSize={0.4}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {sector}
          </Text>
          
          <Text
            position={[0, 2.5, 0]}
            fontSize={0.25}
            color={data.avgChange >= 0 ? '#00ff88' : '#ff4444'}
            anchorX="center"
            anchorY="middle"
          >
            {data.avgChange >= 0 ? '+' : ''}{data.avgChange.toFixed(1)}%
          </Text>
          
          {/* Stocks in sector */}
          {data.stocks.map((stock, stockIndex) => {
            const angle = (stockIndex / data.stocks.length) * Math.PI * 2;
            const radius = 2.5;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            
            return (
              <mesh key={stock.symbol} position={[x, 1, z]} scale={0.4}>
                <boxGeometry args={[1, 1, 1]} />
                <meshStandardMaterial 
                  color={stock.changePercent >= 0 ? '#00ff7f' : '#ff7f00'}
                  transparent
                  opacity={0.8}
                />
              </mesh>
            );
          })}
        </group>
      ))}
    </>
  );
}

// Real correlation network
function RealCorrelationNetwork({ marketData }: { marketData: RealMarketData[] }) {
  const connectionsRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (connectionsRef.current) {
      connectionsRef.current.rotation.y += 0.003;
    }
  });

  const correlations = useMemo(() => {
    const corrs = [];
    for (let i = 0; i < marketData.length - 1; i++) {
      for (let j = i + 1; j < marketData.length; j++) {
        const stock1 = marketData[i];
        const stock2 = marketData[j];
        
        // Real correlation based on price movement and sector
        const priceCorr = 1 - Math.abs(stock1.changePercent - stock2.changePercent) / 10;
        const sectorCorr = stock1.sector === stock2.sector ? 0.3 : 0;
        const correlation = Math.max(0, priceCorr + sectorCorr);
        
        if (correlation > 0.6) {
          corrs.push({
            from: i,
            to: j,
            strength: correlation,
            stock1,
            stock2
          });
        }
      }
    }
    return corrs;
  }, [marketData]);

  return (
    <group ref={connectionsRef}>
      {correlations.map((corr, index) => {
        const fromPos = [
          (corr.from % 6) * 2.5 - 6,
          Math.floor(corr.from / 6) * 2.5 - 3,
          0
        ];
        const toPos = [
          (corr.to % 6) * 2.5 - 6,
          Math.floor(corr.to / 6) * 2.5 - 3,
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
              color={corr.strength > 0.8 ? '#00ff00' : '#ffaa00'} 
              transparent
              opacity={corr.strength * 0.8}
            />
          </line>
        );
      })}
    </group>
  );
}

// Enhanced particle system
function EnhancedParticleSystem({ marketData }: { marketData: RealMarketData[] }) {
  const particlesRef = useRef<THREE.Points>(null);
  const particleCount = 2000;
  
  const { positions, colors, velocities } = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const col = new Float32Array(particleCount * 3);
    const vel = new Float32Array(particleCount * 3);
    
    // Market sentiment affects particle behavior
    const avgChange = marketData.length > 0 
      ? marketData.reduce((sum, stock) => sum + stock.changePercent, 0) / marketData.length 
      : 0;
    
    for (let i = 0; i < particleCount; i++) {
      // Position
      pos[i * 3] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 50;
      
      // Color based on market sentiment
      if (avgChange > 0) {
        col[i * 3] = 0.2 + Math.random() * 0.3; // Red
        col[i * 3 + 1] = 0.6 + Math.random() * 0.4; // Green
        col[i * 3 + 2] = 0.3 + Math.random() * 0.3; // Blue
      } else {
        col[i * 3] = 0.6 + Math.random() * 0.4; // Red
        col[i * 3 + 1] = 0.2 + Math.random() * 0.3; // Green
        col[i * 3 + 2] = 0.3 + Math.random() * 0.3; // Blue
      }
      
      // Velocity
      vel[i * 3] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
    }
    
    return { positions: pos, colors: col, velocities: vel };
  }, [marketData]);

  useFrame(() => {
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
      <pointsMaterial size={0.08} vertexColors transparent opacity={0.7} />
    </points>
  );
}

// Main 3D Scene
function Enhanced3DScene() {
  const { marketData, isLoading, error, lastUpdate, refresh } = useRealMarketData();
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [viewMode, setViewMode] = useState<'grid' | 'sectors' | 'network' | 'heatmap'>('grid');
  const [shapeMode, setShapeMode] = useState<'cubes' | 'spheres' | 'cylinders'>('cubes');
  const [showParticles, setShowParticles] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);

  const filteredData = useMemo(() => {
    return marketData.slice(0, 25); // Limit for performance
  }, [marketData]);

  return (
    <div className="w-full h-full relative bg-black">
      {/* Enhanced Control Panel */}
      <div className="absolute top-2 sm:top-4 left-2 sm:left-4 z-10 bg-gray-900/90 border border-gray-600 rounded-lg p-2 sm:p-4 backdrop-blur-sm max-w-[200px] sm:max-w-none">
        <h3 className="text-white font-bold mb-2 sm:mb-3 flex items-center text-sm sm:text-base">
          <Box className="w-4 h-4 sm:w-5 sm:h-5 mr-1 sm:mr-2 text-blue-400" />
          <span className="hidden sm:inline">3D Market Control</span>
          <span className="sm:hidden">Control</span>
        </h3>
        
        <div className="space-y-2 sm:space-y-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">View Mode</label>
            <div className="grid grid-cols-2 gap-1">
              {(['grid', 'sectors', 'network', 'heatmap'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-1 sm:px-2 py-1 rounded text-xs transition-colors ${
                    viewMode === mode ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <span className="hidden sm:inline">{mode.charAt(0).toUpperCase() + mode.slice(1)}</span>
                  <span className="sm:hidden">{mode.charAt(0).toUpperCase()}</span>
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="text-xs text-gray-400 block mb-1">Shape</label>
            <div className="grid grid-cols-3 gap-1">
              {(['cubes', 'spheres', 'cylinders'] as const).map((shape) => (
                <button
                  key={shape}
                  onClick={() => setShapeMode(shape)}
                  className={`px-1 sm:px-2 py-1 rounded text-xs transition-colors ${
                    shapeMode === shape ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {shape === 'cubes' && <Box className="w-3 h-3" />}
                  {shape === 'spheres' && <Circle className="w-3 h-3" />}
                  {shape === 'cylinders' && <BarChart3 className="w-3 h-3" />}
                </button>
              ))}
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400 hidden sm:inline">Particles</span>
            <span className="text-xs text-gray-400 sm:hidden">FX</span>
            <button
              onClick={() => setShowParticles(!showParticles)}
              className={`px-1 sm:px-2 py-1 rounded text-xs transition-colors ${
                showParticles ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'
              }`}
            >
              {showParticles ? 'ON' : 'OFF'}
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">Auto Rotate</span>
            <button
              onClick={() => setAutoRotate(!autoRotate)}
              className={`px-2 py-1 rounded text-xs transition-colors ${
                autoRotate ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'
              }`}
            >
              {autoRotate ? 'ON' : 'OFF'}
            </button>
          </div>
          
          <button
            onClick={refresh}
            disabled={isLoading}
            className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-xs text-white transition-colors flex items-center justify-center"
          >
            <RefreshCw className={`w-3 h-3 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh Data
          </button>
        </div>
      </div>

      {/* Status Panel */}
      <div className="absolute top-4 right-4 z-10 bg-gray-900/90 border border-gray-600 rounded-lg p-4 backdrop-blur-sm">
        <h4 className="text-white font-bold mb-2 flex items-center">
          <Activity className={`w-4 h-4 mr-2 ${marketData.length > 0 ? 'text-green-400' : 'text-red-400'}`} />
          Market Status
        </h4>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-400">Stocks:</span>
            <span className={`${marketData.length > 0 ? 'text-white' : 'text-red-400'}`}>{marketData.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Selected:</span>
            <span className="text-blue-400">{selectedSymbol || 'None'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Updated:</span>
            <span className="text-white">{lastUpdate.toLocaleTimeString()}</span>
          </div>
          {isLoading && (
            <div className="text-blue-400 text-xs mt-2 flex items-center">
              <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
              Loading market data...
            </div>
          )}
          {error && (
            <div className="text-red-400 text-xs mt-2">
              {error}
            </div>
          )}
          {marketData.length === 0 && !isLoading && (
            <div className="text-yellow-400 text-xs mt-2">
              No market data available
            </div>
          )}
        </div>
      </div>

      {/* Performance Stats */}
      <div className="absolute bottom-4 left-4 z-10 bg-gray-900/90 border border-gray-600 rounded-lg p-4 backdrop-blur-sm">
        <div className="text-white text-sm space-y-2">
          <div className="flex items-center text-green-400">
            <TrendingUp size={16} className="mr-2" />
            Gainers: {marketData.filter(s => s.changePercent > 0).length}
          </div>
          <div className="flex items-center text-red-400">
            <TrendingDown size={16} className="mr-2" />
            Losers: {marketData.filter(s => s.changePercent < 0).length}
          </div>
          <div className="flex items-center text-blue-400">
            <Target size={16} className="mr-2" />
            Avg Change: {marketData.length > 0 ? (marketData.reduce((sum, s) => sum + s.changePercent, 0) / marketData.length).toFixed(2) : '0.00'}%
          </div>
          {marketData.length === 0 && (
            <div className="text-yellow-400 text-xs mt-2">
              Start backend server to load market data
            </div>
          )}
        </div>
      </div>

      <Canvas
        camera={{ 
          position: [15, 15, 15], 
          fov: 60,
          near: 0.1,
          far: 1000
        }}
        style={{ width: '100%', height: '100%' }}
      >
        {/* Enhanced Lighting */}
        <ambientLight intensity={0.4} />
        <pointLight position={[15, 15, 15]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-15, -15, -15]} intensity={0.8} color="#4444ff" />
        <spotLight position={[0, 25, 0]} angle={0.4} penumbra={0.2} intensity={1.5} />
        <directionalLight position={[10, 10, 5]} intensity={0.5} />

        {/* Background particles */}
        {showParticles && <EnhancedParticleSystem marketData={marketData.length > 0 ? marketData : []} />}

        {/* Market visualization based on view mode */}
        {viewMode === 'grid' && (
          <>
            {filteredData.length > 0 ? (
              filteredData.map((stock, index) => (
                <Enhanced3DStockCube
                  key={stock.symbol}
                  data={stock}
                  position={[
                    (index % 6) * 3 - 7.5,
                    Math.floor(index / 6) * 3 - 6,
                    0
                  ]}
                  isSelected={selectedSymbol === stock.symbol}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                  viewMode={shapeMode}
                />
              ))
            ) : (
              <Text
                position={[0, 0, 0]}
                fontSize={1}
                color="white"
                anchorX="center"
                anchorY="middle"
              >
                {isLoading ? 'Loading Market Data...' : 'No Market Data Available'}
              </Text>
            )}
          </>
        )}

        {viewMode === 'sectors' && (
          filteredData.length > 0 ? (
            <RealSectorClusters marketData={filteredData} />
          ) : (
            <Text
              position={[0, 0, 0]}
              fontSize={1}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {isLoading ? 'Loading Sector Data...' : 'No Sector Data Available'}
            </Text>
          )
        )}

        {viewMode === 'network' && (
          filteredData.length > 0 ? (
            <>
              {filteredData.slice(0, 20).map((stock, index) => (
                <Enhanced3DStockCube
                  key={stock.symbol}
                  data={stock}
                  position={[
                    (index % 5) * 3 - 6,
                    Math.floor(index / 5) * 3 - 6,
                    0
                  ]}
                  isSelected={selectedSymbol === stock.symbol}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                  viewMode={shapeMode}
                />
              ))}
              <RealCorrelationNetwork marketData={filteredData.slice(0, 20)} />
            </>
          ) : (
            <Text
              position={[0, 0, 0]}
              fontSize={1}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {isLoading ? 'Loading Network Data...' : 'No Network Data Available'}
            </Text>
          )
        )}

        {viewMode === 'heatmap' && (
          filteredData.length > 0 ? (
            <>
              {filteredData.map((stock, index) => {
                const row = Math.floor(index / 8);
                const col = index % 8;
                return (
                  <Enhanced3DStockCube
                    key={stock.symbol}
                    data={stock}
                    position={[
                      col * 2 - 7,
                      row * 2 - 4,
                      Math.abs(stock.changePercent) * 0.5
                    ]}
                    isSelected={selectedSymbol === stock.symbol}
                    onClick={() => setSelectedSymbol(stock.symbol)}
                    viewMode={shapeMode}
                  />
                );
              })}
            </>
          ) : (
            <Text
              position={[0, 0, 0]}
              fontSize={1}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {isLoading ? 'Loading Heatmap Data...' : 'No Heatmap Data Available'}
            </Text>
          )
        )}

        {/* Enhanced Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          zoomSpeed={0.8}
          panSpeed={1.0}
          rotateSpeed={0.5}
          autoRotate={autoRotate}
          autoRotateSpeed={1.0}
          maxDistance={50}
          minDistance={5}
        />
      </Canvas>
    </div>
  );
}

export default function Enhanced3DMarketVisualization({ onSymbolSelect }: Enhanced3DVisualizationProps) {
  return (
    <div className="w-full h-full">
      <Enhanced3DScene />
    </div>
  );
}