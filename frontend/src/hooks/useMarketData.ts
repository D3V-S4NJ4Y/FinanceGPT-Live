import { useState, useEffect } from 'react';

interface MarketTick {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
}

interface MarketData {
  stocks: MarketTick[];
  indices: MarketTick[];
  crypto: MarketTick[];
  lastUpdated: string;
}

export const useMarketData = () => {
  const [marketData, setMarketData] = useState<MarketData>({
    stocks: [],
    indices: [],
    crypto: [],
    lastUpdated: new Date().toISOString()
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    let fetchController: AbortController | null = null;

    const fetchMarketData = async () => {
      if (!isMounted) return;

      try {
        console.log('📈 Fetching market data...');
        setIsLoading(true);
        setError(null); // Clear previous errors
        
        // Create fresh controller for each request
        fetchController = new AbortController();
        
        const response = await fetch('http://127.0.0.1:8001/api/market/latest', {
          signal: fetchController.signal,
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          }
        });
        
        if (!isMounted) return; // Component unmounted during fetch
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();

        console.log('📊 Raw market API response:', data);

        if (response.ok && Array.isArray(data)) {
          // API returns flat array of market data, categorize it
          const stocks = data.filter(item => 
            !item.symbol.startsWith('^') && 
            !item.symbol.includes('USD') && 
            !item.symbol.includes('BTC') &&
            !['SPY', 'QQQ', 'DIA', 'IWM'].includes(item.symbol)
          );
          
          const indices = data.filter(item => 
            item.symbol.startsWith('^') || ['SPY', 'QQQ', 'DIA', 'IWM'].includes(item.symbol)
          );
          
          const crypto = data.filter(item => 
            item.symbol.includes('USD') || item.symbol.includes('BTC')
          );

          const transformedData = {
            stocks: stocks.map(item => ({
              symbol: item.symbol,
              price: item.price,
              change: item.change,
              change_percent: item.changePercent,
              volume: item.volume
            })),
            indices: indices.map(item => ({
              symbol: item.symbol,
              price: item.price,
              change: item.change,
              change_percent: item.changePercent,
              volume: item.volume
            })),
            crypto: crypto.map(item => ({
              symbol: item.symbol,
              price: item.price,
              change: item.change,
              change_percent: item.changePercent,
              volume: item.volume
            })),
            lastUpdated: new Date().toISOString()
          };
          
          console.log('✅ Processed market data:', transformedData.stocks.length, 'stocks,', transformedData.indices.length, 'indices,', transformedData.crypto.length, 'crypto');
          if (isMounted) {
            setMarketData(transformedData);
          }
          setError(null);
        } else {
          throw new Error('Invalid response format from market data API');
        }
      } catch (error) {
        if (!isMounted) return; // Component unmounted during error handling
        
        console.error('❌ Error fetching market data:', error);
        
        let errorMessage = 'Failed to connect to market data API';
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            console.log('Request was aborted - component unmounting or new request started');
            return; // Don't set error state for intentional aborts
          } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Connection Error: Cannot reach backend server';
          } else {
            errorMessage = error.message;
          }
        }
        
        setError(errorMessage);
        
        // Don't use fake/mock data - just keep empty arrays
        console.log('🚫 Not using fallback mock data - keeping current data');
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
        fetchController = null;
      }
    };

    // Initial fetch
    fetchMarketData();

    // Set up interval with proper cleanup
    const interval = setInterval(() => {
      if (isMounted) {
        fetchMarketData();
      }
    }, 30000); // Increased to 30 seconds to reduce load

    return () => {
      console.log('🧹 useMarketData cleanup - cancelling requests and intervals');
      isMounted = false;
      
      // Cancel any ongoing fetch
      if (fetchController) {
        fetchController.abort();
      }
      
      // Clear interval
      clearInterval(interval);
    };
  }, []);

  return { marketData, isLoading, error };
};
