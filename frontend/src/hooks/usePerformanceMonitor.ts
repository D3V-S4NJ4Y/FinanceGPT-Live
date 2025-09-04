/**
 * Performance Monitor Hook
 * Tracks and optimizes component performance for the Command Center
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';

interface PerformanceMetrics {
  renderTime: number;
  updateFrequency: number;
  memoryUsage: number;
  apiLatency: number;
  wsLatency: number;
  errorRate: number;
}

interface PerformanceConfig {
  enableMetrics: boolean;
  maxRenderTime: number;
  maxUpdateFrequency: number;
  memoryThreshold: number;
}

export const usePerformanceMonitor = (config: PerformanceConfig = {
  enableMetrics: true,
  maxRenderTime: 16, // 60fps target
  maxUpdateFrequency: 10, // max 10 updates per second
  memoryThreshold: 50 // MB
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    updateFrequency: 0,
    memoryUsage: 0,
    apiLatency: 0,
    wsLatency: 0,
    errorRate: 0
  });

  const renderStartTime = useRef<number>(0);
  const updateCount = useRef<number>(0);
  const lastUpdateTime = useRef<number>(Date.now());
  const errorCount = useRef<number>(0);
  const totalRequests = useRef<number>(0);
  const performanceObserver = useRef<PerformanceObserver | null>(null);

  // Track render performance
  const startRenderMeasurement = useCallback(() => {
    if (!config.enableMetrics) return;
    renderStartTime.current = performance.now();
  }, [config.enableMetrics]);

  const endRenderMeasurement = useCallback(() => {
    if (!config.enableMetrics || renderStartTime.current === 0) return;
    
    const renderTime = performance.now() - renderStartTime.current;
    
    setMetrics(prev => ({
      ...prev,
      renderTime: Math.round(renderTime * 100) / 100
    }));

    // Warn if render time exceeds threshold
    if (renderTime > config.maxRenderTime) {
      console.warn(`⚠️ Slow render detected: ${renderTime.toFixed(2)}ms (threshold: ${config.maxRenderTime}ms)`);
    }

    renderStartTime.current = 0;
  }, [config.enableMetrics, config.maxRenderTime]);

  // Track update frequency
  const trackUpdate = useCallback(() => {
    if (!config.enableMetrics) return;

    const now = Date.now();
    updateCount.current++;

    // Calculate frequency every second
    if (now - lastUpdateTime.current >= 1000) {
      const frequency = updateCount.current;
      
      setMetrics(prev => ({
        ...prev,
        updateFrequency: frequency
      }));

      // Warn if update frequency is too high
      if (frequency > config.maxUpdateFrequency) {
        console.warn(`⚠️ High update frequency: ${frequency}/sec (threshold: ${config.maxUpdateFrequency}/sec)`);
      }

      updateCount.current = 0;
      lastUpdateTime.current = now;
    }
  }, [config.enableMetrics, config.maxUpdateFrequency]);

  // Track memory usage
  const trackMemoryUsage = useCallback(() => {
    if (!config.enableMetrics || !(performance as any).memory) return;

    const memory = (performance as any).memory;
    const usedMB = memory.usedJSHeapSize / 1024 / 1024;
    
    setMetrics(prev => ({
      ...prev,
      memoryUsage: Math.round(usedMB * 100) / 100
    }));

    // Warn if memory usage exceeds threshold
    if (usedMB > config.memoryThreshold) {
      console.warn(`⚠️ High memory usage: ${usedMB.toFixed(2)}MB (threshold: ${config.memoryThreshold}MB)`);
    }
  }, [config.enableMetrics, config.memoryThreshold]);

  // Track API latency
  const measureApiLatency = useCallback(async (apiCall: () => Promise<any>): Promise<any> => {
    if (!config.enableMetrics) return apiCall();

    const startTime = performance.now();
    totalRequests.current++;

    try {
      const result = await apiCall();
      const latency = performance.now() - startTime;
      
      setMetrics(prev => ({
        ...prev,
        apiLatency: Math.round(latency * 100) / 100
      }));

      return result;
    } catch (error) {
      errorCount.current++;
      const errorRate = (errorCount.current / totalRequests.current) * 100;
      
      setMetrics(prev => ({
        ...prev,
        errorRate: Math.round(errorRate * 100) / 100
      }));

      throw error;
    }
  }, [config.enableMetrics]);

  // Track WebSocket latency
  const measureWsLatency = useCallback((sendTime: number) => {
    if (!config.enableMetrics) return;

    const latency = Date.now() - sendTime;
    
    setMetrics(prev => ({
      ...prev,
      wsLatency: latency
    }));
  }, [config.enableMetrics]);

  // Performance optimization suggestions
  const getOptimizationSuggestions = useCallback((): string[] => {
    const suggestions: string[] = [];

    if (metrics.renderTime > config.maxRenderTime) {
      suggestions.push('Consider using React.memo() or useMemo() for expensive calculations');
      suggestions.push('Reduce the number of DOM elements or simplify component structure');
    }

    if (metrics.updateFrequency > config.maxUpdateFrequency) {
      suggestions.push('Implement debouncing or throttling for frequent updates');
      suggestions.push('Use useCallback() to prevent unnecessary re-renders');
    }

    if (metrics.memoryUsage > config.memoryThreshold) {
      suggestions.push('Check for memory leaks in event listeners or intervals');
      suggestions.push('Consider implementing virtual scrolling for large lists');
    }

    if (metrics.apiLatency > 1000) {
      suggestions.push('Implement request caching or reduce API call frequency');
      suggestions.push('Consider using WebSocket for real-time data instead of polling');
    }

    if (metrics.errorRate > 5) {
      suggestions.push('Implement better error handling and retry logic');
      suggestions.push('Add circuit breaker pattern for failing services');
    }

    return suggestions;
  }, [metrics, config]);

  // Initialize performance monitoring
  useEffect(() => {
    if (!config.enableMetrics) return;

    // Set up Performance Observer for detailed metrics
    if ('PerformanceObserver' in window) {
      performanceObserver.current = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'measure' && entry.name.includes('React')) {
            console.log(`React performance: ${entry.name} took ${entry.duration.toFixed(2)}ms`);
          }
        });
      });

      try {
        performanceObserver.current.observe({ entryTypes: ['measure', 'navigation'] });
      } catch (error) {
        console.warn('Performance Observer not fully supported:', error);
      }
    }

    // Set up memory monitoring interval
    const memoryInterval = setInterval(trackMemoryUsage, 5000);

    return () => {
      if (performanceObserver.current) {
        performanceObserver.current.disconnect();
      }
      clearInterval(memoryInterval);
    };
  }, [config.enableMetrics, trackMemoryUsage]);

  // Throttle function for high-frequency operations
  const throttle = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    limit: number
  ): T => {
    let inThrottle: boolean;
    return ((...args: any[]) => {
      if (!inThrottle) {
        func.apply(null, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    }) as T;
  }, []);

  // Debounce function for delayed operations
  const debounce = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    delay: number
  ): T => {
    let timeoutId: number;
    return ((...args: any[]) => {
      clearTimeout(timeoutId);
      timeoutId = window.setTimeout(() => func.apply(null, args), delay);
    }) as T;
  }, []);

  // Batch updates to reduce render frequency
  const batchUpdates = useCallback((updates: (() => void)[]): void => {
    // Use React's unstable_batchedUpdates if available
    if ((React as any).unstable_batchedUpdates) {
      (React as any).unstable_batchedUpdates(() => {
        updates.forEach(update => update());
      });
    } else {
      // Fallback: execute all updates in the same tick
      updates.forEach(update => update());
    }
  }, []);

  // Memoization helper with performance tracking
  const memoizeWithPerformance = useCallback(<T>(
    fn: (...args: any[]) => T,
    keyFn?: (...args: any[]) => string
  ) => {
    const cache = new Map<string, { value: T; timestamp: number }>();
    const cacheTimeout = 60000; // 1 minute cache

    return (...args: any[]): T => {
      const key = keyFn ? keyFn(...args) : JSON.stringify(args);
      const cached = cache.get(key);
      const now = Date.now();

      if (cached && (now - cached.timestamp) < cacheTimeout) {
        return cached.value;
      }

      const startTime = performance.now();
      const result = fn(...args);
      const duration = performance.now() - startTime;

      if (duration > 10) {
        console.log(`Expensive calculation cached: ${duration.toFixed(2)}ms for key: ${key}`);
      }

      cache.set(key, { value: result, timestamp: now });

      // Clean up old cache entries
      if (cache.size > 100) {
        const entries = Array.from(cache.entries());
        entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
        entries.slice(0, 50).forEach(([key]) => cache.delete(key));
      }

      return result;
    };
  }, []);

  return {
    metrics,
    startRenderMeasurement,
    endRenderMeasurement,
    trackUpdate,
    measureApiLatency,
    measureWsLatency,
    getOptimizationSuggestions,
    throttle,
    debounce,
    batchUpdates,
    memoizeWithPerformance
  };
};

export default usePerformanceMonitor;