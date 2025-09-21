import asyncio
import time
from functools import wraps
from typing import Dict, List, Any, Callable, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.cache_ttl = {}
        self.request_counts = {}
        self.rate_limits = {}
        
    def async_cache(self, ttl_seconds: int = 300):
        """Async caching decorator with TTL"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Check cache
                if cache_key in self.cache:
                    if time.time() < self.cache_ttl.get(cache_key, 0):
                        return self.cache[cache_key]
                    else:
                        # Expired, remove from cache
                        del self.cache[cache_key]
                        if cache_key in self.cache_ttl:
                            del self.cache_ttl[cache_key]
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                self.cache[cache_key] = result
                self.cache_ttl[cache_key] = time.time() + ttl_seconds
                
                return result
            return wrapper
        return decorator
    
    def rate_limit(self, max_calls: int = 10, window_seconds: int = 60):
        """Rate limiting decorator"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                func_name = func.__name__
                current_time = time.time()
                
                # Initialize rate limit tracking
                if func_name not in self.request_counts:
                    self.request_counts[func_name] = []
                
                # Clean old requests outside window
                self.request_counts[func_name] = [
                    req_time for req_time in self.request_counts[func_name]
                    if current_time - req_time < window_seconds
                ]
                
                # Check rate limit
                if len(self.request_counts[func_name]) >= max_calls:
                    raise Exception(f"Rate limit exceeded for {func_name}")
                
                # Record this request
                self.request_counts[func_name].append(current_time)
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def batch_process(self, items: List[Any], batch_size: int = 10, max_concurrent: int = 3):
        """Process items in batches with concurrency control"""
        async def process_batches(processor_func: Callable):
            results = []
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_batch(batch):
                async with semaphore:
                    return await processor_func(batch)
            
            # Create batches
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            # Process batches concurrently
            tasks = [process_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"Batch processing error: {batch_result}")
                    continue
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            
            return results
        
        return process_batches
    
    def memory_efficient_processing(self, data_stream, chunk_size: int = 1000):
        """Process large datasets in memory-efficient chunks"""
        def process_chunks(processor_func: Callable):
            results = []
            
            for i in range(0, len(data_stream), chunk_size):
                chunk = data_stream[i:i + chunk_size]
                try:
                    chunk_result = processor_func(chunk)
                    results.append(chunk_result)
                except Exception as e:
                    logger.error(f"Chunk processing error: {e}")
                    continue
            
            return results
        
        return process_chunks
    
    def parallel_execution(self, tasks: List[Callable], max_workers: int = 4):
        """Execute tasks in parallel using thread pool"""
        async def execute_parallel():
            loop = asyncio.get_event_loop()
            
            # Execute tasks in thread pool
            futures = []
            for task in tasks:
                future = loop.run_in_executor(self.thread_pool, task)
                futures.append(future)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Filter out exceptions
            successful_results = [
                result for result in results 
                if not isinstance(result, Exception)
            ]
            
            return successful_results
        
        return execute_parallel()
    
    def optimize_dataframe_operations(self, df_operations: List[Callable]):
        """Optimize pandas DataFrame operations"""
        def optimized_pipeline(df):
            # Use method chaining for better performance
            result = df.copy()
            
            for operation in df_operations:
                try:
                    result = operation(result)
                except Exception as e:
                    logger.error(f"DataFrame operation error: {e}")
                    continue
            
            return result
        
        return optimized_pipeline
    
    def smart_retry(self, max_retries: int = 3, backoff_factor: float = 1.5):
        """Smart retry with exponential backoff"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            break
                        
                        # Calculate backoff delay
                        delay = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                
                raise last_exception
            return wrapper
        return decorator
    
    def performance_monitor(self, func: Callable):
        """Monitor function performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = self._get_memory_usage()
            
            try:
                result = await func(*args, **kwargs)
                
                # Log performance metrics
                execution_time = time.time() - start_time
                memory_after = self._get_memory_usage()
                memory_delta = memory_after - memory_before
                
                logger.info(f"Performance: {func.__name__} - {execution_time:.3f}s, Memory: {memory_delta:.2f}MB")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Performance: {func.__name__} failed after {execution_time:.3f}s - {e}")
                raise
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.cache_ttl.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_ttl:
                del self.cache_ttl[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "cache_size": len(self.cache),
            "active_rate_limits": len(self.request_counts),
            "thread_pool_active": self.thread_pool._threads,
            "memory_usage_mb": self._get_memory_usage()
        }

# Global optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience decorators
async_cache = performance_optimizer.async_cache
rate_limit = performance_optimizer.rate_limit
smart_retry = performance_optimizer.smart_retry
performance_monitor = performance_optimizer.performance_monitor