"""
🔧 FinanceGPT Live Configuration System
================================    # 🔌 WebSocket Configuration  
    websocket_max_connections: int = Field(default=1000, env="WEBSOCKET_MAX_CONNECTIONS")
    websocket_heartbeat_interval: int = Field(default=5, env="WEBSOCKET_HEARTBEAT_INTERVAL")  # Super fast heartbeat!==

Advanced configuration management for enterprise-grade deployment
with environment-specific settings, secrets management, and performance tuning.

"""

import os
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    🎯 Comprehensive Application Settings
    
    Handles all configuration aspects:
    - API keys and secrets
    - Database connections
    - Performance parameters
    - Feature flags
    - Security settings
    """
    
    # 🏷️ Application Info
    app_name: str = "FinanceGPT Live"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # 🌐 Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8001, env="PORT")  # Updated to 8001
    reload: bool = Field(default=True, env="RELOAD")
    workers: int = Field(default=4, env="WORKERS")
    
    # 🔐 Security Settings
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # 🤖 AI/LLM Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    default_model: str = Field(default="gpt-4-turbo-preview", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # 📊 Financial Data APIs
    alpha_vantage_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_KEY")
    finnhub_key: Optional[str] = Field(None, env="FINNHUB_KEY")
    polygon_key: Optional[str] = Field(None, env="POLYGON_KEY")
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    
    # 🗄️ Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./financegpt.db",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # 📦 Redis/Cache Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # 🔥 Pathway Configuration
    pathway_license_key: Optional[str] = Field(None, env="PATHWAY_LICENSE_KEY")
    pathway_threads: int = Field(default=4, env="PATHWAY_THREADS")
    pathway_persistence_dir: str = Field(default="./pathway_data", env="PATHWAY_PERSISTENCE_DIR")
    
    # ⚡ WebSocket Configuration
    websocket_max_connections: int = Field(default=1000, env="WEBSOCKET_MAX_CONNECTIONS")
    websocket_heartbeat_interval: int = Field(default=30, env="WEBSOCKET_HEARTBEAT_INTERVAL")
    
    # 📈 Market Data Configuration
    market_data_update_interval: int = Field(default=2, env="MARKET_DATA_UPDATE_INTERVAL")  # seconds - super fast updates!
    max_symbols_per_request: int = Field(default=100, env="MAX_SYMBOLS_PER_REQUEST")
    historical_data_days: int = Field(default=365, env="HISTORICAL_DATA_DAYS")
    
    # 🤖 Agent Configuration
    agent_concurrency_limit: int = Field(default=10, env="AGENT_CONCURRENCY_LIMIT")
    agent_timeout_seconds: int = Field(default=30, env="AGENT_TIMEOUT_SECONDS")
    max_agent_retries: int = Field(default=3, env="MAX_AGENT_RETRIES")
    
    # 📊 Analytics Configuration
    analytics_batch_size: int = Field(default=1000, env="ANALYTICS_BATCH_SIZE")
    analytics_processing_interval: int = Field(default=60, env="ANALYTICS_PROCESSING_INTERVAL")
    
    # 🛡️ Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=1000, env="RATE_LIMIT_RPM")
    rate_limit_burst_size: int = Field(default=100, env="RATE_LIMIT_BURST")
    
    # 📱 Frontend Configuration
    frontend_url: str = Field(default="http://localhost:3000", env="FRONTEND_URL")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # 🔍 Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # 🧪 Feature Flags
    enable_real_time_streaming: bool = Field(default=True, env="ENABLE_REAL_TIME_STREAMING")
    enable_advanced_analytics: bool = Field(default=True, env="ENABLE_ADVANCED_ANALYTICS")
    enable_ai_agents: bool = Field(default=True, env="ENABLE_AI_AGENTS")
    enable_portfolio_tracking: bool = Field(default=True, env="ENABLE_PORTFOLIO_TRACKING")
    enable_risk_monitoring: bool = Field(default=True, env="ENABLE_RISK_MONITORING")
    enable_news_sentiment: bool = Field(default=True, env="ENABLE_NEWS_SENTIMENT")
    
    # 🚀 Performance Tuning
    async_pool_size: int = Field(default=100, env="ASYNC_POOL_SIZE")
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    request_timeout: int = Field(default=60, env="REQUEST_TIMEOUT")
    
    # 📊 Data Storage
    data_retention_days: int = Field(default=30, env="DATA_RETENTION_DAYS")
    backup_interval_hours: int = Field(default=6, env="BACKUP_INTERVAL_HOURS")
    
    # 🔐 Security Headers
    security_headers: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;"
    }
    
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "url": self.redis_url,
            "ttl": self.cache_ttl,
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI/LLM configuration dictionary"""
        return {
            "openai_api_key": self.openai_api_key,
            "default_model": self.default_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
    
    def get_market_data_config(self) -> Dict[str, Any]:
        """Get market data configuration dictionary"""
        return {
            "alpha_vantage_key": self.alpha_vantage_key,
            "finnhub_key": self.finnhub_key,
            "polygon_key": self.polygon_key,
            "update_interval": self.market_data_update_interval,
            "max_symbols": self.max_symbols_per_request,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """
    🔧 Get cached application settings
    
    Uses LRU cache to ensure settings are loaded once and reused.
    """
    logger.info("Loading application settings...")
    return Settings()

# Global settings instance
settings = get_settings()

# 🚀 Export commonly used configurations
DATABASE_CONFIG = settings.get_database_config()
REDIS_CONFIG = settings.get_redis_config()
AI_CONFIG = settings.get_ai_config()
MARKET_DATA_CONFIG = settings.get_market_data_config()
