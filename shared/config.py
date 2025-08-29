"""Configuration management utilities"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Database configuration
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/ai_pipeline")
    
    # Redis configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Logging configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    
    # Service configuration
    service_name: str = os.getenv("SERVICE_NAME", "unknown-service")
    service_version: str = os.getenv("SERVICE_VERSION", "1.0.0")
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_config() -> Settings:
    """Get application configuration"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_config() -> Settings:
    """Reload configuration from environment"""
    global _settings
    _settings = Settings()
    return _settings