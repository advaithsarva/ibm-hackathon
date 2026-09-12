from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application
    APP_NAME: str = "Disaster Early Warning Network"
    DEBUG: bool = False
    API_VERSION: str = "v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./disasters.db"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Security
    SECRET_KEY: str = "change-this-secret-key-in-production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
