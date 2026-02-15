"""
Environment configuration management using environment variables.
Provides fallback to YAML config for backward compatibility.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Settings:
    """Application settings from environment variables."""
    
    # Application
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Database
    POSTGRES_URI: Optional[str] = os.getenv("POSTGRES_URI")
    POSTGRES_TABLE: str = os.getenv("POSTGRES_TABLE", "financials")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "risk_prediction")
    
    # Model
    MODEL_RANDOM_STATE: int = int(os.getenv("MODEL_RANDOM_STATE", "42"))
    MODEL_TEST_SIZE: float = float(os.getenv("MODEL_TEST_SIZE", "0.2"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    
    # Streamlit
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.APP_ENV.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.APP_ENV.lower() == "development"


settings = Settings()
