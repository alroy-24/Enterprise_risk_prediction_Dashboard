"""
Centralized logging configuration for the Risk Prediction Platform.
Uses loguru for structured, colored, and file-based logging.
"""
import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Get log level and file from environment
try:
    from env_config import settings
    log_level = settings.LOG_LEVEL
    log_file = settings.LOG_FILE
except ImportError:
    log_level = "INFO"
    log_file = "logs/app.log"

# Ensure logs directory exists
log_path = Path(log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)

# Add console handler with colors
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level,
    colorize=True,
)

# Add file handler with rotation
logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=log_level,
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    serialize=False,
)

# Add JSON file handler for structured logs (production-ready)
logger.add(
    log_path.parent / "app_json.log",
    format="{message}",
    level=log_level,
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    serialize=True,  # JSON format
)


def get_logger(name: str):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(module=name)


# Example usage logger
__all__ = ["logger", "get_logger"]
