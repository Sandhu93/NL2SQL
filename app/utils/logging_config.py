"""
Module: logging_config.py
Description: Logging configuration and setup utilities
Dependencies: Standard library logging
Author: AI Agent
Created: 2025-11-22
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Standard library imports
import logging
import sys
from datetime import datetime
from typing import Optional

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = 'nl2sql.log'
NOISY_LOGGERS = ['httpx', 'chromadb']


def setup_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
    include_console: bool = True
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Log file name (default: nl2sql.log)
        include_console: Whether to include console output (default: True)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file is None:
        log_file = f"nl2sql_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create handlers list
    handlers = []
    
    # Add file handler
    handlers.append(logging.FileHandler(log_file))
    
    # Add console handler if requested
    if include_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=handlers
    )
    
    # Set specific log levels for noisy third-party libraries
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {logging.getLevelName(log_level)}, File: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)