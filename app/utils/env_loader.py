"""
Module: env_loader.py
Description: Environment variable loading and validation utilities
Dependencies: python-dotenv
Author: Sandeep B Kadam
Created: 2025-11-22
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Optional

# Third-party imports
from dotenv import load_dotenv

# Constants
DEFAULT_DB_PORT = "3306"
DEFAULT_LANGSMITH_TRACING = "false"
REQUIRED_ENV_VARS = {
    'OPENAI_API_KEY': 'OpenAI API key for language model',
    'DB_USER': 'Database username',
    'DB_PASSWORD': 'Database password', 
    'DB_HOST': 'Database host address',
    'DB_NAME': 'Database name'
}
OPTIONAL_ENV_VARS = {
    'DB_PORT': DEFAULT_DB_PORT,
    'LANGSMITH_API_KEY': None,
    'LANGSMITH_TRACING': DEFAULT_LANGSMITH_TRACING
}

logger = logging.getLogger(__name__)


def load_environment_variables() -> Dict[str, str]:
    """
    Load and validate all required environment variables.
    
    Returns:
        Dict[str, str]: Dictionary containing all environment variables
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Loading environment variables from .env file")
    
    env_vars = {}
    missing_vars = []
    
    # Check required variables
    for var_name, description in REQUIRED_ENV_VARS.items():
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(f"{var_name} ({description})")
        else:
            env_vars[var_name] = value
            logger.info(f"[OK] {var_name} loaded successfully")
    
    # Load optional variables
    for var_name, default_value in OPTIONAL_ENV_VARS.items():
        value = os.getenv(var_name, default_value)
        env_vars[var_name] = value
        if value:
            logger.info(f"[OK] {var_name} loaded: {value if var_name != 'LANGSMITH_API_KEY' else '[HIDDEN]'}")
    
    # Check for missing required variables
    if missing_vars:
        error_msg = f"Missing required environment variables:\n" + "\n".join(f"  - {var}" for var in missing_vars)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return env_vars


def setup_openai_api(api_key: str) -> None:
    """
    Setup OpenAI API key in environment.
    
    Args:
        api_key: OpenAI API key from environment variables
    """
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OpenAI API key configured successfully")


def setup_langsmith_tracing(api_key: Optional[str], tracing_enabled: str) -> None:
    """
    Setup LangSmith tracing if credentials are provided.
    
    Args:
        api_key: LangSmith API key (optional)
        tracing_enabled: Whether tracing is enabled ("true"/"false")
    """
    if api_key and tracing_enabled.lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        logger.info("LangSmith tracing enabled")
    else:
        logger.info("LangSmith tracing disabled or no API key provided")