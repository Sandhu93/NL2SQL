"""
Module: constants.py
Description: Application-wide constants and configuration values
Author: AI Agent
Created: 2025-11-22
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Database Configuration Constants
DEFAULT_DB_PORT = "3306"
DEFAULT_SAMPLE_ROWS = 2
MAX_CONNECTION_RETRIES = 3

# Environment Variable Names
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_DB_USER = "DB_USER"
ENV_DB_PASSWORD = "DB_PASSWORD"
ENV_DB_HOST = "DB_HOST"
ENV_DB_NAME = "DB_NAME"
ENV_DB_PORT = "DB_PORT"
ENV_LANGSMITH_API_KEY = "LANGSMITH_API_KEY"
ENV_LANGSMITH_TRACING = "LANGSMITH_TRACING"

# LangSmith Configuration Constants
DEFAULT_LANGSMITH_TRACING = "false"
LANGSMITH_TRACING_V2 = "LANGCHAIN_TRACING_V2"
LANGSMITH_API_KEY_ENV = "LANGCHAIN_API_KEY"

# OpenAI Model Configuration Constants
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
MAX_RETRIES = 3

# Logging Configuration Constants
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = 'nl2sql.log'

# Performance Constants
DEFAULT_CACHE_SIZE = 128
DEFAULT_TIMEOUT_SECONDS = 300

# Test Query Constants
TEST_QUESTION_MUSTANG = "what is price of `1968 Ford Mustang`"
TEST_QUERIES = [
    "Show me all products priced above $100",
    "How many customers are there?",
    "List all offices in the USA"
]

# Security Constants
DANGEROUS_SQL_KEYWORDS = [
    "DROP", "DELETE", "TRUNCATE", "INSERT", 
    "UPDATE", "ALTER", "CREATE", "EXEC"
]

# Display Constants
SEPARATOR_LENGTH = 60
HEADER_CHAR = "="
SUBHEADER_CHAR = "-"