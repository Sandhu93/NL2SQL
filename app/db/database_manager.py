"""
Module: database_manager.py
Description: Database connection management and utilities
Dependencies: langchain-community, pymysql, sqlalchemy
Author: Sandeep B Kadam
Created: 2025-11-22
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Standard library imports
import logging
from typing import Dict, Optional
from urllib.parse import quote_plus, unquote

# Third-party imports
from langchain_community.utilities.sql_database import SQLDatabase

# Constants
DEFAULT_SAMPLE_ROWS = 2
MAX_CONNECTION_RETRIES = 3

logger = logging.getLogger(__name__)


def create_database_uri(env_vars: Dict[str, str]) -> str:
    """
    Create a properly formatted database URI from environment variables.
    
    Args:
        env_vars: Dictionary containing database environment variables
        
    Returns:
        str: Properly formatted database URI
    """
    # Get database connection parameters
    db_user = env_vars['DB_USER']
    db_password = env_vars['DB_PASSWORD']
    db_host = env_vars['DB_HOST']
    db_name = env_vars['DB_NAME']
    db_port = env_vars['DB_PORT']
    
    # First decode the password if it's already URL encoded, then encode it properly
    if '%' in db_password:
        # Password is already URL encoded in .env, decode it first
        decoded_password = unquote(db_password)
        logger.info("URL decoded password from .env file")
    else:
        decoded_password = db_password
    
    # Now URL encode for the connection string
    encoded_password = quote_plus(decoded_password)
    
    # Create database URI
    db_uri = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    # Log URI without password for security
    safe_uri = f"mysql+pymysql://{db_user}:****@{db_host}:{db_port}/{db_name}"
    logger.info(f"Database URI created: {safe_uri}")
    
    return db_uri


def initialize_database(db_uri: str) -> Optional[SQLDatabase]:
    """
    Initialize SQLDatabase connection with error handling.
    
    Args:
        db_uri: Database connection URI
        
    Returns:
        SQLDatabase: Database connection object or None if failed
    """
    try:
        logger.info("Initializing database connection...")
        
        # Create database connection
        db = SQLDatabase.from_uri(
            db_uri,
            sample_rows_in_table_info=DEFAULT_SAMPLE_ROWS,  # Include sample data in table info
            include_tables=None,  # Include all tables (can be customized)
            custom_table_info=None  # Custom table descriptions (can be added)
        )
        
        logger.info("Database connection established successfully")
        return db
        
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")
        return None


def display_database_info(db: SQLDatabase) -> None:
    """
    Display database information including dialect, tables, and schema.
    
    Args:
        db: SQLDatabase connection object
    """
    try:
        print("\n" + "="*60)
        print("DATABASE CONNECTION SUCCESSFUL!")
        print("="*60)
        
        # Display database dialect
        print(f"Database Dialect: {db.dialect}")
        
        # Display available tables
        table_names = db.get_usable_table_names()
        print(f"Available Tables ({len(table_names)}):")
        for table in table_names:
            print(f"  - {table}")
        
        print("\n" + "-"*60)
        print("DATABASE SCHEMA INFORMATION:")
        print("-"*60)
        
        # Display table information
        table_info = db.get_table_info()
        print(table_info)
        
        print("\n" + "="*60)
        logger.info("Database information displayed successfully")
        
    except Exception as e:
        logger.error(f"Failed to display database information: {e}")
        print(f"Error displaying database information: {e}")