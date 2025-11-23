"""
Module: main.py
Description: NL2SQL Tutorial Main Application Entry Point
Dependencies: See requirements.txt
Author: Sandeep B Kadam
Created: 2025-11-21
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Standard library imports
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from app.utils.logging_config import setup_logging
from app.utils.env_loader import load_environment_variables, setup_openai_api, setup_langsmith_tracing
from app.db.database_manager import create_database_uri, initialize_database, display_database_info
from app.services.nl2sql_service import execute_first_query, execute_refined_query

# Setup logging
logger = setup_logging()


def main():
    """Main application entry point."""
    logger.info("Starting NL2SQL Tutorial Application")
    
    try:
        # Load and validate environment variables
        env_vars = load_environment_variables()
        
        # Setup OpenAI API
        setup_openai_api(env_vars['OPENAI_API_KEY'])
        
        # Setup LangSmith tracing (optional)
        setup_langsmith_tracing(
            env_vars.get('LANGSMITH_API_KEY'), 
            env_vars.get('LANGSMITH_TRACING', 'false')
        )
        
        # Create database URI
        db_uri = create_database_uri(env_vars)
        
        # Initialize database connection
        db = initialize_database(db_uri)
        if not db:
            logger.error("Failed to initialize database. Exiting.")
            sys.exit(1)
        
        # Display database information
        display_database_info(db)
        
        logger.info("Application initialized successfully")
        
        # Test first query functionality
        execute_first_query(db)
        execute_refined_query(db)
        
        logger.info("Ready for NL2SQL operations!")
        
        return db  # Return database object for potential further use
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    database = main()
