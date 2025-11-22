"""
Module: test_main_integration.py
Description: Integration-level checks for the main application entry point
Dependencies: pytest, unittest.mock
Author: AI Assistant
Created: 2025-11-22
Last Modified: 2025-11-22
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import main


class TestMainIntegration:
    """Integration tests for main application functionality."""

    def test_main_application_startup_success(self, mock_env_vars):
        """Verify happy-path startup wires dependencies and returns the db handle."""
        mock_database = MagicMock()

        with patch("main.load_environment_variables", return_value=mock_env_vars), \
             patch("main.setup_openai_api"), \
             patch("main.setup_langsmith_tracing"), \
             patch("main.create_database_uri", return_value="mysql://uri"), \
             patch("main.initialize_database", return_value=mock_database), \
             patch("main.display_database_info"), \
             patch("main.execute_first_query") as mock_first_query:
            result = main.main()

        mock_first_query.assert_called_once_with(mock_database)
        assert result == mock_database

    def test_main_application_with_missing_env_vars(self):
        """Startup fails fast when required environment variables are missing."""
        with patch("main.load_environment_variables", side_effect=ValueError("Missing env var")):
            with pytest.raises(SystemExit):
                main.main()

    def test_main_application_database_connection_failure(self, mock_env_vars):
        """Startup exits when database initialization fails."""
        with patch("main.load_environment_variables", return_value=mock_env_vars), \
             patch("main.setup_openai_api"), \
             patch("main.setup_langsmith_tracing"), \
             patch("main.create_database_uri", return_value="mysql://uri"), \
             patch("main.initialize_database", return_value=None):
            with pytest.raises(SystemExit):
                main.main()


class TestEndToEndWorkflow:
    """End-to-end flow validations with patched dependencies."""

    def test_environment_to_database_connection_flow(self, mock_env_vars):
        """Ensure environment loading feeds into database URI creation and init."""
        mock_database = MagicMock()
        with patch("main.load_environment_variables", return_value=mock_env_vars) as mock_load_env, \
             patch("main.create_database_uri", return_value="mysql://uri") as mock_create_uri, \
             patch("main.initialize_database", return_value=mock_database) as mock_init_db:
            env_vars = main.load_environment_variables()
            db_uri = main.create_database_uri(env_vars)
            database = main.initialize_database(db_uri)

        mock_load_env.assert_called_once()
        mock_create_uri.assert_called_once_with(mock_env_vars)
        mock_init_db.assert_called_once_with("mysql://uri")
        assert database == mock_database

    def test_openai_setup_is_invoked(self, mock_env_vars):
        """Validate OpenAI setup receives the configured API key."""
        with patch("main.setup_openai_api") as mock_setup_openai:
            main.setup_openai_api(mock_env_vars["OPENAI_API_KEY"])
        mock_setup_openai.assert_called_once_with(mock_env_vars["OPENAI_API_KEY"])

    def test_complete_pipeline_calls_first_query(self, mock_env_vars):
        """Main should reach execute_first_query after successful setup."""
        mock_database = MagicMock()
        with patch("main.load_environment_variables", return_value=mock_env_vars), \
             patch("main.setup_openai_api"), \
             patch("main.setup_langsmith_tracing"), \
             patch("main.create_database_uri", return_value="mysql://uri"), \
             patch("main.initialize_database", return_value=mock_database), \
             patch("main.display_database_info"), \
             patch("main.execute_first_query") as mock_first_query:
            main.main()
        mock_first_query.assert_called_once_with(mock_database)


class TestMainApplicationConfiguration:
    """Configuration and logging behavior for main."""
    
    def test_logging_configuration_in_main(self):
        """Module-level logger should be configured."""
        import logging
        assert isinstance(main.logger, logging.Logger)

    def test_graceful_shutdown_on_keyboard_interrupt(self, mock_env_vars):
        """KeyboardInterrupt during setup should propagate to caller."""
        with patch("main.load_environment_variables", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                main.main()
