"""
Integration Tests for NL2SQL System
Author: AI Assistant
Created: 2025-11-21
Python Version: 3.11
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import main, load_environment_variables, initialize_database, execute_first_query


class TestDatabaseIntegration:
    """Ensure database helpers can be chained together."""

    @pytest.mark.integration
    def test_database_connection_integration(self, mock_sql_database, mock_env_vars):
        mock_sql_database.dialect = "mysql"
        mock_sql_database.get_usable_table_names.return_value = ["customers", "products"]

        with patch("app.db.database_manager.SQLDatabase") as mock_sql_db, \
             patch("main.create_database_uri", return_value="mysql://uri"):
            mock_sql_db.from_uri.return_value = mock_sql_database
            db = initialize_database("mysql://uri")

        assert db is mock_sql_database
        assert db.get_usable_table_names() == ["customers", "products"]


class TestFullSystemIntegration:
    """Check main() orchestrates environment loading and query execution."""

    @pytest.mark.integration
    @patch("main.execute_first_query")
    @patch("main.initialize_database")
    def test_complete_system_flow(self, mock_init_db, mock_first_query, mock_env_vars):
        mock_db = MagicMock()
        mock_init_db.return_value = mock_db
        with patch("main.load_environment_variables", return_value=mock_env_vars), \
             patch("main.setup_openai_api"), \
             patch("main.setup_langsmith_tracing"), \
             patch("main.create_database_uri", return_value="mysql://uri"):
            result = main()

        assert result == mock_db
        mock_first_query.assert_called_once_with(mock_db)

    @pytest.mark.integration
    def test_system_with_missing_openai_key(self):
        with patch("main.load_environment_variables", side_effect=ValueError("missing OPENAI_API_KEY")):
            with pytest.raises(SystemExit):
                main()
