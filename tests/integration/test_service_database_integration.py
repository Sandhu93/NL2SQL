"""
Module: test_service_database_integration.py
Description: Integration-oriented tests for NL2SQL service and database utilities
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

from app.services.nl2sql_service import NL2SQLService
from app.db.database_manager import create_database_uri, initialize_database, display_database_info


class TestServiceDatabaseIntegration:
    """Integration checks for NL2SQLService with a mocked SQLDatabase."""

    @pytest.fixture
    def service_with_db(self, mock_env_vars):
        """Provide an NL2SQLService wired to a mocked SQLDatabase."""
        with patch("app.db.database_manager.SQLDatabase") as mock_sql_db, \
             patch("app.services.nl2sql_service.ChatOpenAI") as mock_openai, \
             patch("app.services.nl2sql_service.create_sql_query_chain") as mock_chain, \
             patch("app.services.nl2sql_service.QuerySQLDataBaseTool") as mock_tool:
            mock_db = MagicMock()
            mock_db.dialect = "mysql"
            mock_db.get_usable_table_names.return_value = ["customers", "products"]
            mock_sql_db.from_uri.return_value = mock_db

            mock_openai_instance = MagicMock()
            mock_openai.return_value = mock_openai_instance

            mock_chain_instance = MagicMock()
            mock_chain.return_value = mock_chain_instance

            mock_tool_instance = MagicMock()
            mock_tool.return_value = mock_tool_instance

            service = NL2SQLService(db=mock_db, model_name="gpt-3.5-turbo")
            return service, mock_db, mock_chain_instance, mock_tool_instance

    def test_service_processes_question_with_mocked_components(self, service_with_db):
        """NL2SQLService.process_question should orchestrate chain and executor."""
        service, mock_db, mock_chain_instance, mock_tool_instance = service_with_db
        question = "How many customers are there?"
        generated_sql = "SELECT COUNT(*) FROM customers"
        query_result = "count: 10"

        mock_chain_instance.invoke.return_value = generated_sql
        mock_tool_instance.invoke.return_value = query_result

        sql, result = service.process_question(question)

        mock_chain_instance.invoke.assert_called_once_with({"question": question})
        mock_tool_instance.invoke.assert_called_once_with(generated_sql)
        assert sql == generated_sql
        assert result == query_result

    def test_service_raises_on_empty_question(self, service_with_db):
        """Empty questions should raise ValueError before hitting LLM/DB."""
        service, _, _, _ = service_with_db
        with pytest.raises(ValueError):
            service.process_question("   ")


class TestDatabaseManagerIntegration:
    """Integration checks for database manager helpers."""

    def test_database_uri_creation_and_initialization_flow(self, mock_env_vars):
        """Ensure URI formatting feeds initialize_database correctly."""
        with patch("app.db.database_manager.SQLDatabase") as mock_sql_db:
            mock_db = MagicMock()
            mock_sql_db.from_uri.return_value = mock_db

            db_uri = create_database_uri(mock_env_vars)
            db = initialize_database(db_uri)

        mock_sql_db.from_uri.assert_called_once_with(
            db_uri,
            sample_rows_in_table_info=2,
            include_tables=None,
            custom_table_info=None,
        )
        assert db is mock_db

    def test_display_database_info_outputs_schema(self, capsys):
        """display_database_info prints dialect and table info without raising."""
        mock_db = MagicMock()
        mock_db.dialect = "mysql"
        mock_db.get_usable_table_names.return_value = ["customers", "products"]
        mock_db.get_table_info.return_value = "CREATE TABLE customers (...);"

        display_database_info(mock_db)
        captured = capsys.readouterr()
        assert "DATABASE CONNECTION SUCCESSFUL" in captured.out
        assert "customers" in captured.out
