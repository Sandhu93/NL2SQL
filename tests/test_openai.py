"""
OpenAI-related tests for NL2SQL System
Author: AI Assistant
Created: 2025-11-21
Python Version: 3.11
"""

import os
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_openai_api, execute_first_query


class TestOpenAIAPISetup:
    """Test OpenAI API configuration and setup."""

    def test_setup_openai_api_key(self):
        """setup_openai_api should place the key in the environment."""
        test_key = "sk-test-1234567890"
        with patch.dict(os.environ, {}, clear=True):
            setup_openai_api(test_key)
            assert os.environ["OPENAI_API_KEY"] == test_key

    def test_openai_api_key_persistence(self):
        """Subsequent calls should leave the key unchanged."""
        test_key = "sk-test-persistent"
        with patch.dict(os.environ, {}, clear=True):
            setup_openai_api(test_key)
            setup_openai_api(test_key)
            assert os.environ.get("OPENAI_API_KEY") == test_key


class TestFirstQueryFlow:
    """Verify execute_first_query orchestrates the chain and executor."""

    @patch("app.services.nl2sql_service.NL2SQLService")
    def test_first_query_success(self, mock_service_class, mock_sql_database):
        mock_service = MagicMock()
        mock_service.process_question.return_value = (
            "SELECT 1",
            [{"value": 1}],
        )
        mock_service_class.return_value = mock_service

        execute_first_query(mock_sql_database)

        mock_service_class.assert_called_once()
        assert mock_service.process_question.call_count == 4

    @patch("app.services.nl2sql_service.NL2SQLService")
    def test_first_query_handles_errors(self, mock_service_class, mock_sql_database, capsys):
        mock_service = MagicMock()
        mock_service.process_question.side_effect = Exception("API Error")
        mock_service_class.return_value = mock_service

        execute_first_query(mock_sql_database)
        captured = capsys.readouterr()
        assert "Error during first query test" in captured.out or "Error during query test" in captured.out
