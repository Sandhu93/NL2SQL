"""
Unit Tests for Main Application Functions
Author: AI Agent
Created: 2025-11-21
Python Version: 3.11
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import tempfile
from decimal import Decimal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    load_environment_variables,
    setup_openai_api,
    setup_langsmith_tracing,
    create_database_uri,
    initialize_database,
    display_database_info,
    execute_first_query
)

class TestEnvironmentVariables:
    """Test environment variable loading and validation."""
    
    def test_load_environment_variables_success(self, mock_env_vars):
        """Test successful loading of environment variables."""
        with patch('app.utils.env_loader.load_dotenv'):
            result = load_environment_variables()
            
            assert result['OPENAI_API_KEY'] == 'sk-test-key-12345'
            assert result['DB_USER'] == 'test_user'
            assert result['DB_PASSWORD'] == 'test_password%23'
            assert result['DB_HOST'] == 'localhost'
            assert result['DB_NAME'] == 'test_classicmodels'
            assert result['DB_PORT'] == '3306'
    
    def test_load_environment_variables_missing_required(self):
        """Test error when required environment variables are missing."""
        with patch('app.utils.env_loader.load_dotenv'), \
             patch.dict(os.environ, {'DB_USER': 'test'}, clear=True):
            
            with pytest.raises(ValueError) as exc_info:
                load_environment_variables()
            
            assert "Missing required environment variables" in str(exc_info.value)
            assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_load_environment_variables_optional_defaults(self, mock_env_vars):
        """Test that optional variables get default values."""
        # Remove optional variables
        env_vars = mock_env_vars.copy()
        del env_vars['LANGSMITH_API_KEY']
        del env_vars['LANGSMITH_TRACING']
        
        with patch('app.utils.env_loader.load_dotenv'), \
             patch.dict(os.environ, env_vars, clear=True):
            
            result = load_environment_variables()
            
            assert result['DB_PORT'] == '3306'  # Default value
            assert result['LANGSMITH_TRACING'] == 'false'  # Default value

class TestAPISetup:
    """Test API setup functions."""
    
    def test_setup_openai_api(self):
        """Test OpenAI API key setup."""
        test_key = "sk-test-key-12345"
        
        with patch.dict(os.environ, {}, clear=True):
            setup_openai_api(test_key)
            assert os.environ["OPENAI_API_KEY"] == test_key
    
    def test_setup_langsmith_tracing_enabled(self):
        """Test LangSmith tracing when enabled."""
        api_key = "test-langsmith-key"
        
        with patch.dict(os.environ, {}, clear=True):
            setup_langsmith_tracing(api_key, "true")
            
            assert os.environ["LANGCHAIN_API_KEY"] == api_key
            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    
    def test_setup_langsmith_tracing_disabled(self):
        """Test LangSmith tracing when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            setup_langsmith_tracing("test-key", "false")
            
            assert "LANGCHAIN_API_KEY" not in os.environ
            assert "LANGCHAIN_TRACING_V2" not in os.environ

class TestDatabaseURI:
    """Test database URI creation."""
    
    def test_create_database_uri_simple_password(self):
        """Test URI creation with simple password."""
        env_vars = {
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'simplepass',
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'testdb'
        }
        
        result = create_database_uri(env_vars)
        expected = "mysql+pymysql://testuser:simplepass@localhost:3306/testdb"
        assert result == expected
    
    def test_create_database_uri_encoded_password(self):
        """Test URI creation with URL-encoded password."""
        env_vars = {
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'pass%23word',  # Contains encoded #
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'testdb'
        }
        
        result = create_database_uri(env_vars)
        # Password should be decoded then re-encoded
        assert "pass%23word" in result
    
    def test_create_database_uri_special_characters(self):
        """Test URI creation with special characters in password."""
        env_vars = {
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'p@ssw0rd!',
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'testdb'
        }
        
        result = create_database_uri(env_vars)
        # Special characters should be URL encoded
        assert "p%40ssw0rd%21" in result

class TestDatabaseInitialization:
    """Test database initialization."""
    
    @patch('app.db.database_manager.SQLDatabase')
    def test_initialize_database_success(self, mock_sql_database):
        """Test successful database initialization."""
        mock_db = MagicMock()
        mock_sql_database.from_uri.return_value = mock_db
        
        uri = "mysql+pymysql://user:pass@localhost:3306/testdb"
        result = initialize_database(uri)
        
        assert result == mock_db
        mock_sql_database.from_uri.assert_called_once_with(
            uri,
            sample_rows_in_table_info=2,
            include_tables=None,
            custom_table_info=None
        )
    
    @patch('app.db.database_manager.SQLDatabase')
    def test_initialize_database_failure(self, mock_sql_database):
        """Test database initialization failure."""
        mock_sql_database.from_uri.side_effect = Exception("Connection failed")
        
        uri = "mysql+pymysql://user:pass@localhost:3306/testdb"
        result = initialize_database(uri)
        
        assert result is None

class TestDatabaseInfo:
    """Test database information display."""
    
    def test_display_database_info(self, mock_sql_database, capsys):
        """Test database information display."""
        mock_sql_database.dialect = "mysql"
        mock_sql_database.get_usable_table_names.return_value = ["customers", "products"]
        mock_sql_database.get_table_info.return_value = "CREATE TABLE customers (id INT);"

        display_database_info(mock_sql_database)
        
        captured = capsys.readouterr()
        assert "DATABASE CONNECTION SUCCESSFUL!" in captured.out
        assert "mysql" in captured.out
        assert "customers" in captured.out
        assert "products" in captured.out
    
    def test_display_database_info_error(self, capsys):
        """Test database info display with error."""
        mock_db = MagicMock()
        mock_db.dialect = "mysql"
        mock_db.get_usable_table_names.side_effect = Exception("DB Error")
        
        # Should not raise exception, should handle gracefully
        display_database_info(mock_db)
        
        # Function should complete without raising exception

class TestFirstQuery:
    """Test first query functionality."""
    
    @patch('app.services.nl2sql_service.NL2SQLService')
    def test_first_query_success(self, mock_service_class, mock_sql_database, capsys):
        """Test successful first query execution using NL2SQLService."""
        mock_service = MagicMock()
        mock_service.process_question.return_value = (
            "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'",
            [{"buyPrice": Decimal("95.34")}],
        )
        mock_service_class.return_value = mock_service

        execute_first_query(mock_sql_database)

        mock_service_class.assert_called_once_with(mock_sql_database)
        assert mock_service.process_question.call_count == 4
        captured = capsys.readouterr()
        assert "FIRST QUERY TEST - NL2SQL CONVERSION" in captured.out
    
    @patch('app.services.nl2sql_service.NL2SQLService')
    def test_first_query_openai_error(self, mock_service_class, mock_sql_database, capsys):
        """Test first query with service error is handled gracefully."""
        mock_service = MagicMock()
        mock_service.process_question.side_effect = Exception("API Error")
        mock_service_class.return_value = mock_service

        execute_first_query(mock_sql_database)

        captured = capsys.readouterr()
        assert "Error during first query test" in captured.out or "Error during query test" in captured.out

class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('main.initialize_database')
    @patch('main.execute_first_query')
    def test_main_function_success(self, mock_test_query, mock_init_db, mock_env_vars, mock_sql_database):
        """Test main function successful execution."""
        from main import main
        
        # Setup mocks
        mock_init_db.return_value = mock_sql_database
        
        with patch.dict(os.environ, mock_env_vars):
            result = main()
        
        # Verify execution flow
        mock_init_db.assert_called_once()
        mock_test_query.assert_called_once_with(mock_sql_database)
        assert result == mock_sql_database
    
    def test_main_function_missing_env_vars(self):
        """Test main function with missing environment variables."""
        from main import main
        
        with patch.dict(os.environ, {}, clear=True), \
             pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    @patch('main.initialize_database')
    def test_main_function_db_connection_failure(self, mock_init_db, mock_env_vars):
        """Test main function with database connection failure."""
        from main import main
        
        mock_init_db.return_value = None  # Simulate connection failure
        
        with patch.dict(os.environ, mock_env_vars), \
             pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
