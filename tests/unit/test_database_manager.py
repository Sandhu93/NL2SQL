"""
Module: test_database_manager.py
Description: Unit tests for database connection management and utilities
Dependencies: pytest, unittest.mock, sqlalchemy
Author: AI Assistant
Created: 2025-11-22
Last Modified: 2025-11-22
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.database_manager import (
    create_database_uri,
    initialize_database,
    display_database_info
)


class TestDatabaseManager:
    """Test database connection management functionality."""
    
    def test_create_database_uri_success(self):
        """Test successful database URI creation."""
        # Arrange
        db_config = {
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'testpass%23',
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'testdb'
        }
        
        # Act
        result = create_database_uri(db_config)
        
        # Assert
        expected = 'mysql+pymysql://testuser:testpass%23@localhost:3306/testdb'
        assert result == expected
    
    def test_create_database_uri_with_special_characters(self):
        """Test database URI creation with special characters in password."""
        # Arrange
        db_config = {
            'DB_USER': 'user@domain',
            'DB_PASSWORD': 'p@ss!w0rd#$%',
            'DB_HOST': 'db.example.com',
            'DB_PORT': '3306',
            'DB_NAME': 'mydb'
        }
        
        # Act
        result = create_database_uri(db_config)
        
        # Assert
        assert 'mysql+pymysql://' in result
        assert 'user@domain' in result  # Username not encoded
        assert 'p%40ss%21w0rd%23%24%25' in result  # Password is encoded
        assert 'db.example.com:3306/mydb' in result
    
    def test_create_database_uri_missing_required_field(self):
        """Test database URI creation with missing required field."""
        # Arrange
        incomplete_config = {
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'testpass',
            'DB_HOST': 'localhost'
            # Missing DB_PORT and DB_NAME
        }
        
        # Act & Assert
        with pytest.raises(KeyError):
            create_database_uri(incomplete_config)
    
    def test_initialize_database_success(self, mock_database_connection):
        """Test successful database initialization."""
        # Arrange
        test_uri = 'mysql+pymysql://user:pass@localhost:3306/testdb'
        
        # Mock the database connection and table info
        mock_database_connection.dialect.name = 'mysql'
        mock_table_info = "CREATE TABLE customers (id INT PRIMARY KEY);"
        
        # Act
        with patch('app.db.database_manager.SQLDatabase') as mock_sql_db:
            mock_sql_db.from_uri.return_value = mock_database_connection
            mock_database_connection.table_info = mock_table_info
            
            result = initialize_database(test_uri)
        
        # Assert
        mock_sql_db.from_uri.assert_called_once_with(
            test_uri,
            sample_rows_in_table_info=2,
            include_tables=None,
            custom_table_info=None
        )
        assert result == mock_database_connection
        assert result.table_info == mock_table_info
    
    def test_initialize_database_connection_failure(self):
        """Test database initialization with connection failure."""
        # Arrange
        test_uri = 'mysql+pymysql://invalid:creds@badhost:3306/nonexistent'
        
        # Act & Assert
        with patch('app.db.database_manager.SQLDatabase') as mock_sql_db:
            mock_sql_db.from_uri.side_effect = Exception("Connection failed")
            
            result = initialize_database(test_uri)
            
            # Function returns None on failure instead of raising exception
            assert result is None
    
    def test_display_database_info_success(self, mock_database_connection, mock_logger):
        """Test successful database info display."""
        # Arrange
        mock_database_connection.dialect.name = 'mysql'
        mock_database_connection.get_usable_table_names.return_value = [
            'customers', 'products', 'orders'
        ]
        table_info = "CREATE TABLE customers (id INT);\nCREATE TABLE products (id INT);"
        mock_database_connection.get_table_info.return_value = table_info
        
        # Act
        with patch('app.db.database_manager.logger', mock_logger), \
             patch('builtins.print') as mock_print:
            display_database_info(mock_database_connection)
        
        # Assert - function uses print statements for output, only logs success
        mock_logger.info.assert_called_with("Database information displayed successfully")
        # Verify print was called for console output
        assert mock_print.call_count > 0
    
    def test_display_database_info_with_no_tables(self, mock_database_connection, mock_logger):
        """Test database info display when no tables are available."""
        # Arrange
        mock_database_connection.dialect.name = 'mysql'
        mock_database_connection.get_usable_table_names.return_value = []
        mock_database_connection.get_table_info.return_value = ""
        
        # Act
        with patch('app.db.database_manager.logger', mock_logger), \
             patch('builtins.print') as mock_print:
            display_database_info(mock_database_connection)
        
        # Assert - function uses print statements, only logs success
        mock_logger.info.assert_called_with("Database information displayed successfully")
        # Verify print was called for console output
        assert mock_print.call_count > 0
    
    def test_display_database_info_error_handling(self, mock_logger):
        """Test error handling in database info display."""
        # Arrange
        mock_db = Mock()
        mock_db.dialect.name = 'mysql'
        mock_db.get_usable_table_names.side_effect = Exception("Table access error")
        
        # Act
        with patch('app.db.database_manager.logger', mock_logger), \
             patch('builtins.print') as mock_print:
            display_database_info(mock_db)  # Function handles errors gracefully
        
        # Assert
        mock_logger.error.assert_called_once_with("Failed to display database information: Table access error")
        # Verify error was also printed to console
        assert mock_print.call_count > 0


class TestDatabaseManagerEdgeCases:
    """Test edge cases and error scenarios for database manager."""
    
    def test_create_uri_with_empty_values(self):
        """Test database URI creation with empty values."""
        # Arrange
        db_config = {
            'DB_USER': '',
            'DB_PASSWORD': 'pass',
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'test'
        }
        
        # Act
        result = create_database_uri(db_config)
        
        # Assert
        assert result.startswith('mysql+pymysql://:')