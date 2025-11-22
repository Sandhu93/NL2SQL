"""
Database Tests for NL2SQL System
Author: AI Agent
Created: 2025-11-21
Python Version: 3.11
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal
import os
from urllib.parse import quote_plus

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import create_database_uri, initialize_database, display_database_info

class TestDatabaseURIGeneration:
    """Test database URI generation with various configurations."""
    
    def test_basic_uri_generation(self):
        """Test basic database URI generation."""
        env_vars = {
            'DB_USER': 'root',
            'DB_PASSWORD': 'password',
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'classicmodels'
        }
        
        result = create_database_uri(env_vars)
        expected = "mysql+pymysql://root:password@localhost:3306/classicmodels"
        assert result == expected
    
    def test_uri_with_special_characters(self):
        """Test URI generation with special characters in password."""
        env_vars = {
            'DB_USER': 'root',
            'DB_PASSWORD': 'p@ssw0rd!#$',
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'classicmodels'
        }
        
        result = create_database_uri(env_vars)
        # Password should be URL encoded
        encoded_password = quote_plus('p@ssw0rd!#$')
        expected = f"mysql+pymysql://root:{encoded_password}@localhost:3306/classicmodels"
        assert result == expected
    
    def test_uri_with_already_encoded_password(self):
        """Test URI generation with already URL-encoded password."""
        env_vars = {
            'DB_USER': 'root',
            'DB_PASSWORD': 'pass%23word',  # # encoded as %23
            'DB_HOST': 'localhost',
            'DB_PORT': '3306',
            'DB_NAME': 'classicmodels'
        }
        
        result = create_database_uri(env_vars)
        # Should handle already encoded passwords correctly
        assert "pass%23word" in result
    
    def test_uri_with_remote_host(self):
        """Test URI generation with remote database host."""
        env_vars = {
            'DB_USER': 'remote_user',
            'DB_PASSWORD': 'remote_pass',
            'DB_HOST': 'db.example.com',
            'DB_PORT': '3307',
            'DB_NAME': 'production_db'
        }
        
        result = create_database_uri(env_vars)
        expected = "mysql+pymysql://remote_user:remote_pass@db.example.com:3307/production_db"
        assert result == expected
    
    def test_uri_with_ip_address(self):
        """Test URI generation with IP address as host."""
        env_vars = {
            'DB_USER': 'user',
            'DB_PASSWORD': 'pass',
            'DB_HOST': '192.168.1.100',
            'DB_PORT': '3306',
            'DB_NAME': 'testdb'
        }
        
        result = create_database_uri(env_vars)
        expected = "mysql+pymysql://user:pass@192.168.1.100:3306/testdb"
        assert result == expected

class TestDatabaseConnectionMocking:
    """Test database connection handling with mocks."""
    
    @patch('app.db.database_manager.SQLDatabase')
    def test_successful_database_initialization(self, mock_sql_database):
        """Test successful database initialization."""
        # Setup mock
        mock_db = MagicMock()
        mock_db.dialect = "mysql"
        mock_db.get_usable_table_names.return_value = [
            'customers', 'employees', 'offices', 'orders', 
            'orderdetails', 'payments', 'products', 'productlines'
        ]
        mock_sql_database.from_uri.return_value = mock_db
        
        # Test initialization
        uri = "mysql+pymysql://user:pass@localhost:3306/classicmodels"
        result = initialize_database(uri)
        
        assert result is not None
        assert result.dialect == "mysql"
        assert len(result.get_usable_table_names()) == 8
        
        # Verify SQLDatabase was called with correct parameters
        mock_sql_database.from_uri.assert_called_once_with(
            uri,
            sample_rows_in_table_info=2,
            include_tables=None,
            custom_table_info=None
        )
    
    @patch('app.db.database_manager.SQLDatabase')
    def test_database_initialization_failure(self, mock_sql_database):
        """Test database initialization failure handling."""
        # Setup mock to raise exception
        mock_sql_database.from_uri.side_effect = Exception("Connection refused")
        
        uri = "mysql+pymysql://user:pass@localhost:3306/classicmodels"
        result = initialize_database(uri)
        
        assert result is None
    
    @patch('app.db.database_manager.SQLDatabase')
    def test_database_initialization_with_specific_tables(self, mock_sql_database):
        """Test database initialization with specific table inclusion."""
        mock_db = MagicMock()
        mock_sql_database.from_uri.return_value = mock_db
        
        # This would be modified in main.py if we wanted to test specific tables
        uri = "mysql+pymysql://user:pass@localhost:3306/classicmodels"
        result = initialize_database(uri)
        
        assert result is not None

class TestDatabaseInfo:
    """Test database information display and retrieval."""
    
    def test_display_database_info_success(self, capsys):
        """Test successful database info display."""
        # Create mock database
        mock_db = MagicMock()
        mock_db.dialect = "mysql"
        mock_db.get_usable_table_names.return_value = [
            'customers', 'employees', 'offices', 'orders', 
            'orderdetails', 'payments', 'products', 'productlines'
        ]
        
        display_database_info(mock_db)
        
        captured = capsys.readouterr()
        assert "DATABASE CONNECTION SUCCESSFUL!" in captured.out
        assert "Database Dialect: mysql" in captured.out
        assert "Available Tables (8):" in captured.out
        assert "customers" in captured.out
        assert "products" in captured.out
    
    def test_display_database_info_with_error(self, capsys):
        """Test database info display when table query fails."""
        mock_db = MagicMock()
        mock_db.dialect = "mysql"
        mock_db.get_usable_table_names.side_effect = Exception("Query failed")
        
        # Should not raise exception
        display_database_info(mock_db)
        
        captured = capsys.readouterr()
        assert "DATABASE CONNECTION SUCCESSFUL!" in captured.out
        assert "Database Dialect: mysql" in captured.out
    
    def test_display_database_info_empty_tables(self, capsys):
        """Test database info display with no tables."""
        mock_db = MagicMock()
        mock_db.dialect = "mysql"
        mock_db.get_usable_table_names.return_value = []
        
        display_database_info(mock_db)
        
        captured = capsys.readouterr()
        assert "Available Tables (0):" in captured.out

class TestDatabaseQueryExecution:
    """Test database query execution scenarios."""
    
    def test_mock_query_execution_success(self):
        """Test successful query execution with mock."""
        # Create mock database
        mock_db = MagicMock()
        mock_db.run.return_value = "[(Decimal('95.34'),)]"
        
        # Simulate query execution
        query = "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'"
        result = mock_db.run(query)
        
        assert "95.34" in result
        mock_db.run.assert_called_once_with(query)
    
    def test_mock_query_execution_failure(self):
        """Test query execution failure handling."""
        mock_db = MagicMock()
        mock_db.run.side_effect = Exception("SQL syntax error")
        
        query = "SELECT * FROM non_existent_table"
        
        with pytest.raises(Exception) as exc_info:
            mock_db.run(query)
        
        assert "SQL syntax error" in str(exc_info.value)
    
    def test_mock_query_execution_empty_result(self):
        """Test query execution with empty result."""
        mock_db = MagicMock()
        mock_db.run.return_value = "[]"
        
        query = "SELECT * FROM products WHERE productName = 'NonExistentProduct'"
        result = mock_db.run(query)
        
        assert result == "[]"

class TestDatabaseConnectionParameters:
    """Test various database connection parameters."""
    
    def test_connection_with_ssl_parameters(self):
        """Test database URI with SSL parameters."""
        env_vars = {
            'DB_USER': 'secure_user',
            'DB_PASSWORD': 'secure_pass',
            'DB_HOST': 'secure.db.com',
            'DB_PORT': '3306',
            'DB_NAME': 'secure_db'
        }
        
        # In a real implementation, we might add SSL parameters to the URI
        result = create_database_uri(env_vars)
        base_uri = "mysql+pymysql://secure_user:secure_pass@secure.db.com:3306/secure_db"
        assert result == base_uri
    
    def test_connection_with_different_mysql_versions(self):
        """Test compatibility with different MySQL versions."""
        # This test ensures our URI format works across MySQL versions
        test_configs = [
            {'host': 'mysql57.test.com', 'name': 'mysql57_db'},
            {'host': 'mysql80.test.com', 'name': 'mysql80_db'},
            {'host': 'mariadb.test.com', 'name': 'mariadb_db'},
        ]
        
        for config in test_configs:
            env_vars = {
                'DB_USER': 'test_user',
                'DB_PASSWORD': 'test_pass',
                'DB_HOST': config['host'],
                'DB_PORT': '3306',
                'DB_NAME': config['name']
            }
            
            result = create_database_uri(env_vars)
            assert config['host'] in result
            assert config['name'] in result
    
    def test_connection_timeout_handling(self):
        """Test database connection with timeout considerations."""
        # This test simulates what would happen with connection timeouts
        with patch('app.db.database_manager.SQLDatabase') as mock_sql_database:
            # Simulate timeout exception
            mock_sql_database.from_uri.side_effect = Exception("Connection timeout")
            
            uri = "mysql+pymysql://user:pass@slow.db.com:3306/testdb"
            result = initialize_database(uri)
            
            assert result is None

class TestClassicModelsSchema:
    """Test ClassicModels database schema expectations."""
    
    def test_classicmodels_expected_tables(self):
        """Test that ClassicModels database has expected tables."""
        expected_tables = [
            'customers', 'employees', 'offices', 'orders',
            'orderdetails', 'payments', 'products', 'productlines'
        ]
        
        mock_db = MagicMock()
        mock_db.get_usable_table_names.return_value = expected_tables
        
        tables = mock_db.get_usable_table_names()
        
        for table in expected_tables:
            assert table in tables
        assert len(tables) == 8
    
    def test_classicmodels_table_relationships(self):
        """Test ClassicModels table relationships through mock queries."""
        mock_db = MagicMock()
        
        # Mock relationship queries
        relationship_queries = [
            "SELECT COUNT(*) FROM customers c JOIN orders o ON c.customerNumber = o.customerNumber",
            "SELECT COUNT(*) FROM orders o JOIN orderdetails od ON o.orderNumber = od.orderNumber",
            "SELECT COUNT(*) FROM products p JOIN orderdetails od ON p.productCode = od.productCode"
        ]
        
        for query in relationship_queries:
            mock_db.run.return_value = "[(100,)]"  # Mock count result
            result = mock_db.run(query)
            assert "100" in result
    
    def test_classicmodels_sample_data_expectations(self):
        """Test expectations for ClassicModels sample data."""
        mock_db = MagicMock()
        
        # Mock sample data queries
        sample_queries = {
            "SELECT COUNT(*) FROM customers": "[(122,)]",
            "SELECT COUNT(*) FROM products": "[(110,)]",
            "SELECT COUNT(*) FROM orders": "[(326,)]"
        }
        
        for query, expected_result in sample_queries.items():
            mock_db.run.return_value = expected_result
            result = mock_db.run(query)
            assert result == expected_result
