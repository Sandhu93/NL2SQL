"""
Module: conftest.py
Description: Shared pytest fixtures and configuration for all tests
Dependencies: pytest, unittest.mock
Author: AI Assistant
Created: 2025-11-21
Last Modified: 2025-11-22
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import tempfile
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test constants
TEST_ENV_CONFIG = {
    'DB_USER': 'test_user',
    'DB_PASSWORD': 'test_password%23',
    'DB_HOST': 'localhost',
    'DB_PORT': '3306',
    'DB_NAME': 'test_classicmodels',
    'OPENAI_API_KEY': 'sk-test-key-12345',
    'LANGSMITH_API_KEY': 'test_langsmith_key',
    'LANGSMITH_TRACING': 'false'
}

SAMPLE_TABLE_INFO = """
CREATE TABLE customers (
    customerNumber INT PRIMARY KEY,
    customerName VARCHAR(50) NOT NULL,
    contactLastName VARCHAR(50) NOT NULL,
    city VARCHAR(50)
);

CREATE TABLE products (
    productCode VARCHAR(15) PRIMARY KEY,
    productName VARCHAR(70) NOT NULL,
    buyPrice DECIMAL(10,2) NOT NULL
);
"""

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, TEST_ENV_CONFIG, clear=True):
        yield TEST_ENV_CONFIG

@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    temp_dir = tempfile.mkdtemp()
    env_file = Path(temp_dir) / '.env'
    
    env_content = '\n'.join([f"{key}={value}" for key, value in TEST_ENV_CONFIG.items()])
    env_file.write_text(env_content)
    
    yield str(env_file)
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {'buyPrice': 95.34, 'productName': '1968 Ford Mustang'}
    ]
    return mock_conn

@pytest.fixture
def mock_database_manager():
    """Mock DatabaseManager for testing."""
    mock_manager = MagicMock()
    mock_manager.get_table_info.return_value = SAMPLE_TABLE_INFO
    mock_manager.get_table_names.return_value = ['customers', 'products', 'orders']
    mock_manager.execute_query.return_value = [
        {'buyPrice': 95.34, 'productName': '1968 Ford Mustang'}
    ]
    return mock_manager

@pytest.fixture
def mock_sql_database():
    """Mock SQLDatabase (LangChain) for testing."""
    from langchain_community.utilities.sql_database import SQLDatabase
    
    mock_db = MagicMock(spec=SQLDatabase)
    mock_db.run.return_value = "[(1, 'test')]"
    mock_db.get_usable_table_names.return_value = ["customers", "products", "orders"]
    mock_db.get_table_info.return_value = SAMPLE_TABLE_INFO
    mock_db.dialect = "sqlite"
    
    return mock_db

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""
    return MagicMock(
        content="SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'"
    )

@pytest.fixture
def mock_nl2sql_service():
    """Mock NL2SQLService for testing."""
    from app.services.nl2sql_service import NL2SQLService
    
    mock_service = MagicMock(spec=NL2SQLService)
    mock_service.generate_sql.return_value = {
        'sql': "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'",
        'confidence': 0.9,
        'tables_used': ['products']
    }
    mock_service.execute_query.return_value = [
        {'buyPrice': 95.34, 'productName': '1968 Ford Mustang'}
    ]
    return mock_service

@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {
            'question': 'What is the price of 1968 Ford Mustang?',
            'expected_sql': "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'",
            'expected_result': [{'buyPrice': 95.34}]
        },
        {
            'question': 'How many customers are there?',
            'expected_sql': 'SELECT COUNT(*) as customer_count FROM customers',
            'expected_result': [{'customer_count': 122}]
        },
        {
            'question': 'Show me all products priced above $100',
            'expected_sql': 'SELECT * FROM products WHERE buyPrice > 100.00',
            'expected_result': [
                {'productName': '1962 Lancia Delta 16V', 'buyPrice': 103.42},
                {'productName': '1998 Chrysler Plymouth Prowler', 'buyPrice': 101.51}
            ]
        }
    ]

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger

@pytest.fixture
def valid_sql_queries():
    """Valid SQL queries for testing."""
    return [
        "SELECT * FROM customers",
        "SELECT customerName, city FROM customers WHERE city = 'Paris'",
        "SELECT COUNT(*) FROM products",
        "SELECT p.productName, p.buyPrice FROM products p WHERE p.buyPrice > 50.00"
    ]

@pytest.fixture
def invalid_sql_queries():
    """Invalid SQL queries for testing."""
    return [
        "DROP TABLE customers",
        "DELETE FROM products WHERE buyPrice > 100",
        "INSERT INTO customers VALUES (1, 'Test')",
        "UPDATE products SET buyPrice = 0",
        "ALTER TABLE customers ADD COLUMN test VARCHAR(50)",
        "CREATE TABLE test (id INT)",
        "TRUNCATE TABLE orders"
    ]

@pytest.fixture
def non_select_queries():
    """Non-SELECT queries for testing."""
    return [
        "UPDATE customers SET city = 'London' WHERE customerNumber = 1",
        "INSERT INTO products (productCode, productName, buyPrice) VALUES ('TEST', 'Test Product', 99.99)",
        "DELETE FROM orders WHERE orderNumber = 1"
    ]