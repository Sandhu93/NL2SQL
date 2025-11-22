"""
Module: test_nl2sql_service.py
Description: Unit tests for NL2SQL service business logic
Dependencies: pytest, unittest.mock
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

from app.services.nl2sql_service import NL2SQLService


class TestNL2SQLService:
    """Test NL2SQL service functionality."""
    
    @pytest.fixture
    def nl2sql_service(self, mock_sql_database):
        """Create NL2SQLService instance for testing."""
        with patch('app.services.nl2sql_service.ChatOpenAI') as mock_openai, \
             patch('app.services.nl2sql_service.create_sql_query_chain') as mock_chain, \
             patch('app.services.nl2sql_service.QuerySQLDataBaseTool') as mock_tool, \
             patch('app.services.nl2sql_service.logger') as mock_logger:
            
            mock_openai.return_value = MagicMock()
            mock_chain.return_value = MagicMock()
            mock_tool.return_value = MagicMock()
            mock_logger.info = MagicMock()  # Mock logger to prevent logging conflicts
            
            service = NL2SQLService(
                db=mock_sql_database,
                model_name="gpt-3.5-turbo"
            )
            service.llm = mock_openai.return_value
            return service
    
    def test_generate_sql_success(self, nl2sql_service):
        """Test successful SQL generation from natural language."""
        # Arrange
        question = "What is the price of 1968 Ford Mustang?"
        expected_sql = "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'"
        
        # Mock the query chain to return SQL string
        nl2sql_service.query_chain = MagicMock()
        nl2sql_service.query_chain.invoke.return_value = expected_sql
        
        # Act
        result = nl2sql_service.generate_sql(question)
        
        # Assert
        assert isinstance(result, str)
        assert result == expected_sql
        nl2sql_service.query_chain.invoke.assert_called_once_with({"question": question})
    
    def test_generate_sql_empty_question(self, nl2sql_service):
        """Test SQL generation with empty question."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            nl2sql_service.generate_sql("")
        
        assert "question cannot be empty" in str(exc_info.value).lower()
    
    def test_generate_sql_whitespace_only_question(self, nl2sql_service):
        """Test SQL generation with whitespace-only question."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            nl2sql_service.generate_sql("   \n\t   ")
        
        assert "question cannot be empty" in str(exc_info.value).lower()
    
    def test_execute_query_success(self, nl2sql_service):
        """Test successful query execution."""
        # Arrange
        sql_query = "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'"
        expected_result = "buyPrice: 95.34"
        
        # Mock the query executor tool
        nl2sql_service.query_executor = MagicMock()
        nl2sql_service.query_executor.invoke.return_value = expected_result
        
        # Act
        result = nl2sql_service.execute_query(sql_query)
        
        # Assert
        assert result == expected_result
        nl2sql_service.query_executor.invoke.assert_called_once_with(sql_query)
    
    def test_process_question_end_to_end(self, nl2sql_service):
        """Test complete question processing pipeline."""
        # Arrange
        question = "What is the price of 1968 Ford Mustang?"
        expected_sql = "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'"
        expected_result = "buyPrice: 95.34"
        
        # Mock both the query chain and executor
        nl2sql_service.query_chain = MagicMock()
        nl2sql_service.query_chain.invoke.return_value = expected_sql
        nl2sql_service.query_executor = MagicMock()
        nl2sql_service.query_executor.invoke.return_value = expected_result
        
        # Act
        result = nl2sql_service.process_question(question)
        
        # Assert - actual method returns tuple (sql, result)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == expected_sql
        assert result[1] == expected_result
    
    def test_process_question_with_sql_generation_error(self, nl2sql_service):
        """Test question processing when SQL generation fails."""
        # Arrange
        question = "What is the price of 1968 Ford Mustang?"
        nl2sql_service.query_chain = MagicMock()
        nl2sql_service.query_chain.invoke.side_effect = Exception("OpenAI API error")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            nl2sql_service.process_question(question)
        
        assert "OpenAI API error" in str(exc_info.value)
    
    def test_process_question_with_database_error(self, nl2sql_service):
        """Test question processing when database execution fails."""
        # Arrange
        question = "What is the price of 1968 Ford Mustang?"
        expected_sql = "SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'"
        
        # Mock successful SQL generation but failed execution
        nl2sql_service.query_chain = MagicMock()
        nl2sql_service.query_chain.invoke.return_value = expected_sql
        nl2sql_service.query_executor = MagicMock()
        nl2sql_service.query_executor.invoke.side_effect = Exception("Database connection error")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            nl2sql_service.process_question(question)
        
        assert "Database connection error" in str(exc_info.value)


class TestNL2SQLServiceErrorHandling:
    """Test error handling scenarios for NL2SQL service."""
    
    def test_service_initialization_with_invalid_model(self, mock_sql_database):
        """Test service initialization with invalid model name."""
        # Act & Assert
        with patch('app.services.nl2sql_service.ChatOpenAI') as mock_openai, \
             patch('app.services.nl2sql_service.create_sql_query_chain') as mock_chain, \
             patch('app.services.nl2sql_service.QuerySQLDataBaseTool') as mock_tool, \
             patch('app.services.nl2sql_service.logger') as mock_logger:
            
            mock_openai.side_effect = ValueError("Invalid model name")
            
            with pytest.raises(ValueError):
                NL2SQLService(db=mock_sql_database, model_name="invalid-model")
    
    def test_generate_sql_with_api_timeout(self, mock_sql_database):
        """Test SQL generation with API timeout."""
        # Arrange
        question = "What is the price of 1968 Ford Mustang?"
        
        with patch('app.services.nl2sql_service.ChatOpenAI') as mock_openai, \
             patch('app.services.nl2sql_service.create_sql_query_chain') as mock_chain, \
             patch('app.services.nl2sql_service.QuerySQLDataBaseTool') as mock_tool, \
             patch('app.services.nl2sql_service.logger') as mock_logger:
            
            mock_openai.return_value = MagicMock()
            mock_chain.return_value = MagicMock()
            mock_tool.return_value = MagicMock()
            mock_logger.info = MagicMock()
            
            service = NL2SQLService(db=mock_sql_database, model_name="gpt-3.5-turbo")
            service.query_chain.invoke.side_effect = TimeoutError("API timeout")
        
        # Act & Assert
        with pytest.raises(TimeoutError):
            service.generate_sql(question)
    
    def test_process_question_with_malformed_response(self, mock_sql_database):
        """Test question processing with malformed LLM response."""
        # Arrange
        question = "What is the price of 1968 Ford Mustang?"
        
        with patch('app.services.nl2sql_service.ChatOpenAI') as mock_openai, \
             patch('app.services.nl2sql_service.create_sql_query_chain') as mock_chain, \
             patch('app.services.nl2sql_service.QuerySQLDataBaseTool') as mock_tool, \
             patch('app.services.nl2sql_service.logger') as mock_logger:
            
            mock_openai.return_value = MagicMock()
            mock_chain.return_value = MagicMock()
            mock_tool.return_value = MagicMock()
            mock_logger.info = MagicMock()
            
            service = NL2SQLService(db=mock_sql_database, model_name="gpt-3.5-turbo")
            service.query_chain.invoke.side_effect = Exception("Malformed response")
        
        # Act & Assert
        with pytest.raises(Exception):
            service.process_question(question)
