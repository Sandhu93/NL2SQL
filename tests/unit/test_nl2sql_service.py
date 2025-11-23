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

from app.services.nl2sql_service import (
    NL2SQLService,
    execute_refined_query,
    interactive_query_runner,
    REFINED_TEST_QUESTION,
)


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

    def test_rephrase_answer_success(self, nl2sql_service):
        """Test rephrasing pipeline returns a natural language answer."""
        expected_answer = "There are 2 customers with an order count over 5."
        mock_pipeline = MagicMock()
        mock_pipeline.__or__.return_value = mock_pipeline
        mock_pipeline.invoke.return_value = expected_answer
        
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_pipeline
        
        with patch('app.services.nl2sql_service.ANSWER_PROMPT', mock_prompt), \
             patch('app.services.nl2sql_service.StrOutputParser') as mock_parser:
            mock_parser.return_value = MagicMock()
            answer = nl2sql_service.rephrase_answer(
                question="How many customers have an order count greater than 5?",
                sql_query="SELECT COUNT(*) FROM customers WHERE orderCount > 5",
                result=[{"count": 2}]
            )
        
        assert answer == expected_answer
        mock_pipeline.invoke.assert_called_once()

    def test_process_question_rephrased(self, nl2sql_service):
        """Test end-to-end refined flow returns sql, raw result, and answer."""
        question = "List all offices in the USA"
        expected_sql = "SELECT * FROM offices WHERE country = 'USA'"
        expected_result = [{"officeCode": "1", "city": "NYC"}]
        expected_answer = "There is 1 office in the USA: NYC."
        
        nl2sql_service.generate_sql = MagicMock(return_value=expected_sql)
        nl2sql_service.execute_query = MagicMock(return_value=expected_result)
        nl2sql_service.rephrase_answer = MagicMock(return_value=expected_answer)
        
        response = nl2sql_service.process_question_rephrased(question)
        
        assert response["sql"] == expected_sql
        assert response["result"] == expected_result
        assert response["answer"] == expected_answer
        nl2sql_service.generate_sql.assert_called_once_with(question)
        nl2sql_service.execute_query.assert_called_once_with(expected_sql)
        nl2sql_service.rephrase_answer.assert_called_once_with(
            question, expected_sql, expected_result
        )

    def test_generate_sql_with_examples(self, nl2sql_service):
        """Test SQL generation with few-shot prompt."""
        question = "How many customers have an order count greater than 5?"
        expected_sql = "SELECT COUNT(*) FROM customers WHERE orderCount > 5;"
        
        mock_prompt = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_sql
        chained_once = MagicMock()
        chained_once.__or__.return_value = mock_chain
        mock_prompt.__or__.return_value = chained_once
        nl2sql_service.build_few_shot_prompt = MagicMock(return_value=mock_prompt)
        
        nl2sql_service._select_relevant_tables = MagicMock(return_value=["customers"])
        nl2sql_service._format_table_context = MagicMock(return_value="table info")
        nl2sql_service._build_semantic_selector = MagicMock()
        
        with patch('app.services.nl2sql_service.StrOutputParser') as mock_parser:
            mock_parser.return_value = MagicMock()
            sql_query = nl2sql_service.generate_sql_with_examples(question)
        
        assert sql_query == expected_sql
        mock_prompt.__or__.assert_called()
        mock_chain.invoke.assert_called_once()
        nl2sql_service._select_relevant_tables.assert_called_once_with(question, top_k=2)
        nl2sql_service._format_table_context.assert_called_once_with(["customers"])
        nl2sql_service._build_semantic_selector.assert_called_once()


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


class TestExecuteRefinedQuery:
    """Test refined query demonstration helper."""
    
    def test_execute_refined_query_calls_service_methods(self, mock_sql_database):
        """Ensure refined demo delegates to service methods."""
        refined_result = {
            "sql": "SELECT COUNT(*) FROM customers WHERE orderCount > 5",
            "result": [{"count": 2}],
            "answer": "There are 2 customers with an order count over 5."
        }
        with patch('app.services.nl2sql_service.NL2SQLService') as mock_service_cls, \
             patch('app.services.nl2sql_service.logger') as mock_logger, \
             patch('builtins.print') as mock_print:
            
            mock_service = MagicMock()
            mock_service.process_question_rephrased.return_value = refined_result
            mock_service.rephrase_answer.return_value = refined_result["answer"]
            mock_service_cls.return_value = mock_service
            
            execute_refined_query(mock_sql_database)
        
        mock_service.process_question_rephrased.assert_called_once_with(
            REFINED_TEST_QUESTION
        )
        mock_service.rephrase_answer.assert_called_once_with(
            REFINED_TEST_QUESTION,
            refined_result["sql"],
            refined_result["result"]
        )
        mock_logger.info.assert_called()
        mock_print.assert_called()


class TestInteractiveQueryRunner:
    """Test interactive menu handler."""
    
    def test_interactive_exit_immediately(self, mock_sql_database):
        """Ensure interactive runner exits cleanly when user chooses exit."""
        with patch('app.services.nl2sql_service.NL2SQLService') as mock_service_cls, \
             patch('app.services.nl2sql_service.logger') as mock_logger, \
             patch('builtins.print'), \
             patch('builtins.input', side_effect=["4"]):
            
            interactive_query_runner(mock_sql_database)
        
        mock_service_cls.assert_called_once_with(mock_sql_database)
        mock_logger.info.assert_called()

    def test_interactive_uses_default_question_when_empty(self, mock_sql_database):
        """Ensure blank input falls back to default refined test question."""
        refined_result = {
            "sql": "SELECT COUNT(*) FROM customers WHERE orderCount > 5",
            "result": [{"count": 2}],
            "answer": "There are 2 customers with an order count over 5."
        }
        with patch('app.services.nl2sql_service.NL2SQLService') as mock_service_cls, \
             patch('app.services.nl2sql_service.logger') as mock_logger, \
             patch('builtins.print'), \
             patch('builtins.input', side_effect=["2", "", "exit"]):
            
            mock_service = MagicMock()
            mock_service.process_question_rephrased.return_value = refined_result
            mock_service_cls.return_value = mock_service
            
            interactive_query_runner(mock_sql_database)
        
        mock_service.process_question_rephrased.assert_called_once_with(REFINED_TEST_QUESTION)
        mock_logger.info.assert_called()

    def test_interactive_few_shot_option(self, mock_sql_database):
        """Ensure few-shot path is invoked via menu option 3."""
        few_shot_result = {
            "sql": "SELECT COUNT(*) FROM customers WHERE orderCount > 5",
            "result": [{"count": 2}],
            "answer": "There are 2 customers with an order count over 5."
        }
        with patch('app.services.nl2sql_service.NL2SQLService') as mock_service_cls, \
             patch('app.services.nl2sql_service.logger') as mock_logger, \
             patch('builtins.print'), \
             patch('builtins.input', side_effect=["3", "question here", "exit"]):
            
            mock_service = MagicMock()
            mock_service.process_question_few_shot.return_value = few_shot_result
            mock_service_cls.return_value = mock_service
            
            interactive_query_runner(mock_sql_database)
        
        mock_service.process_question_few_shot.assert_called_once()
        mock_logger.info.assert_called()
