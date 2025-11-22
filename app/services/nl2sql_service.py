"""
Module: nl2sql_service.py
Description: Natural Language to SQL conversion service
Dependencies: langchain, langchain-openai, langchain-community
Author: AI Agent
Created: 2025-11-22
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Standard library imports
import logging
from typing import List, Optional

# LangChain / OpenAI imports
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

# Constants
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0
TEST_QUERIES = [
    "Show me all products priced above $100",
    "How many customers are there?",
    "List all offices in the USA"
]

logger = logging.getLogger(__name__)


class NL2SQLService:
    """
    Service for converting natural language queries to SQL and executing them.
    
    This class provides functionality to translate human-readable queries 
    into proper SQL syntax using OpenAI's language models.
    
    Attributes:
        llm: ChatOpenAI language model instance
        db: SQLDatabase connection object
        query_chain: SQL query generation chain
        query_executor: SQL query execution tool
    """
    
    def __init__(self, db: SQLDatabase, model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE):
        """
        Initialize NL2SQL service with database connection and model configuration.
        
        Args:
            db: SQLDatabase connection object
            model_name: OpenAI model name to use
            temperature: Model temperature setting (0.0 to 1.0)
        """
        self.db = db
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.query_chain = create_sql_query_chain(self.llm, self.db)
        self.query_executor = QuerySQLDataBaseTool(db=self.db)
        
        logger.info(f"NL2SQL service initialized with model: {model_name}")
    
    def generate_sql(self, question: str) -> str:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            str: Generated SQL query
            
        Raises:
            ValueError: If question is empty or invalid
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            logger.info(f"Generating SQL for question: {question[:100]}...")
            generated_query = self.query_chain.invoke({"question": question})
            logger.info("SQL query generated successfully")
            return generated_query
        except Exception as e:
            logger.error(f"Failed to generate SQL query: {e}")
            raise
    
    def execute_query(self, sql_query: str) -> str:
        """
        Execute SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            str: Query execution results
            
        Raises:
            ValueError: If SQL query is empty
        """
        if not sql_query or not sql_query.strip():
            raise ValueError("SQL query cannot be empty")
        
        try:
            logger.info(f"Executing SQL query: {sql_query[:100]}...")
            result = self.query_executor.invoke(sql_query)
            logger.info("SQL query executed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to execute SQL query: {e}")
            raise
    
    def process_question(self, question: str) -> tuple:
        """
        Process natural language question and return both SQL and results.
        
        Args:
            question: Natural language question
            
        Returns:
            tuple: (generated_sql, execution_result)
        """
        sql_query = self.generate_sql(question)
        result = self.execute_query(sql_query)
        return sql_query, result


def execute_first_query(db: SQLDatabase) -> None:
    """
    Execute the first query test to demonstrate NL2SQL functionality.
    
    Args:
        db: SQLDatabase connection object
    """
    try:
        logger.info("Starting first query test")
        
        # Initialize NL2SQL service
        nl2sql_service = NL2SQLService(db)
        
        # Test query: Find price of 1968 Ford Mustang
        test_question = "what is price of `1968 Ford Mustang`"
        logger.info(f"Processing question: {test_question}")
        
        print("\n" + "="*60)
        print("FIRST QUERY TEST - NL2SQL CONVERSION")
        print("="*60)
        print(f"Question: {test_question}")
        print("-"*60)
        
        # Process the question
        generated_query, result = nl2sql_service.process_question(test_question)
        
        print("Generated SQL Query:")
        print(generated_query)
        print("-"*60)
        print("Query Results:")
        print(result)
        print("="*60)
        
        logger.info("First query test completed successfully")
        
        # Execute additional test queries
        _execute_additional_queries(nl2sql_service)
        
    except Exception as e:
        logger.error(f"First query test failed: {e}")
        print(f"Error during first query test: {e}")


def _execute_additional_queries(nl2sql_service: NL2SQLService) -> None:
    """
    Execute additional test queries to demonstrate functionality.
    
    Args:
        nl2sql_service: Initialized NL2SQL service instance
    """
    print("\n" + "="*60)
    print("ADDITIONAL QUERY TESTS")
    print("="*60)
    
    for i, question in enumerate(TEST_QUERIES, 1):
        try:
            print(f"\n{i}. Question: {question}")
            print("-" * 50)
            
            # Process the question
            sql_query, result = nl2sql_service.process_question(question)
            
            print(f"Generated SQL: {sql_query.strip()}")
            print(f"Result: {result}")
            
        except Exception as e:
            logger.error(f"Failed to process query {i}: {e}")
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    logger.info("Additional query tests completed")