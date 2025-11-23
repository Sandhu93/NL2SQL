"""
Module: nl2sql_service.py
Description: Natural Language to SQL conversion service
Dependencies: langchain, langchain-openai, langchain-community
Author: Sandeep B Kadam
Created: 2025-11-22
Last Modified: 2025-11-22
Python Version: 3.11
"""

# Standard library imports
import csv
import logging
import os
from operator import itemgetter
from typing import List, Optional

# LangChain / OpenAI imports
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Constants
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0
TEST_QUERIES = [
    "Show me all products priced above $100",
    "How many customers are there?",
    "List all offices in the USA"
]
REFINED_TEST_QUESTION = "How many customers have an order count greater than 5?"
FEW_SHOT_EXAMPLES = [
    {
        "input": "List all customers in France with a credit limit over 20,000.",
        "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;",
    },
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;",
    },
    {
        "input": "How many products cost more than $100?",
        "query": "SELECT COUNT(*) FROM products WHERE buyPrice > 100;",
    },
    {
        "input": "Show the names of employees in the sales department.",
        "query": "SELECT firstName, lastName FROM employees WHERE jobTitle LIKE '%Sales%';",
    },
]
TABLE_METADATA_CSV = os.getenv("TABLE_METADATA_CSV", "classicmodels_tables_llm.csv")
SYSTEM_TABLE_PROMPT = (
    "Return the names of ALL the SQL tables that MIGHT be relevant to the user question. "
    "Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure they are needed.\n\n"
    "Table reference:\n{table_details}"
)
ANSWER_PROMPT = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""
)

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
        self.table_metadata = self._load_table_metadata(TABLE_METADATA_CSV)
        if os.getenv("TABLE_METADATA_CSV"):
            logger.info("Using TABLE_METADATA_CSV override from environment")
        self.rephrase_chain = (
            RunnablePassthrough.assign(query=self.query_chain).assign(
                result=itemgetter("query") | self.query_executor
            )
            | ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        logger.info(f"NL2SQL service initialized with model: {model_name}")

    def build_few_shot_prompt(
        self,
        examples: Optional[List[dict]] = None,
        use_dynamic_selector: bool = False,
        k: int = 2,
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate with few-shot examples for SQL generation.
        
        Args:
            examples: Optional list of example dicts with 'input' and 'query'.
            use_dynamic_selector: Whether to use semantic similarity selection.
            k: Number of examples to select when using dynamic selector.
        
        Returns:
            ChatPromptTemplate: Prompt configured with few-shot examples.
        """
        selected_examples = examples or FEW_SHOT_EXAMPLES
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}\nSQLQuery:"),
                ("ai", "{query}"),
            ]
        )
        
        if use_dynamic_selector:
            # Dynamic selector placeholder; enables future semantic similarity selection.
            selector = SemanticSimilarityExampleSelector.from_examples(
                selected_examples,
                OpenAIEmbeddings(),
                k=k,
                input_keys=["input"],
            )
            few_shot = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                example_selector=selector,
                input_variables=["input"],
            )
        else:
            few_shot = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=selected_examples,
                input_variables=["input"],
            )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a MySQL expert. Given an input question, create a syntactically "
                    "correct MySQL query to run. Use the provided examples as guidance.\n\n"
                    "Table context:\n{table_info}\n\nUse up to {top_k} relevant tables.",
                ),
                few_shot,
                ("human", "{input}"),
            ]
        )
        return prompt

    @staticmethod
    def _load_table_metadata(csv_path: str) -> List[dict]:
        """
        Load table metadata from a CSV file.
        
        Args:
            csv_path: Path to the CSV containing table metadata.
            
        Returns:
            List[dict]: List of table metadata dictionaries.
        """
        metadata: List[dict] = []
        try:
            with open(csv_path, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    metadata.append({
                        "table": row.get("table", "").strip(),
                        "description": row.get("description", "").strip(),
                        "important_columns": (
                            row.get("important_columns")
                            or row.get("important_coulmns")
                            or ""
                        ).strip(),
                    })
            logger.info("Loaded table metadata for dynamic selection")
        except FileNotFoundError:
            logger.warning(f"Table metadata CSV not found at path: {csv_path}")
        except Exception as exc:
            logger.error(f"Failed to load table metadata: {exc}")
        return metadata

    def _table_details_text(self) -> str:
        """
        Create table details text block for prompts from loaded metadata.
        
        Returns:
            str: Concatenated table details text.
        """
        if not self.table_metadata:
            return ""
        details: List[str] = []
        for row in self.table_metadata:
            table_name = row.get("table", "")
            description = row.get("description", "")
            important_cols = row.get("important_columns", "")
            section = f"Table Name: {table_name}\nTable Description: {description}"
            if important_cols:
                section += f"\nImportant Columns: {important_cols}"
            details.append(section)
        return "\n\n".join(details)

    def _select_relevant_tables(self, question: str, top_k: int = 3) -> List[str]:
        """
        Select relevant tables using LLM extraction over table metadata.
        
        Args:
            question: User natural language question.
            top_k: Number of tables to suggest.
            
        Returns:
            List[str]: Selected table names.
        """
        if not self.table_metadata:
            return list(self.db.get_usable_table_names())
        
        class Table(BaseModel):
            """Table in SQL database."""
            name: str = Field(description="Name of table in SQL database.")
        
        table_prompt = SYSTEM_TABLE_PROMPT.format(table_details=self._table_details_text())
        try:
            table_chain = create_extraction_chain_pydantic(
                Table,
                self.llm,
                system_message=table_prompt,
            )
            tables = table_chain.invoke({"input": question})
            table_names = [table.name for table in tables][:top_k]
            return table_names or list(self.db.get_usable_table_names())
        except Exception as exc:
            logger.error(f"Failed to select relevant tables: {exc}")
            return list(self.db.get_usable_table_names())

    def _format_table_context(self, table_names: List[str]) -> str:
        """
        Build table context string limited to selected tables.
        
        Args:
            table_names: List of table names to include.
            
        Returns:
            str: Formatted table context for prompts.
        """
        if not table_names:
            return self.db.get_table_info()
        if not self.table_metadata:
            return self.db.get_table_info()
        
        selected = [row for row in self.table_metadata if row.get("table") in table_names]
        if not selected:
            return self.db.get_table_info()
        
        details: List[str] = []
        for row in selected:
            table_name = row.get("table", "")
            description = row.get("description", "")
            important_cols = row.get("important_columns", "")
            section = f"Table Name: {table_name}\nTable Description: {description}"
            if important_cols:
                section += f"\nImportant Columns: {important_cols}"
            details.append(section)
        return "\n\n".join(details)
    
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

    def generate_sql_with_examples(
        self,
        question: str,
        use_dynamic_selector: bool = False,
        k: int = 2,
    ) -> str:
        """
        Generate SQL using few-shot examples to guide the model.
        
        Args:
            question: Natural language question
            use_dynamic_selector: Whether to select examples dynamically
            k: Number of examples to select when dynamic selection is enabled
        
        Returns:
            str: Generated SQL query
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        prompt = self.build_few_shot_prompt(
            use_dynamic_selector=use_dynamic_selector,
            k=k,
        )
        try:
            logger.info("Generating SQL with few-shot examples")
            chain = prompt | self.llm | StrOutputParser()
            table_names = self._select_relevant_tables(question, top_k=k)
            table_info = self._format_table_context(table_names)
            generated_query = chain.invoke(
                {"input": question, "table_info": table_info, "top_k": k}
            )
            logger.info("SQL query generated with few-shot examples")
            return generated_query
        except Exception as exc:
            logger.error(f"Failed to generate SQL with examples: {exc}")
            raise

    def process_question_few_shot(
        self,
        question: str,
        use_dynamic_selector: bool = False,
        k: int = 2,
    ) -> dict:
        """
        Generate SQL with few-shot examples, execute it, and return results.
        
        Args:
            question: Natural language question
            use_dynamic_selector: Whether to use dynamic example selection
            k: Number of examples to select when dynamic selection is enabled
        
        Returns:
            dict: Dictionary containing 'sql', 'result', and 'answer'
        """
        sql_query = self.generate_sql_with_examples(
            question,
            use_dynamic_selector=use_dynamic_selector,
            k=k,
        )
        result = self.execute_query(sql_query)
        answer = self.rephrase_answer(question, sql_query, result)
        return {"sql": sql_query, "result": result, "answer": answer}

    def rephrase_answer(self, question: str, sql_query: str, result: str) -> str:
        """
        Convert raw SQL results into a concise natural language answer.
        
        Args:
            question: Original user question
            sql_query: Generated SQL query
            result: Raw SQL execution result
            
        Returns:
            str: Rephrased, user-friendly answer
        """
        try:
            logger.info("Rephrasing SQL result for user-friendly answer")
            response = (ANSWER_PROMPT | self.llm | StrOutputParser()).invoke(
                {"question": question, "query": sql_query, "result": result}
            )
            logger.info("Rephrasing completed successfully")
            return response
        except Exception as e:
            logger.error(f"Failed to rephrase answer: {e}")
            raise

    def process_question_rephrased(self, question: str) -> dict:
        """
        Generate SQL, execute it, and return SQL, raw result, and rephrased answer.
        
        Args:
            question: Natural language question
            
        Returns:
            dict: Dictionary containing 'sql', 'result', and 'answer'
        """
        sql_query = self.generate_sql(question)
        result = self.execute_query(sql_query)
        answer = self.rephrase_answer(question, sql_query, result)
        return {"sql": sql_query, "result": result, "answer": answer}


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


def execute_refined_query(db: SQLDatabase) -> None:
    """
    Demonstrate refined NL2SQL flow with rephrased answer output.
    
    Args:
        db: SQLDatabase connection object
    """
    try:
        logger.info("Starting refined query demonstration")
        
        nl2sql_service = NL2SQLService(db)
        test_question = REFINED_TEST_QUESTION
        
        print("\n" + "="*60)
        print("REFINED QUERY TEST - NL2SQL WITH REPHRASED ANSWER")
        print("="*60)
        print(f"Question: {test_question}")
        print("-"*60)
        
        response = nl2sql_service.process_question_rephrased(test_question)
        
        print("Generated SQL Query:")
        print(response["sql"])
        print("-"*60)
        print("Raw Query Results:")
        print(response["result"])
        print("-"*60)
        print("Rephrased Answer:")
        print(response["answer"])
        
        direct_answer = nl2sql_service.rephrase_answer(
            test_question,
            response["sql"],
            response["result"]
        )
        print("-"*60)
        print("Direct Rephrase Call (tutorial-style):")
        print(direct_answer)
        print("="*60)
        
        logger.info("Refined query demonstration completed successfully")
    except Exception as e:
        logger.error(f"Refined query demonstration failed: {e}")
        print(f"Error during refined query test: {e}")


def interactive_query_runner(db: SQLDatabase) -> None:
    """
    Provide an interactive menu to run demos or ad-hoc refined queries.
    
    Args:
        db: SQLDatabase connection object
    """
    try:
        logger.info("Starting interactive query runner")
        nl2sql_service = NL2SQLService(db)
        
        print("\n" + "="*60)
        print("INTERACTIVE NL2SQL TESTING")
        print("="*60)
        print("Options:")
        print("  1) Run first query demo (includes additional samples)")
        print("  2) Run refined query with your own question")
        print("  3) Run refined query with few-shot guidance")
        print("  4) Exit interactive mode")
        print("-"*60)
        print("Type the option number or 'exit'/'q' to quit.\n")
        
        while True:
            choice = input("Select an option [1/2/3/4]: ").strip().lower()
            
            if choice in {"4", "exit", "q", "quit"}:
                print("Exiting interactive mode.")
                logger.info("Interactive mode exited by user")
                break
            
            if choice == "1":
                execute_first_query(db)
                continue
            
            if choice == "2":
                question = input(
                    f"Enter your question (default: {REFINED_TEST_QUESTION}): "
                ).strip()
                if not question:
                    question = REFINED_TEST_QUESTION
                try:
                    response = nl2sql_service.process_question_rephrased(question)
                    
                    print("\nGenerated SQL Query:")
                    print(response["sql"])
                    print("-"*60)
                    print("Raw Query Results:")
                    print(response["result"])
                    print("-"*60)
                    print("Rephrased Answer:")
                    print(response["answer"])
                    print("="*60 + "\n")
                    
                    logger.info("Interactive refined query executed successfully")
                except Exception as exc:
                    logger.error(f"Interactive refined query failed: {exc}")
                    print(f"Error executing refined query: {exc}")
                continue
            
            if choice == "3":
                question = input(
                    f"Enter your question for few-shot guidance (default: {REFINED_TEST_QUESTION}): "
                ).strip()
                if not question:
                    question = REFINED_TEST_QUESTION
                try:
                    response = nl2sql_service.process_question_few_shot(question)
                    
                    print("\nGenerated SQL Query (few-shot):")
                    print(response["sql"])
                    print("-"*60)
                    print("Raw Query Results:")
                    print(response["result"])
                    print("-"*60)
                    print("Rephrased Answer:")
                    print(response["answer"])
                    print("="*60 + "\n")
                    
                    logger.info("Interactive few-shot refined query executed successfully")
                except Exception as exc:
                    logger.error(f"Interactive few-shot query failed: {exc}")
                    print(f"Error executing few-shot query: {exc}")
                continue
            
            print("Invalid selection. Please choose 1, 2, 3, or 4 (or 'exit').")
    except Exception as exc:
        logger.error(f"Interactive query runner encountered an error: {exc}")


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
