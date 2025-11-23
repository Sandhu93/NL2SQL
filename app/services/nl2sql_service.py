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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from pydantic import BaseModel, Field
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
MAX_RESULT_ROWS = 10  # Maximum rows to show in rephrased answers
MAX_RESULT_LENGTH = 2000  # Maximum character length for result string
SYSTEM_TABLE_PROMPT = (
    "Return the names of ALL the SQL tables that MIGHT be relevant to the user question. "
    "Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure they are needed.\n\n"
    "Table reference:\n{table_details}"
)
ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a careful, precise data analyst helping a user understand SQL query results.

You are given:
- The user's natural language question.
- The SQL query that was executed.
- A SQL result OR a PREVIEW of that result.

WHEN THE RESULT IS A SINGLE VALUE:
- Sometimes the SQL Result is effectively a single scalar value, such as:
  - "[(2,)]"
  - "[(42,)]"
  - "2"
  - or a 1-row, 1-column table.
- In those cases, interpret that value directly and answer in natural language, e.g.:
  - "There are 2 customers who have more than 5 orders."
- DO NOT invent any IDs, names, or extra fields that are not present in the result.
- If the query is an aggregate (COUNT, SUM, AVG, MIN, MAX, etc.), only answer what that aggregate tells you.

IMPORTANT ABOUT PREVIEWS:
- Sometimes the SQL Result is a PREVIEW only (for example: "SQL query returned N rows in total. Preview of the first K rows only: [...]").
- In those cases, DO NOT assume you can see every row.
- Use the preview plus the total row count to summarize the answer.
- Make it clear in your answer if you are only showing or describing the first K rows out of N.
- DO NOT fabricate or hallucinate rows that are not visible in the preview.

STRICT RULES ABOUT GLOBAL STATEMENTS:
- You must NEVER claim that a value is the global minimum, maximum, "highest", or "lowest" for the FULL result set UNLESS:
  - The query itself is a global aggregate (e.g. SELECT MAX(...), MIN(...), COUNT(*), SUM(...)), OR
  - The result clearly includes ALL rows (not a preview).
- If you only see a preview and you talk about extremes, you MUST say things like:
  - "in this preview" or "among the shown rows".
- Do NOT claim "in the whole dataset" or "overall" when you only have a preview.

FORMATTING RULES FOR MULTIPLE ROWS:
- If the result or preview clearly shows multiple rows (for example, a Python-style list of tuples),
  then:
  1. Infer reasonable column names from the question and/or query (for example, 'customerNumber', 'customerName', 'totalOrders').
  2. Render ONLY the visible rows as a compact Markdown table.
  3. After the table, add a short natural language summary that includes:
     - The total number of rows returned (N), if this information is present in the result/preview text.
     - How many rows you actually showed (K = number of rows in the table).
     - Any obvious highlights visible in the preview (for example, which shown row has the highest value IN THE PREVIEW).
- When the preview text includes phrases like "SQL query returned 122 rows in total" and "Preview of the first 10 rows", use those concrete numbers (122 and 10) in your answer. Do NOT write placeholders like "N" or "K" in the final answer.
- Do NOT just restate the preview text verbatim.
- Avoid generic sentences like "this pattern continues"; be specific about what you can see in the preview.

Question: {question}
SQL Query: {query}
SQL Result (or preview): {result}

Provide a clear, concise answer for the user based on the information available.
If there are many rows, give an overview and mention how many rows exist in total, and how many you are actually showing.
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
        use_dynamic_selector: bool = True,
        k: int = 2,
        include_messages: bool = False,
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate with few-shot examples for SQL generation.
        
        Args:
            examples: Optional list of example dicts with 'input' and 'query'.
            use_dynamic_selector: Whether to use semantic similarity selection (vectorstore-backed).
            k: Number of examples to select when using dynamic selector.
            include_messages: Whether to include prior chat history (for memory).
        
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
            selector = self._build_semantic_selector(selected_examples, k=k)
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
                MessagesPlaceholder(variable_name="messages") if include_messages else None,
                ("human", "{input}"),
            ]
        )
        # Remove any None entries (when messages placeholder not requested)
        prompt = ChatPromptTemplate.from_messages([m for m in prompt.messages if m])
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

    def _build_semantic_selector(self, examples: List[dict], k: int = 2) -> SemanticSimilarityExampleSelector:
        """
        Build a semantic similarity example selector using FAISS vectorstore.
        
        Args:
            examples: Few-shot examples containing 'input' and 'query'.
            k: Number of examples to retrieve.
        
        Returns:
            SemanticSimilarityExampleSelector: Selector configured with vectorstore.
        """
        embeddings = OpenAIEmbeddings()
        selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            FAISS,
            k=k,
            input_keys=["input"],
        )
        return selector

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
        
        table_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TABLE_PROMPT.format(
                        table_details=self._table_details_text()
                    ),
                ),
                ("human", "{input}"),
            ]
        )
        try:
            structured_llm = self.llm.with_structured_output(Table)
            table_chain = table_prompt | structured_llm
            tables = table_chain.invoke({"input": question})
            # ensure list handling for consistent downstream behavior
            table_objs = tables if isinstance(tables, list) else [tables]
            table_names = [table.name for table in table_objs][:top_k]
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

    @staticmethod
    def _summarize_result(
        result: object,
        max_rows: int = MAX_RESULT_ROWS,
        max_length: int = MAX_RESULT_LENGTH,
    ) -> str:
        """
        Summarize potentially large query results for LLM prompting.
        
        - Small results are returned as-is (stringified).
        - Larger results include only a preview plus metadata so the LLM
          knows it is not seeing all rows.
        
        Args:
            result: Raw query result (string, list, tuple, etc.)
            max_rows: Maximum rows to include in the preview
            max_length: Maximum characters to include in the preview string
        
        Returns:
            str: Summary string suitable for prompt injection
        """
        import ast
        
        try:
            if isinstance(result, str):
                try:
                    parsed = ast.literal_eval(result)
                    if isinstance(parsed, (list, tuple)):
                        result = parsed
                except Exception:
                    return result if len(result) <= max_length else result[:max_length] + " ... (truncated)"
            
            if isinstance(result, (list, tuple)):
                total_count = len(result)
                if total_count == 0:
                    return "SQL query returned 0 rows."
                if total_count <= max_rows:
                    full_str = str(result)
                    if len(full_str) > max_length:
                        full_str = full_str[:max_length] + " ... (truncated)"
                    return f"SQL query returned {total_count} rows:\n{full_str}"
                
                preview = result[:max_rows]
                preview_str = str(preview)
                if len(preview_str) > max_length:
                    preview_str = preview_str[:max_length] + " ... (preview truncated)"
                
                summary = (
                    f"SQL query returned {total_count} rows in total.\n"
                    f"Preview of the first {max_rows} rows only:\n"
                    f"{preview_str}\n"
                    f"(Note: Only a preview is shown here to save tokens. "
                    f"Shown rows: {len(preview)} of {total_count}.)"
                )
                return summary
            
            result_str = str(result)
            if len(result_str) > max_length:
                result_str = result_str[:max_length] + " ... (truncated)"
            return result_str
        
        except Exception:
            result_str = str(result)
            if len(result_str) > max_length:
                result_str = result_str[:max_length] + " ... (truncated)"
            return result_str

    @staticmethod
    def _format_display_result(result: object, max_rows: int = 20) -> str:
        """
        Format results for console display with truncation metadata.
        
        Args:
            result: Raw query result (string or sequence)
            max_rows: Maximum rows to display before truncating
        
        Returns:
            str: Human-friendly display string
        """
        try:
            if isinstance(result, str):
                return result
            if isinstance(result, (list, tuple)):
                if len(result) <= max_rows:
                    return str(result)
                preview = result[:max_rows]
                return (
                    f"{preview}\n... ({len(result)} total rows, showing first {max_rows})"
                )
            if isinstance(result, dict) and "preview" in result:
                preview = result.get("preview")
                total = result.get("total_rows")
                note = result.get("note", "")
                return f"{preview}\n... ({total} total rows) {note}"
            return str(result)
        except Exception:
            return str(result)
    
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
        use_dynamic_selector: bool = True,
        k: int = 2,
        messages: Optional[List] = None,
    ) -> str:
        """
        Generate SQL using few-shot examples to guide the model.
        
        Args:
            question: Natural language question
            use_dynamic_selector: Whether to select examples dynamically
            k: Number of examples to select when dynamic selection is enabled
            messages: Prior chat messages for conversational memory
        
        Returns:
            str: Generated SQL query
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        prompt = self.build_few_shot_prompt(
            use_dynamic_selector=use_dynamic_selector,
            k=k,
            include_messages=messages is not None,
        )
        try:
            logger.info("Generating SQL with few-shot examples")
            chain = prompt | self.llm | StrOutputParser()
            table_names = self._select_relevant_tables(question, top_k=k)
            table_info = self._format_table_context(table_names)
            payload = {
                "input": question,
                "table_info": table_info,
                "top_k": k,
            }
            if messages is not None:
                payload["messages"] = messages
            generated_query = chain.invoke(payload)
            logger.info("SQL query generated with few-shot examples")
            return generated_query
        except Exception as exc:
            logger.error(f"Failed to generate SQL with examples: {exc}")
            raise

    def process_question_few_shot(
        self,
        question: str,
        use_dynamic_selector: bool = True,
        k: int = 2,
        history: Optional[ChatMessageHistory] = None,
    ) -> dict:
        """
        Generate SQL with few-shot examples, execute it, and return results.
        
        Args:
            question: Natural language question
            use_dynamic_selector: Whether to use dynamic example selection
            k: Number of examples to select when dynamic selection is enabled
            history: Optional chat history for conversational memory
        
        Returns:
            dict: Dictionary containing 'sql', 'result', and 'answer'
        """
        sql_query = self.generate_sql_with_examples(
            question,
            use_dynamic_selector=use_dynamic_selector,
            k=k,
            messages=history.messages if history else None,
        )
        result = self.execute_query(sql_query)
        q_lower = question.lower().strip()
        is_list_all = q_lower.startswith("list ") or q_lower.startswith("show all") or "list all " in q_lower

        summarized = self._summarize_result(
            result,
            max_rows=MAX_RESULT_ROWS,
            max_length=MAX_RESULT_LENGTH,
        )

        if is_list_all and isinstance(result, (list, tuple)) and len(result) > MAX_RESULT_ROWS:
            total_rows = len(result)
            answer = (
                f"The query returned {total_rows} rows. "
                f"Full results are available below; only a preview is summarized here."
            )
        else:
            answer = self.rephrase_answer(question, sql_query, summarized)
        if history is not None:
            history.add_user_message(question)
            history.add_ai_message(answer)
        return {"sql": sql_query, "result": result, "answer": answer}

    def rephrase_answer(self, question: str, sql_query: str, result: object) -> str:
        """
        Convert raw SQL results into a concise natural language answer.
        
        Args:
            question: Original user question
            sql_query: Generated SQL query
            result: Raw SQL execution result (may be pre-summarized)
            
        Returns:
            str: Rephrased, user-friendly answer
        """
        try:
            logger.info("Rephrasing SQL result for user-friendly answer")
            
            response = (ANSWER_PROMPT | self.llm | StrOutputParser()).invoke(
                {
                    "question": question,
                    "query": sql_query,
                    "result": result,
                }
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
        summarized = self._summarize_result(
            result,
            max_rows=MAX_RESULT_ROWS,
            max_length=MAX_RESULT_LENGTH,
        )
        answer = self.rephrase_answer(question, sql_query, summarized)
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
        print("  4) Run few-shot guidance with conversational memory")
        print("  5) Exit interactive mode")
        print("-"*60)
        print("Type the option number or 'exit'/'q' to quit.\n")
        
        while True:
            choice = input("Select an option [1/2/3/4/5]: ").strip().lower()
            
            if choice in {"5", "exit", "q", "quit"}:
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
                    print(nl2sql_service._format_display_result(response["result"]))
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
                    print(nl2sql_service._format_display_result(response["result"]))
                    print("-"*60)
                    print("Rephrased Answer:")
                    print(response["answer"])
                    print("="*60 + "\n")
                    
                    logger.info("Interactive few-shot refined query executed successfully")
                except Exception as exc:
                    logger.error(f"Interactive few-shot query failed: {exc}")
                    print(f"Error executing few-shot query: {exc}")
                continue
            
            if choice == "4":
                question = input(
                    f"Enter your question for few-shot with memory (default: {REFINED_TEST_QUESTION}): "
                ).strip()
                if not question:
                    question = REFINED_TEST_QUESTION
                chat_history = ChatMessageHistory()
                try:
                    response = nl2sql_service.process_question_few_shot(
                        question,
                        history=chat_history,
                    )
                    
                    print("\nGenerated SQL Query (few-shot with memory):")
                    print(response["sql"])
                    print("-"*60)
                    print("Raw Query Results:")
                    print(nl2sql_service._format_display_result(response["result"]))
                    print("-"*60)
                    print("Rephrased Answer:")
                    print(response["answer"])
                    print("="*60 + "\n")
                    
                    logger.info("Interactive few-shot with memory executed successfully")
                    
                    while True:
                        follow_up = input(
                            "Ask a follow-up (or press Enter to finish this session): "
                        ).strip()
                        if not follow_up:
                            break
                        follow_resp = nl2sql_service.process_question_few_shot(
                            follow_up,
                            history=chat_history,
                        )
                        print("\nGenerated SQL Query (few-shot with memory):")
                        print(follow_resp["sql"])
                        print("-"*60)
                        print("Raw Query Results:")
                        print(nl2sql_service._format_display_result(follow_resp["result"]))
                        print("-"*60)
                        print("Rephrased Answer:")
                        print(follow_resp["answer"])
                        print("="*60 + "\n")
                except Exception as exc:
                    logger.error(f"Interactive few-shot with memory failed: {exc}")
                    print(f"Error executing few-shot with memory: {exc}")
                continue
            
            print("Invalid selection. Please choose 1, 2, 3, 4, or 5 (or 'exit').")
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
