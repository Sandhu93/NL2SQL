This file SHOULD live at the repository root as `AI_AGENT_RULES.md` and be provided as context to any AI coding agent working on this project.

# AI Agent Coding Rules for NL2SQL Tutorial Project

These rules apply to **all code, tests, docs, and terminal commands** generated or modified by the AI agent in this repository.

### How the AI should use this file

- Treat every **MUST** / **MUST NOT** as hard rules.
- Treat **SHOULD** as recommended, unless a higher-priority rule conflicts.
- Before changing code, the AI SHOULD:
  - Look for existing modules with similar behavior and follow their patterns.
  - Prefer editing existing functions over creating near-duplicates.
- If user instructions conflict with these rules:
  - Prefer **environment safety** and **security** first.
  - Otherwise follow the **most specific** instruction (e.g., file-level rule beats general rule).

### 0.1 Core Rules Summary (for the AI)

1. Always use `venv\Scripts\activate;` before any terminal command.
2. Only use Python 3.11 for runtime and dependencies.
3. Use snake_case for variables/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants.
4. Put API logic in FastAPI routes, business logic in `services`, DB logic in `db`, vectorstore logic in `vectorstores`.
5. Never hardcode secrets; always read from environment variables.
6. All SQL must be validated, sanitized, and executed with parameters; only SELECT queries are allowed.
7. All public functions and classes need type hints, docstrings, and proper error handling.
8. Log important events and errors; don't swallow exceptions.
9. Add/maintain tests for new or changed behavior, under the appropriate `tests/` folder.
10. Follow existing patterns in this repository before introducing new ones.

## 0. Scope & Priorities

When generating or editing code, the AI agent MUST prioritize in this order:

1. **Environment safety**
   - Never break or bypass the required virtual environment and Python version.
2. **Correctness & security**
   - Code must be functionally correct, safe, and not leak secrets.
3. **Compatibility with existing project**
   - Match existing patterns, structure, and dependencies in this repo.
4. **Maintainability**
   - Clear structure, type hints, logging, docstrings, and tests.
5. **Performance**
   - Reasonable performance for typical NL2SQL workloads; no premature micro-optimizations.

If any instruction conflicts with these rules, **follow this file unless explicitly told otherwise.**

---

## 1. Virtual Environment & Python Version

### 1.1 MANDATORY Virtual Environment Activation

**EVERY SINGLE terminal command MUST be executed within the activated virtual environment named `venv`.**

On Windows (PowerShell):

```bash
venv\Scripts\activate; <your_command_here>
```

Examples of CORRECT terminal usage:

```bash
# Installing packages
venv\Scripts\activate; pip install package_name

# Running Python scripts
venv\Scripts\activate; python script.py

# Running tests
venv\Scripts\activate; pytest

# Checking installed packages
venv\Scripts\activate; pip list

# Running Jupyter
venv\Scripts\activate; jupyter notebook
```

NEVER run commands like this (they are WRONG):

```bash
pip install package_name
python script.py
jupyter notebook
pytest
uvicorn main:app --reload
```

### 1.2 Virtual Environment Verification

Before doing environment-sensitive operations (installing packages, running migrations, etc.), the agent SHOULD verify:

```bash
venv\Scripts\activate; python -c "import sys; print('Virtual env active:', 'venv' in sys.executable)"
```

### 1.3 Python Version Requirements

**ONLY use Python 3.11** in this project.

Python 3.12 is NOT allowed (ChromaDB and other dependencies may fail).

Always verify version inside the virtual environment:

```bash
venv\Scripts\activate; python --version
```

## 2. Code Generation & Structure

### 2.1 Function & Class Structure Pattern

When creating core processing functions, follow this pattern (adapt names as needed):

```python
from typing import Optional, List, Dict, Any, Union
import logging
from dataclasses import dataclass
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

def process_natural_language_query(
    query: str,
    database_schema: Dict[str, Any],
    model_name: str = "gpt-4"
) -> Dict[str, Union[str, float, List[str]]]:
    """
    Convert natural language query to SQL statement.

    Args:
        query: Natural language query from user.
        database_schema: Dictionary containing table schemas.
        model_name: OpenAI model to use for conversion.

    Returns:
        Dictionary containing SQL query and metadata, e.g.:
        {
            "sql": "...",
            "confidence": 0.0-1.0,
            "tables_used": ["table1", "table2"]
        }

    Raises:
        ValueError: If query is empty or invalid.
        ConnectionError: If OpenAI API is unavailable.
        RuntimeError: For other unrecoverable errors.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        logger.info("Processing query: %s", query[:100])
        # Implementation here...
        generated_sql: str = ...
        confidence_score: float = ...
        tables_list: List[str] = ...

        return {
            "sql": generated_sql,
            "confidence": confidence_score,
            "tables_used": tables_list,
        }
    except Exception as e:
        logger.error("Query processing failed: %s", e, exc_info=True)
        raise
```

**Rules:**

All public functions MUST:
- Use type hints.
- Have meaningful docstrings with Args / Returns / Raises.
- Log at least one high-level message on entry or significant events.
- Avoid extremely long functions; split into smaller helpers when logic is complex.

## 3. Naming Conventions (MANDATORY)

The AI agent MUST follow these naming conventions consistently across the project.

### 3.1 Modules, Files, and Packages

- Python files (modules): **snake_case**, all lowercase, words separated by underscores  
  - Examples: `nl2sql_converter.py`, `database_manager.py`, `vector_store_manager.py`
- Packages (directories with `__init__.py`): **snake_case**  
  - Examples: `app`, `api`, `db`, `models`, `services`, `utils`

### 3.2 Variables

- Regular variables: **snake_case**, descriptive
  - ✅ `user_id`, `query_text`, `database_schema`, `confidence_score`
  - ❌ `x`, `tmp`, `data1`, `foo`, `bar` (except inside very small, obvious scopes)
- Boolean variables should read like yes/no questions:
  - ✅ `is_active`, `has_error`, `should_retry`, `is_valid_sql`
- Temporary variables:
  - Loop counters may use `i`, `j`, `idx` in short loops.
  - For anything non-trivial, use meaningful names: `row_index`, `table_name`, `column_mapping`.

### 3.3 Global Variables & Constants

- Global **mutable** variables are strongly discouraged.
  - Prefer function parameters, dependency injection, or configuration objects.
- Constants:
  - Use **UPPER_SNAKE_CASE** at module top:
    - ✅ `DEFAULT_CONFIDENCE_THRESHOLD = 0.8`
    - ✅ `CHROMA_PERSIST_DIR = "chroma_db"`
- If a global is truly needed (e.g., a shared `settings` object), document it clearly and treat it as read-only wherever possible.

### 3.4 Function and Method Names

- Use **snake_case**, start with a verb, describe what the function does:
  - ✅ `convert_query_to_sql`, `load_database_schema`, `execute_safe_query`, `validate_nl_query`
- Async functions still use snake_case:
  - ✅ `async def process_nl_query_async(...)`
- Private/internal helpers:
  - Prefix with a single underscore:
    - ✅ `_build_prompt`, `_normalize_schema`, `_create_db_engine`

### 3.5 Arguments / Parameters

- Use descriptive names, not `a`, `b`, `v1`:
  - ✅ `query_text`, `database_name`, `model_name`, `max_results`
- Do not reuse names with different meanings in the same module.
- For optional parameters, signal that in the name if it helps:
  - Example: `timeout_seconds: int | None = None`

### 3.6 Class and Exception Names

- Classes: **PascalCase**
  - ✅ `NL2SQLConverter`, `DatabaseManager`, `VectorStoreManager`, `ConfigSettings`
- Exception classes: PascalCase and end with `Error` where appropriate:
  - ✅ `SQLValidationError`, `VectorStoreInitializationError`
- Pydantic models:
  - Also PascalCase:
    - ✅ `QueryRequest`, `QueryResponse`, `DatabaseSchemaModel`

### 3.7 Object / Instance Names

- Instance variables: **snake_case**, often the singular of the class:
  - ✅ `db_manager = DatabaseManager(DATABASE_URL)`
  - ✅ `converter = NL2SQLConverter(model_name="gpt-4")`
- Self attributes inside classes:
  - ✅ `self.database_url`, `self.vector_store`, `self.schema_cache`

## 4. Error Handling Rules

**General principles:**
- Prefer specific exceptions over catching Exception when possible.
- Do not swallow exceptions silently.
- Log errors with context and exc_info=True when needed.

**Pattern:**

```python
try:
    result = risky_operation()
except OpenAIError as e:
    logger.error("OpenAI API error: %s", e, exc_info=True)
    raise
except DatabaseError as e:
    logger.error("Database connection failed: %s", e, exc_info=True)
    raise
except ValueError as e:
    logger.warning("Validation error: %s", e)
    raise
except Exception as e:
    logger.error("Unexpected error: %s", e, exc_info=True)
    raise
```

## 5. Import Organization Rules

For each Python module, organize imports in this order:

```python
# Standard library imports
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy import create_engine, text

# LangChain / OpenAI imports
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Local imports
from .models import QueryRequest, QueryResponse
from .database import DatabaseManager
from .utils import validate_sql, sanitize_input
```

**Rules:**
- Remove unused imports.
- Do not create circular imports; refactor into utils or other shared modules if necessary.

## 6. Environment Variables & Configuration

```python
import os
from dotenv import load_dotenv

# ALWAYS load environment variables at module level
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "mysql://localhost:3306/default")
```

**Rules:**
- NEVER hardcode secrets or API keys.
- Always use os.getenv (optionally with safe defaults for non-secret values).
- If a required env var is missing, fail fast with a clear error message.

## 7. Database Connection Rules

```python
from contextlib import contextmanager
from typing import Dict, List
import logging

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        connection = self.engine.connect()
        trans = connection.begin()
        try:
            yield connection
            trans.commit()
        except Exception as e:
            trans.rollback()
            logger.error("Database operation failed: %s", e, exc_info=True)
            raise
        finally:
            connection.close()

    def execute_query(self, sql: str, params: Dict | None = None) -> List[Dict]:
        """Execute SQL query with proper error handling."""
        with self.get_connection() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result.fetchall()]
```

**Rules:**
- Always use parameterized queries (text(sql), params) rather than string concatenation.
- Database writes (if any) must be explicit; NL2SQL queries SHOULD be read-only (see security rules).

## 8. FastAPI Endpoint Rules

```python
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NL2SQL API",
    description="Convert natural language to SQL queries",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    database_name: Optional[str] = Field(None, description="Target database name")

class QueryResponse(BaseModel):
    sql: str
    confidence: float
    execution_time: float
    tables_used: List[str]

@app.post("/convert", response_model=QueryResponse)
async def convert_query(request: QueryRequest) -> QueryResponse:
    """Convert natural language query to SQL."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = await process_nl_query(request.query, request.database_name)
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Query conversion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

**Rules:**
- Always validate input and return appropriate HTTP status codes.
- Never expose internal exception details in HTTP responses.

## 9. Testing Strategy & Rules (MANDATORY)

The AI agent MUST treat tests as **first-class code**, not optional extras.

### 9.1 Types of Tests (What We Use)

The project uses (or SHOULD use) these test types:

1. **Unit Tests**
   - Small, fast tests of a single function or class.
   - External dependencies (DB, OpenAI, vector stores, network) are **mocked**.
   - Live under: `tests/unit/`.

2. **Integration Tests**
   - Test how multiple components work together (e.g., FastAPI endpoint + service + DB).
   - May use a test DB or test vectorstore.
   - Live under: `tests/integration/`.

3. **End-to-End (E2E) / API Tests** (optional in this project)
   - Hit the running API as a client would (HTTP requests).
   - Useful for the public `/convert` endpoint.

4. **Regression Tests**
   - Tests added specifically to prove a **previous bug** never comes back.
   - Usually unit or integration tests with a reference to the bug/ticket.

5. **Smoke Tests**
   - Very basic sanity checks (e.g., "API starts", "/health" returns 200).

The AI MUST choose the **smallest appropriate test type** for the change:
- Logic-only change → unit test.
- Change touching multiple layers → add or update integration / E2E tests.

### 9.2 When to Write or Update Tests

The AI MUST add or update tests in these situations:

- **New functionality**:
  - At least one **unit test** for each public function or method with meaningful logic.
  - If it affects HTTP behavior, at least one **integration/API test** for the endpoint.

- **Bug fix**:
  - First add a **regression test** that fails with the old behavior.
  - Then change the code until the new test passes.
  - The regression test MUST remain in the test suite permanently.

- **Refactor**:
  - If behavior is intended to stay the same:
    - Do not weaken or delete tests.
    - Update tests only if signatures change, keeping expectations identical.
  - If behavior intentionally changes:
    - Update tests to reflect the new contract.
    - Add new tests for edge cases introduced by the change.

- **Security / validation change**:
  - Add tests that:
    - Prove dangerous inputs are rejected.
    - Prove valid inputs still work.

The AI MUST NOT:
- Delete or mark tests as skipped (`@pytest.mark.skip`) just to make a failing test pass.
- Comment out tests instead of fixing the underlying issue.

### 9.3 Test File Structure & Naming

#### 9.3.1 Directory Layout

Tests SHOULD be organized as:

```text
tests/
  unit/
    test_nl2sql_service.py
    test_sql_sanitizer.py
    test_database_manager.py
  integration/
    test_api_convert_endpoint.py
    test_vector_store_integration.py
```

**Rules:**
- Unit tests go in `tests/unit/`.
- Integration/E2E tests go in `tests/integration/`.

#### 9.3.2 Test File Names

Test modules MUST be named: `test_<module_name>.py`

**Examples:**
- `app/services/nl2sql_service.py` → `tests/unit/test_nl2sql_service.py`
- `app/utils/sql_sanitizer.py` → `tests/unit/test_sql_sanitizer.py`
- `app/db/database_manager.py` → `tests/unit/test_database_manager.py`

#### 9.3.3 Test Class & Function Names

**Test classes:**
- PascalCase starting with `Test`:
  - ✅ `class TestNL2SQLService:`
  - ✅ `class TestSqlSanitizer:`

**Test functions and methods:**
- Must start with `test_`.
- Name describes behavior, not implementation:
  - ✅ `test_convert_query_returns_select_statement`
  - ✅ `test_empty_query_raises_value_error`
  - ✅ `test_sanitize_sql_rejects_drop_statements`

### 9.4 Test Design & Patterns

#### 9.4.1 Arrange – Act – Assert

Each test SHOULD follow the AAA pattern:

```python
def test_convert_query_returns_select_statement(converter: NL2SQLConverter):
    # Arrange
    query = "Show all active users"

    # Act
    result = converter.convert_query(query)

    # Assert
    assert result["sql"].startswith("SELECT")
    assert "users" in result["sql"]
```

The AI MUST keep tests:
- **Focused**: one behavior per test.
- **Deterministic**: no random time-based or external-system flakiness.

#### 9.4.2 Use of Fixtures

Reusable setup belongs in pytest fixtures, not repeated in every test.

Fixtures live in:
- `tests/conftest.py` for shared ones, or
- Inside the relevant test module.

Example fixture:

```python
import pytest
from app.services.nl2sql_service import NL2SQLService

@pytest.fixture
def nl2sql_service() -> NL2SQLService:
    return NL2SQLService(model_name="gpt-3.5-turbo")
```

#### 9.4.3 Mocking External Dependencies

In unit tests, the AI MUST mock:
- OpenAI / LLM calls.
- Database operations.
- Network calls.
- Vector store operations (Chroma, FAISS) unless the test is explicitly integration-level.

Example:

```python
from unittest.mock import patch

def test_simple_query_conversion(nl2sql_service, mock_openai_response):
    with patch("app.services.nl2sql_service.openai.ChatCompletion.create") as mock_openai:
        mock_openai.return_value = mock_openai_response

        result = nl2sql_service.convert_query("Show all active users")

        assert "SELECT" in result["sql"]
        assert "users" in result["sql"]
```

### 9.5 Test Documentation

Tests themselves are executable documentation. The AI MUST also:

**Document non-obvious tests** with short comments:
- Why does this test exist?
- What bug or edge case does it cover?

```python
def test_sanitize_sql_rejects_drop_statements():
    """Regression test for bug #123: DROP statements must be rejected."""
    with pytest.raises(ValueError):
        sanitize_sql_input("DROP TABLE users;")
```

**Maintain a simple TESTING.md** at project root (or update it if it exists) describing:
- How to run unit tests.
- How to run integration tests.
- Any required env vars for tests (e.g. test DB URL).

Example snippet for TESTING.md:

```markdown
# Testing

## Unit Tests
venv\Scripts\activate; python -m pytest tests/unit -v

## Integration Tests
venv\Scripts\activate; python -m pytest tests/integration -v
```

The AI SHOULD update TESTING.md when test commands or structure change.

### 9.6 Versioning Tests Alongside Code

Tests MUST evolve together with the production code:

**Every feature branch or change** that modifies behavior MUST:
- Update or add tests in the same commit/PR.

**When there is a breaking change** in behavior:
- Update tests to match the new contract.
- Keep regression tests around where they still make sense.

**When a test is intentionally changed:**
- The AI SHOULD update test names/docstrings to match the new behavior.
- If the change is disabling a test (skip/xfail), the reason MUST be documented:

```python
@pytest.mark.xfail(reason="Known bug #456, not fixed yet")
```

The AI MUST NOT:
- Remove tests just to make the test suite pass.
- Reduce assertion strength (e.g., checking only `status_code == 200` instead of also checking response body) without a good reason and documentation.

### 9.7 Minimum Testing Expectations per Change

For each non-trivial change, the AI SHOULD ensure at least:

**New function / class:**
- 1+ unit tests covering:
  - Normal behavior.
  - At least one error/edge case.

**New API endpoint:**
- 1+ integration/API tests using TestClient that:
  - Verify valid input → 2xx response with correct body.
  - Verify invalid input → 4xx error with correct message.

**Security behavior change:**
- Tests that:
  - Reject dangerous input (e.g., SQL injection, forbidden keywords).
  - Accept valid input (to avoid over-blocking).

## 10. Vector Store Implementation Rules

```python
from typing import List, Dict, Optional

from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        store_type: str = "chroma",
        persist_directory: str = "./chroma_db",
    ):
        self.embeddings = OpenAIEmbeddings()
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.vector_store = self._initialize_store()

    def _initialize_store(self):
        """Initialize vector store based on type."""
        if self.store_type == "chroma":
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        if self.store_type == "faiss":
            # NOTE: Caller should later replace dummy data with real texts.
            return FAISS.from_texts(
                texts=["dummy"],
                embedding=self.embeddings,
            )
        raise ValueError(f"Unsupported vector store type: {self.store_type}")

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add documents to vector store."""
        try:
            self.vector_store.add_texts(documents, metadatas=metadata)
            if self.store_type == "chroma":
                self.vector_store.persist()
        except Exception as e:
            logger.error("Failed to add documents: %s", e, exc_info=True)
            raise
```

## 11. Dependency Management Rules

**The AI MUST NOT introduce new major frameworks or ORMs without explicit instruction.**

When adding a new Python package:
- **Prefer libraries already present in `requirements.txt`.**
- If a new package is truly needed, add it with:
  ```bash
  venv\Scripts\activate; pip install package_name; pip freeze > requirements.txt
  ```
- **Avoid overlapping libraries** that do the same job (e.g., multiple HTTP clients) unless justified.
- **Check compatibility** with Python 3.11 and existing dependencies before installation.

**Examples of what to avoid:**
- Adding both `requests` and `httpx` for HTTP calls
- Installing `sqlalchemy` when `pymysql` already handles DB connections
- Adding `pandas` when native Python data structures suffice

## 12. Follow Existing Patterns Rule

**Before creating new abstractions, the AI MUST:**
- **Search the existing codebase** for similar patterns (naming, error handling, logging).
- **Match those patterns** instead of inventing new ones.
- **Reuse existing utilities** and helper functions where applicable.
- **Extend existing classes** rather than creating parallel implementations.

**Examples:**
- If the codebase uses a specific logging format, maintain that format.
- If error handling follows a particular pattern, use the same pattern.
- If there's an existing database manager, extend it rather than creating a new one.
- If validation functions exist, use them rather than writing new validators.

## 13. Project, File, and Component Structure

These rules define **where code should live**, how to split it, and how to avoid "god files" with everything in one place.

### 13.1 High-Level Project Layout (Recommended)

A typical layout for this project SHOULD look like:

```text
project_root/
  app/                      # Main application code
    __init__.py
    main.py                 # FastAPI app / entrypoint
    api/                    # API routers & endpoints
      __init__.py
      routes_nl2sql.py
    services/               # Business logic / use-cases
      __init__.py
      nl2sql_service.py
    db/                     # Database access & models
      __init__.py
      database_manager.py
    vectorstores/           # Vector store logic
      __init__.py
      vector_store_manager.py
    models/                 # Pydantic models & domain models
      __init__.py
      request_response.py
    config/                 # Settings, env handling
      __init__.py
      settings.py
    utils/                  # Shared helpers (logging, validation, etc.)
      __init__.py
      logging_config.py
      sql_sanitizer.py

  tests/
    unit/
    integration/

  requirements.txt
  README.md
  .env.example
```

You don't have to copy this exactly, but files MUST follow the same general principle: API → services → DB/vectorstore/utils, not the other way around.

### 13.2 Component Responsibility Rules

**FastAPI / API layer** (`app/main.py`, `app/api/`):
- Handles:
  - HTTP endpoints, request parsing, response formatting.
  - Input validation using Pydantic models.
- MUST NOT contain:
  - Raw SQL.
  - Direct OpenAI/LLM calls.
  - Heavy business logic.
- SHOULD delegate to service functions (e.g., `nl2sql_service.convert_query(...)`).

**Service layer** (`app/services/`):
- Contains application/business logic:
  - Orchestrates LLM calls, SQL generation, and DB access.
- May:
  - Call OpenAI/LLM clients.
  - Use DatabaseManager and vector stores.
- Should stay framework-agnostic where possible (e.g., no FastAPI types here).
- **Main entrypoints stay thin**: keep `main.py` limited to wiring/bootstrapping; put interactive helpers, demos, or business logic in services/utils instead of adding functions directly to `main.py`.

**Database layer** (`app/db/`):
- Contains all DB-specific code:
  - Engine/connection creation, transaction management.
  - Query execution helpers.
- No HTTP or API-related logic here.

**Vector store layer** (`app/vectorstores/`):
- Manages Chroma/FAISS initialization, persistence, and document operations.
- No FastAPI or request handling logic.

**Models layer** (`app/models/`):
- Contains Pydantic models for requests/responses.
- Optionally domain models / DTOs.

**Utils layer** (`app/utils/`):
- Shared helpers that do not belong in a specific feature domain:
  - Logging setup, configuration helpers, SQL sanitization, common validation.

### 13.3 When to Create a New File vs. Add to Existing

MUST consider creating a new file/module when:
- A file is exceeding 300–400 lines and has multiple unrelated responsibilities.
- You are adding a concept that is logically distinct (e.g., new service, new vector store type).
- You find yourself scrolling a lot to find something or adding many `# region` style comments.
- You risk circular imports by putting everything in a single file.

Examples:
- If `main.py` starts to contain DB logic → move that to `db/database_manager.py`.
- If nl2sql logic grows too large → split into:
  - `nl2sql_service.py` (orchestrates),
  - `prompt_builder.py` (builds prompts),
  - `sql_validator.py` (validates SQL), etc.

### 13.4 Directory Structure Rules

- New feature or capability?
  - Add under the appropriate existing directory (e.g., new endpoint under `app/api/`, new service under `app/services/`).
- Avoid parallel "duplicate" structures:
  - ❌ `nl2sql_service.py` in `app/` and another in `app/services/`.
- Keep file names consistent with their main class or purpose:
  - `nl2sql_service.py` → contains `NL2SQLService` or similar.
  - `database_manager.py` → contains `DatabaseManager`.

## 14. File Creation and Modification Rules

**MUST include proper headers and documentation:**
```python
"""
Module: example_module.py
Description: Brief description of the module's purpose and responsibilities
Dependencies: List major dependencies (langchain, chromadb, etc.)
Author: AI Assistant
Created: 2024-01-01
Last Modified: 2024-01-01
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class ExampleClass:
    """
    Brief description of the class.
    
    Attributes:
        attribute_name: Description of the attribute
        
    Methods:
        method_name: Brief description of what it does
    """
    
    def __init__(self, param: str) -> None:
        """Initialize with proper type hints and validation."""
        self.param = param
        logger.info(f"Initialized {self.__class__.__name__} with param: {param}")
```

**MUST follow project structure conventions:**
- Use appropriate extensions (.py, .md, .txt, .json, .yml)
- Place files in correct directories (tests/, src/, config/, etc.)
- Follow naming conventions (snake_case for Python files)

### When Modifying Existing Files
- ALWAYS preserve existing functionality
- Add comprehensive comments for new code
- Update docstrings if function signatures change
- Add appropriate error handling
- Update type hints

## 15. Logging Configuration

```python
import logging
import sys
from datetime import datetime

def setup_logging() -> logging.Logger:
    """Configure logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(
                f"nl2sql_{datetime.now().strftime('%Y%m%d')}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific log levels for noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    return logging.getLogger(__name__)

logger = setup_logging()
```

## 16. Comments & Documentation Rules

### 16.1 Docstrings

All public functions, methods, and classes MUST have docstrings.

Use clear, concise descriptions and include Args/Returns/Raises where applicable.

Example:

```python
def convert_query_to_sql(query: str, database_schema: dict) -> dict:
    """
    Convert a natural language query into a SQL statement.

    Args:
        query: Natural language query text.
        database_schema: Dictionary representing available tables and columns.

    Returns:
        Dict with keys:
        - "sql": Generated SQL statement.
        - "confidence": Confidence score between 0.0 and 1.0.
        - "tables_used": List of tables referenced in the query.

    Raises:
        ValueError: If query is empty or cannot be converted safely.
    """
```

### 16.2 Inline Comments

Use inline comments to explain **why**, not **what**:
- ✅ `# Using a cached schema to avoid extra DB calls`
- ❌ `# increment i` when the code already says `i += 1`

Avoid large blocks of commented-out code:
- If logic is no longer needed, delete it.
- If you want to preserve it, add it to version control or a design/ADR document.

### 16.3 Module Headers

For important modules, include a short header at the top:

```python
"""
Module: nl2sql_service.py
Description: Business logic for converting natural language queries to SQL.
Author: AI Agent
Created: 2025-11-21
Python Version: 3.11
Dependencies: see requirements.txt
"""
```

Keep headers updated when responsibilities change.

## 17. Security Rules (CRITICAL)

**MUST NEVER hardcode secrets or credentials:**
```python
# ❌ NEVER do this
API_KEY = "sk-1234567890abcdef"

# ✅ ALWAYS use environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
```

**MUST validate and sanitize all inputs:**
```python
def sanitize_sql_input(query: str) -> str:
    """
    Sanitize SQL input to prevent injection attacks.
    
    Args:
        query: Raw SQL query string
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If dangerous SQL keywords detected or not a SELECT query
    """
    stripped = query.strip().rstrip(";")
    query_upper = stripped.upper()
    
    # Enforce SELECT-only policy
    if not query_upper.startswith("SELECT"):
        logger.warning(f"Blocked non-SELECT query: {query[:50]}...")
        raise ValueError("Only SELECT queries are allowed in this project.")
    
    dangerous_keywords = [
        "DROP", "DELETE", "TRUNCATE", "INSERT", 
        "UPDATE", "ALTER", "CREATE", "EXEC"
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            logger.warning(f"Blocked dangerous SQL keyword: {keyword}")
            raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
    
    return stripped

# MUST use parameterized queries
def execute_safe_query(sql: str, params: Optional[Dict] = None) -> List[Dict]:
    """Execute SQL with parameters to prevent injection."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise
```

## 18. Code Quality & Formatting Rules (MANDATORY)

**All new or modified Python code MUST:**
- Be formatted with `black` before submission
- Pass `flake8` linting without errors
- Have maximum line length of 88 characters (black default)

**Required commands to run before code submission:**
```powershell
# Format code with black
venv\Scripts\activate; black src/ app/ tests/

# Check for linting errors
venv\Scripts\activate; flake8 src/ app/ tests/ --max-line-length=88
```

**The AI agent MUST:**
- Run these commands after generating or modifying Python files
- Fix any formatting or linting issues before considering the task complete
- Ensure consistent code style across the entire project

## 19. Performance Optimization Rules

**MUST implement caching for expensive operations:**
```python
import asyncio
from functools import lru_cache, wraps
import time
from typing import Dict, List, Any, Callable

@lru_cache(maxsize=128)
def get_database_schema(database_name: str) -> Dict[str, Any]:
    """
    Cache database schema to avoid repeated queries.
    
    Args:
        database_name: Name of the database
        
    Returns:
        Dictionary containing schema information
    """
    logger.info(f"Fetching schema for database: {database_name}")
    # Implementation would go here
    return {"tables": [], "relationships": []}

# MUST use async for I/O operations
async def process_multiple_queries(queries: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple queries concurrently for better performance.
    
    Args:
        queries: List of SQL query strings
        
    Returns:
        List of query results
    """
    logger.info(f"Processing {len(queries)} queries concurrently")
    tasks = [process_single_query(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]

# MUST add timing for performance monitoring
def timed_operation(func: Callable) -> Callable:
    """Decorator to measure and log execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper
```

## 20. MANDATORY Pre-Generation Checklist

**Before generating or modifying ANY code, AI agents MUST verify:**

### Environment & Compatibility
- [ ] Virtual environment activation command included in instructions
- [ ] Python 3.11 compatibility verified (required for ChromaDB)
- [ ] All required packages in requirements.txt

### Code Quality Standards
- [ ] Type hints added to ALL function parameters and return values
- [ ] Comprehensive docstrings with Args, Returns, Raises sections
- [ ] Input validation implemented for all user inputs
- [ ] Proper error handling with try/except blocks
- [ ] Logging statements added for debugging and monitoring
- [ ] Code formatted with black and passes flake8 without errors

### Security Requirements
- [ ] No hardcoded secrets or API keys
- [ ] Input sanitization for SQL queries
- [ ] Parameterized database queries only
- [ ] Environment variable usage for configuration

### Performance & Reliability
- [ ] Async/await for I/O operations where applicable
- [ ] Caching implemented for expensive operations
- [ ] Performance timing decorators added
- [ ] Resource cleanup (context managers, try/finally)

### Testing & Documentation
- [ ] Unit tests written for new functions
- [ ] Integration tests for external dependencies
- [ ] Mock objects used for external services
- [ ] README updates if functionality changes

**If ANY checklist item is missing, STOP and complete it before proceeding.**

## 21. Standard Terminal Command Templates

**Platform Note:** These commands assume Windows PowerShell. On Unix-like systems (Linux/macOS), replace `venv\Scripts\activate` with `source venv/bin/activate`.

**MUST use these exact command patterns for Windows PowerShell:**

### Environment Management
```powershell
# Activate virtual environment (REQUIRED before any Python operation)
venv\Scripts\activate

# Verify Python version
venv\Scripts\activate; python --version

# Create new virtual environment
python -m venv venv
```

### Package Management  
```powershell
# Install specific package and update requirements
venv\Scripts\activate; pip install package_name; pip freeze > requirements.txt

# Install from requirements.txt
venv\Scripts\activate; pip install -r requirements.txt

# Upgrade package
venv\Scripts\activate; pip install --upgrade package_name
```

### Code Execution
```powershell
# Run Python script
venv\Scripts\activate; python script_name.py

# Run with arguments
venv\Scripts\activate; python main.py --arg1 value1 --arg2 value2

# Run module
venv\Scripts\activate; python -m module_name
```

### Testing
```powershell
# Run all tests with verbose output
venv\Scripts\activate; python -m pytest tests/ -v

# Run specific test file
venv\Scripts\activate; python -m pytest tests/test_specific.py -v

# Run with coverage
venv\Scripts\activate; python -m pytest tests/ --cov=src --cov-report=html
```

### Development Tools
```powershell
# Start Jupyter Notebook
venv\Scripts\activate; jupyter notebook

# Format code with black
venv\Scripts\activate; black src/ tests/

# Lint with flake8
venv\Scripts\activate; flake8 src/ tests/
```

**CRITICAL: Always include `venv\Scripts\activate;` prefix for Windows PowerShell commands.**

**Starting FastAPI:**
```bash
venv\Scripts\activate; uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Remember: Every terminal operation MUST begin with virtual environment activation!
