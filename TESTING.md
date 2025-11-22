# Testing Guide

This document describes how to run tests for the NL2SQL Tutorial project.

## Test Structure

The project follows a layered testing approach:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_env_loader.py   # Environment variable loading tests
│   ├── test_database_manager.py  # Database connection tests
│   ├── test_nl2sql_service.py    # NL2SQL service logic tests
│   └── test_logging_config.py    # Logging configuration tests
└── integration/             # Integration tests for component interactions
    ├── test_main_integration.py           # Main application integration tests
    └── test_service_database_integration.py  # Service-database integration tests
```

## Running Tests

### Prerequisites

Ensure your virtual environment is activated and dependencies are installed:

```powershell
venv\Scripts\activate; pip install -r requirements.txt
```

### Unit Tests

Run all unit tests with verbose output:

```powershell
venv\Scripts\activate; python -m pytest tests/unit -v
```

Run a specific unit test file:

```powershell
venv\Scripts\activate; python -m pytest tests/unit/test_nl2sql_service.py -v
```

Run a specific test function:

```powershell
venv\Scripts\activate; python -m pytest tests/unit/test_nl2sql_service.py::TestNL2SQLService::test_generate_sql_success -v
```

### Integration Tests

Run all integration tests:

```powershell
venv\Scripts\activate; python -m pytest tests/integration -v
```

Run a specific integration test file:

```powershell
venv\Scripts\activate; python -m pytest tests/integration/test_main_integration.py -v
```

### All Tests

Run the entire test suite:

```powershell
venv\Scripts\activate; python -m pytest tests/ -v
```

### Test Coverage

Run tests with coverage reporting:

```powershell
venv\Scripts\activate; python -m pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
```

This generates an HTML coverage report in `htmlcov/index.html`.

## Test Categories

### Unit Tests
- **Purpose**: Test individual functions and classes in isolation
- **Dependencies**: All external dependencies (OpenAI, database, file system) are mocked
- **Speed**: Fast execution (< 1 second per test)
- **Scope**: Single module or class functionality

### Integration Tests  
- **Purpose**: Test how multiple components work together
- **Dependencies**: May use test databases or mock external services
- **Speed**: Slower execution (1-10 seconds per test)
- **Scope**: Cross-component interactions and workflows

## Environment Variables for Testing

Tests use mock environment variables defined in `conftest.py`. For integration tests that require real connections, create a `.env.test` file:

```
# Test environment variables
OPENAI_API_KEY=your_test_api_key
DB_USER=test_user
DB_PASSWORD=test_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=test_database
LANGSMITH_TRACING=false
```

## Test Fixtures

Common fixtures are available in `conftest.py`:

- `mock_env_vars`: Mock environment variables
- `mock_database_manager`: Mock database manager
- `mock_openai_response`: Mock OpenAI API responses
- `sample_queries`: Sample test queries and expected results
- `valid_sql_queries`: Valid SELECT queries for testing
- `invalid_sql_queries`: Invalid queries with dangerous keywords

## Writing New Tests

### Unit Test Example

```python
def test_generate_sql_success(self, nl2sql_service, mock_openai_response):
    """Test successful SQL generation from natural language."""
    # Arrange
    question = "What is the price of 1968 Ford Mustang?"
    nl2sql_service.llm.invoke.return_value = mock_openai_response
    
    # Act
    result = nl2sql_service.generate_sql(question)
    
    # Assert
    assert isinstance(result, dict)
    assert 'sql' in result
    assert 'confidence_score' in result
```

### Regression Test Example

```python
def test_sanitize_sql_rejects_drop_statements():
    """Regression test for bug #123: DROP statements must be rejected."""
    with pytest.raises(ValueError) as exc_info:
        sanitize_sql_input("DROP TABLE customers;")
    
    assert "only select queries are allowed" in str(exc_info.value).lower()
```

## Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `TestModuleName`
- Test functions: `test_<behavior_description>`
- Use descriptive names that explain the expected behavior

## Debugging Tests

### Run with debugging output:

```powershell
venv\Scripts\activate; python -m pytest tests/ -v -s --tb=long
```

### Run a single test with maximum verbosity:

```powershell
venv\Scripts\activate; python -m pytest tests/unit/test_nl2sql_service.py::TestNL2SQLService::test_generate_sql_success -vvv -s
```

### Use pytest debugger:

```powershell
venv\Scripts\activate; python -m pytest tests/ --pdb
```

## Performance Testing

For performance-sensitive components, add timing assertions:

```python
import time

def test_query_processing_performance(self, nl2sql_service):
    """Test that query processing completes within acceptable time."""
    start_time = time.time()
    
    result = nl2sql_service.process_question("How many customers are there?")
    
    execution_time = time.time() - start_time
    assert execution_time < 5.0  # Should complete within 5 seconds
```

## Continuous Integration

The test suite is designed to run in CI environments. Ensure:

1. All tests pass with zero exit code
2. No external dependencies (real APIs, databases) in unit tests
3. Mock all I/O operations and network calls
4. Tests are deterministic and don't rely on timing or random data

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Mock Failures**: Verify mock paths match actual import paths
3. **Fixture Errors**: Check fixture dependencies and scopes
4. **Slow Tests**: Move to integration tests if using real external services

### Getting Help

1. Check test output for specific error messages
2. Review the test file and fixture definitions
3. Ensure all dependencies are installed in the virtual environment
4. Verify environment variables are properly mocked