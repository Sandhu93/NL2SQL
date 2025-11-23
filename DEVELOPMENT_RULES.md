# Project Development Rules and Guidelines

## Virtual Environment Rules

### MANDATORY: Always Use Virtual Environment
1. **ALL terminal commands must be executed within the activated virtual environment**
2. Before any terminal operation, verify virtual environment is active by checking for `(venv)` prefix in terminal prompt
3. If virtual environment is not active, run: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Never install packages globally - always use the project's virtual environment
5. Always run `pip freeze > requirements.txt` after installing new packages

### Virtual Environment Verification
```bash
# Check if virtual environment is active
echo $VIRTUAL_ENV  # Should show path to venv directory

# Verify Python path
python -c "import sys; print(sys.executable)"  # Should point to venv/Scripts/python.exe
```

## Code Quality Standards

### Python Code Standards
1. **PEP 8 Compliance**: Follow Python Enhancement Proposal 8 style guide
2. **Type Hints**: Use type annotations for all function parameters and return values
3. **Docstrings**: All functions, classes, and modules must have descriptive docstrings
4. **Error Handling**: Implement proper exception handling with specific exception types
5. **Constants**: Use UPPER_CASE for constants and define them at module level
6. **Keep entrypoints thin**: `main.py` should only wire/boot the app; move demos, interactive flows, and business logic into services/utils instead of adding functions directly to the entrypoint.

### Code Structure Rules
```python
# Example of proper function structure
def process_sql_query(query: str, database_url: str) -> dict:
    """
    Process natural language query and convert to SQL.
    
    Args:
        query: Natural language query string
        database_url: Database connection URL
        
    Returns:
        dict: Contains SQL query and execution results
        
    Raises:
        ValueError: If query is empty or invalid
        ConnectionError: If database connection fails
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    try:
        # Implementation here
        return {"sql": sql_query, "results": results}
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise
```

## Git and Version Control Rules

### Commit Standards
1. **Commit Messages**: Use clear, descriptive commit messages
   - Format: `type(scope): description`
   - Examples: `feat(api): add new endpoint for SQL conversion`
2. **Atomic Commits**: Each commit should represent a single logical change
3. **Branch Naming**: Use descriptive branch names (`feature/nl2sql-improvement`, `fix/database-connection`)
4. **No Secrets**: Never commit API keys, passwords, or sensitive data

### Branching and Environment Flow
1. **Branch Roles**
   - `main`: Production-ready code only; tagged releases are deployed to production.
   - `staging`: Pre-production; mirrors production config and is fed by release branches for final validation.
   - `develop`: Integration branch for day-to-day work; all feature branches merge here first.
2. **Feature Work**
   - Create branches from `develop` using `feature/<short-description>` for new features.
   - Create branches from `develop` using `fix/<short-description>` for bug fixes.
   - Open PRs back into `develop` with tests and notes; no direct commits to `main`.
3. **Release Flow**
   - Cut `release/x.y.z` branches from `develop` when stabilizing a version; deploy them to `staging`.
   - After sign-off, merge `release/x.y.z` into `main`, tag the version, and deploy to production.
   - Back-merge `release/x.y.z` into `develop` to keep history aligned.

### Versioning Rules
1. **Semantic Versioning**: Use `MAJOR.MINOR.PATCH` with meaning:
   - **MAJOR**: Backward-incompatible changes.
   - **MINOR**: Backward-compatible features.
   - **PATCH**: Backward-compatible fixes and docs.
2. **Current Baseline**: Start from version `0.0.1` for the current codebase.
3. **Tagging**: Every production deployment must be tagged (`vX.Y.Z`) on `main`.
4. **Changelog**: Update README (and/or CHANGELOG if added) with each tagged release, noting changes and migration steps if any.

### File Management
1. **Always update `.gitignore`** to exclude:
   - Virtual environment (`venv/`)
   - Environment files (`.env`)
   - Cache files (`__pycache__/`, `*.pyc`)
   - IDE files (`.vscode/`, `.idea/`)
   - Log files (`*.log`)

## Environment and Configuration Rules

### Environment Variables
1. **All sensitive data must be stored in `.env` file**
2. **Provide `.env.example` template** with placeholder values
3. **Never hardcode API keys or secrets** in source code
4. **Use environment variables for configuration**

```python
# Correct way to handle environment variables
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
```

## Testing Rules

### Test Coverage Requirements
1. **Minimum 80% test coverage** for all new code
2. **Unit tests** for all functions and methods
3. **Integration tests** for database connections and API endpoints
4. **Test file naming**: `test_<module_name>.py`

### Test Structure
```python
import pytest
from unittest.mock import Mock, patch

class TestSQLConverter:
    def setup_method(self):
        """Setup test data before each test method."""
        self.converter = SQLConverter()
    
    def test_convert_simple_query(self):
        """Test basic query conversion functionality."""
        query = "Show all users"
        result = self.converter.convert(query)
        assert "SELECT * FROM users" in result
    
    @patch('module.external_api_call')
    def test_api_integration(self, mock_api):
        """Test external API integration with mocking."""
        mock_api.return_value = {"status": "success"}
        # Test implementation
```

## Database and Data Rules

### Database Operations
1. **Always use connection pooling** for database connections
2. **Implement proper transaction handling** with rollback on errors
3. **Use parameterized queries** to prevent SQL injection
4. **Close connections properly** using context managers or try/finally blocks

### Data Validation
1. **Validate all input data** before processing
2. **Sanitize user inputs** to prevent injection attacks
3. **Use Pydantic models** for data validation and serialization
4. **Log all data operations** for debugging and auditing

## Security Rules

### API Security
1. **Implement authentication** for all API endpoints
2. **Use HTTPS** for all external communications
3. **Validate and sanitize** all input parameters
4. **Implement rate limiting** to prevent abuse
5. **Log security events** and failed authentication attempts

### Data Protection
1. **Encrypt sensitive data** at rest and in transit
2. **Use secure random tokens** for session management
3. **Implement proper access controls** based on user roles
4. **Regular security audits** of dependencies using `pip audit`

## Documentation Rules

### Code Documentation
1. **README.md**: Must include installation, usage, and troubleshooting
2. **API Documentation**: Use FastAPI automatic documentation
3. **Inline Comments**: Explain complex logic and business rules
4. **Changelog**: Maintain version history and breaking changes

### Documentation Standards
```python
class NL2SQLConverter:
    """
    Converts natural language queries to SQL statements.
    
    This class provides functionality to translate human-readable
    queries into proper SQL syntax using OpenAI's language models.
    
    Attributes:
        model_name (str): OpenAI model identifier
        temperature (float): Model creativity parameter (0.0 to 1.0)
    
    Example:
        >>> converter = NL2SQLConverter(model_name="gpt-4")
        >>> sql = converter.convert("Show all active users")
        >>> print(sql)
        'SELECT * FROM users WHERE status = "active"'
    """
```

## Performance Rules

### Code Optimization
1. **Profile before optimizing** - measure performance bottlenecks
2. **Use appropriate data structures** for the task
3. **Implement caching** for expensive operations
4. **Optimize database queries** - avoid N+1 problems
5. **Use async/await** for I/O bound operations

### Resource Management
1. **Monitor memory usage** and prevent memory leaks
2. **Implement proper cleanup** for file handles and connections
3. **Use connection pooling** for database and API connections
4. **Set appropriate timeouts** for external API calls

## Error Handling and Logging Rules

### Logging Standards
```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Usage examples
logger.info("Application started")
logger.warning("Database connection slow")
logger.error("Failed to process query", exc_info=True)
```

### Exception Handling
1. **Catch specific exceptions** rather than bare `except:`
2. **Log exceptions** with full stack traces
3. **Provide meaningful error messages** to users
4. **Implement graceful degradation** when possible

## Deployment Rules

### Pre-deployment Checklist
1. **All tests pass** in clean environment
2. **Requirements.txt updated** with exact versions
3. **Environment variables documented** in README
4. **Security scan completed** with no critical issues
5. **Performance benchmarks** meet acceptance criteria

### Environment Consistency
1. **Use same Python version** across development, testing, and production
2. **Pin all package versions** in requirements.txt
3. **Document system dependencies** and OS requirements
4. **Test deployment process** in staging environment

## Terminal Command Rules

### Mandatory Virtual Environment Check
Before executing ANY terminal command:
```bash
# 1. Check if virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment not active. Activating..."
    source venv/bin/activate  # Linux/macOS
    # OR
    venv\Scripts\activate     # Windows
fi

# 2. Verify correct environment
python -c "import sys; print('Python path:', sys.executable)"
pip show langchain  # Verify project packages are available
```

### Standard Command Workflow
```bash
# Always start with virtual environment activation
venv\Scripts\activate

# Then proceed with your commands
pip install new_package
python script.py
pytest tests/
jupyter notebook
```

This rules file must be followed by all team members and automated tools working on this project.
