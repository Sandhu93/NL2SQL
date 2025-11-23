# NL2SQL Tutorial with LangChain

A comprehensive tutorial for building Natural Language to SQL applications using LangChain, OpenAI, and MySQL database. This project demonstrates how to connect natural language queries to SQL databases with robust error handling and environment management.

**Author**: Sandeep B Kadam  
**Created**: November 21, 2025  
**Status**: Active Development  
**Current Version**: 0.0.1 (dev)

## Learning Resources & Attribution

This project follows the comprehensive tutorial **"Mastering Natural Language to SQL with LangChain"** by Pradip Nichite:

- 📝 **Tutorial Blog**: [FutureSmart AI - NL2SQL with LangChain](https://blog.futuresmart.ai/mastering-natural-language-to-sql-with-langchain-nl2sql)
- 📺 **Video Tutorial**: [YouTube - NL2SQL Tutorial](https://www.youtube.com/watch?v=fss6CrmQU2Y)
- 💻 **Reference Notebook**: [GitHub - Langchain_NL2SQL_2024.ipynb](https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Langchain_NL2SQL_2024.ipynb)

### What You'll Learn

This tutorial covers building production-ready NL2SQL systems from basics to advanced concepts:

1. **Building a Basic NL2SQL Model**
   - Understanding NL2SQL fundamentals
   - Setting up LangChain with OpenAI integration
   - Creating your first natural language query
   - Executing SQL and viewing results

2. **Rephrasing Answers for Enhanced Clarity** ✅ Implemented
   - Converting raw SQL results to natural language responses
   - Using prompt templates for user-friendly output
   - Implementing answer rephrasing with LangChain chains
   - Integrated RunnablePassthrough for streamlined processing

3. **Enhancing Models with Few-Shot Learning** ✅ Implemented
   - Incorporating example queries for better accuracy
   - Creating few-shot learning templates with FewShotChatMessagePromptTemplate
   - Improving model understanding with curated examples
   - Building dynamic few-shot prompts with system context

4. **Dynamic Few-Shot Example Selection**
   - Semantic similarity-based example selection
   - Using vector embeddings (ChromaDB/FAISS) for context matching
   - Tailoring examples to query context automatically

5. **Dynamic Relevant Table Selection**
   - Optimizing for databases with 100+ tables
   - Reducing prompt token usage and costs
   - Improving performance with focused table selection
   - Using table descriptions for intelligent filtering

6. **Adding Memory for Conversational Context**
   - Implementing chat message history
   - Handling follow-up questions intelligently
   - Maintaining conversational context across queries
   - Building interactive chatbot experiences

### Tutorial Author

**Pradip Nichite** - Top Rated Plus NLP freelancer and founder of FutureSmart AI
- Specializes in custom NLP solutions using LangChain, Transformers, and Vector Databases
- YouTube Channel: [@aidemos.videos](https://www.youtube.com/@aidemos.videos)
- Website: [aidemos.com](http://aidemos.com/)

## System Requirements

### Python Version
**Python 3.11 is required** for this project. Python 3.12 is not supported due to compatibility issues with ChromaDB and related C++ dependencies.

### Operating System
- Windows 10/11 (tested)
- macOS (should work)
- Linux (should work)

## Installation Instructions

### Step 1: Check Python Version
```bash
python --version
```
Ensure you have Python 3.11.x installed. If not, download and install Python 3.11 from [python.org](https://www.python.org/downloads/).

### Step 2: Clone or Download the Project
```bash
git clone <repository-url>
cd NL2SQL_tutorial
```

### Step 3: Create Virtual Environment
```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

## Key Dependencies

### Core Framework
- **LangChain 0.3.7**: Main framework for LLM applications
- **LangChain OpenAI 0.2.8**: OpenAI integration
- **LangChain Community 0.3.7**: Community extensions and integrations

### Vector Databases
- **ChromaDB 0.5.18**: Vector database for embeddings
- **FAISS CPU 1.13.0**: Facebook AI Similarity Search
- **LangChain Chroma 0.1.4**: ChromaDB integration for LangChain

### Web Framework
- **FastAPI 0.115.5**: Modern web framework for APIs
- **Uvicorn 0.32.0**: ASGI server for FastAPI

### Database
- **PyMySQL 1.1.1**: MySQL database connector
- **SQLAlchemy 2.0.35**: SQL toolkit and ORM  
- **Cryptography 46.0.3**: Required for MySQL authentication

### AI/ML
- **OpenAI 1.109.1**: OpenAI API client
- **TikToken 0.12.0**: Tokenizer for OpenAI models
- **NumPy 1.26.4**: Numerical computing

### Development Tools
- **Python-dotenv**: Environment variable management
- **Logging**: Comprehensive application logging

## Project Structure

```
NL2SQL_tutorial/
├── venv/                          # Virtual environment (Python 3.11)
├── .env                           # Environment variables (not in git)
├── .gitignore                     # Git ignore rules
├── .pytest_cache/                 # Pytest cache directory
├── __pycache__/                   # Python bytecode cache
│
├── app/                           # Main application package
│   ├── __init__.py               
│   ├── db/                        # Database layer
│   │   ├── __init__.py
│   │   └── database_manager.py    # MySQL connection & utilities
│   ├── services/                  # Business logic layer
│   │   ├── __init__.py
│   │   └── nl2sql_service.py      # NL2SQL conversion service
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── constants.py           # Application constants
│       ├── env_loader.py          # Environment variable management
│       └── logging_config.py      # Logging configuration
│
├── tests/                         # Comprehensive test suite
│   ├── conftest.py               # Pytest fixtures and configuration
│   ├── pytest.log                # Test execution logs
│   ├── unit/                     # Unit tests (isolated component testing)
│   │   ├── test_database_manager.py
│   │   ├── test_env_loader.py
│   │   ├── test_logging_config.py
│   │   └── test_nl2sql_service.py
│   ├── integration/              # Integration tests (component interaction)
│   │   ├── test_main_integration.py
│   │   └── test_service_database_integration.py
│   ├── test_config.py           # Legacy test files
│   ├── test_database.py
│   ├── test_integration.py
│   ├── test_main.py
│   └── test_openai.py
│
├── AI_AGENT_RULES.md             # AI coding guidelines & testing strategy
├── DEVELOPMENT_RULES.md          # Development standards & git workflow
├── TESTING.md                    # Comprehensive testing documentation
├── Langchain_NL2SQL_2024.ipynb   # Main tutorial notebook
├── main.py                       # Application entry point
├── run_tests.py                  # Test execution script
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Python dependencies (129 packages)
├── nl2sql.log                    # Application logs
└── README.md                     # This file
```

### Architecture Layers

**1. Database Layer (`app/db/`)**
- MySQL connection management
- Database URI construction
- Schema exploration utilities
- Connection validation

**2. Services Layer (`app/services/`)**
- NL2SQL conversion logic
- Query chain creation
- Query execution
- Result processing
- Answer rephrasing with prompt templates
- RunnablePassthrough chain integration
- Few-shot learning with curated examples
- Dynamic prompt building with example selectors

**3. Utilities Layer (`app/utils/`)**
- Environment variable loading & validation
- Logging configuration & setup
- Application constants
- Reusable helper functions

**4. Testing Layer (`tests/`)**
- **Unit Tests**: Isolated component testing with mocks
- **Integration Tests**: End-to-end workflow testing
- **Fixtures**: Reusable test data and mocks in `conftest.py`
- **Coverage**: Comprehensive test coverage with pytest

## Environment Setup

### Required: Create `.env` File
Create a `.env` file in the project root directory with the following variables:

```env
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration (Required)
DB_USER=root
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=classicmodels

# Optional: LangSmith Tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
```

### Special Characters in Passwords
If your database password contains special characters like `#`, `@`, `%`, etc., URL encode them:
- `#` becomes `%23`
- `@` becomes `%40`
- `%` becomes `%25`
- ` ` (space) becomes `%20`

**Example**: `mypassword#123` should be written as `mypassword%23123`

### Environment Variable Validation
The application automatically validates all required environment variables on startup and provides clear error messages for missing configurations.

## Usage

### Running the Application

The main entry point is `main.py`, which provides a complete NL2SQL pipeline demonstration with answer rephrasing:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the application
python main.py
```

### Current Implementation: NL2SQL Query System with Answer Rephrasing
The current `main.py` provides a complete NL2SQL pipeline with natural language answer generation:

#### Database Setup
- Complete database connection setup
- Environment variable validation
- Database schema exploration
- Connection testing and logging

#### Natural Language Processing
- **GPT-3.5-turbo Integration**: Converts natural language to SQL
- **Query Chain Creation**: Uses LangChain's `create_sql_query_chain`
- **Automatic SQL Generation**: Intelligent query construction
- **Query Execution**: Direct database query execution with results
- **Answer Rephrasing**: Converts raw SQL results to user-friendly natural language
- **RunnablePassthrough Chains**: Streamlined processing pipeline
- **Few-Shot Learning**: Curated example queries guide SQL generation
- **Interactive Query Runner**: Menu-driven interface for testing queries

```bash
# Activate virtual environment and run
venv\Scripts\activate
python main.py
```

**Expected Output:**
- Database connection validation and schema information
- **First Query Test**: "what is price of `1968 Ford Mustang`"
  - Generated SQL: `SELECT buyPrice FROM products WHERE productName = '1968 Ford Mustang'`
  - Result: `$95.34`
- **Additional Query Tests**:
  - "Show me all products priced above $100" → Returns high-value products
  - "How many customers are there?" → Returns total customer count (122)
  - "List all offices in the USA" → Returns all US office locations
- **Refined Query Test**: "How many customers have an order count greater than 5?"
  - Generated SQL query
  - Raw SQL results
  - Rephrased natural language answer
- **Interactive Query Runner**: Menu-driven interface
  - Option 1: Run first query demo with additional samples
  - Option 2: Enter your own natural language questions
  - Option 3: Use few-shot learning guidance for improved SQL generation
  - Option 4: Exit interactive mode
  - Real-time query processing with SQL generation, execution, and answer rephrasing

### Query Examples Working Now
```python
# Natural Language → SQL → Results
"what is price of 1968 Ford Mustang" → "SELECT buyPrice FROM products..." → $95.34
"Show me all products priced above $100" → "SELECT * FROM products WHERE..." → Product list
"How many customers are there?" → "SELECT COUNT(*) FROM customers" → 122
"List all offices in the USA" → "SELECT * FROM offices WHERE country='USA'" → Office list

# With Answer Rephrasing
"How many customers have an order count greater than 5?"
  → SQL: "SELECT COUNT(*) FROM customers..."
  → Raw Result: "[(42,)]"
  → Rephrased: "There are 42 customers who have placed more than 5 orders."

# Interactive Query Runner
User interaction:
  Select an option [1/2/3/4]: 2
  Enter your question: How many products are in stock?
  → Generated SQL Query: SELECT COUNT(*) FROM products WHERE quantityInStock > 0
  → Raw Query Results: [(110,)]
  → Rephrased Answer: There are 110 products currently in stock.

# Few-Shot Learning
Curated examples guide the model:
  Example 1: "List all customers in France with a credit limit over 20,000"
  → SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;
  
  Example 2: "Get the highest payment amount made by any customer"
  → SELECT MAX(amount) FROM payments;
  
User query with few-shot guidance (Option 3):
  Enter your question: Show me products that cost more than $150
  → Generated SQL Query: SELECT * FROM products WHERE buyPrice > 150;
  → Uses example patterns for better accuracy
```

### Future Implementations (Coming Soon)
- Advanced query result formatting
- Query history and user sessions
- Web API endpoints with FastAPI
- Real-time query suggestions
- Query validation and optimization hints

### Running Jupyter Notebook
```bash
venv\Scripts\activate
jupyter notebook Langchain_NL2SQL_2024.ipynb
```

## Database Setup

### Supported Database: MySQL ClassicModels
The project uses the **ClassicModels** sample database with the following tables:
- **customers**: Customer information and contact details
- **employees**: Employee data and organizational structure  
- **offices**: Office locations and contact information
- **orders**: Order records and status tracking
- **orderdetails**: Individual order line items
- **payments**: Payment transactions and history
- **products**: Product catalog and inventory
- **productlines**: Product categories and descriptions

### Database Connection Requirements
- MySQL 8.0+ (tested with 8.0.44)
- ClassicModels sample database installed
- User with SELECT permissions (minimum)
- Network access to MySQL server

## Troubleshooting

### Database Connection Issues

**Error**: `Access denied for user 'root'@'localhost'`
- Verify credentials in `.env` file
- Check if MySQL server is running
- Ensure user has proper permissions

**Error**: `Can't connect to MySQL server`
- Verify `DB_HOST` and `DB_PORT` in `.env`
- Check MySQL server status
- Verify network connectivity

**Error**: `Unknown database 'classicmodels'`
- Install ClassicModels sample database
- Verify database name in `.env` file

### Python 3.12 Compatibility Issues
If you encounter build errors with ChromaDB or chroma-hnswlib, ensure you are using Python 3.11. Python 3.12 has compatibility issues with C++ dependencies.

### Missing Cryptography Package
If you see `'cryptography' package is required for sha256_password`:
```bash
venv\Scripts\activate
pip install cryptography
```

### Visual Studio Build Tools (Windows)
If you see C++ compilation errors:
1. Install Visual Studio Build Tools 2022
2. Include Windows 10/11 SDK  
3. Include MSVC v143 compiler toolset

### Environment Variable Issues
Check the application logs (`nl2sql.log`) for detailed error messages about missing or invalid environment variables.

## Current Features

### ✅ Implemented
- **Layered Architecture**: Clean separation of database, services, and utilities layers
- **Database Connection**: Robust MySQL connection with error handling
- **Environment Management**: Secure credential management via .env files
- **Schema Exploration**: Complete database schema analysis and display
- **Natural Language to SQL**: Convert natural language questions to SQL queries
- **Query Execution**: Execute generated SQL queries and return results
- **Answer Rephrasing**: Convert raw SQL results to natural language responses
- **Prompt Templates**: User-friendly answer formatting with LangChain prompts
- **RunnablePassthrough Chains**: Streamlined query-to-answer pipeline
- **Few-Shot Learning**: Curated example queries improve SQL generation accuracy
- **FewShotChatMessagePromptTemplate**: LangChain few-shot prompt integration
- **Interactive Query Runner**: Menu-driven interface for testing and exploring queries
- **OpenAI Integration**: GPT-3.5-turbo for intelligent query generation
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Error Handling**: Graceful error handling with specific error messages
- **Password Security**: URL encoding/decoding for special characters
- **Connection Validation**: Automatic database connection testing
- **Multiple Vector Stores**: ChromaDB, FAISS, and LangChain Chroma support
- **Testing Suite**: Comprehensive unit and integration tests with pytest (20/20 tests passing)

### 🚧 In Development
- Advanced query result formatting
- FastAPI web interface with RESTful endpoints
- Query history and caching
- Advanced error recovery
- Query optimization suggestions
- Dynamic few-shot example selection with semantic similarity

### 📋 Planned Features
- Real-time query suggestions
- Query history and caching
- Multi-database support
- Advanced NL understanding
- Query performance analytics
- User authentication and sessions

## Development

### Versioning and Release Flow
- Semantic Versioning: `MAJOR.MINOR.PATCH`
  - MAJOR: Backward-incompatible changes
  - MINOR: Backward-compatible features
  - PATCH: Backward-compatible fixes/docs
- Branch roles:
  - `main` = production (tagged releases only)
  - `staging` = pre-production validation
  - `develop` = integration branch for daily work
  - `feature/<name>` or `fix/<name>` = branch from `develop`, merge via PR
- Release flow:
  1. Cut `release/x.y.z` from `develop` and deploy to `staging`.
  2. After sign-off, merge `release/x.y.z` into `main`, tag `vX.Y.Z`, deploy to production.
  3. Back-merge `release/x.y.z` into `develop` to keep history aligned.

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions and classes

### Testing
The project includes a comprehensive test suite with both unit and integration tests:

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/unit/test_database_manager.py

# Run with coverage report
python -m pytest --cov=app --cov-report=html

# Run only unit tests
python -m pytest tests/unit/

# Run only integration tests
python -m pytest tests/integration/
```

**Test Coverage:**
- ✅ **Unit Tests**: Database manager, environment loader, logging config, NL2SQL service
- ✅ **Integration Tests**: End-to-end workflow, service-database interaction
- ✅ **Fixtures**: Comprehensive mocking in `conftest.py`
- 📊 **Target Coverage**: 80%+ for all modules

See `TESTING.md` for detailed testing guidelines and `AI_AGENT_RULES.md` for testing strategy.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check this README first
2. Review the Jupyter notebook documentation
3. Create an issue in the repository

## Development Guidelines

This project includes comprehensive development documentation:
- **AI_AGENT_RULES.md**: Guidelines for AI agents, testing strategy, and code quality standards
- **DEVELOPMENT_RULES.md**: Development standards, git workflow, and versioning rules
- **TESTING.md**: Comprehensive testing documentation, fixtures, and test patterns

### Code Quality Standards
- Python 3.11 required
- Type hints mandatory for all functions
- Comprehensive error handling
- Detailed logging with structured format
- Environment variable validation
- Security-first approach (no hardcoded credentials)
- PEP 8 compliance
- Comprehensive docstrings for all modules, classes, and functions

## Version History

### 0.0.1 (Current)
- Baseline NL2SQL pipeline: environment validation, MySQL connectivity, schema display.
- GPT-3.5-turbo integration with LangChain query chain and execution.
- First-query demo plus additional sample queries.
- Answer rephrasing functionality with prompt templates and RunnablePassthrough chains.
- Natural language response generation from raw SQL results.
- Interactive query runner with menu-driven interface for ad-hoc query testing.
- Few-shot learning with curated example queries (4 examples covering various SQL patterns).
- FewShotChatMessagePromptTemplate integration for guided SQL generation.
- Dynamic prompt building with example selectors (static and semantic similarity ready).
- Four demo modes: first query test, refined query with rephrasing, few-shot learning, and interactive mode.
- Comprehensive logging, error handling, and password URL-encoding support.
- Full test coverage (20/20 unit tests passing).

> The project now follows semantic versioning from this baseline. Tag v0.0.1 when promoting the current code to production.

