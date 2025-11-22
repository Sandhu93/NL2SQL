# NL2SQL Tutorial with LangChain

A comprehensive tutorial for building Natural Language to SQL applications using LangChain, OpenAI, and MySQL database. This project demonstrates how to connect natural language queries to SQL databases with robust error handling and environment management.

**Author**: Sandeep B Kadam  
**Created**: November 21, 2025  
**Status**: Active Development  
**Current Version**: 0.0.1 (dev)

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
├── AI_AGENT_RULES.md              # AI coding guidelines
├── DEVELOPMENT_RULES.md           # Development guidelines
├── Langchain_NL2SQL_2024.ipynb    # Main tutorial notebook
├── main.py                        # Database connection & setup
├── requirements.txt               # Python dependencies (129 packages)
├── nl2sql.log                     # Application logs
└── README.md                      # This file
```

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

### Current Implementation: NL2SQL Query System
The current `main.py` provides a complete NL2SQL pipeline:

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

### Query Examples Working Now
```python
# Natural Language → SQL → Results
"what is price of 1968 Ford Mustang" → "SELECT buyPrice FROM products..." → $95.34
"Show me all products priced above $100" → "SELECT * FROM products WHERE..." → Product list
"How many customers are there?" → "SELECT COUNT(*) FROM customers" → 122
"List all offices in the USA" → "SELECT * FROM offices WHERE country='USA'" → Office list
```

### Future Implementations (Coming Soon)
- Interactive query interface
- Advanced query result formatting
- Query history and user sessions
- Web API endpoints
- Real-time query suggestions

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
- **Database Connection**: Robust MySQL connection with error handling
- **Environment Management**: Secure credential management via .env files
- **Schema Exploration**: Complete database schema analysis and display
- **Natural Language to SQL**: Convert natural language questions to SQL queries
- **Query Execution**: Execute generated SQL queries and return results
- **OpenAI Integration**: GPT-3.5-turbo for intelligent query generation
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Error Handling**: Graceful error handling with specific error messages
- **Password Security**: URL encoding/decoding for special characters
- **Connection Validation**: Automatic database connection testing
- **Multiple Vector Stores**: ChromaDB, FAISS, and LangChain Chroma support

### 🚧 In Development
- Interactive query interface
- Advanced query result formatting
- FastAPI web interface  
- Query history and caching
- Advanced error recovery
- Query optimization suggestions

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
Run tests with:
```bash
python -m pytest
```

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

This project includes comprehensive development rules:
- **AI_AGENT_RULES.md**: Guidelines for AI agents working on this project
- **DEVELOPMENT_RULES.md**: General development standards and practices

### Code Quality Standards
- Python 3.11 required
- Type hints mandatory
- Comprehensive error handling
- Detailed logging
- Environment variable validation
- Security-first approach

## Version History

### 0.0.1 (Current)
- Baseline NL2SQL pipeline: environment validation, MySQL connectivity, schema display.
- GPT-3.5-turbo integration with LangChain query chain and execution.
- First-query demo plus additional sample queries.
- Comprehensive logging, error handling, and password URL-encoding support.

> The project now follows semantic versioning from this baseline. Tag v0.0.1 when promoting the current code to production.

