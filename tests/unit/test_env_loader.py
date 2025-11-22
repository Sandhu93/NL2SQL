"""
Module: test_env_loader.py
Description: Unit tests for environment variable loading and validation
Dependencies: pytest, unittest.mock
Author: AI Assistant
Created: 2025-11-22
Last Modified: 2025-11-22
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.env_loader import (
    load_environment_variables,
    setup_openai_api,
    setup_langsmith_tracing,
    REQUIRED_ENV_VARS,
    OPTIONAL_ENV_VARS
)


class TestEnvLoader:
    """Test environment variable loading and validation."""
    
    def test_load_environment_variables_success(self, mock_env_vars):
        """Test successful loading of environment variables."""
        # Arrange
        expected_keys = set(list(REQUIRED_ENV_VARS.keys()) + list(OPTIONAL_ENV_VARS.keys()))
        
        # Act
        with patch('app.utils.env_loader.load_dotenv'):
            result = load_environment_variables()
        
        # Assert
        assert isinstance(result, dict)
        assert result['OPENAI_API_KEY'] == 'sk-test-key-12345'
        assert result['DB_USER'] == 'test_user'
        assert result['DB_PASSWORD'] == 'test_password%23'
        assert result['DB_HOST'] == 'localhost'
        assert result['DB_NAME'] == 'test_classicmodels'
        assert result['DB_PORT'] == '3306'
    
    def test_load_environment_variables_missing_required(self):
        """Test error when required environment variables are missing."""
        # Arrange
        incomplete_env = {'DB_USER': 'test', 'DB_HOST': 'localhost'}
        
        # Act & Assert
        with patch('app.utils.env_loader.load_dotenv'), \
             patch.dict(os.environ, incomplete_env, clear=True):
            
            with pytest.raises(ValueError) as exc_info:
                load_environment_variables()
            
            assert "required environment variable" in str(exc_info.value).lower()
    
    def test_load_environment_variables_with_defaults(self):
        """Test loading with optional variables using defaults."""
        # Arrange
        minimal_env = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'user',
            'DB_PASSWORD': 'pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test',
            'DB_PORT': '3306'
        }
        
        # Act
        with patch('app.utils.env_loader.load_dotenv'), \
             patch.dict(os.environ, minimal_env, clear=True):
            result = load_environment_variables()
        
        # Assert
        assert result['LANGSMITH_TRACING'] == 'false'  # Default value
    
    def test_setup_openai_api_success(self):
        """Test successful OpenAI API setup."""
        # Arrange
        api_key = 'sk-test-key'

        # Act
        setup_openai_api(api_key)

        # Assert
        assert os.environ.get('OPENAI_API_KEY') == api_key
    
    def test_setup_openai_api_with_empty_key(self):
        """Test OpenAI API setup with empty key."""
        # Arrange & Act
        setup_openai_api('')
        
        # Assert - empty key is still set (validation happens elsewhere)
        assert os.environ.get('OPENAI_API_KEY') == ''
    
    def test_setup_langsmith_tracing_enabled(self):
        """Test LangSmith tracing setup when enabled."""
        # Arrange
        api_key = 'test_key'
        
        # Act
        setup_langsmith_tracing(api_key, 'true')
        
        # Assert
        assert os.environ.get('LANGCHAIN_TRACING_V2') == 'true'
        assert os.environ.get('LANGCHAIN_API_KEY') == api_key
    
    def test_setup_langsmith_tracing_disabled(self):
        """Test LangSmith tracing setup when disabled."""        
        # Arrange - Clean environment
        clean_env = {k: v for k, v in os.environ.items() 
                    if k not in ['LANGCHAIN_TRACING_V2', 'LANGCHAIN_API_KEY']}
        
        # Act
        with patch.dict(os.environ, clean_env, clear=True):
            setup_langsmith_tracing('test_key', 'false')
            
            # Assert - These variables should not be set when tracing is disabled
            assert 'LANGCHAIN_TRACING_V2' not in os.environ
            assert 'LANGCHAIN_API_KEY' not in os.environ
    
    def test_environment_validation_edge_cases(self):
        """Test edge cases in environment validation."""
        # Test with whitespace in values
        env_with_whitespace = {
            'OPENAI_API_KEY': '  sk-test-key  ',
            'DB_USER': ' test_user ',
            'DB_PASSWORD': 'pass ',
            'DB_HOST': ' localhost',
            'DB_NAME': 'test',
            'DB_PORT': '3306'
        }
        
        with patch('app.utils.env_loader.load_dotenv'), \
             patch.dict(os.environ, env_with_whitespace, clear=True):
            result = load_environment_variables()
        
        # Values are returned as-is (no stripping in current implementation)
        assert result['OPENAI_API_KEY'] == '  sk-test-key  '
        assert result['DB_USER'] == ' test_user '


class TestEnvLoaderErrorHandling:
    """Test error handling in environment loader."""
    
    def test_load_with_malformed_env_file(self, temp_env_file):
        """Test handling of malformed .env file."""
        # Arrange - Create malformed .env file
        Path(temp_env_file).write_text("INVALID_FORMAT_NO_EQUALS")
        
        # Act & Assert
        with patch('app.utils.env_loader.load_dotenv') as mock_load:
            mock_load.side_effect = Exception("Malformed env file")
            
            with pytest.raises(Exception):
                load_environment_variables()
    
    def test_setup_with_special_characters(self):
        """Test setup with special characters in API key."""
        # Arrange
        special_key = 'sk-test$key&123'
        
        # Act
        setup_openai_api(special_key)
        
        # Assert
        assert os.environ.get('OPENAI_API_KEY') == special_key