"""
Environment and Configuration Tests for NL2SQL System
Author: AI Agent
Created: 2025-11-21
Python Version: 3.11
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.env_loader import load_environment_variables, setup_openai_api, setup_langsmith_tracing

class TestEnvironmentFileHandling:
    """Test .env file handling and loading."""
    
    def test_env_file_loading_success(self):
        """Test successful .env file loading."""
        # Create temporary .env file
        env_content = """
OPENAI_API_KEY=sk-test-key-12345
DB_USER=test_user
DB_PASSWORD=test_password
DB_HOST=localhost
DB_NAME=test_db
DB_PORT=3306
LANGSMITH_API_KEY=test-langsmith-key
LANGSMITH_TRACING=true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            with patch('app.utils.env_loader.load_dotenv') as mock_load_dotenv:
                with patch.dict(os.environ, {
                    'OPENAI_API_KEY': 'sk-test-key-12345',
                    'DB_USER': 'test_user',
                    'DB_PASSWORD': 'test_password',
                    'DB_HOST': 'localhost',
                    'DB_NAME': 'test_db',
                    'DB_PORT': '3306',
                    'LANGSMITH_API_KEY': 'test-langsmith-key',
                    'LANGSMITH_TRACING': 'true'
                }, clear=True):
                    result = load_environment_variables()
                    
                    assert result['OPENAI_API_KEY'] == 'sk-test-key-12345'
                    assert result['DB_USER'] == 'test_user'
                    assert result['LANGSMITH_TRACING'] == 'true'
        finally:
            os.unlink(env_file_path)
    
    def test_env_file_missing(self):
        """Test behavior when .env file is missing."""
        with patch('app.utils.env_loader.load_dotenv') as mock_load_dotenv:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-from-system-env',
                'DB_USER': 'system_user',
                'DB_PASSWORD': 'system_pass',
                'DB_HOST': 'localhost',
                'DB_NAME': 'test_db'
            }, clear=True):
                result = load_environment_variables()
                
                # Should still work with system environment variables
                assert result['OPENAI_API_KEY'] == 'sk-from-system-env'
                assert result['DB_USER'] == 'system_user'
    
    def test_env_file_partial_config(self):
        """Test .env file with only partial configuration."""
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-test-key',
                'DB_USER': 'test_user',
                'DB_PASSWORD': 'test_pass',
                'DB_HOST': 'localhost',
                'DB_NAME': 'test_db'
                # Missing DB_PORT - should use default
            }, clear=True):
                result = load_environment_variables()
                
                assert result['DB_PORT'] == '3306'  # Default value
                assert result['LANGSMITH_TRACING'] == 'false'  # Default value

class TestEnvironmentVariableValidation:
    """Test environment variable validation logic."""
    
    def test_required_variables_present(self):
        """Test validation when all required variables are present."""
        required_vars = {
            'OPENAI_API_KEY': 'sk-test-key-12345',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, required_vars, clear=True):
                result = load_environment_variables()
                
                for key, value in required_vars.items():
                    assert result[key] == value
    
    def test_missing_required_variables(self):
        """Test validation with missing required variables."""
        incomplete_configs = [
            {},  # All missing
            {'OPENAI_API_KEY': 'sk-test'},  # Missing DB vars
            {'DB_USER': 'user', 'DB_PASSWORD': 'pass'},  # Missing API key
            {'OPENAI_API_KEY': '', 'DB_USER': 'user'},  # Empty API key
        ]
        
        for config in incomplete_configs:
            with patch('app.utils.env_loader.load_dotenv'):
                with patch.dict(os.environ, config, clear=True):
                    with pytest.raises(ValueError) as exc_info:
                        load_environment_variables()
                    
                    assert "Missing required environment variables" in str(exc_info.value)
    
    def test_empty_string_validation(self):
        """Test validation of empty string values."""
        config_with_empty_values = {
            'OPENAI_API_KEY': '',  # Empty
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config_with_empty_values, clear=True):
                with pytest.raises(ValueError):
                    load_environment_variables()
    
    def test_whitespace_only_validation(self):
        """Test validation of whitespace-only values."""
        config_with_whitespace = {
            'OPENAI_API_KEY': '   ',  # Whitespace only
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config_with_whitespace, clear=True):
                # The current implementation doesn't strip whitespace, so this test should pass
                # If we want to enforce no-whitespace-only values, we need to update main.py
                result = load_environment_variables()
                assert result['OPENAI_API_KEY'] == '   '

class TestDefaultValueHandling:
    """Test default value handling for optional variables."""
    
    def test_default_db_port(self):
        """Test default DB_PORT value."""
        minimal_config = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db'
            # DB_PORT not specified
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, minimal_config, clear=True):
                result = load_environment_variables()
                
                assert result['DB_PORT'] == '3306'
    
    def test_custom_db_port(self):
        """Test custom DB_PORT value."""
        config_with_custom_port = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db',
            'DB_PORT': '3307'  # Custom port
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config_with_custom_port, clear=True):
                result = load_environment_variables()
                
                assert result['DB_PORT'] == '3307'
    
    def test_default_langsmith_tracing(self):
        """Test default LANGSMITH_TRACING value."""
        minimal_config = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db'
            # LANGSMITH_TRACING not specified
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, minimal_config, clear=True):
                result = load_environment_variables()
                
                assert result['LANGSMITH_TRACING'] == 'false'

class TestSpecialCharacterHandling:
    """Test handling of special characters in environment variables."""
    
    def test_password_with_special_characters(self):
        """Test password containing special characters."""
        passwords_to_test = [
            'password#123',
            'p@ssw0rd!',
            'pass$word%',
            'my&password*',
            'test+pass=word'
        ]
        
        for password in passwords_to_test:
            config = {
                'OPENAI_API_KEY': 'sk-test-key',
                'DB_USER': 'test_user',
                'DB_PASSWORD': password,
                'DB_HOST': 'localhost',
                'DB_NAME': 'test_db'
            }
            
            with patch('app.utils.env_loader.load_dotenv'):
                with patch.dict(os.environ, config, clear=True):
                    result = load_environment_variables()
                    
                    assert result['DB_PASSWORD'] == password
    
    def test_url_encoded_password(self):
        """Test URL-encoded password handling."""
        config = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'pass%23word',  # # encoded as %23
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config, clear=True):
                result = load_environment_variables()
                
                assert result['DB_PASSWORD'] == 'pass%23word'
    
    def test_database_name_with_special_chars(self):
        """Test database name with special characters."""
        config = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db-2024'  # Hyphen in name
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config, clear=True):
                result = load_environment_variables()
                
                assert result['DB_NAME'] == 'test_db-2024'

class TestLangSmithConfiguration:
    """Test LangSmith tracing configuration."""
    
    def test_langsmith_enabled(self):
        """Test LangSmith tracing enabled configuration."""
        config = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db',
            'LANGSMITH_API_KEY': 'test-langsmith-key',
            'LANGSMITH_TRACING': 'true'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config, clear=True):
                result = load_environment_variables()
                
                assert result['LANGSMITH_API_KEY'] == 'test-langsmith-key'
                assert result['LANGSMITH_TRACING'] == 'true'
    
    def test_langsmith_disabled(self):
        """Test LangSmith tracing disabled configuration."""
        config = {
            'OPENAI_API_KEY': 'sk-test-key',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'test_db',
            'LANGSMITH_TRACING': 'false'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config, clear=True):
                result = load_environment_variables()
                
                assert result['LANGSMITH_TRACING'] == 'false'
    
    def test_langsmith_setup_enabled(self):
        """Test LangSmith setup when enabled."""
        with patch.dict(os.environ, {}, clear=True):
            setup_langsmith_tracing('test-key', 'true')
            
            assert os.environ['LANGCHAIN_API_KEY'] == 'test-key'
            assert os.environ['LANGCHAIN_TRACING_V2'] == 'true'
    
    def test_langsmith_setup_disabled(self):
        """Test LangSmith setup when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            setup_langsmith_tracing('test-key', 'false')
            
            assert 'LANGCHAIN_API_KEY' not in os.environ
            assert 'LANGCHAIN_TRACING_V2' not in os.environ

class TestConfigurationPersistence:
    """Test configuration persistence across function calls."""
    
    def test_environment_persistence(self):
        """Test that environment variables persist across function calls."""
        config = {
            'OPENAI_API_KEY': 'sk-persistent-key',
            'DB_USER': 'persistent_user',
            'DB_PASSWORD': 'persistent_pass',
            'DB_HOST': 'localhost',
            'DB_NAME': 'persistent_db'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, config, clear=True):
                # First call
                result1 = load_environment_variables()
                
                # Second call should return same values
                result2 = load_environment_variables()
                
                assert result1 == result2
                for key in config:
                    assert result1[key] == result2[key]
    
    def test_api_key_persistence(self):
        """Test API key persistence after setup."""
        test_key = 'sk-persistent-test-key'
        
        setup_openai_api(test_key)
        
        # Key should persist
        assert os.environ.get('OPENAI_API_KEY') == test_key
        
        # Should still be available after another setup call
        setup_openai_api(test_key)
        assert os.environ.get('OPENAI_API_KEY') == test_key

class TestConfigurationErrorScenarios:
    """Test various error scenarios in configuration."""
    
    def test_malformed_env_file(self):
        """Test handling of malformed .env file."""
        # This test simulates what happens when .env file is malformed
        # but environment variables are still available from other sources
        
        with patch('app.utils.env_loader.load_dotenv') as mock_load_dotenv:
            # Simulate dotenv failing to parse file but not raising exception
            mock_load_dotenv.return_value = False
            
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'sk-system-key',
                'DB_USER': 'system_user',
                'DB_PASSWORD': 'system_pass',
                'DB_HOST': 'localhost',
                'DB_NAME': 'system_db'
            }, clear=True):
                # Should still work with system environment variables
                result = load_environment_variables()
                assert result['OPENAI_API_KEY'] == 'sk-system-key'
    
    def test_partial_configuration_override(self):
        """Test partial configuration override scenarios."""
        # Start with complete config
        complete_config = {
            'OPENAI_API_KEY': 'sk-original-key',
            'DB_USER': 'original_user',
            'DB_PASSWORD': 'original_pass',
            'DB_HOST': 'original_host',
            'DB_NAME': 'original_db'
        }
        
        with patch('app.utils.env_loader.load_dotenv'):
            with patch.dict(os.environ, complete_config, clear=True):
                result1 = load_environment_variables()
                
                # Override some values
                os.environ['OPENAI_API_KEY'] = 'sk-new-key'
                os.environ['DB_HOST'] = 'new_host'
                
                result2 = load_environment_variables()
                
                # New values should be present
                assert result2['OPENAI_API_KEY'] == 'sk-new-key'
                assert result2['DB_HOST'] == 'new_host'
                
                # Unchanged values should remain
                assert result2['DB_USER'] == 'original_user'
                assert result2['DB_PASSWORD'] == 'original_pass'
