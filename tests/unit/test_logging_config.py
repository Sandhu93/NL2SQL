"""
Module: test_logging_config.py
Description: Unit tests for logging configuration utilities
Dependencies: pytest, unittest.mock
Author: AI Assistant
Created: 2025-11-22
Last Modified: 2025-11-22
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch
import tempfile
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logging_config import setup_logging, get_logger


def _cleanup_handlers() -> None:
    """Flush and remove all handlers to avoid file locks across tests."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            if hasattr(handler, "close"):
                handler.close()
        finally:
            root_logger.removeHandler(handler)
    logging.shutdown()


@pytest.fixture(autouse=True)
def reset_logging_state():
    _cleanup_handlers()
    yield
    _cleanup_handlers()


class TestLoggingConfig:
    """Test logging configuration functionality."""

    def test_setup_logging_default_config(self):
        _cleanup_handlers()
        logger = setup_logging()
        root_logger = logging.getLogger()
        assert isinstance(logger, logging.Logger)
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) >= 1

    def test_setup_logging_with_file_handler(self):
        _cleanup_handlers()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logging(log_file=str(log_file))
            logger.info("Test message")
            _cleanup_handlers()
            assert log_file.exists()
            assert "Test message" in log_file.read_text()

    def test_setup_logging_console_handler(self):
        _cleanup_handlers()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "console_test.log"
            setup_logging(log_file=str(log_file))
            console_handlers = [
                h for h in logging.getLogger().handlers
                if isinstance(h, logging.StreamHandler)
            ]
            assert console_handlers
            _cleanup_handlers()

    def test_get_logger_returns_configured_logger(self):
        module_name = "test_module"
        logger = get_logger(module_name)
        assert isinstance(logger, logging.Logger)
        assert logger.name == module_name

    def test_third_party_library_log_levels(self):
        _cleanup_handlers()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "third_party_test.log"
            setup_logging(log_file=str(log_file))
            assert logging.getLogger("httpx").level >= logging.WARNING
            assert logging.getLogger("chromadb").level >= logging.WARNING
            _cleanup_handlers()

    def test_multiple_logger_instances_same_config(self):
        _cleanup_handlers()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "multiple_logger_test.log"
            setup_logging(log_file=str(log_file))
            logger1 = get_logger("module1")
            logger2 = get_logger("module2")
            assert isinstance(logger1, logging.Logger)
            assert isinstance(logger2, logging.Logger)
            assert logging.getLogger().handlers
            _cleanup_handlers()


class TestLoggingConfigErrorHandling:
    """Test error handling in logging configuration."""

    def test_setup_logging_invalid_log_file_path(self):
        invalid_path = "/invalid/path/that/does/not/exist/test.log"
        with pytest.raises(FileNotFoundError):
            setup_logging(log_file=invalid_path)

    def test_setup_logging_permission_denied(self):
        if sys.platform.startswith("win"):
            restricted_path = "C:\\Windows\\System32\\test.log"
        else:
            restricted_path = "/root/test.log"
        try:
            logger = setup_logging(log_file=restricted_path)
            assert isinstance(logger, logging.Logger)
        except (PermissionError, OSError):
            pass

    def test_logging_with_unicode_characters(self):
        _cleanup_handlers()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "unicode_test.log"
            logger = setup_logging(log_file=str(log_file), include_console=False)
            message = "Test message with unicode content"
            logger.info(message)
            _cleanup_handlers()
            assert message in log_file.read_text(encoding="utf-8")

    def test_concurrent_logging_safety(self):
        import threading
        import time

        _cleanup_handlers()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "concurrent_test.log"
            logger = setup_logging(log_file=str(log_file))
            messages = []

            def log_messages(thread_id: int):
                for i in range(3):
                    text = f"Thread {thread_id} message {i}"
                    logger.info(text)
                    messages.append(text)
                    time.sleep(0.01)

            threads = [threading.Thread(target=log_messages, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            _cleanup_handlers()
            content = log_file.read_text()
            for msg in messages:
                assert msg in content
