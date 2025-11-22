"""
Test Runner Script for NL2SQL System
Author: AI Agent
Created: 2025-11-21
Python Version: 3.11

This script provides easy test execution with different configurations.
Run with: python run_tests.py [options]
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_pytest_installation():
    """Check if pytest is installed."""
    try:
        import pytest
        print(f"✅ pytest is installed: {pytest.__version__}")
        return True
    except ImportError:
        print("❌ pytest is not installed")
        print("Install with: pip install pytest")
        return False

def install_test_dependencies():
    """Install test dependencies."""
    print("Installing test dependencies...")
    
    test_deps = [
        "pytest>=7.0.0",
        "pytest-mock>=3.10.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",  # For parallel testing
        "pytest-html>=3.1.0",  # For HTML reports
    ]
    
    for dep in test_deps:
        cmd = [sys.executable, "-m", "pip", "install", dep]
        success = run_command(cmd, f"Installing {dep}")
        if not success:
            print(f"Failed to install {dep}")
            return False
    
    return True

def run_unit_tests():
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_main.py",
        "tests/test_database.py", 
        "tests/test_config.py",
        "tests/test_openai.py",
        "-v",
        "--tb=short"
    ]
    
    return run_command(cmd, "Unit Tests")

def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_integration.py",
        "-v",
        "--tb=short",
        "-m", "integration"
    ]
    
    return run_command(cmd, "Integration Tests")

def run_all_tests():
    """Run all tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "-v",
        "--tb=short"
    ]
    
    return run_command(cmd, "All Tests")

def run_tests_with_coverage():
    """Run tests with coverage report."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "--cov=main",
        "--cov-report=html",
        "--cov-report=term",
        "-v"
    ]
    
    return run_command(cmd, "Tests with Coverage")

def run_tests_parallel():
    """Run tests in parallel."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "-n", "auto",  # Use all available CPUs
        "-v"
    ]
    
    return run_command(cmd, "Parallel Tests")

def run_specific_test(test_name):
    """Run a specific test."""
    cmd = [
        sys.executable, "-m", "pytest", 
        f"tests/{test_name}",
        "-v",
        "--tb=long"
    ]
    
    return run_command(cmd, f"Specific Test: {test_name}")

def generate_html_report():
    """Generate HTML test report."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "--html=reports/test_report.html",
        "--self-contained-html",
        "-v"
    ]
    
    # Create reports directory
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    return run_command(cmd, "HTML Test Report")

def check_test_environment():
    """Check if test environment is properly set up."""
    print("Checking test environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 11):
        print("⚠️  Warning: Python 3.11+ is recommended")
    else:
        print("✅ Python version is compatible")
    
    # Check if tests directory exists
    tests_dir = PROJECT_ROOT / "tests"
    if tests_dir.exists():
        print("✅ Tests directory found")
        
        # List test files
        test_files = list(tests_dir.glob("test_*.py"))
        print(f"📝 Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"   - {test_file.name}")
    else:
        print("❌ Tests directory not found")
        return False
    
    # Check main.py exists
    main_file = PROJECT_ROOT / "main.py"
    if main_file.exists():
        print("✅ main.py found")
    else:
        print("❌ main.py not found")
        return False
    
    return True

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="NL2SQL Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--check-env", action="store_true", help="Check test environment")
    parser.add_argument("--test", type=str, help="Run specific test file (e.g., test_main.py)")
    
    args = parser.parse_args()
    
    print("🧪 NL2SQL Test Runner")
    print("=" * 50)
    
    # Check environment first
    if args.check_env or not any(vars(args).values()):
        if not check_test_environment():
            print("❌ Test environment check failed")
            sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            print("❌ Failed to install test dependencies")
            sys.exit(1)
    
    # Check pytest installation
    if not check_pytest_installation():
        print("Installing pytest...")
        if not install_test_dependencies():
            sys.exit(1)
    
    success = True
    
    # Run specific test
    if args.test:
        success = run_specific_test(args.test)
    
    # Run unit tests
    elif args.unit:
        success = run_unit_tests()
    
    # Run integration tests
    elif args.integration:
        success = run_integration_tests()
    
    # Run tests with coverage
    elif args.coverage:
        success = run_tests_with_coverage()
    
    # Run tests in parallel
    elif args.parallel:
        success = run_tests_parallel()
    
    # Generate HTML report
    elif args.html:
        success = generate_html_report()
    
    # Run all tests by default
    else:
        success = run_all_tests()
    
    # Final status
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("💥 SOME TESTS FAILED!")
    print("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()