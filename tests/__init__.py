"""
Test package for lazy_transcode.

This package contains unit tests for all core modules before refactoring
to consolidate duplicate code patterns.
"""

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # Default timeout for tests
    'temp_cleanup': True,  # Whether to clean up temp files
    'mock_subprocess': True,  # Whether to mock subprocess calls by default
}

# Test data paths (if needed)  
TEST_DATA_DIR = None  # Will be set if test data is needed

__version__ = "1.0.0"
