# Test Suite Documentation

## Overview

This directory contains the comprehensive test suite for the lazy_transcode system. The tests are organized into logical categories for better maintainability and easy execution.

## Directory Structure

```
tests/
├── unit/                    # Individual module unit tests (19 files)
├── regression/              # Regression tests to prevent bug reoccurrence (6 files)  
├── integration/            # End-to-end workflow tests (3 files)
├── utils/                  # Test utilities and validation tools (1 file)
├── __init__.py             # Test package configuration
├── test_overview.py        # Test status summary tool
└── README.md               # This documentation
```

**Total: 29 test files, 69 test classes**

## Test Categories

### 1. Unit Tests (`unit/`)

**Purpose**: Test individual modules and functions in isolation

**Files**:
- `test_system_utils.py` - System utilities (file handling, formatting, temp files)
- `test_media_utils.py` - Media analysis (ffprobe, VMAF, codec detection) 
- `test_file_manager.py` - File discovery and filtering logic
- `test_encoder_config.py` - Hardware encoder configuration
- `test_vbr_optimizer.py` - VBR bitrate optimization algorithms
- `test_transcoding_engine.py` - Core transcoding operations
- `test_vbr.py` - VBR-specific functionality
- `test_config.py` - Configuration management
- `test_core_functionality.py` - Core module patterns

**Usage**:
```bash
# Run all unit tests
python -m unittest discover tests/unit -v

# Run specific module tests
python -m unittest tests.unit.test_system_utils -v
```

### 2. Regression Tests (`regression/`)

**Purpose**: Prevent previously fixed bugs from reoccurring

**Files**:
- `test_stream_preservation_regression.py` - Stream preservation bug protection (✅ Working)
- `test_file_discovery_regression.py` - File filtering and sample detection (❌ Found 11 bugs)
- `test_media_metadata_regression.py` - Metadata extraction and caching (❌ Found 39 bugs)
- `test_vbr_optimization_regression.py` - VBR workflow protection (⚠️ Signature issues)
- `test_temp_file_management_regression.py` - Temp file safety (✅ Working)
- `test_progress_tracking_regression.py` - Progress calculation accuracy (🟡 Mock-based)

**Usage**:
```bash
# Run all regression tests
python -m unittest discover tests/regression -v

# Run specific regression suite
python -m unittest tests.regression.test_stream_preservation_regression -v
```
### 3. Integration Tests (`integration/`)

**Purpose**: Test complete workflows and component interactions

**Files**:
- `test_enhanced_transcoding.py` - End-to-end transcoding workflows
- `test_stream_preservation.py` - Stream preservation integration
- `test_progress_monitoring.py` - Progress tracking across operations

**Usage**:
```bash
# Run all integration tests
python -m unittest discover tests/integration -v

# Run specific integration test
python -m unittest tests.integration.test_enhanced_transcoding -v
```

### 4. Test Utils (`utils/`)

**Purpose**: Test utilities and validation tools

**Files**:
- `test_command_validation.py` - FFmpeg command validation utility

**Usage**:
```bash
# Run utility tests
python -m unittest discover tests/utils -v
```

## How to Run Tests

### Quick Start

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific categories
python -m unittest discover tests/unit -v
python -m unittest discover tests/regression -v
python -m unittest discover tests/integration -v

# Get test overview
python tests/test_overview.py
```

### Focused Testing

```bash
# Test specific functionality
python -m unittest tests.unit.test_system_utils.TestSystemUtils -v

# Test specific bug protection
python -m unittest tests.regression.test_stream_preservation_regression -v

# Test complete workflows
python -m unittest tests.integration.test_enhanced_transcoding -v
```

## Current Test Status

### ✅ Working Test Suites
- **Unit Tests**: 103 tests with some expected failures from code bugs
- **Stream Preservation Regression**: Bug protection working
- **Temp File Management Regression**: Safety tests passing
- **Integration Tests**: Core workflows validated

### ❌ Known Issues
- **File Discovery Regression**: Found 11 bugs in sample detection logic
- **Media Metadata Regression**: Found 39 bugs in cache/codec detection
- **VBR Optimization**: Function signature mismatches need fixing

## Writing New Tests

1. **Unit Tests**: Place in `unit/` directory, test single functions/classes
2. **Regression Tests**: Place in `regression/` directory, test specific bug scenarios  
3. **Integration Tests**: Place in `integration/` directory, test complete workflows
4. **Use descriptive names**: `test_codec_detection_with_valid_h264_file`
5. **Include docstrings**: Explain test purpose clearly

## Contributing

When adding tests:
1. Choose appropriate directory (`unit/`, `regression/`, `integration/`, `utils/`)
2. Follow existing naming conventions  
3. Add comprehensive docstrings
4. Ensure tests are deterministic
5. Update this README if needed

---

*Last updated: September 2025*  
*Total: 29 test files, 69 test classes*  
*Status: Cleaned and organized*
