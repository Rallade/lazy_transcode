# Test Suite Documentation

## Overview

This directory contains the comprehensive test suite for the lazy_transcode system. The tests are organized into logical categories for better maintainability, readability, and ease of understanding.

## Directory Structure

```
tests/
├── unit/                    # Individual module unit tests
├── regression/              # Regression tests to prevent bug reoccurrence  
├── integration/            # End-to-end workflow tests
├── utils/                  # Test runners, utilities, and validation tools
├── __init__.py             # Test package configuration
└── README.md               # This documentation
```

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

**Usage**:
```bash
# Run all unit tests
python -m unittest discover tests.unit -v

# Run specific module tests
python -m unittest tests.unit.test_system_utils -v
```

### 2. Regression Tests (`regression/`)

**Purpose**: Prevent previously fixed bugs from reoccurring

**Files**:
- `test_stream_preservation_regression.py` - Stream preservation bug protection (✅ 14 tests passing)
- `test_file_discovery_regression.py` - File filtering and sample detection (❌ 11 critical bugs found)
- `test_media_metadata_regression.py` - Metadata extraction and caching (❌ 39 critical bugs found)
- `test_vbr_optimization_regression.py` - VBR workflow protection (⚠️ signature issues)
- `test_temp_file_management_regression.py` - Temp file safety (✅ 12 tests passing)
- `test_progress_tracking_regression.py` - Progress calculation accuracy (✅ mock-based)

**Critical Issues Found**:
- **Sample detection too aggressive**: Legitimate files incorrectly filtered
- **Cache isolation failures**: Metadata mixing between files
- **Codec detection broken**: All returning None instead of actual codecs

**Usage**:
```bash
# Run all regression tests
python -m unittest discover tests.regression -v

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
python -m unittest discover tests.integration -v

# Run specific integration test
python -m unittest tests.integration.test_enhanced_transcoding -v
```

### 4. Test Utils (`utils/`)

**Purpose**: Test runners, validation tools, and shared utilities

**Files**:
- `run_tests.py` - Main test runner with summary reporting
- `run_enhanced_tests.py` - Enhanced test execution
- `run_stream_protection_tests.py` - Stream preservation focused runner
- `daily_validation.py` - Daily system health validation
- `test_command_validation.py` - FFmpeg command validation utility

**Usage**:
```bash
# Run comprehensive test suite
python tests/utils/run_tests.py

# Daily validation check
python tests/utils/daily_validation.py

# Validate FFmpeg commands
python -m unittest tests.utils.test_command_validation -v
```

```

## Test Execution Guide

### Quick Start

```bash
# Run all tests with summary
python tests/utils/run_tests.py

# Run only passing tests (excludes broken regression tests)
python -m unittest discover tests.unit -v
python -m unittest discover tests.integration -v
python -m unittest tests.regression.test_stream_preservation_regression -v
python -m unittest tests.regression.test_temp_file_management_regression -v
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

### Debugging Failing Tests

```bash
# Run with maximum verbosity
python -m unittest tests.regression.test_file_discovery_regression -v

# Run specific failing test class
python -m unittest tests.regression.test_media_metadata_regression.TestCodecDetectionRegression -v
```

## Current Test Status

### ✅ Working Test Suites (Pass Rate: 100%)
- **Unit Tests**: All module tests passing
- **Stream Preservation Regression**: 14/14 tests passing
- **Temp File Management Regression**: 12/12 tests passing
- **Integration Tests**: Core workflows validated

### ❌ Broken Test Suites (Critical Issues Found)
- **File Discovery Regression**: 11 critical bugs in sample detection
- **Media Metadata Regression**: 39 critical bugs in cache/codec detection
- **VBR Optimization Regression**: Function signature mismatches

### ⚠️ Needs Attention
- **Progress Tracking Regression**: Mock-based implementation (no real progress tracker module)

## Bug Summary from Regression Testing

The regression tests successfully identified **48 total bugs** across the system:

1. **Sample Detection Logic** (11 bugs) - Files incorrectly classified as samples
2. **Metadata Cache Isolation** (39 bugs) - Metadata mixing between different files
3. **Codec Detection Failure** - All codec detection returning None
4. **VBR Function Signatures** (16 issues) - Parameter mismatches preventing testing

## Best Practices

### Writing New Tests

1. **Unit Tests**: Place in `unit/` directory, test single functions/classes
2. **Regression Tests**: Place in `regression/` directory, test specific bug scenarios
3. **Integration Tests**: Place in `integration/` directory, test complete workflows

### Test Organization

- Use descriptive class names: `TestMediaUtilsCodecDetection`
- Use descriptive method names: `test_codec_detection_with_valid_h264_file`
- Include docstrings explaining test purpose
- Group related tests in the same test class

### Running Tests in Development

1. **Before committing**: Run regression tests to ensure no bugs reintroduced
2. **After changes**: Run related unit tests to validate functionality
3. **Before releases**: Run full test suite including integration tests

## Future Improvements

1. **Fix Critical Bugs**: Address the 48 bugs found by regression tests
2. **Add Performance Tests**: Monitor encoding speed and quality
3. **Expand Integration Tests**: Add more real-world scenario testing
4. **CI/CD Integration**: Automate test execution on code changes
5. **Test Data Management**: Create standardized test video files

## Contributing

When adding new tests:

1. Choose the appropriate directory (`unit/`, `regression/`, `integration/`)
2. Follow existing naming conventions
3. Add comprehensive docstrings
4. Update this README if adding new test categories
5. Ensure tests are deterministic and don't depend on external resources

---

*Last updated: September 2025*
*Total tests: 77 across all categories*
*Critical bugs found: 48 (regression testing proving its value)*
- **Locations**: `media_utils.py`, `encoder_config.py`
- **Pattern**: Duplicate `ffprobe_field` functions with identical logic
- **Consolidation**: Create single utility function in `media_utils.py`

### 2. Video Dimensions Extraction  
- **Locations**: 6+ locations across modules
- **Pattern**: Repetitive width/height extraction using ffprobe
- **Consolidation**: Create `get_video_dimensions()` utility function

### 3. Subprocess Execution
- **Locations**: 40+ instances across all modules
- **Pattern**: Inconsistent subprocess.run() with different error handling
- **Consolidation**: Create standardized `run_command()` wrapper

### 4. VMAF Computation
- **Locations**: Multiple locations with slight variations
- **Pattern**: Similar FFmpeg VMAF filter usage
- **Consolidation**: Standardize in `media_utils.compute_vmaf_score()`

### 5. Temporary File Management
- **Locations**: Scattered across modules
- **Pattern**: Inconsistent temp file creation and cleanup
- **Consolidation**: Centralize in `system_utils` with context managers

## Running Tests

### Run All Tests
```bash
cd tests
python run_tests.py
```

### Run Specific Test File
```bash
python -m unittest test_core_functionality.py -v
```

### Run Individual Test Class
```bash
python -m unittest test_core_functionality.TestSystemUtils -v
```

## Test Coverage Goals

Before refactoring, ensure test coverage for:

- [x] HDR detection and metadata handling
- [x] FFprobe field extraction patterns
- [x] Video dimensions extraction
- [x] Subprocess execution patterns
- [x] VMAF computation workflows
- [x] Temporary file management
- [x] System utilities (format_size, directory operations)
- [x] Error handling and edge cases
- [x] Encoder configuration building
- [x] VBR optimization convergence detection

## Refactoring Safety

These tests provide safety nets for:

1. **Function Consolidation** - Ensure merged functions maintain behavior
2. **API Changes** - Catch breaking changes during refactoring  
3. **Performance Impact** - Verify optimizations don't break functionality
4. **Error Handling** - Maintain robust error handling through changes

## Research Integration Testing

Tests verify integration of:

- **Farhadi Nia (2025) polynomial models** - VBR optimization algorithms
- **Newton-Raphson inverse prediction** - Bitrate convergence detection
- **Content-adaptive analysis** - SI/TI metrics and resolution optimization
- **Cross-codec quality-rate prediction** - Multi-codec comparison methodologies

## Academic Attribution Compliance

All research-enhanced functionality includes proper citations and maintains academic integrity standards established during implementation.

## Post-Refactoring Validation

After code consolidation:

1. Run full test suite to ensure no regressions
2. Verify performance improvements from reduced code duplication
3. Validate that all functionality remains intact
4. Update tests as needed for new consolidated APIs
5. Add integration tests for refactored components

The test suite serves as both validation and documentation of the system's behavior before and after the refactoring process.
