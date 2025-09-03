# Test Suite Documentation

This directory contains comprehensive test cases for the lazy_transcode core modules, created before refactoring to consolidate duplicate code patterns.

## Test Files

### Core Module Tests

1. **`test_core_functionality.py`** - Basic functionality tests that work with current imports
   - System utilities (format_size, DEBUG flag, TEMP_FILES)
   - HDR detection functionality
   - Common code patterns that will be refactored

2. **`test_media_utils.py`** - Media utilities functionality
   - FFprobe field extraction
   - Video codec detection  
   - VMAF computation
   - Duration extraction
   - Caching behavior

3. **`test_encoder_config.py`** - Encoder configuration (existing, 223 lines)
   - EncoderConfigBuilder functionality
   - AMF and NVENC command generation
   - Hardware encoder detection

4. **`test_transcoding_engine_corrected.py`** - Transcoding engine functionality
   - HDR content detection
   - Command building for different encoders
   - Edge cases and error handling

5. **`test_vbr_optimizer_corrected.py`** - VBR optimization functionality
   - Bounds calculation algorithms
   - Optimization convergence detection
   - Research-enhanced polynomial models

6. **`test_system_utils.py`** - System utilities
   - File size formatting
   - Directory management
   - Temporary file handling

### Test Infrastructure

- **`run_tests.py`** - Test runner with summary reporting
- **`__init__.py`** - Test package configuration

## Code Duplication Analysis Results

The following duplicate patterns were identified for refactoring:

### 1. FFprobe Operations
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
