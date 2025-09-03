# Stream Preservation Bug Prevention Tests

## Overview

This document outlines the comprehensive testing strategy designed to prevent the audio and subtitle stream preservation bug that was discovered in the transcoding system. The original bug occurred because `transcode_file_vbr` was using an incomplete `build_vbr_encode_cmd` function that didn't include proper stream mapping flags.

## Root Cause Analysis

**Original Problem:**
- The `transcoding_engine.py` had its own incomplete `build_vbr_encode_cmd` function
- This function was missing critical stream preservation flags: `-map 0`, `-map_metadata 0`, `-map_chapters 0`, `-c:a copy`, `-c:s copy`, etc.
- The comprehensive version in `vbr_optimizer.py` was not being used

**Impact:**
- Audio streams were not copied to output files
- Subtitle streams were lost during transcoding
- Metadata and chapter information was not preserved
- Only video was properly transcoded

## Test Suite Architecture

### 1. Critical Tests (Would Have Caught the Original Bug)

#### `test_command_validation.py`
- **`TestFFmpegCommandValidation.test_command_contains_all_required_stream_preservation_flags`**
  - ✅ **CRITICAL**: Validates ALL required flags are present in generated commands
  - Tests for: `-map 0`, `-map_metadata 0`, `-map_chapters 0`, `-c:a copy`, `-c:s copy`, `-c:d copy`, `-c:t copy`, `-copy_unknown`
  - **Would have caught**: Missing stream preservation flags immediately

#### `test_stream_preservation_regression.py`
- **`TestStreamPreservationRegression.test_vbr_function_uses_comprehensive_encoder`**
  - ✅ **CRITICAL**: Ensures VBR transcoding uses the comprehensive encoder from `vbr_optimizer.py`
  - **Would have caught**: Wrong encoder function being imported/used

- **`TestCommandGenerationIntegration.test_end_to_end_command_generation_includes_streams`**
  - ✅ **CRITICAL**: Tests end-to-end command generation pipeline
  - **Would have caught**: Integration failure between modules

### 2. High Priority Tests (Command Structure & Syntax)

#### Command Structure Validation
- **`test_command_structure_is_valid_ffmpeg_syntax`**: Validates FFmpeg command syntax rules
- **`test_no_conflicting_flags_present`**: Ensures no contradictory flags (e.g., `-an`, `-sn`)
- **`test_vbr_optimizer_generates_comprehensive_commands`**: Direct testing of VBR optimizer

#### Pattern Validation
- **`test_comprehensive_stream_mapping_pattern`**: Tests for complete mapping pattern
- **`test_complete_stream_copying_pattern`**: Validates all stream types are copied
- **`test_no_stream_exclusion_patterns`**: Ensures no exclusion patterns present

#### Module Integration
- **`test_transcoding_engine_imports_correct_vbr_function`**: Validates correct function import
- **`test_no_duplicate_vbr_functions_exist`**: Prevents multiple conflicting implementations

### 3. Hardware Encoder Coverage

Tests ensure ALL encoder types preserve streams:
- **NVIDIA NVENC** (`hevc_nvenc`): Hardware acceleration with stream preservation
- **AMD AMF** (`hevc_amf`): AMD hardware encoding with full streams
- **Intel QuickSync** (`hevc_qsv`): Intel hardware encoding validation

### 4. Consistency & Edge Case Tests

#### Cross-Parameter Consistency
- **Bitrate Independence**: Stream preservation works regardless of bitrate (1000-20000 kbps)
- **Resolution Independence**: Works across 720p, 1080p, 4K resolutions
- **Encoder Independence**: Consistent behavior across all supported encoders

#### Edge Cases
- **HDR Content**: 10-bit HDR processing doesn't interfere with stream preservation
- **Debug Mode**: Additional logging doesn't break stream mapping
- **Complex Media**: Multi-audio, multi-subtitle files handled correctly

## Test Organization by Priority

### Critical (Must Pass - Prevents Original Bug)
```python
critical_tests = [
    'test_command_contains_all_required_stream_preservation_flags',
    'test_vbr_function_uses_comprehensive_encoder', 
    'test_end_to_end_command_generation_includes_streams',
]
```

### High Priority (Command Validation)
```python
high_priority_tests = [
    'test_command_structure_is_valid_ffmpeg_syntax',
    'test_no_conflicting_flags_present',
    'test_comprehensive_stream_mapping_pattern',
    'test_complete_stream_copying_pattern',
    'test_transcoding_engine_imports_correct_vbr_function',
]
```

### Medium Priority (Hardware & Consistency)
```python
medium_priority_tests = [
    'TestHardwareEncoderStreamPreservation',
    'TestCommandGenerationConsistency', 
    'TestRealWorldScenarios',
]
```

## How These Tests Prevent the Original Bug

### 1. Direct Flag Validation
```python
# This test would have IMMEDIATELY caught the missing flags
required_flags = [
    r'-map\s+0\b',                    # ❌ Was missing in original bug
    r'-map_metadata\s+0\b',           # ❌ Was missing in original bug  
    r'-map_chapters\s+0\b',           # ❌ Was missing in original bug
    r'-c:a\s+copy\b',                 # ❌ Was missing in original bug
    r'-c:s\s+copy\b',                 # ❌ Was missing in original bug
]

for pattern in required_flags:
    self.assertRegex(cmd_str, pattern, f"Missing required flag: {pattern}")
```

### 2. Function Usage Validation
```python
# This test prevents using the wrong encoder function
def test_vbr_function_uses_comprehensive_encoder(self):
    # Ensures transcoding_engine uses vbr_optimizer's comprehensive function
    # Not its own incomplete version
```

### 3. Integration Testing
```python
# This catches integration failures between modules
def test_end_to_end_command_generation_includes_streams(self):
    # Tests the complete pipeline from transcode_file_vbr to final command
    # Would catch if wrong function is being called
```

## Test Execution Strategy

### Automated Detection
The test runner prioritizes tests to catch critical issues first:

1. **Critical Tests First**: If these fail, the original bug could still occur
2. **Pattern-Based Categorization**: Automatically categorizes failures by impact
3. **Exit Code Strategy**:
   - `0`: All tests pass - full protection in place
   - `1`: Non-critical failures - original bug protection intact
   - `2`: Critical failures - original bug protection compromised

### Sample Test Run Output
```
🚀 Starting Stream Preservation Test Suite
============================================================

🔥 CRITICAL Tests:
   ✓ TestFFmpegCommandValidation.test_command_contains_all_required_stream_preservation_flags
   ✓ TestStreamPreservationRegression.test_vbr_function_uses_comprehensive_encoder
   ✓ TestCommandGenerationIntegration.test_end_to_end_command_generation_includes_streams
   📊 Added 3 tests

✅ All CRITICAL tests passed (Original bug would be caught!)

🎉 All tests passed! Stream preservation is well protected.
```

## Files in the Test Suite

### Test Files
1. **`test_command_validation.py`** - FFmpeg command validation and structure testing
2. **`test_stream_preservation_regression.py`** - Regression tests targeting the specific bug
3. **`test_enhanced_transcoding.py`** - Enhanced functionality and logging tests
4. **`test_stream_preservation.py`** - General stream preservation testing

### Test Runners
1. **`run_stream_protection_tests.py`** - Priority-based comprehensive test runner
2. **`run_enhanced_tests.py`** - General enhanced functionality test runner

## Key Insights

### What Would Have Prevented the Original Bug

1. **Command Flag Validation**: Testing that generated commands contain required flags
2. **Function Usage Validation**: Ensuring the correct encoder function is used
3. **Integration Testing**: End-to-end pipeline testing catches module interaction issues
4. **Pattern Matching**: Regex-based validation of command structure

### Test Design Principles

1. **Fail Fast**: Critical tests run first to catch major issues immediately
2. **Comprehensive Coverage**: Test all encoder types, resolutions, and edge cases
3. **Realistic Scenarios**: Test with actual anime/movie content patterns
4. **Negative Testing**: Verify exclusion patterns are NOT present

## Future Maintenance

### Adding New Tests
When adding new transcoding features:
1. Add corresponding stream preservation tests
2. Update the critical test patterns if new flags are required
3. Test new encoder types for stream preservation
4. Add edge case tests for new functionality

### Monitoring Strategy
- Run critical tests on every commit affecting transcoding
- Full suite before releases
- Regression tests when stream-related bugs are reported

This comprehensive test suite ensures that the stream preservation bug (missing audio/subtitle streams) can never occur again without being immediately detected during development.
