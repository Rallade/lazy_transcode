# Critical Areas for Regression Testing Beyond Stream Preservation

Based on the codebase analysis, here are the key areas that would benefit from comprehensive regression tests similar to our successful stream preservation testing approach:

## 1. 🎯 **FILE DISCOVERY & FILTERING PIPELINE** (High Priority)

### Critical Functions in `file_manager.py`:
- `discover_video_files()` - Core file discovery logic
- `check_codec_and_filter()` - Codec filtering that could accidentally exclude valid files
- `_is_sample_or_artifact()` - Sample clip detection (could filter out real files)
- `startup_scavenge()` - Cleanup that could accidentally delete important files

### **Potential Bugs These Tests Would Catch:**
- **File Loss Bug**: Overly aggressive filtering excluding valid video files
- **Hidden File Bug**: Accidentally processing system/hidden files that should be ignored
- **Extension Bug**: Not recognizing valid video file extensions
- **Sample Detection Bug**: Real files being classified as sample clips and ignored

### **Regression Test Examples:**
```python
def test_file_discovery_preserves_all_valid_video_files(self):
    """Ensure all legitimate video files are discovered."""
    
def test_codec_filtering_never_excludes_h264_files(self):
    """H.264 files should always be candidates for transcoding."""
    
def test_sample_detection_never_excludes_real_episodes(self):
    """Episode files should never be classified as samples."""
```

---

## 2. 🔧 **TEMPORARY FILE MANAGEMENT** (High Priority)

### Critical Functions:
- `cleanup_temp_file()` - Could accidentally delete non-temp files
- `startup_scavenge()` - Cleanup patterns might be too broad
- `register_temp_file()` - Tracking failures could lead to orphaned files

### **Potential Bugs:**
- **Data Loss Bug**: Cleanup accidentally deleting original files
- **Orphaned Files Bug**: Temp files not being tracked/cleaned properly
- **Path Confusion Bug**: Temp file paths conflicting with real file paths

### **Regression Test Examples:**
```python
def test_cleanup_never_touches_original_files(self):
    """Cleanup should only affect files in temp directories."""
    
def test_temp_file_patterns_are_specific_enough(self):
    """Cleanup patterns shouldn't match legitimate files."""
```

---

## 3. 🎬 **MEDIA METADATA EXTRACTION** (High Priority)

### Critical Functions in `media_utils.py`:
- `ffprobe_field()` - Core metadata extraction (cached with LRU)
- `get_video_dimensions()` - Resolution detection for encoding
- `get_duration_sec()` - Duration calculation for progress tracking
- `get_video_codec()` - Codec detection for filtering decisions

### **Potential Bugs:**
- **Metadata Cache Bug**: Wrong cached values being returned for different files
- **Resolution Detection Bug**: Wrong dimensions leading to encoding failures
- **Duration Bug**: Incorrect duration affecting progress and clip extraction
- **Codec Detection Bug**: Wrong codec info leading to incorrect processing decisions

### **Regression Test Examples:**
```python
def test_ffprobe_cache_never_returns_wrong_file_metadata(self):
    """LRU cache should never mix up metadata between files."""
    
def test_video_dimensions_detection_handles_all_formats(self):
    """Resolution detection should work for all supported formats."""
    
def test_codec_detection_identifies_all_major_codecs(self):
    """Should correctly identify h264, h265, hevc, av1, etc."""
```

---

## 4. ⚙️ **VBR OPTIMIZATION BOUNDS CALCULATION** (Medium-High Priority)

### Critical Functions in `vbr_optimizer.py`:
- `get_intelligent_bounds()` - Initial bitrate range calculation
- `get_adaptive_lower_bounds()` - Dynamic bounds adjustment
- `should_terminate_early()` - Optimization convergence detection
- `calculate_intelligent_vbr_bounds()` - Master bounds calculation

### **Potential Bugs:**
- **Infinite Loop Bug**: Optimization never terminating due to bad bounds
- **Quality Target Bug**: Never reaching target VMAF due to incorrect bounds
- **Performance Bug**: Bounds too wide causing excessive iterations
- **Precision Bug**: Bounds too narrow missing optimal bitrate

### **Regression Test Examples:**
```python
def test_vbr_bounds_always_converge_within_reasonable_iterations(self):
    """Optimization should converge in <10 iterations for typical content."""
    
def test_intelligent_bounds_never_exclude_optimal_bitrate(self):
    """Initial bounds should always include the optimal bitrate."""
    
def test_early_termination_never_stops_before_reaching_target(self):
    """Should not terminate if target VMAF hasn't been reached."""
```

---

## 5. 📊 **VMAF COMPUTATION PIPELINE** (Medium Priority)

### Critical Functions in `vmaf_evaluator.py`:
- `compute_vmaf_score()` - Core VMAF calculation
- `create_sample_clip()` - Sample extraction for testing
- `test_qp_on_sample()` - QP testing workflow
- `_compute_vmaf_with_retry()` - Retry logic for failed computations

### **Potential Bugs:**
- **VMAF Accuracy Bug**: Incorrect scores due to scaling/format issues
- **Sample Representativeness Bug**: Non-representative samples leading to wrong decisions
- **Retry Logic Bug**: Infinite retries or giving up too early
- **Memory Leak Bug**: Large temp files not being cleaned up

---

## 6. 🔄 **COMMAND BUILDING INTEGRATION** (Medium Priority)

### Beyond stream preservation, test integration between:
- `encoder_config.py` - Command building
- `transcoding_engine.py` - Workflow orchestration  
- `vbr_optimizer.py` - Optimization logic

### **Potential Integration Bugs:**
- **Parameter Mismatch Bug**: Encoder config not matching optimizer expectations
- **HDR Pipeline Bug**: HDR settings being lost in command building
- **Hardware Encoder Bug**: Hardware-specific flags being applied incorrectly

---

## 7. 🎮 **ARGUMENT PARSING & VALIDATION** (Medium Priority)

### Critical Functions in main entry points:
- Argument validation in `main.py`, `manager.py`, `transcode.py`
- Mode selection logic (VBR vs QP vs Auto)
- Path validation and normalization

### **Potential Bugs:**
- **Path Traversal Bug**: Invalid paths causing crashes or security issues
- **Mode Conflict Bug**: Conflicting arguments not being caught
- **Default Value Bug**: Wrong defaults being applied

---

## 🏗️ **Recommended Test Implementation Strategy**

### Phase 1: Critical Path Protection (Week 1)
1. **File Discovery Tests** - Protect against file loss
2. **Temp File Safety Tests** - Prevent data deletion  
3. **Metadata Extraction Tests** - Ensure correct media info

### Phase 2: Quality & Performance (Week 2)
1. **VBR Bounds Testing** - Prevent optimization failures
2. **VMAF Accuracy Tests** - Ensure quality metrics are correct
3. **Integration Tests** - Cross-module compatibility

### Phase 3: Edge Cases & Robustness (Week 3)
1. **Error Handling Tests** - Graceful failure modes
2. **Edge Case Coverage** - Unusual file formats, corrupted media
3. **Performance Regression Tests** - Prevent performance degradation

---

## 🔍 **Testing Approach Similar to Stream Preservation**

### Pattern to Follow:
```python
class TestFileDiscoveryRegression(unittest.TestCase):
    """Prevent file discovery regressions that could cause file loss."""
    
    def test_discover_video_files_finds_all_valid_extensions(self):
        """CRITICAL: Ensure all supported extensions are discovered."""
        
    def test_codec_filtering_never_excludes_h264_content(self):
        """REGRESSION: Ensure H.264 files are always considered for transcoding."""
        
    def test_sample_detection_uses_conservative_patterns(self):
        """SAFETY: Sample detection should err on the side of inclusion."""
```

### Key Principles:
1. **Focus on Critical Path** - Test functions that could cause data loss
2. **Test Integration Points** - Where modules interact
3. **Validate Assumptions** - Test the logic behind decisions
4. **Prevent Common Mistakes** - Test patterns that could be accidentally broken

---

## 💡 **Why These Areas Need Regression Tests**

1. **File Manager**: File loss is catastrophic - much worse than encoding issues
2. **Media Utils**: Wrong metadata leads to wrong encoding decisions
3. **VBR Optimizer**: Complex optimization logic prone to edge case failures
4. **Integration**: Multiple modules working together increases failure surface area

These areas have **high complexity** and **high impact**, making them prime candidates for comprehensive regression testing similar to what we built for stream preservation.

Would you like me to implement regression tests for any of these specific areas?
