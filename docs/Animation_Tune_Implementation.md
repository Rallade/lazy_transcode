# Animation Tune Support Implementation

## Overview

Added comprehensive animation tune support to the lazy_transcode encoder configuration system, enabling automatic detection and optimization for animated content.

## Features Implemented

### 1. Core Encoder Configuration Support

**File**: `lazy_transcode/core/modules/config/encoder_config.py`

- Added `tune` parameter to `EncoderConfigBuilder.set_encoder()` method
- Added tune support to both `build_standard_encode_cmd()` and `build_vbr_encode_cmd()` methods
- Automatic filtering - tune is only applied to software encoders (libx264, libx265)
- Hardware encoders (nvenc, qsv, amf) correctly skip tune parameters

**Available Tune Options**:
- **x264**: `film`, `animation`, `grain`, `stillimage`, `psnr`, `ssim`, `fastdecode`, `zerolatency`
- **x265**: `psnr`, `ssim`, `grain`, `zerolatency`, `fastdecode`, `animation`

### 2. Animation Content Detection

**File**: `lazy_transcode/core/modules/analysis/animation_detector.py`

**Detection Methods**:
1. **Content Analysis**: Uses existing content analyzer to detect low spatial info + high temporal info patterns typical of animation
2. **Filename Keywords**: Detects anime/animation keywords and series indicators
3. **Hybrid Approach**: Combines content metrics with filename analysis for robust detection

**Detection Criteria**:
- Spatial Info < 20 AND Temporal Info > 40 (content-based)
- Very low spatial info < 15 (flat animated content)
- Animation keywords: `anime`, `animation`, `cartoon`, `animated`
- Series indicators: `ep`, `episode`, `s01`, `s02`, etc. + moderate spatial info

### 3. Automatic VBR Integration

**File**: `lazy_transcode/core/modules/optimization/vbr_optimizer.py`

- Modified `build_vbr_encode_cmd()` to include `auto_tune` parameter (default: True)
- Automatic animation detection and tune application for software encoders
- Logging integration to show when auto-tune is applied
- Backward compatible - existing code continues to work unchanged

### 4. Helper Functions

**Animation-Optimized Settings**:
```python
get_animation_optimized_settings(input_file, encoder)
```
Returns recommended settings for animation content:
- `tune`: 'animation'
- `suggested_preset`: 'fast' (animation compresses well)
- `refs`: 2 (reduced reference frames)
- `bf`: 2 (optimized B-frames)

**Content-Based Tune Detection**:
```python
get_optimal_tune_for_content(input_file, encoder, sample_duration=30)
```
Automatically determines optimal tune setting based on content analysis.

## Usage Examples

### Manual Tune Application
```python
from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder

builder = EncoderConfigBuilder()
builder.set_base_config("input.mkv", "output.mkv")
builder.set_encoder("libx265", preset="medium", crf=23, tune="animation")
cmd = builder.build_command()
```

### Automatic Detection in VBR Mode
```python
from lazy_transcode.core.modules.optimization.vbr_optimizer import build_vbr_encode_cmd

# Animation tune automatically detected and applied for anime content
cmd = build_vbr_encode_cmd(
    anime_file, output_file, 
    encoder='libx265', encoder_type='software',
    max_bitrate=5000, avg_bitrate=4000
)
```

### Content Analysis
```python
from lazy_transcode.core.modules.analysis.animation_detector import detect_animation_content

is_anime = detect_animation_content(Path("Your.Name.2016.mkv"))
print(f"Animation detected: {is_anime}")
```

## Benefits for Anime Content

The animation tune provides several optimizations specifically beneficial for animated content:

1. **Better Compression Efficiency**: Optimized for flat colors and gradients common in animation
2. **Quality Preservation**: Better handling of sharp edges and text overlays in anime
3. **Psychovisual Tuning**: Adjusted for cartoon-like content perception
4. **Bitrate Efficiency**: Often achieves same quality at lower bitrates

## Testing

### Unit Tests
**File**: `tests/unit/test_animation_tune.py`

Comprehensive test suite covering:
- Basic tune functionality for all encoders
- Hardware encoder filtering (tune skipped)
- VBR and standard command integration
- Animation detection logic
- Content-based tune recommendations

All 11 tests pass, ensuring robust functionality.

### Test Coverage
- ✅ Tune parameter handling in encoder config
- ✅ Software vs hardware encoder filtering
- ✅ Animation content detection (metrics + filename)
- ✅ VBR integration with auto-tune
- ✅ Standard encoding command tune support
- ✅ Optimal tune recommendations

## Integration Notes

### Backward Compatibility
- All existing code continues to work unchanged
- Tune parameter is optional in all functions
- Auto-tune can be disabled by setting `auto_tune=False`

### Performance Impact
- Minimal - content analysis only runs when auto-tune is enabled
- Uses existing content analyzer infrastructure
- Graceful fallback if analysis fails

### Anime-Optimized Workflow
Perfect integration with existing anime-focused features:
- Coverage-based VBR clip selection for anime
- Opening theme detection
- Episode ordering
- Content-aware optimization

## Future Enhancements

Potential areas for expansion:
1. **Additional Tune Detection**: Film grain, still image content
2. **Encoder-Specific Optimizations**: Tune-aware preset selection  
3. **CLI Integration**: Direct tune parameter exposure
4. **Quality Validation**: VMAF testing with different tune settings

## Configuration

The animation tune functionality works out-of-the-box with no configuration required. For advanced users:

- Content analysis duration can be adjusted in `detect_animation_content(sample_duration=30)`
- Detection thresholds can be modified in the animation detector module
- Auto-tune can be disabled per-operation if needed

This implementation provides a solid foundation for anime-optimized encoding while maintaining full compatibility with existing workflows.
