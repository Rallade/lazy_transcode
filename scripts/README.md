# Utility Scripts

This directory contains utility scripts for testing and optimization analysis.

## Scripts

### `qp_comparison_efficient.py`
**Purpose**: Comprehensive QP optimization technique comparison testing
- Compares different QP optimization approaches across multiple shows
- Efficient pipeline that tests up to 10 random files per show
- Ensures both methods succeed on the same files for fair comparison
- Generates detailed performance comparisons and results tables

**Usage**:
```bash
python scripts/qp_comparison_efficient.py /path/to/shows/root --target-comparisons 20
```

### `vmaf_center_crop.py`
**Purpose**: VMAF center crop optimization implementation
- Uses center cropping technique to accelerate VMAF calculations
- Reduces computational load (4x fewer pixels) while maintaining accuracy
- Achieves 2-4x speed improvements staying within 0.3 VMAF points of full-frame analysis
- Based on Reddit optimization techniques for VMAF acceleration

**Usage**:
```bash
python scripts/vmaf_center_crop.py --input video.mkv --reference ref.mkv
```

## Development Notes

These scripts are development and testing utilities. They are not part of the main package functionality but provide valuable tools for:
- Performance analysis
- Quality optimization research  
- Benchmarking different approaches
- Algorithm validation

The main transcoding functionality is in the `lazy_transcode` package itself.
