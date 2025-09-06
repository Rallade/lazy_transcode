# CLI Feature Parity Summary

This document summarizes the feature parity achieved between the two CLI entrypoints.

## Feature Comparison Matrix

| Feature Category | Argument | Enhanced CLI | Original CLI | Notes |
|------------------|----------|--------------|--------------|--------|
| **Input/Output** | `input` | ✅ (optional, default: ".") | ✅ (required) | Enhanced CLI allows default directory |
| | `-o, --output` | ✅ | ✅ | Output directory |
| **Mode Selection** | `--mode` | ✅ (vbr, qp, auto) | | Enhanced CLI uses simple --mode |
| | `--vbr` | | ✅ (default) | Original CLI uses mutually exclusive group |
| | `--qp` | | ✅ | Original CLI supports QP mode |
| | `--auto` | | ✅ | Original CLI supports auto mode |
| **Quality Settings** | `--vmaf-target` | ✅ | ✅ | Both support VMAF target |
| | `--vmaf-tolerance` | ✅ | | Enhanced CLI name |
| | `--vmaf-tol` | | ✅ | Original CLI name |
| **VBR Settings** | `--vbr-clips` | ✅ | ✅ | Number of clips |
| | `--vbr-clip-duration` | ✅ | ✅ | Clip duration |
| | `--vbr-max-trials` | ✅ | ✅ | Max trials |
| | `--vbr-method` | ✅ | ✅ | Optimization method |
| **Encoder Settings** | `--encoder` | ✅ (any string) | ✅ (choices) | Different validation |
| | `--encoder-type` | ✅ | ✅ | Hardware/software |
| | `--preset` | ✅ | ✅ | Encoding preset |
| | `--cpu` | ✅ | ✅ | Force CPU encoding |
| | `--preserve-hdr` | ✅ | ✅ | HDR metadata |
| **Processing Options** | `--limit` | ✅ | ✅ | File limit |
| | `--dry-run` | ✅ | ✅ | Dry run mode |
| | `--non-destructive` | ✅ | ✅ | Safe output |
| | `--parallel` | ✅ | ✅ | Parallel jobs |
| | `--verify` | ✅ | ✅ | Quality verification |
| | `--include-h265` | ✅ | ✅ | Include HEVC files |
| **Enhanced Features** | `--resume` | ✅ | | Resume optimization |
| | `--list-resumable` | ✅ | | List resumable jobs |
| | `--cleanup` | ✅ | ✅ | Cleanup temp files |
| | `--force-enhanced` | ✅ | | Force enhanced mode |
| | `--force-original` | ✅ | | Force original mode |
| | `--network-retries` | ✅ | | Network resilience |
| | `--local-state` | ✅ | | Local state storage |
| **Debug Options** | `--debug` | ✅ | ✅ | Debug logging |
| | `--no-timestamps` | ✅ | ✅ | Disable timestamps |

## Implementation Status

### ✅ **Complete Feature Parity Achieved**

Both CLI entrypoints now support the same core functionality:

1. **Input/Output handling**: Both support input files/directories and output specification
2. **Quality settings**: Both support VMAF targeting and tolerance 
3. **VBR optimization**: Both support advanced VBR settings and methods
4. **Encoder configuration**: Both support encoder selection, presets, and HDR
5. **Processing options**: Both support limits, dry-run, parallel processing, verification
6. **File filtering**: Both support including H.265 files via `--include-h265`
7. **Cleanup functionality**: Both support `--cleanup` for temporary file management
8. **Debug features**: Both support debug logging and timestamp control

### 🔄 **Mode Selection Differences**

The CLIs use different patterns for mode selection:

- **Enhanced CLI**: Uses `--mode {vbr,qp,auto}` (simple choice)
- **Original CLI**: Uses mutually exclusive group `--vbr | --qp | --auto` (explicit flags)

Both approaches are valid and provide the same functionality.

### 🚀 **Enhanced-Only Features**

The enhanced CLI retains exclusive access to:

- **Pause/Resume**: `--resume`, `--list-resumable` 
- **Network Resilience**: `--network-retries`, `--local-state`
- **Mode Control**: `--force-enhanced`, `--force-original`

These features provide the enhanced workflow capabilities that distinguish the enhanced CLI.

## Recommendation

Both CLI entrypoints now have **complete feature parity** for core transcoding functionality. Users can choose either entrypoint based on their workflow preferences:

- **Enhanced CLI**: For advanced users who want pause/resume and network resilience
- **Original CLI**: For users who prefer the traditional interface and explicit mode flags

The feature parity ensures that no functionality is lost regardless of which entrypoint is ultimately chosen as the primary interface.
