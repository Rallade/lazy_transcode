# Lazy Transcode

A smart video transcoding utility with VMAF-based quality optimization.

## Features

- **VMAF-based Quality Optimization**: Automatically finds the optimal quality point using VMAF scores
- **Batch Processing**: Process multiple directories with a single command
- **Smart Codec Detection**: Skip files that already use efficient codecs (H.265/HEVC, AV1)
- **Pause/Resume**: Graceful pause and resume functionality with Ctrl+C
- **Progress Tracking**: Real-time progress bars and timing information
- **Non-destructive Mode**: Option to save transcoded files to separate directories

## Installation

Install from PyPI:

```bash
pip install lazy-transcode
```

Or install from source:

```bash
git clone https://github.com/Rallade/lazy_transcode.git
cd lazy-transcode
pip install -e .
```

## Requirements

- Python 3.8+
- FFmpeg with HEVC support
- Optional: `psutil` for CPU monitoring (install with `pip install lazy-transcode[full]`)

## Usage

### Single File/Directory Transcoding

```bash
lazy-transcode /path/to/video/file.mkv
```

### Batch Processing

```bash
lazy-transcode-manager --root /path/to/shows --execute
```

### Common Options

- `--vmaf-target 95.0`: Target VMAF score (default: 95.0)
- `--samples 6`: Number of samples for quality analysis
- `--dry-run`: Analyze only, don't transcode
- `--test-mode`: Fast testing with shorter samples
- `--non-destructive`: Save to 'Transcoded' subdirectory

## Configuration

Create a `.env` file in your project directory:

```env
test_root=M:/Shows
```

## Examples

Analyze all subdirectories under `/media/shows`:
```bash
lazy-transcode-manager --root /media/shows --dry-run
```

Batch transcode with custom quality target:
```bash
lazy-transcode-manager --root /media/shows --vmaf-target 93.0 --execute
```

## License

MIT License - see LICENSE file for details.
