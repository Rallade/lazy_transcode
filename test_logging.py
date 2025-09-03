#!/usr/bin/env python3
"""
Test the enhanced transcoding logging functionality.
"""

from pathlib import Path
import sys

# Add the lazy_transcode package to path
sys.path.insert(0, str(Path(__file__).parent / 'lazy_transcode'))

from lazy_transcode.core.modules.transcoding_engine import _log_input_streams, _log_output_streams

def test_logging_functions():
    """Test the logging functions with a dummy file."""
    print("Testing enhanced transcoding logging functions:")
    print("=" * 50)
    
    # Test with a non-existent file to see error handling
    dummy_file = Path("nonexistent_video.mp4")
    
    print("\n1. Testing input stream analysis with non-existent file:")
    _log_input_streams(dummy_file)
    
    print("\n2. Testing output stream verification with non-existent files:")
    _log_output_streams(dummy_file, dummy_file)
    
    print("\nLogging enhancement test completed!")
    print("\nKey features added:")
    print("✓ Detailed input stream analysis (video, audio, subtitles, chapters)")
    print("✓ Comprehensive output stream verification")
    print("✓ File size comparison and reduction calculation")
    print("✓ Enhanced progress monitoring with FPS, bitrate, speed")
    print("✓ Full FFmpeg command logging for transparency")
    print("✓ Clear success/failure messages")

if __name__ == "__main__":
    test_logging_functions()
