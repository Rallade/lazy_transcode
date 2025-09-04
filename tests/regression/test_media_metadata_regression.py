"""
Media Metadata Extraction Regression Tests

These tests prevent critical bugs in media metadata extraction that could cause:
- Wrong codec detection leading to incorrect processing decisions
- Incorrect duration affecting progress tracking and clip extraction
- Wrong resolution leading to encoding failures
- Cache corruption causing metadata mix-ups between files

The media utilities are foundational to all transcoding operations.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from functools import lru_cache

from lazy_transcode.core.modules.analysis.media_utils import (
    ffprobe_field, get_video_dimensions, get_duration_sec, get_video_codec
)


class TestFFprobeFieldRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent ffprobe_field regressions that could cause metadata corruption.
    
    ffprobe_field is used throughout the system and has LRU caching that could
    mix up metadata between different files.
    """
    
    def setUp(self):
        """Clear the LRU cache before each test."""
        ffprobe_field.cache_clear()
    
    def test_ffprobe_cache_never_mixes_up_different_files(self):
        """
        CRITICAL TEST: LRU cache should never return wrong file's metadata.
        
        This would catch cache key collisions that could cause file A's metadata
        to be returned when requesting file B's metadata.
        """
        file1 = Path("/fake/file1.mkv")
        file2 = Path("/fake/file2.mkv") 
        file3 = Path("/fake/file3.mp4")
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.run') as mock_run:
            # Set up different responses for each file
            def mock_ffprobe_responses(cmd, **kwargs):
                file_path = cmd[cmd.index('-i') + 1]
                mock_result = MagicMock()
                mock_result.returncode = 0
                
                if 'file1.mkv' in file_path:
                    mock_result.stdout = "1920x1080"
                elif 'file2.mkv' in file_path:
                    mock_result.stdout = "1280x720"
                elif 'file3.mp4' in file_path:
                    mock_result.stdout = "3840x2160"
                else:
                    mock_result.stdout = "unknown"
                
                return mock_result
            
            mock_run.side_effect = mock_ffprobe_responses
            
            # Request metadata multiple times in different orders
            for _ in range(3):  # Multiple iterations to test cache behavior
                result1 = ffprobe_field(file1, "stream=width,height")
                result2 = ffprobe_field(file2, "stream=width,height") 
                result3 = ffprobe_field(file3, "stream=width,height")
                
                # Each file should always return its own metadata
                with self.subTest(iteration=_, file="file1"):
                    self.assertEqual(result1, "1920x1080", "File1 should always return 1920x1080")
                
                with self.subTest(iteration=_, file="file2"):
                    self.assertEqual(result2, "1280x720", "File2 should always return 1280x720")
                
                with self.subTest(iteration=_, file="file3"):
                    self.assertEqual(result3, "3840x2160", "File3 should always return 3840x2160")
    
    def test_ffprobe_handles_identical_filenames_in_different_paths(self):
        """
        CACHE COLLISION TEST: Files with same name in different paths should not collide.
        
        This tests that the cache uses the full path, not just the filename.
        """
        file1 = Path("/path1/episode01.mkv")
        file2 = Path("/path2/episode01.mkv")  # Same filename, different path
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.run') as mock_run:
            def mock_different_paths(cmd, **kwargs):
                file_path = cmd[cmd.index('-i') + 1]
                mock_result = MagicMock()
                mock_result.returncode = 0
                
                if '/path1/' in file_path:
                    mock_result.stdout = "h264"
                elif '/path2/' in file_path:
                    mock_result.stdout = "hevc"
                else:
                    mock_result.stdout = "unknown"
                
                return mock_result
            
            mock_run.side_effect = mock_different_paths
            
            # Get codec for both files
            codec1 = ffprobe_field(file1, "stream=codec_name")
            codec2 = ffprobe_field(file2, "stream=codec_name")
            
            self.assertEqual(codec1, "h264", "Path1 file should return h264")
            self.assertEqual(codec2, "hevc", "Path2 file should return hevc")
            
            # Test cache consistency with multiple calls
            codec1_cached = ffprobe_field(file1, "stream=codec_name")
            codec2_cached = ffprobe_field(file2, "stream=codec_name")
            
            self.assertEqual(codec1_cached, "h264", "Cached path1 should still return h264")
            self.assertEqual(codec2_cached, "hevc", "Cached path2 should still return hevc")
    
    def test_ffprobe_error_handling_is_consistent(self):
        """
        ERROR HANDLING TEST: ffprobe failures should be handled consistently.
        
        This ensures error conditions don't corrupt the cache or cause inconsistent behavior.
        """
        test_file = Path("/fake/corrupted.mkv")
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.run') as mock_run:
            # Mock ffprobe failure
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Input/output error"
            mock_run.return_value = mock_result
            
            # Multiple calls should all return None consistently
            for i in range(3):
                with self.subTest(call=i):
                    result = ffprobe_field(test_file, "stream=codec_name")
                    self.assertIsNone(result, f"Error case should always return None (call {i})")
    
    def test_ffprobe_different_fields_are_cached_separately(self):
        """
        CACHE GRANULARITY TEST: Different fields for same file should be cached separately.
        
        This ensures requesting different metadata fields doesn't interfere with each other.
        """
        test_file = Path("/fake/test.mkv")
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.run') as mock_run:
            def mock_field_specific_response(cmd, **kwargs):
                mock_result = MagicMock()
                mock_result.returncode = 0
                
                # Check which field is being requested
                cmd_str = ' '.join(cmd)
                if 'codec_name' in cmd_str:
                    mock_result.stdout = "h264"
                elif 'width,height' in cmd_str:
                    mock_result.stdout = "1920x1080"
                elif 'duration' in cmd_str:
                    mock_result.stdout = "7200.0"
                else:
                    mock_result.stdout = "unknown"
                
                return mock_result
            
            mock_run.side_effect = mock_field_specific_response
            
            # Request different fields
            codec = ffprobe_field(test_file, "stream=codec_name")
            dimensions = ffprobe_field(test_file, "stream=width,height") 
            duration = ffprobe_field(test_file, "format=duration")
            
            self.assertEqual(codec, "h264", "Codec field should be correct")
            self.assertEqual(dimensions, "1920x1080", "Dimensions field should be correct")
            self.assertEqual(duration, "7200.0", "Duration field should be correct")
            
            # Verify each field is cached independently
            codec_cached = ffprobe_field(test_file, "stream=codec_name")
            dimensions_cached = ffprobe_field(test_file, "stream=width,height")
            duration_cached = ffprobe_field(test_file, "format=duration")
            
            self.assertEqual(codec_cached, "h264", "Cached codec should be consistent")
            self.assertEqual(dimensions_cached, "1920x1080", "Cached dimensions should be consistent")
            self.assertEqual(duration_cached, "7200.0", "Cached duration should be consistent")


class TestVideoDimensionsRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent video dimensions detection regressions.
    
    Wrong dimensions lead to encoding failures and incorrect optimization decisions.
    """
    
    def setUp(self):
        """Clear caches before each test."""
        get_video_dimensions.cache_clear()
    
    def test_video_dimensions_handles_all_common_resolutions(self):
        """
        COMPATIBILITY TEST: Should correctly parse all common video resolutions.
        
        This prevents bugs where certain resolutions are misinterpreted.
        """
        test_cases = [
            ("1920x1080", (1920, 1080)),  # 1080p
            ("1280x720", (1280, 720)),    # 720p  
            ("3840x2160", (3840, 2160)),  # 4K
            ("1440x1080", (1440, 1080)),  # 4:3 1080p
            ("2560x1440", (2560, 1440)),  # 1440p
            ("7680x4320", (7680, 4320)),  # 8K
        ]
        
        for resolution_str, expected_tuple in test_cases:
            with self.subTest(resolution=resolution_str):
                test_file = Path(f"/fake/{resolution_str.replace('x', '_')}.mkv")
                
                with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
                    mock_ffprobe.return_value = resolution_str
                    
                    result = get_video_dimensions(test_file)
                    self.assertEqual(result, expected_tuple,
                                   f"Resolution {resolution_str} should parse to {expected_tuple}")
    
    def test_video_dimensions_handles_malformed_input_gracefully(self):
        """
        ROBUSTNESS TEST: Malformed resolution strings should not crash the system.
        
        This prevents crashes when ffprobe returns unexpected formats.
        """
        malformed_cases = [
            "",           # Empty string
            "1920",       # Missing height
            "x1080",      # Missing width  
            "1920x",      # Missing height with x
            "widthxheight", # Non-numeric
            "1920×1080",  # Unicode multiplication symbol
            "1920 x 1080", # Spaces
            None,         # ffprobe returned None
        ]
        
        for malformed_input in malformed_cases:
            with self.subTest(input=repr(malformed_input)):
                test_file = Path(f"/fake/malformed_{hash(str(malformed_input))}.mkv")
                
                with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
                    mock_ffprobe.return_value = malformed_input
                    
                    # Should not crash - should return some sensible default or raise specific exception
                    try:
                        result = get_video_dimensions(test_file)
                        # If it returns something, it should be a tuple of two integers
                        self.assertIsInstance(result, tuple, "Should return tuple if successful")
                        self.assertEqual(len(result), 2, "Tuple should have exactly 2 elements")
                        self.assertIsInstance(result[0], int, "Width should be integer")
                        self.assertIsInstance(result[1], int, "Height should be integer")
                    except (ValueError, TypeError) as e:
                        # Acceptable to raise specific exceptions for malformed input
                        pass
    
    def test_video_dimensions_cache_works_correctly(self):
        """
        CACHE TEST: Dimensions caching should not mix up different files.
        
        Similar to ffprobe_field, this tests cache isolation between files.
        """
        file1 = Path("/fake/hd.mkv")
        file2 = Path("/fake/uhd.mkv")
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
            def mock_resolution_responses(file_path, field):
                if 'hd.mkv' in str(file_path):
                    return "1920x1080"
                elif 'uhd.mkv' in str(file_path):
                    return "3840x2160"
                else:
                    return "unknown"
            
            mock_ffprobe.side_effect = mock_resolution_responses
            
            # Get dimensions multiple times
            for i in range(3):
                dims1 = get_video_dimensions(file1)
                dims2 = get_video_dimensions(file2)
                
                with self.subTest(iteration=i):
                    self.assertEqual(dims1, (1920, 1080), f"HD file should be 1920x1080 (iteration {i})")
                    self.assertEqual(dims2, (3840, 2160), f"UHD file should be 3840x2160 (iteration {i})")


class TestDurationExtractionRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent duration extraction regressions.
    
    Wrong duration affects progress tracking, clip extraction, and optimization decisions.
    """
    
    def setUp(self):
        """Clear caches before each test."""
        get_duration_sec.cache_clear()
    
    def test_duration_parsing_handles_various_formats(self):
        """
        FORMAT TEST: Should handle all common duration formats from ffprobe.
        
        ffprobe can return duration in different formats depending on the container.
        """
        duration_test_cases = [
            ("7200.000000", 7200.0),    # Standard decimal format
            ("7200", 7200.0),           # Integer format
            ("7200.5", 7200.5),         # Half second
            ("0.000000", 0.0),          # Zero duration
            ("3661.234567", 3661.234567), # Precise decimal
        ]
        
        for duration_str, expected_seconds in duration_test_cases:
            with self.subTest(duration=duration_str):
                test_file = Path(f"/fake/duration_{duration_str.replace('.', '_')}.mkv")
                
                with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
                    mock_ffprobe.return_value = duration_str
                    
                    result = get_duration_sec(test_file)
                    self.assertAlmostEqual(result, expected_seconds, places=5,
                                         msg=f"Duration {duration_str} should parse to {expected_seconds}")
    
    def test_duration_handles_malformed_input_gracefully(self):
        """
        ROBUSTNESS TEST: Malformed duration should not crash the system.
        
        This prevents crashes when ffprobe returns unexpected duration formats.
        """
        malformed_durations = [
            "",              # Empty
            "N/A",          # ffprobe N/A response
            "unknown",      # Text response
            "1:30:00",      # Time format (should be converted)
            None,           # ffprobe failure
            "inf",          # Infinity
            "-100",         # Negative duration
        ]
        
        for malformed_duration in malformed_durations:
            with self.subTest(duration=repr(malformed_duration)):
                test_file = Path(f"/fake/bad_duration_{hash(str(malformed_duration))}.mkv")
                
                with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
                    mock_ffprobe.return_value = malformed_duration
                    
                    try:
                        result = get_duration_sec(test_file)
                        # If it succeeds, result should be a valid float
                        self.assertIsInstance(result, (int, float), "Duration should be numeric")
                        self.assertGreaterEqual(result, 0, "Duration should be non-negative")
                    except (ValueError, TypeError):
                        # Acceptable to raise specific exceptions for malformed input
                        pass
    
    def test_duration_cache_isolation_between_files(self):
        """
        CACHE ISOLATION TEST: Duration cache should not mix up different files.
        
        Critical for progress tracking - wrong duration would show incorrect progress.
        """
        short_file = Path("/fake/short_clip.mkv")
        long_file = Path("/fake/full_movie.mkv")
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
            def mock_duration_responses(file_path, field):
                if 'short_clip.mkv' in str(file_path):
                    return "60.0"  # 1 minute
                elif 'full_movie.mkv' in str(file_path):
                    return "7200.0"  # 2 hours
                else:
                    return "0.0"
            
            mock_ffprobe.side_effect = mock_duration_responses
            
            # Test multiple times to ensure cache consistency
            for i in range(3):
                short_duration = get_duration_sec(short_file)
                long_duration = get_duration_sec(long_file)
                
                with self.subTest(iteration=i):
                    self.assertEqual(short_duration, 60.0, f"Short file should be 60s (iteration {i})")
                    self.assertEqual(long_duration, 7200.0, f"Long file should be 7200s (iteration {i})")


class TestCodecDetectionRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent codec detection regressions.
    
    Wrong codec detection leads to incorrect processing decisions and can cause
    files to be skipped when they should be transcoded or vice versa.
    """
    
    def test_codec_detection_identifies_all_major_codecs(self):
        """
        COMPATIBILITY TEST: Should correctly identify all commonly used codecs.
        
        This ensures the system can make correct processing decisions for all
        types of video content.
        """
        codec_test_cases = [
            # H.264 variants
            ("h264", "h264"),
            ("avc1", "avc1"), 
            ("x264", "x264"),
            
            # H.265/HEVC variants  
            ("hevc", "hevc"),
            ("h265", "h265"),
            ("x265", "x265"),
            
            # Other modern codecs
            ("av1", "av1"),
            ("vp9", "vp9"),
            ("vp8", "vp8"),
            
            # Legacy codecs
            ("mpeg4", "mpeg4"),
            ("mpeg2video", "mpeg2video"),
            ("xvid", "xvid"),
        ]
        
        for codec_input, expected_output in codec_test_cases:
            with self.subTest(codec=codec_input):
                test_file = Path(f"/fake/{codec_input}_video.mkv")
                
                with patch('lazy_transcode.core.modules.analysis.media_utils.run_command') as mock_run:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = codec_input
                    mock_run.return_value = mock_result
                    
                    result = get_video_codec(test_file)
                    self.assertEqual(result, expected_output,
                                   f"Codec {codec_input} should be detected as {expected_output}")
    
    def test_codec_detection_handles_ffprobe_failure_gracefully(self):
        """
        ERROR HANDLING TEST: ffprobe failures should not cause crashes.
        
        When codec detection fails, the system should handle it gracefully
        rather than crashing.
        """
        test_file = Path("/fake/unreadable.mkv")
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.run_command') as mock_run:
            # Mock ffprobe failure
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "No such file or directory"
            mock_run.return_value = mock_result
            
            result = get_video_codec(test_file)
            self.assertIsNone(result, "Failed codec detection should return None")
    
    def test_codec_detection_handles_empty_or_malformed_output(self):
        """
        ROBUSTNESS TEST: Empty or malformed ffprobe output should be handled.
        
        This prevents issues when ffprobe returns unexpected output formats.
        """
        malformed_cases = [
            "",              # Empty output
            " ",             # Whitespace only
            "unknown",       # Unknown codec
            "N/A",          # N/A response
            "codec_name=h264", # Key=value format
            None,           # No stdout
        ]
        
        for malformed_output in malformed_cases:
            with self.subTest(output=repr(malformed_output)):
                test_file = Path(f"/fake/malformed_{hash(str(malformed_output))}.mkv")
                
                with patch('lazy_transcode.core.modules.analysis.media_utils.run_command') as mock_run:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = malformed_output or ""
                    mock_run.return_value = mock_result
                    
                    result = get_video_codec(test_file)
                    # Should either return None or a valid string, but not crash
                    self.assertIsInstance(result, (str, type(None)),
                                        "Codec detection should return string or None")


class TestMetadataExtractionIntegration(unittest.TestCase):
    """
    INTEGRATION TESTS: Test metadata extraction functions working together.
    
    These tests ensure that the various metadata extraction functions don't
    interfere with each other and work correctly as a complete system.
    """
    
    def setUp(self):
        """Clear all metadata caches before each test."""
        ffprobe_field.cache_clear()
        get_video_dimensions.cache_clear() 
        get_duration_sec.cache_clear()
    
    def test_metadata_extraction_workflow_consistency(self):
        """
        WORKFLOW TEST: Complete metadata extraction should be consistent.
        
        This tests that extracting all metadata for a file works correctly
        and doesn't interfere between different extraction calls.
        """
        test_file = Path("/fake/complete_test.mkv")
        
        # Mock comprehensive file metadata
        mock_responses = {
            "stream=codec_name": "h264",
            "stream=width,height": "1920x1080", 
            "format=duration": "7200.123",
            "stream=pix_fmt": "yuv420p",
            "format=bit_rate": "5000000",
        }
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
            def mock_metadata_response(file_path, field):
                return mock_responses.get(field, "unknown")
            
            mock_ffprobe.side_effect = mock_metadata_response
            
            # Extract all metadata
            codec = get_video_codec(test_file)
            dimensions = get_video_dimensions(test_file)
            duration = get_duration_sec(test_file)
            pixel_format = ffprobe_field(test_file, "stream=pix_fmt")
            bitrate = ffprobe_field(test_file, "format=bit_rate")
            
            # Verify all metadata is correct
            self.assertEqual(codec, "h264", "Codec should be h264")
            self.assertEqual(dimensions, (1920, 1080), "Dimensions should be 1920x1080")
            self.assertAlmostEqual(duration, 7200.123, places=3, msg="Duration should be 7200.123")
            self.assertEqual(pixel_format, "yuv420p", "Pixel format should be yuv420p")
            self.assertEqual(bitrate, "5000000", "Bitrate should be 5000000")
            
            # Test that repeated calls return same results (cache working)
            codec2 = get_video_codec(test_file)
            dimensions2 = get_video_dimensions(test_file)
            duration2 = get_duration_sec(test_file)
            
            self.assertEqual(codec, codec2, "Cached codec should match")
            self.assertEqual(dimensions, dimensions2, "Cached dimensions should match")
            self.assertEqual(duration, duration2, "Cached duration should match")
    
    def test_multiple_files_metadata_isolation(self):
        """
        ISOLATION TEST: Metadata for multiple files should not interfere.
        
        This is critical for batch processing - ensuring metadata from one
        file doesn't leak into another file's metadata.
        """
        files = [
            Path("/fake/anime_episode.mkv"),
            Path("/fake/movie_bluray.mkv"),
            Path("/fake/documentary_4k.mp4"),
        ]
        
        # Different metadata for each file
        expected_metadata = {
            str(files[0]): {"codec": "h264", "dimensions": (1920, 1080), "duration": 1440.0},
            str(files[1]): {"codec": "hevc", "dimensions": (3840, 2160), "duration": 7200.0},
            str(files[2]): {"codec": "av1", "dimensions": (7680, 4320), "duration": 3600.0},
        }
        
        def mock_file_specific_metadata(file_path, field):
            file_str = str(file_path)
            metadata = expected_metadata.get(file_str, {})
            
            if field == "stream=codec_name":
                return metadata.get("codec", "unknown")
            elif field == "stream=width,height":
                dims = metadata.get("dimensions", (0, 0))
                return f"{dims[0]}x{dims[1]}"
            elif field == "format=duration":
                return str(metadata.get("duration", 0.0))
            else:
                return "unknown"
        
        with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field') as mock_ffprobe:
            mock_ffprobe.side_effect = mock_file_specific_metadata
            
            # Extract metadata for all files multiple times in random order
            import random
            for iteration in range(3):
                shuffled_files = files.copy()
                random.shuffle(shuffled_files)
                
                for file_path in shuffled_files:
                    expected = expected_metadata[str(file_path)]
                    
                    codec = get_video_codec(file_path)
                    dimensions = get_video_dimensions(file_path)
                    duration = get_duration_sec(file_path)
                    
                    with self.subTest(iteration=iteration, file=file_path.name):
                        self.assertEqual(codec, expected["codec"],
                                       f"{file_path.name} codec should be {expected['codec']}")
                        self.assertEqual(dimensions, expected["dimensions"],
                                       f"{file_path.name} dimensions should be {expected['dimensions']}")
                        self.assertAlmostEqual(duration, expected["duration"], places=1,
                                             msg=f"{file_path.name} duration should be {expected['duration']}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
