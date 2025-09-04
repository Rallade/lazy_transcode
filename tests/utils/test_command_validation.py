"""
Command validation tests to prevent transcoding bugs.

These tests validate the actual FFmpeg commands generated to ensure
they include all necessary flags for stream preservation and are
syntactically correct.
"""

import unittest
import re
from pathlib import Path
from unittest.mock import patch

from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder
from lazy_transcode.core.modules.optimization.vbr_optimizer import build_vbr_encode_cmd


class TestFFmpegCommandValidation(unittest.TestCase):
    """Validate generated FFmpeg commands for correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = EncoderConfigBuilder()
    
    def test_command_contains_all_required_stream_preservation_flags(self):
        """
        CRITICAL TEST: Ensure all required stream preservation flags are present.
        
        This test would have caught the original bug by verifying that
        ALL required flags are present in the generated command.
        """
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = self.builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                cmd_str = ' '.join(cmd)
                
                # CRITICAL: These flags MUST be present
                required_flags = [
                    # Stream mapping (most critical)
                    r'-map\s+0\b',                    # Map all input streams
                    r'-map_metadata\s+0\b',           # Copy metadata
                    r'-map_chapters\s+0\b',           # Copy chapters
                    
                    # Stream copying
                    r'-c:a\s+copy\b',                 # Copy audio
                    r'-c:s\s+copy\b',                 # Copy subtitles
                    r'-c:d\s+copy\b',                 # Copy data streams
                    r'-c:t\s+copy\b',                 # Copy timecode
                    r'-copy_unknown\b',               # Copy unknown streams
                ]
                
                for pattern in required_flags:
                    with self.subTest(pattern=pattern):
                        self.assertRegex(cmd_str, pattern, 
                                       f"Missing required flag pattern: {pattern}")
    
    def test_command_structure_is_valid_ffmpeg_syntax(self):
        """
        SYNTAX TEST: Validate command follows proper FFmpeg syntax rules.
        """
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = self.builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                # Basic structure validation
                self.assertGreater(len(cmd), 10, "Command too short")
                self.assertEqual(cmd[0], 'ffmpeg', "Must start with ffmpeg")
                self.assertIn('-i', cmd, "Must have input flag")
                self.assertIn('input.mkv', cmd, "Must have input file")
                self.assertEqual(cmd[-1], 'output.mkv', "Must end with output file")
                
                # Flag-argument pairing validation
                flag_value_pairs = [
                    ('-i', 'input.mkv'),
                    ('-c:v', 'libx265'),
                    ('-c:a', 'copy'),
                    ('-c:s', 'copy'),
                    ('-c:d', 'copy'),
                    ('-c:t', 'copy'),
                    ('-map', '0'),
                    ('-map_metadata', '0'),
                    ('-map_chapters', '0'),
                ]
                
                for flag, expected_value in flag_value_pairs:
                    with self.subTest(flag=flag, value=expected_value):
                        try:
                            flag_idx = cmd.index(flag)
                            actual_value = cmd[flag_idx + 1]
                            self.assertEqual(actual_value, expected_value,
                                           f"Flag {flag} should have value {expected_value}, got {actual_value}")
                        except (ValueError, IndexError):
                            self.fail(f"Flag {flag} not found or missing value")
    
    def test_no_conflicting_flags_present(self):
        """
        CONFLICT TEST: Ensure no conflicting flags are present.
        
        This catches cases where contradictory flags might cause issues.
        """
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = self.builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                cmd_str = ' '.join(cmd)
                
                # Flags that would conflict with stream preservation
                conflicting_flags = [
                    '-an',              # Disable audio
                    '-sn',              # Disable subtitles
                    '-dn',              # Disable data streams
                    '-map_chapters -1', # Disable chapters
                    '-map_metadata -1', # Disable metadata
                ]
                
                for bad_flag in conflicting_flags:
                    self.assertNotIn(bad_flag, cmd_str, 
                                   f"Conflicting flag found: {bad_flag}")
    
    def test_vbr_optimizer_generates_comprehensive_commands(self):
        """
        INTEGRATION TEST: Test vbr_optimizer command generation.
        
        This ensures the VBR optimizer (which should be used) generates
        comprehensive commands with full stream preservation.
        """
        with patch('lazy_transcode.core.modules.analysis.media_utils.get_video_dimensions') as mock_dims:
            mock_dims.return_value = (1920, 1080)
            
            cmd = build_vbr_encode_cmd(
                Path("input.mkv"), Path("output.mkv"),
                "libx265", "software", 5000, 4000
            )
            
            cmd_str = ' '.join(cmd)
            
            # Verify comprehensive stream preservation
            self.assertIn('-map 0', cmd_str, "VBR optimizer missing stream mapping")
            self.assertIn('-map_metadata 0', cmd_str, "VBR optimizer missing metadata")
            self.assertIn('-map_chapters 0', cmd_str, "VBR optimizer missing chapters")
            self.assertIn('-c:a copy', cmd_str, "VBR optimizer missing audio copy")
            self.assertIn('-c:s copy', cmd_str, "VBR optimizer missing subtitle copy")
            self.assertIn('-c:d copy', cmd_str, "VBR optimizer missing data copy")
            self.assertIn('-c:t copy', cmd_str, "VBR optimizer missing timecode copy")
            self.assertIn('-copy_unknown', cmd_str, "VBR optimizer missing unknown copy")


class TestStreamPreservationPatterns(unittest.TestCase):
    """Test specific patterns that indicate proper stream preservation."""
    
    def test_comprehensive_stream_mapping_pattern(self):
        """
        PATTERN TEST: Verify the comprehensive stream mapping pattern.
        
        Tests for the specific pattern: -map 0 -map_metadata 0 -map_chapters 0
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                cmd_str = ' '.join(cmd)
                
                # Test for the comprehensive mapping pattern
                comprehensive_pattern = r'-map\s+0.*-map_metadata\s+0.*-map_chapters\s+0'
                self.assertRegex(cmd_str, comprehensive_pattern, 
                               "Missing comprehensive mapping pattern")
    
    def test_complete_stream_copying_pattern(self):
        """
        PATTERN TEST: Verify all stream types are set to copy.
        
        Tests for: -c:a copy -c:s copy -c:d copy -c:t copy
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                cmd_str = ' '.join(cmd)
                
                # Test for complete stream copying pattern
                copy_patterns = [
                    r'-c:a\s+copy',     # Audio copy
                    r'-c:s\s+copy',     # Subtitle copy
                    r'-c:d\s+copy',     # Data copy
                    r'-c:t\s+copy',     # Timecode copy
                ]
                
                for pattern in copy_patterns:
                    with self.subTest(pattern=pattern):
                        self.assertRegex(cmd_str, pattern, 
                                       f"Missing stream copy pattern: {pattern}")
    
    def test_no_stream_exclusion_patterns(self):
        """
        NEGATIVE TEST: Ensure no stream exclusion patterns are present.
        
        Verifies that patterns that would exclude streams are not present.
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                cmd_str = ' '.join(cmd)
                
                # Patterns that would exclude streams (should NOT be present)
                exclusion_patterns = [
                    r'-map\s+-0:a',     # Exclude audio
                    r'-map\s+-0:s',     # Exclude subtitles
                    r'-an\b',           # Disable audio
                    r'-sn\b',           # Disable subtitles
                    r'-dn\b',           # Disable data
                    r'-c:a\s+none',     # No audio codec
                    r'-c:s\s+none',     # No subtitle codec
                ]
                
                for pattern in exclusion_patterns:
                    with self.subTest(pattern=pattern):
                        self.assertNotRegex(cmd_str, pattern, 
                                          f"Found stream exclusion pattern: {pattern}")


class TestHardwareEncoderStreamPreservation(unittest.TestCase):
    """Test stream preservation specifically for hardware encoders."""
    
    def test_nvenc_preserves_all_streams(self):
        """Test NVIDIA NVENC preserves all streams."""
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            mock_ffprobe.return_value = "yuv420p"
            
            cmd = builder.build_vbr_encode_cmd(
                "input.mkv", "output.mkv", "hevc_nvenc", "medium", 5000,
                3, 3, 1920, 1080
            )
            
            cmd_str = ' '.join(cmd)
            
            # NVENC must preserve streams just like software encoder
            self.assertIn('-map 0', cmd_str)
            self.assertIn('-map_metadata 0', cmd_str)
            self.assertIn('-map_chapters 0', cmd_str)
            self.assertIn('-c:a copy', cmd_str)
            self.assertIn('-c:s copy', cmd_str)
    
    def test_amf_preserves_all_streams(self):
        """Test AMD AMF preserves all streams."""
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            mock_ffprobe.return_value = "yuv420p"
            
            cmd = builder.build_vbr_encode_cmd(
                "input.mkv", "output.mkv", "hevc_amf", "medium", 5000,
                3, 3, 1920, 1080
            )
            
            cmd_str = ' '.join(cmd)
            
            # AMF must preserve streams
            self.assertIn('-map 0', cmd_str)
            self.assertIn('-map_metadata 0', cmd_str)
            self.assertIn('-map_chapters 0', cmd_str)
            self.assertIn('-c:a copy', cmd_str)
            self.assertIn('-c:s copy', cmd_str)
    
    def test_qsv_preserves_all_streams(self):
        """Test Intel QuickSync preserves all streams."""
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            mock_ffprobe.return_value = "yuv420p"
            
            cmd = builder.build_vbr_encode_cmd(
                "input.mkv", "output.mkv", "hevc_qsv", "medium", 5000,
                3, 3, 1920, 1080
            )
            
            cmd_str = ' '.join(cmd)
            
            # QSV must preserve streams
            self.assertIn('-map 0', cmd_str)
            self.assertIn('-map_metadata 0', cmd_str)
            self.assertIn('-map_chapters 0', cmd_str)
            self.assertIn('-c:a copy', cmd_str)
            self.assertIn('-c:s copy', cmd_str)


class TestCommandGenerationConsistency(unittest.TestCase):
    """Test that command generation is consistent across different scenarios."""
    
    def test_consistent_stream_preservation_across_bitrates(self):
        """
        CONSISTENCY TEST: Stream preservation should be consistent regardless of bitrate.
        """
        builder = EncoderConfigBuilder()
        bitrates = [1000, 5000, 10000, 20000]  # Different bitrate scenarios
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                for bitrate in bitrates:
                    with self.subTest(bitrate=bitrate):
                        cmd = builder.build_vbr_encode_cmd(
                            "input.mkv", "output.mkv", "libx265", "medium", bitrate,
                            3, 3, 1920, 1080
                        )
                        
                        cmd_str = ' '.join(cmd)
                        
                        # Stream preservation must be consistent regardless of bitrate
                        self.assertIn('-map 0', cmd_str)
                        self.assertIn('-c:a copy', cmd_str)
                        self.assertIn('-c:s copy', cmd_str)
                        self.assertIn('-map_chapters 0', cmd_str)
    
    def test_consistent_stream_preservation_across_resolutions(self):
        """
        CONSISTENCY TEST: Stream preservation should be consistent regardless of resolution.
        """
        builder = EncoderConfigBuilder()
        resolutions = [
            (1280, 720),    # 720p
            (1920, 1080),   # 1080p
            (3840, 2160),   # 4K
        ]
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                for width, height in resolutions:
                    with self.subTest(resolution=f"{width}x{height}"):
                        cmd = builder.build_vbr_encode_cmd(
                            "input.mkv", "output.mkv", "libx265", "medium", 5000,
                            3, 3, width, height
                        )
                        
                        cmd_str = ' '.join(cmd)
                        
                        # Stream preservation must be consistent regardless of resolution
                        self.assertIn('-map 0', cmd_str)
                        self.assertIn('-c:a copy', cmd_str)
                        self.assertIn('-c:s copy', cmd_str)
                        self.assertIn('-map_chapters 0', cmd_str)


class TestCommandGenerationEdgeCases(unittest.TestCase):
    """Test command generation in edge cases that might break stream preservation."""
    
    def test_hdr_content_still_preserves_streams(self):
        """
        EDGE CASE TEST: HDR content should still preserve all streams.
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p10le"  # 10-bit HDR
                
                cmd = builder.build_vbr_encode_cmd(
                    "hdr_input.mkv", "hdr_output.mkv", "libx265", "medium", 8000,
                    3, 3, 3840, 2160, preserve_hdr=True
                )
                
                cmd_str = ' '.join(cmd)
                
                # HDR processing must not interfere with stream preservation
                self.assertIn('-map 0', cmd_str)
                self.assertIn('-map_metadata 0', cmd_str)
                self.assertIn('-map_chapters 0', cmd_str)
                self.assertIn('-c:a copy', cmd_str)
                self.assertIn('-c:s copy', cmd_str)
                
                # Should also have HDR settings
                self.assertIn('main10', cmd_str)  # 10-bit profile
    
    def test_debug_mode_still_preserves_streams(self):
        """
        EDGE CASE TEST: Debug mode should not affect stream preservation.
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.config.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080, debug=True
                )
                
                cmd_str = ' '.join(cmd)
                
                # Debug mode must not interfere with stream preservation
                self.assertIn('-map 0', cmd_str)
                self.assertIn('-c:a copy', cmd_str)
                self.assertIn('-c:s copy', cmd_str)
                self.assertIn('-map_chapters 0', cmd_str)
                
                # Should have debug logging
                self.assertIn('-loglevel info', cmd_str)


if __name__ == '__main__':
    unittest.main()
