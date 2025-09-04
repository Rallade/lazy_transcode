"""
Unit tests for stream preservation and encoder integration.

Tests the comprehensive encoder configuration and stream preservation
functionality that ensures audio, subtitles, chapters, and metadata
are properly preserved during transcoding.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.encoder_config import EncoderConfigBuilder


class TestEncoderConfigBuilderStreamPreservation(unittest.TestCase):
    """Test EncoderConfigBuilder stream preservation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.builder = EncoderConfigBuilder()
        self.input_file = "test_input.mkv"
        self.output_file = "test_output.mkv"
    
    def test_default_stream_copying_enabled(self):
        """Test that stream copying is enabled by default in VBR commands."""
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = self.builder.build_vbr_encode_cmd(
                    self.input_file, self.output_file,
                    encoder="libx265", preset="medium", bitrate=5000,
                    bf=3, refs=3, width=1920, height=1080
                )
        
        cmd_str = ' '.join(cmd)
        
        # Verify comprehensive stream mapping
        self.assertIn('-map 0', cmd_str)
        self.assertIn('-map_metadata 0', cmd_str)
        self.assertIn('-map_chapters 0', cmd_str)
        
        # Verify stream copying
        self.assertIn('-c:a copy', cmd_str)
        self.assertIn('-c:s copy', cmd_str)
        self.assertIn('-c:d copy', cmd_str)
        self.assertIn('-c:t copy', cmd_str)
        self.assertIn('-copy_unknown', cmd_str)
    
    def test_comprehensive_stream_mapping(self):
        """Test that all stream types are properly mapped."""
        self.builder.set_base_config("input.mkv", "output.mkv")
        self.builder.set_output_format(copy_streams=True)
        
        cmd = self.builder.build_command()
        cmd_str = ' '.join(cmd)
        
        # Should include comprehensive mapping when copy_streams=True
        self.assertIn('-map 0', cmd_str)
        self.assertIn('-map_metadata 0', cmd_str)
        self.assertIn('-map_chapters 0', cmd_str)
    
    def test_custom_stream_mapping(self):
        """Test custom stream mapping overrides default behavior."""
        self.builder.set_base_config("input.mkv", "output.mkv")
        self.builder.set_output_format(
            map_streams=['0:v:0', '0:a:0'],
            copy_streams=False
        )
        
        cmd = self.builder.build_command()
        cmd_str = ' '.join(cmd)
        
        # Should use custom mapping
        self.assertIn('-map 0:v:0', cmd_str)
        self.assertIn('-map 0:a:0', cmd_str)
        # Should not include default comprehensive mapping
        self.assertNotIn('-map 0', cmd_str)
    
    def test_hardware_encoder_stream_preservation(self):
        """Test stream preservation with hardware encoders."""
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            mock_ffprobe.return_value = "yuv420p"
            
            # Test NVENC
            cmd = self.builder.build_vbr_encode_cmd(
                self.input_file, self.output_file,
                encoder="hevc_nvenc", preset="medium", bitrate=5000,
                bf=3, refs=3, width=1920, height=1080
            )
        
        cmd_str = ' '.join(cmd)
        
        # Verify stream preservation is included even with hardware encoders
        self.assertIn('-map 0', cmd_str)
        self.assertIn('-map_metadata 0', cmd_str)
        self.assertIn('-map_chapters 0', cmd_str)
        self.assertIn('-c:a copy', cmd_str)
        self.assertIn('-c:s copy', cmd_str)
    
    def test_hdr_preservation_with_stream_copying(self):
        """Test HDR preservation combined with stream copying."""
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p10le"  # 10-bit format
                
                cmd = self.builder.build_vbr_encode_cmd(
                    self.input_file, self.output_file,
                    encoder="libx265", preset="medium", bitrate=5000,
                    bf=3, refs=3, width=1920, height=1080,
                    preserve_hdr=True
                )
        
        cmd_str = ' '.join(cmd)
        
        # Should have both HDR preservation and stream copying
        self.assertIn('main10', cmd_str)  # HDR profile
        self.assertIn('-c:a copy', cmd_str)  # Stream copying
        self.assertIn('-map 0', cmd_str)     # Comprehensive mapping


class TestVBROptimizerIntegration(unittest.TestCase):
    """Test VBR optimizer integration with comprehensive encoder."""
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.EncoderConfigBuilder')
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_video_dimensions')
    def test_vbr_optimizer_uses_comprehensive_builder(self, mock_get_dims, mock_builder_class):
        """Test that VBR optimizer uses the comprehensive EncoderConfigBuilder."""
        from lazy_transcode.core.modules.vbr_optimizer import build_vbr_encode_cmd
        
        # Setup mocks
        mock_get_dims.return_value = (1920, 1080)
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        
        expected_cmd = [
            'ffmpeg', '-y', '-i', 'input.mkv',
            '-map', '0', '-map_metadata', '0', '-map_chapters', '0',
            '-c:v', 'libx265', '-c:a', 'copy', '-c:s', 'copy',
            'output.mkv'
        ]
        mock_builder.build_vbr_encode_cmd.return_value = expected_cmd
        
        # Call the function
        result = build_vbr_encode_cmd(
            Path("input.mkv"), Path("output.mkv"),
            "libx265", "software", 5000, 4000
        )
        
        # Verify EncoderConfigBuilder was instantiated and used
        mock_builder_class.assert_called_once()
        mock_builder.build_vbr_encode_cmd.assert_called_once()
        
        # Verify the parameters passed to the builder
        call_args = mock_builder.build_vbr_encode_cmd.call_args
        self.assertEqual(call_args[0][0], "input.mkv")  # input file
        self.assertEqual(call_args[0][1], "output.mkv")  # output file
        self.assertEqual(call_args[0][2], "libx265")     # encoder
        
        self.assertEqual(result, expected_cmd)


class TestStreamPreservationCommand(unittest.TestCase):
    """Test stream preservation in actual FFmpeg commands."""
    
    def test_comprehensive_stream_preservation_command(self):
        """Test that generated commands include comprehensive stream preservation."""
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv",
                    encoder="libx265", preset="medium", bitrate=5000,
                    bf=3, refs=3, width=1920, height=1080,
                    preserve_hdr=False, debug=False
                )
        
        # Convert to string for easier testing
        cmd_str = ' '.join(cmd)
        
        # Test comprehensive stream preservation
        preservation_checks = [
            ('-map 0', 'Map all input streams'),
            ('-map_metadata 0', 'Copy all metadata'),
            ('-map_chapters 0', 'Copy all chapters'),
            ('-c:a copy', 'Copy audio streams'),
            ('-c:s copy', 'Copy subtitle streams'),
            ('-c:d copy', 'Copy data streams'),
            ('-c:t copy', 'Copy timecode streams'),
            ('-copy_unknown', 'Copy unknown stream types')
        ]
        
        for flag, description in preservation_checks:
            with self.subTest(flag=flag, description=description):
                self.assertIn(flag, cmd_str, f"Missing {description}: {flag}")
    
    def test_progress_reporting_included(self):
        """Test that progress reporting is included by default."""
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv",
                    encoder="libx265", preset="medium", bitrate=5000,
                    bf=3, refs=3, width=1920, height=1080
                )
        
        cmd_str = ' '.join(cmd)
        
        # Should include progress reporting
        self.assertIn('-progress pipe:1', cmd_str)
        self.assertIn('-nostats', cmd_str)
    
    def test_different_encoder_types_preserve_streams(self):
        """Test that different encoder types all preserve streams."""
        builder = EncoderConfigBuilder()
        
        encoders_to_test = [
            ("libx265", "software"),
            ("hevc_nvenc", "hardware"),
            ("hevc_amf", "hardware"),
            ("hevc_qsv", "hardware")
        ]
        
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            mock_ffprobe.return_value = "yuv420p"
            
            for encoder, encoder_type in encoders_to_test:
                with self.subTest(encoder=encoder, encoder_type=encoder_type):
                    cmd = builder.build_vbr_encode_cmd(
                        "input.mkv", "output.mkv",
                        encoder=encoder, preset="medium", bitrate=5000,
                        bf=3, refs=3, width=1920, height=1080
                    )
                    
                    cmd_str = ' '.join(cmd)
                    
                    # All encoder types should preserve streams
                    self.assertIn('-map 0', cmd_str)
                    self.assertIn('-c:a copy', cmd_str)
                    self.assertIn('-c:s copy', cmd_str)


class TestErrorHandlingInStreamPreservation(unittest.TestCase):
    """Test error handling in stream preservation functionality."""
    
    def test_builder_handles_missing_input_gracefully(self):
        """Test that builder handles missing input file gracefully."""
        builder = EncoderConfigBuilder()
        
        # Test with non-existent input file
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                # Simulate ffprobe failure
                mock_ffprobe.return_value = None
                
                cmd = builder.build_vbr_encode_cmd(
                    "nonexistent.mkv", "output.mkv",
                    encoder="libx265", preset="medium", bitrate=5000,
                    bf=3, refs=3, width=1920, height=1080
                )
        
        # Should still generate a valid command with stream preservation
        cmd_str = ' '.join(cmd)
        self.assertIn('ffmpeg', cmd_str)
        self.assertIn('-map 0', cmd_str)
        self.assertIn('-c:a copy', cmd_str)
    
    def test_builder_handles_invalid_parameters_gracefully(self):
        """Test that builder handles invalid parameters gracefully."""
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                # Test with edge case parameters
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv",
                    encoder="libx265", preset="medium", bitrate=1,  # Very low bitrate
                    bf=0, refs=0, width=1, height=1  # Minimal values
                )
        
        # Should still generate valid command
        cmd_str = ' '.join(cmd)
        self.assertIn('ffmpeg', cmd_str)
        self.assertIn('-map 0', cmd_str)


if __name__ == '__main__':
    unittest.main()
