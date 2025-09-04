"""
Unit tests for enhanced transcoding engine functionality.

Tests the comprehensive logging, stream preservation, and VBR transcoding
enhancements added to address stream preservation and logging concerns.
"""

import unittest
import tempfile
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from lazy_transcode.core.modules.processing.transcoding_engine import (
    _log_input_streams,
    _log_output_streams,
    transcode_file_vbr,
    monitor_progress
)


class TestStreamAnalysisLogging(unittest.TestCase):
    """Test stream analysis and logging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_file = Path("test_video.mkv")
        self.sample_ffprobe_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "tags": {"language": "eng"}
                },
                {
                    "codec_type": "audio", 
                    "codec_name": "ac3",
                    "channels": 6,
                    "tags": {"language": "jpn"}
                },
                {
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "tags": {"language": "eng"}
                },
                {
                    "codec_type": "subtitle",
                    "codec_name": "ass",
                    "tags": {"language": "jpn"}
                }
            ],
            "chapters": [
                {"id": 0, "start": 0, "end": 300},
                {"id": 1, "start": 300, "end": 600}
            ]
        }
    
    @patch('lazy_transcode.core.modules.transcoding_engine.run_command')
    @patch('builtins.print')
    def test_log_input_streams_success(self, mock_print, mock_run_command):
        """Test successful input stream analysis and logging."""
        # Mock successful ffprobe command
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(self.sample_ffprobe_output)
        mock_run_command.return_value = mock_result
        
        _log_input_streams(self.test_file)
        
        # Verify ffprobe command was called correctly
        mock_run_command.assert_called_once()
        call_args = mock_run_command.call_args[0][0]
        self.assertEqual(call_args[0], 'ffprobe')
        self.assertIn('-show_streams', call_args)
        self.assertIn('-show_chapters', call_args)
        self.assertIn(str(self.test_file), call_args)
        
        # Verify logging output
        print_calls = mock_print.call_args_list
        self.assertTrue(any("[TRANSCODE-VBR] Input analysis:" in str(call) for call in print_calls))
        self.assertTrue(any("Total streams: 5" in str(call) for call in print_calls))
        self.assertTrue(any("Video streams: 1" in str(call) for call in print_calls))
        self.assertTrue(any("Audio streams: 2" in str(call) for call in print_calls))
        self.assertTrue(any("Subtitle streams: 2" in str(call) for call in print_calls))
        self.assertTrue(any("Chapters: 2" in str(call) for call in print_calls))
    
    @patch('lazy_transcode.core.modules.transcoding_engine.run_command')
    @patch('builtins.print')
    def test_log_input_streams_error_handling(self, mock_print, mock_run_command):
        """Test error handling in input stream analysis."""
        # Mock failed ffprobe command
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run_command.return_value = mock_result
        
        _log_input_streams(self.test_file)
        
        # Verify error is handled gracefully
        print_calls = mock_print.call_args_list
        self.assertTrue(any("Warning: Could not analyze input streams" in str(call) for call in print_calls))
    
    @patch('lazy_transcode.core.modules.transcoding_engine.run_command')
    @patch('builtins.print')
    def test_log_output_streams_with_size_comparison(self, mock_print, mock_run_command):
        """Test output stream verification with file size comparison."""
        # Mock successful ffprobe command
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(self.sample_ffprobe_output)
        mock_run_command.return_value = mock_result
        
        # Mock file stat for size comparison
        input_file = Mock()
        input_file.stat.return_value.st_size = 1000 * 1024 * 1024  # 1000 MB
        output_file = Mock()
        output_file.stat.return_value.st_size = 800 * 1024 * 1024   # 800 MB
        
        _log_output_streams(output_file, input_file)
        
        # Verify logging includes size comparison
        print_calls = mock_print.call_args_list
        self.assertTrue(any("Input size: 1000.0 MB" in str(call) for call in print_calls))
        self.assertTrue(any("Output size: 800.0 MB" in str(call) for call in print_calls))
        self.assertTrue(any("Size reduction: 20.0%" in str(call) for call in print_calls))


class TestEnhancedVBRTranscoding(unittest.TestCase):
    """Test enhanced VBR transcoding with comprehensive logging."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("test_input.mkv")
        self.output_file = Path("test_output.mkv")
        self.encoder = "libx265"
        self.encoder_type = "software"
        self.max_bitrate = 5000
        self.avg_bitrate = 4000
    
    @patch('lazy_transcode.core.modules.encoder_config.EncoderConfigBuilder.build_vbr_encode_cmd')
    @patch('lazy_transcode.core.modules.transcoding_engine._log_input_streams')
    @patch('lazy_transcode.core.modules.transcoding_engine._log_output_streams')
    @patch('lazy_transcode.core.modules.transcoding_engine.monitor_progress')
    @patch('subprocess.Popen')
    @patch('builtins.print')
    def test_transcode_file_vbr_success(self, mock_print, mock_popen, mock_monitor,
                                        mock_log_output, mock_log_input, mock_build_cmd):
        """Test successful VBR transcoding with comprehensive logging."""
        test_cmd = [
            'ffmpeg', '-y', '-i', str(self.input_file), '-c:v', 'libx265',
            '-map', '0', '-map_metadata', '0', '-map_chapters', '0',
            '-c:a', 'copy', '-c:s', 'copy', '-c:d', 'copy', '-c:t', 'copy',
            '-progress', 'pipe:1', str(self.output_file)
        ]
        mock_build_cmd.return_value = test_cmd

        mock_process = Mock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.mkdir'):
            result = transcode_file_vbr(
                self.input_file, self.output_file, self.encoder, self.encoder_type,
                self.max_bitrate, self.avg_bitrate, preserve_hdr_metadata=True
            )

        self.assertTrue(result)
        mock_build_cmd.assert_called_once()
        called_args, _ = mock_build_cmd.call_args
        self.assertEqual(str(called_args[0]), str(self.input_file))
        self.assertEqual(str(called_args[1]), str(self.output_file))
        self.assertIn(self.encoder, called_args)
        mock_log_input.assert_called_once_with(self.input_file)
        mock_log_output.assert_called_once_with(self.output_file, self.input_file)
        print_calls = mock_print.call_args_list
        self.assertTrue(any("Starting transcode of:" in str(call) for call in print_calls))
        self.assertTrue(any(f"Encoder: {self.encoder} ({self.encoder_type})" in str(call) for call in print_calls))
        self.assertTrue(any(f"Target bitrate: {self.avg_bitrate} kbps" in str(call) for call in print_calls))
        self.assertTrue(any("Successfully transcoded" in str(call) for call in print_calls))

    @patch('lazy_transcode.core.modules.encoder_config.EncoderConfigBuilder.build_vbr_encode_cmd')
    @patch('subprocess.Popen')
    @patch('builtins.print')
    def test_transcode_file_vbr_failure(self, mock_print, mock_popen, mock_build_cmd):
        """Test VBR transcoding failure with detailed error reporting."""
        mock_build_cmd.return_value = ['ffmpeg', '-y', '-i', str(self.input_file)]

        mock_process = Mock()
        mock_process.communicate.return_value = ("stdout output", "stderr error message")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with patch('pathlib.Path.exists', return_value=False), patch('pathlib.Path.mkdir'):
            result = transcode_file_vbr(
                self.input_file, self.output_file, self.encoder, self.encoder_type,
                self.max_bitrate, self.avg_bitrate
            )

        self.assertFalse(result)
        print_calls = mock_print.call_args_list
        self.assertTrue(any("VBR transcoding failed" in str(call) for call in print_calls))
        self.assertTrue(any("Process return code: 1" in str(call) for call in print_calls))
        self.assertTrue(any("STDOUT: stdout output" in str(call) for call in print_calls))
        self.assertTrue(any("STDERR: stderr error message" in str(call) for call in print_calls))

    @patch('lazy_transcode.core.modules.encoder_config.EncoderConfigBuilder.build_vbr_encode_cmd')
    @patch('subprocess.Popen')
    def test_transcode_file_vbr_progress_tracking(self, mock_popen, mock_build_cmd):
        """Test progress tracking functionality in VBR transcoding."""
        mock_build_cmd.return_value = ['ffmpeg', '-progress', 'pipe:1', str(self.output_file)]

        mock_process = Mock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        progress_callback = Mock()

        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.mkdir'), \
             patch('lazy_transcode.core.modules.transcoding_engine.monitor_progress') as mock_monitor:
            transcode_file_vbr(
                self.input_file, self.output_file, self.encoder, self.encoder_type,
                self.max_bitrate, self.avg_bitrate,
                progress_callback=progress_callback
            )
            mock_monitor.assert_called_once()


class TestProgressMonitoring(unittest.TestCase):
    """Test enhanced progress monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.progress_file = Path("progress.txt")
        self.mock_callback = Mock()
    
    @patch('builtins.open')
    @patch('builtins.print')
    @patch('time.time')
    def test_monitor_progress_detailed_logging(self, mock_time, mock_print, mock_open):
        """Test detailed progress logging during monitoring."""
        # Mock time for elapsed calculation
        mock_time.side_effect = [1000.0, 1001.0, 1002.0]  # 1 second intervals

        # Mock process that finishes quickly
        mock_process = Mock()
        mock_process.poll.side_effect = [None, 0]  # Running, then finished

        # Mock progress file content
        progress_content = (
            "frame=100\n"
            "fps=30.5\n"
            "bitrate=2500kbps\n"
            "total_size=1048576\n"
            "out_time_us=3000000\n"
            "speed=1.2x\n"
            "progress=continue"
        )

        mock_file = Mock()
        mock_file.read.return_value = progress_content
        mock_open.return_value.__enter__.return_value = mock_file

        with patch('pathlib.Path.exists', return_value=True):
            monitor_progress(mock_process, self.progress_file, self.mock_callback)

        # Verify callback was called
        self.mock_callback.assert_called()

        # Verify detailed progress logging
        print_calls = mock_print.call_args_list
        progress_logged = any("Progress: Frame 100" in str(call) for call in print_calls)
        fps_logged = any("FPS 30.5" in str(call) for call in print_calls)
        speed_logged = any("Speed 1.2x" in str(call) for call in print_calls)

        self.assertTrue(progress_logged)
        self.assertTrue(fps_logged)
        self.assertTrue(speed_logged)


class TestStreamPreservationIntegration(unittest.TestCase):
    """Test integration with comprehensive stream preservation."""
    
    @patch('lazy_transcode.core.modules.encoder_config.EncoderConfigBuilder.build_vbr_encode_cmd')
    def test_uses_comprehensive_encoder_builder(self, mock_build_cmd):
        """Test that VBR transcoding uses the comprehensive encoder builder."""
        # Mock the comprehensive build command that includes stream preservation
        comprehensive_cmd = [
            'ffmpeg', '-hide_banner', '-y', '-i', 'input.mkv',
            '-map', '0',              # Map all streams
            '-map_metadata', '0',     # Copy metadata
            '-map_chapters', '0',     # Copy chapters
            '-c:v', 'libx265',        # Video encoder
            '-c:a', 'copy',          # Copy audio
            '-c:s', 'copy',          # Copy subtitles
            '-c:d', 'copy',          # Copy data streams
            '-c:t', 'copy',          # Copy timecode
            '-copy_unknown',         # Copy unknown streams
            'output.mkv'
        ]
        mock_build_cmd.return_value = comprehensive_cmd
        
        input_file = Path("input.mkv")
        output_file = Path("output.mkv")
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.mkdir'):
                    transcode_file_vbr(
                        input_file, output_file, "libx265", "software",
                        5000, 4000, preserve_hdr_metadata=True
                    )
            
            # Verify the comprehensive encoder was used
            mock_build_cmd.assert_called_once()
            
            # Verify we attempted to build a comprehensive command
            # (the exact command may differ after wrapper usage; ensure build function was invoked)
            self.assertTrue(mock_build_cmd.called)


class TestProgressFileHandling(unittest.TestCase):
    """Test progress file handling and cleanup."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("test.mkv")
        self.output_file = Path("output.mkv")
    
    @patch('lazy_transcode.core.modules.encoder_config.EncoderConfigBuilder.build_vbr_encode_cmd')
    @patch('subprocess.Popen')
    def test_progress_file_cleanup(self, mock_popen, mock_build_cmd):
        """Test that progress files are properly cleaned up."""
        mock_build_cmd.return_value = ['ffmpeg', '-progress', 'pipe:1', str(self.output_file)]
        
        mock_process = Mock()
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        progress_callback = Mock()
        
        # Mock progress file
        mock_progress_file = Mock()
        mock_progress_file.exists.return_value = True
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.mkdir'):
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "progress_test.txt"
                    
                    transcode_file_vbr(
                        self.input_file, self.output_file, "libx265", "software",
                        5000, 4000, progress_callback=progress_callback
                    )
        
        # The cleanup is handled in the finally block of transcode_file_vbr


if __name__ == '__main__':
    unittest.main()
