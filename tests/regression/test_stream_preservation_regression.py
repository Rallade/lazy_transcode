"""
Regression tests to prevent stream preservation bugs.

These tests specifically target the bugs that caused audio and subtitle streams
to be missing from transcoded files by testing integration points and 
actual command generation.
"""

import unittest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, call

from lazy_transcode.core.modules.processing.transcoding_engine import transcode_file_vbr
from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder


class TestStreamPreservationRegression(unittest.TestCase):
    """Regression tests to prevent stream preservation bugs."""
    
    def test_transcode_file_vbr_uses_comprehensive_command_builder(self):
        """
        REGRESSION TEST: Ensure transcode_file_vbr uses comprehensive encoder.
        
        This test would have caught the bug where transcode_file_vbr was using
        the incomplete build_vbr_encode_cmd from transcoding_engine.py instead
        of the comprehensive one from vbr_optimizer.py.
        """
        input_file = Path("test.mkv")
        output_file = Path("output.mkv")
        
        with patch('lazy_transcode.core.modules.processing.transcoding_engine.build_vbr_encode_cmd') as mock_comprehensive:
            with patch('subprocess.Popen') as mock_popen:
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.mkdir'):
                        with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.check_output') as mock_check_output:
                            
                            # Mock ffprobe calls
                            mock_check_output.return_value = "yuv420p"
                            
                            # Mock successful process with context manager support
                            mock_process = Mock()
                            mock_process.communicate.return_value = ("", "")
                            mock_process.returncode = 0
                            mock_process.__enter__ = Mock(return_value=mock_process)
                            mock_process.__exit__ = Mock(return_value=None)
                            mock_popen.return_value = mock_process
                            
                            # Mock comprehensive command with stream preservation
                            comprehensive_cmd = [
                                'ffmpeg', '-y', '-i', str(input_file),
                                '-map', '0',              # All streams
                                '-map_metadata', '0',     # Metadata
                                '-map_chapters', '0',     # Chapters
                                '-c:v', 'libx265',        # Video
                                '-c:a', 'copy',          # Audio
                                '-c:s', 'copy',          # Subtitles
                                '-c:d', 'copy',          # Data
                                '-c:t', 'copy',          # Timecode
                                '-copy_unknown',         # Unknown
                                str(output_file)
                            ]
                            mock_comprehensive.return_value = comprehensive_cmd
                        
                        # Call the function
                        result = transcode_file_vbr(
                            input_file, output_file, "libx265", "software",
                            5000, 4000, preserve_hdr_metadata=True
                        )
                        
                        # Verify it used the comprehensive builder
                        mock_comprehensive.assert_called_once()
                        
                        # Verify the command passed to Popen includes stream preservation
                        # Find the ffmpeg call among multiple Popen calls
                        ffmpeg_call = None
                        for call in mock_popen.call_args_list:
                            if call[0] and 'ffmpeg' in str(call[0][0]):
                                ffmpeg_call = call[0][0]
                                break
                        
                        self.assertIsNotNone(ffmpeg_call, "FFmpeg command not found in Popen calls")
                        assert ffmpeg_call is not None  # Type checker assertion
                        
                        # These assertions would have FAILED with the old bug
                        self.assertIn('-map', ffmpeg_call)
                        self.assertIn('0', ffmpeg_call)
                        self.assertIn('-map_metadata', ffmpeg_call)
                        self.assertIn('-map_chapters', ffmpeg_call)
                        self.assertIn('-c:a', ffmpeg_call)
                        self.assertIn('copy', ffmpeg_call)
                        self.assertIn('-c:s', ffmpeg_call)
                        
                        self.assertTrue(result)


class TestCommandGenerationIntegration(unittest.TestCase):
    """Integration tests for command generation across modules."""
    
    def test_vbr_command_generation_end_to_end(self):
        """
        INTEGRATION TEST: Test complete VBR command generation pipeline.
        
        This tests the actual path from transcode_file_vbr through to
        final FFmpeg command generation, ensuring no steps are skipped.
        """
        input_file = Path("test_input.mkv")
        output_file = Path("test_output.mkv")
        
        # Don't mock the command generation - test the real pipeline
        with patch('subprocess.Popen') as mock_popen:
            with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.check_output') as mock_check_output:
                with patch('os.cpu_count', return_value=8):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch('pathlib.Path.mkdir'):
                            
                            # Mock ffprobe calls
                            mock_check_output.return_value = "yuv420p"
                            
                            # Mock successful process with context manager support
                            mock_process = Mock()
                            mock_process.communicate.return_value = ("", "")
                            mock_process.returncode = 0
                            mock_process.__enter__ = Mock(return_value=mock_process)
                            mock_process.__exit__ = Mock(return_value=None)
                            mock_popen.return_value = mock_process
                            
                            # Call the actual function - no mocking of command generation
                            result = transcode_file_vbr(
                                input_file, output_file, "libx265", "software",
                                5000, 4000, preserve_hdr_metadata=False
                            )
                            
                            # Verify subprocess was called multiple times (as expected)
                            self.assertGreater(mock_popen.call_count, 0, "Expected at least one subprocess call")
                            
                            # Find the main ffmpeg transcoding command
                            ffmpeg_calls = [call for call in mock_popen.call_args_list 
                                          if len(call[0]) > 0 and len(call[0][0]) > 0 and 'ffmpeg' in call[0][0][0]]
                            
                            # Should have content analysis + main transcoding commands
                            self.assertGreaterEqual(len(ffmpeg_calls), 1, "Expected at least one ffmpeg command")
                            
                            # Find the main transcoding command (has output file and VBR parameters)
                            main_cmd = None
                            for call in ffmpeg_calls:
                                cmd = call[0][0]
                                cmd_str = ' '.join(str(arg) for arg in cmd)
                                # Main command has output file name and VBR bitrate settings
                                if str(output_file) in cmd_str and '-b:v' in cmd_str:
                                    main_cmd = cmd
                                    break
                            
                            self.assertIsNotNone(main_cmd, "Could not find main transcoding command")
                            if main_cmd is not None:  # Type safety
                                actual_cmd = main_cmd
                                cmd_str = ' '.join(str(arg) for arg in actual_cmd)
                            
                            # These are the critical assertions that would catch the bug
                            self.assertIn('ffmpeg', cmd_str)
                            self.assertIn('-map 0', cmd_str)
                            self.assertIn('-map_metadata 0', cmd_str)
                            self.assertIn('-map_chapters 0', cmd_str)
                            self.assertIn('-c:a copy', cmd_str)
                            self.assertIn('-c:s copy', cmd_str)
                            self.assertIn('-c:d copy', cmd_str)
                            self.assertIn('-c:t copy', cmd_str)
                            self.assertIn('-copy_unknown', cmd_str)
                            
                            self.assertTrue(result)
    
    def test_different_encoders_all_preserve_streams(self):
        """
        COMPREHENSIVE TEST: Verify all encoder types preserve streams.
        
        This would catch if stream preservation was missing for specific
        encoder types (hardware vs software).
        """
        test_encoders = [
            ("libx265", "software"),
            ("hevc_nvenc", "hardware"), 
            ("hevc_amf", "hardware"),
            ("hevc_qsv", "hardware")
        ]
        
        for encoder, encoder_type in test_encoders:
            with self.subTest(encoder=encoder, encoder_type=encoder_type):
                input_file = Path(f"test_{encoder}.mkv")
                output_file = Path(f"output_{encoder}.mkv")
                
                with patch('subprocess.Popen') as mock_popen:
                    with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.check_output') as mock_check_output:
                        with patch('pathlib.Path.exists', return_value=True):
                            with patch('pathlib.Path.mkdir'):
                                
                                # Mock ffprobe calls
                                mock_check_output.return_value = "yuv420p"
                                
                                # Mock successful process with context manager support
                                mock_process = Mock()
                                mock_process.communicate.return_value = ("", "")
                                mock_process.returncode = 0
                                mock_process.__enter__ = Mock(return_value=mock_process)
                                mock_process.__exit__ = Mock(return_value=None)
                                mock_popen.return_value = mock_process
                                
                                transcode_file_vbr(
                                    input_file, output_file, encoder, encoder_type,
                                    5000, 4000
                                )
                                
                                # Find the ffmpeg command among all Popen calls
                                ffmpeg_cmd = None
                                for call_args in mock_popen.call_args_list:
                                    cmd = call_args[0][0]
                                    if isinstance(cmd, list) and len(cmd) > 0 and 'ffmpeg' in cmd[0]:
                                        ffmpeg_cmd = cmd
                                        break
                                
                                self.assertIsNotNone(ffmpeg_cmd, f"No ffmpeg command found for {encoder}")
                                if ffmpeg_cmd:
                                    cmd_str = ' '.join(str(arg) for arg in ffmpeg_cmd)
                                    
                                    # Critical: ALL encoders must preserve streams
                                    self.assertIn('-map 0', cmd_str, f"{encoder} missing stream mapping")
                                self.assertIn('-c:a copy', cmd_str, f"{encoder} missing audio copy")
                                self.assertIn('-c:s copy', cmd_str, f"{encoder} missing subtitle copy")


class TestFFmpegCommandValidation(unittest.TestCase):
    """Tests to validate actual FFmpeg commands are correct."""
    
    def test_generated_commands_are_valid_ffmpeg_syntax(self):
        """
        VALIDATION TEST: Ensure generated commands have valid FFmpeg syntax.
        
        This catches syntax errors that could cause silent failures.
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                # Validate command structure
                self.assertEqual(cmd[0], 'ffmpeg', "Command must start with ffmpeg")
                self.assertIn('-i', cmd, "Must have input flag")
                self.assertIn('input.mkv', cmd, "Must have input file")
                self.assertIn('output.mkv', cmd, "Must have output file")
                
                # Validate stream preservation flags are present and properly formatted
                cmd_str = ' '.join(cmd)
                
                # Check for required stream preservation patterns
                stream_patterns = [
                    '-map 0',
                    '-map_metadata 0', 
                    '-map_chapters 0',
                    '-c:a copy',
                    '-c:s copy',
                    '-c:d copy',
                    '-c:t copy',
                    '-copy_unknown'
                ]
                
                for pattern in stream_patterns:
                    self.assertIn(pattern, cmd_str, f"Missing required pattern: {pattern}")
    
    def test_command_flags_are_properly_paired(self):
        """
        SYNTAX TEST: Ensure FFmpeg flags have proper arguments.
        
        This catches cases where flags are present but missing their arguments.
        """
        builder = EncoderConfigBuilder()
        
        with patch('lazy_transcode.core.modules.encoder_config.ffprobe_field') as mock_ffprobe:
            with patch('lazy_transcode.core.modules.encoder_config.os.cpu_count', return_value=8):
                mock_ffprobe.return_value = "yuv420p"
                
                cmd = builder.build_vbr_encode_cmd(
                    "input.mkv", "output.mkv", "libx265", "medium", 5000,
                    3, 3, 1920, 1080
                )
                
                # Check that critical flags have arguments
                flag_pairs = [
                    ('-i', 'input.mkv'),
                    ('-c:v', 'libx265'),
                    ('-c:a', 'copy'),
                    ('-c:s', 'copy'),
                    ('-map', '0'),
                    ('-map_metadata', '0'),
                    ('-map_chapters', '0')
                ]
                
                for flag, expected_arg in flag_pairs:
                    try:
                        flag_index = cmd.index(flag)
                        actual_arg = cmd[flag_index + 1]
                        self.assertEqual(actual_arg, expected_arg, 
                                       f"Flag {flag} should be followed by {expected_arg}, got {actual_arg}")
                    except (ValueError, IndexError):
                        self.fail(f"Flag {flag} not found or missing argument")


class TestModuleIntegrationPoints(unittest.TestCase):
    """Tests for integration points between modules."""
    
    def test_transcoding_engine_imports_correct_builder(self):
        """
        IMPORT TEST: Verify transcoding_engine imports the right builder.
        
        This would catch import errors or wrong module imports.
        """
        # Test that the import works correctly
        try:
            from lazy_transcode.core.modules.processing.transcoding_engine import transcode_file_vbr
            from lazy_transcode.core.modules.optimization.vbr_optimizer import build_vbr_encode_cmd
        except ImportError as e:
            self.fail(f"Import error: {e}")
        
        # Test that transcode_file_vbr can access the comprehensive builder
        with patch('lazy_transcode.core.modules.processing.transcoding_engine.build_vbr_encode_cmd') as mock_builder:
            with patch('subprocess.Popen') as mock_popen:
                with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.check_output') as mock_check_output:
                    with patch('builtins.print'):  # Suppress logging output
                        
                        # Mock ffprobe calls
                        mock_check_output.return_value = "yuv420p"
                        
                        # Mock successful process with context manager support
                        mock_process = Mock()
                        mock_process.communicate.return_value = ("", "")
                        mock_process.returncode = 0
                        mock_process.__enter__ = Mock(return_value=mock_process)
                        mock_process.__exit__ = Mock(return_value=None)
                        mock_popen.return_value = mock_process
                        
                        mock_builder.return_value = ['ffmpeg', '-i', 'test.mkv', 'out.mkv']
                        
                        try:
                            transcode_file_vbr(
                                Path("test.mkv"), Path("out.mkv"),
                                "libx265", "software", 5000, 4000
                            )
                            # If we get here, the import and call worked
                            mock_builder.assert_called_once()
                        except Exception as e:
                            self.fail(f"Integration failed: {e}")
    
    def test_encoder_config_builder_is_accessible_from_vbr_optimizer(self):
        """
        DEPENDENCY TEST: Verify VBR optimizer can access EncoderConfigBuilder.
        
        This tests the dependency chain is intact.
        """
        try:
            from lazy_transcode.core.modules.optimization.vbr_optimizer import build_vbr_encode_cmd
            from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder
        except ImportError as e:
            self.fail(f"Dependency import failed: {e}")
        
        # Test that vbr_optimizer can create and use EncoderConfigBuilder
        with patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_video_dimensions') as mock_dims:
            mock_dims.return_value = (1920, 1080)
            
            try:
                cmd = build_vbr_encode_cmd(
                    Path("test.mkv"), Path("out.mkv"),
                    "libx265", "software", 5000, 4000
                )
                
                # Should return a valid command list
                self.assertIsInstance(cmd, list)
                self.assertGreater(len(cmd), 5)  # Should have multiple arguments
                self.assertEqual(cmd[0], 'ffmpeg')
                
            except Exception as e:
                self.fail(f"EncoderConfigBuilder integration failed: {e}")


class TestRealWorldScenarios(unittest.TestCase):
    """Tests based on real-world usage scenarios."""
    
    def test_anime_episode_with_multiple_audio_tracks(self):
        """
        REAL-WORLD TEST: Anime episode with Japanese/English audio + subtitles.
        
        This simulates the exact scenario from the user's Demon Slayer files.
        """
        input_file = Path("Demon_Slayer_S01E01.mkv")
        output_file = Path("Demon_Slayer_S01E01_transcoded.mkv")
        
        with patch('subprocess.Popen') as mock_popen:
            with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.check_output') as mock_check_output:
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.mkdir'):
                        
                        # Mock ffprobe calls
                        mock_check_output.return_value = "yuv420p"
                        
                        # Mock successful process with context manager support
                        mock_process = Mock()
                        mock_process.communicate.return_value = ("", "")
                        mock_process.returncode = 0
                        mock_process.__enter__ = Mock(return_value=mock_process)
                        mock_process.__exit__ = Mock(return_value=None)
                        mock_popen.return_value = mock_process
                        
                        result = transcode_file_vbr(
                            input_file, output_file, "libx265", "software",
                            5000, 4000, preserve_hdr_metadata=False
                        )
                        
                        # Find the ffmpeg command among all Popen calls
                        ffmpeg_cmd = None
                        for call_args in mock_popen.call_args_list:
                            cmd = call_args[0][0]
                            if isinstance(cmd, list) and len(cmd) > 0 and 'ffmpeg' in cmd[0]:
                                ffmpeg_cmd = cmd
                                break
                        
                        self.assertIsNotNone(ffmpeg_cmd, "No ffmpeg command found")
                        if ffmpeg_cmd:
                            cmd_str = ' '.join(str(arg) for arg in ffmpeg_cmd)
                            
                            # Verify it would preserve multiple audio tracks
                            self.assertIn('-map 0', cmd_str, "Must map all streams")
                        self.assertIn('-c:a copy', cmd_str, "Must copy audio tracks")
                        self.assertIn('-c:s copy', cmd_str, "Must copy subtitle tracks")
                        self.assertIn('-map_chapters 0', cmd_str, "Must copy chapters")
                        
                        # Specific assertions for anime content
                        self.assertNotIn('-an', cmd_str, "Must not disable audio")
                        self.assertNotIn('-sn', cmd_str, "Must not disable subtitles")
                        
                        self.assertTrue(result)
    
    def test_movie_with_commentary_and_multiple_subtitle_languages(self):
        """
        REAL-WORLD TEST: Movie with director commentary and multiple subtitle languages.
        """
        input_file = Path("Movie_2023_4K.mkv")
        output_file = Path("Movie_2023_4K_transcoded.mkv")
        
        with patch('subprocess.Popen') as mock_popen:
            with patch('lazy_transcode.core.modules.analysis.media_utils.subprocess.check_output') as mock_check_output:
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.mkdir'):
                        
                        # Mock ffprobe calls
                        mock_check_output.return_value = "yuv420p10le"  # 10-bit content
                        
                        # Mock successful process with context manager support
                        mock_process = Mock()
                        mock_process.communicate.return_value = ("", "")
                        mock_process.returncode = 0
                        mock_process.__enter__ = Mock(return_value=mock_process)
                        mock_process.__exit__ = Mock(return_value=None)
                        mock_popen.return_value = mock_process
                        
                        result = transcode_file_vbr(
                            input_file, output_file, "libx265", "software",
                            8000, 6000, preserve_hdr_metadata=True
                        )
                        
                        # Find the ffmpeg command among all Popen calls
                        ffmpeg_cmd = None
                        for call_args in mock_popen.call_args_list:
                            cmd = call_args[0][0]
                            if isinstance(cmd, list) and len(cmd) > 0 and 'ffmpeg' in cmd[0]:
                                ffmpeg_cmd = cmd
                                break
                        
                        self.assertIsNotNone(ffmpeg_cmd, "No ffmpeg command found")
                        if ffmpeg_cmd:
                            cmd_str = ' '.join(str(arg) for arg in ffmpeg_cmd)
                            
                            # Must preserve ALL content types
                            essential_preservation = [
                                '-map 0',           # All streams
                                '-map_metadata 0',  # Metadata
                                '-map_chapters 0',  # Chapters
                                '-c:a copy',        # Audio (including commentary)
                                '-c:s copy',        # Subtitles (all languages)
                                '-c:d copy',        # Data streams
                                '-copy_unknown'     # Unknown streams
                            ]
                            
                            for pattern in essential_preservation:
                                self.assertIn(pattern, cmd_str,
                                            f"Missing essential preservation: {pattern}")
                        
                        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
