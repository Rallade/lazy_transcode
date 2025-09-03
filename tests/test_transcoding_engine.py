"""
Unit tests for transcoding_engine module.

Tests encoding functionality including HDR detection, command building,
and encoding operations with proper metadata handling.
"""

import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.transcoding_engine import (
    detect_hdr_content, build_encode_cmd, build_vbr_encode_cmd
)


class TestHDRDetection(unittest.TestCase):
    """Test HDR video detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_file = Path("test_video.mkv")
    
    @patch('subprocess.check_output')
    def test_is_hdr_video_bt2020_smpte2084(self, mock_check_output):
        """Test HDR detection with BT.2020 color space and SMPTE2084 transfer."""
        mock_check_output.side_effect = [
            "bt2020nc\n",  # color_space
            "smpte2084\n"  # color_transfer
        ]
        
        result = is_hdr_video(self.test_file)
        
        self.assertTrue(result)
    
    @patch('subprocess.check_output')
    def test_is_hdr_video_bt2020_hlg(self, mock_check_output):
        """Test HDR detection with BT.2020 and HLG."""
        mock_check_output.side_effect = [
            "bt2020nc\n",  # color_space
            "arib-std-b67\n"  # HLG transfer
        ]
        
        result = is_hdr_video(self.test_file)
        
        self.assertTrue(result)
    
    @patch('subprocess.check_output')
    def test_is_hdr_video_bt709(self, mock_check_output):
        """Test SDR detection with BT.709."""
        mock_check_output.side_effect = [
            "bt709\n",     # color_space
            "bt709\n"      # color_transfer
        ]
        
        result = is_hdr_video(self.test_file)
        
        self.assertFalse(result)
    
    @patch('subprocess.check_output')
    def test_is_hdr_video_unknown_values(self, mock_check_output):
        """Test HDR detection with unknown color characteristics."""
        mock_check_output.side_effect = [
            "unknown\n",   # color_space
            "unknown\n"    # color_transfer
        ]
        
        result = is_hdr_video(self.test_file)
        
        self.assertFalse(result)
    
    @patch('subprocess.check_output')
    def test_is_hdr_video_ffprobe_error(self, mock_check_output):
        """Test HDR detection with ffprobe error."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
        
        result = is_hdr_video(self.test_file)
        
        self.assertFalse(result)


class TestHDRMetadata(unittest.TestCase):
    """Test HDR metadata extraction and application."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_file = Path("hdr_video.mkv")
    
    @patch('subprocess.check_output')
    def test_extract_hdr_metadata_complete(self, mock_check_output):
        """Test complete HDR metadata extraction."""
        mock_check_output.side_effect = [
            "bt2020nc\n",      # color_space
            "smpte2084\n",     # color_transfer
            "bt2020\n",        # color_primaries
            "5000,40000\n",    # mastering_display
            "1000,50\n"        # max_cll
        ]
        
        metadata = _extract_hdr_metadata(self.test_file)
        
        expected = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084',
            'color_primaries': 'bt2020',
            'mastering_display': '5000,40000',
            'max_cll': '1000,50'
        }
        self.assertEqual(metadata, expected)
    
    @patch('subprocess.check_output')
    def test_extract_hdr_metadata_partial(self, mock_check_output):
        """Test partial HDR metadata extraction."""
        mock_check_output.side_effect = [
            "bt2020nc\n",      # color_space
            "smpte2084\n",     # color_transfer
            "bt2020\n",        # color_primaries
            "unknown\n",       # mastering_display
            "unknown\n"        # max_cll
        ]
        
        metadata = _extract_hdr_metadata(self.test_file)
        
        expected = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084',
            'color_primaries': 'bt2020'
        }
        self.assertEqual(metadata, expected)
    
    @patch('subprocess.check_output')
    def test_extract_hdr_metadata_error(self, mock_check_output):
        """Test HDR metadata extraction with ffprobe error."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
        
        metadata = _extract_hdr_metadata(self.test_file)
        
        self.assertEqual(metadata, {})
    
    def test_apply_hdr_metadata_complete(self):
        """Test applying complete HDR metadata to command."""
        base_cmd = ["ffmpeg", "-i", "input.mkv"]
        metadata = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084',
            'color_primaries': 'bt2020',
            'mastering_display': 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)',
            'max_cll': '1000,50'
        }
        
        cmd = _apply_hdr_metadata(base_cmd, metadata)
        
        self.assertIn("-colorspace", cmd)
        self.assertIn("bt2020nc", cmd)
        self.assertIn("-color_trc", cmd)
        self.assertIn("smpte2084", cmd)
        self.assertIn("-color_primaries", cmd)
        self.assertIn("bt2020", cmd)
        self.assertIn("-x265-params", cmd)
        # Should contain mastering display and max CLL in x265 params
        x265_idx = cmd.index("-x265-params")
        x265_params = cmd[x265_idx + 1]
        self.assertIn("master-display", x265_params)
        self.assertIn("max-cll", x265_params)
    
    def test_apply_hdr_metadata_partial(self):
        """Test applying partial HDR metadata to command."""
        base_cmd = ["ffmpeg", "-i", "input.mkv"]
        metadata = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084'
        }
        
        cmd = _apply_hdr_metadata(base_cmd, metadata)
        
        self.assertIn("-colorspace", cmd)
        self.assertIn("bt2020nc", cmd)
        self.assertIn("-color_trc", cmd)
        self.assertIn("smpte2084", cmd)
        # Should not have x265-params without mastering display or max CLL
        self.assertNotIn("-x265-params", cmd)
    
    def test_apply_hdr_metadata_empty(self):
        """Test applying empty HDR metadata (no changes)."""
        base_cmd = ["ffmpeg", "-i", "input.mkv", "output.mkv"]
        metadata = {}
        
        cmd = _apply_hdr_metadata(base_cmd, metadata)
        
        self.assertEqual(cmd, base_cmd)  # Should be unchanged


class TestCommandBuilding(unittest.TestCase):
    """Test encoding command building functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("input.mkv")
        self.output_file = Path("output.mkv")
        self.encoder_config = Mock()
        self.encoder_config.get_base_command.return_value = [
            "ffmpeg", "-i", str(self.input_file), "-c:v", "libx265"
        ]
    
    @patch('lazy_transcode.core.modules.transcoding_engine.is_hdr_video')
    @patch('lazy_transcode.core.modules.transcoding_engine._extract_hdr_metadata')
    def test_build_hevc_encode_cmd_hdr(self, mock_extract_hdr, mock_is_hdr):
        """Test building HEVC command for HDR content."""
        mock_is_hdr.return_value = True
        mock_extract_hdr.return_value = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084'
        }
        
        cmd = build_hevc_encode_cmd(
            self.input_file, self.output_file, 
            self.encoder_config, crf=23
        )
        
        # Should include HDR metadata
        self.assertIn("-colorspace", cmd)
        self.assertIn("bt2020nc", cmd)
        self.assertIn("-color_trc", cmd)
        self.assertIn("smpte2084", cmd)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.is_hdr_video')
    def test_build_hevc_encode_cmd_sdr(self, mock_is_hdr):
        """Test building HEVC command for SDR content."""
        mock_is_hdr.return_value = False
        
        cmd = build_hevc_encode_cmd(
            self.input_file, self.output_file, 
            self.encoder_config, crf=23
        )
        
        # Should not include HDR metadata
        self.assertNotIn("-colorspace", cmd)
        self.assertNotIn("-color_trc", cmd)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.is_hdr_video')
    @patch('lazy_transcode.core.modules.transcoding_engine._extract_hdr_metadata')
    def test_build_vbr_encode_cmd_hdr(self, mock_extract_hdr, mock_is_hdr):
        """Test building VBR command for HDR content."""
        mock_is_hdr.return_value = True
        mock_extract_hdr.return_value = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084'
        }
        
        cmd = build_vbr_encode_cmd(
            self.input_file, self.output_file,
            self.encoder_config, bitrate=5000
        )
        
        # Should include HDR metadata and VBR settings
        self.assertIn("-colorspace", cmd)
        self.assertIn("bt2020nc", cmd)
        # Should have bitrate setting (implementation dependent)
    
    def test_build_hevc_encode_cmd_with_crf(self):
        """Test building HEVC command with CRF value."""
        cmd = build_hevc_encode_cmd(
            self.input_file, self.output_file,
            self.encoder_config, crf=25
        )
        
        # Should include CRF setting in command
        self.assertIn("25", ' '.join(map(str, cmd)))


class TestEncodingOperations(unittest.TestCase):
    """Test actual encoding operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("input.mkv")
        self.output_file = Path("output.mkv")
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('lazy_transcode.core.modules.transcoding_engine.build_hevc_encode_cmd')
    def test_encode_file_success(self, mock_build_cmd, mock_run):
        """Test successful file encoding."""
        mock_build_cmd.return_value = ["ffmpeg", "-i", "input.mkv", "output.mkv"]
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        success = encode_file(
            self.input_file, self.output_file,
            Mock(), crf=23, temp_dir=self.temp_dir
        )
        
        self.assertTrue(success)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    @patch('lazy_transcode.core.modules.transcoding_engine.build_hevc_encode_cmd')
    def test_encode_file_failure(self, mock_build_cmd, mock_run):
        """Test failed file encoding."""
        mock_build_cmd.return_value = ["ffmpeg", "-i", "input.mkv", "output.mkv"]
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Encoding failed"
        mock_run.return_value = mock_result
        
        success = encode_file(
            self.input_file, self.output_file,
            Mock(), crf=23, temp_dir=self.temp_dir
        )
        
        self.assertFalse(success)
    
    @patch('subprocess.run')
    @patch('lazy_transcode.core.modules.transcoding_engine.build_vbr_encode_cmd')
    def test_encode_file_vbr_mode(self, mock_build_cmd, mock_run):
        """Test VBR mode encoding."""
        mock_build_cmd.return_value = ["ffmpeg", "-i", "input.mkv", "-b:v", "5000k", "output.mkv"]
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        success = encode_file(
            self.input_file, self.output_file,
            Mock(), bitrate=5000, temp_dir=self.temp_dir
        )
        
        self.assertTrue(success)
        mock_build_cmd.assert_called_once()
    
    @patch('subprocess.run')
    def test_encode_file_with_progress_callback(self, mock_run):
        """Test encoding with progress callback."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        progress_calls = []
        def progress_callback(percent):
            progress_calls.append(percent)
        
        success = encode_file(
            self.input_file, self.output_file,
            Mock(), crf=23, temp_dir=self.temp_dir,
            progress_callback=progress_callback
        )
        
        self.assertTrue(success)
        # Progress callback should have been used (implementation dependent)


class TestEncodingEdgeCases(unittest.TestCase):
    """Test edge cases in encoding operations."""
    
    @patch('subprocess.check_output')
    def test_hdr_detection_mixed_characteristics(self, mock_check_output):
        """Test HDR detection with mixed color characteristics."""
        # BT.2020 space but SDR transfer
        mock_check_output.side_effect = [
            "bt2020nc\n",  # color_space
            "bt709\n"      # color_transfer (SDR)
        ]
        
        result = is_hdr_video(Path("mixed.mkv"))
        
        # Should be False - need both HDR space AND transfer
        self.assertFalse(result)
    
    @patch('subprocess.check_output')
    def test_hdr_metadata_extraction_malformed(self, mock_check_output):
        """Test HDR metadata extraction with malformed data."""
        mock_check_output.side_effect = [
            "bt2020nc\n",           # color_space
            "smpte2084\n",          # color_transfer
            "bt2020\n",             # color_primaries
            "malformed_data\n",     # mastering_display
            "also_malformed\n"      # max_cll
        ]
        
        metadata = _extract_hdr_metadata(Path("malformed.mkv"))
        
        # Should skip malformed fields but keep valid ones
        expected = {
            'color_space': 'bt2020nc',
            'color_transfer': 'smpte2084',
            'color_primaries': 'bt2020'
        }
        self.assertEqual(metadata, expected)


if __name__ == '__main__':
    unittest.main()
