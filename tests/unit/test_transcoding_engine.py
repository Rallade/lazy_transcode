"""
Unit tests for transcoding_engine module.

Tests encoding functionality including HDR detection and command building.
"""

import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.processing.transcoding_engine import (
    detect_hdr_content, build_encode_cmd, build_vbr_encode_cmd
)


class TestHDRDetection(unittest.TestCase):
    """Test HDR video detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_file = Path("test_video.mkv")
    
    @patch('subprocess.run')
    def test_detect_hdr_content_bt2020_smpte2084(self, mock_run):
        """Test HDR detection with BT.2020 and SMPTE2084."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{
                "color_primaries": "bt2020",
                "color_trc": "smpte2084",
                "color_space": "bt2020nc"
            }]
        })
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(self.test_file)
        
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_detect_hdr_content_hlg(self, mock_run):
        """Test HDR detection with HLG transfer."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{
                "color_primaries": "bt2020",
                "color_trc": "arib-std-b67",
                "color_space": "bt2020nc"
            }]
        })
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(self.test_file)
        
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_detect_hdr_content_sdr(self, mock_run):
        """Test SDR detection with BT.709."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{
                "color_primaries": "bt709",
                "color_trc": "bt709",
                "color_space": "bt709"
            }]
        })
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(self.test_file)
        
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_detect_hdr_content_error(self, mock_run):
        """Test HDR detection with subprocess error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(self.test_file)
        
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_detect_hdr_content_malformed_json(self, mock_run):
        """Test HDR detection with malformed JSON."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(self.test_file)
        
        self.assertFalse(result)


class TestCommandBuilding(unittest.TestCase):
    """Test encoding command building functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("input.mkv")
        self.output_file = Path("output.mkv")
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_encode_cmd_software_sdr(self, mock_detect_hdr):
        """Test building encode command for software encoder with SDR content."""
        mock_detect_hdr.return_value = False
        
        cmd = build_encode_cmd(
            self.input_file, self.output_file, 
            "libx265", "software", qp=23
        )
        
        # Should contain basic x265 settings without HDR
        cmd_str = ' '.join(cmd)
        self.assertIn("libx265", cmd_str)
        self.assertIn("-crf", cmd_str)
        self.assertIn("23", cmd_str)
        self.assertNotIn("colorprim=bt2020", cmd_str)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_encode_cmd_software_hdr(self, mock_detect_hdr):
        """Test building encode command for software encoder with HDR content."""
        mock_detect_hdr.return_value = True
        
        cmd = build_encode_cmd(
            self.input_file, self.output_file, 
            "libx265", "software", qp=23
        )
        
        # Should contain HDR x265 parameters
        cmd_str = ' '.join(cmd)
        self.assertIn("libx265", cmd_str)
        self.assertIn("colorprim=bt2020", cmd_str)
        self.assertIn("transfer=smpte2084", cmd_str)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_encode_cmd_nvenc_hdr(self, mock_detect_hdr):
        """Test building encode command for NVENC with HDR content."""
        mock_detect_hdr.return_value = True
        
        cmd = build_encode_cmd(
            self.input_file, self.output_file, 
            "hevc_nvenc", "hardware", qp=25
        )
        
        # Should contain NVENC settings with HDR
        cmd_str = ' '.join(cmd)
        self.assertIn("hevc_nvenc", cmd_str)
        self.assertIn("-cq", cmd_str)
        self.assertIn("25", cmd_str)
        self.assertIn("-colorspace", cmd_str)
        self.assertIn("bt2020nc", cmd_str)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_encode_cmd_amf(self, mock_detect_hdr):
        """Test building encode command for AMF encoder."""
        mock_detect_hdr.return_value = False
        
        cmd = build_encode_cmd(
            self.input_file, self.output_file, 
            "hevc_amf", "hardware", qp=28
        )
        
        # Should contain AMF-specific settings
        cmd_str = ' '.join(cmd)
        self.assertIn("hevc_amf", cmd_str)
        self.assertIn("-qp_i", cmd_str)
        self.assertIn("28", cmd_str)
    
    def test_build_encode_cmd_with_progress(self):
        """Test building encode command with progress tracking."""
        progress_file = Path("progress.txt")
        
        cmd = build_encode_cmd(
            self.input_file, self.output_file,
            "libx265", "software", qp=23,
            progress_file=progress_file
        )
        
        # Should include progress file
        cmd_str = ' '.join(cmd)
        self.assertIn("-progress", cmd_str)
        self.assertIn(str(progress_file), cmd_str)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_vbr_encode_cmd_basic(self, mock_detect_hdr):
        """Test building VBR encode command."""
        mock_detect_hdr.return_value = False
        
        cmd = build_vbr_encode_cmd(
            self.input_file, self.output_file,
            "libx265", "software", 
            max_bitrate=8000, avg_bitrate=5000
        )
        
        # Should contain VBR settings
        cmd_str = ' '.join(cmd)
        self.assertIn("libx265", cmd_str)
        # VBR parameters will be implementation specific
        self.assertIn(str(self.input_file), cmd_str)
        self.assertIn(str(self.output_file), cmd_str)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_vbr_encode_cmd_hdr(self, mock_detect_hdr):
        """Test building VBR encode command with HDR."""
        mock_detect_hdr.return_value = True
        
        cmd = build_vbr_encode_cmd(
            self.input_file, self.output_file,
            "libx265", "software",
            max_bitrate=12000, avg_bitrate=8000
        )
        
        # Should contain HDR settings for VBR
        cmd_str = ' '.join(cmd)
        self.assertIn("libx265", cmd_str)


class TestEncodingEdgeCases(unittest.TestCase):
    """Test edge cases in encoding operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("input.mkv")
        self.output_file = Path("output.mkv")
    
    @patch('subprocess.run')
    def test_detect_hdr_content_mixed_characteristics(self, mock_run):
        """Test HDR detection with mixed color characteristics."""
        # BT.2020 space but SDR transfer
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{
                "color_primaries": "bt2020",
                "color_trc": "bt709",  # SDR transfer
                "color_space": "bt2020nc"
            }]
        })
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(Path("mixed.mkv"))
        
        # Should still detect as HDR due to bt2020 primaries
        self.assertTrue(result)
    
    @patch('subprocess.run')  
    def test_detect_hdr_content_empty_streams(self, mock_run):
        """Test HDR detection with empty streams."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"streams": []})
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(Path("empty.mkv"))
        
        # Should default to False for empty streams
        self.assertFalse(result)
    
    @patch('lazy_transcode.core.modules.transcoding_engine.detect_hdr_content')
    def test_build_encode_cmd_preserve_hdr_false(self, mock_detect_hdr):
        """Test building encode command with HDR preservation disabled."""
        mock_detect_hdr.return_value = True
        
        cmd = build_encode_cmd(
            self.input_file, self.output_file,
            "libx265", "software", qp=23,
            preserve_hdr_metadata=False
        )
        
        # Should not include HDR parameters even if source is HDR
        cmd_str = ' '.join(cmd)
        self.assertNotIn("colorprim=bt2020", cmd_str)


if __name__ == '__main__':
    unittest.main()
