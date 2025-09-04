"""
Unit tests for media_utils module.

Tests core media functionality including FFprobe operations,
video metadata extraction, codec detection, and VMAF computation.
"""

import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.analysis.media_utils import (
    ffprobe_field, get_duration_sec, get_video_codec, 
    should_skip_codec, detect_hevc_encoder, compute_vmaf_score
)


class TestMediaUtils(unittest.TestCase):
    """Test media utilities functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_file = Path("test_video.mkv")
    
    @patch('subprocess.check_output')
    def test_ffprobe_field_success(self, mock_check_output):
        """Test successful ffprobe field extraction."""
        mock_check_output.return_value = "1920\n"
        
        result = ffprobe_field(self.test_file, "width")
        
        self.assertEqual(result, "1920")
        mock_check_output.assert_called_once()
        args = mock_check_output.call_args[0][0]
        self.assertIn("ffprobe", args)
        self.assertIn("width", ' '.join(args))
    
    @patch('subprocess.check_output')
    def test_ffprobe_field_unknown_value(self, mock_check_output):
        """Test ffprobe field with unknown value."""
        mock_check_output.return_value = "unknown\n"
        
        result = ffprobe_field(self.test_file, "codec_name")
        
        self.assertIsNone(result)
    
    @patch('subprocess.check_output')
    def test_ffprobe_field_error(self, mock_check_output):
        """Test ffprobe field with subprocess error."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
        
        result = ffprobe_field(self.test_file, "width")
        
        self.assertIsNone(result)
    
    @patch('subprocess.check_output')
    def test_get_duration_sec_success(self, mock_check_output):
        """Test successful duration extraction."""
        mock_check_output.return_value = "3600.5\n"
        
        result = get_duration_sec(self.test_file)
        
        self.assertEqual(result, 3600.5)
    
    @patch('subprocess.check_output')
    def test_get_duration_sec_error(self, mock_check_output):
        """Test duration extraction with error."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
        
        result = get_duration_sec(self.test_file)
        
        self.assertEqual(result, 0.0)
    
    @patch('lazy_transcode.core.modules.analysis.media_utils.run_command')
    def test_get_video_codec_h264(self, mock_run_command):
        """Test video codec detection for H.264."""
        mock_result = type('MockResult', (), {
            'returncode': 0,
            'stdout': 'h264\n'
        })()
        mock_run_command.return_value = mock_result

        result = get_video_codec(self.test_file)

        self.assertEqual(result, "h264")
    
    @patch('lazy_transcode.core.modules.analysis.media_utils.run_command')
    def test_get_video_codec_hevc(self, mock_run_command):
        """Test video codec detection for HEVC."""
        mock_result = type('MockResult', (), {
            'returncode': 0,
            'stdout': 'hevc\n'
        })()
        mock_run_command.return_value = mock_result
        
        result = get_video_codec(self.test_file)
        
        self.assertEqual(result, "hevc")
    
    def test_should_skip_codec_hevc(self):
        """Test codec skip detection for HEVC."""
        self.assertTrue(should_skip_codec("hevc"))
        self.assertTrue(should_skip_codec("h265"))
        self.assertTrue(should_skip_codec("HEVC"))  # Case insensitive
    
    def test_should_skip_codec_av1(self):
        """Test codec skip detection for AV1."""
        self.assertTrue(should_skip_codec("av1"))
        self.assertTrue(should_skip_codec("AV1"))
    
    def test_should_skip_codec_vp9(self):
        """Test codec skip detection for VP9."""
        self.assertTrue(should_skip_codec("vp9"))
    
    def test_should_skip_codec_h264(self):
        """Test codec skip detection for H.264 (should not skip)."""
        self.assertFalse(should_skip_codec("h264"))
        self.assertFalse(should_skip_codec("H264"))
    
    def test_should_skip_codec_none(self):
        """Test codec skip detection with None input."""
        self.assertFalse(should_skip_codec(None))
    
    @patch('subprocess.run')
    def test_detect_hevc_encoder_amd(self, mock_run):
        """Test HEVC encoder detection preferring AMD."""
        mock_result = Mock()
        mock_result.stdout = "VEA hevc_amf AMD AMF HEVC encoder"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        encoder, encoder_type = detect_hevc_encoder()
        
        self.assertEqual(encoder, "hevc_amf")
        self.assertEqual(encoder_type, "hardware")
    
    @patch('subprocess.run')
    def test_detect_hevc_encoder_nvidia(self, mock_run):
        """Test HEVC encoder detection with NVIDIA."""
        mock_result = Mock()
        mock_result.stdout = "V..... hevc_nvenc NVIDIA NVENC HEVC encoder"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        encoder, encoder_type = detect_hevc_encoder()
        
        self.assertEqual(encoder, "hevc_nvenc")
        self.assertEqual(encoder_type, "hardware")
    
    @patch('subprocess.run')
    def test_detect_hevc_encoder_intel(self, mock_run):
        """Test HEVC encoder detection with Intel QSV."""
        mock_result = Mock()
        mock_result.stdout = "V..... hevc_qsv Intel QuickSync HEVC encoder"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        encoder, encoder_type = detect_hevc_encoder()
        
        self.assertEqual(encoder, "hevc_qsv")
        self.assertEqual(encoder_type, "hardware")
    
    @patch('subprocess.run')
    def test_detect_hevc_encoder_software_fallback(self, mock_run):
        """Test HEVC encoder detection falling back to software."""
        mock_result = Mock()
        mock_result.stdout = "V..... libx265 x265 HEVC encoder"
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        encoder, encoder_type = detect_hevc_encoder()
        
        self.assertEqual(encoder, "libx265")
        self.assertEqual(encoder_type, "software")
    
    @patch('subprocess.run')
    def test_detect_hevc_encoder_error(self, mock_run):
        """Test HEVC encoder detection with subprocess error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ffmpeg')
        
        encoder, encoder_type = detect_hevc_encoder()
        
        self.assertEqual(encoder, "libx265")
        self.assertEqual(encoder_type, "software")


class TestVMAFComputation(unittest.TestCase):
    """Test VMAF computation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.ref_file = Path("reference.mkv")
        self.dist_file = Path("distorted.mkv")
    
    @patch('subprocess.run')
    def test_compute_vmaf_score_success(self, mock_run):
        """Test successful VMAF computation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
        [Parsed_vmaf_0 @ 0x123] VMAF score: 95.23
        """
        mock_run.return_value = mock_result
        
        score = compute_vmaf_score(self.ref_file, self.dist_file)
        
        self.assertEqual(score, 95.23)
    
    @patch('subprocess.run')
    def test_compute_vmaf_score_parse_error(self, mock_run):
        """Test VMAF computation with parsing error."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "No VMAF score found"
        mock_run.return_value = mock_result
        
        score = compute_vmaf_score(self.ref_file, self.dist_file)
        
        self.assertIsNone(score)
    
    @patch('subprocess.run')
    def test_compute_vmaf_score_ffmpeg_error(self, mock_run):
        """Test VMAF computation with FFmpeg error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "FFmpeg error"
        mock_run.return_value = mock_result
        
        score = compute_vmaf_score(self.ref_file, self.dist_file)
        
        self.assertIsNone(score)
    
    @patch('subprocess.run')
    def test_compute_vmaf_score_with_threads(self, mock_run):
        """Test VMAF computation with thread specification."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "[Parsed_vmaf_0 @ 0x123] VMAF score: 92.45"
        mock_run.return_value = mock_result
        
        score = compute_vmaf_score(self.ref_file, self.dist_file, n_threads=8)
        
        self.assertEqual(score, 92.45)
        # Verify threads were passed in the command
        call_args = mock_run.call_args[0][0]
        self.assertIn("n_threads=8", ' '.join(call_args))


class TestMediaUtilsCaching(unittest.TestCase):
    """Test caching behavior of media utilities."""
    
    @patch('subprocess.check_output')
    def test_ffprobe_field_caching(self, mock_check_output):
        """Test that ffprobe_field results are cached."""
        mock_check_output.return_value = "1920\n"
        test_file = Path("cached_test.mkv")
        
        # First call
        result1 = ffprobe_field(test_file, "width")
        # Second call with same parameters
        result2 = ffprobe_field(test_file, "width")
        
        self.assertEqual(result1, result2)
        # Should only be called once due to caching
        self.assertEqual(mock_check_output.call_count, 1)
    
    @patch('subprocess.check_output')
    def test_get_duration_sec_caching(self, mock_check_output):
        """Test that duration results are cached."""
        mock_check_output.return_value = "3600.5\n"
        test_file = Path("cached_duration.mkv")
        
        # First call
        duration1 = get_duration_sec(test_file)
        # Second call with same file
        duration2 = get_duration_sec(test_file)
        
        self.assertEqual(duration1, duration2)
        # Should only be called once due to caching
        self.assertEqual(mock_check_output.call_count, 1)


if __name__ == '__main__':
    unittest.main()
