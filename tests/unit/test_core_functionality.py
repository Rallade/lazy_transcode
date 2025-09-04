"""
Unit tests for core modules functionality.

Tests the main functionality that can be tested before refactoring.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Test imports that we know work
try:
    from lazy_transcode.core.modules.system_utils import format_size, DEBUG, TEMP_FILES
    SYSTEM_UTILS_AVAILABLE = True
except ImportError:
    SYSTEM_UTILS_AVAILABLE = False

try:
    from lazy_transcode.core.modules.transcoding_engine import detect_hdr_content
    TRANSCODING_ENGINE_AVAILABLE = True
except ImportError:
    TRANSCODING_ENGINE_AVAILABLE = False


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality that should work."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)


@unittest.skipUnless(SYSTEM_UTILS_AVAILABLE, "system_utils not available")
class TestSystemUtils(TestBasicFunctionality):
    """Test system utility functions."""
    
    def test_format_size_bytes(self):
        """Test size formatting for bytes."""
        self.assertEqual(format_size(500), "500.0 B")
        self.assertEqual(format_size(1000), "1000.0 B")
    
    def test_format_size_kb(self):
        """Test size formatting for kilobytes."""
        result = format_size(1536)  # 1.5 KB
        self.assertIn("KB", result)
        self.assertIn("1.5", result)
    
    def test_format_size_mb(self):
        """Test size formatting for megabytes."""
        result = format_size(1572864)  # 1.5 MB
        self.assertIn("MB", result)
    
    def test_format_size_zero(self):
        """Test size formatting for zero."""
        self.assertEqual(format_size(0), "0.0 B")
    
    def test_debug_flag_type(self):
        """Test that DEBUG flag is boolean."""
        self.assertIsInstance(DEBUG, bool)
    
    def test_temp_files_type(self):
        """Test that TEMP_FILES is a set."""
        self.assertIsInstance(TEMP_FILES, set)


@unittest.skipUnless(TRANSCODING_ENGINE_AVAILABLE, "transcoding_engine not available") 
class TestTranscodingEngine(TestBasicFunctionality):
    """Test transcoding engine functions."""
    
    @patch('subprocess.run')
    def test_detect_hdr_content_hdr_video(self, mock_run):
        """Test HDR detection with HDR video."""
        import json
        
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
        
        result = detect_hdr_content(Path("test_hdr.mkv"))
        
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_detect_hdr_content_sdr_video(self, mock_run):
        """Test HDR detection with SDR video."""
        import json
        
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
        
        result = detect_hdr_content(Path("test_sdr.mkv"))
        
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_detect_hdr_content_error(self, mock_run):
        """Test HDR detection with error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        result = detect_hdr_content(Path("test_error.mkv"))
        
        # Should default to False on error
        self.assertFalse(result)


class TestMediaUtilsPatterns(unittest.TestCase):
    """Test patterns that will be refactored in media_utils."""
    
    def test_ffprobe_field_pattern(self):
        """Test ffprobe field extraction pattern."""
        # This tests the pattern we identified in duplicate code
        with patch('subprocess.check_output') as mock_output:
            mock_output.return_value = "1920\n"
            
            # Simulate the duplicate ffprobe_field pattern
            def mock_ffprobe_field(file_path, field_name):
                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                       "-show_entries", f"stream={field_name}", "-of", "csv=p=0",
                       str(file_path)]
                try:
                    result = mock_output(cmd)
                    return result.strip() if result.strip() != "N/A" else None
                except:
                    return None
            
            result = mock_ffprobe_field(Path("test.mkv"), "width")
            self.assertEqual(result, "1920")
    
    def test_video_dimensions_pattern(self):
        """Test video dimensions extraction pattern."""
        # This tests the pattern we identified for dimensions extraction
        with patch('subprocess.check_output') as mock_output:
            mock_output.side_effect = ["1920\n", "1080\n"]
            
            def get_video_dimensions(file_path):
                # Simulate the repeated pattern we found
                try:
                    width_result = mock_output(["ffprobe", "-v", "error", 
                                              "-select_streams", "v:0", "-show_entries",
                                              "stream=width", "-of", "csv=p=0", str(file_path)])
                    height_result = mock_output(["ffprobe", "-v", "error",
                                               "-select_streams", "v:0", "-show_entries", 
                                               "stream=height", "-of", "csv=p=0", str(file_path)])
                    
                    width = int(width_result.strip()) if width_result.strip().isdigit() else 0
                    height = int(height_result.strip()) if height_result.strip().isdigit() else 0
                    
                    return width, height
                except:
                    return 0, 0
            
            width, height = get_video_dimensions(Path("test.mkv"))
            self.assertEqual(width, 1920)
            self.assertEqual(height, 1080)
    
    def test_subprocess_execution_pattern(self):
        """Test subprocess execution pattern."""
        # This tests the subprocess.run pattern we found repeated
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "success"
            mock_run.return_value = mock_result
            
            def run_subprocess_command(cmd, capture=True):
                # Simulate the repeated subprocess pattern
                try:
                    if capture:
                        result = mock_run(cmd, capture_output=True, text=True, timeout=30)
                    else:
                        result = mock_run(cmd, timeout=30)
                    return result.returncode == 0, result.stdout if capture else ""
                except:
                    return False, ""
            
            success, output = run_subprocess_command(["echo", "test"], capture=True)
            self.assertTrue(success)
            self.assertEqual(output, "success")


class TestCodePatterns(unittest.TestCase):
    """Test code patterns that will be consolidated."""
    
    def test_temp_file_handling_pattern(self):
        """Test temporary file handling pattern."""
        # Test the temp file pattern we identified
        temp_files = []
        
        def create_temp_file(suffix=".mkv"):
            temp_file = Path(tempfile.mktemp(suffix=suffix))
            temp_files.append(temp_file)
            return temp_file
        
        def cleanup_temp_files():
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            temp_files.clear()
        
        # Test pattern
        temp1 = create_temp_file()
        temp2 = create_temp_file(".mp4")
        
        self.assertEqual(len(temp_files), 2)
        self.assertTrue(str(temp1).endswith(".mkv"))
        self.assertTrue(str(temp2).endswith(".mp4"))
        
        cleanup_temp_files()
        self.assertEqual(len(temp_files), 0)
    
    def test_vmaf_computation_pattern(self):
        """Test VMAF computation pattern."""
        # Test the VMAF pattern we identified
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "[Parsed_vmaf_0 @ 0x123] VMAF score: 92.5"
            mock_run.return_value = mock_result
            
            def compute_vmaf_pattern(ref_file, dist_file):
                # Simulate the VMAF computation pattern
                cmd = [
                    "ffmpeg", "-i", str(ref_file), "-i", str(dist_file),
                    "-lavfi", "vmaf", "-f", "null", "-"
                ]
                
                try:
                    result = mock_run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse VMAF score from output
                        for line in result.stdout.split('\n'):
                            if "VMAF score:" in line:
                                score_str = line.split("VMAF score:")[-1].strip()
                                return float(score_str)
                    return None
                except:
                    return None
            
            score = compute_vmaf_pattern(Path("ref.mkv"), Path("dist.mkv"))
            self.assertEqual(score, 92.5)


if __name__ == '__main__':
    unittest.main()
