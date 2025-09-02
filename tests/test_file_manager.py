"""Unit tests for FileManager module."""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from lazy_transcode.core.modules.file_manager import FileManager, FileDiscoveryResult, CodecCheckResult


class TestFileManager(unittest.TestCase):
    """Test FileManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.file_manager = FileManager(debug=False)
        
        # Create test files
        self.test_files = [
            self.temp_dir / "video1.mkv",
            self.temp_dir / "video2.mp4", 
            self.temp_dir / "._hidden.mkv",  # Hidden file
            self.temp_dir / ".DS_Store",     # Mac hidden file
            self.temp_dir / "sample.sample_clip.mkv",  # Sample artifact
            self.temp_dir / "test_qp25_sample.mkv"     # QP sample artifact
        ]
        
        # Create the files
        for f in self.test_files:
            f.touch()
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_discover_video_files(self):
        """Test video file discovery."""
        result = self.file_manager.discover_video_files(self.temp_dir, "mkv,mp4")
        
        self.assertIsInstance(result, FileDiscoveryResult)
        # Should find video files but filter out hidden and sample files
        expected_files = ["video1.mkv", "video2.mp4"]
        found_names = [f.name for f in result.files_to_transcode]
        
        for expected in expected_files:
            self.assertIn(expected, found_names)
        
        # Should filter out hidden and sample files
        self.assertNotIn("._hidden.mkv", found_names)
        self.assertNotIn(".DS_Store", found_names)
        self.assertNotIn("sample.sample_clip.mkv", found_names)
        self.assertNotIn("test_qp25_sample.mkv", found_names)
    
    def test_is_sample_or_artifact(self):
        """Test sample and artifact detection."""
        test_cases = [
            ("normal.mkv", False),
            ("video.sample_clip.mkv", True),
            ("test_sample.mkv", True),
            ("clip1.sample.mkv", True),
            ("qp25_sample.mkv", True),
            (".hidden.mkv", False),  # Hidden files handled separately
        ]
        
        for filename, expected in test_cases:
            test_path = Path(filename)
            result = self.file_manager._is_sample_or_artifact(test_path)
            self.assertEqual(result, expected, f"Failed for {filename}")
    
    @patch('subprocess.run')
    def test_check_video_codec(self, mock_subprocess):
        """Test video codec checking."""
        # Mock ffprobe output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "h264"
        mock_subprocess.return_value = mock_result
        
        test_file = self.temp_dir / "test.mkv"
        test_file.touch()
        
        result = self.file_manager.check_video_codec(test_file)
        
        self.assertIsInstance(result, CodecCheckResult)
        self.assertEqual(result.codec, "h264")
        self.assertFalse(result.should_skip)  # h264 should not be skipped
    
    @patch('subprocess.run')
    def test_check_video_codec_efficient(self, mock_subprocess):
        """Test codec checking with efficient codec."""
        # Mock ffprobe output for efficient codec
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hevc"
        mock_subprocess.return_value = mock_result
        
        test_file = self.temp_dir / "test.mkv"
        test_file.touch()
        
        result = self.file_manager.check_video_codec(test_file)
        
        self.assertEqual(result.codec, "hevc")
        self.assertTrue(result.should_skip)  # hevc should be skipped
    
    def test_should_skip_codec(self):
        """Test codec skip logic."""
        test_cases = [
            ("h264", False),
            ("hevc", True),
            ("h265", True),
            ("av1", True),
            ("mpeg2", False),
            ("", False),
            (None, False)
        ]
        
        for codec, expected in test_cases:
            result = self.file_manager._should_skip_codec(codec)
            self.assertEqual(result, expected, f"Failed for codec: {codec}")
    
    def test_temp_file_management(self):
        """Test temporary file management."""
        test_file = self.temp_dir / "original.mkv"
        test_file.touch()
        
        # Test temp file creation
        temp_path = self.file_manager.create_temp_file_path(test_file, "_test", ".mkv")
        expected_name = "original_test.mkv"
        self.assertEqual(temp_path.name, expected_name)
        
        # Test registration
        self.assertIn(str(temp_path), self.file_manager.temp_files)
        
        # Test unregistration
        self.file_manager.unregister_temp_file(temp_path)
        self.assertNotIn(str(temp_path), self.file_manager.temp_files)
    
    def test_validate_file_access(self):
        """Test file access validation."""
        # Test existing file
        test_file = self.temp_dir / "test.mkv"
        test_file.write_bytes(b"test content")
        
        valid, message = self.file_manager.validate_file_access(test_file)
        self.assertTrue(valid)
        self.assertEqual(message, "")
        
        # Test non-existent file
        missing_file = self.temp_dir / "missing.mkv"
        valid, message = self.file_manager.validate_file_access(missing_file)
        self.assertFalse(valid)
        self.assertIn("File not found", message)
        
        # Test empty file
        empty_file = self.temp_dir / "empty.mkv"
        empty_file.touch()
        valid, message = self.file_manager.validate_file_access(empty_file)
        self.assertFalse(valid)
        self.assertIn("File is empty", message)
    
    def test_startup_scavenge(self):
        """Test startup cleanup of stale files."""
        # Create mock stale files
        stale_files = [
            self.temp_dir / "test.sample_clip.mkv",
            self.temp_dir / "video_sample.mkv", 
            self.temp_dir / "vbr_ref_clip_0_30.mkv",
            self.temp_dir / "vbr_enc_clip_1_120.mkv"
        ]
        
        for f in stale_files:
            f.touch()
        
        # Also might pick up the test files we created in setUp
        initial_file_count = len(list(self.temp_dir.iterdir()))
        
        # Run scavenge
        removed_count = self.file_manager.startup_scavenge(self.temp_dir)
        
        # Should have removed at least the stale files we created
        self.assertGreaterEqual(removed_count, len(stale_files))
        
        # Our specific stale files should no longer exist
        for f in stale_files:
            self.assertFalse(f.exists())
    
    @patch('subprocess.run')
    def test_get_file_stats(self, mock_subprocess):
        """Test file statistics generation."""
        # Mock codec detection
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "h264"
        mock_subprocess.return_value = mock_result
        
        # Create test files with known sizes
        test_files = []
        for i in range(3):
            test_file = self.temp_dir / f"test{i}.mkv"
            test_file.write_bytes(b"0" * (1024 * 1024 * (i + 1)))  # 1MB, 2MB, 3MB
            test_files.append(test_file)
        
        stats = self.file_manager.get_file_stats(test_files)
        
        self.assertEqual(stats['total_files'], 3)
        self.assertAlmostEqual(stats['total_size_gb'], 6 / 1024, places=3)  # ~6MB
        self.assertEqual(stats['codec_distribution']['h264'], 3)
        self.assertAlmostEqual(stats['average_file_size_mb'], 2.0, places=1)


if __name__ == '__main__':
    unittest.main()
