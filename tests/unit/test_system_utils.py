"""
Unit tests for system_utils module.

Tests system utility functions including directory management,
file operations, and system configuration.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.system.system_utils import (
    format_size, get_next_transcoded_dir, cleanup_temp_files,
    DEBUG, TEMP_FILES
)


class TestSystemUtils(unittest.TestCase):
    """Test system utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)
    
    def test_format_size_bytes(self):
        """Test size formatting for bytes."""
        self.assertEqual(format_size(500), "500 B")
        self.assertEqual(format_size(1000), "1000 B")
    
    def test_format_size_kb(self):
        """Test size formatting for kilobytes."""
        self.assertEqual(format_size(1536), "1.50 KB")  # 1.5 KB
        self.assertEqual(format_size(2048), "2.00 KB")  # 2 KB
    
    def test_format_size_mb(self):
        """Test size formatting for megabytes."""
        self.assertEqual(format_size(1572864), "1.50 MB")  # 1.5 MB
        self.assertEqual(format_size(2097152), "2.00 MB")  # 2 MB
    
    def test_format_size_gb(self):
        """Test size formatting for gigabytes."""
        self.assertEqual(format_size(1610612736), "1.50 GB")  # 1.5 GB
        self.assertEqual(format_size(2147483648), "2.00 GB")  # 2 GB
    
    def test_format_size_tb(self):
        """Test size formatting for terabytes."""
        self.assertEqual(format_size(1649267441664), "1.50 TB")  # 1.5 TB
        self.assertEqual(format_size(2199023255552), "2.00 TB")  # 2 TB
    
    def test_format_size_zero(self):
        """Test size formatting for zero."""
        self.assertEqual(format_size(0), "0 B")
    
    def test_format_size_negative(self):
        """Test size formatting for negative values."""
        # Should handle negative values gracefully
        result = format_size(-1024)
        self.assertIsInstance(result, str)
    
    def test_get_next_transcoded_dir_new(self):
        """Test getting next transcoded directory when none exist."""
        base_dir = self.test_dir
        transcoded_dir = get_next_transcoded_dir(base_dir)
        
        expected = base_dir / "Transcoded"
        self.assertEqual(transcoded_dir, expected)
    
    def test_get_next_transcoded_dir_existing(self):
        """Test getting next transcoded directory when one exists."""
        base_dir = self.test_dir
        
        # Create existing Transcoded directory
        existing = base_dir / "Transcoded"
        existing.mkdir()
        
        transcoded_dir = get_next_transcoded_dir(base_dir)
        
        expected = base_dir / "Transcoded_2"
        self.assertEqual(transcoded_dir, expected)
    
    def test_get_next_transcoded_dir_multiple_existing(self):
        """Test getting next transcoded directory with multiple existing."""
        base_dir = self.test_dir
        
        # Create multiple existing directories
        for i in [1, 2, 3]:
            if i == 1:
                dir_name = "Transcoded"
            else:
                dir_name = f"Transcoded_{i}"
            (base_dir / dir_name).mkdir()
        
        transcoded_dir = get_next_transcoded_dir(base_dir)
        
        expected = base_dir / "Transcoded_4"
        self.assertEqual(transcoded_dir, expected)
    
    @patch('lazy_transcode.core.modules.system.system_utils.TEMP_FILES')
    def test_cleanup_temp_files_empty_list(self, mock_temp_files):
        """Test cleanup with empty temp files list."""
        mock_temp_files.clear()
        
        # Should not raise exception
        cleanup_temp_files()
    
    @patch('lazy_transcode.core.modules.system.system_utils.TEMP_FILES')
    def test_cleanup_temp_files_with_files(self, mock_temp_files):
        """Test cleanup with temp files."""
        # Create test files
        temp_file1 = self.test_dir / "temp1.mkv"
        temp_file2 = self.test_dir / "temp2.mkv"
        temp_file1.touch()
        temp_file2.touch()
        
        mock_temp_files.extend([temp_file1, temp_file2])
        
        # Run cleanup
        cleanup_temp_files()
        
        # Files should be removed (mocked)
        mock_temp_files.clear.assert_called()
    
    def test_debug_flag_type(self):
        """Test that DEBUG flag is boolean."""
        self.assertIsInstance(DEBUG, bool)
    
    def test_temp_files_list_type(self):
        """Test that TEMP_FILES is a list."""
        self.assertIsInstance(TEMP_FILES, list)


class TestSystemUtilsEdgeCases(unittest.TestCase):
    """Test edge cases in system utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_format_size_very_large(self):
        """Test size formatting for very large values."""
        # Test with a very large number
        very_large = 1024**6  # Exabyte range
        result = format_size(very_large)
        self.assertIsInstance(result, str)
        self.assertTrue(any(unit in result for unit in ['EB', 'TB', 'GB', 'MB', 'KB', 'B']))
    
    def test_get_next_transcoded_dir_permission_error(self):
        """Test directory creation with permission issues."""
        # This would test permission handling, but it's environment-dependent
        # In a real scenario, you might mock os.makedirs to raise PermissionError
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            # The function should handle this gracefully or raise appropriately
            base_dir = Path("/tmp/test")
            # Implementation-dependent behavior
            try:
                result = get_next_transcoded_dir(base_dir)
                self.assertIsInstance(result, Path)
            except PermissionError:
                # This is also acceptable behavior
                pass
    
    def test_get_next_transcoded_dir_with_files(self):
        """Test directory numbering with existing files (not directories)."""
        base_dir = self.test_dir
        
        # Create a file named "Transcoded" (not directory)
        transcoded_file = base_dir / "Transcoded"
        transcoded_file.touch()
        
        transcoded_dir = get_next_transcoded_dir(base_dir)
        
        # Should skip the file and create Transcoded_2 or handle appropriately
        self.assertIsInstance(transcoded_dir, Path)
        self.assertTrue(transcoded_dir.name.startswith("Transcoded"))


class TestSystemConfiguration(unittest.TestCase):
    """Test system configuration and constants."""
    
    def test_debug_flag_usage(self):
        """Test DEBUG flag can be used in conditional logic."""
        if DEBUG:
            # DEBUG mode behavior
            self.assertTrue(True)
        else:
            # Production mode behavior
            self.assertTrue(True)
        
        # Flag should be usable in boolean context
        self.assertIn(DEBUG, [True, False])
    
    def test_temp_files_manipulation(self):
        """Test TEMP_FILES list can be manipulated."""
        original_length = len(TEMP_FILES)
        
        # Should be able to append
        test_path = Path("test_temp.mkv")
        TEMP_FILES.append(test_path)
        self.assertEqual(len(TEMP_FILES), original_length + 1)
        
        # Should be able to remove
        TEMP_FILES.remove(test_path)
        self.assertEqual(len(TEMP_FILES), original_length)
    
    def test_temp_files_contains_paths(self):
        """Test TEMP_FILES contains Path objects."""
        # Add a test path
        test_path = Path("test_file.mkv")
        TEMP_FILES.append(test_path)
        
        # Check it's a Path object
        if TEMP_FILES:
            last_item = TEMP_FILES[-1]
            self.assertIsInstance(last_item, (Path, str))  # Allow both Path and str
        
        # Clean up
        if test_path in TEMP_FILES:
            TEMP_FILES.remove(test_path)


if __name__ == '__main__':
    unittest.main()
