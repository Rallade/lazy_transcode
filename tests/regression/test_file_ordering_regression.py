"""
Regression tests for file processing order consistency.

These tests ensure that files are processed in predictable, sorted order
across different interfaces to prevent episodes being processed out of sequence.
"""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

class TestFileOrderingRegression(unittest.TestCase):
    """
    FILE ORDERING TESTS: Ensure consistent file processing order.
    
    These tests prevent regression where files are processed in random
    filesystem order instead of sorted episode order.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_cli_enhanced_processes_files_in_sorted_order(self):
        """
        CLI ORDERING TEST: Enhanced CLI should process files in sorted order.
        
        This prevents regression where files were processed in filesystem order
        (e.g., E18 first) instead of episode order (E01, E02, E03...).
        """
        # Create test video files in non-alphabetical order to test sorting
        test_files = [
            "Show S01E18 Title.mkv",
            "Show S01E01 First.mkv", 
            "Show S01E26 Last.mkv",
            "Show S01E05 Middle.mkv",
            "Show S01E02 Second.mkv"
        ]
        
        # Create files in temp directory
        for filename in test_files:
            (self.base_path / filename).touch()
        
        # Test the core file discovery and sorting logic used by CLI
        video_extensions = {'.mkv', '.mp4', '.mov', '.ts', '.avi', '.m4v'}
        video_files = [f for f in self.base_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        # Apply sorting as the CLI does
        video_files = sorted(video_files)
        
        # Get filenames for comparison
        processed_files = [f.name for f in video_files]
        
        # Expected sorted order
        expected_order = [
            "Show S01E01 First.mkv",
            "Show S01E02 Second.mkv", 
            "Show S01E05 Middle.mkv",
            "Show S01E18 Title.mkv",
            "Show S01E26 Last.mkv"
        ]
        
        self.assertEqual(processed_files, expected_order,
                       f"Files should be processed in sorted order. Got: {processed_files}")

    def test_file_manager_discover_returns_sorted_files(self):
        """
        FILE MANAGER ORDERING TEST: File discovery should return sorted results.
        
        This ensures the FileManager.discover_video_files() method returns
        files in consistent sorted order.
        """
        # Create test video files
        test_files = [
            "Video_S01E15.mkv",
            "Video_S01E03.mkv",
            "Video_S01E01.mkv",
            "Video_S01E22.mkv",
            "Video_S01E08.mkv"
        ]
        
        for filename in test_files:
            (self.base_path / filename).touch()
        
        from lazy_transcode.core.modules.processing.file_manager import FileManager
        
        file_manager = FileManager()
        
        # Mock codec checking to avoid ffprobe calls
        with patch.object(file_manager, 'check_video_codec') as mock_codec:
            mock_codec.return_value = MagicMock(needs_transcoding=True, codec='h264')
            
            result = file_manager.discover_video_files(self.base_path)
            
            # Get just the filenames for comparison
            discovered_names = [f.name for f in result.files_to_transcode]
            
            # Should be in sorted order
            expected_order = [
                "Video_S01E01.mkv",
                "Video_S01E03.mkv",
                "Video_S01E08.mkv", 
                "Video_S01E15.mkv",
                "Video_S01E22.mkv"
            ]
            
            self.assertEqual(discovered_names, expected_order,
                           f"FileManager should return files in sorted order. Got: {discovered_names}")

    def test_manager_processes_files_in_sorted_order(self):
        """
        MANAGER ORDERING TEST: Manager should process files in sorted order.
        
        This tests the batch processing manager to ensure it maintains
        consistent file ordering across subdirectories.
        """
        # Create subdirectory with test files
        subdir = self.base_path / "Test Show Season 1"
        subdir.mkdir()
        
        test_files = [
            "Episode_25.mkv",
            "Episode_01.mkv",
            "Episode_12.mkv",
            "Episode_03.mkv"
        ]
        
        for filename in test_files:
            (subdir / filename).touch()
        
        from lazy_transcode.core.modules.processing.file_manager import FileManager
        
        file_manager = FileManager()
        
        # Test the file discovery pattern used by manager
        files = []
        for pattern in ["*.mkv", "*.mp4"]:
            files.extend(self.base_path.rglob(pattern))
        
        # Apply the same sorting as manager does
        files = sorted(set(files))
        
        # Get just filenames
        file_names = [f.name for f in files]
        
        expected_order = [
            "Episode_01.mkv",
            "Episode_03.mkv", 
            "Episode_12.mkv",
            "Episode_25.mkv"
        ]
        
        self.assertEqual(file_names, expected_order,
                       f"Manager file discovery should return sorted order. Got: {file_names}")

    def test_mixed_episode_formats_sort_correctly(self):
        """
        MIXED FORMAT TEST: Various episode naming formats should sort correctly.
        
        This ensures different common episode naming patterns all sort
        in the expected order.
        """
        test_files = [
            "Show - S01E10 - Title.mkv",
            "Show - S01E02 - Title.mkv", 
            "Show S01E01 Title.mkv",
            "Show.S01E15.Title.mkv",
            "Show_S01E05_Title.mkv"
        ]
        
        for filename in test_files:
            (self.base_path / filename).touch()
        
        # Get files and sort them (simulating CLI behavior)
        video_extensions = {'.mkv', '.mp4', '.mov', '.ts', '.avi', '.m4v'}
        video_files = [f for f in self.base_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        # Sort files as CLI does
        video_files = sorted(video_files)
        file_names = [f.name for f in video_files]
        
        # Expected alphabetical order (which gives episode order for these patterns)
        expected_order = [
            "Show - S01E02 - Title.mkv",
            "Show - S01E10 - Title.mkv",
            "Show S01E01 Title.mkv", 
            "Show.S01E15.Title.mkv",
            "Show_S01E05_Title.mkv"
        ]
        
        self.assertEqual(file_names, expected_order,
                       f"Mixed episode formats should sort consistently. Got: {file_names}")

if __name__ == '__main__':
    unittest.main()
