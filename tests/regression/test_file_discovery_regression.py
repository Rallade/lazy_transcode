"""
File Discovery & Filtering Regression Tests

These tests prevent critical bugs in the file discovery pipeline that could
cause legitimate video files to be lost, ignored, or incorrectly processed.

The file discovery pipeline is the entry point for all transcoding operations,
making it critical to get right.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from lazy_transcode.core.modules.processing.file_manager import FileManager, FileDiscoveryResult, CodecCheckResult


class TestFileDiscoveryRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent file discovery regressions that could cause file loss.
    
    These tests would catch bugs where legitimate video files are incorrectly
    excluded from processing, leading to content not being transcoded.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = FileManager(debug=True)
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_directory_structure(self):
        """Create a realistic directory structure for testing."""
        self.temp_dir = tempfile.mkdtemp()
        base_path = Path(self.temp_dir)
        
        # Create legitimate video files
        legitimate_files = [
            "Episode 01.mkv",
            "Episode 02.mp4", 
            "Movie.2023.1080p.BluRay.x264.mkv",
            "Series.S01E03.720p.HDTV.h264.mkv",
            "Documentary.4K.HEVC.mp4",
            "Anime.EP04.1080p.mkv",
            "subfolder/Another.Episode.mkv"
        ]
        
        # Create files that should be filtered out
        filtered_files = [
            ".hidden_file.mkv",           # Hidden file
            "._resource_fork.mkv",        # macOS resource fork
            "sample.mkv",                 # Sample file
            "Episode.01.sample_clip.mkv", # Sample clip
            "test_sample.mkv",            # Sample file
            "Episode.01.qp25_sample.mkv", # QP test sample
            "vbr_ref_clip_001.mkv",       # VBR reference clip
        ]
        
        # Create all files
        for file_path in legitimate_files + filtered_files:
            full_path = base_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
        
        return base_path, legitimate_files, filtered_files
    
    def test_discover_video_files_finds_all_legitimate_files(self):
        """
        CRITICAL TEST: Ensure all legitimate video files are discovered.
        
        This test would have caught bugs where valid video files are 
        incorrectly excluded from processing.
        """
        base_path, legitimate_files, filtered_files = self._create_test_directory_structure()
        
        result = self.file_manager.discover_video_files(base_path)
        
        # Check that all legitimate files were found
        found_file_names = [f.name for f in result.files_to_transcode]
        
        for legit_file in legitimate_files:
            file_name = Path(legit_file).name
            with self.subTest(file=file_name):
                self.assertIn(file_name, found_file_names, 
                             f"Legitimate file {file_name} was not discovered")
        
        # Ensure we found the expected number of legitimate files
        self.assertEqual(len(result.files_to_transcode), len(legitimate_files),
                        f"Expected {len(legitimate_files)} files, found {len(result.files_to_transcode)}")
    
    def test_file_filtering_excludes_hidden_and_sample_files(self):
        """
        SAFETY TEST: Ensure hidden files and sample clips are properly filtered.
        
        This prevents processing system files and temporary artifacts.
        """
        base_path, legitimate_files, filtered_files = self._create_test_directory_structure()
        
        result = self.file_manager.discover_video_files(base_path)
        found_file_names = [f.name for f in result.files_to_transcode]
        
        # Ensure filtered files are NOT in the results
        for filtered_file in filtered_files:
            file_name = Path(filtered_file).name
            with self.subTest(file=file_name):
                self.assertNotIn(file_name, found_file_names,
                               f"Filtered file {file_name} should not be discovered")
        
        # Check that we reported the correct number of hidden files skipped
        expected_hidden = sum(1 for f in filtered_files if f.startswith('.'))
        self.assertEqual(result.hidden_files_skipped, expected_hidden,
                        f"Expected {expected_hidden} hidden files skipped")
    
    def test_extension_filtering_works_correctly(self):
        """
        COMPATIBILITY TEST: Ensure all supported extensions are discovered.
        
        This prevents bugs where certain video formats are ignored.
        """
        base_path = Path(tempfile.mkdtemp())
        self.temp_dir = str(base_path)
        
        # Create files with different extensions
        test_extensions = {
            "mkv": "should_be_found.mkv",
            "mp4": "should_be_found.mp4", 
            "mov": "should_be_found.mov",
            "ts": "should_be_found.ts",
            "avi": "should_not_be_found.avi",  # Not in default extensions
            "txt": "should_not_be_found.txt"   # Not a video file
        }
        
        for ext, filename in test_extensions.items():
            (base_path / filename).touch()
        
        result = self.file_manager.discover_video_files(base_path, extensions="mkv,mp4,mov,ts")
        found_extensions = {f.suffix[1:] for f in result.files_to_transcode}
        
        # Should find supported extensions
        supported_extensions = {"mkv", "mp4", "mov", "ts"}
        self.assertEqual(found_extensions, supported_extensions,
                        f"Should find exactly {supported_extensions}, found {found_extensions}")
    
    def test_recursive_directory_search_works(self):
        """
        FUNCTIONALITY TEST: Ensure recursive directory search finds nested files.
        
        This prevents bugs where files in subdirectories are missed.
        """
        base_path = Path(tempfile.mkdtemp())
        self.temp_dir = str(base_path)
        
        # Create nested directory structure
        nested_files = [
            "root_file.mkv",
            "subdir1/nested_file1.mkv",
            "subdir1/subdir2/deeply_nested.mkv",
            "another_dir/another_file.mp4"
        ]
        
        for file_path in nested_files:
            full_path = base_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
        
        result = self.file_manager.discover_video_files(base_path)
        found_file_names = {f.name for f in result.files_to_transcode}
        expected_names = {Path(f).name for f in nested_files}
        
        self.assertEqual(found_file_names, expected_names,
                        f"Should find all nested files. Expected: {expected_names}, Found: {found_file_names}")


class TestCodecFilteringRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent codec filtering regressions that could skip files needing transcoding.
    
    These tests ensure that files with inefficient codecs (like H.264) are always
    considered for transcoding, while files with efficient codecs are properly skipped.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = FileManager(debug=True)
    
    def test_h264_files_are_never_skipped_by_codec_filter(self):
        """
        CRITICAL TEST: H.264 files should always be candidates for transcoding.
        
        This would catch regressions where H.264 files are incorrectly
        classified as "already efficient" and skipped.
        """
        with patch('lazy_transcode.core.modules.file_manager.run_command') as mock_run_command:
            # Mock ffprobe to return h264
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "h264"
            mock_run_command.return_value = mock_result
            
            test_file = Path("/fake/h264_video.mkv")
            codec_result = self.file_manager.check_video_codec(test_file)
            
            self.assertEqual(codec_result.codec, "h264")
            self.assertFalse(codec_result.should_skip, 
                           "H.264 files should never be skipped - they need transcoding to HEVC")
            self.assertEqual(codec_result.reason, "h264")
    
    def test_hevc_files_are_correctly_skipped(self):
        """
        EFFICIENCY TEST: HEVC files should be skipped as already efficient.
        
        This ensures we don't waste time re-encoding already efficient files.
        """
        with patch('lazy_transcode.core.modules.file_manager.run_command') as mock_run_command:
            # Mock ffprobe to return hevc
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "hevc"
            mock_run_command.return_value = mock_result
            
            test_file = Path("/fake/hevc_video.mkv")
            codec_result = self.file_manager.check_video_codec(test_file)
            
            self.assertEqual(codec_result.codec, "hevc")
            self.assertTrue(codec_result.should_skip,
                          "HEVC files should be skipped as already efficient")
            self.assertEqual(codec_result.reason, "already hevc")
    
    def test_codec_detection_handles_ffprobe_failure_gracefully(self):
        """
        ROBUSTNESS TEST: Codec detection failure should not skip files.
        
        When codec detection fails, we should err on the side of processing
        the file rather than skipping it.
        """
        with patch('lazy_transcode.core.modules.file_manager.run_command') as mock_run_command:
            # Mock ffprobe failure
            mock_result = MagicMock()
            mock_result.returncode = 1  # Failure
            mock_result.stdout = ""
            mock_result.stderr = "ffprobe error"
            mock_run_command.return_value = mock_result
            
            test_file = Path("/fake/unknown_video.mkv")
            codec_result = self.file_manager.check_video_codec(test_file)
            
            self.assertIsNone(codec_result.codec)
            self.assertFalse(codec_result.should_skip,
                           "Files with unknown codecs should not be skipped - better safe than sorry")
            self.assertEqual(codec_result.reason, "codec detection failed")
    
    def test_all_efficient_codecs_are_recognized(self):
        """
        COMPLETENESS TEST: All known efficient codecs should be properly identified.
        
        This ensures the efficient codec list is comprehensive and up-to-date.
        """
        efficient_codecs = ["hevc", "h265", "av1"]
        
        for codec in efficient_codecs:
            with self.subTest(codec=codec):
                with patch('lazy_transcode.core.modules.file_manager.run_command') as mock_run_command:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = codec
                    mock_run_command.return_value = mock_result
                    
                    test_file = Path(f"/fake/{codec}_video.mkv")
                    codec_result = self.file_manager.check_video_codec(test_file)
                    
                    self.assertTrue(codec_result.should_skip,
                                  f"{codec} should be recognized as efficient and skipped")
                    self.assertEqual(codec_result.reason, f"already {codec}")


class TestSampleDetectionRegression(unittest.TestCase):
    """
    SAFETY TESTS: Prevent sample detection regressions that could exclude real episodes.
    
    These tests ensure that the sample detection logic is conservative and doesn't
    accidentally classify legitimate episodes as sample clips.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = FileManager(debug=True)
    
    def test_legitimate_episode_files_are_never_classified_as_samples(self):
        """
        CRITICAL TEST: Real episode files should never be classified as samples.
        
        This prevents legitimate content from being excluded due to overly
        aggressive sample detection.
        """
        legitimate_episode_names = [
            "Attack on Titan S04E01.mkv",
            "Demon Slayer Ep 12.mkv", 
            "One Piece Episode 1000.mp4",
            "Naruto Shippuden 500.mkv",
            "Movie.2023.1080p.BluRay.mkv",
            "Series.Name.S01E01.720p.HDTV.mkv",
            "Documentary.2023.4K.mkv"
        ]
        
        for episode_name in legitimate_episode_names:
            with self.subTest(episode=episode_name):
                test_path = Path(f"/fake/{episode_name}")
                is_sample = self.file_manager._is_sample_or_artifact(test_path)
                
                self.assertFalse(is_sample, 
                               f"Legitimate episode '{episode_name}' incorrectly classified as sample")
    
    def test_actual_sample_files_are_correctly_identified(self):
        """
        FUNCTIONALITY TEST: Actual sample clips should be identified correctly.
        
        This ensures sample detection is working and filters out actual artifacts.
        """
        actual_sample_names = [
            "Episode.01.sample_clip.mkv",
            "Movie_sample.mkv",
            "test.sample.mkv",
            "Episode.01.qp25_sample.mkv",
            "vbr_ref_clip_001.mkv",
            "vbr_enc_clip_002.mkv",
            "clip1_sample.mkv"
        ]
        
        for sample_name in actual_sample_names:
            with self.subTest(sample=sample_name):
                test_path = Path(f"/fake/{sample_name}")
                is_sample = self.file_manager._is_sample_or_artifact(test_path)
                
                self.assertTrue(is_sample,
                              f"Sample file '{sample_name}' should be identified as sample")
    
    def test_edge_case_filenames_are_handled_correctly(self):
        """
        EDGE CASE TEST: Files with similar but legitimate names should not be filtered.
        
        This tests the boundaries of sample detection to ensure it's not too broad.
        """
        edge_case_names = [
            "Sample.Anime.Series.EP01.mkv",     # "Sample" is part of title
            "The.Sample.Documentary.mkv",        # "Sample" is part of title
            "Clinical.Sample.Study.mkv",         # Legitimate use of word "sample"
            "Sampling.Theory.Lecture.mkv",       # Contains "sampl" but legitimate
            "Example.Episode.mkv",               # "Example" not "sample"
        ]
        
        for edge_case in edge_case_names:
            with self.subTest(edge_case=edge_case):
                test_path = Path(f"/fake/{edge_case}")
                is_sample = self.file_manager._is_sample_or_artifact(test_path)
                
                self.assertFalse(is_sample,
                               f"Edge case file '{edge_case}' should not be classified as sample")


class TestFileProcessingWorkflowIntegration(unittest.TestCase):
    """
    INTEGRATION TESTS: Test the complete file processing workflow.
    
    These tests ensure that the file discovery, filtering, and codec checking
    work together correctly as an integrated pipeline.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = FileManager(debug=True)
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow_processes_correct_files(self):
        """
        WORKFLOW TEST: Complete processing should result in correct file categorization.
        
        This tests the integration of discovery + filtering + codec checking.
        """
        # Create test directory
        self.temp_dir = tempfile.mkdtemp()
        base_path = Path(self.temp_dir)
        
        # Create test files with known characteristics
        test_files = {
            "h264_episode.mkv": "h264",      # Should be processed
            "hevc_episode.mkv": "hevc",      # Should be skipped
            "av1_episode.mkv": "av1",        # Should be skipped
            "unknown_codec.mkv": "unknown",   # Should be processed (safe default)
        }
        
        # Create sample files that should be filtered out before codec checking
        sample_files = [
            "episode_sample.mkv",
            "test.sample_clip.mkv"
        ]
        
        # Create all files
        for filename in list(test_files.keys()) + sample_files:
            (base_path / filename).touch()
        
        # Mock codec detection
        def mock_codec_check(file_path):
            filename = file_path.name
            if filename in test_files:
                expected_codec = test_files[filename]
                if expected_codec == "hevc" or expected_codec == "av1":
                    return CodecCheckResult(
                        codec=expected_codec,
                        should_skip=True,
                        reason=f"already {expected_codec}"
                    )
                elif expected_codec == "h264":
                    return CodecCheckResult(
                        codec=expected_codec,
                        should_skip=False,
                        reason=""
                    )
                else:  # unknown
                    return CodecCheckResult(
                        codec=None,
                        should_skip=False,
                        reason="unknown codec - will transcode for safety"
                    )
            return CodecCheckResult(codec=None, should_skip=False, reason="unknown")

        with patch.object(self.file_manager, 'check_video_codec', side_effect=mock_codec_check):
            result = self.file_manager.process_files_with_codec_filtering(base_path)
            
            # Should find files that need transcoding (h264 and unknown)
            files_to_transcode = {f.name for f in result.files_to_transcode}
            expected_to_transcode = {"h264_episode.mkv", "unknown_codec.mkv"}
            self.assertEqual(files_to_transcode, expected_to_transcode,
                           f"Expected to transcode {expected_to_transcode}, got {files_to_transcode}")
            
            # Should skip efficient codecs
            skipped_files = {f[0].name for f in result.skipped_files}
            expected_skipped = {"hevc_episode.mkv", "av1_episode.mkv"}
            self.assertEqual(skipped_files, expected_skipped,
                           f"Expected to skip {expected_skipped}, got {skipped_files}")
            
            # Sample files should not appear in either category
            all_processed = files_to_transcode | skipped_files
            for sample_file in sample_files:
                self.assertNotIn(sample_file, all_processed,
                               f"Sample file {sample_file} should not be processed at all")


class TestCleanupSafetyRegression(unittest.TestCase):
    """
    SAFETY TESTS: Prevent cleanup regressions that could cause data loss.
    
    These tests ensure that cleanup operations are conservative and never
    accidentally delete legitimate files.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = FileManager(debug=True)
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_startup_scavenge_never_deletes_legitimate_files(self):
        """
        CRITICAL SAFETY TEST: Cleanup should never touch legitimate video files.
        
        This prevents catastrophic data loss from overly aggressive cleanup.
        """
        self.temp_dir = tempfile.mkdtemp()
        base_path = Path(self.temp_dir)
        
        # Create legitimate files that should NEVER be deleted
        legitimate_files = [
            "Episode.01.mkv",
            "Movie.2023.mkv",
            "sample_video.mkv",  # Contains "sample" but is legitimate
            "clip_from_movie.mkv"  # Contains "clip" but is legitimate
        ]
        
        # Create files that SHOULD be cleaned up
        cleanup_files = [
            "Episode.01.sample_clip.001.mkv",
            "test_sample.mkv",
            "Episode.01.qp25_sample.mkv",
            "vbr_ref_clip_001.mkv",
            "vbr_enc_clip_002.mkv"
        ]
        
        # Create all files
        all_files = legitimate_files + cleanup_files
        for filename in all_files:
            (base_path / filename).touch()
        
        # Run cleanup
        removed_count = self.file_manager.startup_scavenge(base_path)
        
        # Verify legitimate files still exist
        for legit_file in legitimate_files:
            file_path = base_path / legit_file
            with self.subTest(file=legit_file):
                self.assertTrue(file_path.exists(),
                              f"Legitimate file '{legit_file}' was incorrectly deleted by cleanup")
        
        # Verify cleanup files were removed
        for cleanup_file in cleanup_files:
            file_path = base_path / cleanup_file
            with self.subTest(file=cleanup_file):
                self.assertFalse(file_path.exists(),
                               f"Cleanup file '{cleanup_file}' should have been removed")
        
        # Verify correct count
        self.assertEqual(removed_count, len(cleanup_files),
                        f"Should have removed {len(cleanup_files)} files, removed {removed_count}")
    
    def test_cleanup_patterns_are_sufficiently_specific(self):
        """
        PATTERN SAFETY TEST: Cleanup patterns should be specific enough to avoid false positives.
        
        This ensures cleanup patterns don't accidentally match legitimate files.
        """
        # Test patterns that should NOT match legitimate files
        safe_filenames = [
            "Sample.Anime.Episode.01.mkv",      # "Sample" in title
            "Documentary.Sample.Study.mkv",      # "Sample" in title
            "The.Clip.Show.EP01.mkv",           # "Clip" in title
            "Music.Video.Clips.Collection.mkv", # "Clips" in title
        ]
        
        for filename in safe_filenames:
            with self.subTest(filename=filename):
                test_path = Path(f"/fake/{filename}")
                
                # Test that it's not classified as a sample/artifact
                is_sample = self.file_manager._is_sample_or_artifact(test_path)
                self.assertFalse(is_sample,
                               f"Filename '{filename}' should not match cleanup patterns")


if __name__ == '__main__':
    # Run with high verbosity to see detailed test results
    unittest.main(verbosity=2)
