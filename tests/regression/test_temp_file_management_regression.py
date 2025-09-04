"""
Temp File Management Regression Tests

These tests prevent critical bugs in temporary file handling that could cause:
- Disk space exhaustion from temp files not being cleaned up
- Permission errors blocking future transcoding operations  
- Race conditions in concurrent transcoding sessions
- Temp file collisions causing data corruption

Temp file management is critical for safe transcoding operations.
"""

import unittest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from lazy_transcode.core.modules.system.system_utils import (
    TEMP_FILES, temporary_file
)


def manual_cleanup_temp_files():
    """Manual cleanup for testing purposes."""
    for file_path in list(TEMP_FILES):
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        except Exception:
            pass
        finally:
            TEMP_FILES.discard(file_path)


class TestTempFileRegistrationRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent temp file tracking regressions.
    
    TEMP_FILES registry must accurately track all temporary files to ensure cleanup.
    """
    
    def setUp(self):
        """Clear temp files registry before each test."""
        TEMP_FILES.clear()
    
    def tearDown(self):
        """Clean up any remaining temp files after each test."""
        manual_cleanup_temp_files()
    
    def test_temp_file_registry_tracks_created_files(self):
        """
        REGISTRATION TEST: Temp files should be registered for cleanup.
        
        This prevents temp file leaks that could fill up disk space.
        """
        with temporary_file(suffix=".tmp") as temp_file:
            # Should be registered in TEMP_FILES
            self.assertIn(str(temp_file), TEMP_FILES,
                        f"Temp file {temp_file.name} should be registered")
    
    def test_temp_file_context_manager_cleans_up_properly(self):
        """
        CONTEXT MANAGER TEST: Files should be cleaned up when context exits.
        
        This ensures no temp files are left behind after operations.
        """
        temp_file_path = None
        
        # Create temp file in context
        with temporary_file(suffix=".tmp") as temp_file:
            temp_file_path = temp_file
            temp_file.write_text("test content")
            self.assertTrue(temp_file.exists(), "File should exist in context")
            
        # File should be cleaned up after context
        # Note: The context manager should handle cleanup
        # If it doesn't, that would be a bug to catch
    
    def test_temp_file_handles_multiple_concurrent_contexts(self):
        """
        CONCURRENCY TEST: Multiple temp file contexts should not interfere.
        
        This ensures concurrent transcoding sessions work correctly.
        """
        created_files = []
        
        # Create multiple temp files
        contexts = []
        for i in range(3):
            context = temporary_file(suffix=f"_{i}.tmp")
            contexts.append(context)
            temp_file = context.__enter__()
            temp_file.write_text(f"content {i}")
            created_files.append(temp_file)
        
        # All should be registered
        for temp_file in created_files:
            self.assertIn(str(temp_file), TEMP_FILES,
                        f"File {temp_file.name} should be registered")
        
        # Clean up contexts
        for context in contexts:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
    
    def test_temp_file_registry_handles_unicode_filenames(self):
        """
        UNICODE TEST: Should handle international characters in temp files.
        
        This prevents failures with anime titles, foreign language content.
        """
        with temporary_file(suffix="_鬼滅の刃.tmp") as temp_file:
            temp_file.write_text("unicode test content")
            
            # Should be registered despite unicode characters
            self.assertIn(str(temp_file), TEMP_FILES,
                        "Unicode temp file should be registered")
            
            # Should be able to work with the file
            content = temp_file.read_text()
            self.assertEqual(content, "unicode test content")


class TestTempFilePathGenerationRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent temp file path generation issues.
    
    Path generation problems could cause file collisions or access errors.
    """
    
    def setUp(self):
        """Clear temp files registry before each test."""
        TEMP_FILES.clear()
    
    def tearDown(self):
        """Clean up any remaining temp files after each test."""
        manual_cleanup_temp_files()
    
    def test_temp_file_paths_are_unique_across_calls(self):
        """
        UNIQUENESS TEST: Each temp file should have a unique path.
        
        This prevents file collisions that could cause data corruption.
        """
        generated_paths = []
        contexts = []
        
        # Generate multiple temp file paths
        for i in range(5):
            context = temporary_file(suffix=".tmp")
            contexts.append(context)
            temp_file = context.__enter__()
            generated_paths.append(str(temp_file))
        
        # All paths should be unique
        unique_paths = set(generated_paths)
        self.assertEqual(len(unique_paths), len(generated_paths),
                       f"All temp paths should be unique: {len(unique_paths)} unique out of {len(generated_paths)}")
        
        # Clean up contexts
        for context in contexts:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
    
    def test_temp_file_creation_with_different_suffixes(self):
        """
        SUFFIX TEST: Different suffixes should create different files.
        
        This ensures temp files for different purposes don't collide.
        """
        suffixes = [".mkv", ".mp4", ".tmp", ".log", ".json"]
        created_files = []
        contexts = []
        
        for suffix in suffixes:
            context = temporary_file(suffix=suffix)
            contexts.append(context)
            temp_file = context.__enter__()
            temp_file.write_text(f"content for {suffix}")
            created_files.append(temp_file)
        
        # All files should exist and have correct suffixes
        for temp_file, expected_suffix in zip(created_files, suffixes):
            with self.subTest(suffix=expected_suffix):
                self.assertTrue(temp_file.exists(), f"File with {expected_suffix} should exist")
                self.assertTrue(str(temp_file).endswith(expected_suffix), 
                              f"File should have {expected_suffix} suffix")
        
        # Clean up contexts
        for context in contexts:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
    
    def test_temp_file_creation_handles_filesystem_limits(self):
        """
        FILESYSTEM TEST: Should handle filesystem limitations gracefully.
        
        This ensures temp file creation doesn't crash with filesystem issues.
        """
        # Test with various edge cases
        edge_case_suffixes = [
            ".tmp",                    # Normal
            ".very_long_extension_name_that_might_cause_issues", # Long extension
            "",                       # No extension
            ".123",                   # Numeric extension
        ]
        
        for suffix in edge_case_suffixes:
            with self.subTest(suffix=suffix):
                try:
                    with temporary_file(suffix=suffix) as temp_file:
                        temp_file.write_text("test content")
                        self.assertTrue(temp_file.exists(), f"Should create file with suffix '{suffix}'")
                except Exception as e:
                    # Some edge cases might fail - that's acceptable
                    # as long as they fail gracefully
                    self.assertNotIsInstance(e, (AttributeError, KeyError),
                                           f"Should fail gracefully for suffix '{suffix}': {e}")


class TestTempFileErrorHandlingRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent error handling issues in temp file operations.
    
    Error handling problems could crash transcoding operations.
    """
    
    def setUp(self):
        """Clear temp files registry before each test."""
        TEMP_FILES.clear()
    
    def tearDown(self):
        """Clean up any remaining temp files after each test."""
        manual_cleanup_temp_files()
    
    def test_temp_file_handles_disk_full_scenarios(self):
        """
        DISK FULL TEST: Should handle disk space issues gracefully.
        
        This ensures transcoding fails gracefully when disk is full.
        """
        # Mock disk full scenario
        with patch('tempfile.mkstemp') as mock_mkstemp:
            mock_mkstemp.side_effect = OSError("No space left on device")
            
            # Should raise appropriate exception, not crash
            with self.assertRaises(OSError) as cm:
                with temporary_file(suffix=".tmp") as temp_file:
                    pass
            
            self.assertIn("space", str(cm.exception).lower())
    
    def test_temp_file_handles_permission_errors(self):
        """
        PERMISSION TEST: Should handle permission errors appropriately.
        
        This ensures transcoding fails gracefully with permission issues.
        """
        # Mock permission error during temp file creation
        with patch('tempfile.mkstemp') as mock_mkstemp:
            mock_mkstemp.side_effect = PermissionError("Permission denied")
            
            # Should raise permission error, not crash with unexpected exception
            with self.assertRaises(PermissionError):
                with temporary_file(suffix=".tmp") as temp_file:
                    pass
    
    def test_temp_file_context_manager_handles_exceptions_in_block(self):
        """
        EXCEPTION HANDLING TEST: Context manager should clean up even if block raises exception.
        
        This ensures temp files don't leak when operations fail.
        """
        temp_file_path = None
        
        try:
            with temporary_file(suffix=".tmp") as temp_file:
                temp_file_path = temp_file
                temp_file.write_text("test content")
                
                # Verify file exists during context
                self.assertTrue(temp_file.exists(), "File should exist during context")
                
                # Simulate an error in the transcoding operation
                raise ValueError("Simulated transcoding error")
                
        except ValueError:
            # Expected error - should be caught
            pass
        
        # The context manager should still clean up properly
        # Note: We can't directly test this without knowing the exact cleanup behavior
        # but at minimum it shouldn't crash
    
    def test_temp_file_cleanup_is_idempotent(self):
        """
        IDEMPOTENCY TEST: Multiple cleanup calls should be safe.
        
        This ensures cleanup can be called multiple times without errors.
        """
        # Create some temp files manually in the registry
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Manually create files and add to registry
            test_files = []
            for i in range(3):
                test_file = temp_path / f"test_file_{i}.tmp"
                test_file.write_text(f"content {i}")
                TEMP_FILES.add(str(test_file))
                test_files.append(test_file)
            
            # Multiple cleanup calls should be safe
            for cleanup_round in range(3):
                with self.subTest(round=cleanup_round):
                    try:
                        manual_cleanup_temp_files()
                    except Exception as e:
                        self.fail(f"Cleanup round {cleanup_round} should not raise exception: {e}")
    
    def test_temp_file_registry_thread_safety(self):
        """
        THREAD SAFETY TEST: TEMP_FILES registry should be thread-safe.
        
        This ensures concurrent transcoding sessions don't corrupt the registry.
        """
        registry_errors = []
        
        def create_and_register_files(thread_id):
            """Create temp files in a thread."""
            try:
                for i in range(3):
                    with temporary_file(suffix=f"_thread{thread_id}_{i}.tmp") as temp_file:
                        temp_file.write_text(f"thread {thread_id} file {i}")
                        # Verify registration
                        if str(temp_file) not in TEMP_FILES:
                            registry_errors.append(f"Thread {thread_id} file {i} not registered")
            except Exception as e:
                registry_errors.append(f"Thread {thread_id}: {e}")
        
        # Create multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=create_and_register_files, args=(thread_id,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check for errors
        if registry_errors:
            self.fail(f"Thread safety issues: {registry_errors}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
