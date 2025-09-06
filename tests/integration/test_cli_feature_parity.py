"""
Feature parity validation test.

This test verifies that both CLI entrypoints support the same core functionality
and can handle basic operations without errors.
"""

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
from pathlib import Path

# Add the project root to sys.path to ensure imports work
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestCLIFeatureParity(unittest.TestCase):
    """Test feature parity between CLI entrypoints."""

    def test_both_clis_show_help_without_error(self):
        """
        Test that both CLIs can display help without errors.
        
        This validates that argument parsing works correctly in both entrypoints.
        """
        # Test enhanced CLI help
        try:
            result_enhanced = subprocess.run([
                sys.executable, "-m", "lazy_transcode.cli_enhanced", "--help"
            ], capture_output=True, text=True, cwd=project_root)
            
            self.assertEqual(result_enhanced.returncode, 0, 
                           "Enhanced CLI help should execute without error")
            self.assertIn("--limit", result_enhanced.stdout,
                         "Enhanced CLI should support --limit flag")
            self.assertIn("--include-h265", result_enhanced.stdout,
                         "Enhanced CLI should support --include-h265 flag")
            self.assertIn("--cleanup", result_enhanced.stdout,
                         "Enhanced CLI should support --cleanup flag")
            
        except Exception as e:
            self.fail(f"Enhanced CLI help failed: {e}")
        
        # Test original CLI help  
        try:
            result_original = subprocess.run([
                sys.executable, "-m", "lazy_transcode.core.main", "--help"
            ], capture_output=True, text=True, cwd=project_root)
            
            self.assertEqual(result_original.returncode, 0,
                           "Original CLI help should execute without error")
            self.assertIn("--limit", result_original.stdout,
                         "Original CLI should support --limit flag")
            self.assertIn("--include-h265", result_original.stdout,
                         "Original CLI should support --include-h265 flag")
            self.assertIn("--cleanup", result_original.stdout,
                         "Original CLI should support --cleanup flag")
            
        except Exception as e:
            self.fail(f"Original CLI help failed: {e}")

    def test_both_clis_support_core_arguments(self):
        """
        Test that both CLIs support the same core arguments.
        
        Verifies that key arguments are present in both help outputs.
        """
        # Get help output from both CLIs
        enhanced_help = subprocess.run([
            sys.executable, "-m", "lazy_transcode.cli_enhanced", "--help"
        ], capture_output=True, text=True, cwd=project_root).stdout
        
        original_help = subprocess.run([
            sys.executable, "-m", "lazy_transcode.core.main", "--help"
        ], capture_output=True, text=True, cwd=project_root).stdout
        
        # Core arguments that should be in both
        core_args = [
            "--vmaf-target",
            "--limit", 
            "--dry-run",
            "--debug",
            "--cpu",
            "--parallel",
            "--verify",
            "--include-h265",
            "--cleanup",
            "--preserve-hdr",
            "--encoder",
            "--non-destructive",
            "--no-timestamps"
        ]
        
        for arg in core_args:
            self.assertIn(arg, enhanced_help, 
                         f"Enhanced CLI should support {arg}")
            self.assertIn(arg, original_help,
                         f"Original CLI should support {arg}")

    def test_both_clis_handle_invalid_arguments(self):
        """
        Test that both CLIs handle invalid arguments gracefully.
        """
        # Test enhanced CLI with invalid argument
        result_enhanced = subprocess.run([
            sys.executable, "-m", "lazy_transcode.cli_enhanced", ".", "--invalid-arg"
        ], capture_output=True, text=True, cwd=project_root)
        
        self.assertNotEqual(result_enhanced.returncode, 0,
                          "Enhanced CLI should fail with invalid argument")
        self.assertIn("unrecognized arguments", result_enhanced.stderr,
                     "Enhanced CLI should show error for invalid argument")
        
        # Test original CLI with invalid argument
        result_original = subprocess.run([
            sys.executable, "-m", "lazy_transcode.core.main", ".", "--invalid-arg"
        ], capture_output=True, text=True, cwd=project_root)
        
        self.assertNotEqual(result_original.returncode, 0,
                          "Original CLI should fail with invalid argument")
        self.assertIn("unrecognized arguments", result_original.stderr,
                     "Original CLI should show error for invalid argument")


if __name__ == '__main__':
    unittest.main()
