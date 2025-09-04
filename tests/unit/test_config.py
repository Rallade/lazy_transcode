"""Test configuration loading."""

import unittest
import tempfile
from pathlib import Path

from lazy_transcode.config import load_env_file, get_config


class TestConfig(unittest.TestCase):
    
    def test_load_env_file(self):
        """Test loading environment variables from .env file."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('test_root=M:/Shows\n')
            f.write('vmaf_target=95.0\n')
            f.write('# This is a comment\n')
            f.write('debug=true\n')
            temp_path = Path(f.name)
        
        try:
            env_vars = load_env_file(temp_path)
            
            self.assertEqual(env_vars['test_root'], 'M:/Shows')
            self.assertEqual(env_vars['vmaf_target'], '95.0')
            self.assertEqual(env_vars['debug'], 'true')
            self.assertNotIn('# This is a comment', env_vars)
            
        finally:
            temp_path.unlink()
    
    def test_get_config(self):
        """Test getting configuration."""
        config = get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('vmaf_target', config)
        self.assertIn('vmaf_threads', config)
        self.assertIn('sample_duration', config)
        self.assertIn('debug', config)


class TestPackageImport(unittest.TestCase):
    
    def test_package_imports(self):
        """Test that package imports work."""
        import lazy_transcode
        
        # Test basic imports
        self.assertTrue(hasattr(lazy_transcode, 'get_config'))
        self.assertTrue(hasattr(lazy_transcode, 'get_test_root'))
        
        # Test function getters work
        transcode_funcs = lazy_transcode.get_transcode_functions()
        self.assertIsInstance(transcode_funcs, dict)
        
        manager_funcs = lazy_transcode.get_manager_functions()
        self.assertIsInstance(manager_funcs, dict)


if __name__ == '__main__':
    unittest.main()
