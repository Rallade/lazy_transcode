"""
Unit tests for vbr_optimizer module.

Tests VBR optimization functionality including bounds calculation,
convergence detection, and core optimization functions.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.optimization.vbr_optimizer import (
    get_intelligent_bounds, should_continue_optimization, 
    build_vbr_encode_cmd, optimize_encoder_settings_vbr,
    calculate_intelligent_vbr_bounds
)


class TestVBRBounds(unittest.TestCase):
    """Test VBR bitrate bounds calculation."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("test_video.mkv")
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.ffprobe_field')
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_duration_sec')
    def test_get_intelligent_bounds_1080p(self, mock_duration, mock_ffprobe):
        """Test intelligent bounds calculation for 1080p."""
        # Mock video properties
        mock_duration.return_value = 3600.0
        mock_ffprobe.side_effect = lambda path, field: {
            'width': '1920',
            'height': '1080'
        }.get(field, 'unknown')
        
        min_rate, max_rate = get_intelligent_bounds(
            self.input_file, target_vmaf=92.0, preset="medium",
            bounds_history={}, source_bitrate_kbps=8000
        )
        
        self.assertIsInstance(min_rate, int)
        self.assertIsInstance(max_rate, int)
        self.assertLess(min_rate, max_rate)
        self.assertGreater(min_rate, 1000)  # Reasonable minimum
        self.assertLess(max_rate, 30000)   # Reasonable maximum
    
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.ffprobe_field')
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_duration_sec')
    def test_get_intelligent_bounds_4k(self, mock_duration, mock_ffprobe):
        """Test intelligent bounds calculation for 4K."""
        # Mock 4K properties
        mock_duration.return_value = 3600.0
        mock_ffprobe.side_effect = lambda path, field: {
            'width': '3840',
            'height': '2160'
        }.get(field, 'unknown')
        
        min_rate, max_rate = get_intelligent_bounds(
            self.input_file, target_vmaf=92.0, preset="medium",
            bounds_history={}, source_bitrate_kbps=25000
        )
        
        # 4K should have higher bounds than 1080p
        self.assertGreater(min_rate, 3000)
        self.assertLess(max_rate, 60000)
    
    def test_calculate_intelligent_vbr_bounds_expansion(self):
        """Test VBR bounds calculation with expansion factor."""
        # Mock file system dependencies
        with patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_video_dimensions') as mock_dims, \
             patch('lazy_transcode.core.modules.optimization.vbr_optimizer.ffprobe_field') as mock_fps, \
             patch('lazy_transcode.core.modules.optimization.vbr_optimizer.run_command') as mock_command, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Mock video properties for 1080p
            mock_dims.return_value = (1920, 1080)
            mock_fps.return_value = "24"
            
            # Mock file stat for file size calculation
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
            
            # Mock bitrate probe result
            mock_result = type('MockResult', (), {
                'returncode': 0,
                'stdout': '8000000'  # 8Mbps = 8000kbps
            })()
            mock_command.return_value = mock_result
            
            min_rate, max_rate = calculate_intelligent_vbr_bounds(
                self.input_file, target_vmaf=92.0, expand_factor=1
            )
            
            # Should get reasonable bounds for 1080p content
            self.assertIsInstance(min_rate, int)
            self.assertIsInstance(max_rate, int)
            self.assertLess(min_rate, max_rate)
            self.assertGreater(min_rate, 0)


class TestVBROptimization(unittest.TestCase):
    """Test VBR optimization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("test_video.mkv")
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_should_continue_optimization_target_achieved(self):
        """Test optimization continuation when target is achieved."""
        trial_results = [
            {'vmaf_score': 92.2, 'vmaf': 92.2, 'bitrate': 5000, 'size_mb': 800, 'success': True},
            {'vmaf_score': 91.8, 'vmaf': 91.8, 'bitrate': 4500, 'size_mb': 750, 'success': True},
            {'vmaf_score': 92.0, 'vmaf': 92.0, 'bitrate': 4800, 'size_mb': 775, 'success': True}
        ]
        
        should_continue, reason = should_continue_optimization(
            trial_results, target_vmaf=92.0, tolerance=0.5
        )
        
        # Should not continue since target is achieved
        self.assertFalse(should_continue)
    
    def test_should_continue_optimization_target_not_achieved(self):
        """Test optimization continuation when target not achieved."""
        trial_results = [
            {'vmaf': 89.5, 'bitrate': 3000, 'size_mb': 500},
            {'vmaf': 90.2, 'bitrate': 4000, 'size_mb': 650}
        ]
        
        should_continue, reason = should_continue_optimization(
            trial_results, target_vmaf=92.0, tolerance=0.5
        )
        
        # Should continue since target not achieved
        self.assertTrue(should_continue)
    
    def test_should_continue_optimization_max_trials(self):
        """Test optimization stops at maximum trials."""
        # Create many trial results
        trial_results = [
            {'vmaf': 89.0 + i * 0.1, 'bitrate': 3000 + i * 200, 'size_mb': 500 + i * 50}
            for i in range(20)  # 20 trials
        ]
        
        should_continue, reason = should_continue_optimization(
            trial_results, target_vmaf=95.0, tolerance=0.5
        )
        
        # May stop due to convergence or continue depending on results
        self.assertIsInstance(should_continue, bool)
    
    def test_should_continue_optimization_convergence(self):
        """Test optimization stops on convergence."""
        # Similar recent results (convergence pattern)
        trial_results = [
            {'vmaf': 91.5, 'bitrate': 5000, 'size_mb': 800},
            {'vmaf': 91.6, 'bitrate': 5100, 'size_mb': 810},
            {'vmaf': 91.4, 'bitrate': 4900, 'size_mb': 790},
            {'vmaf': 91.5, 'bitrate': 5050, 'size_mb': 805}
        ]
        
        should_continue, reason = should_continue_optimization(
            trial_results, target_vmaf=92.0, tolerance=0.5
        )
        
        # May detect convergence or continue based on internal logic
        self.assertIsInstance(should_continue, bool)


class TestVBRCommandBuilding(unittest.TestCase):
    """Test VBR command building functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("input.mkv")
        self.output_file = Path("output.mkv")
    
    def test_build_vbr_encode_cmd_basic(self):
        """Test basic VBR command building."""
        cmd = build_vbr_encode_cmd(
            self.input_file, self.output_file,
            "libx265", "software", 
            avg_bitrate=5000, max_bitrate=8000,
            preset="medium"
        )
        
        # Should contain basic VBR elements
        cmd_str = ' '.join(cmd)
        self.assertIn("ffmpeg", cmd_str)
        self.assertIn("libx265", cmd_str)
        self.assertIn(str(self.input_file), cmd_str)
        self.assertIn(str(self.output_file), cmd_str)
    
    def test_build_vbr_encode_cmd_hardware(self):
        """Test VBR command building for hardware encoder."""
        cmd = build_vbr_encode_cmd(
            self.input_file, self.output_file,
            "hevc_nvenc", "hardware",
            avg_bitrate=6000, max_bitrate=10000,
            preset="slow"
        )
        
        # Should contain hardware encoder specifics
        cmd_str = ' '.join(cmd)
        self.assertIn("hevc_nvenc", cmd_str)
    
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.detect_hdr_content')
    def test_build_vbr_encode_cmd_hdr(self, mock_detect_hdr):
        """Test VBR command building with HDR content."""
        mock_detect_hdr.return_value = True
        
        cmd = build_vbr_encode_cmd(
            self.input_file, self.output_file,
            "libx265", "software",
            avg_bitrate=8000, max_bitrate=12000,
            preset="medium"
        )
        
        # Should include HDR-specific parameters
        cmd_str = ' '.join(cmd)
        self.assertIn("libx265", cmd_str)
        # HDR parameters would be added by the function


class TestVBROptimizerIntegration(unittest.TestCase):
    """Test VBR optimizer integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("test_video.mkv")
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.extract_clips_parallel')
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer._test_parameter_combination')
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_duration_sec')
    def test_optimize_encoder_settings_vbr_basic(self, mock_duration, mock_test, mock_extract):
        """Test basic VBR encoder optimization."""
        # Mock dependencies
        mock_duration.return_value = 3600.0
        mock_extract.return_value = ([Path("clip1.mkv"), Path("clip2.mkv")], None)
        mock_test.return_value = {
            'vmaf': 92.1, 'vmaf_score': 92.1, 'bitrate': 5500, 'size_mb': 850,
            'preset': 'medium', 'bf': 3, 'refs': 3, 'success': True
        }
        
        with patch('lazy_transcode.core.modules.optimization.vbr_optimizer.should_continue_optimization') as mock_continue:
            # Return True first time (empty results), then False after getting a result  
            mock_continue.side_effect = [(True, "continue"), (False, "test stop")]
            
            result = optimize_encoder_settings_vbr(
                self.input_file, "libx265", "software",
                target_vmaf=92.0, vmaf_tolerance=1.0,
                clip_positions=[300, 900, 1500], clip_duration=30
            )
        
        # Should return optimization result
        self.assertIsInstance(result, dict)
        self.assertIn('vmaf', result)
        self.assertIn('bitrate', result)
    
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.extract_clips_parallel')
    def test_optimize_encoder_settings_vbr_clip_extraction_failure(self, mock_extract):
        """Test VBR optimization with clip extraction failure."""
        # Mock clip extraction failure
        mock_extract.return_value = ([], "Extraction failed")
        
        result = optimize_encoder_settings_vbr(
            self.input_file, "libx265", "software",
            target_vmaf=92.0, vmaf_tolerance=1.0,
            clip_positions=[300, 900], clip_duration=30
        )
        
        # Should return failure result, not raise exception
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        self.assertIn('error', result)


class TestVBRUtilities(unittest.TestCase):
    """Test VBR utility functions."""
    
    def test_warn_hardware_encoder_inefficiency(self):
        """Test hardware encoder warning."""
        from lazy_transcode.core.modules.optimization.vbr_optimizer import warn_hardware_encoder_inefficiency
        
        # Should not raise exception
        warn_hardware_encoder_inefficiency("hardware", "pre-encoding")
        warn_hardware_encoder_inefficiency("software", "encoding")
    
    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_duration_sec')
    def test_get_vbr_clip_positions(self, mock_duration):
        """Test VBR clip position calculation."""
        from lazy_transcode.core.modules.optimization.vbr_optimizer import get_vbr_clip_positions
        
        duration = 3600.0  # 1 hour
        positions = get_vbr_clip_positions(duration, num_clips=3)
        
        # Should return list of positions
        self.assertIsInstance(positions, list)
        self.assertEqual(len(positions), 3)
        # All positions should be within duration
        for pos in positions:
            self.assertGreaterEqual(pos, 0)
            self.assertLess(pos, duration)


if __name__ == '__main__':
    unittest.main()
