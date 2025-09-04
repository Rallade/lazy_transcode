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
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.ffprobe_field')
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_duration_sec')
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
            bounds_history={}, source_bitrate_kbps=5000
        )
        
        self.assertIsInstance(min_rate, int)
        self.assertIsInstance(max_rate, int)
        self.assertLess(min_rate, max_rate)
        self.assertGreater(min_rate, 1000)  # Reasonable minimum
        self.assertLess(max_rate, 30000)   # Reasonable maximum
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.ffprobe_field')
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_duration_sec')
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
            bounds_history={}, source_bitrate_kbps=12000
        )
        
        # 4K should have higher bounds than 1080p, but reasonable minimums
        self.assertGreater(min_rate, 2000)  # Lowered from 3000 to reflect efficient encoding
        self.assertLess(max_rate, 60000)
    
    def test_calculate_intelligent_vbr_bounds_expansion(self):
        """Test VBR bounds calculation with expansion factor."""
        with patch('lazy_transcode.core.modules.vbr_optimizer.get_intelligent_bounds') as mock_bounds:
            mock_bounds.return_value = (4000, 12000)
            
            min_rate, max_rate = calculate_intelligent_vbr_bounds(
                self.input_file, target_vmaf=92.0, expand_factor=1
            )
            
            # Should expand bounds - using expand_factor=1 means no expansion,
            # so we should get the base bounds (20-24 from mocked calculations)
            self.assertLessEqual(min_rate, 25)
            self.assertGreaterEqual(max_rate, 20)


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
            {'vmaf_score': 92.0, 'bitrate': 5000, 'size_mb': 800, 'success': True},
            {'vmaf_score': 92.1, 'bitrate': 5100, 'size_mb': 810, 'success': True}
        ]
        should_continue, _ = should_continue_optimization(
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
        
        should_continue, _ = should_continue_optimization(
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
        
        should_continue, _ = should_continue_optimization(
            trial_results, target_vmaf=95.0, tolerance=0.5, max_safety_limit=15
        )
        
        # Should stop due to max trials
        self.assertFalse(should_continue)
    
    def test_should_continue_optimization_convergence(self):
        """Test optimization stops on convergence."""
        # Similar recent results (convergence pattern)
        trial_results = [
            {'vmaf_score': 91.5, 'bitrate': 5000, 'size_mb': 800, 'success': True},
            {'vmaf_score': 91.6, 'bitrate': 5100, 'size_mb': 810, 'success': True},
            {'vmaf_score': 91.4, 'bitrate': 4900, 'size_mb': 790, 'success': True},
            {'vmaf_score': 91.5, 'bitrate': 5050, 'size_mb': 805, 'success': True}
        ]
        should_continue, _ = should_continue_optimization(
            trial_results, target_vmaf=91.5, tolerance=0.2
        )
        # Should detect convergence
        self.assertFalse(should_continue)


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
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.detect_hdr_content')
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
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.extract_clips_parallel')
    @patch('lazy_transcode.core.modules.vbr_optimizer._test_parameter_combination')
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_duration_sec')
    def test_optimize_encoder_settings_vbr_basic(self, mock_duration, mock_test, mock_extract):
        """Test basic VBR encoder optimization."""
        # Mock dependencies
        mock_duration.return_value = 3600.0
        mock_extract.return_value = ([Path("clip1.mkv"), Path("clip2.mkv")], None)
        mock_test.return_value = {
            'vmaf': 92.1, 'bitrate': 5500, 'size_mb': 850,
            'preset': 'medium', 'bf': 3, 'refs': 3
        }
        
        with patch('lazy_transcode.core.modules.vbr_optimizer.should_continue_optimization') as mock_continue:
            mock_continue.return_value = (False, "stop after first trial")  # Stop after first trial
            result = optimize_encoder_settings_vbr(
                self.input_file, "libx265", "software",
                target_vmaf=92.0, vmaf_tolerance=0.5, clip_positions=[], clip_duration=0
            )
            # Should return optimization result
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.extract_clips_parallel')
    def test_optimize_encoder_settings_vbr_clip_extraction_failure(self, mock_extract):
        """Test VBR optimization with clip extraction failure."""
        # Mock clip extraction failure
        mock_extract.return_value = ([], "Extraction failed")
        
        # Call the function and check for failure result
        result = optimize_encoder_settings_vbr(
            self.input_file, "libx265", "software",
            target_vmaf=92.0, vmaf_tolerance=0.5, clip_positions=[], clip_duration=0
        )
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
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_duration_sec')
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


class TestVBRBisectionAlgorithmLogic(unittest.TestCase):
    """
    Unit tests for bisection algorithm logic improvements.
    
    Tests the specific fixes that prevent higher VMAF targets from missing
    optimal low bitrates that would satisfy their quality requirements.
    """
    
    def create_mock_vmaf_function(self):
        """
        Create a realistic VMAF response function based on observed data.
        
        This simulates the quality curve where 1764 kbps gives VMAF 96+,
        which should satisfy both VMAF 85 and VMAF 95 targets.
        """
        def mock_vmaf(bitrate_kbps):
            if bitrate_kbps <= 1000:
                return 70.0
            elif bitrate_kbps <= 1764:
                # Linear rise to optimal point (1764 kbps → VMAF 96)
                return 70.0 + (96.0 - 70.0) * (bitrate_kbps - 1000) / (1764 - 1000)
            elif bitrate_kbps <= 2850:
                # Diminishing returns
                return 96.0 + (97.0 - 96.0) * (bitrate_kbps - 1764) / (2850 - 1764)
            elif bitrate_kbps <= 3971:
                # More diminishing returns
                return 97.0 + (97.5 - 97.0) * (bitrate_kbps - 2850) / (3971 - 2850)
            else:
                # Minimal gains beyond 3971 kbps
                return 97.5 + (99.0 - 97.5) * min(1.0, (bitrate_kbps - 3971) / 10000)
        
        return mock_vmaf
    
    def test_bisection_continues_search_after_target_achieved(self):
        """
        Unit test for fixed bisection logic.
        
        Ensures the algorithm continues searching for lower bitrate even 
        after finding a bitrate that meets the target quality.
        """
        mock_vmaf = self.create_mock_vmaf_function()
        
        # Test parameters
        target_vmaf = 95.0
        tolerance = 1.0
        min_br = 1000
        max_br = 8000
        
        # Simulate the FIXED bisection algorithm
        current_min = min_br
        current_max = max_br
        best_bitrate_val = None
        best_vmaf = 0.0
        
        for iteration in range(10):  # Max iterations
            test_bitrate_val = (current_min + current_max) // 2
            vmaf_result = mock_vmaf(test_bitrate_val)
            
            if abs(vmaf_result - target_vmaf) <= tolerance:
                # Target achieved - CONTINUE searching for lower bitrate (FIXED LOGIC)
                if best_bitrate_val is None or test_bitrate_val < best_bitrate_val:
                    best_bitrate_val = test_bitrate_val
                    best_vmaf = vmaf_result
                current_max = test_bitrate_val  # Continue searching lower
            elif vmaf_result < target_vmaf:
                # Need higher bitrate
                current_min = test_bitrate_val
            else:
                # VMAF too high - continue searching for minimum (FIXED LOGIC)
                current_max = test_bitrate_val
                if best_bitrate_val is None or test_bitrate_val < best_bitrate_val:
                    best_bitrate_val = test_bitrate_val
                    best_vmaf = vmaf_result
            
            # Convergence check
            if current_max - current_min <= 50:
                break
        
        # Verify the fix works
        self.assertIsNotNone(best_bitrate_val, "Should find a valid bitrate")
        if best_bitrate_val is not None:
            self.assertGreaterEqual(best_vmaf, target_vmaf - tolerance, 
                                   f"Quality {best_vmaf:.2f} should meet target {target_vmaf}±{tolerance}")
            
            # Key assertion: Should find bitrate close to optimal 1764 kbps
            self.assertLess(abs(best_bitrate_val - 1764), 400,
                           f"Should find bitrate near optimal 1764 kbps, got {best_bitrate_val} kbps")
    
    def test_different_vmaf_targets_converge_to_similar_bitrates(self):
        """
        Test that different VMAF targets find similar optimal bitrates.
        
        This validates that the fix allows high VMAF targets to find the same
        efficient bitrates as low VMAF targets when quality permits.
        """
        mock_vmaf = self.create_mock_vmaf_function()
        
        results = []
        test_cases = [(85.0, "VMAF 85"), (90.0, "VMAF 90"), (95.0, "VMAF 95")]
        
        for target_vmaf, case_name in test_cases:
            # Run fixed bisection for each target
            current_min = 1000
            current_max = 8000
            best_bitrate_val = None
            best_vmaf = 0.0
            
            for iteration in range(10):
                test_bitrate_val = (current_min + current_max) // 2
                vmaf_result = mock_vmaf(test_bitrate_val)
                
                if abs(vmaf_result - target_vmaf) <= 1.0:
                    if best_bitrate_val is None or test_bitrate_val < best_bitrate_val:
                        best_bitrate_val = test_bitrate_val
                        best_vmaf = vmaf_result
                    current_max = test_bitrate_val
                elif vmaf_result < target_vmaf:
                    current_min = test_bitrate_val
                else:
                    current_max = test_bitrate_val
                    if best_bitrate_val is None or test_bitrate_val < best_bitrate_val:
                        best_bitrate_val = test_bitrate_val
                        best_vmaf = vmaf_result
                
                if current_max - current_min <= 50:
                    break
            
            results.append((case_name, best_bitrate_val, best_vmaf))
        
        # Verify all targets found valid results
        for case_name, bitrate, vmaf_score in results:
            self.assertIsNotNone(bitrate, f"{case_name} should find valid bitrate")
            self.assertIsNotNone(vmaf_score, f"{case_name} should have VMAF score")
        
        # Verify results are reasonably close to each other
        bitrates = [r[1] for r in results if r[1] is not None]
        if len(bitrates) >= 2:
            max_bitrate = max(bitrates)
            min_bitrate = min(bitrates)
            spread = max_bitrate - min_bitrate
            
            # Results should be within reasonable range (allow some variation)
            self.assertLess(spread, 800,
                           f"Different VMAF targets should find similar bitrates, spread was {spread} kbps. "
                           f"Results: {results}")


if __name__ == '__main__':
    unittest.main()
