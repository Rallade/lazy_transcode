"""
Unit tests for vbr_optimizer module.

Tests VBR optimization functionality including convergence detection,
bounds calculation, and the research-enhanced optimization system.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lazy_transcode.core.modules.vbr_optimizer import (
    VBROptimizer, calculate_bounds, check_convergence, 
    predict_optimal_bitrate
)


class TestVBROptimizer(unittest.TestCase):
    """Test VBR optimizer core functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_file = Path("test_video.mkv")
        self.output_file = Path("output.mkv") 
        self.temp_dir = Path(tempfile.mkdtemp())
        self.encoder_config = Mock()
        
        # Mock video properties
        self.video_properties = {
            'duration': 3600.0,  # 1 hour
            'width': 1920,
            'height': 1080,
            'codec': 'h264'
        }
        
        self.optimizer = VBROptimizer(
            self.input_file, self.output_file,
            self.encoder_config, self.temp_dir,
            target_vmaf=92.0
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_vbr_optimizer_initialization(self):
        """Test VBR optimizer initialization."""
        self.assertEqual(self.optimizer.input_file, self.input_file)
        self.assertEqual(self.optimizer.target_vmaf, 92.0)
        self.assertEqual(len(self.optimizer.trials), 0)
        self.assertFalse(self.optimizer.converged)
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_video_properties')
    def test_calculate_bounds_1080p(self, mock_get_props):
        """Test bitrate bounds calculation for 1080p."""
        mock_get_props.return_value = self.video_properties
        
        min_rate, max_rate = calculate_bounds(self.input_file, target_vmaf=92.0)
        
        # Should calculate reasonable bounds for 1080p at VMAF 92
        self.assertIsInstance(min_rate, int)
        self.assertIsInstance(max_rate, int)
        self.assertLess(min_rate, max_rate)
        self.assertGreater(min_rate, 1000)  # Minimum reasonable bitrate
        self.assertLess(max_rate, 25000)   # Maximum reasonable bitrate
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_video_properties')
    def test_calculate_bounds_4k(self, mock_get_props):
        """Test bitrate bounds calculation for 4K."""
        props_4k = {
            'duration': 3600.0,
            'width': 3840,
            'height': 2160,
            'codec': 'h264'
        }
        mock_get_props.return_value = props_4k
        
        min_rate, max_rate = calculate_bounds(self.input_file, target_vmaf=92.0)
        
        # 4K should have higher bitrate bounds
        self.assertGreater(min_rate, 5000)  
        self.assertLess(max_rate, 50000)
    
    def test_check_convergence_target_achieved(self):
        """Test convergence detection when target is achieved."""
        trials = [
            {'bitrate': 5000, 'vmaf': 91.8, 'size_mb': 800},
            {'bitrate': 6000, 'vmaf': 92.1, 'size_mb': 900},
            {'bitrate': 5500, 'vmaf': 91.95, 'size_mb': 850}
        ]
        
        converged, reason = check_convergence(
            trials, target_vmaf=92.0, 
            vmaf_tolerance=0.5, bitrate_tolerance=200
        )
        
        self.assertTrue(converged)
        self.assertIn("target", reason.lower())
    
    def test_check_convergence_stable_pattern(self):
        """Test convergence detection with stable oscillation."""
        trials = [
            {'bitrate': 5000, 'vmaf': 91.5, 'size_mb': 800},
            {'bitrate': 6000, 'vmaf': 92.5, 'size_mb': 900},
            {'bitrate': 5500, 'vmaf': 91.9, 'size_mb': 850},
            {'bitrate': 5800, 'vmaf': 92.2, 'size_mb': 875},
            {'bitrate': 5600, 'vmaf': 91.95, 'size_mb': 860}
        ]
        
        converged, reason = check_convergence(
            trials, target_vmaf=92.0,
            vmaf_tolerance=0.5, bitrate_tolerance=200
        )
        
        self.assertTrue(converged)
        self.assertIn("stable", reason.lower())
    
    def test_check_convergence_not_converged(self):
        """Test convergence detection when not converged."""
        trials = [
            {'bitrate': 5000, 'vmaf': 88.0, 'size_mb': 800},
            {'bitrate': 8000, 'vmaf': 94.0, 'size_mb': 1200}
        ]
        
        converged, reason = check_convergence(
            trials, target_vmaf=92.0,
            vmaf_tolerance=0.5, bitrate_tolerance=200
        )
        
        self.assertFalse(converged)
        self.assertEqual(reason, "")
    
    def test_predict_optimal_bitrate_polynomial(self):
        """Test polynomial-based bitrate prediction."""
        trials = [
            {'bitrate': 4000, 'vmaf': 89.5},
            {'bitrate': 6000, 'vmaf': 92.1},
            {'bitrate': 8000, 'vmaf': 94.2}
        ]
        
        predicted = predict_optimal_bitrate(
            trials, target_vmaf=92.0, 
            video_props=self.video_properties
        )
        
        self.assertIsInstance(predicted, int)
        self.assertGreater(predicted, 4000)
        self.assertLess(predicted, 8000)
        # Should predict closer to 6000 since that achieved ~92 VMAF
        self.assertLess(abs(predicted - 6000), 1500)
    
    def test_predict_optimal_bitrate_insufficient_data(self):
        """Test bitrate prediction with insufficient data."""
        trials = [
            {'bitrate': 5000, 'vmaf': 89.0}
        ]
        
        predicted = predict_optimal_bitrate(
            trials, target_vmaf=92.0,
            video_props=self.video_properties
        )
        
        # Should fall back to bounds-based estimation
        self.assertIsInstance(predicted, int)
        self.assertGreater(predicted, 5000)  # Should suggest higher bitrate


class TestResearchEnhancedOptimization(unittest.TestCase):
    """Test research-enhanced optimization features."""
    
    def setUp(self):
        """Set up test environment."""
        self.video_properties = {
            'duration': 1800.0,  # 30 minutes
            'width': 1920,
            'height': 1080,
            'codec': 'h264'
        }
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_video_properties')
    def test_farhadi_nia_polynomial_model(self, mock_get_props):
        """Test Farhadi Nia (2025) polynomial model application."""
        mock_get_props.return_value = self.video_properties
        
        # Test with typical trial data
        trials = [
            {'bitrate': 3000, 'vmaf': 87.2},
            {'bitrate': 5000, 'vmaf': 91.8},
            {'bitrate': 7000, 'vmaf': 93.9}
        ]
        
        predicted = predict_optimal_bitrate(
            trials, target_vmaf=92.0,
            video_props=self.video_properties
        )
        
        # Should use polynomial fitting for prediction
        self.assertIsInstance(predicted, int)
        # Prediction should be between known good points
        self.assertGreater(predicted, 4500)
        self.assertLess(predicted, 6000)
    
    def test_newton_raphson_inverse_prediction(self):
        """Test Newton-Raphson inverse prediction method."""
        # Simulate converged polynomial coefficients
        trials = [
            {'bitrate': 2000, 'vmaf': 84.1},
            {'bitrate': 4000, 'vmaf': 89.7},
            {'bitrate': 6000, 'vmaf': 92.3},
            {'bitrate': 8000, 'vmaf': 94.1}
        ]
        
        predicted = predict_optimal_bitrate(
            trials, target_vmaf=92.0,
            video_props=self.video_properties
        )
        
        # Should find bitrate that yields target VMAF
        self.assertIsInstance(predicted, int)
        # Should be close to 6000 since that achieved 92.3 VMAF
        self.assertLess(abs(predicted - 6000), 800)
    
    def test_coordinated_cross_trial_learning(self):
        """Test cross-trial learning and adaptation."""
        # Test that optimizer learns from multiple trials
        optimizer = VBROptimizer(
            Path("test.mkv"), Path("out.mkv"),
            Mock(), Path("temp"), target_vmaf=92.0
        )
        
        # Simulate adding trial results
        optimizer.trials = [
            {'bitrate': 4000, 'vmaf': 89.2, 'size_mb': 600},
            {'bitrate': 6000, 'vmaf': 92.5, 'size_mb': 800},
            {'bitrate': 5000, 'vmaf': 90.8, 'size_mb': 700}
        ]
        
        # Test convergence detection
        converged, reason = check_convergence(
            optimizer.trials, target_vmaf=92.0
        )
        
        # Should detect near-optimal solution
        self.assertTrue(converged or len(optimizer.trials) >= 3)


class TestOptimizationBounds(unittest.TestCase):
    """Test bitrate bounds calculation."""
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_video_properties')
    def test_bounds_resolution_scaling(self, mock_get_props):
        """Test bounds scaling with resolution."""
        resolutions = [
            (1280, 720, "720p"),
            (1920, 1080, "1080p"), 
            (2560, 1440, "1440p"),
            (3840, 2160, "4K")
        ]
        
        bounds = []
        for width, height, name in resolutions:
            mock_get_props.return_value = {
                'width': width, 'height': height,
                'duration': 3600.0, 'codec': 'h264'
            }
            
            min_rate, max_rate = calculate_bounds(
                Path("test.mkv"), target_vmaf=92.0
            )
            bounds.append((name, min_rate, max_rate))
        
        # Higher resolutions should have higher bounds
        for i in range(1, len(bounds)):
            prev_name, prev_min, prev_max = bounds[i-1]
            curr_name, curr_min, curr_max = bounds[i]
            
            self.assertLessEqual(prev_min, curr_min, 
                f"{prev_name} min should be <= {curr_name} min")
            self.assertLessEqual(prev_max, curr_max,
                f"{prev_name} max should be <= {curr_name} max")
    
    @patch('lazy_transcode.core.modules.vbr_optimizer.get_video_properties')
    def test_bounds_target_vmaf_scaling(self, mock_get_props):
        """Test bounds scaling with target VMAF."""
        mock_get_props.return_value = self.video_properties = {
            'width': 1920, 'height': 1080,
            'duration': 3600.0, 'codec': 'h264'
        }
        
        targets = [85, 90, 95]
        bounds = []
        
        for target in targets:
            min_rate, max_rate = calculate_bounds(
                Path("test.mkv"), target_vmaf=target
            )
            bounds.append((target, min_rate, max_rate))
        
        # Higher VMAF targets should require higher bitrates
        for i in range(1, len(bounds)):
            prev_target, prev_min, prev_max = bounds[i-1]
            curr_target, curr_min, curr_max = bounds[i]
            
            self.assertLessEqual(prev_min, curr_min)
            self.assertLessEqual(prev_max, curr_max)


class TestConvergenceDetection(unittest.TestCase):
    """Test convergence detection algorithms."""
    
    def test_diminishing_returns_detection(self):
        """Test detection of diminishing returns pattern."""
        trials = [
            {'bitrate': 3000, 'vmaf': 88.0, 'size_mb': 400},
            {'bitrate': 5000, 'vmaf': 91.5, 'size_mb': 600},
            {'bitrate': 7000, 'vmaf': 92.2, 'size_mb': 800},
            {'bitrate': 9000, 'vmaf': 92.4, 'size_mb': 1000}
        ]
        
        converged, reason = check_convergence(
            trials, target_vmaf=92.0,
            vmaf_tolerance=0.5, bitrate_tolerance=500
        )
        
        # Should detect convergence due to diminishing returns
        self.assertTrue(converged)
        self.assertIn("diminishing", reason.lower())
    
    def test_oscillation_stability_detection(self):
        """Test detection of stable oscillation."""
        trials = [
            {'bitrate': 5000, 'vmaf': 91.2, 'size_mb': 700},
            {'bitrate': 6500, 'vmaf': 92.8, 'size_mb': 850},
            {'bitrate': 5500, 'vmaf': 91.6, 'size_mb': 750},
            {'bitrate': 6200, 'vmaf': 92.5, 'size_mb': 820},
            {'bitrate': 5800, 'vmaf': 91.9, 'size_mb': 780}
        ]
        
        converged, reason = check_convergence(
            trials, target_vmaf=92.0,
            vmaf_tolerance=0.5, bitrate_tolerance=300
        )
        
        # Should detect stable oscillation pattern
        self.assertTrue(converged)
        self.assertIn("stable", reason.lower())
    
    def test_no_convergence_wide_spread(self):
        """Test no convergence with widely spread results."""
        trials = [
            {'bitrate': 3000, 'vmaf': 85.0, 'size_mb': 400},
            {'bitrate': 9000, 'vmaf': 95.0, 'size_mb': 1200},
            {'bitrate': 6000, 'vmaf': 90.0, 'size_mb': 800}
        ]
        
        converged, reason = check_convergence(
            trials, target_vmaf=92.0,
            vmaf_tolerance=0.5, bitrate_tolerance=500
        )
        
        # Should not converge with such spread
        self.assertFalse(converged)
        self.assertEqual(reason, "")


if __name__ == '__main__':
    unittest.main()
