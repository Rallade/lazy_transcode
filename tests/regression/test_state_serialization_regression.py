"""
Regression tests for state management serialization issues.

These tests ensure that state management properly serializes all data types
and prevents "Object of type WindowsPath is not JSON serializable" errors.
"""

import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

class TestStateSerializationRegression(unittest.TestCase):
    """
    STATE SERIALIZATION TESTS: Ensure all state data can be JSON serialized.
    
    These tests prevent regression where Path objects or other non-serializable
    types are stored in state causing JSON serialization failures.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.test_input = self.base_path / "test_video.mkv"
        self.test_input.touch()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_vbr_optimization_state_serializes_without_error(self):
        """
        VBR STATE SERIALIZATION TEST: VBROptimizationState should be JSON serializable.
        
        This prevents regression where clips_result containing Path objects
        caused "Object of type WindowsPath is not JSON serializable" errors.
        """
        from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import VBROptimizationState
        
        # Create state with typical data including clips
        state_manager = VBROptimizationState(
            input_file=self.test_input,
            target_vmaf=95.0,
            vmaf_tolerance=1.0,
            encoder="libx265",
            encoder_type="software"
        )
        
        # Simulate clips extraction result with Path objects (the problematic case)
        mock_clips = [
            self.base_path / "clip_0_300.mkv",
            self.base_path / "clip_1_900.mkv", 
            self.base_path / "clip_2_1500.mkv"
        ]
        
        # This should update state with clips_result containing Path objects
        # The system should convert these to strings for JSON serialization
        state_manager.update_step('clips_extracted', {
            'clips_extracted': True,
            'clip_positions': [300, 900, 1500],
            'clips_result': mock_clips  # This contains Path objects
        })
        
        # Test that state can be serialized to JSON without error
        try:
            # This should not raise "Object of type WindowsPath is not JSON serializable"
            state_manager.save_state()
            
            # Verify the state file was created and contains valid JSON
            self.assertTrue(state_manager.state_file.exists(), 
                          "State file should have been created")
            
            # Verify we can load the JSON back
            with open(state_manager.state_file, 'r') as f:
                loaded_state = json.load(f)
            
            # Verify clips_result was converted to strings
            self.assertIn('clips_result', loaded_state)
            clips_result = loaded_state['clips_result']
            
            # Should be list of strings, not Path objects
            self.assertIsInstance(clips_result, list)
            for clip_path in clips_result:
                self.assertIsInstance(clip_path, str, 
                                    f"Clip path should be string, got {type(clip_path)}")
                
        except TypeError as e:
            if "not JSON serializable" in str(e):
                self.fail(f"State serialization failed with JSON error: {e}")
            else:
                raise

    def test_state_with_complex_data_serializes_correctly(self):
        """
        COMPLEX STATE TEST: State with various data types should serialize properly.
        
        This ensures that optimization state with bounds, progress data,
        and results all serialize without type errors.
        """
        from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import VBROptimizationState
        
        state_manager = VBROptimizationState(
            input_file=self.test_input,
            target_vmaf=97.5,
            vmaf_tolerance=0.8,
            encoder="hevc_nvenc", 
            encoder_type="hardware"
        )
        
        # Add complex state data that might cause serialization issues
        complex_data = {
            'bounds': {'min_bitrate': 1000, 'max_bitrate': 8000},
            'clips_extracted': True,
            'clip_positions': [300, 900, 1500],
            'clips_result': [str(self.base_path / f"clip_{i}.mkv") for i in range(3)],  # Pre-converted strings
            'optimization_progress': {
                'trial_1': {'bitrate': 4000, 'vmaf': 94.5},
                'trial_2': {'bitrate': 5000, 'vmaf': 96.2}
            },
            'best_result': {
                'bitrate': 4500,
                'vmaf': 95.1,
                'file_size': 1024 * 1024 * 500  # 500MB
            },
            'timestamps': {
                'start_time': 1693856400.123,
                'clip_extraction_time': 1693856450.456
            }
        }
        
        # Update state with complex data
        state_manager.update_step('optimization_complete', complex_data)
        
        # Should serialize without error
        try:
            state_manager.save_state()
            
            # Verify round-trip serialization
            with open(state_manager.state_file, 'r') as f:
                loaded_state = json.load(f)
            
            # Verify all data types are JSON-compatible
            self.assertEqual(loaded_state['target_vmaf'], 97.5)
            self.assertEqual(loaded_state['encoder'], "hevc_nvenc")
            self.assertIsInstance(loaded_state['bounds'], dict)
            self.assertIsInstance(loaded_state['clips_result'], list)
            self.assertIsInstance(loaded_state['optimization_progress'], dict)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Complex state data should serialize without error: {e}")

    def test_path_objects_automatically_converted_to_strings(self):
        """
        PATH CONVERSION TEST: Path objects should be automatically converted to strings.
        
        This specifically tests the fix for the WindowsPath serialization issue
        by ensuring Path objects in state data are converted to strings.
        """
        from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import VBROptimizationState
        
        state_manager = VBROptimizationState(
            input_file=self.test_input,
            target_vmaf=95.0,
            vmaf_tolerance=1.0,
            encoder="libx265",
            encoder_type="software"
        )
        
        # Create some Path objects that would cause serialization failure
        path_objects = [
            self.base_path / "test_clip_1.mkv",
            self.base_path / "test_clip_2.mkv"
        ]
        
        # Simulate the exact scenario that caused the original bug
        state_manager.update_step('clips_extracted', {
            'clips_extracted': True,
            'clip_positions': [300, 900],
            'clips_result': path_objects  # This should trigger Path->string conversion
        })
        
        # Verify that the state now contains strings, not Path objects
        clips_result = state_manager.state.get('clips_result', [])
        for clip in clips_result:
            self.assertIsInstance(clip, str, 
                                f"clips_result should contain strings, found {type(clip)}")
            # Verify it's a valid path string
            self.assertTrue(clip.endswith('.mkv'), 
                          f"Converted path should end with .mkv: {clip}")

    def test_state_serialization_handles_edge_cases(self):
        """
        EDGE CASE TEST: State serialization should handle edge cases gracefully.
        
        This tests serialization with None values, empty lists, and mixed data types.
        """
        from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import VBROptimizationState
        
        state_manager = VBROptimizationState(
            input_file=self.test_input,
            target_vmaf=95.0,
            vmaf_tolerance=1.0,
            encoder="libx265",
            encoder_type="software"
        )
        
        # Test edge cases that might break serialization
        edge_case_data = {
            'empty_list': [],
            'none_value': None,
            'mixed_clips': [
                str(self.base_path / "string_clip.mkv"),  # String
                None,  # None value
            ],
            'nested_data': {
                'level1': {
                    'level2': [1, 2, "three", None]
                }
            }
        }
        
        state_manager.update_step('edge_case_test', edge_case_data)
        
        # Should handle edge cases without error
        try:
            state_manager.save_state()
            
            # Verify JSON round-trip
            with open(state_manager.state_file, 'r') as f:
                loaded_state = json.load(f)
            
            self.assertEqual(loaded_state['empty_list'], [])
            self.assertIsNone(loaded_state['none_value'])
            self.assertIsInstance(loaded_state['nested_data'], dict)
            
        except Exception as e:
            self.fail(f"Edge case serialization should not fail: {e}")

    @patch('lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced.logger')
    def test_serialization_error_is_logged_gracefully(self, mock_logger):
        """
        ERROR HANDLING TEST: Serialization errors should be logged, not crash.
        
        This ensures that if somehow non-serializable data gets into state,
        the error is handled gracefully with logging.
        """
        from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import VBROptimizationState
        
        state_manager = VBROptimizationState(
            input_file=self.test_input,
            target_vmaf=95.0,
            vmaf_tolerance=1.0,
            encoder="libx265",
            encoder_type="software"
        )
        
        # Manually inject non-serializable data to test error handling
        # (This bypasses the normal Path->string conversion)
        state_manager.state['problematic_data'] = object()  # Not JSON serializable
        
        # Should not crash, should log warning
        state_manager.save_state()
        
        # Verify that a warning was logged
        mock_logger.warn.assert_called()
        warning_call = mock_logger.warn.call_args[0][0]
        self.assertIn("Failed to save state", warning_call)

if __name__ == '__main__':
    unittest.main()
