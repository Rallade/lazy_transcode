"""
Unit tests for progress monitoring and logging enhancements.

Tests the enhanced progress monitoring functionality that provides
detailed real-time feedback during transcoding operations.
"""

import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from lazy_transcode.core.modules.transcoding_engine import monitor_progress


class TestProgressMonitoring(unittest.TestCase):
    """Test enhanced progress monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.progress_file = Path("test_progress.txt")
        self.callback = Mock()
        self.process = Mock()
    
    @patch('time.time')
    @patch('time.sleep')
    @patch('builtins.print')
    def test_detailed_progress_logging(self, mock_print, mock_sleep, mock_time):
        """Test detailed progress information is logged."""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1001.0, 1002.0]
        
        # Mock process that runs then completes
        self.process.poll.side_effect = [None, 0]  # Running, then done
        
        # Mock progress file content with realistic FFmpeg progress output
        progress_content = """frame=150
fps=29.97
bitrate=2500kbps
total_size=2097152
out_time_us=5000000
speed=1.2x
progress=continue"""
        
        with patch('builtins.open', mock_open(read_data=progress_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Verify callback was called with progress data
        self.assertTrue(self.callback.called)
        
        # Check detailed progress logging was output
        print_calls = [str(call) for call in mock_print.call_args_list]
        progress_logged = any("Progress: Frame 150" in call for call in print_calls)
        fps_logged = any("FPS 29.97" in call for call in print_calls)
        speed_logged = any("Speed 1.2x" in call for call in print_calls)
        bitrate_logged = any("Bitrate 2500kbps" in call for call in print_calls)
        
        self.assertTrue(progress_logged, "Frame progress should be logged")
        self.assertTrue(fps_logged, "FPS should be logged")
        self.assertTrue(speed_logged, "Encoding speed should be logged")
        self.assertTrue(bitrate_logged, "Bitrate should be logged")
    
    @patch('time.time')
    @patch('time.sleep')
    @patch('builtins.print')
    def test_time_conversion_from_microseconds(self, mock_print, mock_sleep, mock_time):
        """Test conversion of FFmpeg microsecond timestamps to readable time."""
        mock_time.return_value = 1000.0
        self.process.poll.side_effect = [None, 0]
        
        # Progress with 1 hour, 30 minutes, 45 seconds in microseconds
        time_us = (1 * 3600 + 30 * 60 + 45) * 1000000  # 5445000000 microseconds
        progress_content = f"""frame=1000
fps=25.0
out_time_us={time_us}
speed=1.0x
progress=continue"""
        
        with patch('builtins.open', mock_open(read_data=progress_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Check that time was converted correctly (01:30:45)
        print_calls = [str(call) for call in mock_print.call_args_list]
        time_logged = any("Time 01:30:45" in call for call in print_calls)
        self.assertTrue(time_logged, "Time should be converted to HH:MM:SS format")
    
    @patch('time.sleep')
    @patch('builtins.print')
    def test_handles_missing_progress_file(self, mock_print, mock_sleep):
        """Test graceful handling when progress file doesn't exist."""
        self.process.poll.side_effect = [None, None, 0]  # Takes a few iterations
        
        # Progress file doesn't exist
        with patch('pathlib.Path.exists', return_value=False):
            monitor_progress(self.process, self.progress_file, self.callback)
        
        # Should not crash and should still call final callback
        final_call = self.callback.call_args_list[-1]
        final_args = final_call[0][0]  # First positional argument
        self.assertEqual(final_args['progress'], 'end')
    
    @patch('time.sleep')
    @patch('builtins.print')
    def test_handles_file_read_errors(self, mock_print, mock_sleep):
        """Test graceful handling of file read errors."""
        self.process.poll.side_effect = [None, 0]
        
        # Mock file operations that raise exceptions
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Should not crash due to file read error
        # Final callback should still be called
        self.assertTrue(self.callback.called)
    
    @patch('time.sleep')
    @patch('builtins.print')
    def test_parses_malformed_progress_data(self, mock_print, mock_sleep):
        """Test handling of malformed progress data."""
        self.process.poll.side_effect = [None, 0]
        
        # Malformed progress data
        malformed_content = """invalid_line_without_equals
frame=abc
=no_key
progress=continue"""
        
        with patch('builtins.open', mock_open(read_data=malformed_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Should not crash and should still process valid lines
        self.assertTrue(self.callback.called)
        
        # Check that it processed the valid progress line
        call_args = self.callback.call_args_list[0][0][0]
        # First callback should still reflect 'continue'
        self.assertEqual(call_args['progress'], 'continue')
    
    @patch('time.sleep')
    @patch('builtins.print')
    def test_completion_message_logged(self, mock_print, mock_sleep):
        """Test that completion message is logged when encoding finishes."""
        self.process.poll.side_effect = [None, 0]
        
        progress_content = "progress=continue"
        
        with patch('builtins.open', mock_open(read_data=progress_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Check that completion message was logged
        print_calls = [str(call) for call in mock_print.call_args_list]
        completion_logged = any("Encoding completed!" in call for call in print_calls)
        self.assertTrue(completion_logged, "Completion message should be logged")
    
    @patch('time.sleep')
    def test_callback_receives_final_end_status(self, mock_sleep):
        """Test that callback receives final 'end' status."""
        self.process.poll.side_effect = [None, 0]
        
        progress_content = "progress=continue"
        
        with patch('builtins.open', mock_open(read_data=progress_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Check final callback was made with 'end' status
        final_call = self.callback.call_args_list[-1]
        final_args = final_call[0][0]
        self.assertEqual(final_args['progress'], 'end')
    
    @patch('time.sleep')
    @patch('builtins.print')
    def test_monitoring_frequency_increased(self, mock_print, mock_sleep):
        """Test that monitoring frequency is 1 second (increased from 0.5s)."""
        self.process.poll.side_effect = [None, 0]
        
        with patch('builtins.open', mock_open(read_data="progress=continue")):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Verify sleep was called with 1.0 second intervals
        mock_sleep.assert_called_with(1.0)
    
    @patch('time.time')
    @patch('time.sleep')
    @patch('builtins.print')
    def test_handles_zero_time_values(self, mock_print, mock_sleep, mock_time):
        """Test handling of zero or invalid time values."""
        mock_time.return_value = 1000.0
        self.process.poll.side_effect = [None, 0]
        
        # Progress with zero time values
        progress_content = """frame=50
fps=0
out_time_us=0
speed=N/A
progress=continue"""
        
        with patch('builtins.open', mock_open(read_data=progress_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Should handle zero values gracefully
        print_calls = [str(call) for call in mock_print.call_args_list]
        time_logged = any("Time 00:00:00" in call for call in print_calls)
        self.assertTrue(time_logged, "Should display 00:00:00 for zero time")
    
    @patch('time.time')
    @patch('time.sleep') 
    @patch('builtins.print')
    def test_multiple_progress_updates(self, mock_print, mock_sleep, mock_time):
        """Test multiple progress updates during encoding."""
        mock_time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0]
        
        # Process runs for multiple iterations
        self.process.poll.side_effect = [None, None, None, 0]
        
        # Different progress values for each iteration
        progress_contents = [
            "frame=100\nfps=30\nprogress=continue",
            "frame=200\nfps=29\nprogress=continue", 
            "frame=300\nfps=28\nprogress=continue"
        ]
        
        call_count = 0
        def mock_open_multiple(*args, **kwargs):
            nonlocal call_count
            content = progress_contents[min(call_count, len(progress_contents) - 1)]
            call_count += 1
            return mock_open(read_data=content)(*args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open_multiple):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Should have multiple progress updates plus final end
        self.assertGreaterEqual(len(self.callback.call_args_list), 3)
        
        # Final call should have 'end' status
        final_call = self.callback.call_args_list[-1][0][0]
        self.assertEqual(final_call['progress'], 'end')


class TestProgressDataParsing(unittest.TestCase):
    """Test progress data parsing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.progress_file = Path("test_progress.txt")
        self.callback = Mock()
        self.process = Mock()
        self.process.poll.side_effect = [None, 0]  # Run once then complete
    
    @patch('time.sleep')
    def test_parses_all_ffmpeg_progress_fields(self, mock_sleep):
        """Test parsing of all standard FFmpeg progress fields."""
        progress_content = """frame=1500
fps=29.97
stream_0_0_q=23.0
bitrate=2500kbps
total_size=10485760
out_time_us=60000000
out_time=00:01:00.000000
dup=0
drop=5
speed=1.2x
progress=continue"""
        
        with patch('builtins.open', mock_open(read_data=progress_content)):
            with patch('pathlib.Path.exists', return_value=True):
                monitor_progress(self.process, self.progress_file, self.callback)
        
        # Verify all fields were parsed and passed to callback
        call_args = self.callback.call_args_list[0][0][0]
        
        expected_fields = {
            'frame': '1500',
            'fps': '29.97', 
            'bitrate': '2500kbps',
            'total_size': '10485760',
            'out_time_us': '60000000',
            'speed': '1.2x',
            'progress': 'continue'
        }
        
        for field, expected_value in expected_fields.items():
            self.assertEqual(call_args.get(field), expected_value)


if __name__ == '__main__':
    unittest.main()
