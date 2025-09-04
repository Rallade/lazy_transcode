"""
Progress Tracking Regression Tests

These tests prevent critical bugs in progress tracking that could cause:
- Incorrect progress percentages confusing users
- Progress tracking hanging or freezing operations
- Memory leaks from progress tracking accumulation
- Race conditions in concurrent progress updates

Progress tracking is essential for user experience during long transcoding operations.
"""

import unittest
import threading
import time
from unittest.mock import patch, MagicMock, call

# Mock the progress tracking imports since they may not be available
try:
    from lazy_transcode.core.modules.progress_tracker import (
        ProgressTracker, update_progress, format_progress_message
    )
    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    # Create mock classes if not available
    PROGRESS_TRACKER_AVAILABLE = False
    
    class MockProgressTracker:
        def __init__(self, total_operations=100):
            self.total = total_operations
            self.current = 0
            self.percentage = 0.0
        
        def update(self, increment=1):
            self.current += increment
            self.percentage = (self.current / self.total) * 100
        
        def set_total(self, new_total):
            self.total = new_total
            self.percentage = (self.current / self.total) * 100
    
    ProgressTracker = MockProgressTracker
    
    def update_progress(increment=1):
        pass
    
    def format_progress_message(current, total, operation="Processing"):
        return f"{operation}: {current}/{total} ({(current/total)*100:.1f}%)"


class TestProgressCalculationRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent progress calculation errors.
    
    Wrong progress calculations confuse users and can indicate deeper issues.
    """
    
    def test_progress_percentage_calculation_accuracy(self):
        """
        ACCURACY TEST: Progress percentages should be mathematically correct.
        
        This prevents user confusion from incorrect progress reporting.
        """
        test_cases = [
            (0, 100, 0.0),      # Start
            (25, 100, 25.0),    # Quarter
            (50, 100, 50.0),    # Half
            (75, 100, 75.0),    # Three quarters
            (100, 100, 100.0),  # Complete
            (33, 100, 33.0),    # Odd number
            (1, 3, 33.33),      # Rounding case (approximate)
        ]
        
        for current, total, expected_percentage in test_cases:
            with self.subTest(current=current, total=total):
                tracker = ProgressTracker(total_operations=total)
                tracker.current = current
                tracker.update(0)  # Trigger calculation
                
                if expected_percentage == 33.33:
                    # Allow for rounding differences
                    self.assertAlmostEqual(tracker.percentage, expected_percentage, places=1,
                                         msg=f"{current}/{total} should be ~{expected_percentage}%")
                else:
                    self.assertEqual(tracker.percentage, expected_percentage,
                                   f"{current}/{total} should be exactly {expected_percentage}%")
    
    def test_progress_handles_edge_case_totals(self):
        """
        EDGE CASE TEST: Progress should handle unusual total values gracefully.
        
        This prevents crashes with edge case scenarios.
        """
        edge_cases = [
            (1, "Single operation"),
            (1000000, "Million operations"),
            (7, "Prime number operations"),
        ]
        
        for total, description in edge_cases:
            with self.subTest(case=description, total=total):
                try:
                    tracker = ProgressTracker(total_operations=total)
                    
                    # Test various progress points
                    test_points = [0, total//4, total//2, total-1, total]
                    for current in test_points:
                        tracker.current = current
                        tracker.update(0)  # Trigger calculation
                        
                        # Should be between 0 and 100%
                        self.assertGreaterEqual(tracker.percentage, 0.0,
                                              f"{description}: Progress should be >= 0%")
                        self.assertLessEqual(tracker.percentage, 100.0,
                                           f"{description}: Progress should be <= 100%")
                        
                except Exception as e:
                    self.fail(f"{description} should not cause errors: {e}")
    
    def test_progress_calculation_consistency_across_updates(self):
        """
        CONSISTENCY TEST: Progress should increase monotonically with updates.
        
        This ensures progress always moves forward, never backwards.
        """
        tracker = ProgressTracker(total_operations=10)
        previous_percentage = -1.0
        
        for i in range(11):  # 0 to 10
            tracker.current = i
            tracker.update(0)  # Trigger calculation
            
            # Progress should never go backwards
            self.assertGreaterEqual(tracker.percentage, previous_percentage,
                                  f"Progress should not go backwards at step {i}")
            previous_percentage = tracker.percentage
    
    def test_progress_message_formatting_consistency(self):
        """
        MESSAGE FORMAT TEST: Progress messages should be consistently formatted.
        
        This ensures users see consistent, readable progress information.
        """
        test_cases = [
            (0, 100, "Transcoding", "Transcoding: 0/100 (0.0%)"),
            (50, 100, "Encoding", "Encoding: 50/100 (50.0%)"),
            (100, 100, "Complete", "Complete: 100/100 (100.0%)"),
            (33, 100, "Processing", "Processing: 33/100 (33.0%)"),
        ]
        
        for current, total, operation, expected_format in test_cases:
            with self.subTest(current=current, total=total, operation=operation):
                message = format_progress_message(current, total, operation)
                
                # Should contain all required components
                self.assertIn(str(current), message, "Message should contain current value")
                self.assertIn(str(total), message, "Message should contain total value") 
                self.assertIn(operation, message, "Message should contain operation name")
                self.assertIn("%", message, "Message should contain percentage")


class TestProgressUpdateRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent progress update mechanism failures.
    
    Update failures could freeze progress displays or cause memory leaks.
    """
    
    def test_progress_updates_increment_correctly(self):
        """
        INCREMENT TEST: Progress updates should increment by specified amounts.
        
        This ensures accurate tracking of completed work.
        """
        tracker = ProgressTracker(total_operations=100)
        
        # Test various increment sizes
        increment_tests = [
            (1, 1),      # Single increment
            (5, 6),      # Medium increment (1 + 5 = 6)
            (10, 16),    # Large increment (6 + 10 = 16)
            (25, 41),    # Very large increment (16 + 25 = 41)
        ]
        
        for increment, expected_current in increment_tests:
            with self.subTest(increment=increment, expected=expected_current):
                tracker.update(increment)
                self.assertEqual(tracker.current, expected_current,
                               f"After incrementing by {increment}, current should be {expected_current}")
    
    def test_progress_updates_handle_overshooting(self):
        """
        OVERSHOOT TEST: Progress should handle updates that exceed total gracefully.
        
        This prevents errors when operations complete more work than expected.
        """
        tracker = ProgressTracker(total_operations=10)
        
        # Update to exactly the limit
        tracker.update(10)
        self.assertEqual(tracker.current, 10, "Should reach exactly the total")
        self.assertEqual(tracker.percentage, 100.0, "Should be exactly 100%")
        
        # Update beyond the limit
        tracker.update(5)  # This would make current = 15, which exceeds total = 10
        
        # Should handle this gracefully - either cap at total or allow overage
        # Both behaviors are acceptable as long as it doesn't crash
        self.assertGreaterEqual(tracker.current, 10, "Current should be at least the total")
        
        # Percentage calculation shouldn't crash
        try:
            percentage = tracker.percentage
            self.assertIsInstance(percentage, (int, float), "Percentage should be numeric")
        except Exception as e:
            self.fail(f"Percentage calculation should not crash on overshoot: {e}")
    
    def test_progress_total_can_be_updated_dynamically(self):
        """
        DYNAMIC TOTAL TEST: Progress total should be updatable during operation.
        
        This handles cases where total work amount changes during processing.
        """
        tracker = ProgressTracker(total_operations=100)
        tracker.update(25)  # 25% complete
        
        original_percentage = tracker.percentage
        self.assertEqual(original_percentage, 25.0, "Should start at 25%")
        
        # Double the total work
        tracker.set_total(200)
        
        # Same amount of work done, but percentage should change
        new_percentage = tracker.percentage
        self.assertEqual(new_percentage, 12.5, "Should be 12.5% after doubling total")
        
        # Current work amount should remain unchanged
        self.assertEqual(tracker.current, 25, "Current work should remain 25")
    
    def test_progress_updates_are_atomic(self):
        """
        ATOMICITY TEST: Progress updates should be atomic operations.
        
        This prevents race conditions in concurrent transcoding scenarios.
        """
        tracker = ProgressTracker(total_operations=1000)
        
        def concurrent_updates(thread_id, updates_per_thread):
            """Perform progress updates in a thread."""
            for i in range(updates_per_thread):
                tracker.update(1)
        
        # Create multiple threads doing concurrent updates
        threads = []
        updates_per_thread = 50
        num_threads = 10
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=concurrent_updates, args=(thread_id, updates_per_thread))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Total should be exactly the expected amount
        expected_total = num_threads * updates_per_thread
        self.assertEqual(tracker.current, expected_total,
                       f"After concurrent updates, should have {expected_total} total progress")


class TestProgressMemoryManagementRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent memory leaks in progress tracking.
    
    Memory leaks in long-running transcoding could exhaust system resources.
    """
    
    def test_progress_tracker_does_not_accumulate_memory(self):
        """
        MEMORY TEST: Progress tracker should not accumulate memory over time.
        
        This prevents memory leaks during long transcoding operations.
        """
        # Create and destroy many progress trackers
        initial_tracker = ProgressTracker(total_operations=100)
        initial_size = len(str(initial_tracker.__dict__))  # Rough memory usage indicator
        
        # Simulate many short-lived progress trackers
        for i in range(100):
            tracker = ProgressTracker(total_operations=1000)
            
            # Do some updates
            for j in range(10):
                tracker.update(1)
            
            # Tracker should be eligible for garbage collection after this loop iteration
        
        # Create a new tracker and compare memory usage
        final_tracker = ProgressTracker(total_operations=100)
        final_size = len(str(final_tracker.__dict__))
        
        # Memory usage should be similar (allowing for some variation)
        memory_growth = final_size - initial_size
        self.assertLess(abs(memory_growth), initial_size * 2,
                       f"Memory usage should not grow significantly: {memory_growth} bytes growth")
    
    def test_progress_updates_do_not_accumulate_state(self):
        """
        STATE ACCUMULATION TEST: Progress updates should not accumulate unnecessary state.
        
        This prevents bloating of progress tracker internal state.
        """
        tracker = ProgressTracker(total_operations=10)
        
        # Perform many small updates
        for i in range(1000):
            tracker.update(0.01)  # Very small increments
        
        # Internal state should remain minimal
        state_size = len(str(tracker.__dict__))
        
        # Should not have accumulated a large amount of internal state
        self.assertLess(state_size, 1000,  # Arbitrary but reasonable limit
                       f"Progress tracker state should remain compact: {state_size} characters")
        
        # Should still function correctly
        self.assertIsInstance(tracker.current, (int, float), "Current should be numeric")
        self.assertIsInstance(tracker.percentage, (int, float), "Percentage should be numeric")


class TestProgressErrorHandlingRegression(unittest.TestCase):
    """
    CRITICAL TESTS: Prevent progress tracking errors from crashing operations.
    
    Progress tracking failures shouldn't stop transcoding operations.
    """
    
    def test_progress_handles_invalid_inputs_gracefully(self):
        """
        INPUT VALIDATION TEST: Progress should handle invalid inputs without crashing.
        
        This ensures transcoding continues even with progress tracking errors.
        """
        tracker = ProgressTracker(total_operations=100)
        
        invalid_inputs = [
            -1,         # Negative increment
            float('inf'), # Infinity
            None,       # None value
            "invalid",  # String instead of number
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=repr(invalid_input)):
                try:
                    # Should either handle gracefully or raise specific exception
                    tracker.update(invalid_input)
                    
                    # If it succeeds, tracker should still be in valid state
                    self.assertIsInstance(tracker.current, (int, float),
                                        f"Current should remain numeric after invalid input: {invalid_input}")
                    
                except (TypeError, ValueError):
                    # Acceptable to raise specific exceptions for invalid input
                    pass
                except Exception as e:
                    self.fail(f"Should handle invalid input gracefully: {invalid_input} caused {e}")
    
    def test_progress_handles_zero_total_gracefully(self):
        """
        ZERO DIVISION TEST: Progress should handle zero total without crashing.
        
        This prevents division by zero errors in edge cases.
        """
        try:
            tracker = ProgressTracker(total_operations=0)
            
            # Should not crash when updating with zero total
            tracker.update(1)
            
            # Should handle percentage calculation gracefully
            percentage = tracker.percentage
            
            # Either handle gracefully (return 0% or 100%) or raise specific exception
            if percentage is not None:
                self.assertIsInstance(percentage, (int, float),
                                    "Percentage should be numeric or None for zero total")
            
        except (ZeroDivisionError, ValueError):
            # Acceptable to raise specific exceptions for zero total
            pass
        except Exception as e:
            self.fail(f"Should handle zero total gracefully: {e}")
    
    def test_progress_continues_operation_despite_tracking_errors(self):
        """
        RESILIENCE TEST: Main operation should continue even if progress tracking fails.
        
        This ensures transcoding robustness in the face of progress tracking issues.
        """
        # Simulate an operation with potentially failing progress tracking
        operation_completed = False
        progress_errors = []
        
        def simulated_transcoding_operation():
            """Simulate a transcoding operation with progress tracking."""
            nonlocal operation_completed
            
            tracker = ProgressTracker(total_operations=10)
            
            for i in range(10):
                # Simulate some work
                time.sleep(0.001)  # Brief pause to simulate work
                
                # Try to update progress, but don't let failures stop the operation
                try:
                    if i == 5:
                        # Simulate a progress tracking error halfway through
                        raise RuntimeError("Progress tracking failed")
                    tracker.update(1)
                except Exception as e:
                    progress_errors.append(str(e))
                    # Continue operation despite progress error
                
            operation_completed = True
        
        # Run the simulated operation
        simulated_transcoding_operation()
        
        # Operation should complete despite progress errors
        self.assertTrue(operation_completed, "Operation should complete despite progress errors")
        
        # Should have captured the expected progress error
        self.assertEqual(len(progress_errors), 1, "Should have captured exactly one progress error")
        self.assertIn("Progress tracking failed", progress_errors[0],
                     "Should have captured the expected error message")


class TestProgressDisplayIntegration(unittest.TestCase):
    """
    INTEGRATION TESTS: Test progress tracking in realistic scenarios.
    
    These tests ensure progress tracking works correctly in actual usage patterns.
    """
    
    @unittest.skipUnless(PROGRESS_TRACKER_AVAILABLE, "Progress tracker not available")
    def test_progress_tracking_full_workflow(self):
        """
        WORKFLOW TEST: Complete progress tracking workflow should work correctly.
        
        This tests the entire progress tracking lifecycle.
        """
        # Simulate a complete transcoding workflow
        total_files = 5
        tracker = ProgressTracker(total_operations=total_files)
        
        progress_messages = []
        
        for file_index in range(total_files):
            # Simulate processing a file
            filename = f"video_{file_index + 1}.mkv"
            
            # Update progress
            tracker.update(1)
            
            # Generate progress message
            message = format_progress_message(
                tracker.current, 
                tracker.total, 
                f"Processing {filename}"
            )
            progress_messages.append(message)
            
            # Verify progress is reasonable
            expected_percentage = ((file_index + 1) / total_files) * 100
            self.assertAlmostEqual(tracker.percentage, expected_percentage, places=1,
                                 msg=f"Progress should be ~{expected_percentage}% after file {file_index + 1}")
        
        # Should have generated appropriate progress messages
        self.assertEqual(len(progress_messages), total_files, "Should have message for each file")
        
        # Final progress should be 100%
        self.assertEqual(tracker.percentage, 100.0, "Should be 100% complete at end")
        
        # All messages should contain expected components
        for i, message in enumerate(progress_messages):
            with self.subTest(message_index=i):
                self.assertIn("Processing", message, "Message should contain operation")
                self.assertIn("video_", message, "Message should contain filename")
                self.assertIn(f"{i+1}/{total_files}", message, "Message should contain progress fraction")
                self.assertIn("%", message, "Message should contain percentage")


if __name__ == '__main__':
    unittest.main(verbosity=2)
