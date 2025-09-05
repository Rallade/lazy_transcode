"""
Unit tests for the coverage-based VBR clip selection algorithm.

Tests the new algorithm against various scenarios including:
- Opening detection and inclusion
- Coverage percentage targeting
- Edge cases (short videos, no openings)
- Real-world anime file scenarios
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from lazy_transcode.core.modules.optimization.coverage_clips import (
    calculate_coverage_based_clips,
    get_coverage_based_vbr_clip_positions,
    find_opening_chapter,
    get_chapter_metadata
)

class TestCoverageBasedClips(unittest.TestCase):
    """Test the coverage-based VBR clip selection algorithm."""
    
    def test_basic_coverage_calculation(self):
        """Test basic coverage calculation without opening."""
        dummy_file = Path("test_video.mkv")
        
        # Test 24-minute anime episode with 10% target coverage
        duration = 1420.0  # 24 minutes
        ranges, clip_duration = calculate_coverage_based_clips(
            dummy_file, duration, clip_duration=30, target_coverage=0.10
        )
        
        # Should get around 4-5 clips for 10% coverage (120-150s total)
        total_clip_time = sum(clip_len for _, clip_len in ranges)
        actual_coverage = total_clip_time / duration
        
        self.assertGreaterEqual(len(ranges), 2, "Should have at least 2 clips")
        self.assertLessEqual(len(ranges), 8, "Should not have too many clips")
        self.assertGreater(actual_coverage, 0.05, "Coverage should be at least 5%")
        self.assertLess(actual_coverage, 0.20, "Coverage should not exceed 20%")
        
        # Clips should be well-distributed
        for i in range(len(ranges) - 1):
            curr_end = ranges[i][0] + ranges[i][1]
            next_start = ranges[i + 1][0]
            gap = next_start - curr_end
            self.assertGreaterEqual(gap, 0, "Clips should not overlap")
    
    def test_opening_detection_patterns(self):
        """Test opening detection with various chapter patterns."""
        
        # Test cases: (title, should_detect)
        test_cases = [
            ("OP", True),
            ("Opening", True), 
            ("OP1", True),
            ("Opening Theme", True),
            ("Intro", True),  # But would be filtered if at start and short
            ("Cold Open", False),
            ("Scene 1", False),
            ("Credits", False),
            ("Ending", False),
        ]
        
        for title, should_detect in test_cases:
            chapters = [{
                'start_time': 120.0,  # Not at very beginning
                'end_time': 210.0,    # 90 seconds long
                'tags': {'title': title}
            }]
            
            result = find_opening_chapter(chapters)
            
            if should_detect:
                self.assertIsNotNone(result, f"Should detect opening for title: '{title}'")
                if result is not None:
                    start, end, detected_title = result
                    self.assertEqual(detected_title, title)
                    self.assertEqual(start, 120.0)
                    self.assertEqual(end, 210.0)
            else:
                self.assertIsNone(result, f"Should not detect opening for title: '{title}'")
    
    def test_opening_inclusion_in_clips(self):
        """Test that openings are always included in clip selection."""
        
        # Mock chapter metadata with opening
        mock_chapters = [{
            'start_time': 120.0,
            'end_time': 210.0,
            'tags': {'title': 'OP'}
        }]
        
        with patch('lazy_transcode.core.modules.optimization.coverage_clips.get_chapter_metadata', return_value=mock_chapters):
            dummy_file = Path("anime_with_opening.mkv")
            duration = 1420.0  # 24 minutes
            
            ranges, clip_duration = calculate_coverage_based_clips(
                dummy_file, duration, clip_duration=30, target_coverage=0.10
            )
            
            # Opening should be included as full range
            opening_start = 120
            opening_length = 90  # 210 - 120
            
            # Find the opening clip in the ranges
            opening_found = False
            for start_pos, clip_len in ranges:
                if start_pos == opening_start and clip_len == opening_length:
                    opening_found = True
                    break
            
            self.assertTrue(opening_found, "Opening should be included as full clip")
            self.assertGreater(len(ranges), 1, "Should have additional clips beyond opening")
    
    def test_large_opening_coverage_rule(self):
        """Test that openings always get at least 2 clips outside opening."""
        
        # Mock a normal opening (1.5 minutes out of 15 minute video = 10%)
        mock_chapters = [{
            'start_time': 60.0,
            'end_time': 150.0,  # 1.5 minutes long
            'tags': {'title': 'Opening'}
        }]
        
        with patch('lazy_transcode.core.modules.optimization.coverage_clips.get_chapter_metadata', return_value=mock_chapters):
            dummy_file = Path("anime_with_opening.mkv")
            duration = 900.0  # 15 minutes
            
            ranges, clip_duration = calculate_coverage_based_clips(
                dummy_file, duration, clip_duration=30, target_coverage=0.10
            )
            
            # Should have opening clip plus at least 2 outside opening
            opening_start = 60
            opening_length = 90  # 150 - 60
            non_opening_clips = []
            opening_found = False
            
            for start_pos, clip_len in ranges:
                if start_pos == opening_start and clip_len == opening_length:
                    opening_found = True
                else:
                    non_opening_clips.append((start_pos, clip_len))
            
            self.assertTrue(opening_found, "Should include opening")
            self.assertGreaterEqual(len(non_opening_clips), 2, 
                                   "Should always have at least 2 clips outside opening")
    
    def test_short_video_handling(self):
        """Test handling of very short videos."""
        dummy_file = Path("short_video.mkv")
        duration = 45.0  # 45 seconds
        
        ranges, clip_duration = calculate_coverage_based_clips(
            dummy_file, duration, clip_duration=30, target_coverage=0.10
        )
        
        self.assertEqual(len(ranges), 1, "Short video should get exactly 1 clip")
        start_pos, clip_len = ranges[0]
        self.assertGreater(start_pos, 0, "Clip should not be at very beginning")
        self.assertLess(start_pos + clip_len, duration, "Clip should fit within video")
    
    def test_target_coverage_scaling(self):
        """Test that different target coverage percentages scale appropriately."""
        dummy_file = Path("test_video.mkv")
        duration = 1200.0  # 20 minutes
        
        # Test different coverage targets
        coverage_5_ranges, _ = calculate_coverage_based_clips(
            dummy_file, duration, clip_duration=30, target_coverage=0.05
        )
        coverage_10_ranges, _ = calculate_coverage_based_clips(
            dummy_file, duration, clip_duration=30, target_coverage=0.10
        )
        coverage_15_ranges, _ = calculate_coverage_based_clips(
            dummy_file, duration, clip_duration=30, target_coverage=0.15
        )
        
        # Higher coverage should generally mean more clips
        self.assertLessEqual(len(coverage_5_ranges), len(coverage_10_ranges),
                            "10% coverage should have >= clips than 5%")
        self.assertLessEqual(len(coverage_10_ranges), len(coverage_15_ranges),
                            "15% coverage should have >= clips than 10%")
    
    @patch('lazy_transcode.core.modules.optimization.coverage_clips.get_chapter_metadata')
    def test_no_chapters_fallback(self, mock_get_chapters):
        """Test algorithm works correctly when no chapters are present."""
        mock_get_chapters.return_value = None
        
        dummy_file = Path("no_chapters.mkv")
        duration = 1420.0
        
        ranges, clip_duration = calculate_coverage_based_clips(
            dummy_file, duration, clip_duration=30, target_coverage=0.10
        )
        
        # Should still work and provide reasonable coverage
        total_clip_time = sum(clip_len for _, clip_len in ranges)
        actual_coverage = total_clip_time / duration
        
        self.assertGreater(len(ranges), 1, "Should have multiple clips even without chapters")
        self.assertGreater(actual_coverage, 0.05, "Should achieve reasonable coverage")
    
    def test_public_api_function(self):
        """Test the main public API function."""
        dummy_file = Path("api_test.mkv")
        duration = 1420.0
        
        positions = get_coverage_based_vbr_clip_positions(
            dummy_file, duration, clip_duration=30, target_coverage=0.10
        )
        
        self.assertIsInstance(positions, list, "Should return list of positions")
        self.assertGreater(len(positions), 0, "Should return at least one position")
        
        # All positions should be valid integers
        for pos in positions:
            self.assertIsInstance(pos, int, "Positions should be integers")
            self.assertGreater(pos, 0, "Positions should be positive")
            self.assertLess(pos, duration, "Positions should be within video duration")

class TestOpeningDetection(unittest.TestCase):
    """Test opening detection functionality specifically."""
    
    def test_empty_chapters(self):
        """Test with no chapters."""
        result = find_opening_chapter([])
        self.assertIsNone(result)
    
    def test_intro_filtering(self):
        """Test that 'Intro' at very beginning is filtered out."""
        # Short intro at beginning (likely cold open, not opening theme)
        chapters = [{
            'start_time': 0.0,
            'end_time': 30.0,  # Very short
            'tags': {'title': 'Intro'}
        }]
        
        result = find_opening_chapter(chapters)
        self.assertIsNone(result, "Short intro at beginning should be filtered out")
        
        # Longer intro later in video (likely opening theme)
        chapters = [{
            'start_time': 120.0,  # Not at beginning
            'end_time': 210.0,
            'tags': {'title': 'Intro'}
        }]
        
        result = find_opening_chapter(chapters)
        self.assertIsNotNone(result, "Intro later in video should be detected")
    
    def test_chapter_without_title(self):
        """Test chapters without title tags."""
        chapters = [{
            'start_time': 120.0,
            'end_time': 210.0,
            'tags': {}  # No title
        }]
        
        result = find_opening_chapter(chapters)
        self.assertIsNone(result, "Chapter without title should not be detected")

if __name__ == '__main__':
    unittest.main(verbosity=2)
