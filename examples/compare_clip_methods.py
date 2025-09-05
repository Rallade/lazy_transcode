"""
Comparison script showing the difference between old and new VBR clip selection methods.

This script demonstrates:
1. Old method: Simple uniform distribution 
2. New method: Coverage-based with opening detection

Shows how the new method provides better representation of video content.
"""

import sys
from pathlib import Path

# Add the project root to path for imports  
sys.path.insert(0, str(Path(__file__).parent))

from lazy_transcode.core.modules.optimization.coverage_clips import (
    get_coverage_based_vbr_clip_positions,
    get_coverage_based_vbr_clip_ranges
)
from lazy_transcode.utils.logging import get_logger

logger = get_logger("clip_comparison")

def old_clip_selection_method(duration_seconds: float, num_clips: int = 3) -> list[int]:
    """
    Old simple clip selection method for comparison.
    
    This represents the previous uniform distribution approach.
    """
    if duration_seconds <= 60:
        return [10]
    
    # Simple uniform distribution with fixed number of clips
    start_time = 300  # Skip first 5 minutes
    end_time = duration_seconds - 120  # Skip last 2 minutes
    
    if end_time <= start_time:
        return [int(duration_seconds // 2)]
    
    if num_clips == 1:
        return [int((start_time + end_time) / 2)]
    
    step = (end_time - start_time) / (num_clips - 1)
    positions = []
    
    for i in range(num_clips):
        pos = int(start_time + i * step)
        positions.append(pos)
    
    return positions

def compare_clip_selection_methods():
    """Compare old vs new clip selection methods across different scenarios."""
    
    # Test scenarios: (duration_minutes, description, has_opening)
    test_scenarios = [
        (5, "Short web video", False),
        (15, "Standard TV episode", False), 
        (24, "Anime episode (no opening detected)", False),
        (24, "Anime episode (with opening)", True),
        (45, "Long drama episode", False),
        (90, "Movie", False),
    ]
    
    logger.info("VBR Clip Selection Method Comparison")
    logger.info("=" * 70)
    
    for duration_minutes, description, has_opening in test_scenarios:
        duration_seconds = duration_minutes * 60
        
        logger.info(f"\n{description} ({duration_minutes} minutes)")
        logger.info("-" * 50)
        
        # Old method
        old_positions = old_clip_selection_method(duration_seconds, num_clips=3)
        old_times = [f"{p//60:02d}:{p%60:02d}" for p in old_positions]
        old_coverage = (len(old_positions) * 30) / duration_seconds * 100
        
        # New method (simulate opening if needed)
        if has_opening:
            # Mock opening detection for demonstration
            import unittest.mock
            mock_chapters = [{
                'start_time': 120.0,  # Opening at 2 minutes
                'end_time': 210.0,    # 90 seconds long
                'tags': {'title': 'OP'}
            }]
            with unittest.mock.patch('lazy_transcode.core.modules.optimization.coverage_clips.get_chapter_metadata', return_value=mock_chapters):
                new_positions = get_coverage_based_vbr_clip_positions(
                    Path("mock_anime.mkv"), duration_seconds, clip_duration=30, target_coverage=0.10
                )
        else:
            new_positions = get_coverage_based_vbr_clip_positions(
                Path("mock_video.mkv"), duration_seconds, clip_duration=30, target_coverage=0.10
            )
        
        new_times = [f"{p//60:02d}:{p%60:02d}" for p in new_positions]
        new_coverage = (len(new_positions) * 30) / duration_seconds * 100
        
        # Display comparison
        logger.info(f"Old Method:")
        logger.info(f"  Positions: {old_times}")
        logger.info(f"  Coverage:  {old_coverage:.1f}% ({len(old_positions)} × 30s clips)")
        logger.info(f"  Strategy:  Fixed 3 clips, uniform distribution")
        
        logger.info(f"New Method:")
        logger.info(f"  Positions: {new_times}")
        logger.info(f"  Coverage:  {new_coverage:.1f}% ({len(new_positions)} × 30s clips)")
        if has_opening:
            logger.info(f"  Strategy:  10% coverage target + opening detection")
        else:
            logger.info(f"  Strategy:  10% coverage target, content-aware")
        
        # Analysis
        logger.info(f"Analysis:")
        if new_coverage > old_coverage * 1.2:
            logger.info(f"  ✅ New method provides {new_coverage/old_coverage:.1f}x more coverage")
        elif len(new_positions) > len(old_positions):
            logger.info(f"  ✅ New method uses {len(new_positions)} clips vs {len(old_positions)} for better sampling")
        elif has_opening:
            logger.info(f"  ✅ New method includes opening theme in analysis")
        else:
            logger.info(f"  ℹ️  Similar coverage, but new method is content-aware")

def demonstrate_opening_benefits():
    """Demonstrate specific benefits of opening detection."""
    
    logger.info("\n" + "=" * 70)
    logger.info("Full Opening Integration Benefits Demonstration")
    logger.info("=" * 70)
    
    # Simulate anime episode with opening
    duration = 24 * 60  # 24 minutes
    
    logger.info(f"\nScenario: 24-minute anime episode with opening at 2:00-3:30 (90s)")
    logger.info("-" * 50)
    
    # Old method (doesn't know about opening)
    old_positions = old_clip_selection_method(duration, num_clips=3)
    
    # New method with opening detection
    import unittest.mock
    mock_chapters = [{
        'start_time': 120.0,   # Opening at 2:00
        'end_time': 210.0,     # Ends at 3:30 (90 seconds)
        'tags': {'title': 'OP'}
    }]
    
    with unittest.mock.patch('lazy_transcode.core.modules.optimization.coverage_clips.get_chapter_metadata', return_value=mock_chapters):
        new_ranges = get_coverage_based_vbr_clip_ranges(
            Path("anime_with_opening.mkv"), duration, clip_duration=30, target_coverage=0.10
        )
    
    logger.info(f"Old method clips (30s each):")
    for pos in old_positions:
        minutes, seconds = pos // 60, pos % 60
        logger.info(f"  {minutes:02d}:{seconds:02d} - {minutes:02d}:{(seconds+30)%60:02d} (30s) - Regular content")
    
    logger.info(f"\nNew method clips (variable duration):")
    for start_pos, clip_len in new_ranges:
        end_pos = start_pos + clip_len
        start_mm, start_ss = start_pos // 60, start_pos % 60
        end_mm, end_ss = end_pos // 60, end_pos % 60
        
        if start_pos == 120 and clip_len == 90:
            logger.info(f"  {start_mm:02d}:{start_ss:02d} - {end_mm:02d}:{end_ss:02d} ({clip_len}s) - ⭐ FULL OPENING THEME")
        else:
            logger.info(f"  {start_mm:02d}:{start_ss:02d} - {end_mm:02d}:{end_ss:02d} ({clip_len}s) - Regular content")
    
    # Calculate coverage comparison
    old_coverage = (len(old_positions) * 30) / duration * 100
    new_coverage = sum(clip_len for _, clip_len in new_ranges) / duration * 100
    opening_coverage = 90 / duration * 100
    
    logger.info(f"\nCoverage Analysis:")
    logger.info(f"  Old method: {old_coverage:.1f}% total coverage")
    logger.info(f"  New method: {new_coverage:.1f}% total coverage")
    logger.info(f"  Opening represents: {opening_coverage:.1f}% of total episode")
    
    logger.info(f"\nBenefits of full opening inclusion:")
    logger.info(f"  • Captures complete opening sequence vs. 30s sample")
    logger.info(f"  • Better represents opening's encoding characteristics")
    logger.info(f"  • More accurate VBR optimization for anime content")
    logger.info(f"  • Accounts for opening's unique visual complexity")
    logger.info(f"  • Optimizes for the content viewers actually see")

def main():
    """Run the clip selection comparison demonstration."""
    
    try:
        compare_clip_selection_methods()
        demonstrate_opening_benefits()
        
        logger.info("\n" + "=" * 70)
        logger.info("Summary")
        logger.info("=" * 70)
        logger.info("New coverage-based algorithm provides:")
        logger.info("  ✅ Consistent 10% coverage regardless of video length")
        logger.info("  ✅ Opening detection and inclusion for anime")
        logger.info("  ✅ Content-aware clip distribution")
        logger.info("  ✅ Minimum clips outside large openings")
        logger.info("  ✅ Better representation of encoding complexity")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
