"""
Advanced Coverage-Based VBR Clip Selection for lazy-transcode

New algorithm that:
1. Always includes opening if present
2. Scales clips to 30s duration 
3. Targets 10% coverage of original file
4. Ensures at least 2 clips outside opening if opening > 10% coverage
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
from lazy_transcode.utils.logging import get_logger

logger = get_logger("coverage_clips")

def get_chapter_metadata(video_file: Path) -> Optional[List[Dict]]:
    """Extract chapter metadata from video file using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_chapters', str(video_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get('chapters', [])
    except Exception as e:
        logger.debug(f"Error getting chapters from {video_file}: {e}")
    return None

def find_opening_chapter(chapters: List[Dict]) -> Optional[Tuple[float, float, str]]:
    """
    Find opening chapter based on common naming patterns.
    
    Returns (start_time, end_time, title) in seconds if found.
    """
    if not chapters:
        return None
    
    # Common opening patterns (case-insensitive)
    opening_patterns = [
        r'^OP$',               # Exact "OP"
        r'^Opening$',          # Exact "Opening" 
        r'^Intro$',            # Exact "Intro" (but only if not at start)
        r'^OP\s*\d*$',         # OP with optional number
        r'^Opening\s*\d*$',    # Opening with optional number
        r'^Theme$',            # Some shows use "Theme"
        r'.*Opening.*',        # Contains "Opening"
        r'^Title.*',           # Some use "Title"
    ]
    
    for chapter in chapters:
        title = chapter.get('tags', {}).get('title', '').strip()
        if not title:
            continue
            
        start_time = float(chapter.get('start_time', 0))
        end_time = float(chapter.get('end_time', 0))
        
        # Special handling for "Intro" - skip if it's at the very beginning
        # and very short (likely a cold open, not the opening theme)
        if title.lower() == 'intro' and start_time < 30 and (end_time - start_time) < 60:
            continue
            
        for pattern in opening_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                return (start_time, end_time, title)
    
    return None

def calculate_coverage_based_clips(video_file: Path, duration_seconds: float, 
                                 clip_duration: int = 30, 
                                 target_coverage: float = 0.10) -> Tuple[List[Tuple[int, int]], int]:
    """
    Calculate clip positions using coverage-based algorithm.
    
    Algorithm:
    1. Always include opening if present (full opening duration)
    2. Use 30s clips by default for other clips
    3. Target 10% coverage of original file
    4. Ensure at least 2 clips outside opening if opening > 10% coverage
    
    Args:
        video_file: Path to video file
        duration_seconds: Video duration in seconds
        clip_duration: Duration of each clip in seconds (default: 30)
        target_coverage: Target coverage as fraction (default: 0.10 = 10%)
        
    Returns:
        Tuple of (clip_ranges, default_clip_duration)
        clip_ranges: List of (start_time, duration) tuples
    """
def calculate_coverage_based_clips(video_file: Path, duration_seconds: float, 
                                 clip_duration: int = 30, 
                                 target_coverage: float = 0.10) -> Tuple[List[Tuple[int, int]], int]:
    """
    Calculate clip positions using coverage-based algorithm.
    
    Algorithm:
    1. Always include opening if present (full opening duration)
    2. Use 30s clips by default for other clips
    3. Target 10% coverage of original file
    4. Ensure at least 2 clips outside opening if opening > 10% coverage
    
    Args:
        video_file: Path to video file
        duration_seconds: Video duration in seconds
        clip_duration: Duration of each clip in seconds (default: 30)
        target_coverage: Target coverage as fraction (default: 0.10 = 10%)
        
    Returns:
        Tuple of (clip_ranges, default_clip_duration)
        clip_ranges: List of (start_time, duration) tuples
    """
    duration_int = int(duration_seconds)
    
    # For very short videos, use simple logic
    if duration_int <= 60:
        start_pos = 10
        clip_len = min(clip_duration, duration_int - 20)
        return ([(start_pos, clip_len)], clip_duration)
    
    # Calculate target total clip time based on coverage
    target_total_clip_time = duration_seconds * target_coverage
    
    # Try to detect opening
    opening_info = None
    chapters = get_chapter_metadata(video_file)
    if chapters:
        opening_info = find_opening_chapter(chapters)
    
    clip_ranges = []
    opening_clip_time = 0
    
    # Step 1: Handle opening if present (use full opening duration)
    if opening_info:
        start_time, end_time, title = opening_info
        opening_duration = end_time - start_time
        opening_start = int(start_time)
        opening_length = int(opening_duration)
        
        # Always include full opening
        clip_ranges.append((opening_start, opening_length))
        opening_clip_time = opening_duration
        
        opening_coverage = opening_duration / duration_seconds
        
        logger.info(f"Found opening '{title}' at {start_time:.1f}s-{end_time:.1f}s")
        logger.info(f"Opening represents {opening_coverage:.1%} of total duration")
        logger.info(f"Including full opening clip: {opening_start}s for {opening_length}s")
        
        # Calculate remaining clip time needed
        remaining_clip_time = target_total_clip_time - opening_clip_time
        
        # Always ensure at least 2 clips outside opening for better representation
        min_outside_clips = 2
        min_remaining_time = min_outside_clips * clip_duration
        remaining_clip_time = max(remaining_clip_time, min_remaining_time)
        logger.info(f"Ensuring at least {min_outside_clips} clips outside opening for balanced representation")
    else:
        logger.debug("No opening detected")
        remaining_clip_time = target_total_clip_time
    
    # Step 2: Calculate number of additional clips needed
    additional_clips_needed = max(0, int(remaining_clip_time / clip_duration))
    
    # Ensure minimum number of clips for good coverage
    if not opening_info:
        additional_clips_needed = max(2, additional_clips_needed)  # At least 2 clips if no opening
    else:
        additional_clips_needed = max(1, additional_clips_needed)  # At least 1 more clip if opening present
    
    # Step 3: Distribute additional clips in non-opening content
    if additional_clips_needed > 0:
        # Define content regions (avoiding opening and end credits)
        content_regions = []
        
        if opening_info:
            start_time, end_time, _ = opening_info
            
            # Pre-opening content (if substantial)
            if start_time > 90:  # At least 1.5 minutes before opening
                content_regions.append((60, int(start_time - 30)))
            
            # Post-opening content (main content)
            post_opening_start = int(end_time + 30)  # 30s buffer after opening
            post_opening_end = duration_int - 120    # Leave 2min at end for credits
            
            if post_opening_end > post_opening_start:
                content_regions.append((post_opening_start, post_opening_end))
        else:
            # No opening detected, use entire content (avoiding intro/outro)
            content_start = 60  # Skip first minute
            content_end = duration_int - 120  # Leave 2min at end
            if content_end > content_start:
                content_regions.append((content_start, content_end))
        
        # Distribute clips across content regions
        clips_added = 0
        
        for region_start, region_end in content_regions:
            region_duration = region_end - region_start
            if region_duration < 60:  # Skip regions too small
                continue
            
            # Calculate clips for this region based on its proportion
            total_content_duration = sum(end - start for start, end in content_regions)
            region_proportion = region_duration / total_content_duration if total_content_duration > 0 else 1.0
            
            clips_for_region = max(1, int(additional_clips_needed * region_proportion))
            clips_for_region = min(clips_for_region, additional_clips_needed - clips_added)
            
            if clips_for_region > 0:
                # Evenly distribute clips in this region
                if clips_for_region == 1:
                    # Single clip in middle of region
                    pos = region_start + region_duration // 2
                    clip_ranges.append((pos, clip_duration))
                    clips_added += 1
                else:
                    # Multiple clips evenly distributed
                    segment_size = region_duration / (clips_for_region + 1)
                    for i in range(1, clips_for_region + 1):
                        pos = int(region_start + i * segment_size)
                        clip_ranges.append((pos, clip_duration))
                        clips_added += 1
            
            if clips_added >= additional_clips_needed:
                break
        
        # If we still need more clips, add them evenly in the largest region
        while clips_added < additional_clips_needed and content_regions:
            largest_region = max(content_regions, key=lambda r: r[1] - r[0])
            region_start, region_end = largest_region
            
            # Add clip in middle of largest unused space
            pos = region_start + (region_end - region_start) // 2
            clip_ranges.append((pos, clip_duration))
            clips_added += 1
            
            # Split the region to avoid clustering
            mid_point = (region_start + region_end) // 2
            content_regions.remove(largest_region)
            if mid_point - region_start > 60:
                content_regions.append((region_start, mid_point - 30))
            if region_end - mid_point > 60:
                content_regions.append((mid_point + 30, region_end))
    
    # Step 4: Sort and validate ranges
    clip_ranges = sorted(clip_ranges, key=lambda x: x[0])
    
    # Ensure clips are within bounds and have minimum spacing
    final_ranges = []
    for start_pos, clip_len in clip_ranges:
        start_pos = max(10, min(start_pos, duration_int - clip_len - 10))
        
        # Ensure minimum spacing between clips (based on previous clip end)
        if final_ranges:
            prev_end = final_ranges[-1][0] + final_ranges[-1][1]
            if start_pos < prev_end + 10:  # 10 second minimum gap
                start_pos = prev_end + 10
                # Make sure we still fit in the video
                if start_pos + clip_len > duration_int - 10:
                    continue  # Skip this clip if it doesn't fit
        
        final_ranges.append((start_pos, clip_len))
    
    # Step 5: Calculate actual coverage achieved
    total_clip_time = sum(clip_len for _, clip_len in final_ranges)
    actual_coverage = total_clip_time / duration_seconds
    
    logger.info(f"Coverage-based clip selection complete:")
    logger.info(f"  Target coverage: {target_coverage:.1%}")
    logger.info(f"  Actual coverage: {actual_coverage:.1%} ({total_clip_time}s / {duration_seconds:.0f}s)")
    logger.info(f"  Clips selected: {len(final_ranges)} clips")
    
    # Log clip details
    for i, (start_pos, clip_len) in enumerate(final_ranges):
        end_pos = start_pos + clip_len
        logger.info(f"    Clip {i+1}: {start_pos}s-{end_pos}s ({clip_len}s duration)")
    
    return final_ranges, clip_duration

def get_coverage_based_vbr_clip_positions(video_file: Path, duration_seconds: float, 
                                        clip_duration: int = 30,
                                        target_coverage: float = 0.10) -> List[int]:
    """
    Get VBR clip positions using coverage-based algorithm.
    
    This is the main entry point for the new algorithm.
    Returns start positions for backward compatibility with existing VBR code.
    
    Args:
        video_file: Path to video file
        duration_seconds: Video duration in seconds  
        clip_duration: Duration of each clip in seconds (default: 30)
        target_coverage: Target coverage as fraction (default: 0.10 = 10%)
        
    Returns:
        List of clip start positions in seconds (for backward compatibility)
    """
    ranges, _ = calculate_coverage_based_clips(
        video_file, duration_seconds, clip_duration, target_coverage
    )
    # Return just the start positions for backward compatibility
    return [start_pos for start_pos, _ in ranges]

def get_coverage_based_vbr_clip_ranges(video_file: Path, duration_seconds: float, 
                                     clip_duration: int = 30,
                                     target_coverage: float = 0.10) -> List[Tuple[int, int]]:
    """
    Get VBR clip ranges using coverage-based algorithm.
    
    This returns full (start, duration) tuples for advanced usage.
    
    Args:
        video_file: Path to video file
        duration_seconds: Video duration in seconds  
        clip_duration: Duration of each clip in seconds (default: 30)
        target_coverage: Target coverage as fraction (default: 0.10 = 10%)
        
    Returns:
        List of (start_position, duration) tuples in seconds
    """
    ranges, _ = calculate_coverage_based_clips(
        video_file, duration_seconds, clip_duration, target_coverage
    )
    return ranges

def analyze_coverage_algorithm(video_files: List[Path], clip_duration: int = 30,
                             target_coverage: float = 0.10) -> Dict:
    """
    Analyze the coverage algorithm across multiple files for validation.
    
    Returns statistics about coverage, clip counts, opening detection, etc.
    """
    stats = {
        'total_files': len(video_files),
        'files_analyzed': 0,
        'files_with_openings': 0,
        'coverage_stats': {
            'target': target_coverage,
            'achieved': [],
            'clip_counts': [],
            'total_clip_times': []
        },
        'opening_stats': {
            'detected': 0,
            'positions': [],
            'coverage_percentages': []
        }
    }
    
    for video_file in video_files:
        try:
            # Get duration (would normally use get_duration_sec)
            duration = 1420.0  # Placeholder - in real use would detect duration
            
            ranges, actual_clip_duration = calculate_coverage_based_clips(
                video_file, duration, clip_duration, target_coverage
            )
            
            # Calculate actual coverage
            total_clip_time = sum(clip_len for _, clip_len in ranges)
            actual_coverage = total_clip_time / duration
            
            stats['files_analyzed'] += 1
            stats['coverage_stats']['achieved'].append(actual_coverage)
            stats['coverage_stats']['clip_counts'].append(len(ranges))
            stats['coverage_stats']['total_clip_times'].append(total_clip_time)
            
            # Check for opening detection
            chapters = get_chapter_metadata(video_file)
            if chapters:
                opening_info = find_opening_chapter(chapters)
                if opening_info:
                    stats['files_with_openings'] += 1
                    stats['opening_stats']['detected'] += 1
                    
                    start_time, end_time, _ = opening_info
                    opening_coverage = (end_time - start_time) / duration
                    stats['opening_stats']['coverage_percentages'].append(opening_coverage)
                    stats['opening_stats']['positions'].append((start_time, end_time))
            
        except Exception as e:
            logger.debug(f"Error analyzing {video_file}: {e}")
    
    # Calculate summary statistics
    if stats['coverage_stats']['achieved']:
        achieved = stats['coverage_stats']['achieved']
        stats['coverage_stats']['average_achieved'] = sum(achieved) / len(achieved)
        stats['coverage_stats']['min_achieved'] = min(achieved)
        stats['coverage_stats']['max_achieved'] = max(achieved)
    
    return stats
