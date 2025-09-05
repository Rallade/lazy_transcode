"""
Opening-Aware VBR Clip Selection for lazy-transcode

Intelligently selects VBR test clips to always include opening themes
based on chapter metadata analysis.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
from lazy_transcode.utils.logging import get_logger

logger = get_logger("opening_aware_clips")

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

def get_intelligent_clip_positions_with_opening(video_file: Path, duration_seconds: float, 
                                              num_clips: int = 3) -> List[int]:
    """
    Calculate intelligent clip positions that always include opening when possible.
    
    Strategy:
    1. Try to detect opening from chapter metadata
    2. If found, always include one clip in/around the opening
    3. Distribute remaining clips evenly across the content
    4. Fallback to even distribution if no opening found
    """
    duration_int = int(duration_seconds)
    
    # For very short videos, use simple logic
    if duration_int <= 60:
        return [10]
    
    # Try to detect opening
    opening_info = None
    chapters = get_chapter_metadata(video_file)
    if chapters:
        opening_info = find_opening_chapter(chapters)
    
    if opening_info:
        start_time, end_time, title = opening_info
        opening_middle = int((start_time + end_time) / 2)
        
        logger.debug(f"Found opening '{title}' at {start_time:.1f}s-{end_time:.1f}s")
        logger.info(f"Including opening-aware clip at {opening_middle}s")
        
        # Always include opening middle as one clip
        positions = [opening_middle]
        
        # Distribute remaining clips in the non-opening content
        remaining_clips = num_clips - 1
        
        if remaining_clips > 0:
            # Split the non-opening content into segments
            # Prefer content segments over intro/outro
            content_start = max(60, int(end_time + 30))  # Skip 30s after opening
            content_end = duration_int - 120  # Leave 2min at end
            
            if content_end > content_start:
                content_duration = content_end - content_start
                
                if remaining_clips == 1:
                    # Single additional clip in middle of content
                    positions.append(content_start + content_duration // 2)
                else:
                    # Multiple clips distributed across content
                    segment_size = content_duration / (remaining_clips + 1)
                    for i in range(1, remaining_clips + 1):
                        pos = int(content_start + i * segment_size)
                        positions.append(pos)
            else:
                # Fallback: distribute remaining clips before opening
                pre_opening_duration = max(60, int(start_time - 30))
                if pre_opening_duration > 60:
                    segment_size = pre_opening_duration / (remaining_clips + 1)
                    for i in range(1, remaining_clips + 1):
                        pos = int(i * segment_size)
                        positions.append(max(10, pos))
        
        # Sort positions and ensure minimum spacing
        positions = sorted(set(positions))
        
        # Ensure clips are at least 60s apart and within bounds
        final_positions = []
        for pos in positions:
            pos = max(10, min(pos, duration_int - 60))
            if not final_positions or abs(pos - final_positions[-1]) >= 60:
                final_positions.append(pos)
        
        return final_positions[:num_clips]  # Limit to requested number
    
    else:
        # No opening found, use traditional even distribution
        logger.debug("No opening detected, using traditional clip distribution")
        return get_traditional_clip_positions(duration_seconds, num_clips)

def get_traditional_clip_positions(duration_seconds: float, num_clips: int = 3) -> List[int]:
    """Traditional even distribution of clips (fallback method)."""
    duration_int = int(duration_seconds)
    if duration_int <= 60:
        return [10]
    
    positions = []
    segment_size = duration_int / (num_clips + 1)
    
    for i in range(1, num_clips + 1):
        pos = int(i * segment_size)
        pos = min(pos, duration_int - 60)
        positions.append(max(10, pos))
    
    return positions

def analyze_opening_statistics(video_files: List[Path]) -> Dict:
    """Analyze opening patterns across multiple files for debugging."""
    stats = {
        'total_files': len(video_files),
        'files_with_chapters': 0,
        'files_with_openings': 0,
        'opening_positions': [],
        'opening_titles': [],
        'average_opening_start': None,
        'average_opening_duration': None
    }
    
    opening_starts = []
    opening_durations = []
    
    for video_file in video_files:
        chapters = get_chapter_metadata(video_file)
        if not chapters:
            continue
        
        stats['files_with_chapters'] += 1
        
        opening_info = find_opening_chapter(chapters)
        if opening_info:
            start_time, end_time, title = opening_info
            duration = end_time - start_time
            
            stats['files_with_openings'] += 1
            stats['opening_positions'].append((start_time, end_time))
            stats['opening_titles'].append(title)
            
            opening_starts.append(start_time)
            opening_durations.append(duration)
    
    if opening_starts:
        stats['average_opening_start'] = sum(opening_starts) / len(opening_starts)
        stats['average_opening_duration'] = sum(opening_durations) / len(opening_durations)
    
    return stats

# Integration with existing VBR optimizer
def get_opening_aware_vbr_clip_positions(video_file: Path, duration_seconds: float, 
                                       num_clips: Optional[int] = None) -> List[int]:
    """
    Drop-in replacement for get_vbr_clip_positions with opening awareness.
    
    This function can replace the existing get_vbr_clip_positions call in main.py
    """
    # Use auto-scaling logic from main.py if num_clips not specified
    if num_clips is None:
        # Auto-scale clips: 1 clip per 30 minutes, min 2, max 6
        num_clips = max(2, min(6, int(duration_seconds / 1800)))
    
    return get_intelligent_clip_positions_with_opening(video_file, duration_seconds, num_clips)
