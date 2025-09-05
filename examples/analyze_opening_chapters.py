#!/usr/bin/env python3
"""
Chapter Opening Analysis for lazy-transcode

Analyzes chapter metadata in anime shows to detect opening patterns
for intelligent VBR clip placement.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

def get_chapter_metadata(video_file: Path) -> Optional[Dict]:
    """Extract chapter metadata from video file using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_chapters', str(video_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting chapters from {video_file}: {e}")
    return None

def find_opening_chapter(chapters: List[Dict]) -> Optional[Tuple[float, float]]:
    """
    Find opening chapter based on common naming patterns.
    
    Returns (start_time, end_time) in seconds if found.
    """
    if not chapters:
        return None
    
    # Common opening patterns (case-insensitive)
    opening_patterns = [
        r'^OP$',           # Exact "OP"
        r'^Opening$',      # Exact "Opening" 
        r'^Intro$',        # Exact "Intro"
        r'^OP\d*$',        # OP with optional number
        r'^Opening\s*\d*$', # Opening with optional number
        r'^Theme$',        # Some shows use "Theme"
        r'.*Opening.*',    # Contains "Opening"
        r'.*Intro.*',      # Contains "Intro"
    ]
    
    for chapter in chapters:
        title = chapter.get('tags', {}).get('title', '').strip()
        if not title:
            continue
            
        for pattern in opening_patterns:
            if re.match(pattern, title, re.IGNORECASE):
                start_time = float(chapter.get('start_time', 0))
                end_time = float(chapter.get('end_time', 0))
                return (start_time, end_time)
    
    return None

def analyze_show_openings(show_path: Path, max_episodes: int = 3) -> Dict:
    """Analyze opening patterns for a single show."""
    result = {
        'show_name': show_path.name,
        'episodes_analyzed': 0,
        'episodes_with_chapters': 0,
        'episodes_with_openings': 0,
        'opening_patterns': [],
        'opening_timings': [],
        'average_opening_start': None,
        'average_opening_duration': None
    }
    
    # Find video files in the show directory
    video_extensions = {'.mkv', '.mp4', '.avi', '.m4v'}
    video_files = []
    
    for root, dirs, files in os.walk(show_path):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(Path(root) / file)
        
        # Only check first few episodes per season
        if len(video_files) >= max_episodes:
            break
    
    video_files = sorted(video_files)[:max_episodes]
    
    opening_starts = []
    opening_durations = []
    
    for video_file in video_files:
        result['episodes_analyzed'] += 1
        print(f"  Analyzing: {video_file.name}")
        
        chapter_data = get_chapter_metadata(video_file)
        if not chapter_data or not chapter_data.get('chapters'):
            continue
            
        result['episodes_with_chapters'] += 1
        chapters = chapter_data['chapters']
        
        # Find opening chapter
        opening_info = find_opening_chapter(chapters)
        if opening_info:
            start_time, end_time = opening_info
            duration = end_time - start_time
            
            result['episodes_with_openings'] += 1
            opening_starts.append(start_time)
            opening_durations.append(duration)
            
            # Store pattern info
            for chapter in chapters:
                title = chapter.get('tags', {}).get('title', '').strip()
                if title and find_opening_chapter([chapter]):
                    result['opening_patterns'].append(title)
                    result['opening_timings'].append({
                        'start': start_time,
                        'end': end_time,
                        'duration': duration,
                        'title': title
                    })
                    break
    
    # Calculate averages
    if opening_starts:
        result['average_opening_start'] = sum(opening_starts) / len(opening_starts)
        result['average_opening_duration'] = sum(opening_durations) / len(opening_durations)
    
    return result

def main():
    shows_path = Path("M:/Shows")
    
    if not shows_path.exists():
        print(f"Shows directory not found: {shows_path}")
        return
    
    # Focus on anime shows that are likely to have openings
    anime_keywords = [
        'demon', 'slayer', 'kimetsu', 'attack', 'titan', 'jujutsu', 'kaisen',
        'one piece', 'naruto', 'bleach', 'dragon ball', 'mob psycho',
        'hero academia', 'tokyo ghoul', 'death note', 'fullmetal',
        'hunter x hunter', 'assassination classroom', 'spy family',
        'chainsaw man', 'fire force', 'black clover'
    ]
    
    # Find anime shows
    anime_shows = []
    for show_dir in shows_path.iterdir():
        if show_dir.is_dir():
            show_name_lower = show_dir.name.lower()
            if any(keyword in show_name_lower for keyword in anime_keywords):
                anime_shows.append(show_dir)
    
    print(f"Found {len(anime_shows)} potential anime shows")
    
    all_results = []
    
    for show_path in anime_shows[:10]:  # Limit to first 10 shows
        print(f"\nAnalyzing: {show_path.name}")
        result = analyze_show_openings(show_path)
        all_results.append(result)
        
        # Print summary for this show
        if result['episodes_with_openings'] > 0:
            print(f"  ✓ Found openings in {result['episodes_with_openings']}/{result['episodes_analyzed']} episodes")
            print(f"  ✓ Average opening start: {result['average_opening_start']:.1f}s")
            print(f"  ✓ Average opening duration: {result['average_opening_duration']:.1f}s")
            print(f"  ✓ Patterns: {set(result['opening_patterns'])}")
        else:
            print(f"  ✗ No openings found in {result['episodes_analyzed']} episodes")
    
    # Overall analysis
    print("\n" + "="*60)
    print("OVERALL ANALYSIS")
    print("="*60)
    
    shows_with_openings = [r for r in all_results if r['episodes_with_openings'] > 0]
    
    if shows_with_openings:
        all_opening_starts = []
        all_opening_durations = []
        all_patterns = []
        
        for result in shows_with_openings:
            if result['average_opening_start']:
                all_opening_starts.append(result['average_opening_start'])
                all_opening_durations.append(result['average_opening_duration'])
                all_patterns.extend(result['opening_patterns'])
        
        print(f"Shows with detected openings: {len(shows_with_openings)}")
        print(f"Total episodes with openings: {sum(r['episodes_with_openings'] for r in shows_with_openings)}")
        
        if all_opening_starts:
            avg_start = sum(all_opening_starts) / len(all_opening_starts)
            avg_duration = sum(all_opening_durations) / len(all_opening_durations)
            
            print(f"Average opening start time: {avg_start:.1f} seconds")
            print(f"Average opening duration: {avg_duration:.1f} seconds")
            print(f"Typical opening range: {avg_start:.0f}s - {avg_start + avg_duration:.0f}s")
        
        print(f"Common opening titles: {set(all_patterns)}")
        
        # Recommendations for VBR optimization
        print("\n" + "="*60)
        print("VBR OPTIMIZATION RECOMMENDATIONS")
        print("="*60)
        
        if all_opening_starts:
            opening_middle = avg_start + (avg_duration / 2)
            print(f"1. Always include a clip around {opening_middle:.0f}s (middle of opening)")
            print(f"2. Opening detection patterns found:")
            for pattern in sorted(set(all_patterns)):
                print(f"   - '{pattern}'")
            print(f"3. Typical opening starts around {avg_start:.0f}s")
            print(f"4. Consider clip at {opening_middle:.0f}s ± 15s for opening coverage")
    
    else:
        print("No shows with chapter-based openings found")
        print("Consider checking for other opening detection methods")
    
    # Save detailed results
    output_file = Path("opening_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
