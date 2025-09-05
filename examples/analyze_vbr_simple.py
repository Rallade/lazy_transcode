#!/usr/bin/env python3
"""
Simplified VBR variance analysis focusing on file sizes and metadata differences.
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
import re

def run_ffprobe_basic(file_path: Path) -> Dict:
    """Get basic file information using ffprobe with longer timeout."""
    info = {
        'path': str(file_path),
        'filename': file_path.name,
        'size_bytes': file_path.stat().st_size,
        'size_gb': file_path.stat().st_size / (1024**3),
    }
    
    # Extract VBR bitrate from filename
    match = re.search(r'vbr_(\d+)k', file_path.name)
    if match:
        info['filename_bitrate_kbps'] = int(match.group(1))
    
    try:
        # Get duration with longer timeout
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(file_path)
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and result.stdout.strip():
            info['duration_sec'] = float(result.stdout.strip())
            
            # Calculate bitrate from file size and duration
            file_size_bits = info['size_bytes'] * 8
            info['calculated_bitrate_kbps'] = int(file_size_bits / info['duration_sec'] / 1000)
    
    except Exception as e:
        print(f"Error getting duration for {file_path.name}: {e}")
    
    return info

def analyze_simple_variance():
    """Perform simplified variance analysis."""
    base_path = Path(r"M:\Shows\Demon Slayer- Kimetsu no Yaiba\Season 1")
    
    # Find original file
    original_file = None
    for file in base_path.glob("*.mkv"):
        if "vbr" not in file.name.lower() and "E01" in file.name:
            original_file = file
            break
    
    if not original_file:
        print("Could not find original file")
        return
    
    print(f"Original file: {original_file.name}")
    original_info = run_ffprobe_basic(original_file)
    print(f"Original size: {original_info['size_gb']:.2f}GB")
    if 'duration_sec' in original_info:
        print(f"Duration: {original_info['duration_sec']:.1f}s")
        print(f"Calculated bitrate: {original_info.get('calculated_bitrate_kbps', 'unknown')}kbps")
    
    print("\n" + "="*80)
    print("TRANSCODE ANALYSIS")
    print("="*80)
    
    # Analyze transcodes
    transcode_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Transcoded")]
    
    all_results = []
    
    for folder in sorted(transcode_folders):
        print(f"\n{folder.name}:")
        print("-" * 50)
        
        for file in folder.glob("*E01*vbr*.mkv"):
            info = run_ffprobe_basic(file)
            
            # Calculate compression ratio
            compression_ratio = info['size_bytes'] / original_info['size_bytes']
            space_savings = (1 - compression_ratio) * 100
            
            print(f"  File: {file.name}")
            print(f"  Size: {info['size_gb']:.2f}GB")
            print(f"  Target bitrate (filename): {info.get('filename_bitrate_kbps', 'unknown')}kbps")
            if 'calculated_bitrate_kbps' in info:
                print(f"  Calculated bitrate: {info['calculated_bitrate_kbps']}kbps")
                # Compare filename vs calculated
                if 'filename_bitrate_kbps' in info:
                    diff_pct = abs(info['calculated_bitrate_kbps'] - info['filename_bitrate_kbps']) / info['filename_bitrate_kbps'] * 100
                    print(f"  Bitrate difference: {diff_pct:.1f}%")
            print(f"  Compression ratio: {compression_ratio:.3f}")
            print(f"  Space savings: {space_savings:.1f}%")
            
            info['folder'] = folder.name
            info['compression_ratio'] = compression_ratio
            info['space_savings'] = space_savings
            all_results.append(info)
    
    # Statistical analysis
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("VARIANCE ANALYSIS")
        print("="*80)
        
        # Group by similar target bitrates
        bitrate_groups = {}
        for result in all_results:
            target = result.get('filename_bitrate_kbps')
            if target:
                if target not in bitrate_groups:
                    bitrate_groups[target] = []
                bitrate_groups[target].append(result)
        
        print(f"\nResults by target bitrate:")
        for target_bitrate in sorted(bitrate_groups.keys()):
            results = bitrate_groups[target_bitrate]
            print(f"\nTarget {target_bitrate}kbps:")
            
            sizes = []
            calc_bitrates = []
            
            for r in results:
                print(f"  {r['folder']}: {r['size_gb']:.2f}GB", end="")
                if 'calculated_bitrate_kbps' in r:
                    print(f" ({r['calculated_bitrate_kbps']}kbps actual)")
                    calc_bitrates.append(r['calculated_bitrate_kbps'])
                else:
                    print()
                sizes.append(r['size_gb'])
            
            if len(sizes) > 1:
                size_variance = (max(sizes) - min(sizes)) / min(sizes) * 100
                print(f"  Size variance: {size_variance:.1f}%")
                
                if len(calc_bitrates) > 1:
                    bitrate_variance = (max(calc_bitrates) - min(calc_bitrates)) / min(calc_bitrates) * 100
                    print(f"  Bitrate variance: {bitrate_variance:.1f}%")
        
        # Overall statistics
        all_sizes = [r['size_gb'] for r in all_results]
        all_calc_bitrates = [r['calculated_bitrate_kbps'] for r in all_results if 'calculated_bitrate_kbps' in r]
        all_savings = [r['space_savings'] for r in all_results]
        
        print(f"\nOverall statistics:")
        print(f"File sizes: {min(all_sizes):.2f}GB - {max(all_sizes):.2f}GB (std: {statistics.stdev(all_sizes):.2f}GB)")
        if all_calc_bitrates:
            print(f"Calc bitrates: {min(all_calc_bitrates)}-{max(all_calc_bitrates)}kbps (std: {statistics.stdev(all_calc_bitrates):.0f}kbps)")
        print(f"Space savings: {min(all_savings):.1f}%-{max(all_savings):.1f}% (std: {statistics.stdev(all_savings):.1f}%)")
        
        # Identify patterns
        print(f"\nPatterns identified:")
        
        # Check for identical files
        size_matches = {}
        for result in all_results:
            size_key = round(result['size_bytes'] / 1024)  # Round to nearest KB
            if size_key not in size_matches:
                size_matches[size_key] = []
            size_matches[size_key].append(result)
        
        identical_files = [group for group in size_matches.values() if len(group) > 1]
        if identical_files:
            print(f"Identical file sizes found:")
            for group in identical_files:
                print(f"  Size {group[0]['size_gb']:.2f}GB: {[r['folder'] for r in group]}")
        
        # Check for major differences
        unique_target_bitrates = set(r.get('filename_bitrate_kbps') for r in all_results if r.get('filename_bitrate_kbps'))
        if len(unique_target_bitrates) > 1:
            min_target = min(unique_target_bitrates)
            max_target = max(unique_target_bitrates)
            target_range = (max_target - min_target) / min_target * 100
            print(f"Target bitrate range: {min_target}-{max_target}kbps ({target_range:.1f}% variation)")
        
        print(f"\nPotential causes of variance:")
        print(f"1. Different VMAF targets ({len(unique_target_bitrates)} different target bitrates)")
        print(f"2. Different optimization runs with different convergence")
        print(f"3. Different encoder parameters or hardware")
        print(f"4. Different clip extraction positions")
        print(f"5. Different bounds calculation or caching effects")

if __name__ == "__main__":
    analyze_simple_variance()
