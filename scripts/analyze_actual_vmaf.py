#!/usr/bin/env python3
"""
Calculate actual VMAF scores for transcoded files to verify quality.
Uses the built-in VMAF calculation functions from lazy_transcode.

Usage:
    python analyze_actual_vmaf.py <original_file> [transcode_directory]
    python analyze_actual_vmaf.py <original_file> --all-smallest
    
Examples:
    python analyze_actual_vmaf.py "M:/Shows/Show/episode.mkv" "M:/Shows/Show/Transcoded_6"
    python analyze_actual_vmaf.py "M:/Shows/Show/episode.mkv" --all-smallest
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lazy_transcode.core.modules.analysis.media_utils import compute_vmaf_score
from lazy_transcode.utils.logging import get_logger

logger = get_logger("vmaf_analysis")

def find_smallest_files_in_transcode_folders(base_directory: Path, original_filename: str) -> list:
    """Find the smallest transcoded file in each Transcoded folder."""
    transcode_folders = [d for d in base_directory.iterdir() if d.is_dir() and d.name.startswith("Transcoded")]
    smallest_files = []
    
    logger.info(f"Found {len(transcode_folders)} transcode folders")
    
    for folder in sorted(transcode_folders):
        # Look for files that match the original filename pattern
        base_name = Path(original_filename).stem
        matching_files = []
        
        for file in folder.glob("*.mkv"):
            if base_name in file.stem or file.stem.startswith(base_name.split()[0]):
                matching_files.append(file)
        
        if matching_files:
            # Find the smallest file in this folder
            smallest = min(matching_files, key=lambda f: f.stat().st_size)
            smallest_files.append((folder.name, smallest))
            logger.info(f"  {folder.name}: {smallest.name} ({smallest.stat().st_size / (1024**3):.2f}GB)")
    
    return smallest_files

def analyze_specific_files(original_file: Path, transcode_files: list) -> list:
    """Analyze specific transcoded files against the original."""
    results = []
    
    logger.info(f"Original reference: {original_file.name}")
    logger.info(f"Original size: {original_file.stat().st_size / (1024**3):.2f}GB")
    logger.info(f"Analyzing {len(transcode_files)} transcoded files")
    
    for folder_name, file_path in transcode_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {folder_name} - {file_path.name}")
        logger.info(f"{'='*60}")
        
        # Get file info
        size_gb = file_path.stat().st_size / (1024**3)
        
        # Extract target bitrate from filename
        import re
        match = re.search(r'vbr_(\d+)k', file_path.name)
        target_bitrate = int(match.group(1)) if match else None
        
        logger.info(f"File size: {size_gb:.2f}GB")
        logger.info(f"Target bitrate: {target_bitrate}kbps")
        
        # Calculate VMAF score using built-in function with enhanced timeout handling
        logger.info(f"Calculating VMAF for {file_path.name} vs original...")
        logger.info(f"Note: VMAF calculation may take 5-15 minutes for full episodes...")
        start_time = time.time()
        
        try:
            # Use maximum threads available and enable CPU monitoring for long operations
            import os
            max_threads = os.cpu_count() or 8
            vmaf_score = compute_vmaf_score(original_file, file_path, n_threads=max_threads, enable_cpu_monitoring=True)
            elapsed = time.time() - start_time
            logger.info(f"VMAF calculation completed in {elapsed:.1f}s")
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            logger.error(f"VMAF calculation timed out after {elapsed:.1f}s")
            vmaf_score = None
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"VMAF calculation failed after {elapsed:.1f}s: {e}")
            vmaf_score = None
        
        result = {
            'folder': folder_name,
            'filename': file_path.name,
            'filepath': file_path,
            'size_gb': size_gb,
            'target_bitrate': target_bitrate,
            'vmaf_score': vmaf_score
        }
        
        results.append(result)
        
        logger.info(f"VMAF Score: {vmaf_score:.2f}" if vmaf_score else "VMAF Score: Failed")
        
        # Calculate space savings
        original_size = original_file.stat().st_size / (1024**3)
        space_savings = (1 - size_gb / original_size) * 100
        logger.info(f"Space savings: {space_savings:.1f}%")
        
        # Quality assessment
        if vmaf_score:
            if vmaf_score >= 95:
                quality = "Excellent"
            elif vmaf_score >= 90:
                quality = "Very Good"
            elif vmaf_score >= 85:
                quality = "Good"
            elif vmaf_score >= 80:
                quality = "Acceptable"
            else:
                quality = "Poor"
            
            logger.info(f"Quality assessment: {quality}")
            
            # Efficiency calculation (VMAF per GB)
            efficiency = vmaf_score / size_gb
            logger.info(f"Efficiency (VMAF/GB): {efficiency:.1f}")
    
    return results

def analyze_actual_vmaf_scores(original_file: Path, transcode_target=None, all_smallest=False):
    """Analyze actual VMAF scores of the transcoded files."""
    
    if not original_file.exists():
        logger.error(f"Original file not found: {original_file}")
        return
    
    base_directory = original_file.parent
    
    if all_smallest:
        # Find smallest files in all transcode folders
        transcode_files = find_smallest_files_in_transcode_folders(base_directory, original_file.name)
    elif transcode_target:
        # Analyze specific transcode directory or file
        if transcode_target.is_file():
            # Single file
            folder_name = transcode_target.parent.name
            transcode_files = [(folder_name, transcode_target)]
        else:
            # Directory - find all matching files
            transcode_files = []
            base_name = original_file.stem
            for file in transcode_target.glob("*.mkv"):
                if base_name in file.stem or file.stem.startswith(base_name.split()[0]):
                    transcode_files.append((transcode_target.name, file))
    else:
        # Default: find all transcoded files in all folders
        transcode_files = []
        folders = [d for d in base_directory.iterdir() if d.is_dir() and d.name.startswith("Transcoded")]
        
        for folder in sorted(folders):
            base_name = original_file.stem
            for file in folder.glob("*.mkv"):
                if base_name in file.stem or file.stem.startswith(base_name.split()[0]):
                    transcode_files.append((folder.name, file))
    
    if not transcode_files:
        logger.error("No transcoded files found to analyze")
        return
    
    results = analyze_specific_files(original_file, transcode_files)
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['vmaf_score'] is not None]
    
    if not successful_results:
        print("No successful VMAF calculations")
        return
    
    # Sort by file size
    successful_results.sort(key=lambda x: x['size_gb'])
    
    print(f"\nResults sorted by file size:")
    print(f"{'Folder':<15} {'Size(GB)':<10} {'Target(kbps)':<12} {'VMAF':<8} {'Quality':<12} {'Efficiency':<10}")
    print("-" * 75)
    
    best_efficiency = None
    best_efficiency_result = None
    
    for result in successful_results:
        vmaf = result['vmaf_score']
        size = result['size_gb']
        target = result['target_bitrate']
        
        if vmaf >= 95:
            quality = "Excellent"
        elif vmaf >= 90:
            quality = "Very Good"
        elif vmaf >= 85:
            quality = "Good"
        elif vmaf >= 80:
            quality = "Acceptable"
        else:
            quality = "Poor"
        
        efficiency = vmaf / size
        
        if best_efficiency is None or efficiency > best_efficiency:
            best_efficiency = efficiency
            best_efficiency_result = result
        
        print(f"{result['folder']:<15} {size:<10.2f} {target:<12} {vmaf:<8.2f} {quality:<12} {efficiency:<10.1f}")
    
    # Analysis
    print(f"\nKey Insights:")
    
    # Find the smallest file
    smallest = min(successful_results, key=lambda x: x['size_gb'])
    print(f"1. Smallest file: {smallest['folder']} ({smallest['size_gb']:.2f}GB, VMAF {smallest['vmaf_score']:.2f})")
    
    # Find highest VMAF
    highest_vmaf = max(successful_results, key=lambda x: x['vmaf_score'])
    print(f"2. Highest VMAF: {highest_vmaf['folder']} (VMAF {highest_vmaf['vmaf_score']:.2f}, {highest_vmaf['size_gb']:.2f}GB)")
    
    # Best efficiency
    if best_efficiency_result:
        print(f"3. Most efficient: {best_efficiency_result['folder']} ({best_efficiency:.1f} VMAF/GB)")
    
    # Check if smallest file has acceptable quality
    if smallest['vmaf_score'] >= 90:
        print(f"\n✅ HYPOTHESIS CONFIRMED: Smallest file achieves excellent quality!")
        print(f"   The {smallest['target_bitrate']}kbps target produced a {smallest['size_gb']:.2f}GB file")
        print(f"   with VMAF {smallest['vmaf_score']:.2f} (Very Good/Excellent quality)")
    elif smallest['vmaf_score'] >= 85:
        print(f"\n✅ HYPOTHESIS CONFIRMED: Smallest file achieves good quality!")
        print(f"   The {smallest['target_bitrate']}kbps target produced acceptable results")
    else:
        print(f"\n❌ HYPOTHESIS REJECTED: Smallest file has poor quality")
        print(f"   VMAF {smallest['vmaf_score']:.2f} may be too low for good viewing experience")
    
    # Diminishing returns analysis
    print(f"\nDiminishing Returns Analysis:")
    for i, result in enumerate(successful_results[1:], 1):
        prev_result = successful_results[i-1]
        
        vmaf_gain = result['vmaf_score'] - prev_result['vmaf_score']
        size_cost = result['size_gb'] - prev_result['size_gb']
        
        if size_cost > 0:
            vmaf_per_gb_cost = vmaf_gain / size_cost
            print(f"  {prev_result['folder']} → {result['folder']}: +{vmaf_gain:.2f} VMAF for +{size_cost:.2f}GB ({vmaf_per_gb_cost:.1f} VMAF/GB)")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze VMAF scores of transcoded files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific transcoded file
  python analyze_actual_vmaf.py "original.mkv" "Transcoded_6/transcoded.mkv"
  
  # Analyze all files in a transcode directory  
  python analyze_actual_vmaf.py "original.mkv" "Transcoded_6/"
  
  # Find and analyze smallest file from each transcode folder
  python analyze_actual_vmaf.py "original.mkv" --all-smallest
  
  # Analyze all transcoded files in all folders
  python analyze_actual_vmaf.py "original.mkv"
        """
    )
    
    parser.add_argument("original_file", type=Path, 
                       help="Path to the original (reference) video file")
    
    parser.add_argument("transcode_target", type=Path, nargs="?",
                       help="Path to transcoded file or directory (optional)")
    
    parser.add_argument("--all-smallest", action="store_true",
                       help="Find and analyze the smallest file from each Transcoded folder")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate original file
    if not args.original_file.exists():
        print(f"Error: Original file not found: {args.original_file}")
        sys.exit(1)
    
    # Determine analysis mode
    if args.all_smallest:
        print(f"Analyzing smallest files from each transcode folder...")
        analyze_actual_vmaf_scores(args.original_file, all_smallest=True)
    elif args.transcode_target:
        if not args.transcode_target.exists():
            print(f"Error: Transcode target not found: {args.transcode_target}")
            sys.exit(1)
        print(f"Analyzing: {args.transcode_target}")
        analyze_actual_vmaf_scores(args.original_file, args.transcode_target)
    else:
        print(f"Analyzing all transcoded files...")
        analyze_actual_vmaf_scores(args.original_file)
