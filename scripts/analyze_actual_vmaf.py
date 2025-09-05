#!/usr/bin/env python3
"""
Calculate actual VMAF scores for the different transcoded files to verify
if the smallest file achieves acceptable quality despite lower target bitrate.
Uses the built-in VMAF calculation functions from lazy_transcode.
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lazy_transcode.core.modules.analysis.media_utils import compute_vmaf_score
from lazy_transcode.utils.logging import get_logger

logger = get_logger("vmaf_analysis")

def analyze_actual_vmaf_scores():
    """Analyze actual VMAF scores of the transcoded files."""
    
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
    
    logger.info(f"Original reference: {original_file.name}")
    logger.info(f"Original size: {original_file.stat().st_size / (1024**3):.2f}GB")
    
    # Find all transcoded files
    transcoded_files = []
    folders = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Transcoded")]
    
    for folder in sorted(folders):
        for file in folder.glob("*E01*vbr*.mkv"):
            transcoded_files.append((folder.name, file))
    
    logger.info(f"Found {len(transcoded_files)} transcoded files to analyze")
    
    results = []
    
    for folder_name, file_path in transcoded_files:
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
        
        # Calculate VMAF score using built-in function
        logger.info(f"Calculating VMAF for {file_path.name} vs original...")
        start_time = time.time()
        
        try:
            vmaf_score = compute_vmaf_score(original_file, file_path, n_threads=8)
            elapsed = time.time() - start_time
            logger.info(f"VMAF calculation completed in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"VMAF calculation failed: {e}")
            vmaf_score = None
        
        result = {
            'folder': folder_name,
            'filename': file_path.name,
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
    
    # Summary analysis
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

if __name__ == "__main__":
    analyze_actual_vmaf_scores()
