#!/usr/bin/env python3
"""
VMAF Center Crop Optimization Implementation

Based on the technique described in Reddit posts about using center cropping
to accelerate VMAF calculations while maintaining accuracy.

The technique uses a center crop (e.g., 1280x720 crop from 1920x1080) to:
1. Reduce computational load significantly (4x fewer pixels)
2. Maintain representative quality metrics (center typically has most important content)
3. Achieve speed improvements of 2-4x while staying within 0.3 VMAF points of full-frame analysis

Example FFmpeg command:
ffmpeg -hide_banner -r 60 -i distorted.mp4 -r 60 -i reference.mp4 -an -sn -map 0:V -map 1:V 
-lavfi '[0:v]setpts=PTS-STARTPTS,crop=1280:720:320:180[dist];[1:v]setpts=PTS-STARTPTS,crop=1280:720:320:180[ref];[dist][ref]libvmaf=n_threads=4' 
-t 30 -f null -
"""

import subprocess
import shlex
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json


def get_video_resolution(video_path: Path) -> Optional[Tuple[int, int]]:
    """Get video resolution using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", 
            "-of", "csv=s=x:p=0", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            width, height = map(int, result.stdout.strip().split('x'))
            return width, height
    except Exception as e:
        print(f"[DEBUG] Resolution detection failed for {video_path}: {e}")
    return None


def calculate_center_crop(width: int, height: int, target_ratio: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Calculate center crop dimensions and offsets.
    
    Args:
        width, height: Original resolution
        target_ratio: Fraction of original resolution to keep (0.5 = half width/height)
    
    Returns:
        (crop_width, crop_height, x_offset, y_offset)
    """
    # Calculate crop dimensions (ensure even numbers for video encoding)
    crop_width = int(width * target_ratio)
    crop_height = int(height * target_ratio)
    
    # Make sure dimensions are even
    crop_width = crop_width - (crop_width % 2)
    crop_height = crop_height - (crop_height % 2)
    
    # Calculate center offsets
    x_offset = (width - crop_width) // 2
    y_offset = (height - crop_height) // 2
    
    return crop_width, crop_height, x_offset, y_offset


def compute_vmaf_center_crop(reference: Path, distorted: Path, 
                           crop_ratio: float = 0.5, 
                           n_threads: int = 16,
                           duration: Optional[int] = None) -> Optional[float]:
    """
    Compute VMAF score using center cropping for speed optimization.
    
    Args:
        reference: Path to reference video
        distorted: Path to distorted video  
        crop_ratio: Fraction of original resolution to crop (0.5 = 50% width/height)
        n_threads: Number of threads for libvmaf
        duration: Limit analysis to first N seconds (None = full video)
    
    Returns:
        VMAF score or None if failed
    """
    # Validate input files
    if not reference.exists() or not distorted.exists():
        print(f"[WARN] Input files missing: ref={reference.exists()}, dist={distorted.exists()}")
        return None
    
    # Get resolution from distorted video (typically lower resolution)
    dist_resolution = get_video_resolution(distorted)
    if not dist_resolution:
        print(f"[WARN] Could not determine resolution for {distorted}")
        return None
    
    width, height = dist_resolution
    crop_width, crop_height, x_offset, y_offset = calculate_center_crop(width, height, crop_ratio)
    
    print(f"[CENCRO] Using center crop: {crop_width}x{crop_height} from {width}x{height} "
          f"(offset {x_offset},{y_offset}, ratio {crop_ratio:.1%})")
    
    # Build FFmpeg command with center cropping
    duration_args = ["-t", str(duration)] if duration else []
    
    # Crop filter for both videos
    crop_filter = f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset}"
    
    # Build filter graph with cropping and VMAF
    vmaf_opts = [f"n_threads={n_threads}"]
    if n_threads >= 8:  # Adjusted threshold for 16 threads
        vmaf_opts.append("n_subsample=1")  # Process every frame
    
    vmaf_filter = f"libvmaf={':'.join(vmaf_opts)}"
    
    # Complete filter graph: crop both inputs, then run VMAF
    filter_graph = (
        f"[0:v]setpts=PTS-STARTPTS,{crop_filter}[dist];"
        f"[1:v]setpts=PTS-STARTPTS,{crop_filter}[ref];"
        f"[dist][ref]{vmaf_filter}"
    )
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-y"
    ] + duration_args + [
        "-i", str(distorted),   # distorted first (0:v)
        "-i", str(reference),   # reference second (1:v) 
        "-an", "-sn",           # no audio/subtitles
        "-map", "0:V", "-map", "1:V",  # explicit video mapping
        "-lavfi", filter_graph,
        "-f", "null", "-"
    ]
    
    print("[CENCRO-CMD] " + " ".join(shlex.quote(c) for c in cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, encoding='utf-8', errors='ignore')
        
        if result.returncode != 0:
            print(f"[CENCRO] FFmpeg failed (code {result.returncode}): {result.stderr}")
            return None
        
        # Parse VMAF score from stderr output
        stderr_text = result.stderr or ""
        vmaf_score = None
        
        for line in stderr_text.split('\n'):
            if 'VMAF score:' in line:
                try:
                    vmaf_score = float(line.split('VMAF score:')[1].strip())
                    break
                except (IndexError, ValueError):
                    continue
        
        if vmaf_score is not None:
            print(f"[CENCRO] VMAF score: {vmaf_score:.2f} (center crop {crop_ratio:.1%})")
            return vmaf_score
        else:
            print(f"[CENCRO] Could not parse VMAF score from output")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"[CENCRO] VMAF computation timed out after 600s")
        return None
    except Exception as e:
        print(f"[CENCRO] Unexpected error: {e}")
        return None


def test_center_crop_accuracy(reference: Path, distorted: Path, 
                            crop_ratios: list[float] = [0.5, 0.6, 0.75]) -> Dict[str, Any]:
    """
    Test different crop ratios to find optimal speed vs accuracy balance.
    
    Returns dictionary with results for each crop ratio plus full-frame baseline.
    """
    results = {}
    
    # Get full-frame baseline (this will be slow)
    print("[ACCURACY-TEST] Computing full-frame VMAF baseline...")
    full_vmaf = compute_vmaf_center_crop(reference, distorted, crop_ratio=1.0, duration=30)
    results["full_frame"] = {"vmaf": full_vmaf, "crop_ratio": 1.0}
    
    if full_vmaf is None:
        print("[ACCURACY-TEST] Failed to get baseline, aborting test")
        return results
    
    # Test each crop ratio
    for crop_ratio in crop_ratios:
        print(f"[ACCURACY-TEST] Testing crop ratio {crop_ratio:.1%}...")
        crop_vmaf = compute_vmaf_center_crop(reference, distorted, crop_ratio=crop_ratio, duration=30)
        
        if crop_vmaf is not None:
            diff = abs(crop_vmaf - full_vmaf)
            results[f"crop_{crop_ratio:.0%}"] = {
                "vmaf": crop_vmaf,
                "crop_ratio": crop_ratio,
                "difference": diff,
                "acceptable": diff <= 0.3  # Target accuracy threshold
            }
            print(f"[ACCURACY-TEST] Crop {crop_ratio:.1%}: {crop_vmaf:.2f} (diff: {diff:.2f})")
        else:
            results[f"crop_{crop_ratio:.0%}"] = {"vmaf": None, "crop_ratio": crop_ratio}
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VMAF center crop optimization")
    parser.add_argument("reference", help="Reference video file")
    parser.add_argument("distorted", help="Distorted video file")
    parser.add_argument("--crop-ratio", type=float, default=0.5, help="Crop ratio (default: 0.5)")
    parser.add_argument("--threads", type=int, default=16, help="VMAF threads (default: 16)")
    parser.add_argument("--duration", type=int, help="Limit to first N seconds")
    parser.add_argument("--test-accuracy", action="store_true", help="Test multiple crop ratios")
    
    args = parser.parse_args()
    
    ref_path = Path(args.reference)
    dist_path = Path(args.distorted)
    
    if args.test_accuracy:
        results = test_center_crop_accuracy(ref_path, dist_path)
        print(f"\n[RESULTS] Accuracy test results:")
        print(json.dumps(results, indent=2))
    else:
        score = compute_vmaf_center_crop(ref_path, dist_path, 
                                       crop_ratio=args.crop_ratio,
                                       n_threads=args.threads,
                                       duration=args.duration)
        if score is not None:
            print(f"\n[FINAL] VMAF Score: {score:.2f}")
        else:
            print(f"\n[FINAL] VMAF computation failed")
