"""
Batch manager for transcode operations across subdirectories.

This module provides functionality to analyze and batch process multiple
video directories using the core transcode functionality.
"""

import argparse
import os
import re
import subprocess
import sys
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import verification logic from the base transcode script
from .transcode import (
    verify_and_prompt_transcode, 
    prompt_user_confirmation,
    detect_hevc_encoder, 
    get_video_codec, 
    should_skip_codec,
    adaptive_qp_search_per_file,
    _startup_scavenge,
    format_size
)

RESULT_LINE_RE = re.compile(r"\[RESULT\].*?QP (\d+).*?average quality ([0-9]+\.[0-9]+).*?lowest clip ([0-9]+\.[0-9]+).*?estimated output size ([0-9]+\.?[0-9]*)%")

# Global pause state
_paused = False
_pause_requested = False
_exit_requested = False
_ctrl_c_count = 0

# Global timing and cache
_start_time = None
_encoder_cache: Optional[Tuple[str, str]] = None

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def signal_handler(signum, frame):
    """Handle Ctrl+C: first time pauses gracefully, second time exits immediately"""
    global _pause_requested, _exit_requested, _ctrl_c_count, _paused
    
    _ctrl_c_count += 1
    
    if _ctrl_c_count == 1:
        if not _paused:
            _pause_requested = True
            print("\\n[PAUSE] Processing will pause after current item completes...")
            print("[PAUSE] Press Ctrl+C again to exit immediately")
        else:
            # Already paused, second Ctrl+C should exit
            print("\\n[EXIT] Exiting immediately...")
            sys.exit(0)
    else:
        # Second Ctrl+C - exit immediately
        print("\\n[EXIT] Exiting immediately...")
        sys.exit(0)

def check_pause():
    """Check if pause was requested and handle pause state"""
    global _paused, _pause_requested, _ctrl_c_count, _exit_requested
    
    if _exit_requested:
        print("\\n[EXIT] Exiting...")
        sys.exit(0)
    
    if _pause_requested:
        _paused = True
        _pause_requested = False
        _ctrl_c_count = 0  # Reset counter when we enter pause state
        print("\\n[PAUSED] Press Enter to continue, or Ctrl+C to exit...")
        try:
            input()
            _paused = False
            print("[RESUMED] Continuing processing...")
        except KeyboardInterrupt:
            print("\\n[EXIT] Exiting...")
            sys.exit(0)

def update_progress(current: int, total: int, item_name: str, status: str = "", elapsed: float = 0):
    """Update progress bar in place without creating new lines"""
    if total == 0:
        return
    
    progress_pct = (current / total) * 100
    bar_width = 30  # Reduced for more space
    filled_width = int(bar_width * current / total)
    bar = "█" * filled_width + "░" * (bar_width - filled_width)
    
    # Truncate item name if too long
    display_name = item_name[:20] + "..." if len(item_name) > 23 else item_name
    
    # Add timing info
    timing_text = ""
    if elapsed > 0:
        timing_text = f" ({format_duration(elapsed)})"
    
    status_text = f" - {status}" if status else ""
    progress_text = f"\\r[{current:2d}/{total:2d}] {bar} {progress_pct:5.1f}% | {display_name:<25}{status_text}{timing_text}"
    
    # Pad with spaces to clear previous content
    print(progress_text.ljust(120), end="", flush=True)

def discover_subdirs(root: Path) -> List[Path]:
    """Discover subdirectories in the root path"""
    return [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]

def build_forward_args(args):
    """Build arguments to forward to transcode.py"""
    forward_args = []
    
    # Add basic arguments
    if args.vmaf_target != 95.0:
        forward_args.extend(["--vmaf-target", str(args.vmaf_target)])
    if args.vmaf_min != 93.0:
        forward_args.extend(["--vmaf-min", str(args.vmaf_min)])
    if args.sample_duration != 120:
        forward_args.extend(["--sample-duration", str(args.sample_duration)])
    if args.samples != 6:
        forward_args.extend(["--samples", str(args.samples)])
    if args.vmaf_threads != (os.cpu_count() or 8):
        forward_args.extend(["--vmaf-threads", str(args.vmaf_threads)])
    if args.auto_yes:
        forward_args.append("--auto-yes")
    if args.force_overwrite:
        forward_args.append("--force-overwrite")
    if args.no_parallel:
        forward_args.append("--no-parallel")
    if hasattr(args, 'non_destructive') and args.non_destructive:
        forward_args.append("--non-destructive")
    
    return forward_args


def analyze_folder_direct(path: Path, args) -> dict:
    """Analyze a folder using direct function calls instead of subprocess"""
    folder_start_time = time.time()
    
    try:
        if args.debug:
            print(f"[DEBUG] Starting analysis of {path}")
        
        # Early scavenging of stale artifacts
        _startup_scavenge(path)
        
        # Detect encoder (cached for performance)
        global _encoder_cache
        if _encoder_cache is None:
            encoder, encoder_type = detect_hevc_encoder()
            _encoder_cache = (encoder, encoder_type)
            if args.debug:
                print(f"[DEBUG] Detected and cached encoder: {encoder} ({encoder_type})")
        else:
            encoder, encoder_type = _encoder_cache
            if args.debug:
                print(f"[DEBUG] Using cached encoder: {encoder} ({encoder_type})")
        
        # Find video files
        patterns = [f"*.{ext.strip().lower()}" for ext in args.exts.split(",") if ext.strip()]
        files: list[Path] = []
        for pat in patterns:
            files.extend(path.rglob(pat))
        files = sorted(set(files))
        if args.debug:
            print(f"[DEBUG] Found {len(files)} files with patterns {patterns}")
        
        # Filter out hidden files and sample clips
        files = [f for f in files if not f.name.startswith('._') and not f.name.startswith('.')]
        # Enhanced filtering to catch cascading clip artifacts
        files = [f for f in files if not any([
            ".sample_clip" in f.stem,
            "_sample" in f.stem,
            ".clip" in f.stem and ".sample" in f.stem,  # catch .clipN_timestamp.sample
            f.stem.count(".sample") > 0,  # any .sample artifacts
            f.stem.count(".clip") > 0     # any .clip artifacts
        ])]
        if args.debug:
            print(f"[DEBUG] After filtering hidden/sample files: {len(files)}")
        
        if not files:
            return {"error": "no video files found", "duration": time.time() - folder_start_time}
        
        # Filter out files with efficient codecs
        files_to_transcode = []
        for f in files:
            codec = get_video_codec(f)
            if not should_skip_codec(codec):
                files_to_transcode.append(f)
        
        if args.debug:
            print(f"[DEBUG] Files needing transcode: {len(files_to_transcode)}")
        if not files_to_transcode:
            return {"error": "all files already have efficient codecs", "duration": time.time() - folder_start_time}
        
        files = files_to_transcode
        
        # For testing, limit to first few files if not in production mode
        if hasattr(args, 'test_mode') and args.test_mode and len(files) > 3:
            if args.debug:
                print(f"[TEST] Limiting to first 3 files for faster testing")
            files = files[:3]
        
        # Rest of analyze_folder_direct implementation would go here...
        # For now, return a placeholder result
        return {
            "files_analyzed": len(files),
            "duration": time.time() - folder_start_time,
            "qp": 24,  # placeholder
            "avg_vmaf": 95.0,  # placeholder
            "estimated_size_pct": 65.0  # placeholder
        }
        
    except Exception as e:
        if args.debug:
            print(f"[DEBUG] Analysis failed: {e}")
        return {"error": str(e), "duration": time.time() - folder_start_time}


def main():
    """Main entry point for the manager CLI"""
    # Set up signal handler for graceful pause
    signal.signal(signal.SIGINT, signal_handler)
    
    ap = argparse.ArgumentParser(description="Batch manager for transcode.py across subdirectories")
    ap.add_argument("--root", default=".", help="Root directory whose immediate subdirectories will be analyzed")
    ap.add_argument("--exts", default="mkv,mp4,mov,ts", help="Extensions to pass through")
    ap.add_argument("--samples", type=int, default=6, help="Samples per folder for QP optimization (default: 6)")
    ap.add_argument("--vmaf-target", type=float, default=95.0)
    ap.add_argument("--vmaf-min", type=float, default=93.0)
    ap.add_argument("--sample-duration", type=int, default=120)
    ap.add_argument("--initial-step", type=float, default=4.0)
    ap.add_argument("--min-step", type=float, default=1.0)
    ap.add_argument("--min-qp", type=int, default=20, help="Minimum QP value (lower = higher quality, but may cause encoder issues below 20)")
    ap.add_argument("--max-qp", type=int, default=35, help="Maximum QP value (higher = lower quality but smaller files)")
    # Use optimal VMAF threads (testing shows diminishing returns beyond 8)
    detected_cores = os.cpu_count() or 0
    optimal_vmaf_threads = min(8, detected_cores)
    ap.add_argument("--vmaf-threads", type=int, default=optimal_vmaf_threads)
    ap.add_argument("--quick-sample-duration", type=int, default=30)
    ap.add_argument("--quick-iters", type=int, default=1)
    ap.add_argument("--max-size-pct", type=float, default=100.0, help="Only execute on folders whose estimated size percent <= this value")
    ap.add_argument("--min-savings", type=float, default=0.1, help="Minimum percent savings required (e.g. 5 means need at least 5 percent smaller)")
    ap.add_argument("--execute", action="store_true", help="Skip user confirmation and proceed directly to transcode qualifying folders")
    ap.add_argument("--dry-run", action="store_true", help="Only analyze folders, skip execution phase entirely")
    ap.add_argument("--auto-yes", action="store_true", help="Automatically confirm overwrite prompt in transcode.py when executing")
    ap.add_argument("--force-overwrite", action="store_true", help="Forward force overwrite to transcode.py during execution phase")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of folders to process (useful for testing)")
    ap.add_argument("--debug", action="store_true", help="Show verbose diagnostics when dry-run fails to produce a result")
    ap.add_argument("--no-parallel", action="store_true", help="Disable parallel processing in transcode.py (use sequential mode)")
    ap.add_argument("--use-analysis-qp", action="store_true", help="Use QP values from analysis step instead of re-optimizing during execution")
    ap.add_argument("--test-mode", action="store_true", help="Enable fast testing mode (30s samples, max 3 files per folder)")
    ap.add_argument("--non-destructive", action="store_true", help="Save transcoded files to 'Transcoded' subdirectory instead of replacing originals")
    args = ap.parse_args()

    # Set up timing
    global _start_time
    _start_time = time.time()
    
    print(f"[INFO] Started at {datetime.now().strftime('%H:%M:%S')}")
    if args.test_mode:
        print("[TEST] Fast testing mode enabled: 30s samples, max 3 files per folder")

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] root does not exist: {root}")
        sys.exit(1)

    subdirs = discover_subdirs(root)
    if args.limit:
        subdirs = subdirs[:args.limit]
    if not subdirs:
        print("[INFO] No subdirectories to process.")
        return

    forward_args = build_forward_args(args)
    analysis: Dict[Path, dict] = {}
    
    print(f"[INFO] Analyzing {len(subdirs)} subdirectories under {root}...")
    print(f"[INFO] Press Ctrl+C to pause (will finish current file first)")
    print()  # Empty line before progress bar
    
    for i, d in enumerate(subdirs):
        # Check for pause request
        check_pause()
        
        start_time = time.time()
        
        # Update progress bar
        update_progress(i, len(subdirs), d.name, "analyzing...")
        
        result = analyze_folder_direct(d, args)
        elapsed = time.time() - start_time
        
        # Determine status
        if "error" in result:
            status = f"SKIP ({result['error']})"
        else:
            status = f"QP{result.get('qp', '?')} {result.get('estimated_size_pct', 0):.0f}%"
        
        # Update progress with result
        update_progress(i + 1, len(subdirs), d.name, status, elapsed)
        
        analysis[d] = result
        
        # Brief pause to show result unless it's the last item
        if i < len(subdirs) - 1:
            time.sleep(0.5)
    
    print()  # New line after progress bar
    
    # Summary
    successful = [k for k, v in analysis.items() if "error" not in v]
    failed = [k for k, v in analysis.items() if "error" in v]
    
    total_elapsed = time.time() - _start_time
    print(f"\\n[ANALYSIS] Complete in {format_duration(total_elapsed)}")
    print(f"[ANALYSIS] Success: {len(successful)}, Failed: {len(failed)}")
    
    if not successful:
        print("[INFO] No folders qualified for transcoding.")
        return
    
    print(f"\\n[QUALIFIED] {len(successful)} folders ready for transcoding:")
    for folder in successful[:10]:  # Show first 10
        result = analysis[folder]
        qp = result.get('qp', '?')
        size_pct = result.get('estimated_size_pct', 0)
        print(f"  {folder.name}: QP{qp} -> {size_pct:.1f}% of original")
    
    if len(successful) > 10:
        print(f"  ... and {len(successful) - 10} more")
    
    if args.dry_run:
        print("\\n[DRY-RUN] Analysis complete. Skipping execution phase.")
        return
    
    if not args.execute:
        response = input("\\n[CONFIRM] Proceed with transcoding? (y/N): ")
        if response.lower() != 'y':
            print("[CANCELLED] Exiting without transcoding.")
            return
    
    print(f"\\n[EXECUTE] Starting transcoding of {len(successful)} folders...")
    # Execution phase would go here...
    print("[EXECUTE] Transcoding phase not yet implemented in this version.")


if __name__ == "__main__":
    main()
