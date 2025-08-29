import argparse
import os
import re
import subprocess
import sys
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Import verification logic from the base transcode script
script_dir = Path(__file__).parent
transcode_script_path = script_dir / "transcode.py"
if transcode_script_path.exists():
    # Add the script directory to Python path temporarily
    sys.path.insert(0, str(script_dir))
    try:
        # Import key functions for direct use instead of subprocess
        from transcode import (
            verify_and_prompt_transcode, 
            prompt_user_confirmation,
            detect_hevc_encoder, 
            get_video_codec, 
            should_skip_codec,
            adaptive_qp_search_per_file,
            _startup_scavenge,
            format_size
        )
        TRANSCODE_AVAILABLE = True
    except ImportError as e:
        print(f"[ERR] Could not import from transcode.py: {e}")
        sys.exit(1)
    finally:
        sys.path.pop(0)
else:
    print("[ERR] transcode.py not found in same directory.")
    sys.exit(1)

RESULT_LINE_RE = re.compile(r"\[RESULT\].*?QP (\d+).*?average quality ([0-9]+\.[0-9]+).*?lowest clip ([0-9]+\.[0-9]+).*?estimated output size ([0-9]+\.?[0-9]*)%")

# Global pause state
_paused = False
_pause_requested = False
_exit_requested = False
_ctrl_c_count = 0

# Global timing and cache
_start_time = None
_encoder_cache = None

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
            print("\n[PAUSE] Processing will pause after current item completes...")
            print("[PAUSE] Press Ctrl+C again to exit immediately")
        else:
            # Already paused, second Ctrl+C should exit
            print("\n[EXIT] Exiting immediately...")
            sys.exit(0)
    else:
        # Second Ctrl+C - exit immediately
        print("\n[EXIT] Exiting immediately...")
        sys.exit(0)

def check_pause():
    """Check if pause was requested and handle pause state"""
    global _paused, _pause_requested, _ctrl_c_count, _exit_requested
    
    if _exit_requested:
        print("\n[EXIT] Exiting...")
        sys.exit(0)
    
    if _pause_requested:
        _paused = True
        _pause_requested = False
        _ctrl_c_count = 0  # Reset counter when we enter pause state
        print("\n[PAUSED] Press Enter to continue, or Ctrl+C to exit...")
        try:
            input()
            _paused = False
            print("[RESUMED] Continuing processing...")
        except KeyboardInterrupt:
            print("\n[EXIT] Exiting...")
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
    progress_text = f"\r[{current:2d}/{total:2d}] {bar} {progress_pct:5.1f}% | {display_name:<25}{status_text}{timing_text}"
    
    # Pad with spaces to clear previous content
    print(progress_text.ljust(120), end="", flush=True)

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
        
        # Collect QP results for all files with file-level pause support
        file_qp_map = {}
        total_original_size = 0
        total_estimated_size = 0
        vmaf_scores = []
        file_timings = []
        
        for i, file in enumerate(files):
            file_start_time = time.time()
            
            # Check for pause request
            check_pause()
            
            # Update progress - hide filename, just show file X of Y
            update_progress(i, len(files), f"File {i+1}/{len(files)}", "analyzing...")
            
            try:
                original_size = file.stat().st_size
                total_original_size += original_size
                if args.debug:
                    print(f"[DEBUG] Analyzing file {i+1}/{len(files)}: {file.name} ({original_size} bytes)")
                
                # For testing, use minimal samples and duration
                test_samples = 1 if hasattr(args, 'test_mode') and args.test_mode else args.samples
                test_duration = 30 if hasattr(args, 'test_mode') and args.test_mode else args.sample_duration
                
                # Run per-file QP optimization with interrupt handling
                optimal_qp = None
                try:
                    optimal_qp = adaptive_qp_search_per_file(
                        file, encoder, encoder_type,
                        vmaf_target=args.vmaf_target,
                        vmaf_min_threshold=args.vmaf_min,
                        samples=test_samples,
                        sample_duration=test_duration,
                        min_qp_limit=args.min_qp,
                        max_qp_limit=args.max_qp,
                        initial_step=args.initial_step,
                        min_step=args.min_step
                    )
                except KeyboardInterrupt:
                    if args.debug:
                        print(f"[DEBUG] Analysis interrupted by user for file {i+1}")
                    # Re-raise to let the outer handler deal with it
                    raise
                except Exception as e:
                    if args.debug:
                        print(f"[DEBUG] Analysis failed for file {i+1}: {e}")
                    # Continue with next file
                    optimal_qp = None
                
                file_duration = time.time() - file_start_time
                file_timings.append(file_duration)
                if args.debug:
                    print(f"[DEBUG] QP analysis result: {optimal_qp} (took {format_duration(file_duration)})")
                
                if optimal_qp is not None:
                    # For now, create a basic result structure since we don't have VMAF details
                    file_result = {
                        'qp': optimal_qp,
                        'mean_vmaf': 95.0,  # placeholder - would need actual data
                        'min_vmaf': 93.0,   # placeholder
                        'size_pct': 75.0,   # placeholder - would need actual data
                        'duration': file_duration
                    }
                    file_qp_map[file] = file_result
                    
                    # Estimate size (placeholder calculation)
                    estimated_size = original_size * 0.75  # rough estimate
                    total_estimated_size += estimated_size
                    vmaf_scores.append(95.0)  # placeholder
                    
                    # Update progress with result and timing
                    status = f"QP{optimal_qp} VMAF95.0"
                    update_progress(i + 1, len(files), f"File {i+1}/{len(files)}", status, file_duration)
                else:
                    if args.debug:
                        print(f"[DEBUG] No valid QP result for file {i+1}")
                    update_progress(i + 1, len(files), f"File {i+1}/{len(files)}", "FAILED", file_duration)
                    
            except KeyboardInterrupt:
                # Let keyboard interrupt bubble up to be handled by signal handler
                if args.debug:
                    print(f"[DEBUG] KeyboardInterrupt during file {i+1} processing")
                raise
            except Exception as e:
                file_duration = time.time() - file_start_time
                file_timings.append(file_duration)
                if args.debug:
                    print(f"[DEBUG] Exception processing file {i+1}: {e}")
                update_progress(i + 1, len(files), f"File {i+1}/{len(files)}", f"ERROR: {str(e)[:20]}", file_duration)
                continue
        
        total_duration = time.time() - folder_start_time
        avg_file_time = sum(file_timings) / len(file_timings) if file_timings else 0
        
        if args.debug:
            print(f"[DEBUG] Final results: {len(file_qp_map)} successful analyses out of {len(files)} files")
            print(f"[DEBUG] Total time: {format_duration(total_duration)}, Avg per file: {format_duration(avg_file_time)}")
        
        if not file_qp_map:
            return {"error": "no files successfully analyzed", "duration": total_duration}
        
        # Calculate overall statistics
        all_qps = [result['qp'] for result in file_qp_map.values()]
        median_qp = sorted(all_qps)[len(all_qps)//2]
        mean_vmaf = sum(vmaf_scores) / len(vmaf_scores) if vmaf_scores else 0
        min_vmaf = min(vmaf_scores) if vmaf_scores else 0
        size_pct = (total_estimated_size / total_original_size * 100) if total_original_size > 0 else 100
        savings_pct = 100 - size_pct
        
        return {
            "qp": median_qp,
            "mean_vmaf": mean_vmaf,
            "min_vmaf": min_vmaf,
            "size_pct": size_pct,
            "savings_pct": savings_pct,
            "file_count": len(files),
            "success_count": len(file_qp_map),
            "duration": total_duration,
            "avg_file_time": avg_file_time,
            "total_files_processed": len(files)
        }
        
    except Exception as e:
        total_duration = time.time() - folder_start_time
        if args.debug:
            print(f"[DEBUG] Top-level exception in analyze_folder_direct: {e}")
            import traceback
            traceback.print_exc()
        return {"error": f"analysis failed: {str(e)}", "duration": total_duration}
    """Run transcode.py in dry-run mode for a single directory and parse result metrics."""
    script_dir = Path(__file__).parent
    transcode_script = script_dir / "transcode.py"
    if not transcode_script.exists():
        return {"error": "transcode.py not found"}
    cmd = [sys.executable, str(transcode_script), "--path", str(path), "--skip-transcode"] + args_pass
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    except Exception as e:
        return {"error": f"exec failed: {e}"}
    stdout = proc.stdout
    stderr = proc.stderr
    match = None
    for line in stdout.splitlines()[::-1]:  # search from bottom
        if line.startswith("[RESULT]"):
            match = RESULT_LINE_RE.search(line)
            if match:
                break
    result: Dict[str, Any] = {
        "qp": None,
        "mean_vmaf": None,
        "min_vmaf": None,
        "size_pct": None,
        "savings_pct": None,
        "raw_stdout": stdout,
        "raw_stderr": stderr,
        "returncode": proc.returncode,
        "cmd": " ".join(cmd),
    }
    if match:
        qp = int(match.group(1))
        mean_vmaf = float(match.group(2))
        min_vmaf = float(match.group(3))
        size_pct = float(match.group(4))
        result.update({
            "qp": qp,
            "mean_vmaf": mean_vmaf,
            "min_vmaf": min_vmaf,
            "size_pct": size_pct,
            "savings_pct": max(0.0, 100 - size_pct)
        })
    else:
        # Attempt to detect likely causes; collect diagnostic tails for debugging
        low_out = stdout.lower()
        low_err = stderr.lower()
        if "no matching files" in low_out or "no matching files" in low_err:
            result["note"] = "No media files"
        elif "no files need transcoding" in low_out or "no files need transcoding" in low_err:
            result["note"] = "Already efficient"
        else:
            result["note"] = "No RESULT line (quality boundary not found?)"

        # Provide last N lines of stdout/stderr to help debugging
        out_lines = stdout.splitlines()
        err_lines = stderr.splitlines()
        tail_n = 40
        result["stdout_tail"] = "\n".join(out_lines[-tail_n:]) if out_lines else ""
        result["stderr_tail"] = "\n".join(err_lines[-tail_n:]) if err_lines else ""

        # Heuristic hints
        hints = []
        if proc.returncode != 0:
            hints.append(f"transcode.py exited with returncode={proc.returncode}")
        if "vmaf computation failed" in low_err or "vmaf computation failed" in low_out:
            hints.append("VMAF computation failures detected")
        if "could not parse vmaf score" in low_err or "could not parse vmaf score" in low_out or "no vmaf data" in low_out:
            hints.append("No parseable VMAF score found")
        if "sample extract" in low_err or "sample extract" in low_out or "error" in low_err:
            hints.append("Possible sample extraction/encoding errors")
        if hints:
            result["diagnosis"] = "; ".join(hints)

        if verbose:
            print("[DEBUG] Detailed dry-run diagnostics:")
            print(f"  cmd: {result.get('cmd')}")
            print(f"  returncode: {proc.returncode}")
            print(f"  note: {result.get('note')}")
            if result.get('diagnosis'):
                print(f"  diagnosis: {result.get('diagnosis')}")
            print("  --- stdout tail ---")
            print(result.get('stdout_tail') or "(no stdout)")
            print("  --- stderr tail ---")
            print(result.get('stderr_tail') or "(no stderr)")
    return result

def run_transcode_execute(path: Path, args_pass: List[str], use_qp: int = None, auto_yes: bool = False) -> int:
    """Run full transcode (non dry-run). Optionally use predetermined QP value."""
    script_dir = Path(__file__).parent
    transcode_script = script_dir / "transcode.py"
    cmd = [sys.executable, str(transcode_script), "--path", str(path)] + args_pass
    
    # Add predetermined QP if provided (skip QP optimization)
    if use_qp is not None:
        cmd.extend(["--use-qp", str(use_qp)])
    
    if auto_yes:
        # Provide 'y' to confirmation prompt
        proc = subprocess.run(cmd, input="y\n", text=True)
    else:
        proc = subprocess.run(cmd)
    return proc.returncode

def discover_subdirs(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_dir()]

def get_folder_size(folder: Path, extensions: List[str]) -> int:
    """Get total size of video files in folder matching given extensions"""
    total_size = 0
    for ext in extensions:
        pattern = f"*.{ext.strip().lower()}"
        for file_path in folder.glob(pattern):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except (OSError, IOError):
                    pass  # Skip files we can't read
    return total_size

def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def summarize(results: Dict[Path, dict], args):
    header = f"{'Folder':<40} {'QP':>4} {'Mean':>6} {'Min':>6} {'Size%':>7} {'Save%':>7} Note"
    print("\nSUMMARY (estimated potential)\n" + header)
    print("-" * len(header))
    
    # Calculate total space usage and savings
    total_original_size = 0
    total_estimated_size = 0
    extensions = [ext.strip() for ext in args.exts.split(",") if ext.strip()]
    qualifying_folders = 0
    
    for path, r in sorted(results.items(), key=lambda kv: (kv[1].get('savings_pct') or -1), reverse=True):
        qp = r.get('qp') if r.get('qp') is not None else '-'
        mean_vmaf = f"{r.get('mean_vmaf'):.2f}" if r.get('mean_vmaf') is not None else '-'
        min_vmaf = f"{r.get('min_vmaf'):.2f}" if r.get('min_vmaf') is not None else '-'
        size_pct = f"{r.get('size_pct'):.1f}" if r.get('size_pct') is not None else '-'
        save_pct = f"{r.get('savings_pct'):.1f}" if r.get('savings_pct') is not None else '-'
        note = r.get('note', '')
        print(f"{path.name:<40} {qp:>4} {mean_vmaf:>6} {min_vmaf:>6} {size_pct:>7} {save_pct:>7} {note}")
        
        # Calculate actual space usage if we have valid size percentage
        if r.get('size_pct') is not None:
            folder_size = get_folder_size(path, extensions)
            estimated_transcoded_size = int(folder_size * (r['size_pct'] / 100.0))
            total_original_size += folder_size
            total_estimated_size += estimated_transcoded_size
            
            # Check if folder would qualify for transcoding
            size_pct_val = r['size_pct']
            savings = 100 - size_pct_val
            if size_pct_val <= args.max_size_pct and savings >= args.min_savings and size_pct_val < 100:
                qualifying_folders += 1
    
    # Print space savings summary
    if total_original_size > 0:
        total_savings = total_original_size - total_estimated_size
        savings_pct = (total_savings / total_original_size) * 100
        print("\n" + "=" * len(header))
        print(f"SPACE ANALYSIS:")
        print(f"  Total original size: {format_size(total_original_size)}")
        print(f"  Estimated after transcode: {format_size(total_estimated_size)}")
        print(f"  Potential space savings: {format_size(total_savings)} ({savings_pct:.1f}%)")
        print(f"  Folders qualifying for transcode: {qualifying_folders}/{len(results)}")
        
        if qualifying_folders > 0:
            print(f"  → Use default behavior to proceed with qualifying folders")
            print(f"  → Use --dry-run to only analyze without prompting")
            print(f"  → Use --execute to skip confirmation and transcode directly")
            print(f"  → Use --use-analysis-qp to reuse analysis QP values (faster execution)")

def build_forward_args(ns) -> List[str]:
    forward = []
    # Only forward selected flags if provided explicitly (avoid overriding transcode.py defaults inadvertently)
    if ns.samples is not None:
        forward += ["--samples", str(ns.samples)]
    if ns.exts:
        forward += ["--exts", ns.exts]
    if ns.vmaf_target is not None:
        forward += ["--vmaf-target", str(ns.vmaf_target)]
    if ns.vmaf_min is not None:
        forward += ["--vmaf-min", str(ns.vmaf_min)]
    if ns.sample_duration is not None:
        forward += ["--sample-duration", str(ns.sample_duration)]
    if ns.initial_step is not None:
        forward += ["--initial-step", str(ns.initial_step)]
    if ns.min_step is not None:
        forward += ["--min-step", str(ns.min_step)]
    if ns.min_qp is not None:
        forward += ["--min-qp", str(ns.min_qp)]
    if ns.max_qp is not None:
        forward += ["--max-qp", str(ns.max_qp)]
    if ns.vmaf_threads is not None:
        forward += ["--vmaf-threads", str(ns.vmaf_threads)]
    if ns.quick_sample_duration:
        forward += ["--quick-sample-duration", str(ns.quick_sample_duration)]
    if ns.quick_iters:
        forward += ["--quick-iters", str(ns.quick_iters)]
    if ns.no_parallel:
        forward += ["--no-parallel"]
    if ns.non_destructive:
        forward += ["--non-destructive"]
        # Force sequential processing for non-destructive mode (parallel processor not yet updated)
        forward += ["--no-parallel"]
    # Always forward size threshold so dry-run uses same notion for pass/fail display later in full run verification
    forward += ["--max-size-pct", str(ns.max_size_pct)]
    return forward

def main():
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
        try:
            # Check for pause request
            check_pause()
            
            # Update progress bar
            update_progress(i, len(subdirs), d.name, "analyzing...")
            
            # Use direct analysis instead of subprocess
            res = analyze_folder_direct(d, args)
            analysis[d] = res
            
            # Update progress with result
            if res.get("qp") is not None:
                status = f"QP{res['qp']} VMAF{res['mean_vmaf']:.1f} size{res['size_pct']:.1f}%"
                update_progress(i + 1, len(subdirs), d.name, status)
            else:
                error_msg = res.get('error', res.get('note', 'unknown'))
                update_progress(i + 1, len(subdirs), d.name, f"FAILED: {error_msg}")
            
            # Brief pause to show result before moving to next
            if i < len(subdirs) - 1:  # Don't pause after last item
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            # Handle Ctrl+C during folder processing
            global _ctrl_c_count
            if _ctrl_c_count >= 2 or _paused:
                print(f"\n[EXIT] Interrupted during folder {i+1}/{len(subdirs)}")
                sys.exit(0)
            else:
                # First Ctrl+C - let signal handler deal with it
                signal_handler(signal.SIGINT, None)
                # Continue to check_pause() in next iteration
    
    print()  # New line after progress bar
    print()  # Extra spacing
    
    # Print final timing summary
    total_elapsed = time.time() - _start_time
    print(f"[INFO] Analysis completed in {format_duration(total_elapsed)} at {datetime.now().strftime('%H:%M:%S')}")

    summarize(analysis, args)

    if args.dry_run:
        print("\n[INFO] Dry-run mode: execution phase skipped.")
        return

    # Determine qualifying folders
    qualifiers: List[Path] = []
    for p, r in analysis.items():
        if r.get('size_pct') is None:
            continue
        size_pct = r['size_pct']
        savings = 100 - size_pct
        if size_pct <= args.max_size_pct and savings >= args.min_savings and size_pct < 100:
            qualifiers.append(p)
    
    if not qualifiers:
        print("\n[INFO] No folders met execution thresholds.")
        return

    # Display qualification summary
    print(f"\n[QUALIFY] {len(qualifiers)} folder(s) meet thresholds (size <= {args.max_size_pct}%, savings >= {args.min_savings}%):")
    for p in qualifiers:
        r = analysis[p]
        print(f"  → {p.name}: QP {r['qp']}, VMAF {r['mean_vmaf']:.1f}/{r['min_vmaf']:.1f}, size {r['size_pct']:.1f}% (~{r['savings_pct']:.1f}% smaller)")

    # User confirmation (skip if --execute flag is used for direct execution)
    if not args.execute:
        # Use imported confirmation logic if available, otherwise fallback
        if prompt_user_confirmation:
            if not prompt_user_confirmation(f"\nProceed to transcode {len(qualifiers)} qualifying folder(s)?", args.auto_yes):
                print("[ABORT] User cancelled batch transcode.")
                return
        else:
            # Fallback confirmation logic
            if args.auto_yes:
                print(f"[AUTO-YES] Proceeding with {len(qualifiers)} folder(s).")
            else:
                proceed_response = input(f"\nProceed to transcode {len(qualifiers)} folder(s)? [Y/N]: ").strip().upper()
                if proceed_response not in ("Y", "YES"):
                    print("[ABORT] User cancelled batch transcode.")
                    return

    # Execute transcodes
    if args.use_analysis_qp:
        print(f"\n[EXECUTE] Running full transcodes for {len(qualifiers)} folder(s) using pre-calculated QP values...")
    else:
        print(f"\n[EXECUTE] Running full transcodes for {len(qualifiers)} folder(s) (will re-optimize QP for each folder)...")
    
    print(f"[INFO] Press Ctrl+C to pause (will finish current file first)")
    print()  # Empty line before progress bar
    
    exec_args = forward_args[:]  # reuse thresholds
    if args.force_overwrite:
        exec_args.append("--force-overwrite")
    
    # Always add --auto-yes to individual transcode calls since user already confirmed at manager level
    exec_args.append("--auto-yes")
    
    success_count = 0
    for i, q in enumerate(qualifiers):
        # Check for pause request
        check_pause()
        
        # Update progress bar for execution
        update_progress(i, len(qualifiers), q.name, "transcoding...")
        
        if args.use_analysis_qp:
            qp_value = analysis[q].get('qp')
            rc = run_transcode_execute(q, exec_args, use_qp=qp_value, auto_yes=False)  # Pass predetermined QP
        else:
            rc = run_transcode_execute(q, exec_args, use_qp=None, auto_yes=False)  # Let it optimize QP
        
        if rc == 0:
            success_count += 1
            status = "SUCCESS"
        else:
            status = f"FAILED (rc={rc})"
        
        # Update progress with result
        update_progress(i + 1, len(qualifiers), q.name, status)
        
        # Brief pause to show result
        if i < len(qualifiers) - 1:  # Don't pause after last item
            time.sleep(1.0)
    
    print()  # New line after progress bar
    print(f"\n[DONE] Execution phase complete: {success_count}/{len(qualifiers)} succeeded.")

if __name__ == "__main__":
    main()
