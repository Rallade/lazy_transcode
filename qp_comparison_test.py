#!/usr/bin/env python3
"""
QP Optimization Technique Comparison Test

This script compares the performance of two QP optimization techniques:
1. Traditional binary search (adaptive_qp_search_per_file)
2. Gradient descent optimization (--gradient-descent flag)

The test runs both techniques on random files from different shows and collects
performance metrics including runtime, iterations, and final QP values.
"""

import argparse
import json
import random
import time
from pathlib import Path
import sys
import subprocess
from typing import List, Dict, Any
from datetime import datetime

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from lazy_transcode.core.transcode import (
    detect_hevc_encoder,
    get_video_codec,
    should_skip_codec,
    adaptive_qp_search_per_file,
    _startup_scavenge,
)
from lazy_transcode.config import get_test_root


def find_video_files(root_path: Path, max_files_per_show: int = 3) -> List[Path]:
    """Find random video files from different shows."""
    video_extensions = [".mkv", ".mp4", ".mov", ".ts"]
    all_files = []
    
    print(f"[SCAN] Starting scan of {root_path}")
    print(f"[SCAN] Looking for extensions: {video_extensions}")
    
    # Get all subdirectories (shows)
    try:
        show_dirs = [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        print(f"[SCAN] Found {len(show_dirs)} show directories")
    except Exception as e:
        print(f"[SCAN] ERROR: Could not list directories in {root_path}: {e}")
        return []
    
    for i, show_dir in enumerate(show_dirs, 1):
        print(f"[SCAN] [{i:2d}/{len(show_dirs)}] Scanning {show_dir.name}...", end="", flush=True)
        show_files = []
        
        try:
            # Find video files in this show directory
            for ext in video_extensions:
                found_files = list(show_dir.rglob(f"*{ext}"))
                show_files.extend(found_files)
                if found_files:
                    print(f" {len(found_files)} {ext} files", end="", flush=True)
            
            if not show_files:
                print(" no video files found")
                continue
                
            print(f" | Total: {len(show_files)} files", end="", flush=True)
            
            # Filter out temp/sample files and files with efficient codecs
            filtered_files = []
            for f in show_files:
                # Skip temp/sample files
                if any([
                    f.name.startswith('._'),
                    f.name.startswith('.'),
                    ".sample_clip" in f.stem,
                    "_sample" in f.stem,
                    ".clip" in f.stem and ".sample" in f.stem,
                    f.stem.count(".sample") > 0,
                    f.stem.count(".clip") > 0
                ]):
                    continue
                    
                # Check if file needs transcoding
                try:
                    codec = get_video_codec(f)
                    if not should_skip_codec(codec):
                        filtered_files.append(f)
                except Exception as e:
                    print(f"\n[SCAN] WARNING: Could not check codec for {f.name}: {e}")
                    continue
            
            if filtered_files:
                # Randomly select up to max_files_per_show from this show
                selected = filtered_files[:max_files_per_show] if len(filtered_files) <= max_files_per_show else random.sample(filtered_files, max_files_per_show)
                all_files.extend(selected)
                print(f" | Selected: {len(selected)} suitable files")
            else:
                print(" | No files need transcoding")
                
        except Exception as e:
            print(f" | ERROR: {e}")
            continue
    
    print(f"[SCAN] Scan complete: {len(all_files)} total suitable files found")
    return all_files


def run_binary_search_test(file_path: Path, encoder: str, encoder_type: str, 
                          vmaf_target: float = 92.0, vmaf_min: float = 88.0,
                          min_qp: int = 18, max_qp: int = 32, 
                          test_num: int = 0) -> Dict[str, Any]:
    """Run traditional binary search QP optimization."""
    file_size_mb = file_path.stat().st_size / (1024*1024)
    print(f"[{test_num:2d}/20] BINARY   | {file_path.parent.name[:15]:15} | {file_path.name[:25]:25} | {file_size_mb:5.1f}MB | ", end="", flush=True)
    start_time = time.time()
    
    try:
        print("Starting...", end="", flush=True)
        result_qp = adaptive_qp_search_per_file(
            file_path, encoder, encoder_type,
            vmaf_target=vmaf_target,
            vmaf_min_threshold=vmaf_min,
            samples=3,  # Use same sample count
            sample_duration=60,  # Use same duration
            min_qp_limit=min_qp,
            max_qp_limit=max_qp,
            initial_step=4.0,
            min_step=1.0
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if result_qp is not None:
            print(f" QP {result_qp:2d} | {runtime:6.1f}s | SUCCESS")
            return {
                "success": True,
                "qp": result_qp,
                "runtime": runtime,
                "method": "binary_search",
                "file": str(file_path),
                "file_size": file_path.stat().st_size,
                "show": file_path.parent.name
            }
        else:
            print(f" FAIL | {runtime:6.1f}s | No QP found")
            return {
                "success": False,
                "runtime": runtime,
                "method": "binary_search",
                "file": str(file_path),
                "error": "No suitable QP found",
                "show": file_path.parent.name
            }
            
    except Exception as e:
        end_time = time.time()
        runtime = end_time - start_time
        print(f" FAIL | {runtime:6.1f}s | {str(e)[:30]}")
        return {
            "success": False,
            "runtime": runtime,
            "method": "binary_search",
            "file": str(file_path),
            "error": str(e),
            "show": file_path.parent.name
        }


def run_gradient_descent_test(file_path: Path, 
                             vmaf_target: float = 92.0, vmaf_min: float = 88.0,
                             min_qp: int = 18, max_qp: int = 32,
                             test_num: int = 0) -> Dict[str, Any]:
    """Run gradient descent QP optimization using the CLI."""
    file_size_mb = file_path.stat().st_size / (1024*1024)
    print(f"[{test_num:2d}/20] GRADIENT | {file_path.parent.name[:15]:15} | {file_path.name[:25]:25} | {file_size_mb:5.1f}MB | ", end="", flush=True)
    start_time = time.time()
    
    try:
        print("Starting...", end="", flush=True)
        # Use the CLI with gradient descent - same parameters as binary search
        cmd = [
            sys.executable, "-m", "lazy_transcode.core.transcode",
            "--path", str(file_path.parent),
            "--samples", "3",  # Same as binary search
            "--sample-duration", "60",  # Same as binary search
            "--vmaf-target", str(vmaf_target),
            "--vmaf-min", str(vmaf_min),
            "--min-qp", str(min_qp),
            "--max-qp", str(max_qp),
            "--gradient-descent",
            "--dry-run",  # Don't actually transcode
            "--auto-yes"  # No prompts
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        runtime = end_time - start_time
        
        # Parse the output for QP result
        output_lines = result.stdout.split('\n') if result.stdout else []
        qp_result = None
        
        for line in output_lines:
            if "[RESULT]" in line and "QP" in line:
                # Extract QP from result line
                import re
                qp_match = re.search(r'QP (\d+)', line)
                if qp_match:
                    qp_result = int(qp_match.group(1))
                    break
        
        if qp_result is not None and result.returncode == 0:
            print(f" QP {qp_result:2d} | {runtime:6.1f}s | SUCCESS")
            return {
                "success": True,
                "qp": qp_result,
                "runtime": runtime,
                "method": "gradient_descent",
                "file": str(file_path),
                "file_size": file_path.stat().st_size,
                "show": file_path.parent.name
            }
        else:
            error_msg = "Parse failed" if qp_result is None else f"RC {result.returncode}"
            if result.stderr:
                error_msg += f" | {result.stderr[:50]}"
            print(f" FAIL | {runtime:6.1f}s | {error_msg}")
            return {
                "success": False,
                "runtime": runtime,
                "method": "gradient_descent",
                "file": str(file_path),
                "error": error_msg,
                "show": file_path.parent.name,
                "stdout_sample": result.stdout[-300:] if result.stdout else "",
                "stderr_sample": result.stderr[-300:] if result.stderr else ""
            }
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        runtime = end_time - start_time
        print(f" FAIL | {runtime:6.1f}s | TIMEOUT")
        return {
            "success": False,
            "runtime": runtime,
            "method": "gradient_descent",
            "file": str(file_path),
            "error": "Timeout",
            "show": file_path.parent.name
        }
    except Exception as e:
        end_time = time.time()
        runtime = end_time - start_time
        print(f" FAIL | {runtime:6.1f}s | {str(e)[:30]}")
        return {
            "success": False,
            "runtime": runtime,
            "method": "gradient_descent",
            "file": str(file_path),
            "error": str(e),
            "show": file_path.parent.name
        }


def main():
    parser = argparse.ArgumentParser(description="Compare QP optimization techniques on exactly 20 random files")
    parser.add_argument("--root", help="Root directory to scan for videos (defaults to config test_root)")
    parser.add_argument("--vmaf-target", type=float, default=92.0, help="VMAF target (looser than default)")
    parser.add_argument("--vmaf-min", type=float, default=88.0, help="VMAF minimum (looser than default)")
    parser.add_argument("--min-qp", type=int, default=18, help="Minimum QP")
    parser.add_argument("--max-qp", type=int, default=32, help="Maximum QP")
    parser.add_argument("--output", default="qp_comparison_results.json", help="Output file for results")
    args = parser.parse_args()
    
    # Get root directory
    if args.root:
        root_path = Path(args.root)
    else:
        root_path = get_test_root()
        if not root_path:
            print("[ERROR] No root path specified and no test_root in config")
            sys.exit(1)
    
    if not root_path.exists():
        print(f"[ERROR] Root path does not exist: {root_path}")
        sys.exit(1)
    
    print(f"[INFO] Starting QP optimization comparison test")
    print(f"[INFO] Root directory: {root_path}")
    print(f"[INFO] Testing exactly 20 files with both techniques")
    print(f"[INFO] VMAF target: {args.vmaf_target}, minimum: {args.vmaf_min}")
    print(f"[INFO] QP range: {args.min_qp} - {args.max_qp}")
    print()
    
    # DO NOT clean up the root directory - only individual show directories
    print("[SETUP] Skipping cleanup on root directory (will clean individual show dirs only)")
    
    # Detect encoder once
    print("[SETUP] Detecting HEVC encoder...")
    try:
        encoder, encoder_type = detect_hevc_encoder()
        print(f"[SETUP] Using encoder: {encoder} ({encoder_type})")
    except Exception as e:
        print(f"[SETUP] ERROR: Could not detect encoder: {e}")
        sys.exit(1)
    print()
    
    # Find video files and select randomly each run
    print("[SETUP] Searching for video files...")
    all_video_files = find_video_files(root_path, max_files_per_show=3)
    
    if len(all_video_files) < 20:
        print(f"[ERROR] Only found {len(all_video_files)} suitable files, need at least 20 available")
        if len(all_video_files) > 0:
            print("[INFO] Available files:")
            for f in all_video_files[:10]:
                print(f"  {f.parent.name}: {f.name}")
        sys.exit(1)
    
    # Shuffle for random sampling
    print(f"[SETUP] Shuffling {len(all_video_files)} available files for random selection")
    random.shuffle(all_video_files)
    
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "root_path": str(root_path),
            "encoder": encoder,
            "encoder_type": encoder_type,
            "vmaf_target": args.vmaf_target,
            "vmaf_min": args.vmaf_min,
            "min_qp": args.min_qp,
            "max_qp": args.max_qp,
            "total_available_files": len(all_video_files)
        },
        "successful_tests": [],  # Only store successful comparisons
        "failed_attempts": [],   # Track failed attempts
        "summary": {}
    }
    
    print()
    print("=" * 110)
    print("QP OPTIMIZATION TECHNIQUE COMPARISON - SEARCHING FOR 20 VIABLE FILES")
    print("=" * 110)
    print("Test | Method   | Show           | File                     | Size  | Progress/Result       | Status")
    print("-" * 110)
    
    total_start_time = time.time()
    
    successful_comparisons = 0
    file_index = 0
    
    # Keep testing files until we get 20 successful comparisons
    while successful_comparisons < 20 and file_index < len(all_video_files):
        test_file = all_video_files[file_index]
        file_index += 1
        
        print(f"\n[ATTEMPT {file_index}] Starting tests on: {test_file.parent.name}/{test_file.name}")
        
        # Clean up temp files before each test
        print(f"[ATTEMPT {file_index}] Cleaning temporary files...")
        _startup_scavenge(test_file.parent)
        
        # Test with binary search first
        binary_result = run_binary_search_test(
            test_file, encoder, encoder_type,
            args.vmaf_target, args.vmaf_min, args.min_qp, args.max_qp, 
            successful_comparisons + 1
        )
        
        # Clean up temp files between methods
        print(f"[ATTEMPT {file_index}] Cleaning between methods...")
        _startup_scavenge(test_file.parent)
        time.sleep(0.5)  # Brief pause to ensure cleanup
        
        # Test with gradient descent
        gradient_result = run_gradient_descent_test(
            test_file, args.vmaf_target, args.vmaf_min, args.min_qp, args.max_qp, 
            successful_comparisons + 1
        )
        
        # Check if both methods succeeded
        if binary_result["success"] and gradient_result["success"]:
            successful_comparisons += 1
            
            # Store as successful test with comparison data
            comparison_data = {
                "test_number": successful_comparisons,
                "file": str(test_file),
                "show": test_file.parent.name,
                "file_size_mb": test_file.stat().st_size / (1024*1024),
                "binary_search": binary_result,
                "gradient_descent": gradient_result,
                "runtime_difference": gradient_result["runtime"] - binary_result["runtime"],
                "qp_difference": gradient_result["qp"] - binary_result["qp"],
                "binary_faster": binary_result["runtime"] < gradient_result["runtime"]
            }
            results["successful_tests"].append(comparison_data)
            
            runtime_winner = "Binary" if binary_result["runtime"] < gradient_result["runtime"] else "Gradient"
            runtime_diff = abs(gradient_result["runtime"] - binary_result["runtime"])
            qp_diff = gradient_result["qp"] - binary_result["qp"]
            
            print(f"[SUCCESS] ✓ COMPARISON {successful_comparisons}/20 COMPLETE")
            print(f"          Binary: QP {binary_result['qp']} in {binary_result['runtime']:.1f}s")
            print(f"          Gradient: QP {gradient_result['qp']} in {gradient_result['runtime']:.1f}s")
            print(f"          Winner: {runtime_winner} (faster by {runtime_diff:.1f}s), QP diff: {qp_diff:+d}")
            
        else:
            # Store as failed attempt
            failed_data = {
                "file": str(test_file),
                "show": test_file.parent.name,
                "binary_success": binary_result["success"],
                "gradient_success": gradient_result["success"],
                "binary_error": binary_result.get("error") if not binary_result["success"] else None,
                "gradient_error": gradient_result.get("error") if not gradient_result["success"] else None
            }
            results["failed_attempts"].append(failed_data)
            
            print(f"[FAILED] ✗ COMPARISON FAILED for {test_file.name}")
            if not binary_result["success"]:
                print(f"         Binary failed: {binary_result.get('error', 'unknown error')}")
            if not gradient_result["success"]:
                print(f"         Gradient failed: {gradient_result.get('error', 'unknown error')}")
        
        # Final cleanup after both tests
        print(f"[ATTEMPT {file_index}] Final cleanup...")
        _startup_scavenge(test_file.parent)
        
        # Progress update every few files
        elapsed = time.time() - total_start_time
        if file_index % 3 == 0 or successful_comparisons >= 20:
            success_rate = successful_comparisons / file_index * 100 if file_index > 0 else 0
            estimated_total_time = elapsed * 20 / successful_comparisons if successful_comparisons > 0 else 0
            print(f"[PROGRESS] Files tested: {file_index} | Successful comparisons: {successful_comparisons}/20 ({success_rate:.1f}%)")
            print(f"[PROGRESS] Elapsed: {elapsed/60:.1f}min | Est. total: {estimated_total_time/60:.1f}min")
            print()
    
    total_runtime = time.time() - total_start_time
    
    if successful_comparisons < 20:
        print(f"[WARNING] Only achieved {successful_comparisons}/20 successful comparisons after testing {file_index} files")
    else:
        print(f"[SUCCESS] Completed 20 successful comparisons after testing {file_index} files")
    
    # Calculate summary statistics from successful tests
    if successful_comparisons > 0:
        binary_runtimes = [test["binary_search"]["runtime"] for test in results["successful_tests"]]
        gradient_runtimes = [test["gradient_descent"]["runtime"] for test in results["successful_tests"]]
        binary_qps = [test["binary_search"]["qp"] for test in results["successful_tests"]]
        gradient_qps = [test["gradient_descent"]["qp"] for test in results["successful_tests"]]
        
        binary_avg_runtime = sum(binary_runtimes) / len(binary_runtimes)
        gradient_avg_runtime = sum(gradient_runtimes) / len(gradient_runtimes)
        binary_avg_qp = sum(binary_qps) / len(binary_qps)
        gradient_avg_qp = sum(gradient_qps) / len(gradient_qps)
        
        binary_total_runtime = sum(binary_runtimes)
        gradient_total_runtime = sum(gradient_runtimes)
        
        # Count how many times each method was faster
        binary_faster_count = sum(1 for test in results["successful_tests"] if test["binary_faster"])
        gradient_faster_count = successful_comparisons - binary_faster_count
        
        # Calculate QP differences
        qp_differences = [test["qp_difference"] for test in results["successful_tests"]]
        avg_qp_difference = sum(qp_differences) / len(qp_differences) if qp_differences else 0
        
    else:
        binary_avg_runtime = gradient_avg_runtime = 0
        binary_avg_qp = gradient_avg_qp = 0
        binary_total_runtime = gradient_total_runtime = 0
        binary_faster_count = gradient_faster_count = 0
        avg_qp_difference = 0
    
    results["summary"] = {
        "total_test_time": total_runtime,
        "files_tested": file_index,
        "successful_comparisons": successful_comparisons,
        "failed_attempts": len(results["failed_attempts"]),
        "success_rate": successful_comparisons / file_index * 100 if file_index > 0 else 0,
        "binary_search": {
            "avg_runtime": binary_avg_runtime,
            "avg_qp": binary_avg_qp,
            "total_runtime": binary_total_runtime,
            "faster_count": binary_faster_count
        },
        "gradient_descent": {
            "avg_runtime": gradient_avg_runtime,
            "avg_qp": gradient_avg_qp,
            "total_runtime": gradient_total_runtime,
            "faster_count": gradient_faster_count
        },
        "comparison": {
            "avg_qp_difference": avg_qp_difference,
            "performance_ratio": binary_avg_runtime / gradient_avg_runtime if gradient_avg_runtime > 0 else 0
        }
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(f"Total Test Time: {total_runtime:.1f}s ({total_runtime/60:.1f} minutes)")
    print(f"Files Tested: {file_index} | Successful Comparisons: {successful_comparisons}/20 | Success Rate: {successful_comparisons/file_index*100:.1f}%")
    print()
    
    if successful_comparisons > 0:
        print(f"{'Method':<15} | {'Avg Time':<8} | {'Avg QP':<6} | {'Total Time':<10} | {'Faster':<6}")
        print("-" * 65)
        print(f"{'Binary Search':<15} | {binary_avg_runtime:7.1f}s | {binary_avg_qp:5.1f} | {binary_total_runtime:9.1f}s | {binary_faster_count:2d}/20")
        print(f"{'Gradient Desc':<15} | {gradient_avg_runtime:7.1f}s | {gradient_avg_qp:5.1f} | {gradient_total_runtime:9.1f}s | {gradient_faster_count:2d}/20")
        
        print()
        if binary_avg_runtime > 0 and gradient_avg_runtime > 0:
            speedup = binary_avg_runtime / gradient_avg_runtime
            if speedup > 1:
                print(f"🚀 Gradient Descent is {speedup:.2f}x FASTER than Binary Search")
            else:
                print(f"🚀 Binary Search is {1/speedup:.2f}x FASTER than Gradient Descent")
            
            print(f"📊 Average QP difference: {avg_qp_difference:+.1f} (Gradient - Binary)")
            print(f"⚡ Gradient Descent was faster in {gradient_faster_count}/20 tests ({gradient_faster_count/20*100:.1f}%)")
        
        print()
        print("Individual Test Results:")
        print("-" * 100)
        print("Test | Show           | Binary QP/Time | Gradient QP/Time | Faster Method | QP Diff | Time Diff")
        print("-" * 100)
        
        for i, test in enumerate(results["successful_tests"][:10], 1):  # Show first 10
            binary = test["binary_search"]
            gradient = test["gradient_descent"]
            faster = "Gradient" if test["binary_faster"] == False else "Binary"
            print(f"{i:4d} | {test['show'][:14]:14} | {binary['qp']:2d} / {binary['runtime']:5.1f}s | {gradient['qp']:2d} / {gradient['runtime']:7.1f}s | {faster:8} | {test['qp_difference']:+3.0f} | {test['runtime_difference']:+6.1f}s")
        
        if len(results["successful_tests"]) > 10:
            print(f"     ... and {len(results['successful_tests']) - 10} more tests")
    
    else:
        print("No successful comparisons completed.")
    
    print()
    print(f"Detailed results saved to: {args.output}")
    print("=" * 100)


if __name__ == "__main__":
    main()
