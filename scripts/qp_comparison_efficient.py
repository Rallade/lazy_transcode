#!/usr/bin/env python3
"""
QP Optimization Technique Comparison Test - Efficient Pipeline

Pipeline:
1. Get show directories from M:/Shows
2. For each show, test up to 10 random files until BOTH methods succeed on one file
3. Continue until 20 successful comparisons are completed

This ensures we test both methods on the same files and don't waste time on files
that fail with one method.
"""

import argparse
import json
import multiprocessing
import random
import time
from pathlib import Path
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from lazy_transcode.core.transcode import (
    detect_hevc_encoder,
    get_video_codec,
    should_skip_codec,
    adaptive_qp_search_per_file,
    gradient_descent_qp_search,
    _startup_scavenge,
)
from lazy_transcode.config import get_test_root
from lazy_transcode.utils.logging import (
    log_info, log_error, log_step, log_progress, log_warn,
    log_comparison_start, log_encoder_setup, log_show_scan_start, 
    log_show_files_found, log_file_test_start, log_method_test_start,
    log_method_success, log_method_failure, log_file_success, log_show_skip,
    log_comparison_result, log_final_results, print_results_table,
    log_performance_comparison, print_recent_results, handle_systematic_error,
    create_show_progress_bar, create_overall_progress_bar, format_duration,
    print_section_header, print_separator, create_progress_bar, log_setup,
    # Detailed progress logging functions
    log_gradient_descent_start, log_gradient_descent_params, log_clips_extracted,
    log_gradient_descent_converged, log_gradient_descent_result,
    create_gradient_descent_progress_bar, update_gradient_descent_progress,
    log_binary_search_start, log_binary_search_step, create_adaptive_qp_progress_bar,
    update_adaptive_qp_progress, log_transcoding_progress, parse_qp_from_output
)


def get_show_directories(root_path: Path) -> List[Path]:
    """Step 1: Get show directories from root path."""
    log_step(1, f"Finding show directories in {root_path}")
    
    show_dirs = [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    log_step(1, f"Found {len(show_dirs)} show directories")
    
    # Shuffle for random order
    random.shuffle(show_dirs)
    
    return show_dirs


def find_viable_file_in_show(show_dir: Path, encoder: str, encoder_type: str,
                            vmaf_target: float, vmaf_min: float, 
                            min_qp: int, max_qp: int, vmaf_threads: int) -> Optional[Tuple[Path, Dict, Dict]]:
    """
    Step 2: Find one viable file in a show where BOTH methods succeed.
    
    Tests only 1 random file in the show, then moves to next show.
    Only proceeds to gradient descent if binary search succeeds first.
    Returns (file_path, binary_result, gradient_result) if both succeed, None otherwise.
    """
    video_extensions = [".mkv", ".mp4", ".mov", ".ts"]
    max_attempts = 1  # Only try 1 file per show
    
    log_show_scan_start(show_dir.name)
    
    # Find all video files in show
    show_files = []
    for ext in video_extensions:
        show_files.extend(list(show_dir.rglob(f"*{ext}")))
    
    # Filter out temp/sample files and files with efficient codecs
    filtered_files = []
    for f in show_files:
        if any([
            f.name.startswith('._'), f.name.startswith('.'),
            ".sample_clip" in f.stem, "_sample" in f.stem,
            ".clip" in f.stem and ".sample" in f.stem,
            f.stem.count(".sample") > 0, f.stem.count(".clip") > 0
        ]):
            continue
            
        try:
            codec = get_video_codec(f)
            if not should_skip_codec(codec):
                filtered_files.append(f)
        except Exception:
            continue
    
    if not filtered_files:
        log_show_files_found(show_dir.name, 0)
        return None
    
    # Test only the first random file
    random.shuffle(filtered_files)
    test_files = filtered_files[:max_attempts]  # Only 1 file now
    
    log_show_files_found(show_dir.name, len(test_files))
    
    # Test the single file
    if not test_files:
        return None
        
    test_file = test_files[0]
    
    # Quick validation
    if not test_file.exists():
        print(f"      File missing: {test_file.name}")
        return None
        
    file_size_mb = test_file.stat().st_size / (1024*1024)
    print(f"      [1/1] {test_file.name[:50]}... ({file_size_mb:.0f}MB)")
    
    # Clean up temp files in this show directory only BEFORE starting both tests
    _startup_scavenge(show_dir)
    
    # Test gradient descent first (it creates clip files)
    print(f"        → Gradient Descent testing {test_file.name[:30]}...")
    gradient_result = test_gradient_descent(test_file, vmaf_target, vmaf_min, min_qp, max_qp, 16)  # Use 16 threads

    if not gradient_result["success"]:
        # Check for systematic errors that should cause program exit
        error_type = gradient_result.get("error_type")
        if error_type in ["file_not_found", "permission_error", "disk_full", "clip_extraction_failed"]:
            log_error(gradient_result['error'])
            log_error(f"Error type: {error_type}")
            log_error("This is a systematic issue that prevents testing. Exiting.")
            if "stderr" in gradient_result and gradient_result["stderr"]:
                log_info(f"Debug info: {gradient_result['stderr'][:200]}")
            sys.exit(1)
        
        print(f"        ✗ Gradient Descent FAILED: {gradient_result.get('error', 'unknown')[:40]}")
        # Clean up after gradient descent failure
        _startup_scavenge(show_dir)
        return None

    print(f"        ✓ Gradient Descent SUCCESS: QP{gradient_result['qp']}, {format_duration(gradient_result['runtime'])}")

    # Test binary search only if gradient descent succeeded
    print(f"        → Binary Search testing {test_file.name[:30]}...")
    binary_result = test_binary_search(test_file, encoder, encoder_type, 
                                     vmaf_target, vmaf_min, min_qp, max_qp, 16)  # Use 16 threads
    
    if not binary_result["success"]:
        # Check for systematic errors that should cause program exit
        error_type = binary_result.get("error_type")
        if error_type in ["file_not_found", "permission_error", "disk_full", "clip_extraction_failed"]:
            log_error(binary_result['error'])
            log_error(f"Error type: {error_type}")
            log_error("This is a systematic issue that prevents testing. Exiting.")
            if "stderr" in binary_result and binary_result["stderr"]:
                log_info(f"Debug info: {binary_result['stderr'][:200]}")
            sys.exit(1)
        
        print(f"        ✗ Binary Search FAILED: {binary_result.get('error', 'unknown')[:40]}")
        # Clean up after both tests
        _startup_scavenge(show_dir)
        return None

    # Both succeeded! Clean up before returning
    _startup_scavenge(show_dir)
    print(f"        ✓ Binary Search SUCCESS: QP{binary_result['qp']}, {format_duration(binary_result['runtime'])}")
    print(f"      🎉 SUCCESS! Both methods completed on {test_file.name[:40]}")
    print(f"         → Gradient Descent: QP{gradient_result['qp']} in {format_duration(gradient_result['runtime'])}")
    print(f"         → Binary Search: QP{binary_result['qp']} in {format_duration(binary_result['runtime'])}")
    
    return test_file, binary_result, gradient_result


def test_binary_search(file_path: Path, encoder: str, encoder_type: str, 
                      vmaf_target: float, vmaf_min: float,
                      min_qp: int, max_qp: int, vmaf_threads: int) -> Dict[str, Any]:
    """Test binary search optimization with detailed error reporting and progress logging."""
    start_time = time.time()
    
    # Log binary search start
    log_binary_search_start(file_path.name, vmaf_target, vmaf_min)
    
    try:
        result_qp = adaptive_qp_search_per_file(
            file_path, encoder, encoder_type,
            vmaf_target=vmaf_target,
            vmaf_min_threshold=vmaf_min,
            samples=3, sample_duration=60,
            min_qp_limit=min_qp, max_qp_limit=max_qp,
            initial_step=4.0, min_step=1.0,
            vmaf_threads=vmaf_threads,
            use_gradient_descent=False  # Force binary search, not gradient descent!
        )
        
        runtime = time.time() - start_time
        
        if result_qp is None:
            return {
                "success": False, "qp": 0, "runtime": runtime,
                "method": "binary_search", "file": str(file_path),
                "show": file_path.parent.name, "file_size": file_path.stat().st_size,
                "vmaf_threads_used": vmaf_threads,
                "error": "No suitable QP found within range", "error_type": "no_qp_found"
            }
        
        return {
            "success": True, "qp": result_qp, "runtime": runtime,
            "method": "binary_search", "file": str(file_path),
            "show": file_path.parent.name, "file_size": file_path.stat().st_size,
            "vmaf_threads_used": vmaf_threads, "error": None, "error_type": None
        }
        
    except FileNotFoundError as e:
        runtime = time.time() - start_time
        return {
            "success": False, "qp": 0, "runtime": runtime,
            "method": "binary_search", "file": str(file_path),
            "show": file_path.parent.name, 
            "error": f"File not found: {str(e)[:50]}", "error_type": "file_not_found"
        }
    except PermissionError as e:
        runtime = time.time() - start_time
        return {
            "success": False, "qp": 0, "runtime": runtime,
            "method": "binary_search", "file": str(file_path),
            "show": file_path.parent.name,
            "error": f"Permission denied: {str(e)[:50]}", "error_type": "permission_error"
        }
    except Exception as e:
        runtime = time.time() - start_time
        error_str = str(e)
        
        # Categorize error types
        if "disk" in error_str.lower() and ("full" in error_str.lower() or "space" in error_str.lower()):
            error_type = "disk_full"
        elif "vmaf" in error_str.lower():
            error_type = "vmaf_error"
        else:
            error_type = "unknown_error"
            
        return {
            "success": False, "qp": 0, "runtime": runtime,
            "method": "binary_search", "file": str(file_path),
            "show": file_path.parent.name,
            "error": error_str[:100], "error_type": error_type
        }


def test_gradient_descent(file_path: Path, vmaf_target: float, vmaf_min: float,
                         min_qp: int, max_qp: int, vmaf_threads: int) -> Dict[str, Any]:
    """Test gradient descent optimization using direct function call."""
    start_time = time.time()
    
    # Log gradient descent start with detailed info
    log_gradient_descent_start(file_path.name, vmaf_target, vmaf_min)
    log_clips_extracted(3)  # Log that 3 clips will be extracted
    
    try:
        # Get encoder info (we need this for the function call)
        encoder, encoder_type = detect_hevc_encoder()
        
        # Call the gradient descent function directly
        result_qp, result_details = gradient_descent_qp_search(
            file=file_path,
            encoder=encoder,
            encoder_type=encoder_type,
            samples=3,
            sample_duration=60,
            vmaf_target=vmaf_target,
            vmaf_min_threshold=vmaf_min,
            min_qp_limit=min_qp,
            max_qp_limit=max_qp,
            vmaf_threads=vmaf_threads
        )
        
        runtime = time.time() - start_time
        
        if result_qp is None or result_qp == 0:
            return {
                "success": False, "qp": 0, "runtime": runtime,
                "method": "gradient_descent", "file": str(file_path),
                "show": file_path.parent.name, "file_size": file_path.stat().st_size,
                "vmaf_threads_used": vmaf_threads,
                "error": "Gradient descent failed to find suitable QP", "error_type": "no_qp_found",
                "details": result_details
            }
        
        # Log successful gradient descent result
        if result_details:
            log_gradient_descent_result(
                file_path.name, result_qp, 
                result_details.get('mean_vmaf', 0.0), 
                result_details.get('loss', 0.0),
                result_details.get('min_vmaf', 0.0), 
                result_details.get('worst_drop', 0.0), 
                result_details.get('size_pct', 0.0), 
                result_details.get('samples', 3)
            )
        
        return {
            "success": True, "qp": result_qp, "runtime": runtime,
            "method": "gradient_descent", "file": str(file_path),
            "show": file_path.parent.name, "file_size": file_path.stat().st_size,
            "vmaf_threads_used": vmaf_threads, "error": None, "error_type": None,
            "details": result_details
        }
        
    except FileNotFoundError as e:
        runtime = time.time() - start_time
        return {
            "success": False, "qp": 0, "runtime": runtime,
            "method": "gradient_descent", "file": str(file_path),
            "show": file_path.parent.name, "file_size": file_path.stat().st_size,
            "vmaf_threads_used": vmaf_threads,
            "error": f"File not found: {str(e)[:50]}", "error_type": "file_not_found"
        }
    except PermissionError as e:
        runtime = time.time() - start_time
        return {
            "success": False, "qp": 0, "runtime": runtime,
            "method": "gradient_descent", "file": str(file_path),
            "show": file_path.parent.name, "file_size": file_path.stat().st_size,
            "vmaf_threads_used": vmaf_threads,
            "error": f"Permission denied: {str(e)[:50]}", "error_type": "permission_error"
        }
    except Exception as e:
        runtime = time.time() - start_time
        error_str = str(e)
        
        # Categorize error types
        if "disk" in error_str.lower() and ("full" in error_str.lower() or "space" in error_str.lower()):
            error_type = "disk_full"
        elif "vmaf" in error_str.lower():
            error_type = "vmaf_error"
        else:
            error_type = "unknown_error"
            
        return {
            "success": False, "qp": 0, "runtime": runtime,
            "method": "gradient_descent", "file": str(file_path),
            "show": file_path.parent.name, "file_size": file_path.stat().st_size,
            "vmaf_threads_used": vmaf_threads,
            "error": error_str[:100], "error_type": error_type
        }


def main():
    parser = argparse.ArgumentParser(description="Efficient QP optimization comparison on 20 files")
    parser.add_argument("--root", help="Root directory (defaults to config test_root)")
    parser.add_argument("--vmaf-target", type=float, default=95.0, help="VMAF target")
    parser.add_argument("--vmaf-min", type=float, default=88.0, help="VMAF minimum")
    parser.add_argument("--min-qp", type=int, default=14, help="Minimum QP")
    parser.add_argument("--max-qp", type=int, default=26, help="Maximum QP")
    parser.add_argument("--output", default="qp_comparison_results.json", help="Output file")
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
        log_error(f"Root path does not exist: {root_path}")
        sys.exit(1)
    
    # Use the new logging functions
    log_comparison_start(20, args.vmaf_target, args.vmaf_min, (args.min_qp, args.max_qp))
    log_info(f"Root: {root_path}")
    print()
    
    # Detect encoder
    encoder, encoder_type = detect_hevc_encoder()
    vmaf_threads = 16
    log_encoder_setup(encoder, encoder_type, vmaf_threads)
    print()
    
    # Step 1: Get show directories (no cleanup on root)
    show_dirs = get_show_directories(root_path)
    if len(show_dirs) == 0:
        log_error("No show directories found")
        sys.exit(1)
    
    print()
    
    # Results storage
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "root_path": str(root_path),
            "encoder": encoder, "encoder_type": encoder_type,
            "vmaf_target": args.vmaf_target, "vmaf_min": args.vmaf_min,
            "min_qp": args.min_qp, "max_qp": args.max_qp,
            "total_shows": len(show_dirs)
        },
        "successful_comparisons": [],
        "failed_shows": [],
        "summary": {}
    }
    
    total_start_time = time.time()
    successful_comparisons = 0
    shows_tested = 0
    
    # Step 3: Continue until 20 successful comparisons
    log_info("Starting comparison tests...")
    
    # Create overall progress bar
    with create_progress_bar(total=20, desc="QP Comparison Progress", unit="comparisons") as overall_pbar:
        for show_dir in show_dirs:
            if successful_comparisons >= 20:
                break
                
            shows_tested += 1
            
            # Step 2: Find viable file in this show
            viable_result = find_viable_file_in_show(
                show_dir, encoder, encoder_type,
                args.vmaf_target, args.vmaf_min, args.min_qp, args.max_qp, vmaf_threads
            )
            
            if viable_result is None:
                results["failed_shows"].append(show_dir.name)
                overall_pbar.write(f"    ⚠️  Failed to find viable files in {show_dir.name}")
                continue
            
            # Successful comparison!
            test_file, binary_result, gradient_result = viable_result
            successful_comparisons += 1
            # Extract QP from binary search result (it returns a tuple)
            binary_qp = binary_result["qp"][0] if isinstance(binary_result["qp"], tuple) else binary_result["qp"]
            gradient_qp = gradient_result["qp"]
            
            comparison_data = {
                "comparison_number": successful_comparisons,
                "show": show_dir.name,
                "file": test_file.name,
                "file_path": str(test_file),
                "file_size_mb": test_file.stat().st_size / (1024*1024),
                "binary_search": binary_result,
                "gradient_descent": gradient_result,
                "qp_difference": gradient_qp - binary_qp,
                "runtime_difference": gradient_result["runtime"] - binary_result["runtime"],
                "binary_faster": binary_result["runtime"] < gradient_result["runtime"]
            }
            results["successful_comparisons"].append(comparison_data)
            
            # Update progress bar
            overall_pbar.update(1)
            overall_pbar.set_description(f"QP Comparisons ({successful_comparisons}/20)")
            
            # Progress update
            elapsed = time.time() - total_start_time
            faster = "G" if not comparison_data["binary_faster"] else "B"
            overall_pbar.write(f"    ✅ Comparison {successful_comparisons}/20: "
                              f"B:QP{binary_result['qp']} ({format_duration(binary_result['runtime'])}) "
                              f"G:QP{gradient_result['qp']} ({format_duration(gradient_result['runtime'])}) "
                              f"| {faster} faster | {format_duration(elapsed)} elapsed")
            
            if successful_comparisons % 5 == 0:
                log_progress(f"{successful_comparisons}/20 comparisons complete "
                           f"({shows_tested} shows tested, {format_duration(elapsed)} elapsed)")
    
    total_runtime = time.time() - total_start_time
    
    # Calculate summary
    if successful_comparisons > 0:
        comparisons = results["successful_comparisons"]
        
        binary_runtimes = [c["binary_search"]["runtime"] for c in comparisons]
        gradient_runtimes = [c["gradient_descent"]["runtime"] for c in comparisons]
        
        # Extract QPs properly (binary search returns tuples)
        binary_qps = []
        gradient_qps = []
        for c in comparisons:
            binary_qp = c["binary_search"]["qp"]
            if isinstance(binary_qp, tuple):
                binary_qps.append(binary_qp[0])
            else:
                binary_qps.append(binary_qp)
            gradient_qps.append(c["gradient_descent"]["qp"])
        
        binary_faster_count = sum(1 for c in comparisons if c["binary_faster"])
        
        results["summary"] = {
            "total_runtime": total_runtime,
            "shows_tested": shows_tested,
            "successful_comparisons": successful_comparisons,
            "failed_shows": len(results["failed_shows"]),
            "binary_search": {
                "avg_runtime": sum(binary_runtimes) / len(binary_runtimes),
                "avg_qp": sum(binary_qps) / len(binary_qps),
                "total_runtime": sum(binary_runtimes),
                "faster_count": binary_faster_count
            },
            "gradient_descent": {
                "avg_runtime": sum(gradient_runtimes) / len(gradient_runtimes),
                "avg_qp": sum(gradient_qps) / len(gradient_qps), 
                "total_runtime": sum(gradient_runtimes),
                "faster_count": successful_comparisons - binary_faster_count
            }
        }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print()
    print_section_header("FINAL RESULTS")
    log_info(f"Total Time: {format_duration(total_runtime)}")
    log_info(f"Shows Tested: {shows_tested}")
    log_info(f"Successful Comparisons: {successful_comparisons}/20")
    log_info(f"Failed Shows: {len(results['failed_shows'])}")
    
    if successful_comparisons > 0:
        summary = results["summary"]
        binary = summary["binary_search"]
        gradient = summary["gradient_descent"]
        
        print()
        print(f"{'Method':<15} | {'Avg Runtime':<11} | {'Avg QP':<7} | {'Faster Count':<12}")
        print_separator(55)
        print(f"{'Binary Search':<15} | {format_duration(binary['avg_runtime']):<11} | {binary['avg_qp']:6.1f} | {binary['faster_count']:2d}/20")
        print(f"{'Gradient Desc':<15} | {format_duration(gradient['avg_runtime']):<11} | {gradient['avg_qp']:6.1f} | {gradient['faster_count']:2d}/20")
        
        print()
        speedup = binary["avg_runtime"] / gradient["avg_runtime"]
        if speedup > 1:
            log_info(f"🚀 Gradient Descent is {speedup:.2f}x FASTER than Binary Search")
        else:
            log_info(f"🚀 Binary Search is {1/speedup:.2f}x FASTER than Gradient Descent")
        
        avg_qp_diff = sum(c["qp_difference"] for c in comparisons) / len(comparisons)
        log_info(f"📊 Average QP difference: {avg_qp_diff:+.1f} (Gradient - Binary)")
        log_info(f"⚡ Gradient Descent was faster in {gradient['faster_count']}/20 tests ({gradient['faster_count']/20*100:.1f}%)")
        
        print()
        log_info("Recent Results:")
        print_separator()
        for c in comparisons[-10:]:  # Last 10 results
            faster = "G" if not c["binary_faster"] else "B"
            print(f"{c['comparison_number']:2d} | {c['show'][:20]:20} | B:QP{c['binary_search']['qp']:2d}({format_duration(c['binary_search']['runtime'])}) G:QP{c['gradient_descent']['qp']:2d}({format_duration(c['gradient_descent']['runtime'])}) | {faster}")
        
    print()
    log_info(f"Detailed results saved to: {args.output}")
    print_separator()


if __name__ == "__main__":
    main()
