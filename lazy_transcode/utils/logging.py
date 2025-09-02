"""
Logging utilities for lazy_transcode

Provides consistent logging patterns used throughout the codebase:
- [INFO] for general information
- [WARN] for warnings
- [ERROR] for errors  
- [RESULT] for final results
- [GRADIENT-DESCENT] for gradient descent specific messages
- [CLEANUP] for cleanup operations
- Progress bars with tqdm
- Comparison result formatting
- Status indicators and emojis
"""

import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from tqdm import tqdm as TqdmProgressBar
    tqdm = TqdmProgressBar
except ImportError:  # fallback minimal stub
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, position=None, leave=True, unit=None, unit_scale=False):
            self.iterable = iterable or range(total or 0)
            self.desc = desc or ""
            self.total = total
            self._n = 0
            
        def __iter__(self):
            for x in self.iterable:
                yield x
                
        def update(self, n=1):
            self._n += n
            if self.total:
                print(f"\r{self.desc}: {self._n}/{self.total}", end="", flush=True)
            
        def close(self):
            if self.total:
                print()  # New line after progress
                
        def set_postfix(self, **kw):
            pass
            
        def set_description(self, desc):
            self.desc = desc
            
        def write(self, s):
            print(s)
            
        def __enter__(self):
            return self
            
        def __exit__(self, *exc):
            self.close()
            return False


def log_info(message: str):
    """Log an informational message"""
    print(f"[INFO] {message}")


def log_warn(message: str):
    """Log a warning message"""
    print(f"[WARN] {message}")


def log_error(message: str):
    """Log an error message"""
    print(f"[ERROR] {message}")


def log_result(message: str):
    """Log a result message"""
    print(f"[RESULT] {message}")


def log_gradient_descent(message: str):
    """Log a gradient descent specific message"""
    print(f"[GRADIENT-DESCENT] {message}")


def log_cleanup(message: str):
    """Log a cleanup operation"""
    print(f"[CLEANUP] {message}")


def log_step(step_number: int, message: str):
    """Log a step in a multi-step process"""
    print(f"[STEP {step_number}] {message}")


def log_setup(message: str):
    """Log a setup operation"""
    print(f"[SETUP] {message}")


def log_progress(message: str):
    """Log a progress update"""
    print(f"[PROGRESS] {message}")


def create_progress_bar(total: Optional[int] = None, desc: str = "", unit: str = "it", 
                       position: Optional[int] = None, leave: bool = True) -> tqdm:
    """Create a progress bar with consistent styling"""
    return tqdm(total=total, desc=desc, unit=unit, position=position, leave=leave)


def print_section_header(title: str, width: int = 90):
    """Print a section header with consistent formatting"""
    print("=" * width)
    print(title)
    print("=" * width)


def print_separator(width: int = 90):
    """Print a separator line"""
    print("-" * width)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string"""
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


# QP Comparison specific logging functions
def log_comparison_start(total_target: int, vmaf_target: float, vmaf_min: float, qp_range: tuple):
    """Log the start of a QP comparison test"""
    print_section_header("QP OPTIMIZATION COMPARISON - EFFICIENT PIPELINE")
    log_info(f"Target: {total_target} successful comparisons")
    log_info(f"VMAF: {vmaf_target} target, {vmaf_min} minimum")
    log_info(f"QP range: {qp_range[0]}-{qp_range[1]}")


def log_encoder_setup(encoder: str, encoder_type: str, vmaf_threads: int):
    """Log encoder setup information"""
    log_setup(f"Using encoder: {encoder} ({encoder_type})")
    log_setup(f"Using {vmaf_threads} VMAF threads")


def log_show_scan_start(show_name: str):
    """Log the start of scanning a show for files"""
    log_step(2, f"{show_name:<25} | Scanning...")


def log_show_files_found(show_name: str, file_count: int):
    """Log the number of files found in a show"""
    if file_count == 0:
        print(f"    No suitable files found")
    else:
        log_info(f"    Testing {file_count} files from {show_name}:")


def log_file_test_start(attempt: int, total: int, filename: str, size_mb: float):
    """Log the start of testing a file"""
    print(f"      [{attempt}/{total}] {filename[:50]}... ({size_mb:.0f}MB)")


def log_method_test_start(method: str):
    """Log the start of testing a specific method"""
    print(f"        → {method} testing...")


def log_method_success(method: str, qp: int, duration: float):
    """Log successful method completion"""
    print(f"        ✓ {method} OK: QP{qp}, {format_duration(duration)}")


def log_method_failure(method: str, error: str):
    """Log method failure"""
    print(f"        ✗ {method} FAILED: {error[:40]}")


def log_file_success(filename: str):
    """Log successful completion of both methods on a file"""
    print(f"      🎉 SUCCESS! Both methods completed on {filename[:40]}")


def log_show_skip(reason: str):
    """Log skipping a show"""
    log_warn(reason)


def log_comparison_result(comparison_num: int, show: str, binary_result: Dict[str, Any], 
                         gradient_result: Dict[str, Any], faster: str, elapsed: float):
    """Log a successful comparison result"""
    print(f"    ✅ Comparison {comparison_num}/20: "
          f"B:QP{binary_result['qp']} ({format_duration(binary_result['runtime'])}) "
          f"G:QP{gradient_result['qp']} ({format_duration(gradient_result['runtime'])}) "
          f"| {faster} faster | {format_duration(elapsed)} elapsed")


def log_final_results(total_runtime: float, shows_tested: int, successful: int, failed: int):
    """Log final test results summary"""
    print_section_header("FINAL RESULTS")
    log_info(f"Total Time: {format_duration(total_runtime)}")
    log_info(f"Shows Tested: {shows_tested}")
    log_info(f"Successful Comparisons: {successful}/20")
    log_info(f"Failed Shows: {failed}")


def print_results_table(binary_stats: Dict, gradient_stats: Dict):
    """Print a formatted results comparison table"""
    print()
    print(f"{'Method':<15} | {'Avg Runtime':<11} | {'Avg QP':<7} | {'Faster Count':<12}")
    print_separator(55)
    print(f"{'Binary Search':<15} | {format_duration(binary_stats['avg_runtime']):<11} | "
          f"{binary_stats['avg_qp']:6.1f} | {binary_stats['faster_count']:2d}/20")
    print(f"{'Gradient Desc':<15} | {format_duration(gradient_stats['avg_runtime']):<11} | "
          f"{gradient_stats['avg_qp']:6.1f} | {gradient_stats['faster_count']:2d}/20")


def log_performance_comparison(binary_avg: float, gradient_avg: float, 
                              gradient_faster_count: int, avg_qp_diff: float):
    """Log performance comparison between methods"""
    print()
    speedup = binary_avg / gradient_avg
    if speedup > 1:
        log_info(f"🚀 Gradient Descent is {speedup:.2f}x FASTER than Binary Search")
    else:
        log_info(f"🚀 Binary Search is {1/speedup:.2f}x FASTER than Gradient Descent")
    
    log_info(f"📊 Average QP difference: {avg_qp_diff:+.1f} (Gradient - Binary)")
    log_info(f"⚡ Gradient Descent was faster in {gradient_faster_count}/20 tests ({gradient_faster_count/20*100:.1f}%)")


def print_recent_results(comparisons: List[Dict[str, Any]], count: int = 10):
    """Print recent comparison results in a formatted table"""
    print()
    log_info("Recent Results:")
    print_separator()
    recent = comparisons[-count:] if len(comparisons) > count else comparisons
    
    for c in recent:
        faster = "G" if not c["binary_faster"] else "B"
        print(f"{c['comparison_number']:2d} | {c['show'][:20]:20} | "
              f"B:QP{c['binary_search']['qp']:2d}({format_duration(c['binary_search']['runtime'])}) "
              f"G:QP{c['gradient_descent']['qp']:2d}({format_duration(c['gradient_descent']['runtime'])}) | {faster}")


def handle_systematic_error(result: Dict[str, Any], method: str):
    """Handle systematic errors that should cause program exit"""
    error_type = result.get("error_type")
    if error_type in ["file_not_found", "permission_error", "disk_full", "clip_extraction_failed"]:
        log_error(result['error'])
        log_error(f"Error type: {error_type}")
        log_error("This is a systematic issue that prevents testing. Exiting.")
        if "stderr" in result and result["stderr"]:
            log_info(f"Debug info: {result['stderr'][:200]}")
        return True
    return False


def create_show_progress_bar(total_files: int, show_name: str):
    """Create a progress bar for files in a show"""
    return create_progress_bar(
        total=total_files, 
        desc=f"    {show_name[:25]}", 
        unit="files", 
        leave=False
    )


def create_overall_progress_bar():
    """Create the main progress bar for overall comparison progress"""
    return create_progress_bar(
        total=20, 
        desc="QP Comparison Progress", 
        unit="comparisons"
    )


# Real-time progress logging functions from core transcode
def log_qp_testing_start(candidate_qps: List[int]):
    """Log the start of QP testing with candidate values"""
    log_info(f"Testing QP values {candidate_qps} using VMAF scoring...")


def log_encoding_progress(qp: int, current: Optional[int] = None, total: Optional[int] = None):
    """Log encoding progress for a specific QP"""
    if current and total:
        print(f"  -> Encoding QP{qp} clip {current}/{total}...", end="", flush=True)
    else:
        print(f"    -> Encoding QP{qp}...", end="", flush=True)


def log_vmaf_result(vmaf_score: Optional[float] = None):
    """Log VMAF result after encoding"""
    if vmaf_score is not None:
        print(f" OK, VMAF... {vmaf_score:.1f}")
    else:
        print(f" OK, VMAF... FAILED")


def log_clip_result(idx: int, vmaf_score: Optional[float] = None):
    """Log result for a specific clip"""
    if vmaf_score is not None:
        print(f"  OK Clip {idx}: VMAF {vmaf_score:.1f}")
    else:
        print(f"  FAILED Clip {idx}: VMAF computation failed")


def create_qp_testing_progress_bar(total_qps: int, desc: str = "Testing QPs"):
    """Create progress bar for QP testing"""
    return create_progress_bar(total=total_qps, desc=desc, position=0)


def create_gradient_descent_progress_bar(max_iterations: int, filename: str):
    """Create progress bar for gradient descent iterations"""
    return create_progress_bar(
        total=max_iterations, 
        desc=f"GD Search {filename[:25]}", 
        unit="iterations"
    )


def update_gradient_descent_progress(pbar, filename: str, qp: int, iteration: int, 
                                    vmaf: Optional[float] = None, loss: Optional[float] = None):
    """Update gradient descent progress bar with current QP, VMAF, and loss"""
    if vmaf is not None and loss is not None:
        pbar.set_description(f"GD Search {filename[:25]} - QP{qp}: VMAF {vmaf:.1f}, Loss {loss:.2f}")
    else:
        pbar.set_description(f"GD Search {filename[:25]} - QP{qp} (iter {iteration+1})")


def update_clip_encoding_progress(pbar, qp: int, clip_idx: int, total_clips: int):
    """Update progress bar for clip encoding"""
    pbar.set_description(f"QP{qp} clips - Encoding clip {clip_idx+1}/{total_clips}")


def update_clip_vmaf_progress(pbar, qp: int, clip_idx: int, vmaf_score: Optional[float] = None):
    """Update progress bar with VMAF result for a clip"""
    if vmaf_score is not None:
        pbar.set_description(f"QP{qp} clips - Clip {clip_idx+1}: VMAF {vmaf_score:.1f}")
    else:
        pbar.set_description(f"QP{qp} clips - Clip {clip_idx+1}: VMAF failed")


def log_qp_results_table():
    """Print header for QP results table"""
    print(f"\n{'QP':<3} {'MeanVMAF':<9} {'MinVMAF':<8} {'Sample%':<8} {'EstFull%':<9} {'Decision':<9} Notes")
    print_separator(70)


def log_qp_result_row(qp: int, mean_vmaf: float, min_vmaf: float, 
                     sample_pct: float, full_pct: float, decision: str, notes: str = ""):
    """Log a single row in the QP results table"""
    print(f"{qp:<3} {mean_vmaf:<9.2f} {min_vmaf:<8.2f} {sample_pct:<8.1f} "
          f"{full_pct:<9.1f} {decision:<9} {notes}")


def log_selected_qp(qp: int, vmaf_target: float, vmaf_min: float):
    """Log the selected QP value"""
    print(f"\n[SELECTED] QP {qp} (target VMAF >= {vmaf_target}, min >= {vmaf_min})")


def log_gradient_descent_start(filename: str, vmaf_target: float, vmaf_min: float):
    """Log the start of gradient descent optimization"""
    log_gradient_descent(f"Starting 2D gradient descent QP optimization for {filename}")
    log_gradient_descent(f"Target VMAF: {vmaf_target:.1f}, Min VMAF: {vmaf_min:.1f}")


def log_gradient_descent_params(learning_rate: float, momentum: float, tolerance: float):
    """Log gradient descent parameters"""
    log_gradient_descent(f"Learning rate: {learning_rate:.3f}, Momentum: {momentum:.2f}, Tolerance: {tolerance:.1f}")


def log_clips_extracted(count: int):
    """Log number of clips extracted for gradient descent"""
    log_info(f"Extracted {count} clips for gradient descent optimization")


def log_gradient_descent_converged(vmaf: float, tolerance: float):
    """Log gradient descent convergence"""
    log_gradient_descent(f"Converged: VMAF {vmaf:.2f} within tolerance {tolerance}")


def log_gradient_descent_result(filename: str, qp: int, mean_vmaf: float, loss: float, 
                               min_vmaf: float, worst_drop: float, size_pct: float, samples: int):
    """Log final gradient descent result"""
    log_gradient_descent(f"{filename}: QP {qp} mean {mean_vmaf:.2f} (loss {loss:.2f}), "
                        f"min {min_vmaf:.2f}, worst drop {worst_drop:.2f}, "
                        f"size {size_pct:.1f}% (samples {samples})")


def log_binary_search_start(filename: str, vmaf_target: float, vmaf_min: float):
    """Log the start of binary search optimization"""
    log_info(f"Starting binary search QP optimization for {filename}")
    log_info(f"Target VMAF: {vmaf_target:.1f}, Min VMAF: {vmaf_min:.1f}")


def log_binary_search_step(qp: int, vmaf: float, size_pct: float):
    """Log a binary search step"""
    print(f"    QP{qp}: VMAF {vmaf:.1f}, Size {size_pct:.0f}%")


def create_adaptive_qp_progress_bar(filename: str):
    """Create progress bar for adaptive QP search"""
    return create_progress_bar(
        desc=f"QP Search {filename[:35]}", 
        unit="tests", 
        total=None
    )


def update_adaptive_qp_progress(pbar, filename: str, qp: int, vmaf: Optional[float] = None, size_pct: Optional[float] = None):
    """Update adaptive QP search progress"""
    if vmaf is not None and size_pct is not None:
        pbar.set_description(f"QP Search {filename[:30]} - QP{qp}: VMAF {vmaf:.1f}, Size {size_pct:.0f}%")
    else:
        pbar.set_description(f"QP Search {filename[:35]} - Testing QP{qp}")


def log_transcoding_progress(qp: int, filename: str, vmaf_score: Optional[float] = None):
    """Log transcoding progress with VMAF check"""
    if vmaf_score is not None:
        print(f"Transcoding QP{qp} - VMAF {vmaf_score:.1f} {filename[:15]}")
    else:
        print(f"Transcoding QP{qp} - VMAF check {filename[:20]}...")


def log_adapt_no_vmaf_scores(qp: int):
    """Log when no VMAF scores available at a QP"""
    print(f"[ADAPT] No VMAF scores at QP {qp}; stopping.")


# Subprocess output parsing helpers
def parse_qp_from_output(output: str) -> Optional[int]:
    """Parse QP result from subprocess output"""
    import re
    for line in output.split('\n'):
        if "[RESULT]" in line and "QP" in line:
            qp_match = re.search(r'QP (\d+)', line)
            if qp_match:
                return int(qp_match.group(1))
    return None
