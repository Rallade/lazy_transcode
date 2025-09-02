"""
Main transcoding orchestration module for lazy_transcode.

This module coordinates the transcoding workflow using modular components:
- File discovery and managemen            # Run VBR optimization
            optimal_bitrate, result = optimize_encoder_settings_vbr(
                file, encoder, encoder_type,
                target_vmaf=args.vmaf_target,
                vmaf_tolerance=args.vmaf_tol,
                num_clips=vbr_clips,
                clip_duration=args.vbr_clip_duration,
                max_trials=args.vbr_max_trials
            )
            
            if optimal_bitrate > 0 and result.get('vmaf', 0) > 0:
                # Convert to expected format for compatibility
                vbr_result = {
                    'success': True,
                    'bitrate': optimal_bitrate,
                    'vmaf_score': result.get('vmaf', 0.0)
                }
                vbr_results[file] = vbr_result
                print(f"[VBR-SUCCESS] {file.name}: {optimal_bitrate}kbps, "
                      f"VMAF {result.get('vmaf', 0.0):.2f}")
            else:
                if result.get('error'):
                    print(f"[VBR-FAILED] {file.name}: {result.get('error', 'Unknown error')}")
                else:
                    print(f"[VBR-FAILED] {file.name}: Could not find suitable VBR settings")mization 
- Parallel job processing
- User interaction and verification
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Import modular components
from .modules.encoder_config import EncoderConfigBuilder
from .modules.vmaf_evaluator import VMAfEvaluator
from .modules.file_manager import FileManager
from .modules.system_utils import (
    TEMP_FILES, format_size, get_next_transcoded_dir, 
    start_cpu_monitor, DEBUG
)
from .modules.media_utils import (
    get_duration_sec, get_video_codec, should_skip_codec, 
    compute_vmaf_score
)
from .modules.vbr_optimizer import (
    calculate_intelligent_vbr_bounds, optimize_encoder_settings_vbr,
    get_vbr_clip_positions
)
from .modules.qp_optimizer import (
    find_optimal_qp, adaptive_qp_search_per_file, extract_random_clips,
    test_qp_on_sample
)
from .modules.transcoding_engine import (
    build_encode_cmd, build_vbr_encode_cmd, transcode_file_qp, 
    transcode_file_vbr, detect_best_encoder as detect_hevc_encoder
)
from .modules.job_processor import (
    TranscodeJob, AsyncFileStager, ParallelTranscoder
)
from .modules.user_interface import (
    prompt_user_confirmation, verify_and_prompt_transcode,
    display_encoder_info, display_vbr_results_summary,
    display_qp_optimization_summary
)

# Configuration defaults
EXTS = [".mkv", ".mp4", ".mov", ".ts"]
SAMPLE_COUNT_DEFAULT = 6
CANDIDATE_QPS = [16, 18, 20, 22, 24]
VMAF_TARGET = 95.0
VMAF_MIN_THRESHOLD = 93.0


def encode_with_progress(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                        qp: int, preserve_hdr_metadata: bool = True) -> bool:
    """Encode a file with progress tracking."""
    import time
    import subprocess
    
    # Set up progress tracking
    progress_name = f"progress_{infile.stem}_{int(time.time())}.txt"
    progress_file = infile.parent / progress_name
    TEMP_FILES.add(str(progress_file))
    
    try:
        # Build command with progress tracking
        cmd = build_encode_cmd(infile, outfile, encoder, encoder_type, 
                              qp, preserve_hdr_metadata, progress_file)
        
        if DEBUG:
            print(f"[TRANSCODE] {' '.join(cmd)}")
        
        # Start encoding process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, text=True)
        
        # Monitor progress
        last_progress = {}
        with tqdm(total=100, desc=f"Encoding {infile.name[:30]}", 
                 position=1, leave=False, unit="%") as pbar:
            
            while process.poll() is None:
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            
                        progress_data = {}
                        for line in content.split('\n'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                progress_data[key] = value
                        
                        # Update progress bar
                        if 'progress' in progress_data:
                            if progress_data['progress'] == 'end':
                                pbar.n = 100
                                pbar.refresh()
                                break
                            elif 'out_time_ms' in progress_data and infile.exists():
                                try:
                                    duration_ms = get_duration_sec(infile) * 1000000
                                    current_ms = int(progress_data['out_time_ms'])
                                    progress_pct = min(100, (current_ms / duration_ms) * 100)
                                    pbar.n = progress_pct
                                    pbar.refresh()
                                except:
                                    pass
                                    
                        last_progress = progress_data
                        
                    except (FileNotFoundError, PermissionError):
                        pass
                        
                time.sleep(0.5)  # Check every 500ms
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0 and outfile.exists():
            return True
        else:
            if DEBUG and stderr:
                print(f"[TRANSCODE-ERROR] {stderr}")
            return False
            
    finally:
        # Clean up progress file
        if progress_file.exists():
            try:
                progress_file.unlink()
                TEMP_FILES.discard(str(progress_file))
            except:
                pass


def process_vbr_mode(files: List[Path], encoder: str, encoder_type: str, args, 
                    output_base: Path, preserve_hdr: bool) -> int:
    """Process files in VBR optimization mode."""
    print(f"[VBR] Starting VBR optimization mode (target VMAF: {args.vmaf_target}±{args.vmaf_tol})")
    
    vbr_results = {}
    
    # Standard processing mode (no staging for now - can be added later)
    with tqdm(total=len(files), desc="VBR Optimization", position=0) as vbr_progress:
        for file in files:
            vbr_progress.set_description(f"VBR Optimization - {file.name[:40]}...")
            
            # Get file duration and clip positions for VBR testing
            duration = get_duration_sec(file)
            if duration is None:
                print(f"[VBR-SKIP] {file.name} - Cannot determine duration")
                vbr_progress.update(1)
                continue
            
            # Auto-scale VBR clips based on file duration
            vbr_clips = args.vbr_clips
            vbr_clips_was_explicitly_set = '--vbr-clips' in sys.argv
            
            if duration > 0 and not vbr_clips_was_explicitly_set:
                auto_clips = max(3, min(vbr_clips * 2, int(duration / 1200)))  # 1200s = 20min
                if auto_clips != vbr_clips:
                    print(f"[VBR-INFO] Auto-scaling clips from {vbr_clips} to {auto_clips} based on {duration/60:.1f}min duration")
                    vbr_clips = auto_clips
                    
            clip_positions = get_vbr_clip_positions(int(duration), vbr_clips)
            
            # Run VBR optimization
            optimal_bitrate, result = optimize_encoder_settings_vbr(
                file, encoder, encoder_type,
                target_vmaf=args.vmaf_target,
                vmaf_tolerance=args.vmaf_tol,
                num_clips=vbr_clips,
                clip_duration=args.vbr_clip_duration,
                max_trials=args.vbr_max_trials
            )
            
            if optimal_bitrate > 0 and result.get('vmaf', 0) > 0:
                # Convert to expected format for compatibility
                vbr_result = {
                    'success': True,
                    'bitrate': optimal_bitrate,
                    'vmaf_score': result.get('vmaf', 0.0)
                }
                vbr_results[file] = vbr_result
                print(f"[VBR-SUCCESS] {file.name}: {optimal_bitrate}kbps, "
                      f"VMAF {result.get('vmaf', 0.0):.2f}")
            else:
                if result.get('error'):
                    print(f"[VBR-FAILED] {file.name}: {result.get('error', 'Unknown error')}")
                else:
                    print(f"[VBR-FAILED] {file.name}: Could not find suitable VBR settings")
                    
            vbr_progress.update(1)
    
    # Display results summary
    display_vbr_results_summary(vbr_results, len(files))
    
    # VBR transcoding phase
    if not (args.dry_run or args.skip_transcode) and vbr_results:
        print(f"[VBR] Starting VBR transcoding with optimized settings...")
        
        # Create Transcoded directory if non-destructive mode
        transcoded_dir = None
        if args.non_destructive:
            transcoded_dir = get_next_transcoded_dir(output_base)
            transcoded_dir.mkdir(exist_ok=True)
            print(f"[INFO] Non-destructive mode: saving to {transcoded_dir}")
        
        success_count = 0
        with tqdm(total=len(vbr_results), desc="VBR Transcoding", position=0) as overall:
            for file, vbr_result in vbr_results.items():
                print(f"[VBR-ENCODE] Transcoding {file.name} at {vbr_result['bitrate']}kbps")
                
                if args.non_destructive:
                    # Non-destructive: save to Transcoded subdirectory
                    final_output = transcoded_dir / file.name
                    tmp = transcoded_dir / (file.stem + ".transcode" + file.suffix)
                else:
                    # Destructive: replace original
                    final_output = file
                    tmp = file.with_name(file.stem + ".transcode" + file.suffix)
                    bak = file.with_name(file.stem + ".bak" + file.suffix)

                if tmp.exists(): 
                    try: tmp.unlink()
                    except: pass
                
                # Use VBR-specific transcoding
                ok = transcode_file_vbr(
                    file, tmp, encoder, encoder_type,
                    vbr_result['bitrate'], int(vbr_result['bitrate'] * 0.8),  # avg = 80% of max
                    preserve_hdr_metadata=preserve_hdr
                )
                
                if ok and tmp.exists():
                    # Final VMAF validation
                    print(f"  [VERIFY] Running final VMAF check...")
                    final_vmaf = compute_vmaf_score(file, tmp, n_threads=args.vmaf_threads)
                    
                    if final_vmaf is not None:
                        vmaf_diff = abs(final_vmaf - args.vmaf_target)
                        within_tolerance = vmaf_diff <= args.vmaf_tol
                        
                        if within_tolerance:
                            print(f"  ✓ Final VMAF validation passed: {final_vmaf:.2f}")
                            
                            # Quality validated - proceed with file placement
                            if args.non_destructive:
                                # Non-destructive: simply move temp to final location
                                if final_output.exists():
                                    final_output.unlink()
                                shutil.move(str(tmp), str(final_output))
                                print(f"  ✓ Saved verified transcode to Transcoded/{file.name}")
                                success_count += 1
                            else:
                                # Destructive: atomic swap with backup
                                if bak.exists():
                                    try: bak.unlink()
                                    except: pass
                                shutil.move(str(file), str(bak))
                                shutil.move(str(tmp), str(file))
                                try: bak.unlink()
                                except: pass
                                print(f"  ✓ Verified and replaced {file.name}")
                                success_count += 1
                        else:
                            print(f"  ✗ Final VMAF validation FAILED: {final_vmaf:.2f}")
                            try:
                                tmp.unlink()
                            except:
                                pass
                    else:
                        print(f"  ✗ Final VMAF calculation failed")
                        try:
                            tmp.unlink()
                        except:
                            pass
                else:
                    print(f"  ✗ Failed to transcode {file.name}")
                    
                overall.update(1)
        
        print(f"[VBR] Transcoding complete: {success_count}/{len(vbr_results)} files successful.")
        return success_count
    else:
        print(f"[VBR] Dry run mode - skipping actual transcoding")
        return 0


def process_qp_mode(files: List[Path], encoder: str, encoder_type: str, args,
                   output_base: Path, preserve_hdr: bool) -> int:
    """Process files in QP optimization mode."""
    
    # QP optimization step
    if args.use_qp:
        # Use predetermined QP for all files
        optimal_qp = args.use_qp
        print(f"[INFO] Using predetermined QP {optimal_qp} (skipping optimization)")
        file_qp_map = {f: optimal_qp for f in files}
    else:
        # Per-file QP optimization using random clips
        print(f"[INFO] Starting per-file QP optimization using {args.samples} random clips per file")
        file_qp_map = {}
        optimal_qps_history = []  # Track optimal QPs for adaptive starting points
        
        with tqdm(total=len(files), desc="File QP Optimization", position=0) as file_progress:
            for file in files:
                # Update progress bar with current file
                file_progress.set_description(f"File QP Optimization - Analyzing {file.name[:45]}...")
                
                # Calculate adaptive initial QP based on previous files
                if optimal_qps_history:
                    initial_qp = round(sum(optimal_qps_history) / len(optimal_qps_history))
                    initial_qp = max(10, min(45, initial_qp))  # Clamp to valid range
                    if DEBUG:
                        print(f"[ADAPTIVE-START] Using QP {initial_qp} based on {len(optimal_qps_history)} previous files")
                else:
                    initial_qp = 24  # Default for first file
                    if DEBUG:
                        print(f"[ADAPTIVE-START] Using default QP {initial_qp} for first file")
                
                if args.qp_range:
                    # Use grid search approach
                    provided = [int(q.strip()) for q in args.qp_range.split(',') if q.strip()]
                    provided = sorted(set(provided))
                    median_idx = len(provided)//2
                    file_qp = provided[median_idx] if provided else 24
                    print(f"[INFO] Using median QP {file_qp} for {file.name}")
                else:
                    # Use adaptive search per file
                    file_qp, qp_result = adaptive_qp_search_per_file(
                        file,
                        encoder,
                        encoder_type,
                        samples=args.samples,
                        initial_qp=initial_qp,
                        initial_step=args.initial_step,
                        min_step=args.min_step,
                        vmaf_target=args.vmaf_target,
                        vmaf_min_threshold=args.vmaf_min,
                        sample_duration=args.sample_duration,
                        min_qp_limit=args.min_qp,
                        max_qp_limit=args.max_qp,
                        vmaf_threads=args.vmaf_threads,
                        preserve_hdr_metadata=preserve_hdr,
                        use_gradient_descent=args.gradient_descent
                    )
                
                file_qp_map[file] = file_qp
                
                # Track QPs for adaptive learning
                is_at_min_bound = (file_qp <= args.min_qp)
                is_at_max_bound = (file_qp >= args.max_qp)
                is_bound_constrained = is_at_min_bound or is_at_max_bound
                
                # Check if VMAF target was met despite being at bounds
                if qp_result:
                    file_vmaf = qp_result['mean_vmaf']
                    meets_vmaf_target = file_vmaf >= args.vmaf_target
                else:
                    meets_vmaf_target = False
                
                if not is_bound_constrained or meets_vmaf_target:
                    optimal_qps_history.append(file_qp)
                    if DEBUG:
                        status = "meets target despite bounds" if (is_bound_constrained and meets_vmaf_target) else "not bound-constrained"
                        print(f"[ADAPTIVE-LEARN] Added QP {file_qp} to history ({status})")
                elif DEBUG:
                    vmaf_info = f", VMAF {file_vmaf:.1f}" if qp_result else ""
                    print(f"[ADAPTIVE-LEARN] Skipped QP {file_qp} (bound-constrained{vmaf_info})")
                
                # Update progress bar with result
                file_progress.set_description(f"File QP Optimization - {file.name[:35]}: QP{file_qp}")
                file_progress.update(1)
        
        # Display optimization summary
        optimal_qp = display_qp_optimization_summary(file_qp_map, files)

    if args.dry_run or args.skip_transcode:
        print("[DRY-RUN] Skipping full transcode.")
        return 0

    # Verify quality + size thresholds and get user confirmation
    if not verify_and_prompt_transcode(files, optimal_qp, encoder, encoder_type, args, preserve_hdr):
        return 0

    # Full transcode phase
    print(f"[INFO] Starting full transcode with per-file optimized QPs...")
    
    if not args.no_parallel:
        print(f"[INFO] Using parallel processing (disable with --no-parallel)")
        transcoder = ParallelTranscoder(vmaf_threads=args.vmaf_threads, 
                                      use_experimental_vmaf=args.experimental_vmaf)
        success_count = transcoder.process_files_with_qp_map(
            files, file_qp_map, encoder, encoder_type, 
            args.vmaf_target, args.vmaf_min, preserve_hdr
        )
        print(f"[DONE] Parallel processing completed: {success_count}/{len(files)} files successful.")
        return success_count
    else:
        # Sequential processing with per-file QPs
        print("[INFO] Using sequential processing with per-file QPs")
        success_count = 0
        
        # Create Transcoded directory if non-destructive mode
        transcoded_dir = None
        if args.non_destructive:
            transcoded_dir = get_next_transcoded_dir(output_base)
            transcoded_dir.mkdir(exist_ok=True)
            print(f"[INFO] Non-destructive mode: saving to {transcoded_dir}")
        
        with tqdm(total=len(files), desc="Overall Transcode", position=0) as overall:
            for f in files:
                file_qp = file_qp_map.get(f, optimal_qp)
                print(f"[INFO] Transcoding {f.name} with QP {file_qp}")
                
                if args.non_destructive and transcoded_dir:
                    # Non-destructive: save to Transcoded subdirectory
                    final_output = transcoded_dir / f.name
                    tmp = transcoded_dir / (f.stem + ".transcode" + f.suffix)
                else:
                    # Destructive: replace original
                    final_output = f
                    tmp = f.with_name(f.stem + ".transcode" + f.suffix)
                    bak = f.with_name(f.stem + ".bak" + f.suffix)

                if tmp.exists(): 
                    try: tmp.unlink()
                    except: pass
                    
                ok = encode_with_progress(f, tmp, encoder, encoder_type, file_qp, 
                                        preserve_hdr_metadata=preserve_hdr)
                if ok and tmp.exists():
                    # Final VMAF validation
                    print(f"  [VERIFY] Running final VMAF check on full transcoded file...")
                    final_vmaf = compute_vmaf_score(f, tmp, n_threads=args.vmaf_threads)
                    
                    if final_vmaf is not None:
                        within_target = final_vmaf >= args.vmaf_target
                        within_min = final_vmaf >= args.vmaf_min
                        
                        if within_target and within_min:
                            print(f"  ✓ Final VMAF validation passed: {final_vmaf:.2f}")
                            
                            # Quality validated - proceed with file placement
                            if args.non_destructive:
                                # Non-destructive: simply move temp to final location
                                if final_output.exists():
                                    final_output.unlink()
                                shutil.move(str(tmp), str(final_output))
                                print(f"  ✓ Saved verified transcode to Transcoded/{f.name}")
                                success_count += 1
                            else:
                                # Destructive: atomic swap with backup
                                if bak.exists():
                                    try: bak.unlink()
                                    except: pass
                                shutil.move(str(f), str(bak))
                                shutil.move(str(tmp), str(f))
                                try: bak.unlink()
                                except: pass
                                print(f"  ✓ Verified and replaced {f.name}")
                                success_count += 1
                        else:
                            print(f"  ✗ Final VMAF validation FAILED: {final_vmaf:.2f}")
                            try:
                                tmp.unlink()
                            except:
                                pass
                    else:
                        print(f"  ✗ Final VMAF calculation failed")
                        try:
                            tmp.unlink()
                        except:
                            pass
                else:
                    if tmp.exists():
                        try: tmp.unlink()
                        except: pass
                    TEMP_FILES.discard(str(tmp))
                    print(f"  Failed to transcode {f.name}")
                overall.update(1)
        
        print(f"[DONE] Sequential processing completed: {success_count}/{len(files)} files successful.")
        return success_count


def main():
    """Main transcoding workflow orchestration."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=".", help="Folder to scan (supports UNC)")
    ap.add_argument("--mode", choices=["qp", "vbr"], default="qp", 
                    help="Optimization mode: 'qp' for QP search (default), 'vbr' for VBR bitrate search")
    ap.add_argument("--exts", default=",".join(e.strip(".") for e in EXTS),
                    help="Comma-separated extensions (e.g. mkv,mp4,mov,ts)")
    ap.add_argument("--samples", type=int, default=SAMPLE_COUNT_DEFAULT,
                    help="How many files to sample for QP optimization")
    ap.add_argument("--no-auto-scale-samples", action="store_true", 
                    help="Disable automatic sample count scaling based on video duration")
    ap.add_argument("--qp-range", default=None,
                    help="Comma-separated QP values to test (optional)")
    ap.add_argument("--use-qp", type=int, default=None,
                    help="Skip QP optimization and use this specific QP value")
    ap.add_argument("--vmaf-target", type=float, default=VMAF_TARGET,
                    help="Target mean VMAF score")
    ap.add_argument("--vmaf-min", type=float, default=VMAF_MIN_THRESHOLD,
                    help="Minimum acceptable VMAF score")
    # VBR mode parameters
    ap.add_argument("--vmaf-tol", type=float, default=1,
                    help="VMAF tolerance for VBR mode (default 1.0)")
    ap.add_argument("--vbr-clips", type=int, default=2,
                    help="Number of clips for VBR VMAF testing (default 2)")
    ap.add_argument("--vbr-clip-duration", type=int, default=30,
                    help="Duration in seconds for each VBR test clip (default 30)")
    ap.add_argument("--vbr-max-trials", type=int, default=6,
                    help="Maximum trials for VBR parameter optimization (default 6)")
    ap.add_argument("--sample-duration", type=int, default=600,
                    help="Seconds per sample clip for QP testing (default 600)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only perform optimization analysis; skip actual transcoding")
    ap.add_argument("--skip-transcode", action="store_true",
                    help="Alias for --dry-run")
    ap.add_argument("--force-overwrite", action="store_true",
                    help="Proceed even if quality or size thresholds are not met")
    ap.add_argument("--max-size-pct", type=float, default=100.0,
                    help="Require estimated encoded size to be <= this percent of original")
    ap.add_argument("--quality-prompt-band", type=float, default=1.0,
                    help="VMAF margin for interactive prompting instead of auto-abort")
    ap.add_argument("--min-qp", type=int, default=10, help="Lower bound QP for adaptive search")
    ap.add_argument("--max-qp", type=int, default=45, help="Upper bound QP for adaptive search")
    ap.add_argument("--initial-step", type=float, default=4.0, help="Initial QP step size")
    ap.add_argument("--min-step", type=float, default=1.0, help="Minimum QP step size")
    ap.add_argument("--scale-steps-by-vmaf", action="store_true", 
                    help="Scale adaptive QP step size proportionally to VMAF")
    ap.add_argument("--step-scale", type=float, default=0.5, 
                    help="QP step per 1 VMAF point surplus/deficit")
    ap.add_argument("--max-dynamic-step", type=float, default=8.0, 
                    help="Upper bound on dynamically scaled QP step")
    
    # Threading configuration
    detected_cores = os.cpu_count() or 0
    optimal_vmaf_threads = min(12, detected_cores)
    ap.add_argument("--vmaf-threads", type=int, default=optimal_vmaf_threads,
                    help=f"Threads for libvmaf (default = {optimal_vmaf_threads})")
    ap.add_argument("--experimental-vmaf", action="store_true", 
                    help="Use experimental multiprocess VMAF")
    ap.add_argument("--quick-sample-duration", type=int, default=0,
                    help="Shorter sample duration for first N iterations")
    ap.add_argument("--quick-iters", type=int, default=0,
                    help="Number of initial iterations to use quick sample duration")
    
    # General options
    ap.add_argument("--no-hdr-metadata", action="store_true", 
                    help="Disable HDR10 static metadata preservation")
    ap.add_argument("--gradient-descent", action="store_true", 
                    help="Use 2D gradient descent optimization for QP search")
    ap.add_argument("--debug", action="store_true", 
                    help="Verbose debugging: keep samples, show commands")
    ap.add_argument("--cpu-monitor", action="store_true", 
                    help="Monitor CPU usage during VMAF operations")
    ap.add_argument("--cpu", action="store_true", 
                    help="Force CPU/software encoding (libx265)")
    ap.add_argument("--auto-yes", action="store_true", 
                    help="Automatically answer 'yes' to confirmation prompts")
    ap.add_argument("--no-parallel", action="store_true", 
                    help="Disable parallel processing")
    ap.add_argument("--non-destructive", action="store_true", 
                    help="Save transcoded files to 'Transcoded' subdirectory")
    ap.add_argument("--staging", action="store_true", 
                    help="Enable async file staging for network drives")
    ap.add_argument("--limit", type=int, 
                    help="Limit processing to the first N files")
    
    args = ap.parse_args()

    # Set global debug flag
    global DEBUG
    DEBUG = args.debug
    if DEBUG:
        print("[DEBUG] Debug mode active")

    # Display threading defaults
    if "--vmaf-threads" not in sys.argv and args.vmaf_threads == optimal_vmaf_threads:
        print(f"[INFO] libvmaf threads defaulting to optimal count: {optimal_vmaf_threads}")
    if "--sample-duration" not in sys.argv and args.sample_duration == 600:
        print("[INFO] sample-duration defaulting to 600s (override with --sample-duration)")

    preserve_hdr = not args.no_hdr_metadata

    # Validate path
    base = Path(args.path)
    if not base.exists():
        print(f"[ERR] path not found: {base}")
        sys.exit(1)

    # Detect encoder
    if args.cpu:
        encoder, encoder_type = "libx265", "software"
        display_encoder_info(encoder, encoder_type, cpu_forced=True)
    else:
        encoder, encoder_type = detect_hevc_encoder()
        display_encoder_info(encoder, encoder_type)

    # Initialize file manager for discovery
    file_manager = FileManager(debug=DEBUG, temp_files=TEMP_FILES)
    
    # Perform startup cleanup
    removed_count = file_manager.startup_scavenge(base)
    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} stale temp file(s) from previous run.")
    
    # Discover and filter files
    try:
        if base.is_file():
            # Direct file path provided
            if base.suffix.lower() in [ext.lower() for ext in EXTS]:
                print(f"[INFO] Processing single file: {base.name}")
                files = [base]
                output_base = base.parent
            else:
                print(f"[ERR] Unsupported file type: {base.suffix}")
                sys.exit(1)
        else:
            # Directory path - use FileManager for discovery
            discovery_result = file_manager.process_files_with_codec_filtering(base, args.exts)
            files = discovery_result.files_to_transcode
            output_base = base
            
            if not files:
                if discovery_result.skipped_files:
                    print("[INFO] No files need transcoding (all already use efficient codecs)")
                else:
                    print(f"[INFO] No matching files in {base}")
                return
        
        # Apply limit if specified
        if args.limit and len(files) > args.limit:
            original_count = len(files)
            files = files[:args.limit]
            print(f"[INFO] Limited to first {args.limit} files (out of {original_count} total)")
            
    except ValueError as e:
        print(f"[ERR] {e}")
        sys.exit(1)

    # Route to appropriate processing mode
    if args.mode == "vbr":
        success_count = process_vbr_mode(files, encoder, encoder_type, args, output_base, preserve_hdr)
    else:
        success_count = process_qp_mode(files, encoder, encoder_type, args, output_base, preserve_hdr)

    print("[DONE] All files processed.")


if __name__ == "__main__":
    main()
