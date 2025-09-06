"""
Main transcoding orchestration module for lazy_transcode.

This module coordinates the transcoding workflow using modular components:
- File discovery and management
- VBR and QP optimization
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
from .modules.config.encoder_config import EncoderConfigBuilder
from .modules.analysis.vmaf_evaluator import VMAfEvaluator
from .modules.processing.file_manager import FileManager
from .modules.system.system_utils import (
    TEMP_FILES, format_size, get_next_transcoded_dir, 
    start_cpu_monitor, DEBUG
)
from .modules.analysis.media_utils import (
    get_duration_sec, get_video_codec, should_skip_codec, 
    compute_vmaf_score
)
from ..utils.logging import get_logger, set_debug_mode

# Module logger
logger = get_logger("transcode_main")
from .modules.optimization.vbr_optimizer import (
    calculate_intelligent_vbr_bounds, optimize_encoder_settings_vbr,
    build_vbr_encode_cmd
)
from .modules.optimization.coverage_clips import (
    get_coverage_based_vbr_clip_positions
)
from .modules.optimization.qp_optimizer import (
    find_optimal_qp, adaptive_qp_search_per_file, extract_random_clips,
    test_qp_on_sample
)
from .modules.processing.transcoding_engine import (
    build_encode_cmd, transcode_file_qp, 
    transcode_file_vbr, detect_best_encoder
)
from .modules.processing.job_processor import (
    TranscodeJob, AsyncFileStager, ParallelTranscoder
)
from .modules.interface.user_interface import (
    prompt_user_confirmation, verify_and_prompt_transcode,
    display_vbr_results_summary
)


def process_vbr_mode(args, encoder: str, encoder_type: str, files: List[Path]) -> Dict[Path, Dict]:
    """Process files using VBR optimization mode."""
    vbr_results = {}
    
    logger.vbr(f"Processing {len(files)} files")
    logger.vbr(f"Target VMAF: {args.vmaf_target:.1f} ±{args.vmaf_tol:.1f}")
    logger.vbr(f"Encoder: {encoder} ({encoder_type})")
    
    for file in files:
        logger.vbr(f"\nProcessing: {file.name}")
        
        # Skip if wrong codec
        current_codec = get_video_codec(file)
        if should_skip_codec(current_codec):
            logger.debug(f"SKIP {file.name}: Already HEVC/AV1")
            continue
        
        # Get duration and calculate clips using coverage-based algorithm
        duration = get_duration_sec(file)
        if duration <= 0:
            logger.debug(f"SKIP {file.name}: Could not determine duration")
            continue
            
        # Use new coverage-based clip selection (automatically scales clips for target coverage)
        logger.debug(f"Using coverage-based clip selection for {duration/60:.1f}min video (coverage: {args.vbr_coverage:.1%})")
        clip_positions = get_coverage_based_vbr_clip_positions(
            file, duration, clip_duration=args.vbr_clip_duration, target_coverage=args.vbr_coverage
        )
        
        # Calculate clip target VMAF for minimum quality guarantee approach
        # Target higher VMAF on clips to ensure no quality dips in complex scenes
        clip_target_vmaf = args.vmaf_target - 2.0  # Ensure clips meet high standard for consistent quality
        clip_tolerance = args.vmaf_tol  # Use tolerance as "good enough" threshold, not optimization range
        
        logger.vbr(f"Using minimum quality guarantee approach:")
        logger.vbr(f"  Clip target: {clip_target_vmaf:.1f} (stop if ≥{clip_target_vmaf - clip_tolerance:.1f})")
        logger.vbr(f"  Expected full episode: {args.vmaf_target:.1f}+")
        
        # Run VBR optimization
        if args.vbr_method == "compare":
            # Compare all optimization methods
            from .modules.optimization.vbr_optimizer import optimize_vbr_with_gradient_methods
            
            logger.vbr("Running optimization method comparison")
            comparison_results = optimize_vbr_with_gradient_methods(
                file, file.with_suffix('.hevc.mkv'), clip_target_vmaf,  # Use clip target for comparison
                encoder, encoder_type, args.preserve_hdr,
                methods=["gradient-descent", "quasi-newton", "conjugate-gradient", "bisection"]
            )
            
            # Find the best method result
            successful_methods = {k: v for k, v in comparison_results.items() 
                                if hasattr(v, 'success') and v.success}
            
            if successful_methods:
                # Choose best method by accuracy (closest to target VMAF)
                best_method = min(successful_methods.items(), 
                                key=lambda x: abs(x[1].vmaf_score - clip_target_vmaf) 
                                if x[1].vmaf_score else 999)
                
                method_name, best_result = best_method
                
                # Convert to expected result format
                result = {
                    'success': best_result.success,
                    'bitrate': best_result.bitrate,
                    'vmaf_score': best_result.vmaf_score,
                    'preset': best_result.preset or 'medium',
                    'bf': best_result.bf or 3,
                    'refs': best_result.refs or 3,
                    'filesize': best_result.filesize,
                    'method_used': best_result.method_used,
                    'convergence_time': best_result.convergence_time,
                    'iterations': best_result.iterations
                }
                
                logger.vbr(f"Best method: {method_name.upper()} - "
                          f"{result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f}")
                
                # Log comparison summary
                logger.vbr("Method comparison results:")
                for method, res in successful_methods.items():
                    logger.vbr(f"  {method}: {res.bitrate}kbps, VMAF {res.vmaf_score:.2f}, "
                              f"{res.iterations} iter, {res.convergence_time:.1f}s")
            else:
                logger.vbr("All optimization methods failed, falling back to standard VBR")
                result = optimize_encoder_settings_vbr(
                    file, encoder, encoder_type,
                    target_vmaf=clip_target_vmaf,  # Use clip target for consistency
                    vmaf_tolerance=clip_tolerance,
                    clip_positions=clip_positions,
                    clip_duration=args.vbr_clip_duration,
                    max_safety_limit=args.vbr_max_trials,
                    auto_tune=args.animation_tune
                )
                
        elif args.vbr_method in ["gradient-descent", "quasi-newton", "conjugate-gradient"]:
            # Use specific gradient method
            from .modules.optimization.vbr_optimizer import optimize_vbr_with_gradient_methods
            
            logger.vbr(f"Using {args.vbr_method} optimization method")
            comparison_results = optimize_vbr_with_gradient_methods(
                file, file.with_suffix('.hevc.mkv'), clip_target_vmaf,  # Use clip target
                encoder, encoder_type, args.preserve_hdr,
                methods=[args.vbr_method]
            )
            
            if args.vbr_method in comparison_results and comparison_results[args.vbr_method].success:
                grad_result = comparison_results[args.vbr_method]
                
                # Convert to expected result format
                result = {
                    'success': grad_result.success,
                    'bitrate': grad_result.bitrate,
                    'vmaf_score': grad_result.vmaf_score,
                    'preset': grad_result.preset or 'medium',
                    'bf': grad_result.bf or 3,
                    'refs': grad_result.refs or 3,
                    'filesize': grad_result.filesize,
                    'method_used': grad_result.method_used,
                    'convergence_time': grad_result.convergence_time,
                    'iterations': grad_result.iterations
                }
                
                logger.vbr(f"{args.vbr_method.upper()} result: {result['bitrate']}kbps, "
                          f"VMAF {result['vmaf_score']:.2f}")
            else:
                logger.vbr(f"{args.vbr_method} method failed, falling back to bisection")
                result = optimize_encoder_settings_vbr(
                    file, encoder, encoder_type,
                    target_vmaf=clip_target_vmaf,  # Use clip target for consistency
                    vmaf_tolerance=clip_tolerance,
                    clip_positions=clip_positions,
                    clip_duration=args.vbr_clip_duration,
                    max_safety_limit=args.vbr_max_trials,
                    auto_tune=args.animation_tune
                )
        else:
            # Use standard bisection method with minimum quality guarantee
            result = optimize_encoder_settings_vbr(
                file, encoder, encoder_type,
                target_vmaf=clip_target_vmaf,  # Target higher VMAF on clips for quality guarantee
                vmaf_tolerance=clip_tolerance,  # Use as "good enough" threshold, not optimization range
                clip_positions=clip_positions,
                clip_duration=args.vbr_clip_duration,
                max_safety_limit=args.vbr_max_trials,
                auto_tune=args.animation_tune
            )
        
        if result.get('success', False):
            vbr_results[file] = result
            logger.vbr(f"SUCCESS {file.name}: {result['bitrate']}kbps, "
                      f"VMAF {result['vmaf_score']:.2f}")
        else:
            logger.vbr(f"FAILED {file.name}: Could not find suitable VBR settings")
    
    return vbr_results


def process_qp_mode(args, encoder: str, encoder_type: str, files: List[Path]) -> Dict[Path, int]:
    """Process files using QP optimization mode."""
    
    # Check if we need to find optimal QP
    if args.qp == 0:
        logger.debug(f"Finding optimal QP for {args.vmaf_target:.1f} VMAF")
        optimal_qp = find_optimal_qp(
            files, encoder, encoder_type,
            vmaf_target=args.vmaf_target,
            vmaf_min_threshold=args.vmaf_target - args.vmaf_tol,
            sample_duration=args.vbr_clip_duration
        )
        
        if optimal_qp > 0:
            logger.debug(f"Found optimal QP: {optimal_qp}")
        else:
            logger.debug(f"Failed to find optimal QP, using fallback: 23")
            optimal_qp = 23
    else:
        optimal_qp = args.qp
        logger.debug(f"Using specified QP: {optimal_qp}")
    
    # Process files with QP
    qp_map = {}
    for file in files:
        # Skip if wrong codec
        current_codec = get_video_codec(file)
        if should_skip_codec(current_codec):
            logger.debug(f"SKIP {file.name}: Already HEVC/AV1")
            continue
        
        qp_map[file] = optimal_qp
    
    return qp_map


def process_auto_mode(args, encoder: str, encoder_type: str, files: List[Path]) -> tuple[Dict[Path, Dict], Dict[Path, int]]:
    """Process files using automatic QP/VBR selection."""
    
    # Determine QP vs VBR based on file characteristics
    vbr_files = []
    qp_files = []
    
    for file in files:
        duration = get_duration_sec(file)
        file_size_mb = file.stat().st_size / (1024 * 1024)
        
        # Use VBR for shorter files or smaller files for better optimization
        if duration < 3600 or file_size_mb < 2000:  # Under 1 hour or 2GB
            vbr_files.append(file)
        else:
            qp_files.append(file)
    
    logger.debug(f"VBR optimization: {len(vbr_files)} files")
    logger.debug(f"QP optimization: {len(qp_files)} files")
    
    # Process VBR files
    vbr_results = {}
    if vbr_files:
        vbr_results = process_vbr_mode(args, encoder, encoder_type, vbr_files)
    
    # Process QP files  
    qp_map = {}
    if qp_files:
        qp_map = process_qp_mode(args, encoder, encoder_type, qp_files)
    
    return vbr_results, qp_map


def main():
    """Main entry point for the transcoding application."""
    parser = argparse.ArgumentParser(description="Lazy Transcode - Efficient video transcoding with VMAF optimization")
    
    # Input/output
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: auto-generated)")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--vbr", action="store_true", default=True,
                          help="Use VBR optimization mode (default)")
    mode_group.add_argument("--qp", type=int, default=0, metavar="QP", 
                          help="Use QP mode (0 = auto-find optimal QP)")
    mode_group.add_argument("--auto", action="store_true",
                          help="Auto-select VBR or QP based on file characteristics")
    
    # Quality settings
    parser.add_argument("--vmaf-target", type=float, default=95, 
                       help="Target VMAF score (default: 95.0)")
    parser.add_argument("--vmaf-tol", type=float, default=1.0,
                       help="VMAF tolerance (default: 1.0)")
    
    # VBR settings
    parser.add_argument("--vbr-clips", type=int, default=2,
                       help="Number of clips for VBR optimization (default: 2)")
    parser.add_argument("--vbr-clip-duration", type=int, default=60,
                       help="Duration of each VBR test clip in seconds (default: 60)")
    parser.add_argument("--vbr-coverage", type=float, default=0.10,
                       help="Target coverage percentage for VBR clip sampling (default: 0.10 = 10%%)")
    parser.add_argument("--vbr-max-trials", type=int, default=8,
                       help="Base maximum VBR trials - system adapts based on convergence (default: 8)")
    parser.add_argument("--vbr-method", choices=["bisection", "gradient-descent", "quasi-newton", "conjugate-gradient", "compare"],
                       default="bisection", 
                       help="VBR optimization method: bisection (classic), gradient methods (research-based), or compare all (default: bisection)")
    
    # Encoder settings
    parser.add_argument("--encoder", choices=["cpu", "nvenc", "amf", "qsv", "videotoolbox"],
                       help="Force specific encoder (default: auto-detect)")
    parser.add_argument("--cpu", action="store_true", 
                       help="Force CPU/software encoding (libx265) instead of hardware acceleration")
    parser.add_argument("--animation-tune", action="store_true", default=False,
                       help="Enable automatic animation tune detection for better anime compression (default: disabled)")
    parser.add_argument("--preserve-hdr", action="store_true", default=True,
                       help="Preserve HDR metadata (default: enabled)")
    
    # Processing options
    parser.add_argument("--include-h265", action="store_true",
                       help="Include H.265/HEVC files as candidates for transcoding (do not skip efficient codecs)")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without encoding")
    parser.add_argument("--non-destructive", action="store_true", 
                       help="Save transcoded files to 'Transcoded' subdirectory instead of replacing originals")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel encoding jobs")
    parser.add_argument("--verify", action="store_true", help="Verify quality after encoding")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-timestamps", action="store_true", help="Disable timestamps in log output")
    
    # Enhanced features
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files and samples recursively")
    parser.add_argument("--preset", choices=["fast", "medium", "slow"], default="medium",
                       help="Encoding preset (default: medium)")
    parser.add_argument("--encoder-type", choices=["software", "hardware"], default=None,
                       help="Encoder type (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Handle cleanup command
    if args.cleanup:
        input_path = Path(args.input)
        print(f"[INFO] Cleaning up temporary files in: {input_path}")
        file_manager = FileManager(debug=True, include_h265=getattr(args, 'include_h265', False))
        removed_count = file_manager.startup_scavenge(input_path)
        if removed_count > 0:
            print(f"[INFO] Cleaned up {removed_count} temporary file(s)")
        else:
            print("[INFO] No temporary files found to clean up")
        return 0
    
    # Set debug mode
    if args.debug:
        from ..utils.logging import set_debug_mode
        set_debug_mode(True)
        global DEBUG
        DEBUG = True
        
    # Set timestamp mode
    if args.no_timestamps:
        from ..utils.logging import set_timestamps_enabled
        set_timestamps_enabled(False)
    
    # Setup input/output
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.debug(f"ERROR: Input path does not exist: {input_path}")
        return 1
    
    # Discover files
    file_manager = FileManager(debug=args.debug, include_h265=getattr(args, 'include_h265', False))
    
    if input_path.is_file():
        files = [input_path]
    else:
        discovery_result = file_manager.discover_video_files(input_path)
        files = discovery_result.files_to_transcode
    
    # Apply limit if specified
    if args.limit:
        files = files[:args.limit]
        logger.debug(f"Processing first {len(files)} files")
    
    if not files:
        logger.debug("No video files found to process")
        return 0
    
    logger.discovery(f"Found {len(files)} video files")
    
    # Detect encoder
    encoder, encoder_type = detect_best_encoder()
    if args.cpu:
        # Force CPU encoding
        encoder, encoder_type = 'libx265', 'software'
        logger.debug(f"Forced CPU encoding via --cpu flag")
    elif args.encoder:
        # Override with user's choice
        if args.encoder == 'cpu':
            encoder, encoder_type = 'libx265', 'software'
        elif args.encoder == 'nvidia':
            encoder, encoder_type = 'hevc_nvenc', 'hardware'
        elif args.encoder == 'amd':
            encoder, encoder_type = 'hevc_amf', 'hardware'
        elif args.encoder == 'intel':
            encoder, encoder_type = 'hevc_qsv', 'hardware'
    logger.debug(f"Selected: {encoder} ({encoder_type})")
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        # Default behavior: create numbered Transcoded_X directory (non-destructive)
        output_dir = get_next_transcoded_dir(input_path.parent if input_path.is_file() else input_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")
    
    # Dry run check
    if args.dry_run:
        print("\n[DRY-RUN] Files that would be processed:")
        for file in files:
            print(f"  - {file}")
        return 0
    
    # Process based on mode
    vbr_results = {}
    qp_map = {}
    
    if args.qp > 0:
        print(f"\n[MODE] QP Fixed ({args.qp})")
        qp_map = process_qp_mode(args, encoder, encoder_type, files)
    elif args.auto:
        print(f"\n[MODE] Auto (VBR + QP)")
        vbr_results, qp_map = process_auto_mode(args, encoder, encoder_type, files)
    else:
        print(f"\n[MODE] VBR Optimization")
        vbr_results = process_vbr_mode(args, encoder, encoder_type, files)
    
    # Verification step
    if args.verify and (vbr_results or qp_map):
        # For verification, we need to convert vbr_results to a simple file list
        verify_files = list(vbr_results.keys()) if vbr_results else list(qp_map.keys())
        verify_qp = 23  # Default QP for verification
        if qp_map:
            verify_qp = next(iter(qp_map.values()))  # Get first QP value
        
        success = verify_and_prompt_transcode(verify_files, verify_qp, encoder, encoder_type, args, args.preserve_hdr)
        if not success:
            print("[CANCELLED] User cancelled transcoding")
            return 0
    
    # Execute transcoding
    if vbr_results:
        print(f"\n[TRANSCODE-VBR] Processing {len(vbr_results)} files")
        
        for file, result in vbr_results.items():
            output_file = output_dir / f"{file.stem}_vbr_{result['bitrate']}k{file.suffix}"
            
            print(f"[VBR-ENCODE] {file.name} -> {output_file.name}")
            success = transcode_file_vbr(
                file, output_file, encoder, encoder_type,
                max_bitrate=result['bitrate'], 
                avg_bitrate=int(result['bitrate'] * 0.8),  # 80% of max for average
                preserve_hdr_metadata=args.preserve_hdr,
                auto_tune=args.animation_tune
            )
            
            if success:
                print(f"[VBR-COMPLETE] {output_file.name}")
            else:
                print(f"[VBR-FAILED] {file.name}")
    
    if qp_map:
        print(f"\n[TRANSCODE-QP] Processing {len(qp_map)} files")
        
        if args.parallel > 1:
            # Use parallel processor
            processor = ParallelTranscoder(vmaf_threads=args.parallel)
            processor.process_files_with_qp_map(
                list(qp_map.keys()), qp_map, encoder, encoder_type, 
                vmaf_target=args.vmaf_target, 
                vmaf_min=args.vmaf_target - args.vmaf_tol,
                preserve_hdr_metadata=args.preserve_hdr
            )
        else:
            # Sequential processing
            for file, qp in qp_map.items():
                output_file = output_dir / f"{file.stem}_q{qp}{file.suffix}"
                
                print(f"[QP-ENCODE] {file.name} -> {output_file.name}")
                success = transcode_file_qp(file, output_file, encoder, encoder_type, qp)
                
                if success:
                    print(f"[QP-COMPLETE] {output_file.name}")
                else:
                    print(f"[QP-FAILED] {file.name}")
    
    # Display summary
    if vbr_results:
        display_vbr_results_summary(vbr_results, len(files))
    
    print(f"\n[COMPLETE] Transcoding finished")
    print(f"[OUTPUT] Results in: {output_dir}")
    
    return 0


if __name__ == "__main__":
    import signal
    import atexit
    from .modules.system.system_utils import _cleanup
    
    # Setup cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(1)))
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Cleaning up...")
        _cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        _cleanup()
        sys.exit(1)
