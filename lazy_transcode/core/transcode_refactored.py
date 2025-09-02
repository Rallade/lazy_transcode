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
from ..utils.logging import get_logger, set_debug_mode

# Module logger
logger = get_logger("transcode_main")
from .modules.vbr_optimizer import (
    calculate_intelligent_vbr_bounds, optimize_encoder_settings_vbr,
    get_vbr_clip_positions, build_vbr_encode_cmd
)
from .modules.qp_optimizer import (
    find_optimal_qp, adaptive_qp_search_per_file, extract_random_clips,
    test_qp_on_sample
)
from .modules.transcoding_engine import (
    build_encode_cmd, transcode_file_qp, 
    transcode_file_vbr, detect_best_encoder
)
from .modules.job_processor import (
    TranscodeJob, AsyncFileStager, ParallelTranscoder
)
from .modules.user_interface import (
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
        
        # Get duration and calculate clips
        duration = get_duration_sec(file)
        if duration <= 0:
            logger.debug(f"SKIP {file.name}: Could not determine duration")
            continue
            
        # Auto-scale clips based on video length
        vbr_clips = args.vbr_clips
        if duration > 0:
            # Auto-scale clips: 1 clip per 30 minutes, min 2, max 6
            auto_clips = max(2, min(6, int(duration / 1800)))  # 1800s = 30min
            if auto_clips != vbr_clips:
                logger.debug(f"Auto-scaling clips from {vbr_clips} to {auto_clips} based on {duration/60:.1f}min duration")
                vbr_clips = auto_clips
                
        clip_positions = get_vbr_clip_positions(duration, vbr_clips)
        
        # Run VBR optimization
        result = optimize_encoder_settings_vbr(
            file, encoder, encoder_type,
            target_vmaf=args.vmaf_target,
            vmaf_tolerance=args.vmaf_tol,
            clip_positions=clip_positions,
            clip_duration=args.vbr_clip_duration,
            max_trials=args.vbr_max_trials
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
    mode_group.add_argument("--vbr", action="store_true", help="Use VBR optimization mode")
    mode_group.add_argument("--qp", type=int, default=0, metavar="QP", 
                          help="Use QP mode (0 = auto-find optimal QP)")
    mode_group.add_argument("--auto", action="store_true", default=True,
                          help="Auto-select VBR or QP based on file characteristics (default)")
    
    # Quality settings
    parser.add_argument("--vmaf-target", type=float, default=92.0, 
                       help="Target VMAF score (default: 92.0)")
    parser.add_argument("--vmaf-tol", type=float, default=1.0,
                       help="VMAF tolerance (default: 1.0)")
    
    # VBR settings
    parser.add_argument("--vbr-clips", type=int, default=2,
                       help="Number of clips for VBR optimization (default: 2)")
    parser.add_argument("--vbr-clip-duration", type=int, default=60,
                       help="Duration of each VBR test clip in seconds (default: 60)")
    parser.add_argument("--vbr-max-trials", type=int, default=8,
                       help="Maximum VBR optimization trials (default: 8)")
    
    # Encoder settings
    parser.add_argument("--encoder", choices=["cpu", "nvenc", "amf", "qsv", "videotoolbox"],
                       help="Force specific encoder (default: auto-detect)")
    parser.add_argument("--cpu", action="store_true", 
                       help="Force CPU/software encoding (libx265) instead of hardware acceleration")
    parser.add_argument("--preserve-hdr", action="store_true", default=True,
                       help="Preserve HDR metadata (default: enabled)")
    
    # Processing options
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without encoding")
    parser.add_argument("--non-destructive", action="store_true", 
                       help="Save transcoded files to 'Transcoded' subdirectory instead of replacing originals")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel encoding jobs")
    parser.add_argument("--verify", action="store_true", help="Verify quality after encoding")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-timestamps", action="store_true", help="Disable timestamps in log output")
    
    args = parser.parse_args()
    
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
    file_manager = FileManager(debug=args.debug)
    
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
    
    if args.vbr:
        print(f"\n[MODE] VBR Optimization")
        vbr_results = process_vbr_mode(args, encoder, encoder_type, files)
    elif args.qp > 0:
        print(f"\n[MODE] QP Fixed ({args.qp})")
        qp_map = process_qp_mode(args, encoder, encoder_type, files)
    else:
        print(f"\n[MODE] Auto (VBR + QP)")
        vbr_results, qp_map = process_auto_mode(args, encoder, encoder_type, files)
    
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
                preserve_hdr_metadata=args.preserve_hdr
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
    from .modules.system_utils import _cleanup
    
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
