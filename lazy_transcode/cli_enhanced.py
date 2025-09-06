"""
Enhanced CLI for lazy-transcode with improved logging and user experience.

Provides auto-detection between enhanced vs original mode with backwards compatibility.
Environment variable support: LAZY_TRANSCODE_MODE=enhanced
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

from lazy_transcode.utils.logging import get_logger, set_debug_mode
from lazy_transcode.utils.smart_logging import (
    init_smart_logging, set_log_level_from_args, LogLevel
)

logger = get_logger("cli_enhanced")

def detect_enhanced_mode() -> bool:
    """Detect if enhanced mode should be used."""
    # Check environment variable
    env_mode = os.getenv('LAZY_TRANSCODE_MODE', '').lower()
    if env_mode == 'enhanced':
        return True
    elif env_mode == 'original':
        return False
    
    # Default to enhanced mode - benefits outweigh minimal overhead
    return True

def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create argument parser with enhanced options."""
    parser = argparse.ArgumentParser(
        description="lazy-transcode: Enhanced video transcoding with smart logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  - Smart Logging: Progressive verbosity levels (--quiet, --verbose, --debug)
  - Auto-encoder Detection: Automatically selects best available encoder
  - Animation Tune: Automatic detection and optimization for anime content
  
Environment Variables:
  LAZY_TRANSCODE_MODE=enhanced    Force enhanced mode
  LAZY_TRANSCODE_MODE=original    Force original mode (default: auto-detect)
        """
    )
    
    # Core arguments (backwards compatible)
    parser.add_argument("input", nargs="?", default=".", help="Input video file or directory (default: current directory)")
    parser.add_argument("-o", "--output", help="Output directory (default: auto-generated)")
    parser.add_argument("--mode", choices=["vbr", "qp", "auto"], default="vbr",
                       help="Encoding mode: vbr (default), qp, auto")
    parser.add_argument("--vmaf-target", type=float, default=95.0,
                       help="Target VMAF score (default: 95.0)")
    parser.add_argument("--vmaf-tolerance", type=float, default=1.0,
                       help="VMAF tolerance (default: 1.0)")
    
    # VBR optimization settings
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
    parser.add_argument("--encoder", default=None,
                       help="Video encoder (default: auto-detect)")
    parser.add_argument("--encoder-type", choices=["software", "hardware"], default=None,
                       help="Encoder type (default: auto-detect)")
    parser.add_argument("--preset", choices=["fast", "medium", "slow"], default="medium",
                       help="Encoding preset (default: medium)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU encoding (disable hardware acceleration)")
    parser.add_argument("--animation-tune", action="store_true", default=False,
                       help="Enable automatic animation tune detection for better anime compression (default: disabled)")
    parser.add_argument("--preserve-hdr", action="store_true", default=True,
                       help="Preserve HDR metadata (default: enabled)")
    
    # Processing options
    parser.add_argument("--parallel", type=int, default=1, 
                       help="Number of parallel encoding jobs (default: 1)")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify quality after encoding")
    parser.add_argument("--non-destructive", action="store_true",
                       help="Save transcoded files to 'Transcoded' subdirectory instead of replacing originals")
    parser.add_argument("--local-state", action="store_true",
                       help="Force local state storage (disable network path resolution)")
    
    # Enhanced mode specific arguments
    parser.add_argument("--include-h265", action="store_true",
                       help="Include H.265/HEVC files as candidates for transcoding (do not skip efficient codecs)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files and samples recursively")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of files to process (useful for testing)")
    parser.add_argument("--force-enhanced", action="store_true",
                       help="Force enhanced mode")
    parser.add_argument("--force-original", action="store_true",
                       help="Force original mode")
    parser.add_argument("--network-retries", type=int, default=6,
                       help="Maximum network retry attempts (default: 6)")
    
    # Verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--quiet", action="store_true",
                       help="Minimal output (essential progress only)")
    verbosity_group.add_argument("--verbose", action="store_true",
                       help="Detailed output (normal + key operations)")
    verbosity_group.add_argument("--very-verbose", action="store_true",
                       help="Technical output (verbose + technical details)")
    verbosity_group.add_argument("--debug", action="store_true",
                       help="Debug output (all messages + internal calculations)")
    
    parser.add_argument("--no-timestamps", action="store_true", 
                       help="Disable timestamps in log output")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    return parser

def handle_cleanup(directory: Path):
    """Handle --cleanup command."""
    from lazy_transcode.core.modules.processing.file_manager import FileManager
    
    logger.info(f"Cleaning up temporary files in: {directory}")
    
    from argparse import Namespace
    # Use a dummy args object to allow for future expansion
    args = getattr(handle_cleanup, "args", Namespace())
    include_h265 = getattr(args, "include_h265", False)
    file_manager = FileManager(debug=True, include_h265=include_h265)
    removed_count = file_manager.startup_scavenge(directory)
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} temporary file(s)")
    else:
        logger.info("No temporary files found to clean up")

def run_enhanced_vbr_optimization(args) -> dict:
    """Run VBR optimization in enhanced mode."""
    # Use original path, don't automatically resolve to UNC
    input_path = Path(args.input)
    
    # Handle CPU flag - override encoder selection
    if args.cpu:
        args.encoder_type = "software"
        if args.encoder is None or args.encoder in ["hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox"]:
            args.encoder = "libx265"  # Default software HEVC encoder
        elif args.encoder in ["h264_nvenc", "h264_qsv", "h264_amf"]:
            args.encoder = "libx264"  # Default software H.264 encoder
    
    # Auto-detect encoder if not specified
    if args.encoder is None:
        from lazy_transcode.core.modules.processing.transcoding_engine import detect_best_encoder
        args.encoder, args.encoder_type = detect_best_encoder(force_cpu=getattr(args, 'cpu', False))
    elif args.encoder_type is None:
        # If encoder is specified but type is not, determine type
        args.encoder_type = "software" if args.cpu else ("software" if args.encoder in ["libx265", "libx264"] else "hardware")
    
    # Validate input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    if args.dry_run:
        logger.info("DRY RUN: Would optimize with VBR mode")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Target VMAF: {args.vmaf_target}")
        logger.info(f"  Encoder: {args.encoder} ({args.encoder_type})")
        return {"dry_run": True}
    
    try:
        from lazy_transcode.core.modules.optimization.vbr_optimizer import optimize_encoder_settings_vbr
        from lazy_transcode.core.modules.optimization.coverage_clips import get_coverage_based_vbr_clip_positions
        from lazy_transcode.core.modules.analysis.media_utils import get_duration_sec
        
        # Get clip positions for VBR optimization
        duration_seconds = get_duration_sec(input_path)
        clip_positions = get_coverage_based_vbr_clip_positions(input_path, duration_seconds)
        
        result = optimize_encoder_settings_vbr(
            infile=input_path,
            encoder=args.encoder,
            encoder_type=args.encoder_type,
            target_vmaf=args.vmaf_target,
            vmaf_tolerance=args.vmaf_tolerance,
            clip_positions=clip_positions,
            clip_duration=30,
            auto_tune=getattr(args, 'animation_tune', False)
        )
        
        logger.info("VBR optimization completed successfully!")
        return result
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted")
        sys.exit(130)  # Standard exit code for Ctrl+C

def main_enhanced():
    """Main entry point for enhanced CLI."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    # Initialize smart logging system
    log_level = set_log_level_from_args(args)
    smart_logger = init_smart_logging(
        level=log_level, 
        show_timestamps=not getattr(args, 'no_timestamps', False)
    )
    
    # Configure legacy logging
    if args.debug:
        set_debug_mode(True)
    
    # Determine mode
    use_enhanced = True
    if args.force_original:
        use_enhanced = False
    elif args.force_enhanced:
        use_enhanced = True
    else:
        use_enhanced = detect_enhanced_mode()
    
    if not use_enhanced:
        logger.info("Using original mode - delegating to core CLI")
        # Import and call original CLI
        from lazy_transcode.core.main import main
        main()
        return
    
    # Enhanced mode
    if log_level == LogLevel.ESSENTIAL:
        smart_logger.progress("Starting enhanced mode")
    else:
        logger.info("Running in enhanced mode")
    
    try:
        # Propagate args to cleanup handler for include_h265
        handle_cleanup.args = args

        # Handle special commands
        if args.cleanup:
            directory = Path(args.input).resolve() if args.input != "." else Path.cwd()
            handle_cleanup(directory)
            return

        # Handle current directory or specified folder
        if args.input == "." or Path(args.input).is_dir():
            # Use specified directory or current directory
            search_dir = Path(args.input).resolve() if args.input != "." else Path.cwd()
            video_extensions = {'.mkv', '.mp4', '.mov', '.ts', '.avi', '.m4v'}
            video_files = [f for f in search_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in video_extensions]

            if not video_files:
                logger.error(f"No video files found in folder: {search_dir}")
                sys.exit(1)

            # Sort files for consistent processing order
            video_files = sorted(video_files)

            # Apply limit if specified
            if args.limit is not None:
                original_count = len(video_files)
                video_files = video_files[:args.limit]
                if original_count != len(video_files):
                    logger.info(f"Limited processing to {len(video_files)} of {original_count} files")

            logger.info(f"Found {len(video_files)} video file(s) in folder: {search_dir}")
            for video_file in video_files:
                logger.info(f"Processing: {video_file.name}")
                args.input = str(video_file)
                if args.mode == "vbr":
                    result = run_enhanced_vbr_optimization(args)
                    logger.result(f"Optimization completed for {video_file.name}: {result}")
                elif args.mode == "qp":
                    logger.error("QP mode not yet implemented in enhanced CLI")
                    logger.info("Use --force-original for QP mode")
                    sys.exit(1)
                elif args.mode == "auto":
                    logger.error("Auto mode not yet implemented in enhanced CLI")
                    logger.info("Use --force-original for auto mode")
                    sys.exit(1)
            return

        # Validate input for normal operation - no longer needed since we have default
        # Input is guaranteed to exist due to default="." and directory processing above

        # Run optimization
        if args.mode == "vbr":
            result = run_enhanced_vbr_optimization(args)
            logger.result(f"Optimization completed: {result}")
        elif args.mode == "qp":
            logger.error("QP mode not yet implemented in enhanced CLI")
            logger.info("Use --force-original for QP mode")
            sys.exit(1)
        elif args.mode == "auto":
            logger.error("Auto mode not yet implemented in enhanced CLI")
            logger.info("Use --force-original for auto mode")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Enhanced mode failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main_enhanced()
