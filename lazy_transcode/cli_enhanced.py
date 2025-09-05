"""
Enhanced CLI for lazy-transcode with pause/resume and network resilience.

Provides auto-detection between enhanced vs original mode with backwards compatibility.
Environment variable support: LAZY_TRANSCODE_MODE=enhanced
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

from lazy_transcode.core.modules.system.pause_manager import get_pause_manager, cleanup_pause_manager
from lazy_transcode.core.modules.system.network_utils import get_network_accessor, cleanup_network_accessor
from lazy_transcode.core.modules.optimization.vbr_optimizer_enhanced import (
    optimize_encoder_settings_vbr_enhanced,
    resume_vbr_optimization,
    list_resumable_optimizations
)
from lazy_transcode.utils.logging import get_logger, set_debug_mode

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

def setup_enhanced_environment():
    """Setup enhanced environment with pause/resume and network resilience."""
    logger.info("Initializing enhanced mode...")
    
    # Initialize systems
    pause_manager = get_pause_manager()
    network_accessor = get_network_accessor()
    
    logger.info("Enhanced mode initialized with pause/resume and network resilience")
    return pause_manager, network_accessor

def cleanup_enhanced_environment():
    """Cleanup enhanced environment."""
    logger.debug("Cleaning up enhanced environment...")
    cleanup_pause_manager()
    cleanup_network_accessor()

def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create argument parser with enhanced options."""
    parser = argparse.ArgumentParser(
        description="lazy-transcode: Enhanced video transcoding with pause/resume and network resilience",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  - Pause/Resume: Press Ctrl+C once for graceful pause, twice for immediate exit
  - Network Resilience: Automatic retry and reconnection for network drives
  - State Persistence: Resume interrupted optimizations
  
Environment Variables:
  LAZY_TRANSCODE_MODE=enhanced    Force enhanced mode
  LAZY_TRANSCODE_MODE=original    Force original mode (default: auto-detect)
        """
    )
    
    # Core arguments (backwards compatible)
    parser.add_argument("input", nargs="?", default=".", help="Input video file or directory (default: current directory)")
    parser.add_argument("--mode", choices=["vbr", "qp"], default="vbr",
                       help="Encoding mode (default: vbr)")
    parser.add_argument("--vmaf-target", type=float, default=95.0,
                       help="Target VMAF score (default: 95.0)")
    parser.add_argument("--vmaf-tolerance", type=float, default=1.0,
                       help="VMAF tolerance (default: 1.0)")
    parser.add_argument("--encoder", default=None,
                       help="Video encoder (default: auto-detect)")
    parser.add_argument("--encoder-type", choices=["software", "hardware"], default=None,
                       help="Encoder type (default: auto-detect)")
    parser.add_argument("--preset", choices=["fast", "medium", "slow"], default="medium",
                       help="Encoding preset (default: medium)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU encoding (disable hardware acceleration)")
    parser.add_argument("--local-state", action="store_true",
                       help="Force local state storage (disable network path resolution)")
    
    # Enhanced mode specific arguments
    parser.add_argument("--resume", action="store_true",
                       help="Resume interrupted optimization")
    parser.add_argument("--list-resumable", action="store_true",
                       help="List resumable optimizations in directory")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files and samples recursively")
    parser.add_argument("--force-enhanced", action="store_true",
                       help="Force enhanced mode")
    parser.add_argument("--force-original", action="store_true",
                       help="Force original mode")
    parser.add_argument("--network-retries", type=int, default=6,
                       help="Maximum network retry attempts (default: 6)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    return parser

def handle_list_resumable(directory: Path):
    """Handle --list-resumable command."""
    logger.info(f"Searching for resumable optimizations in: {directory}")
    
    resumable = list_resumable_optimizations(directory)
    
    if not resumable:
        logger.info("No resumable optimizations found")
        return
    
    logger.info(f"Found {len(resumable)} resumable optimization(s):")
    for i, opt in enumerate(resumable, 1):
        logger.info(f"  {i}. {opt['input_file']}")
        logger.info(f"     Step: {opt['current_step']}")
        logger.info(f"     Started: {opt.get('start_time', 'Unknown')}")
        logger.info(f"     Completed: {opt['completed']}")

def handle_cleanup(directory: Path):
    """Handle --cleanup command."""
    from lazy_transcode.core.modules.processing.file_manager import FileManager
    
    logger.info(f"Cleaning up temporary files in: {directory}")
    
    file_manager = FileManager(debug=True)
    removed_count = file_manager.startup_scavenge(directory)
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} temporary file(s)")
    else:
        logger.info("No temporary files found to clean up")

def handle_resume(input_path: Path):
    """Handle --resume command."""
    logger.info(f"Attempting to resume optimization for: {input_path}")
    
    result = resume_vbr_optimization(input_path)
    
    if result:
        logger.info("Optimization resumed and completed successfully!")
        logger.info(f"Result: {result}")
    else:
        logger.error("No resumable optimization found or resume failed")
        sys.exit(1)

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
        logger.info("DRY RUN: Would optimize with enhanced VBR mode")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Target VMAF: {args.vmaf_target}")
        logger.info(f"  Encoder: {args.encoder} ({args.encoder_type})")
        return {"dry_run": True}
    
    try:
        result = optimize_encoder_settings_vbr_enhanced(
            input_file=input_path,
            encoder=args.encoder,
            encoder_type=args.encoder_type,
            target_vmaf=args.vmaf_target,
            vmaf_tolerance=args.vmaf_tolerance
        )
        
        logger.info("Enhanced VBR optimization completed successfully!")
        return result
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted - state saved for resume")
        logger.info("Use --resume to continue the optimization")
        sys.exit(130)  # Standard exit code for Ctrl+C

def main_enhanced():
    """Main entry point for enhanced CLI."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    # Configure logging
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
    logger.info("Running in enhanced mode")
    
    try:
        # Setup enhanced environment
        setup_enhanced_environment()
        
        # Handle special commands
        if args.list_resumable:
            directory = Path(args.input).resolve() if args.input != "." else Path.cwd()
            handle_list_resumable(directory)
            return
        
        if args.cleanup:
            directory = Path(args.input).resolve() if args.input != "." else Path.cwd()
            handle_cleanup(directory)
            return
        
        if args.resume:
            if args.input == ".":
                logger.error("--resume requires specific input file path")
                sys.exit(1)
            handle_resume(Path(args.input).resolve())
            return
        
        # Handle current directory default
        if args.input == ".":
            # Process current directory for video files
            current_dir = Path.cwd()
            video_extensions = {'.mkv', '.mp4', '.mov', '.ts', '.avi', '.m4v'}
            video_files = [f for f in current_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in video_extensions]
            
            if not video_files:
                logger.error(f"No video files found in current directory: {current_dir}")
                sys.exit(1)
            
            # Sort files for consistent processing order
            video_files = sorted(video_files)
            
            logger.info(f"Found {len(video_files)} video file(s) in current directory")
            for video_file in video_files:
                logger.info(f"Processing: {video_file.name}")
                # Use relative path to avoid UNC resolution
                args.input = video_file.name
                if args.mode == "vbr":
                    result = run_enhanced_vbr_optimization(args)
                    logger.result(f"Optimization completed for {video_file.name}: {result}")
                else:
                    logger.error("QP mode not yet implemented in enhanced CLI")
                    logger.info("Use --force-original for QP mode")
                    sys.exit(1)
            return
        
        # Validate input for normal operation - no longer needed since we have default
        # Input is guaranteed to exist due to default="." and directory processing above
        
        # Run optimization
        if args.mode == "vbr":
            result = run_enhanced_vbr_optimization(args)
            logger.result(f"Optimization completed: {result}")
        else:
            logger.error("QP mode not yet implemented in enhanced CLI")
            logger.info("Use --force-original for QP mode")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Enhanced mode failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_enhanced_environment()

if __name__ == "__main__":
    main_enhanced()
