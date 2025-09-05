"""
Enhanced main.py with integrated universal state management.

This demonstrates how to add state management to existing functions
with minimal changes to the core logic.
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# Import the state management system
from lazy_transcode.core.modules.system.transcode_state_manager import (
    UniversalStateManager, TranscodeOperation
)
from lazy_transcode.core.modules.system.state_integration import (
    StateAwareOperation, find_resumable_operations
)

# Import existing modules
from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder
from lazy_transcode.core.modules.analysis.vmaf_evaluator import VMAfEvaluator
from lazy_transcode.core.modules.processing.file_manager import FileManager
from lazy_transcode.core.modules.system.system_utils import (
    TEMP_FILES, format_size, get_next_transcoded_dir, 
    start_cpu_monitor, DEBUG
)
from lazy_transcode.core.modules.analysis.media_utils import (
    get_duration_sec, get_video_codec, should_skip_codec, 
    compute_vmaf_score
)
from lazy_transcode.utils.logging import get_logger, set_debug_mode

logger = get_logger("transcode_main_enhanced")

def process_vbr_mode_with_state(args, encoder: str, encoder_type: str, 
                               files: List[Path]) -> Dict[Path, Dict]:
    """VBR processing with universal state management."""
    
    if len(files) == 1:
        return _process_single_vbr_with_state(args, encoder, encoder_type, files[0])
    else:
        return _process_batch_vbr_with_state(args, encoder, encoder_type, files)

def _process_single_vbr_with_state(args, encoder: str, encoder_type: str, 
                                  input_file: Path) -> Dict[Path, Dict]:
    """Process single file VBR with state management."""
    
    # Create operation description
    operation = TranscodeOperation(
        operation_type='vbr',
        input_file=str(input_file),
        encoder=encoder,
        encoder_type=encoder_type,
        target_vmaf=args.vmaf_target,
        vmaf_tolerance=args.vmaf_tol,
        preserve_hdr=args.preserve_hdr,
        non_destructive=getattr(args, 'non_destructive', False),
        dry_run=getattr(args, 'dry_run', False)
    )
    
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Check if resuming from previous state
        if state_manager.state.current_step != 'initialization':
            logger.info(f"Resuming VBR optimization from: {state_manager.state.current_step}")
            
            # If already completed, return cached result
            if state_manager.state.completed and state_manager.state.best_result:
                logger.info("VBR optimization already completed, using cached result")
                return {input_file: state_manager.state.best_result}
        
        # Step 1: File validation
        state_manager.update_step('file_validation', progress=10.0)
        
        current_codec = get_video_codec(input_file)
        if should_skip_codec(current_codec):
            result = {'skipped': True, 'reason': 'Already HEVC/AV1'}
            state_manager.mark_completed(success=True, result=result)
            return {input_file: result}
        
        # Step 2: Duration analysis
        state_manager.update_step('duration_analysis', progress=20.0)
        
        duration = get_duration_sec(input_file)
        if duration <= 0:
            result = {'skipped': True, 'reason': 'Could not determine duration'}
            state_manager.mark_completed(success=True, result=result)
            return {input_file: result}
        
        # Step 3: Clip position calculation (only if not already done)
        if not state_manager.state.clips_extracted:
            state_manager.update_step('clip_calculation', progress=30.0)
            
            from lazy_transcode.core.modules.optimization.coverage_clips import (
                get_coverage_based_vbr_clip_positions
            )
            
            logger.info(f"Using coverage-based clip selection for {duration/60:.1f}min video")
            clip_positions = get_coverage_based_vbr_clip_positions(
                input_file, duration, clip_duration=30, target_coverage=0.10
            )
            
            # Save clip positions to state
            state_manager.update_step('clips_calculated', progress=40.0, data={
                'clips_extracted': True,
                'clip_positions': clip_positions
            })
        else:
            # Restore clip positions from state
            clip_positions = state_manager.state.operation.clip_positions
            logger.info("Using previously calculated clip positions")
        
        # Step 4: VBR optimization
        state_manager.update_step('vbr_optimization', progress=50.0)
        
        # Import here to avoid circular dependencies
        from lazy_transcode.core.modules.optimization.vbr_optimizer import (
            optimize_encoder_settings_vbr
        )
        
        logger.info("Starting VBR optimization...")
        
        try:
            vbr_result = optimize_encoder_settings_vbr(
                infile=input_file,
                encoder=encoder,
                encoder_type=encoder_type,
                target_vmaf=args.vmaf_target,
                vmaf_tolerance=args.vmaf_tol,
                clip_positions=clip_positions,
                clip_duration=30
            )
            
            state_manager.update_step('optimization_complete', progress=90.0)
            
            # Step 5: Result processing
            result = {
                'success': vbr_result.get('success', False),
                'bitrate': vbr_result.get('bitrate'),
                'vmaf_score': vbr_result.get('vmaf_score'),
                'preset': vbr_result.get('preset', 'medium'),
                'filesize': vbr_result.get('filesize'),
                'optimization_time': time.time() - state_manager.state.start_time
            }
            
            state_manager.update_step('complete', progress=100.0)
            state_manager.state.best_result = result
            
            logger.info(f"VBR optimization completed: {result['bitrate']}kbps @ VMAF {result['vmaf_score']:.2f}")
            
            return {input_file: result}
            
        except Exception as e:
            logger.error(f"VBR optimization failed: {e}")
            result = {'success': False, 'error': str(e)}
            raise

def _process_batch_vbr_with_state(args, encoder: str, encoder_type: str, 
                                 files: List[Path]) -> Dict[Path, Dict]:
    """Process multiple files with batch state management."""
    
    operation = TranscodeOperation(
        operation_type='batch_vbr',
        input_file=str(files[0].parent),
        encoder=encoder,
        encoder_type=encoder_type,
        target_vmaf=args.vmaf_target,
        vmaf_tolerance=args.vmaf_tol,
        extra_args={
            'file_count': len(files),
            'file_list': [str(f) for f in files]
        }
    )
    
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Check for resumable state
        processed_files = set()
        if state_manager.state.optimization_results:
            processed_files = set(Path(f) for f in state_manager.state.optimization_results.keys())
            logger.info(f"Resuming batch: {len(processed_files)} files already processed")
        
        results = state_manager.state.optimization_results or {}
        total_files = len(files)
        
        for i, file in enumerate(files):
            # Skip if already processed
            if file in processed_files:
                continue
                
            state_manager.check_pause_point()
            
            progress = (i / total_files) * 100
            state_manager.update_step(f'processing_{file.name}', progress=progress)
            
            logger.info(f"Processing file {i+1}/{total_files}: {file.name}")
            
            try:
                # Process single file (without nested state management for simplicity)
                file_result = _process_single_vbr_with_state(args, encoder, encoder_type, file)
                results.update(file_result)
                
                # Save intermediate results
                state_manager.state.optimization_results = {str(k): v for k, v in results.items()}
                
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {e}")
                results[file] = {'success': False, 'error': str(e)}
        
        state_manager.update_step('batch_complete', progress=100.0)
        return results

def process_qp_mode_with_state(args, encoder: str, encoder_type: str, 
                              files: List[Path]) -> Dict[Path, int]:
    """QP processing with universal state management."""
    
    operation = TranscodeOperation(
        operation_type='qp',
        input_file=str(files[0]) if len(files) == 1 else str(files[0].parent),
        encoder=encoder,
        encoder_type=encoder_type,
        target_qp=getattr(args, 'qp', None),
        quality_threshold=getattr(args, 'quality_threshold', None)
    )
    
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        logger.info(f"Starting QP optimization for {len(files)} files")
        
        results = {}
        
        for i, file in enumerate(files):
            state_manager.check_pause_point()
            
            progress = (i / len(files)) * 100
            state_manager.update_step(f'qp_analysis_{file.name}', progress=progress)
            
            logger.info(f"Finding optimal QP for {file.name}")
            
            # Import QP optimizer
            from lazy_transcode.core.modules.optimization.qp_optimizer import find_optimal_qp
            
            optimal_qp = find_optimal_qp(
                file, encoder, encoder_type,
                target_quality=getattr(args, 'quality_threshold', 95.0)
            )
            
            results[file] = optimal_qp
            
            # Update state with current results
            state_manager.state.optimization_results = {str(k): v for k, v in results.items()}
        
        state_manager.update_step('qp_complete', progress=100.0)
        return results

# CLI functions for state management

def add_state_management_args(parser: argparse.ArgumentParser):
    """Add state management arguments to CLI parser."""
    state_group = parser.add_argument_group('State Management')
    state_group.add_argument('--resume', action='store_true',
                           help='Resume interrupted operations')
    state_group.add_argument('--list-resumable', action='store_true',
                           help='List resumable operations')
    state_group.add_argument('--cleanup-old-states', type=int, metavar='DAYS',
                           help='Clean up completed states older than N days')

def handle_state_management_cli(args, input_path: Path):
    """Handle state management CLI commands."""
    
    if args.list_resumable:
        from lazy_transcode.core.modules.system.state_integration import list_resumable_cli
        list_resumable_cli(input_path)
        return True
    
    if args.cleanup_old_states:
        from lazy_transcode.core.modules.system.transcode_state_manager import cleanup_completed_states
        cleanup_completed_states(input_path, args.cleanup_old_states)
        logger.info(f"Cleaned up states older than {args.cleanup_old_states} days")
        return True
    
    if args.resume:
        resumable = find_resumable_operations(input_path)
        if resumable:
            logger.info(f"Found {len(resumable)} resumable operations")
            # The regular processing will automatically resume
        else:
            logger.info("No resumable operations found")
    
    return False

# Example of enhanced main function with state management
def main_with_state_management():
    """Enhanced main function with universal state management."""
    
    parser = argparse.ArgumentParser(description="Lazy Transcode with State Management")
    
    # Existing arguments
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--mode", choices=['vbr', 'qp', 'auto'], default='auto')
    parser.add_argument("--vmaf-target", type=float, default=92.0)
    parser.add_argument("--vmaf-tol", type=float, default=1.0)
    parser.add_argument("--encoder", choices=['auto', 'nvenc', 'amf', 'qsv', 'x265'])
    parser.add_argument("--dry-run", action='store_true')
    
    # Add state management arguments
    add_state_management_args(parser)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Handle state management CLI commands
    if handle_state_management_cli(args, input_path):
        return
    
    # Discover files
    file_manager = FileManager()
    files = file_manager.discover_video_files(input_path)
    
    if not files:
        logger.error("No video files found")
        return
    
    # Detect encoder
    from lazy_transcode.core.modules.processing.transcoding_engine import detect_best_encoder
    encoder, encoder_type = detect_best_encoder()
    
    logger.info(f"Using encoder: {encoder} ({encoder_type})")
    logger.info(f"Processing {len(files)} files in {args.mode} mode")
    
    # Process based on mode with state management
    if args.mode == 'vbr':
        results = process_vbr_mode_with_state(args, encoder, encoder_type, files)
    elif args.mode == 'qp':
        results = process_qp_mode_with_state(args, encoder, encoder_type, files)
    else:  # auto mode
        # For auto mode, we could combine both VBR and QP with state management
        logger.info("Auto mode with state management not yet implemented")
        return
    
    # Display results
    successful = sum(1 for r in results.values() if r.get('success', False))
    logger.info(f"Processing complete: {successful}/{len(results)} files successful")

if __name__ == "__main__":
    main_with_state_management()
