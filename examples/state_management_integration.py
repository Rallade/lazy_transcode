"""
Example integration of universal state management with existing transcoding functions.

This shows how to integrate state management with minimal changes to existing code.
"""

from pathlib import Path
from typing import Dict, List, Any

from lazy_transcode.core.modules.system.state_integration import (
    with_state_management, StateAwareOperation
)
from lazy_transcode.core.modules.system.transcode_state_manager import TranscodeOperation
from lazy_transcode.utils.logging import get_logger

logger = get_logger("state_example")

# Example 1: Using decorator for automatic state management
@with_state_management('vbr', resume_check=True)
def process_vbr_mode_with_state(args, encoder: str, encoder_type: str, 
                               files: List[Path], state_manager=None) -> Dict[Path, Dict]:
    """VBR processing with automatic state management."""
    
    # If state_manager is provided, we can use it for progress tracking
    if state_manager:
        state_manager.update_step('starting_vbr_processing')
    
    results = {}
    
    for i, file in enumerate(files):
        if state_manager:
            state_manager.check_pause_point()  # Check for pause requests
            progress = (i / len(files)) * 100
            state_manager.update_step(f'processing_file_{i+1}', progress=progress)
        
        logger.info(f"Processing {file.name}")
        
        # Simulate VBR optimization work
        # In real implementation, this would call the actual VBR optimizer
        result = {
            'success': True,
            'bitrate': 2500,
            'vmaf_score': 93.2,
            'preset': 'medium'
        }
        
        results[file] = result
        
        if state_manager:
            # Save intermediate results
            state_manager.state.optimization_results = results
    
    return results

# Example 2: Using context manager for manual state management
def process_qp_mode_manual_state(args, encoder: str, encoder_type: str, 
                                files: List[Path]) -> Dict[Path, int]:
    """QP processing with manual state management."""
    
    # Create operation description
    operation = TranscodeOperation(
        operation_type='qp',
        input_file=str(files[0]) if files else '',
        encoder=encoder,
        encoder_type=encoder_type,
        target_qp=getattr(args, 'qp', None)
    )
    
    # Use context manager for automatic state handling
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Check if we can resume from a previous step
        if state_manager.state.current_step != 'initialization':
            logger.info(f"Resuming from step: {state_manager.state.current_step}")
            
            # Restore any intermediate results
            if state_manager.state.optimization_results:
                logger.info("Found previous optimization results")
        
        state_manager.update_step('qp_optimization_start')
        
        results = {}
        
        for i, file in enumerate(files):
            state_manager.check_pause_point()
            
            state_manager.update_step(f'analyzing_file_{i+1}', 
                                    progress=(i / len(files)) * 50)
            
            logger.info(f"Finding optimal QP for {file.name}")
            
            # Simulate QP optimization
            optimal_qp = 23  # Would be calculated by real optimizer
            results[file] = optimal_qp
            
            # Update state with current results
            state_manager.state.optimization_results = results
            
            state_manager.update_step(f'transcoding_file_{i+1}',
                                    progress=50 + (i / len(files)) * 50)
            
            logger.info(f"Transcoding {file.name} with QP {optimal_qp}")
            
            # Simulate transcoding work here
        
        state_manager.update_step('qp_optimization_complete', progress=100.0)
        
        # Set final result
        state_manager.state.best_result = results
        
        return results

# Example 3: Integration with existing main.py functions
def integrate_with_existing_main():
    """Example of how to integrate with existing main.py functions."""
    
    # Instead of modifying the original functions, we can wrap them
    from lazy_transcode.core.main import process_vbr_mode, process_qp_mode
    
    # Create state-aware versions
    state_aware_vbr = with_state_management('vbr')(process_vbr_mode)
    state_aware_qp = with_state_management('qp')(process_qp_mode)
    
    # These can now be used as drop-in replacements that automatically
    # handle state management, pause/resume, and progress tracking
    
    return state_aware_vbr, state_aware_qp

# Example 4: Batch processing with state management
def process_batch_with_state(root_directory: Path, operation_type: str = 'auto'):
    """Process multiple directories with batch state management."""
    
    operation = TranscodeOperation(
        operation_type='batch',
        input_file=str(root_directory),
        extra_args={
            'sub_operation_type': operation_type,
            'recursive': True
        }
    )
    
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Discover all video files
        state_manager.update_step('discovering_files')
        
        video_files = []
        for ext in ['.mkv', '.mp4', '.avi']:
            video_files.extend(root_directory.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Process each file
        results = {}
        
        for i, file in enumerate(video_files):
            state_manager.check_pause_point()
            
            progress = (i / len(video_files)) * 100
            state_manager.update_step(f'processing_{file.name}', progress=progress)
            
            try:
                # Process individual file (this would call actual transcoding)
                logger.info(f"Processing {file.name}")
                
                # Simulate processing
                result = {'success': True, 'output_size': '1.2GB'}
                results[file] = result
                
                # Save intermediate results
                state_manager.state.optimization_results = results
                
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {e}")
                results[file] = {'success': False, 'error': str(e)}
        
        state_manager.update_step('batch_complete', progress=100.0)
        return results

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Mock args object
    class MockArgs:
        vmaf_target = 92.0
        vmaf_tol = 1.0
        qp = 23
        preserve_hdr = True
        non_destructive = True
        dry_run = False
    
    args = MockArgs()
    test_files = [Path("test_video.mkv")]
    
    # Test VBR with state management
    logger.info("Testing VBR with state management...")
    vbr_results = process_vbr_mode_with_state(
        args, "hevc_nvenc", "hardware", test_files
    )
    
    # Test QP with manual state management  
    logger.info("Testing QP with manual state management...")
    qp_results = process_qp_mode_manual_state(
        args, "libx265", "cpu", test_files
    )
