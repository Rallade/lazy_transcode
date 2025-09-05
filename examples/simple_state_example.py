"""
Simple, working example of universal state management integration.

This shows the key concepts without the complexity of full integration.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from lazy_transcode.core.modules.system.transcode_state_manager import (
    UniversalStateManager, TranscodeOperation
)
from lazy_transcode.core.modules.system.state_integration import StateAwareOperation
from lazy_transcode.utils.logging import get_logger

logger = get_logger("simple_state_example")

def simple_vbr_with_state(input_file: Path, encoder: str = "hevc_nvenc", 
                         target_vmaf: float = 92.0) -> Dict[str, Any]:
    """Simple VBR optimization with state management."""
    
    # Create operation description
    operation = TranscodeOperation(
        operation_type='vbr',
        input_file=str(input_file),
        encoder=encoder,
        encoder_type='hardware',
        target_vmaf=target_vmaf,
        vmaf_tolerance=1.0
    )
    
    # Use context manager for automatic state handling
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Step 1: Check if resuming
        if state_manager.state.current_step != 'initialization':
            logger.info(f"Resuming from: {state_manager.state.current_step}")
            
            if state_manager.state.completed:
                logger.info("Already completed, returning cached result")
                return state_manager.state.best_result or {'success': True, 'cached': True}
        
        # Step 2: File validation
        state_manager.update_step('file_validation', progress=20.0)
        logger.info(f"Validating {input_file.name}")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        time.sleep(1)  # Simulate work
        
        # Step 3: Clip extraction
        state_manager.update_step('clip_extraction', progress=40.0)
        logger.info("Extracting analysis clips")
        
        clip_positions = [300, 900, 1500]  # Mock clip positions
        state_manager.update_step('clips_extracted', data={
            'clips_extracted': True,
            'clip_positions': clip_positions
        })
        
        time.sleep(2)  # Simulate clip extraction
        
        # Step 4: VBR optimization
        state_manager.update_step('vbr_optimization', progress=70.0)
        logger.info("Running VBR optimization")
        
        # Simulate optimization work
        for i in range(5):
            state_manager.check_pause_point()  # Check for pause requests
            logger.info(f"VBR trial {i+1}/5")
            time.sleep(1)
        
        # Step 5: Generate result
        state_manager.update_step('generating_result', progress=90.0)
        
        result = {
            'success': True,
            'bitrate': 2500,
            'vmaf_score': 93.2,
            'preset': 'medium',
            'optimization_time': time.time() - state_manager.state.start_time,
            'clips_used': clip_positions
        }
        
        state_manager.update_step('complete', progress=100.0)
        state_manager.state.best_result = result
        
        logger.info(f"VBR complete: {result['bitrate']}kbps @ VMAF {result['vmaf_score']}")
        
        return result

def demonstrate_batch_processing():
    """Demonstrate batch processing with state management."""
    
    # Create some mock video files for demonstration
    test_files = [
        Path("video1.mkv"),
        Path("video2.mkv"), 
        Path("video3.mkv")
    ]
    
    # Create batch operation
    operation = TranscodeOperation(
        operation_type='batch',
        input_file=str(Path.cwd()),
        encoder='hevc_nvenc',
        encoder_type='hardware',
        extra_args={
            'file_count': len(test_files),
            'file_list': [str(f) for f in test_files]
        }
    )
    
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Restore any previous results
        results = state_manager.state.optimization_results or {}
        processed_files = set(results.keys()) if results else set()
        
        logger.info(f"Processing {len(test_files)} files")
        if processed_files:
            logger.info(f"Resuming: {len(processed_files)} files already processed")
        
        for i, file in enumerate(test_files):
            # Skip if already processed
            if str(file) in processed_files:
                logger.info(f"Skipping {file.name} (already processed)")
                continue
            
            state_manager.check_pause_point()
            
            progress = (i / len(test_files)) * 100
            state_manager.update_step(f'processing_{file.name}', progress=progress)
            
            logger.info(f"Processing {file.name}")
            
            try:
                # Mock processing
                file_result = {
                    'success': True,
                    'bitrate': 2000 + i * 100,
                    'vmaf_score': 92.0 + i * 0.5
                }
                
                results[str(file)] = file_result
                
                # Save intermediate results
                state_manager.state.optimization_results = results
                
                time.sleep(1)  # Simulate processing time
                
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {e}")
                results[str(file)] = {'success': False, 'error': str(e)}
        
        state_manager.update_step('batch_complete', progress=100.0)
        logger.info("Batch processing complete")
        
        return results

def demonstrate_state_management_cli():
    """Demonstrate CLI state management features."""
    
    from lazy_transcode.core.modules.system.state_integration import (
        list_resumable_cli, find_resumable_operations
    )
    from lazy_transcode.core.modules.system.transcode_state_manager import (
        cleanup_completed_states
    )
    
    current_dir = Path.cwd()
    
    print("=== State Management CLI Demo ===\n")
    
    # List resumable operations
    print("1. Listing resumable operations:")
    list_resumable_cli(current_dir)
    
    # Find resumable programmatically
    resumable = find_resumable_operations(current_dir)
    print(f"Found {len(resumable)} resumable operations")
    
    # Cleanup old states (dry run)
    print("\\n2. Cleaning up old completed states...")
    try:
        cleanup_completed_states(current_dir, max_age_days=7)
        print("Cleanup completed")
    except Exception as e:
        print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    # Example 1: Simple VBR with state management
    print("=== Simple VBR with State Management ===")
    
    test_file = Path("test_video.mkv")
    try:
        result = simple_vbr_with_state(test_file)
        print(f"Result: {result}")
        print()
    except Exception as e:
        print(f"VBR failed: {e}")
        print()
    
    # Example 2: Batch processing with state management
    print("=== Batch Processing with State Management ===")
    try:
        batch_results = demonstrate_batch_processing()
        successful = sum(1 for r in batch_results.values() if r.get('success', False))
        print(f"Batch complete: {successful}/{len(batch_results)} successful")
        print()
    except Exception as e:
        print(f"Batch failed: {e}")
        print()
    
    # Example 3: CLI state management
    demonstrate_state_management_cli()
