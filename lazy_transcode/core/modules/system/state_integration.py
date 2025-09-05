"""
Integration helpers for universal state management.

Provides decorator and wrapper functions to easily add state management 
to existing transcoding workflows without major refactoring.
"""

from functools import wraps
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional
import inspect

from .transcode_state_manager import (
    UniversalStateManager, TranscodeOperation, 
    find_resumable_operations, resume_operation_from_state
)
from lazy_transcode.utils.logging import get_logger

logger = get_logger("state_integration")

def with_state_management(operation_type: str, resume_check: bool = True):
    """
    Decorator to add state management to transcoding functions.
    
    Args:
        operation_type: Type of operation ('vbr', 'qp', 'auto', 'batch')
        resume_check: Whether to check for resumable operations
        
    Usage:
        @with_state_management('vbr')
        def process_vbr_mode(args, encoder, encoder_type, files):
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract parameters from function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Get key parameters for state management
            args_obj = bound_args.arguments.get('args')
            files = bound_args.arguments.get('files', [])
            encoder = bound_args.arguments.get('encoder')
            encoder_type = bound_args.arguments.get('encoder_type')
            
            # If processing single file, add state management
            if len(files) == 1:
                return _process_single_file_with_state(
                    func, operation_type, files[0], args_obj, 
                    encoder, encoder_type, bound_args.arguments
                )
            
            # For multiple files, add state management per file
            else:
                return _process_multiple_files_with_state(
                    func, operation_type, files, args_obj,
                    encoder, encoder_type, bound_args.arguments
                )
        
        return wrapper
    return decorator

def _process_single_file_with_state(func: Callable, operation_type: str, 
                                   input_file: Path, args_obj: Any,
                                   encoder: Optional[str], encoder_type: Optional[str],
                                   all_args: Dict[str, Any]) -> Any:
    """Process a single file with state management."""
    
    # Create operation description
    operation = _create_operation_from_args(
        operation_type, input_file, args_obj, encoder, encoder_type
    )
    
    # Check for resumable operation
    resumable_ops = find_resumable_operations(input_file.parent, operation_type)
    resumable = [op for op in resumable_ops 
                if op['input_file'] == str(input_file) and not op['completed']]
    
    state_manager = None
    
    if resumable:
        logger.info(f"Found resumable {operation_type} operation for {input_file.name}")
        state_file = Path(resumable[0]['state_file'])
        state_manager = resume_operation_from_state(state_file)
        
        if state_manager and state_manager.state.completed:
            logger.info("Operation already completed, returning cached result")
            return state_manager.state.best_result
    
    if not state_manager:
        state_manager = UniversalStateManager(operation)
    
    try:
        # Add state manager to function arguments
        enhanced_args = all_args.copy()
        enhanced_args['state_manager'] = state_manager
        
        # Call original function with state manager
        result = func(**enhanced_args)
        
        # Mark as completed
        state_manager.mark_completed(success=True, result=result)
        state_manager.cleanup_state()
        
        return result
        
    except Exception as e:
        state_manager.mark_completed(success=False, error_message=str(e))
        logger.error(f"Operation failed: {e}")
        raise

def _process_multiple_files_with_state(func: Callable, operation_type: str,
                                      files: List[Path], args_obj: Any,
                                      encoder: Optional[str], encoder_type: Optional[str],
                                      all_args: Dict[str, Any]) -> Any:
    """Process multiple files with batch state management."""
    
    # Create batch operation
    operation = TranscodeOperation(
        operation_type='batch',
        input_file=str(files[0].parent),  # Use parent directory
        encoder=encoder,
        encoder_type=encoder_type,
        extra_args={
            'file_count': len(files),
            'file_list': [str(f) for f in files],
            'sub_operation_type': operation_type
        }
    )
    
    state_manager = UniversalStateManager(operation)
    
    try:
        state_manager.update_step('processing_batch', progress=0.0)
        
        # Process files individually but track overall progress
        results = {}
        total_files = len(files)
        
        for i, file in enumerate(files):
            state_manager.check_pause_point()
            
            # Process single file (without nested state management)
            file_args = all_args.copy()
            file_args['files'] = [file]
            
            logger.info(f"Processing file {i+1}/{total_files}: {file.name}")
            file_result = func(**file_args)
            results[file] = file_result
            
            # Update progress
            progress = ((i + 1) / total_files) * 100
            state_manager.update_step('processing_batch', progress=progress)
        
        state_manager.mark_completed(success=True, result=results)
        state_manager.cleanup_state()
        
        return results
        
    except Exception as e:
        state_manager.mark_completed(success=False, error_message=str(e))
        logger.error(f"Batch operation failed: {e}")
        raise

def _create_operation_from_args(operation_type: str, input_file: Path, 
                               args_obj: Any, encoder: Optional[str], encoder_type: Optional[str]) -> TranscodeOperation:
    """Create TranscodeOperation from function arguments."""
    
    # Extract common parameters from args object
    extra_args = {}
    if hasattr(args_obj, '__dict__'):
        extra_args = {k: v for k, v in args_obj.__dict__.items() 
                     if not k.startswith('_')}
    
    return TranscodeOperation(
        operation_type=operation_type,
        input_file=str(input_file),
        encoder=encoder,
        encoder_type=encoder_type,
        
        # VBR specific
        target_vmaf=getattr(args_obj, 'vmaf_target', None),
        vmaf_tolerance=getattr(args_obj, 'vmaf_tol', None),
        clip_duration=30,  # Default
        
        # QP specific 
        target_qp=getattr(args_obj, 'qp', None),
        quality_threshold=getattr(args_obj, 'quality_threshold', None),
        
        # Common options
        preserve_hdr=getattr(args_obj, 'preserve_hdr', None),
        non_destructive=getattr(args_obj, 'non_destructive', None),
        dry_run=getattr(args_obj, 'dry_run', None),
        
        extra_args=extra_args
    )

# Context manager for step-by-step state management
class StateAwareOperation:
    """Context manager for step-by-step state management in complex operations."""
    
    def __init__(self, operation: TranscodeOperation, resume: bool = True):
        self.operation = operation
        self.resume = resume
        self.state_manager: Optional[UniversalStateManager] = None
    
    def __enter__(self) -> UniversalStateManager:
        # Check for resumable state
        if self.resume:
            input_path = Path(self.operation.input_file)
            resumable_ops = find_resumable_operations(
                input_path.parent if input_path.is_file() else input_path,
                self.operation.operation_type
            )
            
            resumable = [op for op in resumable_ops 
                        if op['input_file'] == self.operation.input_file and not op['completed']]
            
            if resumable:
                state_file = Path(resumable[0]['state_file'])
                self.state_manager = resume_operation_from_state(state_file)
        
        if not self.state_manager:
            self.state_manager = UniversalStateManager(self.operation)
        
        return self.state_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.state_manager:
            if exc_type is None:
                # Success
                self.state_manager.mark_completed(success=True)
                self.state_manager.cleanup_state()
            else:
                # Failure - keep state for debugging
                error_msg = str(exc_val) if exc_val else "Unknown error"
                self.state_manager.mark_completed(success=False, error_message=error_msg)

# Utility functions for CLI integration

def list_resumable_cli(directory: Path, operation_type: Optional[str] = None):
    """List resumable operations for CLI display."""
    resumable = find_resumable_operations(directory, operation_type)
    
    if not resumable:
        print(f"No resumable operations found in {directory}")
        return
    
    print(f"Found {len(resumable)} resumable operation(s):")
    print()
    
    for i, op in enumerate(resumable, 1):
        status = "COMPLETED" if op['completed'] else f"{op['progress_percent']:.1f}%"
        duration = ""
        if op['start_time']:
            import time
            elapsed = time.time() - op['start_time']
            duration = f" ({elapsed/3600:.1f}h elapsed)"
        
        print(f"{i}. {op['operation_type'].upper()}: {Path(op['input_file']).name}")
        print(f"   Step: {op['current_step']} | Progress: {status}{duration}")
        print(f"   File: {op['state_file']}")
        print()

def resume_operation_cli(directory: Path, operation_index: int = 0) -> bool:
    """Resume an operation by index for CLI usage."""
    resumable = find_resumable_operations(directory)
    
    if not resumable:
        print("No resumable operations found")
        return False
    
    if operation_index >= len(resumable):
        print(f"Invalid operation index. Found {len(resumable)} operations")
        return False
    
    selected = resumable[operation_index]
    state_file = Path(selected['state_file'])
    
    print(f"Resuming {selected['operation_type']} operation: {Path(selected['input_file']).name}")
    
    state_manager = resume_operation_from_state(state_file)
    if state_manager:
        print(f"Resumed from step: {state_manager.state.current_step}")
        return True
    else:
        print("Failed to resume operation")
        return False
