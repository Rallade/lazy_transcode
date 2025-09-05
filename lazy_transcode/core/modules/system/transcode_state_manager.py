"""
Universal State Management System for lazy-transcode.

Provides state persistence and resume capability for any transcoding operation:
- VBR optimization
- QP optimization  
- Batch processing
- Individual file transcoding

Features:
- Operation-agnostic state storage
- Parameter serialization/deserialization
- Resume detection and continuation
- Integration with pause manager
- Network-resilient file operations
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict, fields
import threading

from .pause_manager import get_pause_manager
from .network_utils import get_network_accessor, safe_file_operation
from lazy_transcode.utils.logging import get_logger

logger = get_logger("state_manager")

@dataclass
class TranscodeOperation:
    """Represents a transcoding operation with all necessary parameters."""
    operation_type: str  # 'vbr', 'qp', 'auto', 'batch'
    input_file: str
    output_file: Optional[str] = None
    
    # Encoder settings
    encoder: Optional[str] = None
    encoder_type: Optional[str] = None
    preset: Optional[str] = None
    
    # VBR specific
    target_vmaf: Optional[float] = None
    vmaf_tolerance: Optional[float] = None
    clip_positions: Optional[List[int]] = None
    clip_duration: Optional[int] = None
    
    # QP specific
    target_qp: Optional[int] = None
    quality_threshold: Optional[float] = None
    
    # Common options
    preserve_hdr: Optional[bool] = None
    non_destructive: Optional[bool] = None
    dry_run: Optional[bool] = None
    
    # Additional args (for extensibility)
    extra_args: Optional[Dict[str, Any]] = None

@dataclass  
class TranscodeState:
    """Complete state information for a transcoding operation."""
    operation: TranscodeOperation
    
    # State tracking
    operation_id: str
    start_time: float
    current_step: str = 'initialization'
    steps_completed: Optional[List[str]] = None
    progress_percent: float = 0.0
    
    # Results and intermediate data
    clips_extracted: bool = False
    optimization_results: Optional[Dict[str, Any]] = None
    best_result: Optional[Dict[str, Any]] = None
    
    # Completion status
    completed: bool = False
    success: bool = False
    error_message: Optional[str] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []

class UniversalStateManager:
    """Universal state management for any transcoding operation."""
    
    def __init__(self, operation: TranscodeOperation, 
                 state_file: Optional[Path] = None,
                 auto_save: bool = True):
        """
        Initialize state manager for an operation.
        
        Args:
            operation: The transcoding operation to manage
            state_file: Custom state file path (auto-generated if None)
            auto_save: Automatically save state on updates
        """
        self.operation = operation
        self.auto_save = auto_save
        self._lock = threading.Lock()
        
        # Generate operation ID and state file path
        input_path = Path(operation.input_file)
        operation_hash = f"{operation.operation_type}_{int(time.time())}"
        self.operation_id = f"{input_path.stem}_{operation_hash}"
        
        self.state_file = state_file or self._generate_state_file_path(input_path)
        
        # Initialize state
        self.state = TranscodeState(
            operation=operation,
            operation_id=self.operation_id,
            start_time=time.time()
        )
        
        # Register with pause manager for cleanup
        pause_manager = get_pause_manager()
        pause_manager.register_cleanup_callback(self._emergency_save)
        
        logger.info(f"Initialized state manager for {operation.operation_type} operation")
        if self.auto_save:
            self.save_state()
    
    def _generate_state_file_path(self, input_path: Path) -> Path:
        """Generate a state file path for the operation."""
        state_dir = input_path.parent / '.lazy_transcode_state'
        state_dir.mkdir(exist_ok=True)
        
        # Include operation type in filename for clarity
        filename = f"{input_path.stem}_{self.operation.operation_type}.state.json"
        return state_dir / filename
    
    def save_state(self):
        """Save current state to disk."""
        def _save_operation(path: Path):
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert state to JSON-serializable format
            state_dict = self._state_to_dict()
            
            with open(path, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            logger.debug(f"State saved to {path}")
        
        try:
            with self._lock:
                safe_file_operation(self.state_file, _save_operation)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self) -> bool:
        """Load state from disk if it exists."""
        def _load_operation(path: Path) -> bool:
            if not path.exists():
                return False
            
            with open(path, 'r') as f:
                state_dict = json.load(f)
            
            self._state_from_dict(state_dict)
            logger.info(f"Loaded state from {path} - Step: {self.state.current_step}")
            return True
        
        try:
            with self._lock:
                return safe_file_operation(self.state_file, _load_operation)
        except Exception as e:
            logger.debug(f"Failed to load state: {e}")
            return False
    
    def _state_to_dict(self) -> Dict[str, Any]:
        """Convert state to JSON-serializable dictionary."""
        return {
            'operation': asdict(self.state.operation),
            'operation_id': self.state.operation_id,
            'start_time': self.state.start_time,
            'current_step': self.state.current_step,
            'steps_completed': self.state.steps_completed,
            'progress_percent': self.state.progress_percent,
            'clips_extracted': self.state.clips_extracted,
            'optimization_results': self.state.optimization_results,
            'best_result': self.state.best_result,
            'completed': self.state.completed,
            'success': self.state.success,
            'error_message': self.state.error_message,
            'end_time': self.state.end_time,
            'saved_at': time.time()
        }
    
    def _state_from_dict(self, data: Dict[str, Any]):
        """Restore state from dictionary."""
        # Reconstruct operation
        operation_data = data['operation']
        operation = TranscodeOperation(**operation_data)
        
        # Reconstruct state
        self.state = TranscodeState(
            operation=operation,
            operation_id=data['operation_id'],
            start_time=data['start_time'],
            current_step=data['current_step'],
            steps_completed=data.get('steps_completed', []),
            progress_percent=data.get('progress_percent', 0.0),
            clips_extracted=data.get('clips_extracted', False),
            optimization_results=data.get('optimization_results'),
            best_result=data.get('best_result'),
            completed=data.get('completed', False),
            success=data.get('success', False),
            error_message=data.get('error_message'),
            end_time=data.get('end_time')
        )
    
    def update_step(self, step: str, progress: Optional[float] = None, 
                   data: Optional[Dict[str, Any]] = None):
        """Update current step and optional progress/data."""
        with self._lock:
            # Ensure steps_completed is initialized
            if self.state.steps_completed is None:
                self.state.steps_completed = []
                
            # Mark previous step as completed if different
            if step != self.state.current_step and self.state.current_step not in self.state.steps_completed:
                self.state.steps_completed.append(self.state.current_step)
            
            self.state.current_step = step
            
            if progress is not None:
                self.state.progress_percent = progress
            
            if data:
                # Update state with provided data
                for key, value in data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
            
            if self.auto_save:
                self.save_state()
        
        logger.info(f"Step: {step}" + (f" ({progress:.1f}%)" if progress else ""))
    
    def mark_completed(self, success: bool = True, result: Optional[Dict[str, Any]] = None,
                      error_message: Optional[str] = None):
        """Mark operation as completed."""
        with self._lock:
            self.state.completed = True
            self.state.success = success
            self.state.end_time = time.time()
            
            if result:
                self.state.best_result = result
            
            if error_message:
                self.state.error_message = error_message
            
            if self.auto_save:
                self.save_state()
        
        duration = self.state.end_time - self.state.start_time
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Operation {status} in {duration:.1f}s")
    
    def check_pause_point(self):
        """Check for pause requests - integrates with pause manager."""
        pause_manager = get_pause_manager()
        pause_manager.check_pause_point()
    
    def cleanup_state(self):
        """Remove state file after successful completion."""
        def _cleanup_operation(path: Path):
            if path.exists():
                path.unlink()
                # Remove directory if empty
                try:
                    path.parent.rmdir()
                except OSError:
                    pass  # Directory not empty
                logger.debug(f"Cleaned up state file: {path}")
        
        try:
            safe_file_operation(self.state_file, _cleanup_operation)
        except Exception as e:
            logger.debug(f"Failed to cleanup state file: {e}")
    
    def _emergency_save(self):
        """Emergency save on interruption - disable auto-save to prevent recursion."""
        original_auto_save = self.auto_save
        self.auto_save = False
        try:
            self.save_state()
            logger.info("Emergency state save completed")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")
        finally:
            self.auto_save = original_auto_save

# Utility functions for state management integration

def find_resumable_operations(directory: Path, 
                            operation_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find all resumable operations in a directory.
    
    Args:
        directory: Directory to search
        operation_type: Filter by operation type ('vbr', 'qp', 'batch', etc.)
        
    Returns:
        List of resumable operation information
    """
    resumable = []
    state_dir = directory / '.lazy_transcode_state'
    
    if not state_dir.exists():
        return resumable
    
    try:
        for state_file in state_dir.glob("*.state.json"):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Filter by operation type if specified
                op_type = data.get('operation', {}).get('operation_type')
                if operation_type and op_type != operation_type:
                    continue
                
                resumable.append({
                    'state_file': str(state_file),
                    'operation_type': op_type,
                    'input_file': data.get('operation', {}).get('input_file'),
                    'current_step': data.get('current_step'),
                    'progress_percent': data.get('progress_percent', 0),
                    'start_time': data.get('start_time'),
                    'completed': data.get('completed', False),
                    'operation_id': data.get('operation_id')
                })
                
            except Exception as e:
                logger.debug(f"Failed to read state file {state_file}: {e}")
    
    except Exception as e:
        logger.debug(f"Failed to search for state files: {e}")
    
    return resumable

def resume_operation_from_state(state_file: Path) -> Optional[UniversalStateManager]:
    """
    Resume an operation from a state file.
    
    Args:
        state_file: Path to the state file
        
    Returns:
        Initialized state manager if successful, None otherwise
    """
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct operation
        operation_data = data['operation']
        operation = TranscodeOperation(**operation_data)
        
        # Create state manager and load state
        state_manager = UniversalStateManager(operation, state_file=state_file)
        if state_manager.load_state():
            logger.info(f"Successfully resumed operation: {state_manager.state.operation_id}")
            return state_manager
            
    except Exception as e:
        logger.error(f"Failed to resume operation from {state_file}: {e}")
    
    return None

def cleanup_completed_states(directory: Path, max_age_days: int = 7):
    """
    Clean up old completed state files.
    
    Args:
        directory: Directory to clean
        max_age_days: Remove completed states older than this many days
    """
    state_dir = directory / '.lazy_transcode_state'
    if not state_dir.exists():
        return
    
    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    removed_count = 0
    
    try:
        for state_file in state_dir.glob("*.state.json"):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Remove if completed and old
                if (data.get('completed', False) and 
                    data.get('saved_at', 0) < cutoff_time):
                    state_file.unlink()
                    removed_count += 1
                    
            except Exception as e:
                logger.debug(f"Failed to process state file {state_file}: {e}")
        
        # Remove directory if empty
        try:
            state_dir.rmdir()
        except OSError:
            pass  # Directory not empty
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old state files")
            
    except Exception as e:
        logger.debug(f"Failed to cleanup state files: {e}")
