"""
Enhanced VBR Optimizer with pause/resume and network resilience support.

Extends the core VBR optimizer with:
- Pause/resume capability during optimization
- Network resilience for input/output files
- State persistence for resume after interruption
- Enhanced error recovery and progress monitoring
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from lazy_transcode.core.modules.optimization.vbr_optimizer import (
    optimize_encoder_settings_vbr as _base_optimize_vbr,
    extract_clips_parallel
)
from lazy_transcode.core.modules.system.pause_manager import get_pause_manager
from lazy_transcode.core.modules.system.network_utils import get_network_accessor, safe_file_operation
from lazy_transcode.utils.logging import get_logger

logger = get_logger("vbr_enhanced")

class VBROptimizationState:
    """Manages state persistence for VBR optimization."""
    
    def __init__(self, input_file: Path, state_file: Optional[Path] = None, 
                 target_vmaf: float = 95.0, vmaf_tolerance: float = 1.0,
                 encoder: str = "libx265", encoder_type: str = "cpu"):
        self.input_file = input_file
        self.state_file = state_file or input_file.parent / f".{input_file.stem}_vbr_state.json"
        self.state = {
            'input_file': str(input_file),
            'start_time': time.time(),
            'current_step': 'initialization',
            'target_vmaf': target_vmaf,
            'vmaf_tolerance': vmaf_tolerance,
            'encoder': encoder,
            'encoder_type': encoder_type,
            'bounds': None,
            'clips_extracted': False,
            'clip_positions': None,
            'optimization_progress': {},
            'best_result': None,
            'completed': False
        }
    
    def save_state(self):
        """Save current state to disk."""
        def _save_operation(path: Path):
            with open(path, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"State saved to {path}")
        
        try:
            safe_file_operation(self.state_file, _save_operation)
        except Exception as e:
            logger.warn(f"Failed to save state: {e}")
    
    def load_state(self) -> bool:
        """Load state from disk if it exists."""
        def _load_operation(path: Path) -> bool:
            if not path.exists():
                return False
            
            with open(path, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Loaded previous state from {path}")
            return True
        
        try:
            return safe_file_operation(self.state_file, _load_operation)
        except Exception as e:
            logger.debug(f"Failed to load state: {e}")
            return False
    
    def cleanup_state(self):
        """Remove state file after successful completion."""
        def _cleanup_operation(path: Path):
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up state file: {path}")
        
        try:
            safe_file_operation(self.state_file, _cleanup_operation)
        except Exception as e:
            logger.debug(f"Failed to cleanup state file: {e}")
    
    def _convert_paths_to_strings(self, data):
        """Recursively convert Path objects to strings for JSON serialization."""
        if isinstance(data, Path):
            return str(data)
        elif isinstance(data, dict):
            return {key: self._convert_paths_to_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_paths_to_strings(item) for item in data]
        else:
            return data
    
    def update_step(self, step: str, data: Optional[Dict[str, Any]] = None):
        """Update current step and optional data."""
        self.state['current_step'] = step
        if data:
            # Convert any Path objects to strings before storing
            for key, value in data.items():
                self.state[key] = self._convert_paths_to_strings(value)
        self.save_state()

def optimize_encoder_settings_vbr_enhanced(
    input_file: Path,
    encoder: str,
    encoder_type: str,
    target_vmaf: float = 95.0,
    vmaf_tolerance: float = 1.0,
    clip_positions: Optional[List[int]] = None,
    clip_duration: int = 30,
    resume_state: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced VBR optimization with pause/resume and network resilience.
    
    Args:
        input_file: Path to input video file
        encoder: Encoder to use (e.g., 'hevc_nvenc', 'libx265')
        encoder_type: Type of encoder ('hardware' or 'cpu')
        target_vmaf: Target VMAF score
        vmaf_tolerance: VMAF tolerance for optimization
        clip_positions: Positions for clip extraction (seconds)
        clip_duration: Duration of each clip (seconds)
        resume_state: Whether to resume from previous state
        **kwargs: Additional arguments passed to base optimizer
        
    Returns:
        Dictionary with optimization results
    """
    pause_manager = get_pause_manager()
    network_accessor = get_network_accessor()
    
    # Register cleanup callback
    def cleanup_callback():
        logger.info("Enhanced VBR optimization interrupted - state preserved for resume")
    
    pause_manager.register_cleanup_callback(cleanup_callback)
    
    # Initialize state management
    state_manager = VBROptimizationState(
        input_file, target_vmaf=target_vmaf, vmaf_tolerance=vmaf_tolerance,
        encoder=encoder, encoder_type=encoder_type
    )
    
    # Try to resume from previous state
    if resume_state and state_manager.load_state():
        logger.info("Resuming VBR optimization from previous state...")
        logger.info(f"Previous step: {state_manager.state['current_step']}")
        
        # Check if already completed
        if state_manager.state.get('completed'):
            logger.info("Optimization already completed!")
            result = state_manager.state.get('best_result', {})
            state_manager.cleanup_state()
            return result
    
    try:
        # Register input file for network resilience
        network_accessor.register_network_path(input_file)
        
        # Step 1: Verify input file accessibility
        pause_manager.check_pause_point()
        state_manager.update_step('verifying_input')
        
        def verify_input(path: Path):
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            return path.stat().st_size
        
        file_size = safe_file_operation(input_file, verify_input)
        logger.info(f"Input file verified: {input_file} ({file_size / (1024**3):.2f} GB)")
        
        # Step 2: Extract clips (if not already done)
        pause_manager.check_pause_point()
        if not state_manager.state.get('clips_extracted'):
            state_manager.update_step('extracting_clips')
            
            if clip_positions is None:
                # Use default positions if not specified
                clip_positions = [300, 900, 1500]  # 5min, 15min, 25min
            
            logger.info("Extracting analysis clips...")
            clips_result = extract_clips_parallel(
                input_file, clip_positions, clip_duration
            )
            
            state_manager.update_step('clips_extracted', {
                'clips_extracted': True,
                'clip_positions': clip_positions,
                'clips_result': [str(clip) for clip in clips_result] if clips_result else []
            })
        else:
            logger.info("Using previously extracted clips")
            clip_positions = state_manager.state['clip_positions']
        
        # Step 3: Run core VBR optimization with pause points  
        # (Skip bounds calculation for now - base optimizer handles this)
        pause_manager.check_pause_point()
        state_manager.update_step('optimizing')
        
        logger.info("Starting enhanced VBR optimization...")
        
        # Create a wrapped version of the base optimizer that checks pause points
        def pause_aware_optimize():
            # This is where we'd need to modify the base optimizer to support pause points
            # For now, we'll call the base optimizer and check pause points before/after
            pause_manager.check_pause_point()
            
            # Ensure clip_positions is not None
            if clip_positions is None:
                raise ValueError("clip_positions cannot be None for VBR optimization")
            
            result = _base_optimize_vbr(
                infile=input_file,
                encoder=encoder,
                encoder_type=encoder_type,
                target_vmaf=target_vmaf,
                vmaf_tolerance=vmaf_tolerance,
                clip_positions=clip_positions,
                clip_duration=clip_duration,
                **kwargs
            )
            
            pause_manager.check_pause_point()
            return result
        
        optimization_result = pause_aware_optimize()
        
        # Step 5: Save final result
        state_manager.update_step('completed', {
            'best_result': optimization_result,
            'completed': True,
            'end_time': time.time()
        })
        
        logger.info("Enhanced VBR optimization completed successfully!")
        
        # Cleanup state file on success
        state_manager.cleanup_state()
        
        return optimization_result
        
    except KeyboardInterrupt:
        logger.info("VBR optimization paused/interrupted - state saved for resume")
        raise
    except Exception as e:
        logger.error(f"Enhanced VBR optimization failed: {e}")
        # Keep state file for debugging
        raise

def resume_vbr_optimization(input_file: Path) -> Optional[Dict[str, Any]]:
    """
    Resume a previously interrupted VBR optimization.
    
    Args:
        input_file: Path to the input file that was being optimized
        
    Returns:
        Optimization result if resumable state exists, None otherwise
    """
    state_manager = VBROptimizationState(input_file)
    
    if not state_manager.load_state():
        logger.info("No resumable VBR optimization state found")
        return None
    
    logger.info(f"Found resumable optimization state from step: {state_manager.state['current_step']}")
    
    # Extract parameters from saved state
    saved_state = state_manager.state
    return optimize_encoder_settings_vbr_enhanced(
        input_file=input_file,
        encoder=saved_state.get('encoder', 'libx265'),
        encoder_type=saved_state.get('encoder_type', 'cpu'),
        target_vmaf=saved_state.get('target_vmaf', 95.0),
        vmaf_tolerance=saved_state.get('vmaf_tolerance', 1.0),
        resume_state=True
    )

def list_resumable_optimizations(directory: Path) -> List[Dict[str, Any]]:
    """
    List all resumable VBR optimizations in a directory.
    
    Args:
        directory: Directory to search for state files
        
    Returns:
        List of resumable optimization information
    """
    resumable = []
    
    try:
        for state_file in directory.glob(".*_vbr_state.json"):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                resumable.append({
                    'input_file': state.get('input_file'),
                    'state_file': str(state_file),
                    'current_step': state.get('current_step'),
                    'start_time': state.get('start_time'),
                    'completed': state.get('completed', False)
                })
            except Exception as e:
                logger.debug(f"Failed to read state file {state_file}: {e}")
    
    except Exception as e:
        logger.debug(f"Failed to search for state files: {e}")
    
    return resumable
