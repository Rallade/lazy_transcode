# Universal State Management for lazy-transcode

## Overview

The universal state management system provides pause/resume and persistence capabilities for **any** transcoding operation, not just VBR optimization. It's designed to be operation-agnostic and easy to integrate with existing code.

## Key Features

### 🎯 **Operation-Agnostic Design**
- Works with VBR, QP, batch processing, or any custom operation
- Standardized state format across all operation types
- Extensible for future operation types

### 💾 **Persistent State Storage**
- JSON-based state files stored in `.lazy_transcode_state/` directories
- Network-resilient file operations
- Automatic cleanup of completed states
- Resume capability across application restarts

### ⏸️ **Pause/Resume Integration**
- Integrates with existing pause manager
- Graceful interruption handling
- Progress tracking and step management
- Emergency state saving on interruption

### 🔧 **Easy Integration**
- Decorator-based integration for existing functions
- Context manager for manual state management
- Minimal code changes required
- Drop-in replacement capability

## Core Components

### 1. `TranscodeOperation` - Operation Description
```python
@dataclass
class TranscodeOperation:
    operation_type: str          # 'vbr', 'qp', 'auto', 'batch'
    input_file: str
    encoder: Optional[str] = None
    encoder_type: Optional[str] = None
    target_vmaf: Optional[float] = None
    target_qp: Optional[int] = None
    # ... all relevant parameters
```

### 2. `UniversalStateManager` - State Management
```python
class UniversalStateManager:
    def __init__(self, operation: TranscodeOperation)
    def save_state(self)
    def load_state(self) -> bool
    def update_step(self, step: str, progress: float = None)
    def mark_completed(self, success: bool, result: Dict = None)
    def check_pause_point(self)
```

### 3. State File Structure
```json
{
  "operation": {
    "operation_type": "vbr",
    "input_file": "/path/to/video.mkv",
    "encoder": "hevc_nvenc",
    "target_vmaf": 92.0
  },
  "current_step": "vbr_optimization",
  "steps_completed": ["file_validation", "clip_extraction"],
  "progress_percent": 75.0,
  "clips_extracted": true,
  "optimization_results": {...},
  "best_result": {...},
  "completed": false,
  "start_time": 1693824000.0
}
```

## Integration Methods

### Method 1: Decorator Integration (Automatic)
```python
from lazy_transcode.core.modules.system.state_integration import with_state_management

@with_state_management('vbr', resume_check=True)
def process_vbr_mode(args, encoder, encoder_type, files, state_manager=None):
    # Original function logic
    # state_manager is automatically injected
    
    if state_manager:
        state_manager.update_step('processing_files')
        state_manager.check_pause_point()
    
    return results
```

### Method 2: Context Manager (Manual Control)
```python
from lazy_transcode.core.modules.system.state_integration import StateAwareOperation

def my_transcoding_function(input_file, encoder):
    operation = TranscodeOperation(
        operation_type='custom',
        input_file=str(input_file),
        encoder=encoder
    )
    
    with StateAwareOperation(operation, resume=True) as state_manager:
        
        # Check if resuming
        if state_manager.state.current_step != 'initialization':
            logger.info(f"Resuming from: {state_manager.state.current_step}")
        
        # Step-by-step processing with state updates
        state_manager.update_step('file_analysis', progress=25.0)
        # ... do work ...
        
        state_manager.update_step('optimization', progress=75.0)
        state_manager.check_pause_point()  # Pause point
        # ... do work ...
        
        state_manager.update_step('complete', progress=100.0)
        return result
        
    # Automatic completion handling on exit
```

### Method 3: Direct State Manager Usage
```python
from lazy_transcode.core.modules.system.transcode_state_manager import UniversalStateManager

def my_function(input_file):
    operation = TranscodeOperation(
        operation_type='custom',
        input_file=str(input_file)
    )
    
    state_manager = UniversalStateManager(operation)
    
    try:
        # Check for resumable state
        if state_manager.load_state() and state_manager.state.completed:
            return state_manager.state.best_result
        
        # Process with state updates
        state_manager.update_step('working')
        result = do_work()
        
        state_manager.mark_completed(success=True, result=result)
        state_manager.cleanup_state()
        return result
        
    except Exception as e:
        state_manager.mark_completed(success=False, error_message=str(e))
        raise
```

## CLI Integration

### Add State Management Arguments
```python
def add_state_management_args(parser):
    state_group = parser.add_argument_group('State Management')
    state_group.add_argument('--resume', action='store_true')
    state_group.add_argument('--list-resumable', action='store_true')
    state_group.add_argument('--cleanup-old-states', type=int, metavar='DAYS')
```

### Handle State Management Commands
```python
from lazy_transcode.core.modules.system.state_integration import list_resumable_cli

def handle_state_cli(args, input_path):
    if args.list_resumable:
        list_resumable_cli(input_path)
        return True
    
    if args.cleanup_old_states:
        cleanup_completed_states(input_path, args.cleanup_old_states)
        return True
    
    return False
```

## State File Management

### File Locations
- **State Directory**: `.lazy_transcode_state/` in the same directory as input files
- **File Naming**: `{input_filename}_{operation_type}.state.json`
- **Automatic Cleanup**: Completed states removed after successful completion

### Finding Resumable Operations
```python
from lazy_transcode.core.modules.system.transcode_state_manager import find_resumable_operations

# Find all resumable operations
resumable = find_resumable_operations(Path("/videos"))

# Filter by operation type
vbr_resumable = find_resumable_operations(Path("/videos"), operation_type='vbr')

# Get operation details
for op in resumable:
    print(f"Operation: {op['operation_type']}")
    print(f"File: {op['input_file']}")
    print(f"Step: {op['current_step']}")
    print(f"Progress: {op['progress_percent']:.1f}%")
```

### Manual Resume
```python
from lazy_transcode.core.modules.system.transcode_state_manager import resume_operation_from_state

# Resume from specific state file
state_file = Path(".lazy_transcode_state/video_vbr.state.json")
state_manager = resume_operation_from_state(state_file)

if state_manager:
    print(f"Resumed: {state_manager.state.current_step}")
    # Continue processing...
```

## Benefits

### For Users
- **Interruption Recovery**: Never lose progress from long-running operations
- **Pause/Resume**: Graceful interruption with Ctrl+C
- **Progress Visibility**: Clear progress tracking and step information
- **Network Resilience**: Automatic recovery from network drive issues

### For Developers
- **Easy Integration**: Minimal code changes to add state management
- **Operation Agnostic**: Works with any transcoding operation
- **Extensible**: Easy to add new operation types and parameters
- **Consistent**: Standardized state format across all operations

## Migration Path

### Current Functions
Keep existing functions unchanged for backward compatibility:
```python
# Original function still works
def process_vbr_mode(args, encoder, encoder_type, files):
    # Original implementation
    pass
```

### Enhanced Versions
Create enhanced versions with state management:
```python
# Enhanced version with state management
@with_state_management('vbr')
def process_vbr_mode_enhanced(args, encoder, encoder_type, files, state_manager=None):
    # Call original function with added state management
    return process_vbr_mode(args, encoder, encoder_type, files)
```

### Gradual Adoption
- Start with high-value operations (VBR optimization, batch processing)
- Add state management to new features by default
- Gradually enhance existing functions as needed
- Maintain backward compatibility throughout

## Examples

See the `examples/` directory for complete working examples:

- `simple_state_example.py` - Basic state management concepts
- `state_management_integration.py` - Integration patterns
- `main_with_state_management.py` - Enhanced main function

The universal state management system makes lazy-transcode more robust and user-friendly while being easy to integrate with existing code.
