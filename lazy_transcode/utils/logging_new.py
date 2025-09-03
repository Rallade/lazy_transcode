"""
Centralized logging utilities for lazy_transcode

Provides consistent logging patterns with configurable debug levels:
- [INFO] for general information
- [WARN] for warnings
- [ERROR] for errors  
- [RESULT] for final results
- [DEBUG] for debug information
- [VBR] for VBR optimization messages
- [VMAF] for VMAF calculation messages
- [CLEANUP] for cleanup operations
- And many domain-specific loggers

Usage:
    from lazy_transcode.utils.logging import get_logger, set_debug_mode
    
    # Configure logging globally
    set_debug_mode(True)  # Enable debug messages
    
    # Get a logger for your module
    logger = get_logger("vbr_optimizer")
    logger.info("This is an info message")
    logger.debug("This is a debug message")  # Only shows if debug enabled
    logger.vbr("VBR optimization message")
    logger.vmaf("VMAF calculation message")
"""

import sys
import os
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

# Import tqdm with fallback
try:
    from tqdm import tqdm
except ImportError:  # fallback minimal stub
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, position=None, leave=True, unit=None, unit_scale=False):
            self.iterable = iterable or range(total or 0)
            self.desc = desc or ""
            self.total = total
            self._n = 0
            
        def __iter__(self):
            for x in self.iterable:
                yield x
                
        def update(self, n=1):
            self._n += n
            if self.total:
                print(f"\r{self.desc}: {self._n}/{self.total}", end="", flush=True)
            
        def close(self):
            if self.total:
                print()  # New line after progress
                
        def set_postfix(self, **kw):
            pass
            
        def set_description(self, desc):
            self.desc = desc
            
        def write(self, s):
            print(s)
            
        def __enter__(self):
            return self
            
        def __exit__(self, *exc):
            self.close()
            return False

# Global logging configuration
_DEBUG_ENABLED = False
_QUIET_MODE = False
_LOG_LEVEL = "INFO"

class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3

# Initialize debug mode from environment or module state
def _init_debug_mode():
    global _DEBUG_ENABLED
    # Check for DEBUG environment variable
    if os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes'):
        _DEBUG_ENABLED = True
    
    # Check if DEBUG is set in any imported module
    try:
        from lazy_transcode.core.modules.vbr_optimizer import DEBUG
        _DEBUG_ENABLED = DEBUG
    except (ImportError, AttributeError):
        pass

_init_debug_mode()

def set_debug_mode(enabled: bool):
    """Enable or disable debug mode globally"""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled

def set_quiet_mode(enabled: bool):
    """Enable or disable quiet mode (suppress INFO and DEBUG messages)"""
    global _QUIET_MODE
    _QUIET_MODE = enabled

def set_log_level(level: str):
    """Set the global log level: DEBUG, INFO, WARN, ERROR"""
    global _LOG_LEVEL
    _LOG_LEVEL = level.upper()

def get_debug_mode() -> bool:
    """Get current debug mode setting"""
    return _DEBUG_ENABLED

class Logger:
    """Centralized logger with consistent formatting and configurable output"""
    
    def __init__(self, module_name: str = ""):
        self.module_name = module_name
        self.prefix = f"[{module_name}] " if module_name else ""
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on current settings"""
        if _QUIET_MODE and level in (LogLevel.DEBUG, LogLevel.INFO):
            return False
        
        level_hierarchy = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO, 
            "WARN": LogLevel.WARN,
            "ERROR": LogLevel.ERROR
        }
        
        current_level = level_hierarchy.get(_LOG_LEVEL, LogLevel.INFO)
        return level.value >= current_level.value
    
    def _log(self, level: str, message: str, prefix: str = ""):
        """Internal logging function"""
        log_level = LogLevel.INFO
        if level == "DEBUG":
            log_level = LogLevel.DEBUG
        elif level == "WARN":
            log_level = LogLevel.WARN
        elif level == "ERROR":
            log_level = LogLevel.ERROR
            
        if not self._should_log(log_level):
            return
            
        full_prefix = f"{self.prefix}{prefix}" if prefix else self.prefix
        print(f"[{level}] {full_prefix}{message}")
    
    def debug(self, message: str):
        """Log debug message (only if debug mode enabled)"""
        if _DEBUG_ENABLED:
            self._log("DEBUG", message)
    
    def info(self, message: str):
        """Log informational message"""
        self._log("INFO", message)
    
    def warn(self, message: str):
        """Log warning message"""
        self._log("WARN", message)
    
    def error(self, message: str):
        """Log error message"""
        self._log("ERROR", message)
    
    def result(self, message: str):
        """Log result message"""
        if self._should_log(LogLevel.INFO):
            print(f"[RESULT] {self.prefix}{message}")
    
    # Domain-specific logging methods
    def vbr(self, message: str):
        """Log VBR optimization message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR] {message}")
    
    def vbr_debug(self, message: str):
        """Log VBR debug message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[VBR-DEBUG] {message}")
    
    def vbr_bisect(self, message: str):
        """Log VBR bisection message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-BISECT] {message}")
    
    def vbr_bounds(self, message: str):
        """Log VBR bounds calculation message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-BOUNDS] {message}")
    
    def vbr_test(self, message: str):
        """Log VBR test message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-TEST] {message}")
    
    def vbr_cache(self, message: str):
        """Log VBR cache message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-CACHE] {message}")
    
    def vbr_early_exit(self, message: str):
        """Log VBR early exit message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-EARLY-EXIT] {message}")
    
    def vbr_abandon(self, message: str):
        """Log VBR abandonment message"""
        if self._should_log(LogLevel.WARN):
            print(f"[VBR-ABANDON] {message}")
    
    def vbr_skip(self, message: str):
        """Log VBR skip message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-SKIP] {message}")
    
    def vbr_error(self, message: str):
        """Log VBR error message"""
        if self._should_log(LogLevel.ERROR):
            print(f"[VBR-ERROR] {message}")
    
    def vbr_success(self, message: str):
        """Log VBR success message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VBR-SUCCESS] {message}")
    
    def vbr_failed(self, message: str):
        """Log VBR failure message"""
        if self._should_log(LogLevel.WARN):
            print(f"[VBR-FAILED] {message}")
    
    def vmaf(self, message: str):
        """Log VMAF calculation message"""
        if self._should_log(LogLevel.INFO):
            print(f"[VMAF] {message}")
    
    def vmaf_debug(self, message: str):
        """Log VMAF debug message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[VMAF-DEBUG] {message}")
    
    def vmaf_cmd(self, message: str):
        """Log VMAF command message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[VMAF-CMD] {message}")
    
    def vmaf_threads(self, message: str):
        """Log VMAF threading message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[VMAF-THREADS] {message}")
    
    def cleanup(self, message: str):
        """Log cleanup operation"""
        if self._should_log(LogLevel.INFO):
            print(f"[CLEANUP] {message}")
    
    def discovery(self, message: str):
        """Log file discovery message"""
        if self._should_log(LogLevel.INFO):
            print(f"[DISCOVERY] {message}")
    
    def encoder(self, message: str):
        """Log encoder setup message"""
        if self._should_log(LogLevel.INFO):
            print(f"[ENCODER] {message}")
    
    def output(self, message: str):
        """Log output directory message"""
        if self._should_log(LogLevel.INFO):
            print(f"[OUTPUT] {message}")
    
    def mode(self, message: str):
        """Log mode selection message"""
        if self._should_log(LogLevel.INFO):
            print(f"[MODE] {message}")
    
    def hdr(self, message: str):
        """Log HDR processing message"""
        if self._should_log(LogLevel.INFO):
            print(f"[HDR] {message}")
    
    def cmd(self, message: str):
        """Log command execution message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[CMD] {message}")
    
    def sample_extract(self, message: str):
        """Log sample extraction message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[SAMPLE-EXTRACT] {message}")
    
    def sample_encode(self, message: str):
        """Log sample encoding message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[SAMPLE-ENCODE] {message}")
    
    def vbr_encode(self, message: str):
        """Log VBR encoding message"""
        if _DEBUG_ENABLED and self._should_log(LogLevel.DEBUG):
            print(f"[VBR-ENCODE] {message}")

def get_logger(module_name: str = "") -> Logger:
    """Get a logger instance for a module"""
    return Logger(module_name)

# Maintain backward compatibility with existing functions
def log_info(message: str):
    """Log an informational message"""
    if not _QUIET_MODE:
        print(f"[INFO] {message}")

def log_warn(message: str):
    """Log a warning message"""
    print(f"[WARN] {message}")

def log_error(message: str):
    """Log an error message"""
    print(f"[ERROR] {message}")

def log_result(message: str):
    """Log a result message"""
    if not _QUIET_MODE:
        print(f"[RESULT] {message}")

def log_gradient_descent(message: str):
    """Log a gradient descent specific message"""
    if not _QUIET_MODE:
        print(f"[GRADIENT-DESCENT] {message}")

def log_cleanup(message: str):
    """Log a cleanup operation"""
    if not _QUIET_MODE:
        print(f"[CLEANUP] {message}")

def log_step(step_number: int, message: str):
    """Log a step in a multi-step process"""
    if not _QUIET_MODE:
        print(f"[STEP {step_number}] {message}")

def log_setup(message: str):
    """Log a setup operation"""
    if not _QUIET_MODE:
        print(f"[SETUP] {message}")

def log_progress(message: str):
    """Log a progress update"""
    if not _QUIET_MODE:
        print(f"[PROGRESS] {message}")

def create_progress_bar(total: Optional[int] = None, desc: str = "", unit: str = "it", 
                       position: Optional[int] = None, leave: bool = True):
    """Create a progress bar with consistent styling"""
    return tqdm(total=total, desc=desc, unit=unit, position=position, leave=leave)

def print_section_header(title: str, width: int = 90):
    """Print a section header with consistent formatting"""
    print("=" * width)
    print(title)
    print("=" * width)

def print_separator(width: int = 90):
    """Print a separator line"""
    print("-" * width)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string"""
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"
