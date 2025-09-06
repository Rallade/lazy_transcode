"""
Smart Logging System for lazy_transcode

Provides intelligent logging with different verbosity levels:
- ESSENTIAL: Only critical progress and results (default)
- NORMAL: Key operations and important intermediate results  
- VERBOSE: Detailed technical information
- DEBUG: Everything including internal calculations

Key principles:
1. User-centric: Focus on what users need to know
2. Progressive disclosure: More detail with higher verbosity
3. Context-aware: Different levels for different operations
4. Time-aware: Show progress and ETAs where possible
"""

from enum import Enum
from typing import Optional
import sys


class LogLevel(Enum):
    """Smart logging levels with progressive disclosure"""
    ESSENTIAL = 1    # Only progress, results, and errors
    NORMAL = 2       # + key operations and trial details  
    VERBOSE = 3      # + technical details and bounds calculations
    DEBUG = 4        # Everything including internal calculations


class SmartLogger:
    """Intelligent logging with context-aware verbosity"""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.ESSENTIAL, show_timestamps: bool = True):
        self.name = name
        self.level = level
        self.show_timestamps = show_timestamps
        self._current_phase = None
        self._trial_count = 0
        self._total_trials = 0
        
    def set_level(self, level: LogLevel):
        """Change logging level"""
        self.level = level
        
    def _should_log(self, required_level: LogLevel) -> bool:
        """Check if message should be logged at current level"""
        return self.level.value >= required_level.value
        
    def _format_message(self, tag: str, message: str) -> str:
        """Format message with consistent styling"""
        if self.show_timestamps:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            return f"[{timestamp}] [{tag}] {message}"
        return f"[{tag}] {message}"
        
    def _log(self, tag: str, message: str, level: LogLevel = LogLevel.ESSENTIAL, end: str = "\n"):
        """Internal logging method"""
        if self._should_log(level):
            print(self._format_message(tag, message), end=end, flush=True)
            
    # === PROGRESS TRACKING ===
    def progress(self, message: str):
        """Essential progress updates (always shown)"""
        self._log("PROGRESS", message, LogLevel.ESSENTIAL)
        
    def phase_start(self, phase_name: str):
        """Start a new processing phase"""
        self._current_phase = phase_name
        self.progress(f"Phase: {phase_name}")
        
    def phase_complete(self, phase_name: str, result: Optional[str] = None):
        """Complete current processing phase"""
        if result:
            self._log("COMPLETE", f"{phase_name}: {result}", LogLevel.ESSENTIAL)
        else:
            self._log("COMPLETE", phase_name, LogLevel.ESSENTIAL)
        self._current_phase = None
        
    # === RESULTS ===
    def success(self, message: str):
        """Success/completion messages"""
        self._log("SUCCESS", message, LogLevel.ESSENTIAL)
        
    def warning(self, message: str):
        """Warning messages"""
        self._log("WARNING", message, LogLevel.ESSENTIAL)
        
    def error(self, message: str):
        """Error messages"""
        self._log("ERROR", message, LogLevel.ESSENTIAL)
        
    # === VBR-SPECIFIC METHODS ===
    def vbr_start(self, filename: str, target_vmaf: float):
        """Start VBR optimization"""
        self.progress(f"Optimizing {filename} (VMAF target: {target_vmaf})")
        
    def vbr_bounds(self, min_bitrate: int, max_bitrate: int, method: Optional[str] = None):
        """Report bitrate bounds calculation"""
        if method:
            self._log("DETAIL", f"Bounds ({method}): {min_bitrate}-{max_bitrate}kbps", LogLevel.VERBOSE)
        else:
            self._log("DETAIL", f"Bitrate range: {min_bitrate}-{max_bitrate}kbps", LogLevel.NORMAL)
            
    def vbr_trial_start(self, total_trials: int):
        """Start VBR trial phase"""
        self._trial_count = 0
        self._total_trials = total_trials
        self.phase_start("Testing bitrates")
        
    def vbr_trial(self, bitrate: int, vmaf_score: Optional[float] = None, iteration: Optional[int] = None):
        """Report VBR trial result"""
        self._trial_count += 1
        
        # Essential: Progress with percentage
        if self._total_trials > 0:
            progress_pct = int((self._trial_count / self._total_trials) * 100)
            progress_msg = f"Testing bitrates... (trial {self._trial_count}/{self._total_trials}) ({progress_pct}%)"
            self._log("PROGRESS", progress_msg, LogLevel.ESSENTIAL, end="\r")
            
        # Normal: Show individual trial details
        if vmaf_score is not None:
            detail = f"Trial {self._trial_count}: {bitrate}kbps → VMAF {vmaf_score:.1f}"
            self._log("TRIAL", detail, LogLevel.NORMAL)
        else:
            detail = f"Trial {self._trial_count}: Testing {bitrate}kbps"
            self._log("TRIAL", detail, LogLevel.VERBOSE)
            
    def vbr_result(self, bitrate: int, vmaf_score: float, tolerance: Optional[float] = None):
        """Report final VBR optimization result"""
        print()  # Clear progress line
        if tolerance:
            result = f"Found optimal: {bitrate}kbps → VMAF {vmaf_score:.1f} (±{tolerance})"
        else:
            result = f"Found optimal: {bitrate}kbps → VMAF {vmaf_score:.1f}"
        self.success(result)
        
    def vbr_failure(self, reason: str):
        """Report VBR optimization failure"""
        print()  # Clear progress line
        self.error(f"VBR optimization failed: {reason}")
        
    # === BATCH PROCESSING ===
    def batch_start(self, file_count: int):
        """Start batch processing"""
        self.progress(f"Processing {file_count} files")
        
    def batch_file(self, filename: str, index: int, total: int):
        """Report batch file processing"""
        progress = f"File {index}/{total}: {filename}"
        self.progress(progress)
        
    def batch_complete(self, processed: int, skipped: int = 0):
        """Complete batch processing"""
        if skipped > 0:
            result = f"Processed {processed} files ({skipped} skipped)"
        else:
            result = f"Processed {processed} files"
        self.success(result)
        
    # === TECHNICAL DETAILS ===
    def detail(self, message: str):
        """Technical details (verbose level)"""
        self._log("DETAIL", message, LogLevel.VERBOSE)
        
    def debug(self, message: str):
        """Debug information (debug level)"""
        self._log("DEBUG", message, LogLevel.DEBUG)
        
    def technical(self, category: str, message: str):
        """Technical information with category"""
        self._log(f"{category.upper()}", message, LogLevel.VERBOSE)


# === GLOBAL LOGGER MANAGEMENT ===
_global_logger = None


def init_smart_logging(level: LogLevel = LogLevel.ESSENTIAL, show_timestamps: bool = True) -> SmartLogger:
    """Initialize global smart logger"""
    global _global_logger
    _global_logger = SmartLogger("lazy-transcode", level, show_timestamps)
    return _global_logger


def get_smart_logger() -> SmartLogger:
    """Get global smart logger (initialize if needed)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SmartLogger("lazy-transcode")
    return _global_logger


def set_log_level_from_args(args) -> LogLevel:
    """Convert CLI args to LogLevel"""
    if hasattr(args, 'debug') and args.debug:
        return LogLevel.DEBUG
    elif hasattr(args, 'very_verbose') and args.very_verbose:
        return LogLevel.VERBOSE  
    elif hasattr(args, 'verbose') and args.verbose:
        return LogLevel.NORMAL
    elif hasattr(args, 'quiet') and args.quiet:
        return LogLevel.ESSENTIAL
    else:
        return LogLevel.ESSENTIAL  # Default


# === BACKWARD COMPATIBILITY ===
def create_legacy_logger_adapter(smart_logger: SmartLogger):
    """Create adapter for existing logging.py interface"""
    class LegacyAdapter:
        def __init__(self, logger: SmartLogger):
            self.logger = logger
            
        def vbr(self, message: str):
            """Legacy VBR logging"""
            self.logger._log("VBR", message, LogLevel.NORMAL)
            
        def vbr_debug(self, message: str):
            """Legacy VBR debug logging"""
            self.logger.debug(f"VBR: {message}")
            
        def vbr_bisect(self, message: str):
            """Legacy VBR bisect logging"""
            self.logger._log("VBR-BISECT", message, LogLevel.VERBOSE)
            
        def info(self, message: str):
            """Legacy info logging"""
            self.logger._log("INFO", message, LogLevel.NORMAL)
            
        def debug(self, message: str):
            """Legacy debug logging"""
            self.logger.debug(message)
            
        def error(self, message: str):
            """Legacy error logging"""
            self.logger.error(message)
            
        def warning(self, message: str):
            """Legacy warning logging"""
            self.logger.warning(message)
            
    return LegacyAdapter(smart_logger)


# === DEMONSTRATION ===
if __name__ == "__main__":
    """Demo different verbosity levels"""
    
    print("=== ESSENTIAL Level (Default) ===")
    logger = SmartLogger("demo", LogLevel.ESSENTIAL)
    logger.vbr_start("Demon Slayer S01E01.mkv", 95.0)
    logger.vbr_trial_start(8)
    for i in range(8):
        logger.vbr_trial(3000 - i*200, 94.0 + i*0.2)
    logger.vbr_result(2800, 95.1, 1.0)
    
    print("\n=== NORMAL Level ===")
    logger.set_level(LogLevel.NORMAL)
    logger.vbr_start("Demon Slayer S01E01.mkv", 95.0)
    logger.vbr_bounds(2000, 8000)
    logger.vbr_trial_start(4)
    for i in range(4):
        logger.vbr_trial(4000 - i*500, 93.0 + i*0.7)
    logger.vbr_result(2500, 95.2)
    
    print("\n=== VERBOSE Level ===")
    logger.set_level(LogLevel.VERBOSE)
    logger.vbr_start("Demon Slayer S01E01.mkv", 95.0)
    logger.vbr_bounds(2000, 8000, "research-based")
    logger.technical("bounds", "Resolution: 1920×1080 (2,073,600 pixels) → FHD profile")
    logger.vbr_trial_start(3)
    for i in range(3):
        logger.vbr_trial(3500 - i*300)
    logger.vbr_result(2900, 95.0)
