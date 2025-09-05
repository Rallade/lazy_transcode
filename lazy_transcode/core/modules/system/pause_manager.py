"""
Pause/Resume System for lazy-transcode.

Provides graceful interruption handling:
- First Ctrl+C: Graceful pause, wait for current operation to complete
- Second Ctrl+C: Immediate exit

Thread-safe operation with callback support for cleanup operations.
"""

import signal
import threading
import time
from typing import Callable, Optional, List
from lazy_transcode.utils.logging import get_logger

logger = get_logger("pause_manager")

class PauseManager:
    """Manages graceful pause/resume functionality for transcoding operations."""
    
    def __init__(self):
        self._should_pause = threading.Event()
        self._is_paused = threading.Event()
        self._should_exit = threading.Event()
        self._interrupt_count = 0
        self._cleanup_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._original_handler = None
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption."""
        self._original_handler = signal.signal(signal.SIGINT, self._handle_interrupt)
        logger.info("Pause/resume system initialized. Press Ctrl+C once for graceful pause, twice for immediate exit.")
    
    def _handle_interrupt(self, signum, frame):
        """Handle SIGINT (Ctrl+C) interruptions."""
        with self._lock:
            self._interrupt_count += 1
            
            if self._interrupt_count == 1:
                logger.info("Interrupt received. Waiting for current operation to complete...")
                logger.info("Press Ctrl+C again for immediate exit.")
                self._should_pause.set()
                
                # Wait for current operation to pause
                threading.Thread(target=self._wait_for_pause_confirmation, daemon=True).start()
                
            elif self._interrupt_count >= 2:
                logger.warn("Second interrupt received. Performing immediate cleanup and exit...")
                self._should_exit.set()
                self._run_cleanup_callbacks()
                # Restore original handler and re-raise
                signal.signal(signal.SIGINT, self._original_handler)
                raise KeyboardInterrupt("Immediate exit requested")
    
    def _wait_for_pause_confirmation(self):
        """Wait for pause confirmation, then prompt user."""
        # Wait up to 30 seconds for pause confirmation
        if self._is_paused.wait(timeout=30):
            logger.info("Operation paused. Options:")
            logger.info("  - Press Enter to resume")
            logger.info("  - Press Ctrl+C to exit")
            
            try:
                input()  # Wait for user input
                self.resume()
            except KeyboardInterrupt:
                logger.info("Exit requested during pause.")
                self._should_exit.set()
                self._run_cleanup_callbacks()
        else:
            logger.warn("Operation did not pause within timeout. Continuing...")
            self._should_pause.clear()
            self._interrupt_count = 0
    
    def _run_cleanup_callbacks(self):
        """Run all registered cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback to run on exit."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def should_pause(self) -> bool:
        """Check if a pause has been requested."""
        return self._should_pause.is_set()
    
    def should_exit(self) -> bool:
        """Check if an immediate exit has been requested."""
        return self._should_exit.is_set()
    
    def confirm_paused(self):
        """Confirm that the operation has been paused."""
        self._is_paused.set()
        logger.info("Operation successfully paused.")
    
    def resume(self):
        """Resume operations."""
        with self._lock:
            self._should_pause.clear()
            self._is_paused.clear()
            self._interrupt_count = 0
            logger.info("Operations resumed.")
    
    def check_pause_point(self):
        """
        Check if pause is requested and handle it.
        Call this at safe pause points in your operations.
        """
        if self.should_exit():
            raise KeyboardInterrupt("Immediate exit requested")
            
        if self.should_pause():
            self.confirm_paused()
            # Wait for resume or exit
            while self._is_paused.is_set() and not self.should_exit():
                time.sleep(0.1)
            
            if self.should_exit():
                raise KeyboardInterrupt("Exit requested during pause")
    
    def cleanup(self):
        """Cleanup and restore original signal handlers."""
        if self._original_handler:
            signal.signal(signal.SIGINT, self._original_handler)
        self._run_cleanup_callbacks()

# Global instance
_pause_manager: Optional[PauseManager] = None
_pause_manager_lock = threading.Lock()

def get_pause_manager() -> PauseManager:
    """Get the global pause manager instance."""
    global _pause_manager
    with _pause_manager_lock:
        if _pause_manager is None:
            _pause_manager = PauseManager()
        return _pause_manager

def cleanup_pause_manager():
    """Cleanup the global pause manager."""
    global _pause_manager
    with _pause_manager_lock:
        if _pause_manager:
            _pause_manager.cleanup()
            _pause_manager = None
