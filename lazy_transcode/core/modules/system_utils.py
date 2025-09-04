"""
System utilities for lazy_transcode.

This module provides system-level utilities including:
- CPU monitoring
- File system operations
- Process management
- Cleanup functions
"""

import os
import sys
import shlex
import signal
import atexit
import subprocess
import threading
import time
import tempfile
import contextlib
from pathlib import Path

from ...utils.logging import get_logger

logger = get_logger("system_utils")
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Global state
"""TEMP_FILES is expected by tests to behave like a list supporting append/remove and indexing.
Internally we still want uniqueness for cleanup, so we store both an ordered list and a set membership guard.
For simplicity we expose a list and keep semantics minimal (tests only rely on list ops)."""
class _TempFilesList(list):
    _membership: set[str]
    def append(self, item):  # type: ignore[override]
        if isinstance(item, Path):
            key = str(item)
        else:
            key = str(item)
        if key not in self._membership:
            self._membership.add(key)
            super().append(item)

    def remove(self, item):  # type: ignore[override]
        key = str(item)
        if key in self._membership:
            self._membership.remove(key)
        try:
            super().remove(item)
        except ValueError:
            pass

    def discard(self, item):  # convenience used internally
        key = str(item)
        if key in self._membership:
            self._membership.remove(key)
        # remove first matching path/str
        for i, existing in enumerate(list(self)):
            if str(existing) == key:
                del self[i]
                break

    # Provide set-like add used by existing code paths
    def add(self, item):  # type: ignore[override]
        self.append(item)

TEMP_FILES = _TempFilesList()
TEMP_FILES._membership = set()  # type: ignore[attr-defined]
DEBUG = False  # Set by --debug flag


def file_exists(path: "str | os.PathLike[str] | Path") -> bool:
    """Thin existence wrapper to provide a stable patch point for tests.

    Semantics: identical to Path(path).exists() with broad exception safety.
    Returns False on any unexpected OSError to avoid propagating transient FS issues
    during test simulations (patched behaviors). Keep intentionally minimal.
    """
    try:
        return Path(path).exists()
    except Exception:
        return False


def _cleanup():
    """Cleanup temporary files on exit"""
    for f in list(TEMP_FILES):
        try:
            path_str = str(f)
            if file_exists(path_str):
                os.remove(path_str)
                if hasattr(logger, 'cleanup'):
                    try:
                        logger.cleanup(f"removed {path_str}")  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            TEMP_FILES.discard(f)
    try:
        if hasattr(TEMP_FILES, 'clear'):
            TEMP_FILES.clear()  # type: ignore
    except Exception:
        pass


# Register cleanup on exit
atexit.register(_cleanup)
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGABRT):
    signal.signal(sig, lambda s, f: sys.exit(1))


def run_logged(cmd: list[str], **popen_kwargs) -> subprocess.CompletedProcess:
    """Run command with logging"""
    logger.debug("Command: " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, **popen_kwargs)


def format_size(bytes_size: int) -> str:
    """Convert bytes to human readable format matching test expectations:
    - Bytes: integer no decimal ("500 B", "0 B")
    - >= KB: two decimals ("1.50 KB", "2.00 MB")
    """
    negative = bytes_size < 0
    size = float(abs(bytes_size))
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    while unit_index < len(units) - 1 and size >= 1024.0:
        size /= 1024.0
        unit_index += 1
    unit = units[unit_index]
    if unit == 'B':
        formatted = f"{int(size)} {unit}"
    else:
        formatted = f"{size:.2f} {unit}"
    return f"-{formatted}" if negative else formatted

def cleanup_temp_files():
    """Public cleanup function expected by tests (wrapper around _cleanup)."""
    _cleanup()


def run_command(cmd: list[str], timeout: int = 30, capture_output: bool = True,
                text: bool = True, check: bool = False) -> subprocess.CompletedProcess:
    """
    Standardized subprocess command runner with consistent error handling.
    
    Args:
        cmd: Command as list of strings
        timeout: Timeout in seconds (default: 30)
        capture_output: Whether to capture stdout/stderr (default: True)  
        text: Whether to use text mode (default: True)
        check: Whether to raise exception on non-zero exit (default: False)
    
    Returns:
        CompletedProcess object
    """
    try:
        return subprocess.run(
            cmd, 
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s: {' '.join(cmd[:3])}...")
        raise
    except subprocess.CalledProcessError as e:
        if DEBUG:
            logger.error(f"Command failed: {' '.join(cmd[:3])}... (exit code: {e.returncode})")
        raise
    except Exception as e:
        logger.error(f"Unexpected error running command: {e}")
        raise


@contextlib.contextmanager
def temporary_file(suffix: str = ".tmp", prefix: str = "lazy_transcode_"):
    """
    Context manager for temporary files with automatic cleanup.
    
    Ensures temp files are tracked in TEMP_FILES and cleaned up properly.
    
    Args:
        suffix: File extension (default: .tmp)
        prefix: Filename prefix (default: lazy_transcode_)
        
    Yields:
        Path: Path to the temporary file
    """
    temp_file = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # Close the file descriptor, keep the path
        temp_file = Path(temp_path)
        
        # Track in global temp files
        TEMP_FILES.add(str(temp_file))
        
        yield temp_file
        
    finally:
        if temp_file:
            try:
                if file_exists(temp_file):
                    temp_file.unlink()
            except Exception as e:
                if DEBUG:
                    logger.error(f"Failed to cleanup temp file {temp_file}: {e}")
            finally:
                TEMP_FILES.discard(str(temp_file))


def get_next_transcoded_dir(base_path: Path) -> Path:
    """
    Get the next available transcoded directory name.
    
    If 'Transcoded' doesn't exist, returns 'Transcoded'.
    If it exists, returns 'Transcoded_2', 'Transcoded_3', etc.
    """
    transcoded_base = base_path / "Transcoded"
    
    if not file_exists(transcoded_base):
        return transcoded_base
    
    # Find the next available number
    counter = 2
    while True:
        candidate = base_path / f"Transcoded_{counter}"
        if not file_exists(candidate):
            return candidate
        counter += 1


def start_cpu_monitor(duration_seconds: int = 0) -> tuple[threading.Thread, threading.Event]:
    """Start CPU monitoring in background. Returns (thread, stop_event)"""
    stop_event = threading.Event()
    
    def monitor_cpu():
        try:
            if PSUTIL_AVAILABLE:
                logger.debug(f"Starting CPU monitoring (cores: {psutil.cpu_count()})")
                start_time = time.time()
                while not stop_event.is_set():
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
                    active_cores = sum(1 for c in cpu_per_core if c > 10)
                    logger.debug(f"CPU: {cpu_percent:5.1f}% total, {active_cores}/16 cores active (>10%)")
                    if duration_seconds > 0 and time.time() - start_time > duration_seconds:
                        break
            else:
                logger.debug("psutil not available, using basic Windows monitoring")
                start_time = time.time()
                while not stop_event.is_set():
                    try:
                        # Windows-specific CPU monitoring using PowerShell
                        result = subprocess.run(['powershell', '-Command', 
                            'Get-Counter "\\Processor(_Total)\\% Processor Time" -SampleInterval 1 -MaxSamples 1 | Select -ExpandProperty CounterSamples | Select -ExpandProperty CookedValue'], 
                            capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            cpu_usage = float(result.stdout.strip())
                            logger.debug(f"CPU: {cpu_usage:5.1f}% total")
                        time.sleep(2)
                    except Exception:
                        time.sleep(3)
                        if DEBUG:
                            logger.debug("Basic monitoring active")
                    
                    if duration_seconds > 0 and time.time() - start_time > duration_seconds:
                        break
        except Exception as e:
            if DEBUG:
                logger.debug(f"CPU monitoring error: {e}")
    
    thread = threading.Thread(target=monitor_cpu, daemon=True)
    thread.start()
    return thread, stop_event
