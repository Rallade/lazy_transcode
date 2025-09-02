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
from pathlib import Path

from ...utils.logging import get_logger

logger = get_logger("system_utils")
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Global state
TEMP_FILES: set[str] = set()
DEBUG = False  # Set by --debug flag


def _cleanup():
    """Cleanup temporary files on exit"""
    for f in list(TEMP_FILES):
        try:
            if os.path.exists(f):
                os.remove(f)
                logger.cleanup(f"removed {f}")
        except Exception:
            pass
        finally:
            TEMP_FILES.discard(f)


# Register cleanup on exit
atexit.register(_cleanup)
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGABRT):
    signal.signal(sig, lambda s, f: sys.exit(1))


def run_logged(cmd: list[str], **popen_kwargs) -> subprocess.CompletedProcess:
    """Run command with logging"""
    logger.debug("Command: " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, **popen_kwargs)


def format_size(bytes_size: int) -> str:
    """Convert bytes to human readable format"""
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_next_transcoded_dir(base_path: Path) -> Path:
    """
    Get the next available transcoded directory name.
    
    If 'Transcoded' doesn't exist, returns 'Transcoded'.
    If it exists, returns 'Transcoded_2', 'Transcoded_3', etc.
    """
    transcoded_base = base_path / "Transcoded"
    
    if not transcoded_base.exists():
        return transcoded_base
    
    # Find the next available number
    counter = 2
    while True:
        candidate = base_path / f"Transcoded_{counter}"
        if not candidate.exists():
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
