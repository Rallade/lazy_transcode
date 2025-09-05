"""
Network Resilience System for lazy-transcode.

Handles network drive issues, reconnections, and recovery from computer sleep scenarios.
Provides automatic detection and exponential backoff retry logic.
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import subprocess
import platform
from lazy_transcode.utils.logging import get_logger

logger = get_logger("network_utils")

class NetworkAccessor:
    """Manages network resilience for file operations."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._network_drives: Dict[str, Dict[str, Any]] = {}
        self._retry_delays = [1, 2, 5, 10, 30, 60]  # Exponential backoff delays
        self._is_windows = platform.system() == "Windows"
    
    def register_network_path(self, path: Path, mount_info: Optional[Dict[str, str]] = None):
        """
        Register a network path for monitoring.
        
        Args:
            path: Path to monitor
            mount_info: Optional mount information (UNC path, credentials, etc.)
        """
        # Use original path first, only resolve if needed
        original_path = str(path)
        resolved_path = str(path.resolve())
        
        # Prefer the original path if it's accessible
        if self._test_path_access(path):
            path_str = original_path
            effective_path = path
        else:
            path_str = resolved_path  
            effective_path = path.resolve()
        
        with self._lock:
            self._network_drives[path_str] = {
                'path': effective_path,
                'original_path': path,
                'resolved_path': path.resolve(),
                'mount_info': mount_info or {},
                'last_access': time.time(),
                'failure_count': 0,
                'is_network': self._detect_network_drive(effective_path)
            }
        
        if self._network_drives[path_str]['is_network']:
            logger.debug(f"Registered network path: {effective_path}")
        else:
            logger.debug(f"Registered local path: {effective_path}")
    
    def _detect_network_drive(self, path: Path) -> bool:
        """Detect if a path is on a network drive."""
        try:
            path_str = str(path.resolve())
            
            if self._is_windows:
                # Check for UNC paths
                if path_str.startswith('\\\\'):
                    return True
                
                # Check for mapped network drives
                drive_letter = path_str[0:2] if len(path_str) >= 2 else ''
                if drive_letter.endswith(':'):
                    try:
                        result = subprocess.run(['net', 'use', drive_letter], 
                                              capture_output=True, text=True, timeout=5)
                        return result.returncode == 0 and 'Microsoft Windows Network' in result.stdout
                    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                        pass
            else:
                # Unix/Linux: Check for NFS, SMB mounts
                try:
                    result = subprocess.run(['df', '-T', str(path)], 
                                          capture_output=True, text=True, timeout=5)
                    return any(fs_type in result.stdout.lower() 
                             for fs_type in ['nfs', 'cifs', 'smb'])
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    pass
                    
        except Exception as e:
            logger.debug(f"Network detection failed for {path}: {e}")
        
        return False
    
    def _test_path_access(self, path: Path) -> bool:
        """Test if a path is accessible."""
        try:
            return path.exists() and (path.is_file() or path.is_dir())
        except (OSError, PermissionError):
            return False
    
    def _reconnect_network_drive(self, path_info: Dict[str, Any]) -> bool:
        """Attempt to reconnect a network drive."""
        path = path_info['path']
        mount_info = path_info['mount_info']
        
        logger.info(f"Attempting to reconnect network path: {path}")
        
        if self._is_windows and mount_info.get('unc_path'):
            try:
                # Try to reconnect using net use
                cmd = ['net', 'use', str(path), mount_info['unc_path']]
                if mount_info.get('username'):
                    cmd.extend(['/user:' + mount_info['username']])
                if mount_info.get('password'):
                    cmd.append(mount_info['password'])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info(f"Successfully reconnected: {path}")
                    return True
                else:
                    logger.debug(f"Reconnection failed: {result.stderr}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                logger.debug(f"Reconnection command failed: {e}")
        
        # Generic reconnection attempt: try to access the path
        return self._test_path_access(path)
    
    def safe_access(self, path: Path, operation: Callable[[Path], Any], 
                   max_retries: Optional[int] = None) -> Any:
        """
        Safely access a path with automatic retry and reconnection.
        
        Args:
            path: Path to access
            operation: Function to perform on the path
            max_retries: Maximum number of retries (default: len(retry_delays))
            
        Returns:
            Result of the operation
            
        Raises:
            OSError: If all retry attempts fail
        """
        # Try operation with original path first (no network overhead)
        try:
            return operation(path)
        except (OSError, PermissionError, FileNotFoundError):
            # Only use network handling if direct access fails
            pass
        
        # Use original path for registration key
        original_path = str(path)
        max_retries = max_retries or len(self._retry_delays)
        
        # Register path if not already registered  
        if original_path not in self._network_drives:
            self.register_network_path(path)
        
        path_info = self._network_drives[original_path]
        
        for attempt in range(max_retries + 1):
            try:
                # Try both original and resolved paths
                for try_path in [path_info['original_path'], path_info['resolved_path']]:
                    try:
                        # Test access first for network drives
                        if path_info['is_network'] and not self._test_path_access(try_path):
                            continue  # Try next path
                        
                        # Perform the operation
                        result = operation(try_path)
                        
                        # Success: reset failure count and update last access
                        with self._lock:
                            path_info['failure_count'] = 0
                            path_info['last_access'] = time.time()
                        
                        return result
                    except (OSError, PermissionError, FileNotFoundError):
                        continue  # Try next path
                
                # If we get here, both paths failed
                raise OSError(f"Cannot access {path}")
                
            except (OSError, PermissionError, FileNotFoundError) as e:
                with self._lock:
                    path_info['failure_count'] += 1
                
                if attempt >= max_retries:
                    logger.error(f"Failed to access {path} after {max_retries} retries: {e}")
                    raise
                
                # Calculate delay with exponential backoff
                delay_index = min(attempt, len(self._retry_delays) - 1)
                delay = self._retry_delays[delay_index]
                
                logger.warn(f"Access failed for {path} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                logger.info(f"Retrying in {delay} seconds...")
                
                # Try to reconnect network drives
                if path_info['is_network']:
                    logger.info("Attempting network drive reconnection...")
                    if self._reconnect_network_drive(path_info):
                        logger.info("Network reconnection successful, retrying immediately...")
                        continue
                
                time.sleep(delay)
    
    def check_network_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Check the health of all registered network paths.
        
        Returns:
            Dictionary with path health information
        """
        health_report = {}
        
        with self._lock:
            for path_str, path_info in self._network_drives.items():
                path = path_info['path']
                
                health_report[path_str] = {
                    'path': str(path),
                    'is_network': path_info['is_network'],
                    'accessible': self._test_path_access(path),
                    'failure_count': path_info['failure_count'],
                    'last_access': path_info['last_access']
                }
        
        return health_report
    
    def cleanup(self):
        """Cleanup network accessor resources."""
        with self._lock:
            self._network_drives.clear()
        logger.debug("Network accessor cleanup completed")

# Global instance
_network_accessor: Optional[NetworkAccessor] = None
_network_accessor_lock = threading.Lock()

def get_network_accessor() -> NetworkAccessor:
    """Get the global network accessor instance."""
    global _network_accessor
    with _network_accessor_lock:
        if _network_accessor is None:
            _network_accessor = NetworkAccessor()
        return _network_accessor

def cleanup_network_accessor():
    """Cleanup the global network accessor."""
    global _network_accessor
    with _network_accessor_lock:
        if _network_accessor:
            _network_accessor.cleanup()
            _network_accessor = None

def safe_file_operation(file_path: Path, operation: Callable[[Path], Any], 
                       max_retries: Optional[int] = None) -> Any:
    """
    Convenience function for safe file operations with network resilience.
    
    Args:
        file_path: Path to the file
        operation: Function to perform on the file path
        max_retries: Maximum number of retries
        
    Returns:
        Result of the operation
    """
    accessor = get_network_accessor()
    return accessor.safe_access(file_path, operation, max_retries)
