"""
File Processing Workflows Module

Consolidates file discovery, filtering, codec checking, and temporary file management
patterns that are duplicated across the transcoding system (~200+ lines).

This module provides centralized:
- File discovery with extension filtering
- Codec checking and efficient codec filtering  
- Hidden file and sample clip filtering
- Temporary file lifecycle management
- File validation and error handling
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

from ..system.system_utils import run_command
from ....utils.logging import get_logger

# Module logger
logger = get_logger("file_manager")


@dataclass
class FileDiscoveryResult:
    """Result of file discovery operation."""
    files_to_transcode: List[Path]
    skipped_files: List[Tuple[Path, str]]  # (file, reason)
    hidden_files_skipped: int
    total_files_found: int
    

@dataclass
class CodecCheckResult:
    """Result of codec checking operation."""
    codec: Optional[str]
    should_skip: bool
    reason: str


class FileManager:
    """Centralized file processing workflows and temporary file management."""
    
    def __init__(self, debug: bool = False, temp_files: Optional[Set[str]] = None):
        self.debug = debug
        self.temp_files = temp_files or set()
        
    def discover_video_files(self, base_path: Path, extensions: str = "mkv,mp4,mov,ts") -> FileDiscoveryResult:
        """
        Discover video files with comprehensive filtering.
        
        Args:
            base_path: Directory to search
            extensions: Comma-separated list of extensions
            
        Returns:
            FileDiscoveryResult with categorized files
        """
        if not base_path.exists():
            raise ValueError(f"Path not found: {base_path}")
        
        # Build extension patterns
        patterns = [f"*.{ext.strip().lower()}" for ext in extensions.split(",") if ext.strip()]
        
        # Discover files
        files: List[Path] = []
        for pattern in patterns:
            files.extend(base_path.rglob(pattern))
        files = sorted(set(files))
        total_found = len(files)
        
        if self.debug:
            logger.debug(f"Found {total_found} files with patterns {patterns}")
        
        # Filter hidden files and macOS resource forks
        pre_hidden_count = len(files)
        files = [f for f in files if not f.name.startswith('._') and not f.name.startswith('.')]
        hidden_skipped = pre_hidden_count - len(files)
        
        # Filter sample clips and encoding artifacts
        files = [f for f in files if not self._is_sample_or_artifact(f)]
        
        if self.debug:
            logger.debug(f"After filtering hidden/sample files: {len(files)}")
        
        return FileDiscoveryResult(
            files_to_transcode=files,
            skipped_files=[],  # Will be populated by codec checking
            hidden_files_skipped=hidden_skipped,
            total_files_found=total_found
        )
    
    def _is_sample_or_artifact(self, file_path: Path) -> bool:
        """Check if file is a sample clip or encoding artifact."""
        stem = file_path.stem.lower()
        
        # More specific patterns to avoid false positives with legitimate titles
        return any([
            ".sample_clip" in stem,
            stem.endswith("_sample"),  # More specific than "_sample" in stem
            ".clip" in stem and ".sample" in stem,  # cascading clip artifacts
            # Only match .sample when it's clearly an artifact pattern
            stem.endswith(".sample"),  # files ending with .sample
            stem == "sample",  # exact match for generic sample files
            "_sample" in stem and any(x in stem for x in [".clip", "_qp", "vbr_"]),  # sample with encoding artifacts
            "_qp" in stem and "_sample" in stem,  # QP test samples
            # VBR reference and encoded clips (specific patterns)
            "vbr_ref_clip_" in stem,
            "vbr_enc_clip_" in stem,
            # Other specific clip patterns that are artifacts, not titles
            stem.startswith("clip") and "_" in stem,  # clip1_something, clip2_something
        ])
    
    def check_codec_and_filter(self, files: List[Path]) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Check codecs and filter out files with efficient codecs.
        
        Args:
            files: List of files to check
            
        Returns:
            Tuple of (files_to_transcode, skipped_files_with_reasons)
        """
        files_to_transcode = []
        skipped_files = []
        
        for file_path in files:
            codec_result = self.check_video_codec(file_path)
            
            if codec_result.should_skip:
                skipped_files.append((file_path, codec_result.reason))
                if self.debug:
                    print(f"  SKIP: {file_path.name} ({codec_result.reason})")
            else:
                files_to_transcode.append(file_path)
                codec_name = codec_result.codec or 'unknown'
                if self.debug:
                    print(f"  QUEUE: {file_path.name} ({codec_name})")
        
        return files_to_transcode, skipped_files
    
    def check_video_codec(self, file_path: Path) -> CodecCheckResult:
        """
        Check video codec of a file and determine if it should be skipped.
        
        Args:
            file_path: Path to video file
            
        Returns:
            CodecCheckResult with codec info and skip decision
        """
        try:
            # Get codec using ffprobe
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                str(file_path)
            ]
            
            result = run_command(cmd)
            if result.returncode != 0:
                return CodecCheckResult(
                    codec=None,
                    should_skip=False,
                    reason="codec detection failed"
                )
            
            codec = result.stdout.strip()
            should_skip = self._should_skip_codec(codec)
            reason = f"already {codec}" if should_skip else codec
            
            return CodecCheckResult(
                codec=codec,
                should_skip=should_skip,
                reason=reason
            )
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Codec check failed for {file_path.name}: {e}")
            return CodecCheckResult(
                codec=None,
                should_skip=False,
                reason="codec check error"
            )
    
    def _should_skip_codec(self, codec: Optional[str]) -> bool:
        """Check if a codec should be skipped (already efficient)."""
        if not codec:
            return False
        
        # Skip if already using efficient codecs
        efficient_codecs = {"hevc", "h265", "av1"}
        return codec.lower() in efficient_codecs
    
    def process_files_with_codec_filtering(self, base_path: Path, extensions: str = "mkv,mp4,mov,ts") -> FileDiscoveryResult:
        """
        Complete file processing workflow: discovery + codec filtering.
        
        Args:
            base_path: Directory to search
            extensions: Comma-separated extensions
            
        Returns:
            FileDiscoveryResult with complete processing results
        """
        # Discover files
        discovery_result = self.discover_video_files(base_path, extensions)
        
        if not discovery_result.files_to_transcode:
            return discovery_result
        
        # Check codecs and filter
        logger.discovery(f"Found {len(discovery_result.files_to_transcode)} video files")
        
        files_to_transcode, skipped_files = self.check_codec_and_filter(discovery_result.files_to_transcode)
        
        # Update result
        discovery_result.files_to_transcode = files_to_transcode
        discovery_result.skipped_files = skipped_files
        
        # Report results
        if discovery_result.hidden_files_skipped:
            print(f"[INFO] skipped {discovery_result.hidden_files_skipped} hidden/resource-fork file(s) (. / ._ prefix)")
        
        if skipped_files:
            print(f"[INFO] skipped {len(skipped_files)} files with efficient codecs")
            for file_path, reason in skipped_files:
                print(f"  SKIP: {file_path.name} ({reason})")
        
        for file_path in files_to_transcode:
            codec = self.check_video_codec(file_path).codec or 'unknown'
            print(f"  QUEUE: {file_path.name} ({codec})")
        
        return discovery_result
    
    def register_temp_file(self, file_path: Path) -> None:
        """Register a temporary file for cleanup tracking."""
        self.temp_files.add(str(file_path))
    
    def unregister_temp_file(self, file_path: Path) -> None:
        """Unregister a temporary file from cleanup tracking."""
        self.temp_files.discard(str(file_path))
    
    def cleanup_temp_file(self, file_path: Path, force: bool = False) -> bool:
        """
        Clean up a temporary file safely.
        
        Args:
            file_path: Path to temp file
            force: Force cleanup even in debug mode
            
        Returns:
            True if file was cleaned up, False otherwise
        """
        try:
            if file_path.exists() and (force or not self.debug):
                file_path.unlink()
                self.unregister_temp_file(file_path)
                return True
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Failed to cleanup {file_path}: {e}")
        return False
    
    def cleanup_all_temp_files(self, force: bool = False) -> int:
        """
        Clean up all registered temporary files.
        
        Args:
            force: Force cleanup even in debug mode
            
        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        for file_str in list(self.temp_files):
            file_path = Path(file_str)
            if self.cleanup_temp_file(file_path, force):
                cleaned += 1
        return cleaned
    
    def startup_scavenge(self, base_path: Path) -> int:
        """
        Remove stale temporary files from previous runs.
        
        Args:
            base_path: Directory to scavenge
            
        Returns:
            Number of files removed
        """
        patterns = [
            "*.sample_clip.*",
            "*_sample.*", 
            "*.qp*_sample.*",
            "*_qp*_sample.*",
            "vbr_ref_clip_*",
            "vbr_enc_clip_*"
        ]
        
        removed = 0
        for pattern in patterns:
            for stale_file in base_path.rglob(pattern):
                try:
                    if stale_file.is_file():
                        stale_file.unlink()
                        removed += 1
                        if self.debug:
                            logger.cleanup(f"Removed stale: {stale_file.name}")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Could not remove {stale_file}: {e}")
        
        return removed
    
    def validate_file_access(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate that a file exists and is accessible.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not file_path.exists():
                return False, f"File not found: {file_path}"
            
            if not file_path.is_file():
                return False, f"Not a file: {file_path}"
            
            # Check if file is readable
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read first byte
            
            # Check file size
            size = file_path.stat().st_size
            if size == 0:
                return False, f"File is empty: {file_path}"
            
            return True, ""
            
        except PermissionError:
            return False, f"Permission denied: {file_path}"
        except Exception as e:
            return False, f"File access error: {e}"
    
    def create_temp_file_path(self, base_file: Path, suffix: str, extension: Optional[str] = None) -> Path:
        """
        Create a temporary file path based on the base file.
        
        Args:
            base_file: Original file to base temp path on
            suffix: Suffix to add to filename 
            extension: Override extension (uses base file extension if None)
            
        Returns:
            Path for temporary file
        """
        if extension is None:
            extension = base_file.suffix
        
        temp_path = base_file.with_name(f"{base_file.stem}{suffix}{extension}")
        self.register_temp_file(temp_path)
        return temp_path
    
    def get_file_stats(self, files: List[Path]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about a list of files.
        
        Args:
            files: List of files to analyze
            
        Returns:
            Dictionary with file statistics
        """
        total_size = 0
        total_duration = 0
        codec_counts = {}
        
        for file_path in files:
            try:
                # File size
                total_size += file_path.stat().st_size
                
                # Codec counting
                codec_result = self.check_video_codec(file_path)
                codec = codec_result.codec or 'unknown'
                codec_counts[codec] = codec_counts.get(codec, 0) + 1
                
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Stats error for {file_path.name}: {e}")
        
        return {
            'total_files': len(files),
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024**3),
            'codec_distribution': codec_counts,
            'average_file_size_mb': (total_size / len(files) / (1024**2)) if files else 0
        }


# Standalone utility functions for backward compatibility
def get_video_codec(file: Path) -> str | None:
    """Get the video codec name from a file (backward compatibility wrapper)."""
    manager = FileManager()
    result = manager.check_video_codec(file)
    return result.codec


def should_skip_codec(codec: str) -> bool:
    """Check if a codec should be skipped (backward compatibility wrapper)."""
    manager = FileManager()
    return manager._should_skip_codec(codec)
