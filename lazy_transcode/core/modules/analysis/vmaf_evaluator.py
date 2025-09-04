"""
VMAfEvaluator Module

Consolidates all VMAF evaluation patterns including:
- Sample clip creation and management
- QP testing on samples
- VBR bitrate testing with multiple clips  
- VMAF score computation with optimization
- File validation and error handling
- Progress tracking and logging
- Temporary file cleanup

Eliminates ~300+ lines of duplicated VMAF evaluation logic.
"""

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import threading
import time

from ..system.system_utils import run_command


@dataclass
class VMAfResult:
    """Result of VMAF evaluation."""
    vmaf_score: Optional[float]
    encoded_size: int
    reference_size: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class ClipInfo:
    """Information about a video clip."""
    path: Path
    start_time: float
    duration: float
    position_name: str


class VMAfEvaluator:
    """Centralized VMAF evaluation with optimized clip handling and computation."""
    
    def __init__(self, debug: bool = False, temp_files: Optional[Set[str]] = None,
                 baseline_vmaf: Optional[Dict[str, float]] = None):
        self.debug = debug
        self.temp_files = temp_files or set()
        self.baseline_vmaf = baseline_vmaf or {}
        self._cpu_monitor = None
        self._cpu_stop_event = None
        
    def create_sample_clip(self, input_file: Path, sample_duration: int = 60) -> Optional[Path]:
        """Create a sample clip from the middle of the input file."""
        duration = self._get_duration_sec(input_file)
        if duration <= 0:
            return None
            
        start_time = max(0, (duration - sample_duration) / 2) if duration > sample_duration else 0
        sample_clip = input_file.with_name(f"{input_file.stem}.sample_clip{input_file.suffix}")
        self.temp_files.add(str(sample_clip))
        
        extract_cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-loglevel", "error" if not self.debug else "info",
            "-ss", str(start_time),
            "-i", str(input_file),
            "-t", str(sample_duration),
            "-c", "copy",
            str(sample_clip)
        ]
        
        if self.debug:
            print("[SAMPLE-EXTRACT] " + " ".join(shlex.quote(c) for c in extract_cmd))
            
        result = run_command(extract_cmd)
        if result.returncode != 0:
            if self.debug:
                print("[DEBUG] sample extract stderr:\n" + result.stderr)
            return None
            
        return sample_clip if sample_clip.exists() else None
        
    def create_vbr_clips(self, input_file: Path, clip_positions: List[float], 
                        clip_duration: float = 25) -> List[ClipInfo]:
        """Create multiple clips for VBR testing at specified positions."""
        clips = []
        
        for i, position in enumerate(clip_positions):
            clip_path = input_file.parent / f"vbr_ref_clip_{i}_{position}.mkv"
            
            ref_cmd = [
                "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                "-ss", str(position), "-i", str(input_file),
                "-t", str(clip_duration),
                "-c", "copy", str(clip_path)
            ]
            
            result = run_command(ref_cmd)
            if result.returncode == 0 and clip_path.exists():
                clips.append(ClipInfo(
                    path=clip_path,
                    start_time=position,
                    duration=clip_duration,
                    position_name=f"clip_{i}_{position}s"
                ))
                
        return clips
        
    def get_vbr_clip_positions(self, duration_seconds: float, num_clips: int = 2) -> List[float]:
        """Get clip start positions for VBR testing."""
        if duration_seconds < 60:
            return [10.0]  # Single clip for short files
        
        # For longer files, sample from different parts
        positions = []
        segment_size = duration_seconds / (num_clips + 1)
        
        for i in range(1, num_clips + 1):
            pos = float(i * segment_size)
            positions.append(max(30.0, min(pos, duration_seconds - 60.0)))  # Ensure valid range
            
        return positions
        
    def test_qp_on_sample(self, input_file: Path, qp: int, encoder: str, encoder_type: str,
                         sample_duration: int = 60, vmaf_threads: int = 8,
                         preserve_hdr_metadata: bool = True, use_clip_as_sample: bool = False,
                         encode_cmd_builder=None) -> VMAfResult:
        """Test a QP on a sample clip and return VMAF evaluation results."""
        
        if use_clip_as_sample:
            # File is already a clip, use it directly as the sample
            sample_clip = input_file
            skip_cleanup = True  # Don't delete the clip - it's managed by the caller
        else:
            # Create sample clip first
            sample_clip = self.create_sample_clip(input_file, sample_duration)
            if not sample_clip:
                return VMAfResult(None, 0, 0, False, "Failed to create sample clip")
            skip_cleanup = False
    
        encoded_sample = input_file.with_name(f"{input_file.stem}.qp{qp}_sample{input_file.suffix}")
        if not use_clip_as_sample:
            self.temp_files.add(str(encoded_sample))

        try:
            # Encode sample at target QP
            if encode_cmd_builder:
                encode_cmd = encode_cmd_builder(sample_clip, encoded_sample, encoder, encoder_type, qp, preserve_hdr_metadata)
            else:
                # Fallback - this would need to be provided by the caller
                raise ValueError("encode_cmd_builder is required")
            
            # Remove progress args for sample encoding
            encode_cmd = self._clean_encode_command(encode_cmd)
            
            if self.debug:
                print("[SAMPLE-ENCODE] " + " ".join(shlex.quote(c) for c in encode_cmd))
            else:
                # Provide feedback during encoding
                print(f"    -> Encoding QP{qp}...", end="", flush=True)
            
            result = run_command(encode_cmd)
            if result.returncode != 0:
                if not self.debug:
                    print(" FAILED")
                
                error_msg = self._analyze_encoding_error(result, encoder, encoder_type, qp, sample_clip)
                return VMAfResult(None, 0, 0, False, error_msg)
                
            if not encoded_sample.exists():
                if not self.debug:
                    print(" FAILED (no output)")
                return VMAfResult(None, 0, 0, False, f"Encoding produced no output file: {encoded_sample}")
            
            if not self.debug:
                print(" OK", end="", flush=True)

            # Compute baseline VMAF once per file
            self._compute_baseline_vmaf(input_file, sample_clip, vmaf_threads, use_clip_as_sample)
            
            # Compute VMAF encoded vs original sample
            if not self.debug:
                print(", VMAF...", end="", flush=True)
            
            # Validate files before VMAF computation
            validation_error = self._validate_vmaf_files(sample_clip, encoded_sample)
            if validation_error:
                if not self.debug:
                    print(" FAILED")
                return VMAfResult(None, 0, 0, False, validation_error)
            
            vmaf_score = self.compute_vmaf_score(sample_clip, encoded_sample, n_threads=vmaf_threads)
            
            if not self.debug:
                if vmaf_score is not None:
                    print(f" {vmaf_score:.1f}")
                else:
                    print(" FAILED")
            
            encoded_size = encoded_sample.stat().st_size if encoded_sample.exists() else 0
            sample_orig_size = sample_clip.stat().st_size if sample_clip.exists() else 0
            
            success = vmaf_score is not None
            return VMAfResult(vmaf_score, encoded_size, sample_orig_size, success)

        finally:
            if not self.debug and not skip_cleanup:  # keep for inspection when debugging
                self._cleanup_temp_files([encoded_sample])
                
    def test_vbr_settings(self, input_file: Path, encoder: str, encoder_type: str,
                         bitrate_kbps: int, preset: str = "medium", bf: int = 3, refs: int = 3,
                         clip_positions: Optional[List[float]] = None, clip_duration: float = 25,
                         vbr_encode_cmd_builder=None) -> Optional[float]:
        """Test VBR encoding at specific settings and return average VMAF score."""
        
        if not clip_positions:
            duration = self._get_duration_sec(input_file)
            clip_positions = self.get_vbr_clip_positions(duration, num_clips=2)
        
        # Create test clips
        clips = self.create_vbr_clips(input_file, clip_positions, clip_duration)
        if not clips:
            return None
            
        vmaf_scores = []
        
        try:
            for i, clip_info in enumerate(clips):
                # Create encoded clip path
                clip_enc = input_file.parent / f"vbr_enc_clip_{i}_{clip_info.start_time}_{bitrate_kbps}.mkv"
                
                try:
                    # Encode test clip with VBR settings
                    if vbr_encode_cmd_builder:
                        cmd = vbr_encode_cmd_builder(
                            clip_info.path, clip_enc, encoder, encoder_type,
                            bitrate_kbps, preset, bf, refs
                        )
                    else:
                        raise ValueError("vbr_encode_cmd_builder is required")
                    
                    # Run encoding (silent)
                    result = run_command(cmd)
                    if result.returncode != 0 or not clip_enc.exists():
                        if self.debug:
                            print(f"[DEBUG] VBR encoding failed for clip {i}: {result.stderr}")
                        continue
                        
                    # Calculate VMAF on this clip
                    score = self.compute_vmaf_score(clip_info.path, clip_enc, n_threads=16)
                    if score is not None:
                        vmaf_scores.append(score)
                        
                finally:
                    # Clean up encoded clip
                    if clip_enc.exists():
                        clip_enc.unlink()
                        
        finally:
            # Clean up reference clips
            for clip_info in clips:
                if clip_info.path.exists():
                    clip_info.path.unlink()
                    
        if not vmaf_scores:
            return None
            
        # Return average VMAF across clips
        return sum(vmaf_scores) / len(vmaf_scores)
        
    def compute_vmaf_score(self, reference: Path, distorted: Path, n_threads: int = 0,
                          enable_cpu_monitoring: bool = False) -> Optional[float]:
        """Compute VMAF score between reference and distorted video with optimizations."""
        
        # Validate input files
        if not reference.exists():
            if self.debug:
                print(f"[WARN] VMAF reference file missing: {reference}")
            return None
            
        if not distorted.exists():
            if self.debug:
                print(f"[WARN] VMAF distorted file missing: {distorted}")
            return None
        
        # Check for resolution mismatch and pre-scale if needed
        ref_resolution = self._get_video_resolution(reference)
        dist_resolution = self._get_video_resolution(distorted)
        
        actual_reference = reference
        if ref_resolution and dist_resolution and ref_resolution != dist_resolution:
            if self.debug:
                print(f"[VMAF] Resolution mismatch detected: {ref_resolution} vs {dist_resolution}")
            scaled_ref = self._create_scaled_reference(reference, dist_resolution)
            if scaled_ref:
                actual_reference = scaled_ref
            else:
                if self.debug:
                    print("[WARN] Failed to scale reference, proceeding with original")
        
        # Try VMAF computation with retry logic
        score = self._compute_vmaf_with_retry(actual_reference, distorted, n_threads, enable_cpu_monitoring)
        
        # Cleanup scaled reference if created
        if actual_reference != reference and actual_reference.exists():
            self.temp_files.discard(str(actual_reference))
            actual_reference.unlink()
            
        return score
        
    def _compute_vmaf_with_retry(self, reference: Path, distorted: Path, n_threads: int,
                               enable_cpu_monitoring: bool) -> Optional[float]:
        """Compute VMAF with retry logic and error handling."""
        
        # First attempt with specified threads
        returncode, stderr = self._run_vmaf_command(reference, distorted, n_threads, enable_cpu_monitoring)
        
        if returncode == 0:
            return self._extract_vmaf_score(stderr)
        
        # Retry without thread specification if first attempt failed
        if n_threads > 0:
            if self.debug:
                print(f"[VMAF] Retrying without thread specification (first attempt failed)")
            returncode, stderr = self._run_vmaf_command(reference, distorted, 0, enable_cpu_monitoring)
            
            if returncode == 0:
                return self._extract_vmaf_score(stderr)
        
        # Both attempts failed
        if self.debug:
            print(f"[VMAF] Failed to compute score. Return code: {returncode}")
            print(f"[VMAF] stderr: {stderr}")
        
        return None
        
    def _run_vmaf_command(self, reference: Path, distorted: Path, n_threads: int,
                         enable_cpu_monitoring: bool) -> Tuple[int, str]:
        """Execute VMAF command and return result."""
        
        # Enhanced libvmaf options for better threading performance
        opts = []
        if n_threads and n_threads > 0:
            opts.append(f"n_threads={n_threads}")
            # Add subsample option for better threading with high thread counts
            if n_threads >= 4:
                opts.append("n_subsample=1")  # Process every frame for accuracy
        
        # Correct libvmaf syntax with options
        opt_str = f"={':'.join(opts)}" if opts else ""
        filter_graph = f"[0:v][1:v]libvmaf{opt_str}"
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
            "-i", str(distorted),   # distorted first
            "-i", str(reference),   # reference second
            "-lavfi", filter_graph,
            "-f", "null", "-"
        ]
        
        if self.debug:
            print("[VMAF-CMD] " + " ".join(shlex.quote(c) for c in cmd))
            print(f"[VMAF-THREADS] Using {n_threads if n_threads > 0 else 'auto'} threads with enhanced options")
        
        # Start CPU monitoring if requested
        if enable_cpu_monitoring:
            self._cpu_monitor, self._cpu_stop_event = self._start_cpu_monitor()
        
        result = run_command(cmd)
        
        # Stop CPU monitoring
        if enable_cpu_monitoring and self._cpu_monitor and self._cpu_stop_event:
            self._cpu_stop_event.set()
            self._cpu_monitor.join(timeout=1)
        
        return result.returncode, result.stderr or ""
        
    def _extract_vmaf_score(self, stderr: str) -> Optional[float]:
        """Extract VMAF score from ffmpeg stderr output."""
        # Look for the VMAF score in the output
        # Format: VMAF score: 85.123456
        for line in stderr.split('\n'):
            if 'VMAF score:' in line:
                try:
                    # Extract the score value
                    score_str = line.split('VMAF score:')[1].strip()
                    return float(score_str)
                except (IndexError, ValueError):
                    continue
        return None
        
    def _get_duration_sec(self, file: Path) -> float:
        """Get video duration in seconds."""
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                   "-of", "default=nk=1:nw=1", str(file)]
            result = run_command(cmd, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0
        
    def _get_video_resolution(self, video_path: Path) -> Optional[Tuple[int, int]]:
        """Get video resolution using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", 
                "-of", "csv=s=x:p=0", str(video_path)
            ]
            result = run_command(cmd, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                width, height = map(int, result.stdout.strip().split('x'))
                return width, height
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Resolution detection failed for {video_path}: {e}")
        return None
        
    def _create_scaled_reference(self, ref_path: Path, target_resolution: Tuple[int, int]) -> Optional[Path]:
        """Create a temporary scaled version of the reference video."""
        try:
            target_width, target_height = target_resolution
            temp_scaled = Path(f"temp_scaled_{ref_path.stem}_{target_width}x{target_height}.mp4")
            self.temp_files.add(str(temp_scaled))
            
            scale_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(ref_path),
                "-vf", f"scale={target_width}:{target_height}",
                "-c:v", "libx264", "-crf", "18",  # High quality scaling
                "-preset", "fast",  # Balance quality vs speed
                str(temp_scaled)
            ]
            
            if self.debug:
                print(f"[VMAF] Creating scaled reference: {target_width}x{target_height}")
            
            result = run_command(scale_cmd, timeout=300)
            if result.returncode == 0 and temp_scaled.exists():
                return temp_scaled
            else:
                if self.debug:
                    print(f"[WARN] Failed to scale reference video: {result.stderr}")
                self.temp_files.discard(str(temp_scaled))
                return None
                
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Scaling failed: {e}")
            return None
            
    def _clean_encode_command(self, cmd: List[str]) -> List[str]:
        """Remove progress and stats arguments from encode command for sample encoding."""
        cleaned = cmd.copy()
        
        # Remove progress args
        if "-progress" in cleaned:
            idx = cleaned.index("-progress")
            if idx + 1 < len(cleaned):
                cleaned = cleaned[:idx] + cleaned[idx+2:]  # Remove -progress and its value
        
        if "-nostats" in cleaned:
            cleaned.remove("-nostats")
        
        # Ensure error loglevel for quiet sample encoding
        if not self.debug and "-loglevel" not in cleaned:
            cleaned.insert(2, "-loglevel")
            cleaned.insert(3, "error")
            
        return cleaned
        
    def _analyze_encoding_error(self, result: subprocess.CompletedProcess, encoder: str, 
                               encoder_type: str, qp: int, sample_clip: Path) -> str:
        """Analyze encoding error and provide helpful error message."""
        
        # Check for hardware encoder specific issues
        if encoder_type == "hardware" and qp < 20:
            error_msg = f"Hardware encoder {encoder} failed at QP{qp}. Consider using --min-qp 20 or higher."
        else:
            error_msg = f"Encoding failed for {sample_clip.name} at QP{qp}: return code {result.returncode}"
        
        # Enhanced error diagnosis
        if "No such file or directory" in result.stderr:
            if self.debug:
                print(f"[DEBUG] Input file check: {sample_clip} exists = {sample_clip.exists()}")
                if sample_clip.exists():
                    print(f"[DEBUG] Input file size: {sample_clip.stat().st_size} bytes")
        
        if self.debug:
            error_msg += f"\nstderr: {result.stderr}"
        elif result.stderr:
            # Show last error line even in non-debug mode
            error_lines = result.stderr.strip().split('\n')
            if error_lines:
                error_msg += f"\nError: {error_lines[-1]}"
                
        return error_msg
        
    def _compute_baseline_vmaf(self, input_file: Path, sample_clip: Path, vmaf_threads: int, 
                              use_clip_as_sample: bool):
        """Compute baseline (self) VMAF once per file using sample clip."""
        key = str(input_file) if not use_clip_as_sample else str(sample_clip)
        if key not in self.baseline_vmaf and sample_clip.exists():
            base_score = self.compute_vmaf_score(sample_clip, sample_clip, n_threads=vmaf_threads)
            if base_score is not None:
                self.baseline_vmaf[key] = base_score
                
    def _validate_vmaf_files(self, reference: Path, encoded: Path) -> Optional[str]:
        """Validate files before VMAF computation."""
        
        if not reference.exists():
            return f"Reference clip missing for VMAF: {reference}"
            
        if not encoded.exists():
            return f"Encoded sample missing for VMAF: {encoded}"
        
        # Check file sizes to ensure they're not empty/corrupted
        try:
            ref_size = reference.stat().st_size
            enc_size = encoded.stat().st_size
            
            if ref_size == 0:
                return f"Reference clip is empty: {reference}"
            if enc_size == 0:
                return f"Encoded sample is empty: {encoded}"
                
        except OSError as e:
            return f"File access error: {e}"
            
        return None
        
    def _cleanup_temp_files(self, files: List[Path]):
        """Clean up temporary files."""
        for temp_file in files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    self.temp_files.discard(str(temp_file))
                except OSError:
                    pass  # Ignore cleanup errors
                    
    def _start_cpu_monitor(self) -> Tuple[Optional[threading.Thread], Optional[threading.Event]]:
        """Start CPU monitoring thread (placeholder for actual implementation)."""
        # This would need to be implemented based on the existing CPU monitoring logic
        return None, None
        
    def cleanup_all_temp_files(self):
        """Clean up all tracked temporary files."""
        for temp_file_str in list(self.temp_files):
            temp_file = Path(temp_file_str)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass  # Ignore cleanup errors
        self.temp_files.clear()
