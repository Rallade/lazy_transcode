"""
Transcoding engine module for lazy_transcode.

This module handles the core encoding operations including:
- FFmpeg command building
- Encoder configuration
- Progress tracking
- HDR metadata preservation
"""

import subprocess
import shlex
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from .system_utils import DEBUG, get_next_transcoded_dir, run_command


def detect_hdr_content(input_file: Path) -> bool:
    """
    Detect if input video actually contains HDR content.
    
    Returns True only if the video has actual HDR characteristics:
    - HDR color primaries (bt2020)  
    - HDR transfer characteristics (smpte2084/arib-std-b67)
    - Wide color gamut indicators
    """
    try:
        import subprocess
        import json
        
        # Use ffprobe to analyze color metadata
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
            "-select_streams", "v:0", str(input_file)
        ]
        
        result = run_command(cmd, timeout=10)
        if result.returncode != 0:
            return False
            
        data = json.loads(result.stdout)
        video_stream = data.get('streams', [{}])[0]
        
        # Check for HDR indicators
        color_primaries = video_stream.get('color_primaries', '').lower()
        color_trc = video_stream.get('color_trc', '').lower() 
        color_space = video_stream.get('color_space', '').lower()
        
        # HDR indicators
        hdr_primaries = ['bt2020', 'bt2020-10', 'bt2020-12']
        hdr_transfer = ['smpte2084', 'arib-std-b67', 'smpte428', 'hlg']
        hdr_colorspace = ['bt2020nc', 'bt2020c', 'bt2020_ncl', 'bt2020_cl']
        
        is_hdr = (
            any(prim in color_primaries for prim in hdr_primaries) or
            any(trc in color_trc for trc in hdr_transfer) or  
            any(cs in color_space for cs in hdr_colorspace)
        )
        
        return is_hdr
        
    except Exception:
        # If detection fails, assume SDR to avoid false HDR metadata
        return False


def build_encode_cmd(input_file: Path, output_file: Path, encoder: str, encoder_type: str, 
                    qp: int, preserve_hdr_metadata: bool = True, 
                    progress_file: Optional[Path] = None) -> List[str]:
    """Build FFmpeg encoding command with proper HDR detection."""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    
    # Input
    cmd.extend(["-i", str(input_file)])
    
    # CRITICAL FIX: Only apply HDR metadata if source is actually HDR
    apply_hdr = preserve_hdr_metadata and detect_hdr_content(input_file)
    
    # Video encoding
    cmd.extend(["-c:v", encoder])
    
    if encoder_type == "hardware":
        # Hardware encoder settings
        if "nvenc" in encoder:
            cmd.extend(["-preset", "slow", "-cq", str(qp)])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084"])
        elif "amf" in encoder:
            cmd.extend(["-usage", "transcoding", "-quality", "quality", "-qp_i", str(qp), 
                       "-qp_p", str(qp), "-qp_b", str(qp)])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084"])
        elif "videotoolbox" in encoder:
            cmd.extend(["-q:v", str(qp)])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084", "-color_range", "tv"])
        elif "qsv" in encoder:
            cmd.extend(["-preset", "veryslow", "-global_quality", str(qp)])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084"])
    else:
        # Software encoder (x265)
        cmd.extend(["-preset", "slow", "-crf", str(qp)])
        
        # x265 specific parameters
        x265_params = ["rd=6"]
        if apply_hdr:
            x265_params.extend([
                "colorprim=bt2020",
                "transfer=smpte2084", 
                "colormatrix=bt2020nc",
                "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)",
                "max-cll=1000,400"
            ])
        cmd.extend(["-x265-params", ":".join(x265_params)])
    
    # Audio copy
    cmd.extend(["-c:a", "copy"])
    
    # Subtitle copy
    cmd.extend(["-c:s", "copy"])
    
    # Progress tracking
    if progress_file:
        cmd.extend(["-progress", str(progress_file), "-nostats"])
    
    # Output
    cmd.append(str(output_file))
    
    return cmd


def build_vbr_encode_cmd(input_file: Path, output_file: Path, encoder: str, encoder_type: str,
                        max_bitrate: int, avg_bitrate: int, preserve_hdr_metadata: bool = True,
                        progress_file: Optional[Path] = None) -> List[str]:
    """Build FFmpeg VBR encoding command."""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    
    # Input  
    cmd.extend(["-i", str(input_file)])
    
    # CRITICAL FIX: Only apply HDR metadata if source is actually HDR
    apply_hdr = preserve_hdr_metadata and detect_hdr_content(input_file)
    
    # Video encoding
    cmd.extend(["-c:v", encoder])
    
    if encoder_type == "hardware":
        # Hardware encoder VBR settings
        if "nvenc" in encoder:
            cmd.extend(["-preset", "slow", "-rc", "vbr", "-maxrate", f"{max_bitrate}k", 
                       "-b:v", f"{avg_bitrate}k", "-bufsize", f"{max_bitrate * 2}k"])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084"])
        elif "amf" in encoder:
            cmd.extend(["-usage", "transcoding", "-rc", "vbr_peak", "-b:v", f"{avg_bitrate}k",
                       "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate * 2}k"])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084"])
        elif "videotoolbox" in encoder:
            cmd.extend(["-b:v", f"{avg_bitrate}k", "-maxrate", f"{max_bitrate}k",
                       "-bufsize", f"{max_bitrate * 2}k"])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084", "-color_range", "tv"])
        elif "qsv" in encoder:
            cmd.extend(["-preset", "veryslow", "-b:v", f"{avg_bitrate}k", 
                       "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate * 2}k"])
            if apply_hdr:
                cmd.extend(["-colorspace", "bt2020nc", "-color_primaries", "bt2020", 
                           "-color_trc", "smpte2084"])
    else:
        # Software encoder (x265) VBR
        cmd.extend(["-preset", "slow", "-b:v", f"{avg_bitrate}k", 
                   "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate * 2}k"])
        
        # x265 specific parameters
        x265_params = ["rd=6"]
        if apply_hdr:
            x265_params.extend([
                "colorprim=bt2020",
                "transfer=smpte2084",
                "colormatrix=bt2020nc",
                "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)",
                "max-cll=1000,400"
            ])
        cmd.extend(["-x265-params", ":".join(x265_params)])
    
    # Audio copy
    cmd.extend(["-c:a", "copy"])
    
    # Subtitle copy  
    cmd.extend(["-c:s", "copy"])
    
    # Progress tracking
    if progress_file:
        cmd.extend(["-progress", str(progress_file), "-nostats"])
    
    # Output
    cmd.append(str(output_file))
    
    return cmd


def transcode_file_qp(input_file: Path, output_file: Path, encoder: str, encoder_type: str,
                     qp: int, preserve_hdr_metadata: bool = True,
                     progress_callback=None) -> bool:
    """Transcode a file with CRF/QP encoding."""
    # Set up progress tracking
    progress_file = None
    if progress_callback:
        # Create unique progress file
        progress_name = f"progress_{input_file.stem}_{int(time.time())}.txt"
        progress_file = input_file.parent / progress_name
        
    try:
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = build_encode_cmd(input_file, output_file, encoder, encoder_type, 
                              qp, preserve_hdr_metadata, progress_file)
        
        if DEBUG:
            print(f"[TRANSCODE-QP] {' '.join(shlex.quote(c) for c in cmd)}")
        
        # Run encoding
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, text=True)
        
        # Monitor progress if callback provided
        if progress_callback and progress_file:
            monitor_progress(process, progress_file, progress_callback)
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0 and output_file.exists():
            return True
        else:
            print(f"[ERROR] Transcoding failed for {input_file.name}")
            if stderr:
                print(f"[ERROR] {stderr}")
            return False
            
    finally:
        # Clean up progress file
        if progress_file and progress_file.exists():
            try:
                progress_file.unlink()
            except:
                pass


def transcode_file_vbr(input_file: Path, output_file: Path, encoder: str, encoder_type: str,
                      max_bitrate: int, avg_bitrate: int, preserve_hdr_metadata: bool = True,
                      progress_callback=None) -> bool:
    """Transcode a file with VBR encoding."""
    # Set up progress tracking
    progress_file = None
    if progress_callback:
        # Create unique progress file
        progress_name = f"progress_{input_file.stem}_{int(time.time())}.txt"
        progress_file = input_file.parent / progress_name
        
    try:
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = build_vbr_encode_cmd(input_file, output_file, encoder, encoder_type,
                                  max_bitrate, avg_bitrate, preserve_hdr_metadata, progress_file)
        
        if DEBUG:
            print(f"[TRANSCODE-VBR] {' '.join(shlex.quote(c) for c in cmd)}")
        
        # Run encoding
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, text=True)
        
        # Monitor progress if callback provided
        if progress_callback and progress_file:
            monitor_progress(process, progress_file, progress_callback)
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0 and output_file.exists():
            return True
        else:
            print(f"[ERROR] VBR transcoding failed for {input_file.name}")
            if stderr:
                print(f"[ERROR] {stderr}")
            return False
            
    finally:
        # Clean up progress file
        if progress_file and progress_file.exists():
            try:
                progress_file.unlink()
            except:
                pass


def monitor_progress(process: subprocess.Popen, progress_file: Path, callback):
    """Monitor FFmpeg progress file and call progress callback."""
    last_progress = {}
    
    while process.poll() is None:
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                for line in content.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        last_progress[key] = value
                
                # Call progress callback with current state
                if 'progress' in last_progress:
                    callback(last_progress)
                    
            except (FileNotFoundError, PermissionError):
                pass
                
        time.sleep(0.5)  # Check every 500ms
        
    # Final progress update
    if last_progress and callback:
        last_progress['progress'] = 'end'
        callback(last_progress)


def get_encoder_list() -> List[str]:
    """Get list of available encoders."""
    # Run ffmpeg -encoders and parse output
    try:
        result = run_command(['ffmpeg', '-hide_banner', '-encoders'])
        if result.returncode != 0:
            return []
            
        encoders = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('V') and 'h265' in line or 'hevc' in line:
                # Extract encoder name
                parts = line.split()
                if len(parts) >= 2:
                    encoders.append(parts[1])
                    
        return encoders
    except:
        return []


def detect_best_encoder() -> tuple[str, str]:
    """Detect the best available encoder and its type."""
    available_encoders = get_encoder_list()
    
    # Priority order: AMD -> NVIDIA -> Intel -> Software (matches media_utils.py)
    encoder_preferences = [
        ("hevc_amf", "hardware"),
        ("hevc_nvenc", "hardware"), 
        ("hevc_videotoolbox", "hardware"),
        ("hevc_qsv", "hardware"),
        ("libx265", "software")
    ]
    
    for encoder, encoder_type in encoder_preferences:
        if encoder in available_encoders:
            return encoder, encoder_type
            
    # Fallback
    return "libx265", "software"
