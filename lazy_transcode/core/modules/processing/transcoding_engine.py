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

from ..system.system_utils import DEBUG, get_next_transcoded_dir, run_command
from ..analysis.media_utils import get_video_codec  # Re-export for tests
from ..optimization.vbr_optimizer import build_vbr_encode_cmd  # canonical import now; tests should patch here


def _log_input_streams(input_file: Path):
    """Log detailed information about input file streams."""
    try:
        # Run ffprobe to get stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_chapters', str(input_file)
        ]
        
        result = run_command(cmd)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Log stream information
            streams = data.get('streams', [])
            chapters = data.get('chapters', [])
            
            print(f"[TRANSCODE-VBR] Input analysis:")
            print(f"[TRANSCODE-VBR]   Total streams: {len(streams)}")
            
            # Categorize streams
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']
            
            print(f"[TRANSCODE-VBR]   Video streams: {len(video_streams)}")
            for i, stream in enumerate(video_streams):
                codec = stream.get('codec_name', 'unknown')
                resolution = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
                print(f"[TRANSCODE-VBR]     [{i}] {codec} {resolution}")
                
            print(f"[TRANSCODE-VBR]   Audio streams: {len(audio_streams)}")
            for i, stream in enumerate(audio_streams):
                codec = stream.get('codec_name', 'unknown')
                channels = stream.get('channels', '?')
                language = stream.get('tags', {}).get('language', 'und')
                print(f"[TRANSCODE-VBR]     [{i}] {codec} {channels}ch ({language})")
                
            print(f"[TRANSCODE-VBR]   Subtitle streams: {len(subtitle_streams)}")
            for i, stream in enumerate(subtitle_streams):
                codec = stream.get('codec_name', 'unknown')
                language = stream.get('tags', {}).get('language', 'und')
                print(f"[TRANSCODE-VBR]     [{i}] {codec} ({language})")
                
            print(f"[TRANSCODE-VBR]   Chapters: {len(chapters)}")
        else:
            # Non-zero return code without raising an exception – emit warning so tests and users see failure
            print(f"[TRANSCODE-VBR] Warning: Could not analyze input streams (ffprobe return code {result.returncode})")
            
    except Exception as e:
        print(f"[TRANSCODE-VBR] Warning: Could not analyze input streams: {e}")


def _log_output_streams(output_file: Path, input_file: Path):
    """Log detailed information about output file streams and compare with input."""
    try:
        # Run ffprobe to get output stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_chapters', str(output_file)
        ]
        
        result = run_command(cmd)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Log stream information
            streams = data.get('streams', [])
            chapters = data.get('chapters', [])
            
            print(f"[TRANSCODE-VBR] Output verification:")
            print(f"[TRANSCODE-VBR]   Total streams: {len(streams)}")
            
            # Categorize streams
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']
            
            print(f"[TRANSCODE-VBR]   Video streams: {len(video_streams)} (transcoded)")
            print(f"[TRANSCODE-VBR]   Audio streams: {len(audio_streams)} (copied)")
            print(f"[TRANSCODE-VBR]   Subtitle streams: {len(subtitle_streams)} (copied)")
            print(f"[TRANSCODE-VBR]   Chapters: {len(chapters)} (copied)")
            
            # Check file sizes
            input_size = input_file.stat().st_size
            output_size = output_file.stat().st_size
            reduction_pct = ((input_size - output_size) / input_size) * 100
            
            print(f"[TRANSCODE-VBR]   Input size: {input_size / (1024*1024):.1f} MB")
            print(f"[TRANSCODE-VBR]   Output size: {output_size / (1024*1024):.1f} MB")
            print(f"[TRANSCODE-VBR]   Size reduction: {reduction_pct:.1f}%")
            
    except Exception as e:
        print(f"[TRANSCODE-VBR] Warning: Could not verify output streams: {e}")



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
                      progress_callback=None, auto_tune: bool = False) -> bool:
    """Transcode a file with VBR encoding and comprehensive logging."""
    # Set up progress tracking
    progress_file = None
    if progress_callback:
        # Create unique progress file
        progress_name = f"progress_{input_file.stem}_{int(time.time())}.txt"
        progress_file = input_file.parent / progress_name
        
    try:
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Log transcoding start with detailed information
        print(f"\n[TRANSCODE-VBR] Starting transcode of: {input_file.name}")
        print(f"[TRANSCODE-VBR] Output file: {output_file.name}")
        print(f"[TRANSCODE-VBR] Encoder: {encoder} ({encoder_type})")
        print(f"[TRANSCODE-VBR] Target bitrate: {avg_bitrate} kbps (max: {max_bitrate} kbps)")
        print(f"[TRANSCODE-VBR] HDR preservation: {preserve_hdr_metadata}")
        
        # Analyze input streams before transcoding
        _log_input_streams(input_file)
        
        # Build command using the module-level wrapper so tests can patch it
        cmd = build_vbr_encode_cmd(
            input_file, output_file, encoder, encoder_type,
            max_bitrate, avg_bitrate, preserve_hdr_metadata=preserve_hdr_metadata,
            auto_tune=auto_tune
        )
        
        # Add progress tracking if needed
        if progress_callback and progress_file:
            # The comprehensive encoder includes '-progress pipe:1 -nostats'
            # But we need a file-based progress for monitoring, so replace it
            if '-progress' in cmd and 'pipe:1' in cmd:
                progress_idx = cmd.index('-progress')
                cmd[progress_idx + 1] = str(progress_file)  # Replace 'pipe:1' with file path
        
        # Log the full command for transparency
        print(f"[TRANSCODE-VBR] FFmpeg command:")
        print(f"[TRANSCODE-VBR] {' '.join(shlex.quote(c) for c in cmd)}")
        
        # Run encoding
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, text=True)
        
        # Monitor progress if callback provided
        if progress_callback and progress_file:
            monitor_progress(process, progress_file, progress_callback)
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0 and output_file.exists():
            print(f"[TRANSCODE-VBR] ✓ Successfully transcoded {input_file.name}")
            
            # Analyze output streams to confirm preservation
            _log_output_streams(output_file, input_file)
            return True
        else:
            print(f"\n[ERROR] VBR transcoding failed for {input_file.name}")
            print(f"[ERROR] Process return code: {process.returncode}")
            if stdout:
                print(f"[ERROR] STDOUT: {stdout}")
            if stderr:
                print(f"[ERROR] STDERR: {stderr}")
            return False
            
    finally:
        # Clean up progress file
        if progress_file and progress_file.exists():
            try:
                progress_file.unlink()
            except:
                pass


def monitor_progress(process: subprocess.Popen, progress_file: Path, callback):
    """Monitor FFmpeg progress file and call progress callback with enhanced logging."""
    last_progress = {}
    start_time = time.time()
    
    while process.poll() is None:
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                for line in content.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        last_progress[key] = value
                
                # Enhanced progress logging
                if 'progress' in last_progress:
                    progress_status = last_progress['progress']
                    if progress_status == 'continue':
                        # Extract useful progress information
                        frame = last_progress.get('frame', 'N/A')
                        fps = last_progress.get('fps', 'N/A')
                        bitrate = last_progress.get('bitrate', 'N/A')
                        total_size = last_progress.get('total_size', 'N/A')
                        out_time_us = last_progress.get('out_time_us', '0')
                        speed = last_progress.get('speed', 'N/A')
                        
                        # Calculate elapsed time
                        elapsed = time.time() - start_time
                        
                        # Convert microseconds to readable time
                        if out_time_us.isdigit() and int(out_time_us) > 0:
                            seconds = int(out_time_us) / 1000000
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            secs = int(seconds % 60)
                            time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
                        else:
                            time_str = "00:00:00"
                        
                        print(f"[TRANSCODE-VBR] Progress: Frame {frame}, FPS {fps}, "
                              f"Time {time_str}, Speed {speed}, Bitrate {bitrate}")
                
                # Call progress callback with current state (use copy so later mutations don't affect earlier callbacks)
                if 'progress' in last_progress:
                    callback(dict(last_progress))
                    
            except (FileNotFoundError, PermissionError):
                pass
                
        time.sleep(1.0)  # Check every second for better progress visibility
        
    # Final progress update (always emit an 'end' status once process finishes)
    if callback:
        if not last_progress or 'progress' not in last_progress:
            final_progress = {'progress': 'end'}
        else:
            final_progress = dict(last_progress)
            final_progress['progress'] = 'end'
        callback(final_progress)
        print(f"[TRANSCODE-VBR] Encoding completed!")


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


def detect_best_encoder(force_cpu: bool = False) -> tuple[str, str]:
    """Detect the best available encoder and its type."""
    
    if force_cpu:
        # Force software encoding only
        available_encoders = get_encoder_list()
        software_preferences = [
            ("libx265", "software"),
            ("libx264", "software")
        ]
        for encoder, encoder_type in software_preferences:
            if encoder in available_encoders:
                return encoder, encoder_type
        # Fallback to libx265 even if not detected
        return "libx265", "software"
    
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
