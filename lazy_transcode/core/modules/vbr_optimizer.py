"""
VBR (Variable Bitrate) optimization module for lazy_transcode.

This module handles VBR-specific encoding operations including:
- VBR encode command building
- Intelligent bitrate bounds calculation
- Coordinate descent optimization
- VBR-specific progress tracking
"""

import subprocess
import shlex
import time
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

# Import shared utilities
from .media_utils import get_duration_sec, ffprobe_field, _extract_hdr_metadata
from .system_utils import run_logged, TEMP_FILES, DEBUG


def build_vbr_encode_cmd(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                        bitrate_kbps: int, preserve_hdr_metadata: bool = True) -> list[str]:
    """Build VBR encoding command with specified bitrate"""
    pixfmt = ffprobe_field(infile, "pix_fmt")
    prim   = ffprobe_field(infile, "color_primaries")
    trc    = ffprobe_field(infile, "color_transfer")
    matrix = ffprobe_field(infile, "colorspace")
    rng    = ffprobe_field(infile, "color_range")

    is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
    profile = "main10" if is10 else "main"
    pix_out = "p010le" if is10 else "nv12"

    color_args: list[str] = []
    if prim:   color_args += ["-color_primaries:v:0", prim]
    if trc:    color_args += ["-color_trc:v:0",       trc]
    if matrix: color_args += ["-colorspace:v:0",      matrix]
    if rng:    color_args += ["-color_range:v:0",     rng]

    # Base command
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
    ]
    if DEBUG:
        cmd += ["-loglevel", "info"]
    cmd += [
        "-i", str(infile),
        "-map", "0", "-map_metadata", "0", "-map_chapters", "0",
        "-c:v", encoder,
    ]

    # Encoder-specific VBR settings
    if encoder_type == "hardware":
        if encoder == "hevc_amf":
            cmd.extend([
                "-quality", "quality",
                "-rc", "vbr_peak", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{int(bitrate_kbps * 1.2)}k",
                "-profile:v", profile,
                "-pix_fmt", pix_out,
            ])
        elif encoder == "hevc_nvenc":
            cmd.extend([
                "-preset", "medium",
                "-rc", "vbr", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{int(bitrate_kbps * 1.2)}k",
                "-profile:v", profile,
                "-pix_fmt", pix_out,
            ])
        elif encoder == "hevc_qsv":
            cmd.extend([
                "-preset", "medium",
                "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{int(bitrate_kbps * 1.2)}k",
                "-profile:v", profile,
                "-pix_fmt", pix_out,
            ])
    else:  # software (libx265)
        x265_params = [f"bitrate={bitrate_kbps}"]
        if is10:
            # Enable hdr10 signaling if we will embed metadata
            if preserve_hdr_metadata:
                master_display, max_cll = _extract_hdr_metadata(infile)
                if master_display or max_cll:
                    x265_params.append("hdr10=1")
                    x265_params.append("hdr10-opt=1")
                    if master_display:
                        x265_params.append(f"master-display={master_display}")
                    if max_cll:
                        x265_params.append(f"max-cll={max_cll}")
        cmd.extend([
            "-x265-params", ":".join(x265_params),
            "-preset", "medium",
            "-pix_fmt", pix_out,
        ])

    # HDR metadata for hardware encoders
    if preserve_hdr_metadata and encoder_type == "hardware" and is10:
        master_display, max_cll = _extract_hdr_metadata(infile)
        if master_display:
            cmd.extend(["-master_display", master_display])
        if max_cll:
            cmd.extend(["-max_cll", max_cll])

    # Add color args and stream copying
    cmd.extend(color_args)
    cmd.extend([
        "-c:a", "copy", "-c:s", "copy", "-c:d", "copy", "-c:t", "copy",
        "-copy_unknown",
        "-progress", "pipe:1", "-nostats",
        str(outfile)
    ])

    return cmd


def encode_vbr_with_progress(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                           bitrate_kbps: int, preserve_hdr_metadata: bool = True) -> bool:
    """Encode video using VBR with progress tracking"""
    dur = get_duration_sec(infile)
    if dur <= 0:
        print(f"[WARN] could not get duration for {infile}")
    TEMP_FILES.add(str(outfile))

    cmd = build_vbr_encode_cmd(infile, outfile, encoder, encoder_type, bitrate_kbps, preserve_hdr_metadata)
    
    if DEBUG:
        print("[VBR-ENCODE] " + " ".join(shlex.quote(c) for c in cmd))
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Use tqdm for progress tracking
    with tqdm(total=100, desc=f"VBR Encoding {infile.name[:30]}", unit="%", 
             position=1, leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}") as pbar:
        last_pct = 0
        if proc.stdout:
            for line in proc.stdout:
                if any(line.startswith(prefix) for prefix in ["out_time_ms=", "out_time_us=", "out_time="]) and dur > 0:
                    try:
                        time_str = line.split("=",1)[1].strip()
                        if time_str == "N/A":
                            continue
                        
                        # Parse time based on format
                        if line.startswith("out_time_ms="):
                            sec = int(time_str) / 1_000_000
                        elif line.startswith("out_time_us="):
                            sec = int(time_str) / 1_000_000
                        elif line.startswith("out_time="):
                            if ":" in time_str:
                                parts = time_str.split(":")
                                if len(parts) >= 3:
                                    hours = float(parts[0])
                                    minutes = float(parts[1]) 
                                    seconds = float(parts[2])
                                    sec = hours * 3600 + minutes * 60 + seconds
                                else:
                                    continue
                            else:
                                sec = float(time_str)
                        else:
                            continue
                            
                        pct = min(100.0, (sec/dur)*100.0)
                        update_amt = int(pct) - last_pct
                        if update_amt > 0:
                            pbar.update(update_amt)
                            last_pct = int(pct)
                    except (ValueError, IndexError):
                        continue

    proc.wait()
    return proc.returncode == 0


def calculate_intelligent_vbr_bounds(infile: Path, target_vmaf: float, expand_factor: int = 0) -> tuple[int, int]:
    """Calculate intelligent VBR bitrate bounds based on video characteristics"""
    try:
        # Get video properties
        width = ffprobe_field(infile, "width")
        height = ffprobe_field(infile, "height")
        fps = ffprobe_field(infile, "r_frame_rate")
        
        # Parse dimensions
        w = int(width) if width else 1920
        h = int(height) if height else 1080
        
        # Parse framerate
        if fps and "/" in fps:
            num, den = fps.split("/")
            frame_rate = float(num) / float(den)
        else:
            frame_rate = 24.0
        
        # Calculate base bitrate using industry standard formulas
        pixel_count = w * h
        
        # Base bitrate calculation (kbps per 1000 pixels, adjusted for target quality)
        if target_vmaf >= 95:
            # High quality
            base_rate_per_1k_pixels = 0.15
        elif target_vmaf >= 90:
            # Medium-high quality
            base_rate_per_1k_pixels = 0.10
        else:
            # Medium quality
            base_rate_per_1k_pixels = 0.08
        
        base_bitrate = int((pixel_count / 1000) * base_rate_per_1k_pixels * frame_rate)
        
        # Apply expansion factor for exploration
        expansion = 1.0 + (expand_factor * 0.3)  # 30% per expansion level
        
        # Calculate bounds with expansion
        lower_bound = max(500, int(base_bitrate * 0.6 / expansion))
        upper_bound = int(base_bitrate * 1.8 * expansion)
        
        if DEBUG:
            print(f"[VBR-BOUNDS] {w}x{h}@{frame_rate:.1f}fps, target VMAF {target_vmaf}")
            print(f"[VBR-BOUNDS] Base bitrate: {base_bitrate}kbps, expansion: {expansion:.1f}x")
            print(f"[VBR-BOUNDS] Bounds: {lower_bound}-{upper_bound}kbps")
        
        return lower_bound, upper_bound
        
    except Exception as e:
        if DEBUG:
            print(f"[VBR-BOUNDS] Error calculating bounds: {e}")
        # Fallback bounds
        return 1000, 8000


def get_vbr_clip_positions(duration_seconds: int, num_clips: int = 2) -> list[int]:
    """Calculate optimal clip positions for VBR testing"""
    if duration_seconds <= 120:  # Short video
        return [10]  # Single clip near start
    
    # For longer videos, distribute clips evenly
    segment_size = duration_seconds / (num_clips + 1)
    positions = []
    
    for i in range(1, num_clips + 1):
        pos = int(i * segment_size)
        # Ensure we don't go too close to the end
        pos = min(pos, duration_seconds - 60)
        positions.append(max(10, pos))  # At least 10 seconds from start
    
    return positions


def optimize_encoder_settings_vbr(infile: Path, encoder: str, encoder_type: str, 
                                 target_vmaf: float, vmaf_tolerance: float, 
                                 num_clips: int = 2, clip_duration: int = 60,
                                 max_trials: int = 15, preserve_hdr: bool = True,
                                 vmaf_threads: int = 8) -> tuple[int, dict]:
    """
    Optimize VBR bitrate using coordinate descent to find minimum bitrate achieving target VMAF.
    
    Returns:
        Tuple of (optimal_bitrate_kbps, optimization_result_dict)
    """
    from .media_utils import compute_vmaf_score
    
    print(f"[VBR-OPT] Starting VBR optimization for {infile.name}")
    print(f"[VBR-OPT] Target: VMAF {target_vmaf:.1f} ±{vmaf_tolerance:.1f}")
    
    # Get video duration and calculate clip positions
    duration = get_duration_sec(infile)
    if duration <= 0:
        print(f"[VBR-OPT] Could not determine duration for {infile}")
        return 3000, {"error": "no duration"}
    
    clip_positions = get_vbr_clip_positions(duration, num_clips)
    print(f"[VBR-OPT] Using {len(clip_positions)} clips at positions: {clip_positions}s")
    
    # Extract clips for testing
    clips = []
    try:
        for i, start_pos in enumerate(clip_positions):
            clip_path = infile.with_name(f"{infile.stem}_vbr_clip{i}_{start_pos}{infile.suffix}")
            TEMP_FILES.add(str(clip_path))
            
            extract_cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-loglevel", "error" if not DEBUG else "info",
                "-ss", str(start_pos),
                "-i", str(infile),
                "-t", str(clip_duration),
                "-c", "copy",
                str(clip_path)
            ]
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode == 0 and clip_path.exists():
                clips.append(clip_path)
            else:
                print(f"[VBR-OPT] Failed to extract clip {i}")
                return 3000, {"error": f"clip extraction failed: {i}"}
        
        # Calculate initial bounds
        initial_lower, initial_upper = calculate_intelligent_vbr_bounds(infile, target_vmaf)
        
        # Coordinate descent optimization
        best_bitrate = None
        best_vmaf = None
        trial_results = []
        
        # Test function for a specific bitrate
        def test_bitrate(bitrate_kbps: int) -> float:
            """Test a bitrate on all clips and return average VMAF"""
            vmaf_scores = []
            
            for clip_idx, clip in enumerate(clips):
                encoded_clip = clip.with_name(f"{clip.stem}_vbr{bitrate_kbps}{clip.suffix}")
                TEMP_FILES.add(str(encoded_clip))
                
                try:
                    # Encode clip at target bitrate
                    encode_cmd = build_vbr_encode_cmd(clip, encoded_clip, encoder, encoder_type, 
                                                    bitrate_kbps, preserve_hdr)
                    # Remove progress tracking for clips
                    if "-progress" in encode_cmd:
                        idx = encode_cmd.index("-progress")
                        encode_cmd = encode_cmd[:idx] + encode_cmd[idx+2:]
                    if "-nostats" in encode_cmd:
                        encode_cmd.remove("-nostats")
                    
                    result = subprocess.run(encode_cmd, capture_output=True, text=True)
                    if result.returncode != 0 or not encoded_clip.exists():
                        if DEBUG:
                            print(f"[VBR-OPT] Encoding failed for clip {clip_idx} at {bitrate_kbps}kbps")
                        continue
                    
                    # Compute VMAF
                    vmaf_score = compute_vmaf_score(clip, encoded_clip, n_threads=vmaf_threads)
                    if vmaf_score is not None:
                        vmaf_scores.append(vmaf_score)
                    
                    # Cleanup encoded clip immediately
                    if encoded_clip.exists():
                        encoded_clip.unlink()
                        TEMP_FILES.discard(str(encoded_clip))
                        
                except Exception as e:
                    if DEBUG:
                        print(f"[VBR-OPT] Error testing clip {clip_idx}: {e}")
                    continue
            
            return sum(vmaf_scores) / len(vmaf_scores) if vmaf_scores else 0.0
        
        # Coordinate descent search
        current_lower = initial_lower
        current_upper = initial_upper
        
        print(f"[VBR-OPT] Starting coordinate descent: {current_lower}-{current_upper}kbps")
        
        with tqdm(total=max_trials, desc="VBR Optimization", position=0) as pbar:
            for trial in range(max_trials):
                # Test midpoint
                test_bitrate_val = (current_lower + current_upper) // 2
                pbar.set_description(f"VBR Opt - Testing {test_bitrate_val}kbps")
                
                avg_vmaf = test_bitrate(test_bitrate_val)
                trial_results.append((test_bitrate_val, avg_vmaf))
                
                vmaf_error = abs(avg_vmaf - target_vmaf)
                pbar.set_description(f"VBR Opt - {test_bitrate_val}kbps: VMAF {avg_vmaf:.1f}")
                
                if DEBUG:
                    print(f"[VBR-OPT] Trial {trial+1}: {test_bitrate_val}kbps → VMAF {avg_vmaf:.2f} (error: {vmaf_error:.2f})")
                
                # Check convergence
                if vmaf_error <= vmaf_tolerance:
                    best_bitrate = test_bitrate_val
                    best_vmaf = avg_vmaf
                    print(f"[VBR-OPT] Converged at {best_bitrate}kbps (VMAF {best_vmaf:.2f})")
                    break
                
                # Update search bounds
                if avg_vmaf < target_vmaf:
                    # Need higher quality, increase bitrate
                    current_lower = test_bitrate_val
                else:
                    # Quality sufficient, try lower bitrate
                    current_upper = test_bitrate_val
                
                # Check if bounds are too close
                if current_upper - current_lower < 100:  # Less than 100kbps difference
                    best_bitrate = current_upper  # Choose higher bitrate for safety
                    best_vmaf = avg_vmaf
                    print(f"[VBR-OPT] Bounds converged at {best_bitrate}kbps")
                    break
                
                pbar.update(1)
        
        # Fallback if no convergence
        if best_bitrate is None:
            best_bitrate = (current_lower + current_upper) // 2
            best_vmaf = target_vmaf  # Estimate
            print(f"[VBR-OPT] Using fallback bitrate: {best_bitrate}kbps")
        
        return best_bitrate, {
            "vmaf": best_vmaf,
            "trials": len(trial_results),
            "bounds": (initial_lower, initial_upper),
            "final_bounds": (current_lower, current_upper),
            "trial_results": trial_results
        }
        
    finally:
        # Cleanup clips
        if not DEBUG:
            for clip in clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
