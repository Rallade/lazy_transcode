"""
VBR (Variable Bit Rate) optimization module for lazy_transcode.

This module handles VBR-specific optimization operations including:
- Intelligent bitrate bounds calculation with progressive expansion
- Coordinate descent optimization over encoder parameters (preset, bf, refs)
- Bisection search for minimum bitrate achieving target VMAF
- VBR encoding command generation with advanced parameters
"""

import subprocess
import shlex
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .system_utils import TEMP_FILES, DEBUG
from .media_utils import get_duration_sec, compute_vmaf_score, ffprobe_field


def build_vbr_encode_cmd(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                        max_bitrate: int, avg_bitrate: int, preset: str = "medium", 
                        bf: int = 3, refs: int = 3, preserve_hdr_metadata: bool = True) -> List[str]:
    """Build VBR encoding command with advanced encoder parameters."""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    
    # Input
    cmd.extend(["-i", str(infile)])
    
    # Video encoding  
    cmd.extend(["-c:v", encoder])
    
    if encoder_type == "hardware":
        if "nvenc" in encoder:
            cmd.extend(["-preset", preset, "-rc", "vbr", "-maxrate", f"{max_bitrate}k",
                       "-b:v", f"{avg_bitrate}k", "-bufsize", f"{max_bitrate * 2}k",
                       "-bf", str(bf), "-refs", str(refs)])
        elif "amf" in encoder:
            cmd.extend(["-usage", "transcoding", "-rc", "vbr_peak", "-b:v", f"{avg_bitrate}k",
                       "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate * 2}k", 
                       "-bf", str(bf), "-refs", str(refs)])
        elif "videotoolbox" in encoder:
            cmd.extend(["-b:v", f"{avg_bitrate}k", "-maxrate", f"{max_bitrate}k",
                       "-bufsize", f"{max_bitrate * 2}k"])
        elif "qsv" in encoder:
            cmd.extend(["-preset", preset, "-b:v", f"{avg_bitrate}k", 
                       "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate * 2}k",
                       "-bf", str(bf), "-refs", str(refs)])
    else:
        # Software encoder (x265) VBR with advanced parameters
        cmd.extend(["-preset", preset, "-b:v", f"{avg_bitrate}k", 
                   "-maxrate", f"{max_bitrate}k", "-bufsize", f"{max_bitrate * 2}k"])
        
        # x265 specific parameters including advanced options
        x265_params = [
            "rd=6",
            f"bframes={bf}",
            f"ref={refs}",
            "me=3",  # umh motion estimation
            "subme=7",  # high subpixel refinement
            "merange=25",  # motion estimation range
            "b-adapt=2",  # adaptive B-frame decision
            "pmode",  # parallel mode decision
            "pme"  # parallel motion estimation
        ]
        
        if preserve_hdr_metadata:
            x265_params.extend([
                "colorprim=bt2020",
                "transfer=smpte2084", 
                "colormatrix=bt2020nc",
                "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)",
                "max-cll=1000,400"
            ])
        cmd.extend(["-x265-params", ":".join(x265_params)])
    
    # Audio and subtitle copy
    cmd.extend(["-c:a", "copy", "-c:s", "copy"])
    
    # Output
    cmd.append(str(outfile))
    
    return cmd


def calculate_intelligent_vbr_bounds(infile: Path, target_vmaf: float, expand_factor: int = 0) -> tuple[int, int]:
    """Calculate intelligent VBR bitrate bounds with progressive expansion."""
    
    # Get source file bitrate
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=bit_rate", "-of", "csv=p=0", str(infile)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            source_bitrate_bps = int(result.stdout.strip())
            source_bitrate_kbps = source_bitrate_bps // 1000
        else:
            # Fallback: estimate from file size and duration
            duration = get_duration_sec(infile)
            if duration and duration > 0:
                file_size_bits = infile.stat().st_size * 8
                source_bitrate_kbps = int(file_size_bits / duration / 1000)
            else:
                source_bitrate_kbps = 8000  # Conservative fallback
                
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
        source_bitrate_kbps = 8000  # Conservative fallback
    
    print(f"[VBR-BOUNDS] Source bitrate: {source_bitrate_kbps}kbps")
    
    # Calculate estimated target based on VMAF requirements
    if target_vmaf >= 95.0:
        compression_ratio = 0.9  # 90% of original
    elif target_vmaf >= 90.0:
        compression_ratio = 0.5  # 50% of original  
    elif target_vmaf >= 85.0:
        compression_ratio = 0.35 # 35% of original  
    else:
        compression_ratio = 0.25 # 25% of original
    
    target_center = int(source_bitrate_kbps * compression_ratio)
    
    # Start with tight bounds, expand progressively if needed
    base_width = 0.15  # Start with ±15% of target
    expansion_multiplier = 1.8 ** expand_factor  # Expand by 1.8x each time
    current_width = base_width * expansion_multiplier
    
    search_range = int(target_center * current_width)
    
    min_bitrate = max(300, target_center - search_range)  
    max_bitrate = min(20000, target_center + search_range)  
    
    if expand_factor == 0:
        print(f"[VBR-BOUNDS] Starting tight range: {min_bitrate}-{max_bitrate}kbps (center: {target_center}kbps, ±{current_width*100:.0f}%)")
    else:
        print(f"[VBR-BOUNDS] Expanded range #{expand_factor}: {min_bitrate}-{max_bitrate}kbps (±{current_width*100:.0f}%)")
    
    return min_bitrate, max_bitrate


def get_vbr_clip_positions(duration_seconds: float, num_clips: int = 2) -> list[int]:
    """Calculate evenly distributed clip positions for VBR testing."""
    duration_int = int(duration_seconds)
    if duration_int <= 60:
        return [10]  # Single clip near start
    
    # Distribute clips evenly across the timeline
    positions = []
    segment_size = duration_int / (num_clips + 1)
    
    for i in range(1, num_clips + 1):
        pos = int(i * segment_size)
        # Ensure we don't go too close to the end
        pos = min(pos, duration_int - 60)
        positions.append(max(10, pos))  # At least 10 seconds from start
    
    return positions


def optimize_encoder_settings_vbr(infile: Path, encoder: str, encoder_type: str, 
                                  target_vmaf: float, vmaf_tolerance: float,
                                  clip_positions: list[int], clip_duration: int, 
                                  max_trials: int = 15) -> dict:
    """
    Find optimal VBR settings using bisection search and coordinate descent.
    
    Returns dict with:
    - success: bool
    - bitrate: int (kbps) 
    - preset: str
    - bf: int
    - refs: int
    - vmaf_score: float
    - filesize: int (bytes)
    """
    print(f"[VBR] Optimizing {infile.name} for VMAF {target_vmaf:.1f}±{vmaf_tolerance:.1f}")
    
    # Parameter search spaces
    presets = ["fast", "medium", "slow"] if encoder_type == "hardware" else ["fast", "medium", "slow"]
    bf_values = [2, 3, 4] if encoder_type == "hardware" else [3, 4, 6]  
    refs_values = [2, 3, 4] if encoder_type == "hardware" else [3, 4, 6]
    
    best_result = None
    lowest_bitrate = float('inf')
    trials = 0
    abandoned = False  # Flag to break out of all loops when file is hopeless
    
    # Coordinate descent over parameters
    for preset in presets:
        if abandoned:
            break
        for bf in bf_values:
            if abandoned:
                break
            for refs in refs_values:
                if trials >= max_trials or abandoned:
                    break
                    
                print(f"[VBR] Trial {trials+1}/{max_trials}: {preset}, bf={bf}, refs={refs}")
                
                # Progressive bound expansion for this parameter combination
                expand_factor = 0
                max_expansions = 3
                result = None
                
                while expand_factor <= max_expansions:
                    # Calculate bounds with current expansion factor
                    min_bitrate, max_bitrate = calculate_intelligent_vbr_bounds(
                        infile, target_vmaf, expand_factor
                    )
                    
                    # Bisection search on bitrate for this parameter combination
                    result = _bisect_bitrate(
                        infile, encoder, encoder_type, 
                        target_vmaf, vmaf_tolerance, 
                        clip_positions, clip_duration,
                        min_bitrate, max_bitrate,
                        preset, bf, refs,
                        expand_factor
                    )
                    
                    # If successful, break out of expansion loop
                    if result and result['success']:
                        break
                    
                    # If we hit bounds, try expanding
                    if result and result.get('bounds_hit'):
                        # Check if file was abandoned due to impossible quality target
                        if result.get('abandoned'):
                            print(f"[VBR-ABANDON] File cannot reach target quality - stopping all trials")
                            abandoned = True
                            break
                        
                        expand_factor += 1
                        print(f"[VBR] Bounds hit, expanding search range...")
                    else:
                        # Some other failure, don't expand further
                        break
                
                trials += 1
                
                if result and result['success']:
                    if result['bitrate'] < lowest_bitrate:
                        lowest_bitrate = result['bitrate']
                        best_result = result.copy()
                        print(f"[VBR] New best: {result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f}")
                
                # Early termination if we find a very good result
                if best_result and best_result['bitrate'] < 2000:  # Under 2Mbps
                    print(f"[VBR] Early termination - excellent bitrate found")
                    break
    
    if best_result:
        print(f"[VBR] Optimal: {best_result['bitrate']}kbps, "
              f"VMAF {best_result['vmaf_score']:.2f}, "
              f"{best_result['preset']}, bf={best_result['bf']}, refs={best_result['refs']}")
        return best_result
    else:
        print(f"[VBR] Failed to find suitable settings within {max_trials} trials")
        return {'success': False}


def _bisect_bitrate(infile: Path, encoder: str, encoder_type: str,
                   target_vmaf: float, vmaf_tolerance: float,
                   clip_positions: list[int], clip_duration: int,
                   min_br: int, max_br: int,
                   preset: str, bf: int, refs: int,
                   expand_factor: int = 0,
                   max_iterations: int = 8) -> dict:
    """Top-down bisection search to find minimum bitrate achieving target VMAF."""
    
    # Extract clips for testing
    clips = []
    try:
        for i, start_time in enumerate(clip_positions):
            clip_path = infile.with_name(f"{infile.stem}.vbr_clip_{i}_{start_time}{infile.suffix}")
            TEMP_FILES.add(str(clip_path))
            
            extract_cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-loglevel", "error" if not DEBUG else "info",
                "-ss", str(start_time),
                "-i", str(infile),
                "-t", str(clip_duration),
                "-c", "copy",
                str(clip_path)
            ]
            
            if DEBUG:
                print(f"[VBR-EXTRACT] {' '.join(shlex.quote(c) for c in extract_cmd)}")
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode == 0 and clip_path.exists():
                clips.append(clip_path)
            else:
                print(f"[VBR] Failed to extract clip {i}")
                return {'success': False, 'error': f'clip extraction failed: {i}'}
        
        print(f"[VBR] Extracted {len(clips)} clips for bisection search")
        
        def test_bitrate_func(bitrate_kbps: int) -> float:
            """Test a bitrate on all clips and return average VMAF"""
            vmaf_scores = []
            
            for i, clip in enumerate(clips):
                encoded_clip = clip.with_name(f"{clip.stem}.vbr_test_{bitrate_kbps}kbps{clip.suffix}")
                TEMP_FILES.add(str(encoded_clip))
                
                # Build VBR encode command with current parameters
                encode_cmd = build_vbr_encode_cmd(clip, encoded_clip, encoder, encoder_type, 
                                                bitrate_kbps, int(bitrate_kbps * 0.8), 
                                                preset, bf, refs, preserve_hdr_metadata=True)
                
                if not DEBUG:
                    encode_cmd.extend(["-loglevel", "error"])
                
                if DEBUG:
                    print(f"[VBR-TEST] {' '.join(shlex.quote(c) for c in encode_cmd)}")
                
                result = subprocess.run(encode_cmd, capture_output=True, text=True)
                if result.returncode != 0 or not encoded_clip.exists():
                    if DEBUG:
                        print(f"[VBR] Encoding failed for clip {i} at {bitrate_kbps}kbps: {result.stderr}")
                    continue
                
                # Compute VMAF
                vmaf_score = compute_vmaf_score(clip, encoded_clip, n_threads=8)
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                    if DEBUG:
                        print(f"[VBR-TEST] Clip {i}: {bitrate_kbps}kbps -> VMAF {vmaf_score:.2f}")
                
                # Cleanup encoded clip
                if not DEBUG and encoded_clip.exists():
                    try:
                        encoded_clip.unlink()
                        TEMP_FILES.discard(str(encoded_clip))
                    except:
                        pass
            
            avg_vmaf = sum(vmaf_scores) / len(vmaf_scores) if vmaf_scores else 0.0
            print(f"[VBR-TEST] {bitrate_kbps}kbps: Average VMAF {avg_vmaf:.2f} from {len(vmaf_scores)} clips")
            return avg_vmaf
        
        # Bisection search
        current_min = min_br
        current_max = max_br
        best_bitrate_val = current_max
        best_vmaf = 0.0
        bounds_hit = False
        
        print(f"[VBR-BISECT] Starting bisection: {current_min}-{current_max}kbps")
        
        for iteration in range(max_iterations):
            # Test midpoint
            test_bitrate_val = (current_min + current_max) // 2
            
            print(f"[VBR-BISECT] Iteration {iteration+1}: Testing {test_bitrate_val}kbps")
            vmaf_result = test_bitrate_func(test_bitrate_val)
            
            if abs(vmaf_result - target_vmaf) <= vmaf_tolerance:
                # Target achieved
                best_bitrate_val = test_bitrate_val
                best_vmaf = vmaf_result
                print(f"[VBR-BISECT] Target achieved: {test_bitrate_val}kbps, VMAF {vmaf_result:.2f}")
                break
            elif vmaf_result < target_vmaf:
                # Need higher bitrate
                current_min = test_bitrate_val
                if test_bitrate_val >= max_br * 0.95:  # Close to upper bound
                    bounds_hit = True
                    
                    # Early abandonment: progressive thresholds based on expansion factor
                    abandon_threshold = 15.0  # Default for first bounds hit
                    if expand_factor > 0:  # We've already tried expanding bounds
                        # Get more aggressive with each expansion
                        # expand_factor 1: 12 points, expand_factor 2: 9 points, expand_factor 3: 6 points
                        abandon_threshold = max(6.0, 15.0 - (expand_factor * 3.0))
                    
                    vmaf_diff = vmaf_result - target_vmaf
                    if vmaf_diff < -abandon_threshold:
                        print(f"    [ABANDON] VMAF {vmaf_result:.2f} is {abs(vmaf_diff):.1f} points below target")
                        print(f"    [ABANDON] Threshold: {abandon_threshold:.1f} points - file likely cannot reach target quality")
                        result = {
                            'success': False,
                            'bitrate': best_bitrate_val,
                            'preset': preset,
                            'bf': bf,
                            'refs': refs,
                            'vmaf_score': best_vmaf,
                            'filesize': 0,
                            'bounds_hit': True,
                            'abandoned': True,
                            'vmaf_gap': abs(vmaf_diff),
                            'abandon_threshold': abandon_threshold
                        }
                        return result
            else:
                # Can use lower bitrate
                current_max = test_bitrate_val
                best_bitrate_val = test_bitrate_val
                best_vmaf = vmaf_result
            
            # Check convergence
            if current_max - current_min < 100:  # 100kbps convergence
                print(f"[VBR-BISECT] Converged to {best_bitrate_val}kbps, VMAF {best_vmaf:.2f}")
                break
        
        # Check if target was achieved within tolerance
        success = abs(best_vmaf - target_vmaf) <= vmaf_tolerance
        
        # Estimate final file size
        if success:
            duration = get_duration_sec(infile)
            estimated_size = int((best_bitrate_val * 1000 * duration) / 8) if duration > 0 else 0
        else:
            estimated_size = 0
        
        result = {
            'success': success,
            'bitrate': best_bitrate_val,
            'preset': preset,
            'bf': bf,
            'refs': refs,
            'vmaf_score': best_vmaf,
            'filesize': estimated_size,
            'bounds_hit': bounds_hit
        }
        
        return result
        
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


def _test_vbr_encoding(infile: Path, encoder: str, encoder_type: str, bitrate_kbps: int,
                      preserve_hdr_metadata: bool = True) -> bool:
    """Test VBR encoding at specified bitrate."""
    test_output = infile.with_name(f"{infile.stem}.vbr_test_{bitrate_kbps}kbps{infile.suffix}")
    TEMP_FILES.add(str(test_output))
    
    try:
        cmd = build_vbr_encode_cmd(infile, test_output, encoder, encoder_type, 
                                  bitrate_kbps, int(bitrate_kbps * 0.8), 
                                  preserve_hdr_metadata=preserve_hdr_metadata)
        
        if not DEBUG:
            cmd.extend(["-loglevel", "error"])
        
        if DEBUG:
            print(f"[VBR-TEST-ENCODE] {' '.join(shlex.quote(c) for c in cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0 and test_output.exists()
        
        if not success and DEBUG:
            print(f"[VBR-TEST-ENCODE] Failed: {result.stderr}")
        
        return success
        
    finally:
        # Cleanup test file
        if not DEBUG and test_output.exists():
            try:
                test_output.unlink()
                TEMP_FILES.discard(str(test_output))
            except:
                pass
