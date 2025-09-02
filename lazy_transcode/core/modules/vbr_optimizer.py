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
from .media_utils import get_duration_sec, compute_vmaf_score, ffprobe_field, ffprobe_field


def build_vbr_encode_cmd(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                        max_bitrate: int, avg_bitrate: int, preset: str = "medium", 
                        bf: int = 3, refs: int = 3, preserve_hdr_metadata: bool = True) -> List[str]:
    """Build VBR encoding command with proper HDR detection using EncoderConfigBuilder."""
    
    # Use the sophisticated EncoderConfigBuilder which has proper HDR detection
    from .encoder_config import EncoderConfigBuilder
    
    builder = EncoderConfigBuilder()
    
    # Get video dimensions for proper encoding
    try:
        width = int(ffprobe_field(infile, "width") or "1920")
        height = int(ffprobe_field(infile, "height") or "1080")
    except (ValueError, TypeError):
        width, height = 1920, 1080  # Fallback
    
    # Determine threading based on encoder type
    threads = 4 if encoder_type == "hardware" else None  # Let x265 auto-decide for CPU
    
    # Use the sophisticated VBR command builder which properly handles HDR detection
    cmd = builder.build_vbr_encode_cmd(
        str(infile), str(outfile), encoder, preset, avg_bitrate,
        bf, refs, width, height, threads=threads,
        preserve_hdr=preserve_hdr_metadata, debug=DEBUG
    )
    
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
    
    # Dynamic minimum based on content type, resolution, and expansion attempts
    base_min = 300  # Conservative default
    
    # Adjust for target quality - lower quality targets can use much lower bitrates
    if target_vmaf < 85.0:
        base_min = 200  # More aggressive for lower quality targets
    elif target_vmaf < 80.0:
        base_min = 150  # Very aggressive for low quality
    
    # Get resolution to adjust bounds for smaller content
    try:
        width = ffprobe_field(infile, "width")
        height = ffprobe_field(infile, "height")
        if width and height:
            pixels = int(width) * int(height)
            # Scale minimum based on resolution (1080p = reference)
            resolution_factor = pixels / (1920 * 1080)
            base_min = max(100, int(base_min * resolution_factor))
    except:
        pass  # Keep default if we can't get resolution
    
    # Reduce minimum with each expansion - content clearly compresses well
    if expand_factor > 0:
        reduction = expand_factor * 75  # Reduce by 75kbps per expansion
        adjusted_min = max(100, base_min - reduction)  # Never go below 100kbps
        if expand_factor > 0:
            print(f"[VBR-BOUNDS] Lowering minimum by {reduction}kbps due to expansion (attempt #{expand_factor})")
    else:
        adjusted_min = base_min
    
    min_bitrate = max(adjusted_min, target_center - search_range)  
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
                                  max_trials: int = 8) -> dict:
    """
    Find optimal VBR settings using bisection search and intelligent coordinate descent.
    
    Uses a smart parameter search strategy:
    1. Start with most efficient parameters (medium preset)
    2. Test extreme cases only if medium isn't sufficient  
    3. Prioritize encoder parameters by impact on quality/bitrate
    
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
    
    # Intelligent parameter search spaces - prioritize by impact
    # Start with balanced settings, expand only if needed
    if encoder_type == "hardware":
        # Hardware: preset has biggest impact, then refs, then bf
        presets = ["medium"]  # Start with balanced
        bf_values = [3]       # Start with standard  
        refs_values = [3]     # Start with standard
        # Will expand if no good results found
    else:  # CPU
        # CPU: preset has massive impact, refs moderate, bf smaller
        presets = ["medium"]  # Start with balanced
        bf_values = [4]       # Start with good standard
        refs_values = [4]     # Start with good standard
    
    best_result = None
    lowest_bitrate = float('inf')
    trials = 0
    abandoned = False
    
    # Early exit tracking
    last_improvement_gain = 0
    consecutive_small_gains = 0
    target_hit_count = 0
    similar_results = []  # Track very similar results to avoid redundant work
    
    # Calculate source bitrate for efficiency checks
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=bit_rate", "-of", "csv=p=0", str(infile)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            source_bitrate_kbps = int(result.stdout.strip()) // 1000
        else:
            duration = get_duration_sec(infile)
            if duration and duration > 0:
                file_size_bits = infile.stat().st_size * 8
                source_bitrate_kbps = int(file_size_bits / duration / 1000)
            else:
                source_bitrate_kbps = 8000
    except Exception:
        source_bitrate_kbps = 8000
    
    # Phase 1: Test medium preset with standard parameters
    print(f"[VBR] Phase 1: Testing balanced settings")
    for preset in presets:
        if abandoned or trials >= max_trials:
            break
        for bf in bf_values:
            if abandoned or trials >= max_trials:
                break  
            for refs in refs_values:
                if trials >= max_trials or abandoned:
                    break
                    
                trials += 1
                print(f"[VBR] Trial {trials}/{max_trials}: {preset}, bf={bf}, refs={refs}")
                
                result = _test_parameter_combination(
                    infile, encoder, encoder_type, target_vmaf, vmaf_tolerance,
                    clip_positions, clip_duration, preset, bf, refs
                )
                
                if result.get('abandoned'):
                    print(f"[VBR-ABANDON] File cannot reach target quality")
                    abandoned = True
                    break
                
                if result and result.get('success'):
                    # Calculate improvement metrics
                    previous_best = lowest_bitrate if best_result else float('inf')
                    improvement_gain = previous_best - result['bitrate']
                    compression_ratio = result['bitrate'] / source_bitrate_kbps
                    vmaf_margin = result['vmaf_score'] - target_vmaf
                    
                    if result['bitrate'] < lowest_bitrate:
                        lowest_bitrate = result['bitrate']
                        best_result = result.copy()
                        print(f"[VBR] New best: {result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f} "
                              f"(↓{improvement_gain:.0f}kbps, {compression_ratio:.1%} compression)")
                        
                        # Track improvement trends for diminishing returns detection
                        if improvement_gain < 200:  # Less than 200kbps improvement
                            consecutive_small_gains += 1
                        else:
                            consecutive_small_gains = 0
                        last_improvement_gain = improvement_gain
                    
                    # Count how many times we've hit target quality
                    if vmaf_margin >= -vmaf_tolerance:
                        target_hit_count += 1
                    
                    # Check for very similar results (avoid redundant work)
                    is_similar = any(
                        abs(prev_result['bitrate'] - result['bitrate']) < 100 and 
                        abs(prev_result['vmaf_score'] - result['vmaf_score']) < 0.5
                        for prev_result in similar_results
                    )
                    if is_similar:
                        print(f"[VBR-EARLY-EXIT] Very similar result already found - avoiding redundant work")
                        break
                    else:
                        similar_results.append({
                            'bitrate': result['bitrate'], 
                            'vmaf_score': result['vmaf_score']
                        })
                    
                    # EARLY EXIT STRATEGIES
                    
                    # 1. Excellent compression achieved
                    if compression_ratio < 0.25:  # Under 25% of original bitrate
                        print(f"[VBR-EARLY-EXIT] Excellent compression ratio {compression_ratio:.1%} - stopping optimization")
                        break
                    
                    # 2. High quality with very low bitrate
                    if result['bitrate'] < 1500 and vmaf_margin >= 0:
                        print(f"[VBR-EARLY-EXIT] Exceptional result: {result['bitrate']}kbps with {result['vmaf_score']:.2f} VMAF")
                        break
                    
                    # 3. Target hit multiple times with good compression
                    if target_hit_count >= 2 and compression_ratio < 0.5:
                        print(f"[VBR-EARLY-EXIT] Target consistently achieved with good compression - optimization complete")
                        break
                    
                    # 4. Diminishing returns detection
                    if consecutive_small_gains >= 2 and vmaf_margin >= -vmaf_tolerance:
                        print(f"[VBR-EARLY-EXIT] Diminishing returns detected (consecutive gains <200kbps) with acceptable quality")
                        break
                    
                    # 5. Perfect quality hit early
                    if vmaf_margin >= vmaf_tolerance * 2:  # Way above target
                        print(f"[VBR-EARLY-EXIT] Quality significantly exceeds target ({result['vmaf_score']:.2f} vs {target_vmaf:.1f})")
                        break
                    
                    # Early success - check if we should continue optimizing (existing logic)
                    if result['bitrate'] < 2000:  # Excellent result
                        print(f"[VBR-EARLY-EXIT] Excellent bitrate found - skipping further trials")
                        break
    
    # Phase 2: If no success or high bitrate, try more aggressive settings
    if (not best_result or (best_result and best_result['bitrate'] > 3000)) and not abandoned and trials < max_trials:
        print(f"[VBR] Phase 2: Testing aggressive settings (slow preset)")
        
        # Expand parameter space for better compression
        if encoder_type == "hardware":
            aggressive_presets = ["slow"]
            aggressive_refs = [4, 2]  # Try higher then lower refs
        else:
            aggressive_presets = ["slow"] 
            aggressive_refs = [6, 3]  # Try higher then lower refs
            
        for preset in aggressive_presets:
            if abandoned or trials >= max_trials:
                break
            for refs in aggressive_refs:
                if abandoned or trials >= max_trials:
                    break
                
                trials += 1
                print(f"[VBR] Trial {trials}/{max_trials}: {preset}, bf={bf_values[0]}, refs={refs}")
                
                result = _test_parameter_combination(
                    infile, encoder, encoder_type, target_vmaf, vmaf_tolerance,
                    clip_positions, clip_duration, preset, bf_values[0], refs
                )
                
                if result.get('abandoned'):
                    print(f"[VBR-ABANDON] File cannot reach target quality")
                    abandoned = True
                    break
                
                if result and result.get('success') and result['bitrate'] < lowest_bitrate:
                    # Calculate metrics
                    improvement_gain = lowest_bitrate - result['bitrate']
                    compression_ratio = result['bitrate'] / source_bitrate_kbps
                    vmaf_margin = result['vmaf_score'] - target_vmaf
                    
                    lowest_bitrate = result['bitrate']
                    best_result = result.copy()
                    print(f"[VBR] New best: {result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f} "
                          f"(↓{improvement_gain:.0f}kbps, {compression_ratio:.1%} compression)")
                    
                    # Apply same early exit strategies in Phase 2
                    if compression_ratio < 0.25 or (result['bitrate'] < 1500 and vmaf_margin >= 0):
                        print(f"[VBR-EARLY-EXIT] Phase 2: Exceptional result achieved")
                        break
    
    # Phase 3: If still no success, try fast preset for difficult content
    if not best_result and not abandoned and trials < max_trials:
        print(f"[VBR] Phase 3: Testing fast preset for difficult content")
        
        trials += 1
        print(f"[VBR] Trial {trials}/{max_trials}: fast, bf={bf_values[0]}, refs={refs_values[0]}")
        
        result = _test_parameter_combination(
            infile, encoder, encoder_type, target_vmaf, vmaf_tolerance,
            clip_positions, clip_duration, "fast", bf_values[0], refs_values[0]
        )
        
        if result and result.get('success'):
            best_result = result.copy()
            print(f"[VBR] Fast preset result: {result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f}")
    
    if best_result:
        print(f"[VBR] Optimal: {best_result['bitrate']}kbps, "
              f"VMAF {best_result['vmaf_score']:.2f}, "
              f"{best_result['preset']}, bf={best_result['bf']}, refs={best_result['refs']} "
              f"({trials} trials)")
        return best_result
    else:
        print(f"[VBR] Failed to find suitable settings within {trials} trials")
        return {'success': False}


def _test_parameter_combination(infile: Path, encoder: str, encoder_type: str,
                               target_vmaf: float, vmaf_tolerance: float,
                               clip_positions: list[int], clip_duration: int,
                               preset: str, bf: int, refs: int) -> dict:
    """Test a specific parameter combination with progressive bound expansion."""
    expand_factor = 0
    max_expansions = 3
    
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
        
        # If successful, return result
        if result and result['success']:
            return result
        
        # If we hit bounds, try expanding
        if result and result.get('bounds_hit'):
            # Check if file was abandoned due to impossible quality target
            if result.get('abandoned'):
                return result  # Return abandonment result
            
            expand_factor += 1
            print(f"[VBR] Bounds hit, expanding search range...")
        else:
            # Some other failure, don't expand further
            break
    
    # Return last result even if not successful
    return result or {'success': False}


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
