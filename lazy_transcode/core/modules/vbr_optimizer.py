"""
VBR (Variable Bit Rate) optimization module for lazy_transcode.

This module handles VBR-specific optimization operations including:
- Intelligent bitrate bounds calculation with progressive expansion
- Coordinate descent optimization over encoder parameters (preset, bf, refs)
- Bisection search for minimum bitrate achieving target VMAF
- VBR encoding command generation with advanced parameters

RESEARCH INTEGRATION:
Enhanced with academic findings from:
Farhadi Nia, M. (2025). "Cross-Codec Quality-Rate Convex Hulls Relation for 
Adaptive Streaming." Department of Electrical and Computer Engineering,
University of Massachusetts Lowell, Lowell, MA, USA.

Key research-based improvements:
- Polynomial convex hull mathematical models (Section 4.3)
- Content-adaptive analysis via SI/TI metrics (Section 3.1.2)
- Resolution-specific optimization (Section 3.2.2)
- Cross-codec quality-rate prediction methodology

Citation:
Farhadi Nia, M. "Explore Cross-Codec Quality-Rate Convex Hulls Relation for
Adaptive Streaming." University of Massachusetts Lowell, 2025.
Corresponding author: Masoumeh_FarhadiNia@student.uml.edu
"""

import subprocess
import shlex
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .system_utils import TEMP_FILES, DEBUG, format_size, temporary_file, run_command
from .media_utils import get_duration_sec, compute_vmaf_score, ffprobe_field, get_video_dimensions
from .quality_rate_predictor import get_quality_rate_predictor
from .content_analyzer import get_content_analyzer
from .resolution_optimizer import get_resolution_optimizer
from ...utils.logging import get_logger

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.n = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): self.n += n
        def set_description(self, desc): self.desc = desc
        def set_postfix(self, postfix): pass  # No-op for fallback
        def refresh(self): pass  # No-op for fallback
        def close(self): pass

# Module logger
logger = get_logger()


def warn_hardware_encoder_inefficiency(encoder_type: str, operation: str = "pre-encoding"):
    """
    Warn users about hardware encoder storage inefficiency for pre-encoding.
    
    Based on research showing hardware encoders create 2-3x larger files 
    than software encoders for minimal quality gains when pre-encoding.
    """
    if encoder_type in ['hardware', 'nvenc', 'qsv', 'amf']:
        logger.vbr(f"⚠️  Hardware encoders create 2-3x larger files than software encoders for {operation}")
        logger.vbr(f"⚠️  Consider using --cpu flag for storage-efficient pre-encoding")
        logger.vbr(f"⚠️  Hardware encoders are best for real-time transcoding, not file optimization")


def run_ffmpeg_with_progress(cmd: List[str], duration: Optional[float] = None, operation: str = "Encoding") -> subprocess.CompletedProcess:
    """Run FFmpeg command with progress reporting."""
    
    # If duration is available, we can show progress
    if duration and duration > 10:  # Only show progress for longer operations
        # Add progress reporting to FFmpeg
        progress_cmd = cmd.copy()
        # Insert progress options after ffmpeg but before input
        try:
            ffmpeg_idx = next(i for i, arg in enumerate(progress_cmd) if arg.endswith('ffmpeg'))
            # Add progress reporting
            progress_cmd.insert(ffmpeg_idx + 1, '-progress')
            progress_cmd.insert(ffmpeg_idx + 2, 'pipe:1')
            progress_cmd.insert(ffmpeg_idx + 3, '-nostats')
        except StopIteration:
            # Fallback to regular execution
            return run_command(cmd)
        
        # Start process with progress monitoring
        with tqdm(total=100, desc=f"[{operation}]", unit="%", position=0) as pbar:
            process = subprocess.Popen(progress_cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
            
            start_time = time.time()
            
            # Monitor progress
            while True:
                if process.stdout is None:
                    break
                    
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output and 'out_time_ms=' in output:
                    try:
                        # Parse time in microseconds
                        time_str = output.split('out_time_ms=')[1].strip()
                        current_time_us = int(time_str)
                        current_time_s = current_time_us / 1000000.0
                        
                        # Calculate progress percentage
                        progress = min(int((current_time_s / duration) * 100), 100)
                        pbar.n = progress
                        
                        # Update description with time info
                        elapsed = time.time() - start_time
                        if current_time_s > 0:
                            speed = current_time_s / elapsed
                            eta = (duration - current_time_s) / speed if speed > 0 else 0
                            pbar.set_description(f"[{operation}] {progress}% (Speed: {speed:.1f}x, ETA: {eta:.0f}s)")
                        
                        if hasattr(pbar, 'refresh'):
                            pbar.refresh()
                        else:
                            pbar.update(0)  # Force display update
                    except (ValueError, IndexError):
                        continue
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            # Close progress bar
            pbar.n = 100
            if hasattr(pbar, 'refresh'):
                pbar.refresh()
            elif hasattr(pbar, 'update'):
                pbar.update(0)  # Force final display update
            
            return subprocess.CompletedProcess(
                progress_cmd, process.returncode, stdout, stderr
            )
    else:
        # For short operations, just run normally
        return run_command(cmd)


def get_video_duration(file_path: Path) -> float:
    """Get video duration in seconds."""
    try:
        return get_duration_sec(file_path)
    except:
        return 0.0


def extract_single_clip(infile: Path, start_time: int, clip_duration: int, clip_index: int) -> Tuple[Optional[Path], Optional[str]]:
    """Extract a single clip - for parallel processing."""
    clip_path = infile.with_name(f"{infile.stem}.vbr_clip_{clip_index}_{start_time}{infile.suffix}")
    TEMP_FILES.add(str(clip_path))
    
    extract_cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-loglevel", "error",
        "-ss", str(start_time),
        "-i", str(infile),
        "-t", str(clip_duration),
        "-c", "copy",
        str(clip_path)
    ]
    
    try:
        result = run_command(extract_cmd, timeout=30)
        if result.returncode == 0 and clip_path.exists():
            return clip_path, None
        else:
            return None, f"Extraction failed for clip {clip_index}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return None, f"Extraction timeout for clip {clip_index}"
    except Exception as e:
        return None, f"Extraction error for clip {clip_index}: {str(e)}"


def extract_clips_parallel(infile: Path, clip_positions: List[int], clip_duration: int) -> Tuple[List[Path], Optional[str]]:
    """Extract clips in parallel for faster processing."""
    clips = []
    errors = []
    
    # Use ThreadPoolExecutor for parallel extraction
    max_workers = min(len(clip_positions), 4)  # Limit concurrent extractions
    
    logger.vbr(f"Extracting {len(clip_positions)} clips in parallel (max {max_workers} workers)...")
    
    with tqdm(total=len(clip_positions), desc="[VBR-EXTRACT-PAR]", 
             unit="clip", leave=False, position=0) as pbar:
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all extraction tasks
            future_to_index = {
                executor.submit(extract_single_clip, infile, start_time, clip_duration, i): i
                for i, start_time in enumerate(clip_positions)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                clip_index = future_to_index[future]
                try:
                    clip_path, error = future.result()
                    if clip_path:
                        clips.append((clip_index, clip_path))
                        pbar.set_description(f"[VBR-EXTRACT-PAR] Completed {len(clips)}/{len(clip_positions)}")
                        logger.debug(f"Extracted clip {clip_index+1}: {clip_path.name} ({format_size(clip_path.stat().st_size)})")
                    else:
                        if error:
                            errors.append(error)
                            logger.vbr_error(error)
                except Exception as exc:
                    error_msg = f"Clip {clip_index} generated exception: {exc}"
                    errors.append(error_msg)
                    logger.vbr_error(error_msg)
                finally:
                    pbar.update(1)
    
    # Sort clips by their original order
    clips.sort(key=lambda x: x[0])
    final_clips = [clip_path for _, clip_path in clips]
    
    if errors and len(final_clips) == 0:
        return [], f"All clip extractions failed: {'; '.join(errors[:3])}"
    elif errors:
        logger.vbr(f"Extracted {len(final_clips)} clips with {len(errors)} failures")
    else:
        logger.vbr(f"Successfully extracted all {len(final_clips)} clips in parallel")
    
    return final_clips, None


def get_coordinated_adaptive_bounds(trial_results: List[Dict], current_min: int, current_max: int,
                                   target_vmaf: float, tolerance: float) -> Tuple[Tuple[int, int], str]:
    """
    Coordinated cross-trial learning that adjusts both bounds together intelligently.
    
    Replaces separate get_adaptive_lower_bounds() and analyze_quality_gap_dynamically()
    to prevent race conditions and ensure consistent bounds.
    """
    
    if not trial_results:
        return (current_min, current_max), "no previous results"
    
    # Get successful results for analysis
    successful_results = [r for r in trial_results if r.get('success')]
    if not successful_results:
        return (current_min, current_max), "no successful results"
    
    # Extract data for coordinated analysis
    bitrates = [r['bitrate'] for r in successful_results]
    vmaf_scores = [r['vmaf_score'] for r in successful_results]
    quality_excesses = [vmaf - target_vmaf for vmaf in vmaf_scores]
    
    # Find the optimal range from actual results
    min_successful = min(bitrates)
    avg_excess = sum(quality_excesses) / len(quality_excesses)
    min_excess = min(quality_excesses)
    
    # COORDINATED DECISION MAKING (not independent adjustments)
    
    # Strategy 1: Successful results with reasonable quality
    reasonable_results = [(b, v) for b, v in zip(bitrates, vmaf_scores) 
                         if target_vmaf - tolerance <= v <= target_vmaf + tolerance * 2]
    
    if reasonable_results:
        # We have good results - narrow bounds around them
        reasonable_bitrates = [b for b, v in reasonable_results]
        new_min = int(min(reasonable_bitrates) * 0.85)  # 15% below best
        new_max = int(max(reasonable_bitrates) * 1.15)  # 15% above acceptable
        reason = f"narrowing around {len(reasonable_results)} good results ({min(reasonable_bitrates)}-{max(reasonable_bitrates)}kbps)"
        
    elif min_excess > tolerance * 1.5:
        # Strategy 2: All results overshoot significantly - aggressive reduction
        if avg_excess >= 4.0:
            reduction_factor = 0.6  # Aggressive 40% reduction
        elif avg_excess >= 2.5:
            reduction_factor = 0.7  # Moderate 30% reduction  
        else:
            reduction_factor = 0.8  # Conservative 20% reduction
            
        # Apply coordinated reduction to both bounds
        new_max = int(min_successful * reduction_factor)
        new_min = int(new_max * 0.7)  # Ensure reasonable range
        reason = f"coordinated overshoot reduction: avg +{avg_excess:.1f} VMAF, {int((1-reduction_factor)*100)}% reduction"
        
    else:
        # Strategy 3: Results are close to target - minor optimization
        new_min = int(min_successful * 0.9)  # Conservative expansion
        new_max = int(max(bitrates) * 1.1)   # Small upper expansion
        reason = f"minor optimization: results near target (excess: +{avg_excess:.1f})"
    
    # CRITICAL VALIDATION: Ensure bounds are viable
    if new_max <= new_min:
        new_max = new_min + 1000  # Minimum 1Mbps range
        reason += " (adjusted for minimum range)"
    
    # Safety bounds
    new_min = max(new_min, 400)     # Absolute minimum
    new_max = min(new_max, 50000)   # Absolute maximum
    
    return (new_min, new_max), reason


def get_adaptive_lower_bounds(trial_results: List[Dict], current_min: int, 
                             target_vmaf: float, tolerance: float) -> Tuple[int, str]:
    """DEPRECATED: Use get_coordinated_adaptive_bounds() instead for better coordination."""
    
    # Redirect to coordinated function, but only return lower bound for backward compatibility
    (new_min, _), reason = get_coordinated_adaptive_bounds(
        trial_results, current_min, current_min * 2, target_vmaf, tolerance)
    
    return new_min, f"legacy redirect: {reason}"


def analyze_quality_gap_dynamically(trial_results: List[Dict], target_vmaf: float, 
                                   tolerance: float) -> Tuple[bool, int, str]:
    """
    Dynamic Quality Gap Analysis - Enhanced to trigger more aggressively
    
    Analyzes trial results to determine if bounds should be aggressively reduced
    based on consistent quality excess patterns.
    
    Returns: (should_reduce_bounds, suggested_new_max, reason)
    """
    
    if len(trial_results) < 1:  # Trigger after just 1 trial
        return False, 0, "insufficient data"
    
    # Get recent successful results for pattern analysis
    recent_results = [r for r in trial_results[-3:] if r.get('success')]
    
    if len(recent_results) < 1:
        return False, 0, "insufficient successful results"
    
    # Calculate quality excess patterns
    quality_excesses = [r.get('vmaf_score', 0) - target_vmaf for r in recent_results]
    avg_excess = sum(quality_excesses) / len(quality_excesses)
    min_excess = min(quality_excesses)
    
    # CRITICAL FIX: Much more aggressive triggering
    # Check for any overshoot pattern (even from single trial)
    if min_excess > tolerance * 1.5:  # Reduced from 2.5x to 1.5x tolerance
        # Calculate aggressive reduction factor based on excess magnitude
        if avg_excess >= 4.0:  # 4+ points excess = very aggressive reduction
            reduction_factor = 0.55  # Reduce max bounds to 55% of current
            reason = f"severe overshoot: avg +{avg_excess:.1f} VMAF (min +{min_excess:.1f})"
        elif avg_excess >= 2.5:  # 2.5-4 points excess = aggressive reduction  
            reduction_factor = 0.65  # Reduce max bounds to 65% of current
            reason = f"significant overshoot: avg +{avg_excess:.1f} VMAF (min +{min_excess:.1f})"
        else:  # 1.5-2.5 points excess = moderate reduction
            reduction_factor = 0.75  # Reduce max bounds to 75% of current
            reason = f"consistent overshoot: avg +{avg_excess:.1f} VMAF (min +{min_excess:.1f})"
        
        # Calculate new maximum based on lowest bitrate that still overshoots
        lowest_overshoot_bitrate = min(r.get('bitrate', 0) for r in recent_results)
        suggested_max = int(lowest_overshoot_bitrate * reduction_factor)
        
        return True, suggested_max, reason
    
    # Check for identical results pattern (optimization #4 enhancement)
    if len(recent_results) >= 2:  # Reduced from 3 to 2
        # Check if last 2 results have nearly identical VMAF and bitrate
        if len(recent_results) >= 2:
            vmaf_variance = max(quality_excesses) - min(quality_excesses)
            bitrate_variance = max(r.get('bitrate', 0) for r in recent_results) - min(r.get('bitrate', 0) for r in recent_results)
            
            if vmaf_variance <= 0.2 and bitrate_variance <= 75:  # More sensitive detection
                return True, 0, f"near-identical results: VMAF variance {vmaf_variance:.2f}, bitrate variance {bitrate_variance}kbps"
    
    return False, 0, "no aggressive reduction needed"


def get_research_based_intelligent_bounds(video_path: Path, encoder: str, encoder_type: str, 
                                        preset: str, target_vmaf: float, 
                                        vmaf_tolerance: float, trial_results: Optional[List[Dict]] = None) -> Tuple[int, int]:
    """
    REVOLUTIONARY: Research-validated intelligent bounds using polynomial models.
    
    Replaces multiplier-based bounds with mathematically elegant approach based on:
    Farhadi Nia, M. (2025). "Cross-Codec Quality-Rate Convex Hulls Relation for 
    Adaptive Streaming." University of Massachusetts Lowell.
    
    SIMPLIFIED 3-LAYER APPROACH (eliminates content analysis averaging conflicts):
    1. LAYER 1: Polynomial convex hull models (Section 4.3: H.264 6th-order, H.265/VP9 5th-order)
    2. LAYER 2: Resolution-adaptive bounds (Section 3.2.2: research-validated, dominates)
    3. LAYER 3: Cross-trial learning (empirical reality - ultimate authority)
    
    Resolution analysis dominates because it's research-validated and doesn't create
    conflicting intelligence when content complexity disagrees with resolution needs.
    
    Citation:
    Farhadi Nia, M. "Explore Cross-Codec Quality-Rate Convex Hulls Relation for
    Adaptive Streaming." Department of Electrical and Computer Engineering,
    University of Massachusetts Lowell, Lowell, MA, USA, 2025.
    """
    logger.vbr("🔬 Using simplified 3-layer research-validated bounds...")
    
    # Get research-based components
    predictor = get_quality_rate_predictor()
    resolution_optimizer = get_resolution_optimizer()
    
    # LAYER 1: Research-based polynomial bounds (starting point)
    polynomial_bounds = predictor.get_intelligent_bitrate_bounds_research(
        encoder, target_vmaf, vmaf_tolerance)
    
    # LAYER 2: Resolution-adaptive bounds (research-validated, dominates)
    try:
        resolution_bounds = resolution_optimizer.get_resolution_adaptive_bounds(
            video_path, target_vmaf, encoder_type)
        
        # Resolution analysis dominates - it's research-validated and more reliable
        combined_min, combined_max = resolution_bounds
        
        logger.vbr(f"Resolution-dominant bounds: {combined_min}-{combined_max}kbps "
                  f"(polynomial: {polynomial_bounds[0]}-{polynomial_bounds[1]}kbps)")
        
    except Exception as e:
        logger.debug(f"Resolution optimization failed, using polynomial bounds: {e}")
        combined_min, combined_max = polynomial_bounds
    
    # LAYER 3: Coordinated cross-trial learning (empirical reality - ultimate authority)
    if trial_results:
        (adaptive_min, adaptive_max), adaptation_reason = get_coordinated_adaptive_bounds(
            trial_results, combined_min, combined_max, target_vmaf, vmaf_tolerance)
        
        if adaptive_min != combined_min or adaptive_max != combined_max:
            logger.vbr(f"Coordinated cross-trial learning: {adaptation_reason}")
            combined_min, combined_max = adaptive_min, adaptive_max
    
    # Step 5: Safety bounds and validation
    final_min = max(500, combined_min)  # Absolute minimum
    final_max = min(50000, combined_max)  # Absolute maximum
    
    # Ensure reasonable bounds relationship
    if final_max <= final_min:
        final_max = int(final_min * 1.5)
    
    # Log the simplified 3-layer research-based calculation
    model_accuracy = predictor.get_model_accuracy(encoder)
    is_research_validated = predictor.is_research_validated(encoder)
    
    logger.vbr(f"📊 3-Layer bounds: {final_min}-{final_max}kbps")
    logger.vbr(f"   Layer 1 (Polynomial): R²={model_accuracy.get('r_squared', 0):.3f} "
              f"({'validated' if is_research_validated else 'estimated'})")
    logger.vbr(f"   Layer 2 (Resolution-dominant): Research-validated multipliers")
    logger.vbr(f"   Layer 3 (Cross-trial): Empirical learning from results")
    
    return final_min, final_max


def get_intelligent_bounds(infile: Path, target_vmaf: float, preset: str, 
                          bounds_history: Dict, source_bitrate_kbps: int,
                          trial_results: Optional[List[Dict]] = None,
                          vmaf_tolerance: float = 1.0,
                          global_bounds_reduction: Optional[Dict] = None,
                          encoder_type: str = "cpu") -> Tuple[int, int]:
    """Calculate intelligent bounds using previous results and content analysis."""
    
    # OPTIMIZATION #2: Dynamic Quality Gap Analysis - check for aggressive bounds reduction
    if trial_results and len(trial_results) >= 1:
        should_reduce, suggested_max, reason = analyze_quality_gap_dynamically(
            trial_results, target_vmaf, vmaf_tolerance)
        
        if should_reduce and suggested_max > 0:
            # Apply aggressive bounds reduction based on quality gap analysis
            adaptive_min, adaptive_reason = get_adaptive_lower_bounds(
                trial_results, 300, target_vmaf, vmaf_tolerance)
            
            logger.vbr(f"Dynamic bounds reduction: {adaptive_min}-{suggested_max}kbps ({reason})")
            
            # CRITICAL FIX: Store in global bounds reduction for cross-trial learning
            if global_bounds_reduction is not None:
                global_bounds_reduction['max'] = suggested_max
                global_bounds_reduction['min'] = adaptive_min
                global_bounds_reduction['reason'] = reason
            
            return adaptive_min, suggested_max
    
    # CRITICAL FIX: Check global bounds reduction first
    if global_bounds_reduction and 'max' in global_bounds_reduction:
        logger.vbr(f"Using global reduced bounds: {global_bounds_reduction['min']}-{global_bounds_reduction['max']}kbps "
                  f"({global_bounds_reduction['reason']})")
        return global_bounds_reduction['min'], global_bounds_reduction['max']
    
    # EFFICIENCY FIX: Learn from successful trials to narrow bounds immediately
    if trial_results:
        successful_trials = [r for r in trial_results if r.get('success')]
        if successful_trials:
            # Find the best result so far and center bounds around it
            best_trial = min(successful_trials, key=lambda x: x.get('bitrate', float('inf')))
            best_bitrate = best_trial.get('bitrate', 0)
            best_vmaf = best_trial.get('vmaf_score', 0)
            
            # If we found a good result, narrow the search dramatically
            if abs(best_vmaf - target_vmaf) <= vmaf_tolerance:
                # Use tight bounds around the known good result
                narrow_min = max(int(best_bitrate * 0.85), 1000)  # 15% below best
                narrow_max = int(best_bitrate * 1.15)  # 15% above best
                
                logger.vbr(f"Learning from Trial {len(successful_trials)}: Narrowing bounds to {narrow_min}-{narrow_max}kbps (±15% of {best_bitrate}kbps)")
                preset_key = f"{preset}_{target_vmaf}"
                bounds_history[preset_key] = (narrow_min, narrow_max)
                return narrow_min, narrow_max
    
    # Check if we have successful bounds for this preset from previous trials
    preset_key = f"{preset}_{target_vmaf}"
    if preset_key in bounds_history and (trial_results is None or len(trial_results) == 0):
        cached_min, cached_max = bounds_history[preset_key]
        logger.vbr(f"Using cached bounds for {preset}: {cached_min}-{cached_max}kbps")
        return cached_min, cached_max
    
    # Get video properties for smarter initial bounds
    try:
        width, height = get_video_dimensions(infile)
        fps_str = ffprobe_field(infile, "r_frame_rate") or "24"
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
    except:
        width, height, fps = 1920, 1080, 24.0
    
    # Calculate complexity factor
    pixel_rate = width * height * fps
    
    # OPTIMIZATION #3: Aggressive initial bounds calculation for modern content
    # Hardware vs CPU encoding have different efficiency characteristics
    
    # Hardware encoding efficiency factor (hardware is typically less efficient)
    hw_efficiency = 0.85 if encoder_type == "hardware" else 1.0
    
    # Preset-specific bounds calculation with encoder type optimization
    if preset == "fast":
        # Fast preset bounds
        if target_vmaf >= 95:
            base_min = max(int(source_bitrate_kbps * 0.45 * hw_efficiency), 1800)
            base_max = int(source_bitrate_kbps * 0.75 * hw_efficiency)
        elif target_vmaf >= 90:
            base_min = max(int(source_bitrate_kbps * 0.30 * hw_efficiency), 1200)
            base_max = int(source_bitrate_kbps * 0.55 * hw_efficiency)
        else:
            base_min = max(int(source_bitrate_kbps * 0.20 * hw_efficiency), 800)
            base_max = int(source_bitrate_kbps * 0.45 * hw_efficiency)
    elif preset == "medium":
        # Medium preset bounds
        if target_vmaf >= 95:
            base_min = max(int(source_bitrate_kbps * 0.35 * hw_efficiency), 1500)
            base_max = int(source_bitrate_kbps * 0.65 * hw_efficiency)
        elif target_vmaf >= 90:
            base_min = max(int(source_bitrate_kbps * 0.25 * hw_efficiency), 1000)
            base_max = int(source_bitrate_kbps * 0.50 * hw_efficiency)
        else:
            base_min = max(int(source_bitrate_kbps * 0.15 * hw_efficiency), 600)
            base_max = int(source_bitrate_kbps * 0.40 * hw_efficiency)
    else:  # slow
        # Slow preset bounds
        if target_vmaf >= 95:
            base_min = max(int(source_bitrate_kbps * 0.30 * hw_efficiency), 1200)
            base_max = int(source_bitrate_kbps * 0.60 * hw_efficiency)
        elif target_vmaf >= 90:
            base_min = max(int(source_bitrate_kbps * 0.20 * hw_efficiency), 800)
            base_max = int(source_bitrate_kbps * 0.45 * hw_efficiency)
        else:
            base_min = max(int(source_bitrate_kbps * 0.15 * hw_efficiency), 500)
            base_max = int(source_bitrate_kbps * 0.35 * hw_efficiency)
    
    # Apply adaptive lower bound expansion based on previous results
    if trial_results:
        adaptive_min, adaptation_reason = get_adaptive_lower_bounds(
            trial_results, base_min, target_vmaf, vmaf_tolerance)
        
        if adaptive_min != base_min:
            logger.vbr(f"Adaptive bounds: lowering minimum from {base_min} to {adaptive_min}kbps ({adaptation_reason})")
            base_min = adaptive_min
    
    # CRITICAL FIX: Hardware encoder safety bounds
    if encoder_type == "hardware":
        # Hardware encoders often fail at very low bitrates - enforce minimums
        base_min = max(base_min, 1500)  # Never go below 1.5Mbps for hardware
        
        # Also be more conservative with maximum to avoid GPU memory issues
        if base_max > source_bitrate_kbps:
            base_max = min(base_max, int(source_bitrate_kbps * 0.9))
            logger.vbr(f"Hardware encoder: capping max bounds to {base_max}kbps (90% of source)")
    
    # Adjust for resolution complexity
    if pixel_rate > 8000000:  # 4K+
        multiplier = 1.4
    elif pixel_rate > 2000000:  # 1080p+
        multiplier = 1.2
    elif pixel_rate > 500000:  # 720p+
        multiplier = 1.0
    else:  # Lower resolutions
        multiplier = 0.8
    
    final_min = int(base_min * multiplier)
    final_max = int(base_max * multiplier)
    
    # Cache these bounds for future use
    bounds_history[preset_key] = (final_min, final_max)
    
    logger.vbr(f"Intelligent bounds for {preset}: {final_min}-{final_max}kbps "
              f"(resolution: {width}x{height}, complexity: {multiplier:.1f}x)")
    
    return final_min, final_max


def should_terminate_early(trial_results: List[Dict], consecutive_identical: int, 
                          target_vmaf: float, tolerance: float) -> Tuple[bool, str]:
    """
    OPTIMIZATION #4: Enhanced early termination with Dynamic Quality Gap Analysis
    
    Determine if we should terminate the search early based on results patterns.
    """
    
    if len(trial_results) < 2:
        return False, ""
    
    # ENHANCEMENT #1: More aggressive identical results detection
    if consecutive_identical >= 2:  # Reduced from 3 to 2 for faster termination
        return True, f"Found {consecutive_identical} identical results - parameter variations have no impact"
    
    # ENHANCEMENT #2: Quality overshoot detection - terminate if consistently way above target
    recent_successful = [r for r in trial_results[-4:] if r.get('success')]
    if len(recent_successful) >= 3:
        quality_excesses = [r.get('vmaf_score', 0) - target_vmaf for r in recent_successful]
        min_excess = min(quality_excesses)
        avg_excess = sum(quality_excesses) / len(quality_excesses)
        
        # If even the worst recent result is 3+ points above target, we're overdoing it
        if min_excess >= 3.0:
            best_recent = min(recent_successful, key=lambda x: x['bitrate'])
            return True, f"Consistent overshoot: min +{min_excess:.1f} VMAF, avg +{avg_excess:.1f} - best recent {best_recent['bitrate']}kbps"
    
    # ENHANCEMENT #3: Convergence on multiple successful trials
    successful_trials = [r for r in trial_results if r.get('success') and 
                        abs(r.get('vmaf_score', 0) - target_vmaf) <= tolerance]
    
    if len(successful_trials) >= 2:
        # Compare the bitrates - if they're very close, we've converged
        bitrates = [r['bitrate'] for r in successful_trials]
        bitrate_range = max(bitrates) - min(bitrates)
        
        if bitrate_range <= 150:  # Slightly more tolerant range
            best_trial = min(successful_trials, key=lambda x: x['bitrate'])
            return True, f"Converged to {best_trial['bitrate']}kbps (±{bitrate_range//2}kbps) with {len(successful_trials)} trials hitting target"
    
    # ENHANCEMENT #4: VMAF plateau detection - enhanced sensitivity
    if len(trial_results) >= 3:
        recent_results = [r for r in trial_results[-3:] if r.get('success')]
        if len(recent_results) >= 3:
            vmafs = [r['vmaf_score'] for r in recent_results]
            vmaf_range = max(vmafs) - min(vmafs)
            
            # More sensitive plateau detection
            if vmaf_range <= 0.15:  # Reduced from 0.2 to 0.15
                return True, f"VMAF plateau: last 3 trials within ±{vmaf_range:.2f} VMAF"
    
    # ENHANCEMENT #5: Diminishing returns detection
    if len(trial_results) >= 4:
        # Check if last 3 improvements are very small
        recent_successful = [r for r in trial_results[-4:] if r.get('success')]
        if len(recent_successful) >= 3:
            # Sort by trial order (preserve order in list)
            bitrates = [r['bitrate'] for r in recent_successful]
            improvements = []
            
            for i in range(1, len(bitrates)):
                if bitrates[i] < bitrates[i-1]:  # Only count actual improvements
                    improvements.append(bitrates[i-1] - bitrates[i])
            
            # If last 2 improvements are both < 200kbps, diminishing returns
            if len(improvements) >= 2 and all(imp < 200 for imp in improvements[-2:]):
                avg_improvement = sum(improvements[-2:]) / 2
                return True, f"Diminishing returns: last improvements avg {avg_improvement:.0f}kbps"
    
    return False, ""


def _generate_optimization_summary(trial_results: List[Dict], best_result: Dict, 
                                  trials: int, quality_gap_reductions: int, 
                                  consecutive_identical: int) -> str:
    """Generate a summary of optimization effectiveness."""
    
    summary_parts = []
    
    # Count successful vs failed trials
    successful_trials = len([r for r in trial_results if r.get('success')])
    summary_parts.append(f"{successful_trials}/{trials} successful")
    
    # Report dynamic quality gap reductions
    if quality_gap_reductions > 0:
        summary_parts.append(f"{quality_gap_reductions} dynamic bounds reductions")
    
    # Report early termination effectiveness
    if consecutive_identical > 0:
        summary_parts.append(f"{consecutive_identical} redundant results avoided")
    
    # Calculate average VMAF excess
    if trial_results:
        vmaf_excesses = [r.get('vmaf_score', 0) - 92.0 for r in trial_results 
                        if r.get('success') and r.get('vmaf_score', 0) > 0]
        if vmaf_excesses:
            avg_excess = sum(vmaf_excesses) / len(vmaf_excesses)
            summary_parts.append(f"avg +{avg_excess:.1f} VMAF excess")
    
    # Report speed improvement potential
    if trials <= len(get_parameter_combinations("cpu", 92.0, "general")) // 2:
        summary_parts.append("50%+ trial reduction")
    
    return ", ".join(summary_parts) if summary_parts else "standard optimization"


def get_parameter_combinations(encoder_type: str, target_vmaf: float, 
                              content_type: str = "unknown") -> List[Tuple[str, int, int]]:
    """
    OPTIMIZATION #3: Content-adaptive parameter testing
    
    Get optimally ordered parameter combinations for testing based on quality target and encoder type.
    Returns combinations prioritized by efficiency and likelihood of success.
    """
    
    if encoder_type == "hardware":
        # Hardware encoding - start with most reliable combination first
        combinations = [
            # Most reliable combination first - if this works, often others are redundant
            ("medium", 3, 3),
            # Only add variations if medium preset shows promise
            ("fast", 3, 3), ("medium", 4, 3),
            ("fast", 4, 3), ("medium", 3, 4), 
            ("slow", 3, 3)
        ]
        
        # For high quality targets, prioritize slower presets
        if target_vmaf >= 96:
            combinations = [
                ("medium", 3, 3), ("slow", 3, 3),  # Most likely to succeed
                ("medium", 4, 3), ("slow", 4, 3),
                ("fast", 3, 3), ("medium", 3, 4)
            ]
            combinations.insert(-1, ("medium", 4, 4))
            
    else:  # CPU
        # CPU encoding - start with most likely to succeed, then expand
        if target_vmaf <= 90:
            # Lower quality targets - prioritize efficiency
            combinations = [
                ("medium", 3, 3),  # Start with most balanced
                ("fast", 3, 3), ("medium", 4, 3),
                ("fast", 4, 3), ("medium", 3, 4), ("slow", 3, 3)
            ]
        elif target_vmaf <= 94:
            # Medium quality targets - balanced progression
            combinations = [
                ("medium", 3, 3),  # Start with most reliable
                ("medium", 4, 3), ("slow", 3, 3),
                ("fast", 3, 3), ("medium", 3, 4), ("slow", 4, 3)
            ]
        else:
            # High quality targets - prioritize quality presets first
            combinations = [
                ("medium", 3, 3), ("slow", 3, 3),  # Most likely to hit target
                ("medium", 4, 3), ("slow", 4, 3), ("medium", 3, 4),
                ("slow", 3, 4), ("slow", 4, 4), ("slow", 6, 3)
            ]
    
    return combinations


logger = get_logger()


def build_vbr_encode_cmd(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                        max_bitrate: int, avg_bitrate: int, preset: str = "medium", 
                        bf: int = 3, refs: int = 3, preserve_hdr_metadata: bool = True) -> List[str]:
    """Build VBR encoding command with proper HDR detection using EncoderConfigBuilder."""
    
    # Use the sophisticated EncoderConfigBuilder which has proper HDR detection
    from .encoder_config import EncoderConfigBuilder
    
    builder = EncoderConfigBuilder()
    
    # Get video dimensions for proper encoding
    width, height = get_video_dimensions(infile)
    
    # Determine threading based on encoder type
    threads = 4 if encoder_type == "hardware" else None  # Let x265 auto-decide for CPU
    
    # Use the sophisticated VBR command builder which properly handles HDR detection
    cmd = builder.build_vbr_encode_cmd(
        str(infile), str(outfile), encoder, preset, max_bitrate,  # Use max_bitrate as the bitrate parameter
        bf, refs, width, height, threads=threads,
        preserve_hdr=preserve_hdr_metadata, debug=DEBUG
    )
    
    return cmd


def calculate_intelligent_vbr_bounds(infile: Path, target_vmaf: float, expand_factor: int = 0, 
                                    previous_results: Optional[List[Dict]] = None) -> tuple[int, int]:
    """Calculate intelligent VBR bitrate bounds with learning from previous results."""
    
    # Get source file bitrate
    try:
        result = run_command([
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=bit_rate", "-of", "csv=p=0", str(infile)
        ], timeout=30)
        
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
    
    logger.vbr_bounds(f"Source bitrate: {source_bitrate_kbps}kbps")
    
    # Get video properties for intelligent bounds
    try:
        width, height = get_video_dimensions(infile)
        fps_str = ffprobe_field(infile, "r_frame_rate") or "24"
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
    except:
        width, height, fps = 1920, 1080, 24.0
    
    # Resolution complexity factor
    pixel_rate = width * height * fps
    if pixel_rate > 8000000:  # 4K+
        resolution_factor = 1.4
    elif pixel_rate > 2000000:  # 1080p+
        resolution_factor = 1.0
    elif pixel_rate > 900000:   # 720p+
        resolution_factor = 0.7
    else:  # 480p and below
        resolution_factor = 0.5
    
    # Learn from previous results to improve bounds
    if previous_results:
        # Find successful results within target range
        successful_bitrates = []
        for result in previous_results:
            if (result.get('success') and 
                abs(result.get('vmaf_score', 0) - target_vmaf) <= 1.0):
                successful_bitrates.append(result.get('bitrate', 0))
        
        if successful_bitrates:
            # Use successful results to center our search
            avg_successful = sum(successful_bitrates) / len(successful_bitrates)
            target_center = int(avg_successful)
            logger.vbr_bounds(f"Learning from {len(successful_bitrates)} previous results, centering around {target_center}kbps")
        else:
            # No successful results yet, use quality-based estimation
            if target_vmaf >= 95.0:
                target_center = int(source_bitrate_kbps * 0.8 * resolution_factor)
            elif target_vmaf >= 90.0:
                target_center = int(source_bitrate_kbps * 0.6 * resolution_factor)
            elif target_vmaf >= 85.0:
                target_center = int(source_bitrate_kbps * 0.4 * resolution_factor)
            else:
                target_center = int(source_bitrate_kbps * 0.3 * resolution_factor)
    else:
        # First trial - use quality-based estimation
        if target_vmaf >= 95.0:
            target_center = int(source_bitrate_kbps * 0.8 * resolution_factor)
        elif target_vmaf >= 90.0:
            target_center = int(source_bitrate_kbps * 0.6 * resolution_factor)
        elif target_vmaf >= 85.0:
            target_center = int(source_bitrate_kbps * 0.4 * resolution_factor)
        else:
            target_center = int(source_bitrate_kbps * 0.3 * resolution_factor)
    
    # Progressive expansion based on expand_factor
    base_width = 0.15 if not previous_results else 0.10  # Narrower if we have learning data
    expansion_multiplier = 1.8 ** expand_factor
    current_width = base_width * expansion_multiplier
    
    search_range = int(target_center * current_width)
    
    # Smart lower bound with aggressive expansion capability
    if expand_factor == 0:
        # Conservative first attempt
        min_bound = max(target_center - search_range, int(source_bitrate_kbps * 0.1))
    elif expand_factor == 1:
        # More aggressive second attempt
        min_bound = max(target_center - search_range, int(source_bitrate_kbps * 0.05))
    else:
        # Very aggressive for subsequent attempts
        min_bound = max(target_center - search_range, 200)  # Floor at 200kbps
    
    max_bound = min(target_center + search_range, source_bitrate_kbps * 2)  # Never exceed 2x source
    
    logger.vbr_bounds(f"Calculated bounds: {min_bound}-{max_bound}kbps "
                     f"(center: {target_center}kbps, width: ±{current_width:.1%}, "
                     f"resolution: {width}x{height}, factor: {resolution_factor:.1f})")
    
    return min_bound, max_bound


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


def should_continue_optimization(trial_results: List[Dict], target_vmaf: float, 
                               tolerance: float, max_safety_limit: int = 20) -> Tuple[bool, str]:
    """
    Intelligent convergence detection - decides when optimization should stop.
    
    No arbitrary base limits - stops when it makes sense based on:
    - Target achievement with confidence
    - Convergence patterns (stable results)
    - Diminishing returns detection
    - Impossible target detection
    - Safety limit only to prevent infinite loops
    
    Returns (should_continue, reason)
    """
    
    if not trial_results:
        return True, "starting optimization"
    
    # Safety limit to prevent infinite search
    if len(trial_results) >= max_safety_limit:
        return False, f"safety limit reached ({max_safety_limit} trials)"
    
    successful_results = [r for r in trial_results if r.get('success')]
    
    # CASE 1: No successful results yet
    if not successful_results:
        if len(trial_results) <= 3:
            return True, "no success yet - continue exploring"
        elif len(trial_results) <= 6:
            # Check if we're getting closer
            vmaf_scores = [r.get('vmaf_score', 0) for r in trial_results if r.get('vmaf_score', 0) > 0]
            if vmaf_scores:
                best_attempt = max(vmaf_scores)
                gap_to_target = target_vmaf - best_attempt
                if gap_to_target <= 5.0:  # Within 5 VMAF points
                    return True, f"getting closer: best attempt {best_attempt:.1f} vs target {target_vmaf}"
            return True, "no success but continuing exploration"
        else:
            # After 6+ failed trials, likely impossible
            return False, f"target {target_vmaf} likely impossible after {len(trial_results)} failed attempts"
    
    # CASE 2: Analyze successful results for convergence
    vmaf_scores = [r['vmaf_score'] for r in successful_results]
    bitrates = [r['bitrate'] for r in successful_results]
    
    # Find best result (closest to target)
    best_result = min(successful_results, key=lambda r: abs(r['vmaf_score'] - target_vmaf))
    vmaf_error = abs(best_result['vmaf_score'] - target_vmaf)
    
    # CASE 3: Excellent result achieved
    if vmaf_error <= tolerance * 0.3:  # Within 30% of tolerance
        return False, f"excellent result: VMAF {best_result['vmaf_score']:.1f} (error: {vmaf_error:.2f})"
    
    # CASE 4: Target achieved within tolerance
    if vmaf_error <= tolerance:
        if len(successful_results) >= 2:
            # Check stability - do we have consistent good results?
            recent_good_results = [r for r in successful_results[-3:] 
                                 if abs(r['vmaf_score'] - target_vmaf) <= tolerance]
            if len(recent_good_results) >= 2:
                return False, f"stable target achievement: {len(recent_good_results)} recent results within tolerance"
        return True, "target achieved but checking stability"
    
    # CASE 5: Convergence detection (stable results)
    if len(successful_results) >= 3:
        recent_results = successful_results[-3:]
        recent_vmaf = [r['vmaf_score'] for r in recent_results]
        recent_bitrates = [r['bitrate'] for r in recent_results]
        
        vmaf_variance = max(recent_vmaf) - min(recent_vmaf)
        bitrate_variance = max(recent_bitrates) - min(recent_bitrates)
        
        # Very low variance = converged
        if vmaf_variance <= 0.3 and bitrate_variance <= 150:
            return False, f"converged: VMAF±{vmaf_variance/2:.2f}, bitrate±{bitrate_variance/2:.0f}kbps"
    
    # CASE 6: Diminishing returns detection
    if len(successful_results) >= 4:
        # Check if recent results are not improving
        errors = [abs(r['vmaf_score'] - target_vmaf) for r in successful_results]
        if len(errors) >= 4:
            early_error = min(errors[:2])  # Best of first 2
            recent_error = min(errors[-2:])  # Best of last 2
            
            # If no significant improvement in recent trials
            if recent_error >= early_error * 0.9:  # Less than 10% improvement
                return False, f"diminishing returns: error {early_error:.2f}→{recent_error:.2f}"
    
    # CASE 7: Continue optimization
    if len(trial_results) <= 6:
        return True, f"continuing search: best error {vmaf_error:.2f} > tolerance {tolerance}"
    else:
        # After many trials, be more selective about continuing
        if vmaf_error <= tolerance * 1.5:  # Within 150% of tolerance
            return True, f"close to target: error {vmaf_error:.2f} (within 1.5x tolerance)"
        else:
            return False, f"far from target after {len(trial_results)} trials: error {vmaf_error:.2f}"


def optimize_encoder_settings_vbr(infile: Path, encoder: str, encoder_type: str, 
                                  target_vmaf: float, vmaf_tolerance: float,
                                  clip_positions: list[int], clip_duration: int, 
                                  max_safety_limit: int = 20) -> dict:
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
    logger.vbr(f"Optimizing {infile.name} for VMAF {target_vmaf:.1f}±{vmaf_tolerance:.1f}")
    
    # Warn about hardware encoder inefficiency for pre-encoding
    warn_hardware_encoder_inefficiency(encoder_type, "VBR optimization")
    
    # Get source file info and bitrate for intelligent bounds
    try:
        source_size_gb = infile.stat().st_size / (1024**3)
        logger.vbr(f"Source file: {source_size_gb:.2f}GB")
        
        # Get source bitrate for intelligent bounds calculation
        result = run_command([
            "ffprobe", "-v", "error", "-select_streams", "v:0", 
            "-show_entries", "stream=bit_rate", "-of", "csv=p=0", str(infile)
        ], timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            source_bitrate_kbps = int(result.stdout.strip()) // 1000
        else:
            duration = get_duration_sec(infile)
            if duration and duration > 0:
                file_size_bits = infile.stat().st_size * 8
                source_bitrate_kbps = int(file_size_bits / duration / 1000)
            else:
                source_bitrate_kbps = 8000
    except:
        source_size_gb = 0
        source_bitrate_kbps = 8000
    
    # OPTIMIZATION #3: Content-adaptive parameter selection
    # Detect content characteristics for optimized parameter combinations
    content_type = "general"  # Simplified - no specific content detection
    
    # Get intelligent parameter combinations with quality-based adaptation
    param_combinations = get_parameter_combinations(encoder_type, target_vmaf, content_type)
    
    # Track results across trials for intelligent optimization
    trial_results = []  # Store all trial results for cross-trial analysis
    bounds_history = {}  # Cache successful bounds by preset for reuse
    preset_efficiency = {}  # Track preset → bitrate efficiency
    
    # CRITICAL FIX: Global bounds manager to propagate learning across trials
    global_bounds_reduction = {}  # Track the most aggressive bounds discovered
    
    best_result = None
    lowest_bitrate = float('inf')
    trials = 0
    abandoned = False
    
    # OPTIMIZATION #4: Enhanced early termination tracking
    consecutive_identical = 0
    last_vmaf_result = None
    last_bitrate_result = None
    quality_gap_reductions = 0  # Track how many times we've applied dynamic analysis
    
    # EFFICIENCY FIX: Use shorter clips for initial exploration, full clips for final verification
    exploration_duration = max(30, clip_duration // 2)  # Use half duration for speed
    
    # Initialize shared test cache to prevent redundant testing
    test_cache = {}
    
    # EFFICIENCY FIX: Extract clips once and reuse across all trials
    logger.vbr(f"Pre-extracting clips for reuse across all trials (exploration: {exploration_duration}s)...")
    shared_clips, extraction_error = extract_clips_parallel(infile, clip_positions, exploration_duration)
    
    if extraction_error:
        logger.vbr_error(f"Clip extraction failed: {extraction_error}")
        return {'success': False, 'error': extraction_error}
    
    if not shared_clips:
        logger.vbr_error("No clips were successfully extracted")
        return {'success': False, 'error': 'no clips extracted'}
    
    logger.vbr(f"Successfully extracted {len(shared_clips)} clips for reuse")
    
    # PURE CONVERGENCE SYSTEM: No predetermined limits, stop when convergence achieved
    logger.vbr(f"📊 Pure convergence system: Will optimize until natural convergence")
    logger.vbr(f"Safety limit: {max_safety_limit} trials maximum")
    
    logger.vbr(f"Phase 1: Quality-adaptive testing with {len(param_combinations)} combinations")
    logger.vbr(f"Using encoder: {encoder} ({encoder_type}) for VBR optimization")
    logger.vbr(f"Maximum {len(param_combinations)} parameter combinations (convergence-based)")
    
    # Create overall progress bar for parameter testing (dynamic based on safety limit)
    with tqdm(total=max_safety_limit, desc="[VBR-CONVERGENCE]", 
             unit="trial", leave=False, position=0) as coord_pbar:
        
        for preset, bf, refs in param_combinations:
            # PURE CONVERGENCE CHECK: Decide if optimization should continue
            should_continue, convergence_reason = should_continue_optimization(
                trial_results, target_vmaf, vmaf_tolerance)
            
            if not should_continue or abandoned:
                if not should_continue:
                    logger.vbr(f"🎯 Convergence achieved: {convergence_reason}")
                break
            
            # Safety check against infinite loops
            if trials >= max_safety_limit:
                logger.vbr(f"⚠️ Safety limit reached ({max_safety_limit} trials)")
                break
                
            trials += 1
            coord_pbar.set_description(f"[VBR-CONVERGENCE] Trial {trials}: {preset}, bf={bf}, refs={refs}")
            coord_pbar.update(1)
            
            # Check for early termination before expensive operations
            should_terminate, termination_reason = should_terminate_early(
                trial_results, consecutive_identical, target_vmaf, vmaf_tolerance)
            
            if should_terminate:
                logger.vbr(f"Early termination: {termination_reason}")
                abandoned = True
                break
            
            # RESEARCH-BASED BOUNDS: Use polynomial models when available
            try:
                min_bitrate, max_bitrate = get_research_based_intelligent_bounds(
                    infile, encoder, encoder_type, preset, target_vmaf, 
                    vmaf_tolerance, trial_results)
                logger.vbr("Using research-validated polynomial bounds")
            except Exception as e:
                logger.debug(f"Research bounds failed, using traditional: {e}")
                # Fallback to traditional intelligent bounds
                min_bitrate, max_bitrate = get_intelligent_bounds(
                    infile, target_vmaf, preset, bounds_history, source_bitrate_kbps, 
                    trial_results, vmaf_tolerance, global_bounds_reduction, encoder_type)
            
            logger.vbr(f"Trial {trials}: {preset}, bf={bf}, refs={refs}")
            
            result = _test_parameter_combination(
                infile, encoder, encoder_type, target_vmaf, vmaf_tolerance,
                clip_positions, clip_duration, preset, bf, refs, test_cache,
                bounds_history, source_bitrate_kbps, trial_results, global_bounds_reduction, encoder_type,
                shared_clips  # Pass pre-extracted clips
            )
            
            # Store result for cross-trial analysis
            if result:
                trial_results.append(result.copy())
                
                # EFFICIENCY FIX: Immediate success detection - stop if first trial finds great result
                if (len(trial_results) == 1 and result.get('success') and 
                    result.get('vmaf_score', 0) >= target_vmaf - vmaf_tolerance and
                    result.get('bitrate', 0) < source_bitrate_kbps * 0.6):  # Good compression
                    
                    logger.vbr(f"First trial success: {result.get('bitrate')}kbps, VMAF {result.get('vmaf_score', 0):.2f}")
                    logger.vbr(f"Excellent result found early - skipping remaining parameter combinations")
                    best_result = result.copy()
                    abandoned = True
                    break
                
                # OPTIMIZATION #2&4: Dynamic Quality Gap Analysis after each trial
                if len(trial_results) >= 1 and quality_gap_reductions < 3:  # Start analysis after first trial
                    should_reduce, suggested_max, gap_reason = analyze_quality_gap_dynamically(
                        trial_results, target_vmaf, vmaf_tolerance)
                    
                    if should_reduce:
                        if suggested_max == 0:  # Identical results detected
                            logger.vbr(f"Dynamic analysis: {gap_reason} - terminating early")
                            abandoned = True
                            break
                        else:  # Quality overshoot detected
                            quality_gap_reductions += 1
                            logger.vbr(f"Dynamic analysis #{quality_gap_reductions}: {gap_reason}")
                            logger.vbr(f"Applying aggressive bounds reduction for ALL remaining trials")
                            
                            # CRITICAL FIX: Update bounds history for ALL presets immediately
                            for future_preset in ["fast", "medium", "slow"]:
                                future_key = f"{future_preset}_{target_vmaf}"
                                # Calculate adaptive lower bound for this preset
                                adaptive_min, _ = get_adaptive_lower_bounds(
                                    trial_results, 300, target_vmaf, vmaf_tolerance)
                                
                                # Use the suggested max from dynamic analysis
                                new_max = max(suggested_max, adaptive_min + 500)  # Ensure reasonable range
                                
                                # Force update bounds history
                                old_bounds = bounds_history.get(future_key, (adaptive_min, new_max))
                                bounds_history[future_key] = (adaptive_min, new_max)
                                
                                logger.vbr(f"Updated {future_preset} bounds: {adaptive_min}-{new_max}kbps "
                                         f"(was {old_bounds[0] if old_bounds else 'N/A'}-{old_bounds[1] if old_bounds else 'N/A'}kbps)")
                            
                            # Store in global bounds reduction for immediate use
                            global_bounds_reduction['max'] = suggested_max
                            global_bounds_reduction['min'] = adaptive_min
                            global_bounds_reduction['reason'] = gap_reason
                
                # Enhanced identical results detection (optimization #4)  
                if (last_vmaf_result is not None and last_bitrate_result is not None and
                    result.get('success') and abs(result.get('vmaf_score', 0) - last_vmaf_result) < 0.1 and
                    abs(result.get('bitrate', 0) - last_bitrate_result) < 50):  # Slightly less sensitive for real-world variance
                    consecutive_identical += 1
                    logger.vbr(f"Near-identical result #{consecutive_identical}: {result.get('bitrate')}kbps, VMAF {result.get('vmaf_score', 0):.2f}")
                else:
                    consecutive_identical = 0
                
                # CRITICAL FIX: Cross-trial redundancy detection
                if result.get('success') and len(trial_results) >= 2:
                    # Check if this result is very similar to ANY previous result
                    current_bitrate = result.get('bitrate', 0)
                    current_vmaf = result.get('vmaf_score', 0)
                    
                    for prev_result in trial_results[:-1]:  # Exclude current result
                        if (prev_result.get('success') and 
                            abs(prev_result.get('bitrate', 0) - current_bitrate) < 100 and
                            abs(prev_result.get('vmaf_score', 0) - current_vmaf) < 0.3):
                            
                            logger.vbr(f"Cross-trial redundancy detected: {current_bitrate}kbps vs {prev_result.get('bitrate')}kbps")
                            logger.vbr(f"Terminating optimization - parameter variations show minimal impact")
                            abandoned = True
                            break
                    
                    if abandoned:
                        break
                
                # CRITICAL FIX: Smart early success detection
                if result.get('success'):
                    vmaf_score = result.get('vmaf_score', 0)
                    bitrate = result.get('bitrate', 0)
                    vmaf_margin = vmaf_score - target_vmaf
                    
                    # Excellent result detection - stop if we hit a sweet spot
                    if (abs(vmaf_margin) <= vmaf_tolerance * 0.5 and  # Very close to target
                        bitrate < source_bitrate_kbps * 0.4):  # Great compression
                        
                        logger.vbr(f"Excellent result found: {bitrate}kbps, VMAF {vmaf_score:.2f} "
                                  f"(margin: {vmaf_margin:+.1f}, compression: {bitrate/source_bitrate_kbps:.1%})")
                        logger.vbr(f"Stopping optimization - quality target achieved with excellent compression")
                        
                        best_result = result.copy()
                        abandoned = True
                        break
                
                # Update tracking for next iteration
                if result.get('success'):
                    last_vmaf_result = result.get('vmaf_score')
                    last_bitrate_result = result.get('bitrate')
                    
                    # Cache successful bounds for this preset (optimization #3)
                    if preset not in preset_efficiency:
                        preset_efficiency[preset] = []
                    preset_efficiency[preset].append({
                        'bitrate': result['bitrate'],
                        'vmaf': result['vmaf_score'],
                        'bf': bf,
                        'refs': refs
                    })
            
            # RESEARCH INTEGRATION: Record successful measurements for model refinement
            if result.get('success'):
                try:
                    predictor = get_quality_rate_predictor()
                    predictor.add_measurement(
                        encoder=encoder,
                        preset=preset,
                        bitrate=result['bitrate'],
                        vmaf=result['vmaf_score']
                    )
                except Exception as e:
                    logger.debug(f"Failed to record measurement for model refinement: {e}")
            
            # Update progress bar
            coord_pbar.update(1)
            
            # Check early termination after processing result
            should_terminate, termination_reason = should_terminate_early(
                trial_results, consecutive_identical, target_vmaf, vmaf_tolerance)
            
            if should_terminate:
                logger.vbr(f"Early termination: {termination_reason}")
                abandoned = True
                break
            
            if result.get('abandoned'):
                logger.vbr_error("File cannot reach target quality")
                abandoned = True
                break
                
                if result.get('abandoned'):
                    print(f"[VBR-ABANDON] File cannot reach target quality")
                    abandoned = True
                    break
                
                # Handle preset efficiency detection
                if result.get('preset_insufficient'):
                    quality_gap = result.get('quality_gap', 0)
                    print(f"[VBR-SKIP] {preset} preset insufficient ({quality_gap:.1f} points below target) - trying faster preset")
                    break  # Break out of bf/refs loops for this preset
                
                if result.get('preset_overkill'):
                    excess = result.get('excess_quality', 0)
                    print(f"[VBR-SKIP] {preset} preset delivers {excess:.1f} points excess quality - moving to more efficient preset")
                    break  # Break out of bf/refs loops for this preset
                
                if result and result.get('success'):
                    # Calculate improvement metrics
                    previous_best = lowest_bitrate if best_result else float('inf')
                    improvement_gain = previous_best - result['bitrate']
                    compression_ratio = result['bitrate'] / source_bitrate_kbps
                    vmaf_margin = result['vmaf_score'] - target_vmaf
                    
                    # Always save the first successful result, or better compression
                    if result['bitrate'] < lowest_bitrate or best_result is None:
                        lowest_bitrate = result['bitrate']
                        best_result = result.copy()
                        
                        # Store preset result for rollback analysis
                        preset_results.append({
                            'preset': preset,
                            'bitrate': result['bitrate'],
                            'vmaf_score': result['vmaf_score'],
                            'bf': bf,
                            'refs': refs
                        })
                        
                        # Calculate space savings projection
                        try:
                            source_size = infile.stat().st_size
                            duration = get_duration_sec(infile)
                            if duration > 0:
                                projected_size = int((result['bitrate'] * 1000 * duration) / 8)
                                relative_size_pct = (projected_size / source_size) * 100
                                space_savings_gb = (source_size - projected_size) / (1024**3)
                                
                                print(f"[VBR] New best: {result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f} "
                                      f"(↓{improvement_gain:.0f}kbps, {compression_ratio:.1%} compression) "
                                      f"→ ~{relative_size_pct:.1f}% of original (~{space_savings_gb:.2f}GB saved)")
                            else:
                                print(f"[VBR] New best: {result['bitrate']}kbps, VMAF {result['vmaf_score']:.2f} "
                                      f"(↓{improvement_gain:.0f}kbps, {compression_ratio:.1%} compression)")
                        except:
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
                    
                    # 6. Smart preset progression - if slow/medium failing, skip to fast
                    if (preset in ["slow", "medium"] and 
                        vmaf_margin < -vmaf_tolerance * 2 and  # Significantly below target
                        result['bitrate'] > source_bitrate_kbps * 0.8):  # Still high bitrate
                        print(f"[VBR-EARLY-EXIT] {preset} preset insufficient - skipping to faster preset")
                        # Force break out of current preset loops to try fast
                        break
                    
                    # 7. Preset efficiency analysis - detect when we should rollback to previous preset
                    if len(preset_results) >= 2:
                        current_preset_result = preset_results[-1]  # Current result
                        previous_preset_result = preset_results[-2]  # Previous (slower) preset result
                        
                        # Calculate file size difference
                        size_ratio = current_preset_result['bitrate'] / previous_preset_result['bitrate']
                        quality_diff = abs(current_preset_result['vmaf_score'] - previous_preset_result['vmaf_score'])
                        
                        # If current preset gives >105% of previous file size with similar quality, rollback
                        if size_ratio > 1.05 and quality_diff < vmaf_tolerance:
                            print(f"[VBR-ROLLBACK] {preset} preset gives {size_ratio:.1%} size vs {preset_results[-2]['preset']} "
                                  f"with similar quality (Δ{quality_diff:.2f}) - rolling back to {preset_results[-2]['preset']}")
                            # Restore previous result as best
                            best_result = {
                                'success': True,
                                'bitrate': previous_preset_result['bitrate'],
                                'preset': previous_preset_result['preset'],
                                'bf': previous_preset_result['bf'],
                                'refs': previous_preset_result['refs'],
                                'vmaf_score': previous_preset_result['vmaf_score'],
                                'filesize': int((previous_preset_result['bitrate'] * 1000 * get_duration_sec(infile)) / 8) if get_duration_sec(infile) > 0 else 0
                            }
                            lowest_bitrate = previous_preset_result['bitrate']
                            break  # Exit optimization, we found our sweet spot
                    
                    # Early success - check if we should continue optimizing (existing logic)
                    if result['bitrate'] < 2000:  # Excellent result
                        logger.vbr_early_exit("Excellent bitrate found - skipping further trials")
                        break
                    
                    # Update coordinate descent progress
                    coord_pbar.update(1)
                    coord_pbar.set_postfix({
                        "best": f"{best_result['bitrate']}kbps" if best_result else "none",
                        "VMAF": f"{best_result['vmaf_score']:.1f}" if best_result else "0.0"
                    })

    # IMPROVED RESULT SELECTION: Choose the best result from all successful trials
    successful_results = [r for r in trial_results if r.get('success') and r.get('vmaf_score', 0) >= target_vmaf - vmaf_tolerance]
    
    if successful_results:
        # Sort by bitrate (lowest first) among results that meet minimum quality
        successful_results.sort(key=lambda r: r['bitrate'])
        
        # Select the lowest bitrate result that meets quality requirements
        best_candidate = successful_results[0]
        
        # Override best_result if we found a better candidate through this analysis
        if (not best_result or 
            (best_candidate['bitrate'] < best_result.get('bitrate', float('inf')) and 
             best_candidate['vmaf_score'] >= target_vmaf - vmaf_tolerance)):
            best_result = best_candidate.copy()
            logger.vbr(f"Selected optimal result: {best_result['bitrate']}kbps, VMAF {best_result['vmaf_score']:.2f} "
                      f"(lowest bitrate among {len(successful_results)} successful candidates)")

    if best_result:
        # OPTIMIZATION SUMMARY: Report on effectiveness of new optimizations
        optimization_summary = _generate_optimization_summary(
            trial_results, best_result, trials, quality_gap_reductions, consecutive_identical)
        
        # Calculate final efficiency metrics
        try:
            duration = get_duration_sec(infile)
            if duration > 0:
                final_size = int((best_result['bitrate'] * 1000 * duration) / 8)
                final_size_gb = final_size / (1024**3)
                if source_size_gb > 0:
                    relative_size_pct = (final_size / infile.stat().st_size) * 100
                    space_saved_gb = (infile.stat().st_size - final_size) / (1024**3)
                    
                    print(f"[VBR] ✓ Optimal: {best_result['bitrate']}kbps, "
                          f"VMAF {best_result['vmaf_score']:.2f}, "
                          f"{best_result['preset']}, bf={best_result['bf']}, refs={best_result['refs']} "
                          f"({trials} trials)")
                    print(f"[VBR] ✓ Efficiency: {final_size_gb:.2f}GB → {relative_size_pct:.1f}% of original ({space_saved_gb:.2f}GB saved)")
                    print(f"[VBR] ✓ Optimizations: {optimization_summary}")
                else:
                    print(f"[VBR] ✓ Optimal: {best_result['bitrate']}kbps, "
                          f"VMAF {best_result['vmaf_score']:.2f}, "
                          f"{best_result['preset']}, bf={best_result['bf']}, refs={best_result['refs']} "
                          f"({trials} trials)")
                    print(f"[VBR] ✓ Optimizations: {optimization_summary}")
            else:
                print(f"[VBR] ✓ Optimal: {best_result['bitrate']}kbps, "
                      f"VMAF {best_result['vmaf_score']:.2f}, "
                      f"{best_result['preset']}, bf={best_result['bf']}, refs={best_result['refs']} "
                      f"({trials} trials)")
                print(f"[VBR] ✓ Optimizations: {optimization_summary}")
        except:
            print(f"[VBR] ✓ Optimal: {best_result['bitrate']}kbps, "
                  f"VMAF {best_result['vmaf_score']:.2f}, "
                  f"{best_result['preset']}, bf={best_result['bf']}, refs={best_result['refs']} "
                  f"({trials} trials)")
            
        # EFFICIENCY FIX: Cleanup shared clips
        if not DEBUG and shared_clips:
            for clip in shared_clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
            
        return best_result
    else:
        print(f"[VBR] ✗ Failed to find suitable settings within {trials} trials")
        
        # EFFICIENCY FIX: Cleanup shared clips on failure too
        if not DEBUG and shared_clips:
            for clip in shared_clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
                    
        return {'success': False}


def _test_parameter_combination(infile: Path, encoder: str, encoder_type: str,
                               target_vmaf: float, vmaf_tolerance: float,
                               clip_positions: list[int], clip_duration: int,
                               preset: str, bf: int, refs: int,
                               test_cache: dict | None = None,
                               bounds_history: dict | None = None,
                               source_bitrate_kbps: int = 8000,
                               trial_results: Optional[List[Dict]] = None,
                               global_bounds_reduction: Optional[Dict] = None,
                               encoder_type_param: str = "cpu",
                               shared_clips: Optional[List[Path]] = None) -> dict:
    """
    OPTIMIZATION #2&3: Enhanced parameter testing with dynamic bounds and progressive expansion.
    """
    # Initialize cache if not provided
    if test_cache is None:
        test_cache = {}
    if bounds_history is None:
        bounds_history = {}
        
    expand_factor = 0
    max_expansions = 2  # Reduced from 3 - be more decisive
    
    # Pre-flight dynamic analysis check
    if trial_results and len(trial_results) >= 2:
        should_reduce, suggested_max, gap_reason = analyze_quality_gap_dynamically(
            trial_results, target_vmaf, vmaf_tolerance)
        
        if should_reduce and suggested_max == 0:  # Identical results pattern
            logger.vbr(f"Pre-flight check: {gap_reason} - skipping parameter combination")
            return {'success': False, 'early_skip': True, 'reason': gap_reason}
    
    while expand_factor <= max_expansions:
        # Use intelligent bounds calculation with dynamic analysis and global learning
        min_bitrate, max_bitrate = get_intelligent_bounds(
            infile, target_vmaf, preset, bounds_history, source_bitrate_kbps, 
            trial_results, vmaf_tolerance, global_bounds_reduction, encoder_type
        )
        
        # Apply expansion factor if needed
        if expand_factor > 0:
            expansion_factor = 1.6 ** expand_factor  # Less aggressive expansion
            min_bitrate = max(int(min_bitrate / expansion_factor), 100)
            max_bitrate = min(int(max_bitrate * expansion_factor), source_bitrate_kbps * 1.5)  # Cap at 1.5x source
            
            logger.vbr(f"Expanding bounds #{expand_factor}: {min_bitrate}-{max_bitrate}kbps (expansion: {expansion_factor:.1f}x)")
        
        # Bisection search on bitrate for this parameter combination
        result = _bisect_bitrate(
            infile, encoder, encoder_type, 
            target_vmaf, vmaf_tolerance, 
            clip_positions, clip_duration,
            int(min_bitrate), int(max_bitrate),
            preset, bf, refs,
            expand_factor, test_cache=test_cache, shared_clips=shared_clips
        )
        
        # Enhanced result analysis
        if result and result['success']:
            # Check if result shows consistent overshoot pattern
            vmaf_excess = result.get('vmaf_score', 0) - target_vmaf
            if vmaf_excess > vmaf_tolerance * 3:  # Significant overshoot
                logger.vbr(f"Significant overshoot detected: +{vmaf_excess:.1f} VMAF - this will inform future bounds")
            
            return result
        
        # If we hit bounds, try expanding
        if result and result.get('bounds_hit'):
            # Check if file was abandoned due to impossible quality target
            if result.get('abandoned'):
                return result  # Return abandonment result
            
            expand_factor += 1
            logger.vbr(f"Bounds hit, expanding search range...")
        else:
            # Some other failure, don't expand further
            break
    
    # Return last result even if not successful
    return result or {'success': False}


def calculate_adaptive_iterations(min_br: int, max_br: int, target_vmaf: float, 
                                vmaf_tolerance: float, base_iterations: int = 8) -> Tuple[int, str]:
    """
    Calculate adaptive number of bisection iterations based on search space characteristics.
    
    Returns (adaptive_iterations, reasoning)
    
    Strategy:
    - Wide bitrate range → More iterations (need more precision)
    - Narrow range → Fewer iterations (quick convergence)
    - High precision target → More iterations (tight tolerance)
    - Low precision target → Fewer iterations (loose tolerance)
    """
    
    bitrate_range = max_br - min_br
    
    # Range-based adaptation
    if bitrate_range <= 1000:  # Very narrow range (≤1Mbps)
        range_factor = 0.6  # Reduce iterations significantly
        range_desc = "narrow range"
    elif bitrate_range <= 3000:  # Moderate range (≤3Mbps)
        range_factor = 0.8  # Slightly reduce
        range_desc = "moderate range"
    elif bitrate_range >= 10000:  # Very wide range (≥10Mbps)
        range_factor = 1.4  # Increase iterations
        range_desc = "wide range"
    else:  # Standard range (3-10Mbps)
        range_factor = 1.0  # Use base
        range_desc = "standard range"
    
    # Precision-based adaptation
    if vmaf_tolerance <= 0.5:  # High precision
        precision_factor = 1.3  # More iterations needed
        precision_desc = "high precision"
    elif vmaf_tolerance >= 2.0:  # Low precision
        precision_factor = 0.7  # Fewer iterations sufficient
        precision_desc = "low precision"
    else:  # Standard precision
        precision_factor = 1.0
        precision_desc = "standard precision"
    
    # Combined adaptation
    combined_factor = range_factor * precision_factor
    adaptive_iterations = max(3, min(15, int(base_iterations * combined_factor)))
    
    reasoning = f"{range_desc} ({bitrate_range}kbps), {precision_desc} (±{vmaf_tolerance})"
    
    return adaptive_iterations, reasoning


def _bisect_bitrate(infile: Path, encoder: str, encoder_type: str,
                   target_vmaf: float, vmaf_tolerance: float,
                   clip_positions: list[int], clip_duration: int,
                   min_br: int, max_br: int,
                   preset: str, bf: int, refs: int,
                   expand_factor: int = 0,
                   base_iterations: int = 8,
                   test_cache: dict | None = None,
                   shared_clips: Optional[List[Path]] = None) -> dict:
    """Top-down bisection search to find minimum bitrate achieving target VMAF."""
    
    # Initialize cache if not provided
    if test_cache is None:
        test_cache = {}
    
    # Extract clips for testing
    try:
        # EFFICIENCY FIX: Use shared clips if available, otherwise extract new ones
        if shared_clips:
            clips = shared_clips
            logger.vbr(f"Using pre-extracted clips ({len(clips)} clips)")
        else:
            # Fallback: extract clips for this trial only
            clips, extraction_error = extract_clips_parallel(infile, clip_positions, clip_duration)
            
            if extraction_error:
                logger.vbr_error(f"Clip extraction failed: {extraction_error}")
                return {'success': False, 'error': extraction_error}
            
            if not clips:
                logger.vbr_error("No clips were successfully extracted")
                return {'success': False, 'error': 'no clips extracted'}
        
        def test_bitrate_func(bitrate_kbps: int) -> float | None:
            """Test a bitrate on all clips and return average VMAF"""
            # Create cache key based on all parameters that affect encoding
            cache_key = (bitrate_kbps, preset, bf, refs, encoder, encoder_type)
            
            # Check cache first
            if cache_key in test_cache:
                cached_result = test_cache[cache_key]
                logger.vbr_cache(f"{bitrate_kbps}kbps: Using cached VMAF {cached_result:.2f}")
                return cached_result
            
            vmaf_scores = []
            
            # Create progress bar for clip testing at this bitrate
            with tqdm(total=len(clips), desc=f"[VBR-TEST] {bitrate_kbps}kbps", 
                     unit="clip", leave=False, position=1) as pbar:
                
                for i, clip in enumerate(clips):
                    encoded_clip = clip.with_name(f"{clip.stem}.vbr_test_{bitrate_kbps}kbps{clip.suffix}")
                    TEMP_FILES.add(str(encoded_clip))
                    
                    pbar.set_description(f"[VBR-TEST] {bitrate_kbps}kbps clip {i+1}/{len(clips)}")
                    
                    # Build VBR encode command with current parameters
                    encode_cmd = build_vbr_encode_cmd(clip, encoded_clip, encoder, encoder_type, 
                                                    bitrate_kbps, int(bitrate_kbps * 0.8), 
                                                    preset, bf, refs, preserve_hdr_metadata=True)
                    
                    if not DEBUG:
                        encode_cmd.extend(["-loglevel", "error"])
                    
                    if DEBUG:
                        logger.debug(f"Encode command: {' '.join(shlex.quote(c) for c in encode_cmd)}")
                    
                    # Get clip duration for progress reporting
                    clip_duration = get_video_duration(clip)
                    
                    # Encoding step with progress and enhanced error handling
                    pbar.set_postfix({"step": "encoding"})
                    result = run_ffmpeg_with_progress(encode_cmd, clip_duration, 
                                                    f"VBR-ENC {bitrate_kbps}kbps")
                    
                    # Enhanced error checking for hardware encoders
                    if result.returncode != 0:
                        error_msg = result.stderr if result.stderr else "Unknown encoding error"
                        logger.vbr_error(f"Encoding failed for clip {i+1} at {bitrate_kbps}kbps: {error_msg}")
                        
                        # For hardware encoders, log specific common issues
                        if encoder_type == "hardware" and "nvenc" in encoder.lower():
                            if "out of memory" in error_msg.lower():
                                logger.vbr_error(f"GPU memory issue - try lower bitrate or restart GPU")
                            elif "unsupported" in error_msg.lower() or "not found" in error_msg.lower():
                                logger.vbr_error(f"Hardware encoder compatibility issue - consider CPU fallback")
                        
                        pbar.update(1)
                        continue
                    
                    if not encoded_clip.exists():
                        logger.vbr_error(f"Encoded clip {i+1} not created at {bitrate_kbps}kbps")
                        pbar.update(1)
                        continue
                    
                    # Check if encoded file is valid and has content
                    try:
                        encoded_size = encoded_clip.stat().st_size
                        if encoded_size < 1024:  # Less than 1KB indicates failure
                            logger.vbr_error(f"Encoded clip {i+1} too small ({encoded_size} bytes) at {bitrate_kbps}kbps")
                            pbar.update(1)
                            continue
                    except:
                        logger.vbr_error(f"Cannot check encoded clip {i+1} size at {bitrate_kbps}kbps")
                        pbar.update(1)
                        continue
                    
                    # VMAF calculation step with enhanced error handling
                    pbar.set_postfix({"step": "VMAF"})
                    
                    try:
                        vmaf_score = compute_vmaf_score(clip, encoded_clip, n_threads=8)
                        if vmaf_score is not None and vmaf_score > 0:
                            vmaf_scores.append(vmaf_score)
                            pbar.set_postfix({"step": "done", "VMAF": f"{vmaf_score:.1f}"})
                            if DEBUG:
                                logger.debug(f"Clip {i+1}: {bitrate_kbps}kbps -> VMAF {vmaf_score:.2f}")
                        else:
                            logger.vbr_error(f"Invalid VMAF score for clip {i+1}: {vmaf_score}")
                    except Exception as e:
                        logger.vbr_error(f"VMAF calculation exception for clip {i+1}: {str(e)}")
                        # For hardware encoders, this might indicate codec compatibility issues
                        if encoder_type == "hardware":
                            logger.vbr_error(f"Hardware encoder may have produced incompatible output for VMAF analysis")
                    
                    # Cleanup encoded clip
                    if not DEBUG and encoded_clip.exists():
                        try:
                            encoded_clip.unlink()
                            TEMP_FILES.discard(str(encoded_clip))
                        except:
                            pass
                    
                    pbar.update(1)
            
            avg_vmaf = sum(vmaf_scores) / len(vmaf_scores) if vmaf_scores else None
            
            # Handle case where no clips could be processed - CRITICAL for hardware encoders
            if avg_vmaf is None or len(vmaf_scores) == 0:
                logger.vbr_error(f"{bitrate_kbps}kbps: No valid VMAF scores from {len(clips)} clips")
                
                # For hardware encoders, this often indicates compatibility issues
                if encoder_type == "hardware":
                    logger.vbr_error(f"Hardware encoder ({encoder}) may have compatibility issues at {bitrate_kbps}kbps")
                    logger.vbr_error(f"Consider: 1) Lower bitrate, 2) Different preset, 3) CPU fallback")
                    
                    # If this is a high bitrate failure, suggest much lower bitrate
                    if bitrate_kbps > 5000:
                        logger.vbr_error(f"High bitrate ({bitrate_kbps}kbps) failure suggests starting too high")
                        return None  # This will trigger bounds expansion with lower targets
                
                return None
            
            # Store result in cache
            test_cache[cache_key] = avg_vmaf
            
            logger.vbr_test(f"{bitrate_kbps}kbps: Average VMAF {avg_vmaf:.2f} from {len(vmaf_scores)} clips")
            return avg_vmaf
        
        # ADAPTIVE ITERATIONS: Calculate based on search characteristics
        adaptive_iterations, adaptive_reasoning = calculate_adaptive_iterations(
            min_br, max_br, target_vmaf, vmaf_tolerance, base_iterations)
        
        logger.vbr(f"🔄 Adaptive iterations: {adaptive_iterations} ({adaptive_reasoning})")
        
        # Bisection search
        current_min = min_br
        current_max = max_br
        best_bitrate_val = current_max
        best_vmaf = 0.0
        highest_vmaf_achieved = 0.0  # Track highest VMAF regardless of target achievement
        bounds_hit = False
        
        logger.vbr_bisect(f"Starting bisection: {current_min}-{current_max}kbps")
        
        # Calculate source file info for space savings projections
        try:
            source_size = infile.stat().st_size
            duration = get_duration_sec(infile)
        except:
            source_size = 0
            duration = 0
        
        for iteration in range(adaptive_iterations):
            # Test midpoint
            test_bitrate_val = (current_min + current_max) // 2
            
            # Calculate projected space savings
            if duration > 0 and source_size > 0:
                projected_size = int((test_bitrate_val * 1000 * duration) / 8)
                
                print(f"[VBR-BISECT] Iteration {iteration+1}: Testing {test_bitrate_val}kbps "
                      f"[bounds: {current_min}-{current_max}kbps] "
                      f"(Original: {source_size/(1024**3):.2f}GB → Projected: {projected_size/(1024**3):.2f}GB)")
            else:
                print(f"[VBR-BISECT] Iteration {iteration+1}: Testing {test_bitrate_val}kbps "
                      f"[bounds: {current_min}-{current_max}kbps]")
            
            vmaf_result = test_bitrate_func(test_bitrate_val)
            
            # Check for VMAF calculation failure
            if vmaf_result is None or vmaf_result <= 0.0:
                logger.vbr_error(f"VMAF calculation failed at {test_bitrate_val}kbps - aborting trial")
                return {
                    'success': False,
                    'bitrate': test_bitrate_val,
                    'preset': preset,
                    'bf': bf,
                    'refs': refs,
                    'vmaf_score': 0.0,
                    'filesize': 0,
                    'error': 'vmaf_calculation_failed'
                }
            
            # Always track the highest VMAF achieved for better reporting
            if vmaf_result > highest_vmaf_achieved:
                highest_vmaf_achieved = vmaf_result
            
            # OPTIMIZATION #2: Enhanced early detection with dynamic quality gap analysis
            if iteration == 0:
                vmaf_excess = vmaf_result - target_vmaf
                
                if vmaf_result < target_vmaf - (vmaf_tolerance * 2):
                    print(f"[VBR-BISECT] Early detection: {vmaf_result:.2f} VMAF significantly below target")
                    print(f"[VBR-BISECT] {preset} preset insufficient even at high bitrate - will try faster preset")
                    return {
                        'success': False,
                        'preset_insufficient': True,
                        'quality_gap': target_vmaf - vmaf_result,
                        'bitrate': test_bitrate_val,
                        'vmaf_score': vmaf_result
                    }
                elif vmaf_excess > vmaf_tolerance * 3:  # Significant overshoot
                    print(f"[VBR-BISECT] Severe overshoot: {vmaf_result:.2f} VMAF (+{vmaf_excess:.1f}) - applying aggressive reduction")
                    # Immediately jump to much lower bitrate (60% reduction)
                    aggressive_target = int(test_bitrate_val * 0.4)
                    current_max = max(aggressive_target, current_min + 100)
                    print(f"[VBR-BISECT] Aggressive bounds update: {current_min}-{current_max}kbps")
                elif vmaf_excess > vmaf_tolerance * 2:  # Moderate overshoot
                    print(f"[VBR-BISECT] Moderate overshoot: {vmaf_result:.2f} VMAF (+{vmaf_excess:.1f}) - reducing search range")
                    # 40% reduction for moderate overshoot
                    moderate_target = int(test_bitrate_val * 0.6)
                    current_max = max(moderate_target, current_min + 100)
            
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
            else:
                # VMAF too high, reduce bitrate
                current_max = test_bitrate_val
                best_bitrate_val = test_bitrate_val  # Keep this as potential result
                best_vmaf = vmaf_result
            
            # ADAPTIVE EARLY TERMINATION: Check for bisection convergence
            range_remaining = current_max - current_min
            if range_remaining <= 100:  # Less than 100kbps difference
                print(f"[VBR-BISECT] Converged: range {current_min}-{current_max}kbps ({range_remaining}kbps)")
                break
            
            # Check for diminishing returns (VMAF not changing much)
            if iteration >= 3:  # After a few iterations
                vmaf_improvement_needed = abs(vmaf_result - target_vmaf)
                if vmaf_improvement_needed <= vmaf_tolerance * 0.3:  # Very close to target
                    print(f"[VBR-BISECT] Near-optimal: VMAF {vmaf_result:.2f} within {vmaf_improvement_needed:.2f} of target")
                    best_bitrate_val = test_bitrate_val
                    best_vmaf = vmaf_result
                    break
                    
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
                            'vmaf_score': highest_vmaf_achieved if best_vmaf == 0.0 else best_vmaf,
                            'filesize': 0,
                            'bounds_hit': True,
                            'abandoned': True,
                            'vmaf_gap': abs(vmaf_diff),
                            'abandon_threshold': abandon_threshold
                        }
                        return result
            else:
                # OPTIMIZATION #2: Can use lower bitrate - apply dynamic quality gap analysis
                vmaf_excess = vmaf_result - target_vmaf
                
                if vmaf_excess > vmaf_tolerance * 3:  # Severe overshoot (3+ tolerance points)
                    # Apply aggressive reduction - jump much lower
                    aggressive_max = int(test_bitrate_val * 0.6)  # 40% reduction
                    current_max = max(aggressive_max, current_min + 100)
                    print(f"[VBR-BISECT] Severe overshoot (+{vmaf_excess:.1f}) - aggressive jump to {current_max}kbps")
                elif vmaf_excess > vmaf_tolerance * 2:  # Significant overshoot 
                    # Apply moderate reduction - jump lower
                    moderate_max = int(test_bitrate_val * 0.75)  # 25% reduction  
                    current_max = max(moderate_max, current_min + 100)
                    print(f"[VBR-BISECT] Significant overshoot (+{vmaf_excess:.1f}) - moderate jump to {current_max}kbps")
                else:
                    # Standard bisection
                    current_max = test_bitrate_val
                    
                best_bitrate_val = test_bitrate_val
                best_vmaf = vmaf_result
            
            # OPTIMIZATION #4: Enhanced convergence detection
            convergence_threshold = 75 if vmaf_excess > vmaf_tolerance * 2 else 100  # Tighter convergence for overshoot
            
            if current_max - current_min < convergence_threshold:
                display_vmaf = highest_vmaf_achieved if best_vmaf == 0.0 else best_vmaf
                print(f"[VBR-BISECT] Converged to {best_bitrate_val}kbps, VMAF {display_vmaf:.2f} (range: {current_max - current_min}kbps)")
                break
            
            # Smart early exit: enhanced sensitivity for quality overshoot cases
            early_exit_threshold = 0.3 if vmaf_excess > vmaf_tolerance * 2 else 0.5
            range_threshold = 200 if vmaf_excess > vmaf_tolerance * 2 else 300
            
            if (abs(best_vmaf - target_vmaf) <= vmaf_tolerance * early_exit_threshold and 
                current_max - current_min < range_threshold):
                print(f"[VBR-BISECT] Enhanced early convergence (excess: +{vmaf_excess:.1f})")
                break
        
        # Check if target was achieved (minimum quality threshold approach)
        display_vmaf = highest_vmaf_achieved if best_vmaf == 0.0 else best_vmaf
        
        # VBR SUCCESS CRITERIA: Accept any result that meets or exceeds minimum quality
        # Instead of strict tolerance window (target±tolerance), use minimum threshold (target-tolerance)
        minimum_acceptable_vmaf = target_vmaf - vmaf_tolerance
        success = display_vmaf >= minimum_acceptable_vmaf
        
        if success and display_vmaf > target_vmaf + vmaf_tolerance:
            # Log when we exceed target (this is actually GOOD for VBR)
            excess = display_vmaf - target_vmaf
            logger.vbr(f"Quality exceeds target by {excess:.1f} VMAF points - excellent result!")
        
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
            'vmaf_score': display_vmaf,
            'filesize': estimated_size,
            'bounds_hit': bounds_hit
        }
        
        return result
        
    finally:
        # EFFICIENCY FIX: Only cleanup clips if they're NOT shared (shared clips cleaned up by parent)
        if not DEBUG and not shared_clips:  # Only cleanup if we extracted our own clips
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
        
        result = run_command(cmd)
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
