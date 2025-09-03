"""
QP (Quantizer Parameter) optimization module for lazy_transcode.

This module handles QP-specific optimization operations including:
- Sample-based QP testing
- Adaptive QP search algorithms
- Gradient descent QP optimization
- Random clip extraction and evaluation
"""

import subprocess
import shlex
import random
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from .media_utils import get_duration_sec, compute_vmaf_score
from .system_utils import TEMP_FILES, DEBUG, run_command
from .transcoding_engine import build_encode_cmd
from .vbr_optimizer import warn_hardware_encoder_inefficiency


def extract_random_clips(file: Path, num_clips: int, clip_duration: int = 120) -> List[Path]:
    """Extract multiple random non-overlapping clips from a file (best-effort)."""
    duration = get_duration_sec(file)
    if duration <= clip_duration:
        print(f"[WARN] File {file.name} is shorter than clip duration, using full file")
        return [file]  # Use the full file if it's too short
    
    clips: list[Path] = []
    intervals: list[tuple[float, float]] = []  # (start, end)
    buffer = 2.0  # seconds minimal gap between clips
    
    # Get the original base stem (remove any existing .clip, .sample artifacts)
    original_stem = file.stem
    # Remove cascading artifacts to get clean base name
    original_stem = re.sub(r'\.clip\d+_\d+\.sample', '', original_stem)
    original_stem = re.sub(r'\.sample_clip', '', original_stem)
    original_stem = re.sub(r'\.qp\d+_sample', '', original_stem)
    
    for i in range(num_clips):
        max_attempts = 100
        start_time: float | None = None
        for attempt in range(max_attempts):
            candidate = random.uniform(0, duration - clip_duration)
            candidate_interval = (candidate, candidate + clip_duration)
            overlap = any(not (candidate_interval[1] + buffer <= iv[0] or candidate_interval[0] >= iv[1] + buffer) for iv in intervals)
            if not overlap:
                start_time = candidate
                break
        if start_time is None:
            # fallback: allow slight overlap
            start_time = random.uniform(0, duration - clip_duration)
        intervals.append((start_time, start_time + clip_duration))
        
        clip_path = file.with_name(f"{original_stem}.clip{i}_{int(start_time)}.sample{file.suffix}")
        TEMP_FILES.add(str(clip_path))
        
        # Extract clip
        extract_cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-loglevel", "error" if not DEBUG else "info",
            "-ss", str(start_time),
            "-i", str(file),
            "-t", str(clip_duration),
            "-c", "copy",
            str(clip_path)
        ]
        
        if DEBUG:
            print(f"[SAMPLE-EXTRACT] {' '.join(shlex.quote(c) for c in extract_cmd)}")
        
        result = run_command(extract_cmd)
        if result.returncode == 0 and clip_path.exists():
            clips.append(clip_path)
        else:
            print(f"[WARN] Failed to extract clip {i} from {file.name}")
            if clip_path in TEMP_FILES:
                TEMP_FILES.discard(str(clip_path))
    
    return clips


def test_qp_on_sample(file: Path, qp: int, encoder: str, encoder_type: str, 
                     sample_duration: int = 60, vmaf_threads: int = 8, 
                     preserve_hdr_metadata: bool = True, use_clip_as_sample: bool = False) -> tuple[float | None, int, int]:
    """Test a QP on a sample clip and return (VMAF score, encoded sample size, original sample size)"""
    
    if use_clip_as_sample:
        # File is already a clip, use it directly as the sample
        sample_clip = file
        skip_cleanup = True  # Don't delete the clip - it's managed by the caller
    else:
        # Create sample clip first (from middle of video)
        duration = get_duration_sec(file)
        start_time = max(0, (duration - sample_duration) / 2) if duration > sample_duration else 0
        sample_clip = file.with_name(f"{file.stem}.sample_clip{file.suffix}")
        TEMP_FILES.add(str(sample_clip))
        skip_cleanup = False
    
    encoded_sample = file.with_name(f"{file.stem}.qp{qp}_sample{file.suffix}")
    if not use_clip_as_sample:
        TEMP_FILES.add(str(encoded_sample))

    try:
        if not use_clip_as_sample:
            # Extract sample clip
            extract_cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-loglevel", "error" if not DEBUG else "info",
                "-ss", str(start_time),
                "-i", str(file),
                "-t", str(sample_duration),
                "-c", "copy",
                str(sample_clip)
            ]
            
            if DEBUG:
                print("[SAMPLE-EXTRACT] " + " ".join(shlex.quote(c) for c in extract_cmd))
            result = run_command(extract_cmd)
            if result.returncode != 0:
                if DEBUG:
                    print("[DEBUG] sample extract stderr:\n" + result.stderr)
                return None, 0, 0
                
        # Encode sample at target QP
        encode_cmd = build_encode_cmd(sample_clip, encoded_sample, encoder, encoder_type, qp, preserve_hdr_metadata)
        # Remove progress args for sample encoding
        if "-progress" in encode_cmd:
            idx = encode_cmd.index("-progress")
            encode_cmd = encode_cmd[:idx] + encode_cmd[idx+2:]
        if "-nostats" in encode_cmd:
            encode_cmd.remove("-nostats")
        
        # Suppress ffmpeg noise unless debugging
        if not DEBUG and "-loglevel" not in encode_cmd:
            encode_cmd.insert(2, "-loglevel")
            encode_cmd.insert(3, "error")
            
        if DEBUG:
            print("[SAMPLE-ENCODE] " + " ".join(shlex.quote(c) for c in encode_cmd))
        else:
            print(f"    -> Encoding QP{qp}...", end="", flush=True)
        
        result = run_command(encode_cmd)
        if result.returncode != 0:
            if not DEBUG:
                print(" FAILED")
            
            if encoder_type == "hardware" and qp < 20:
                print(f"[WARN] Hardware encoder {encoder} failed at QP{qp}. Consider using --min-qp 20 or higher.")
            
            print(f"[WARN] Encoding failed for {sample_clip.name} at QP{qp}: return code {result.returncode}")
            
            if DEBUG:
                print("[DEBUG] sample encode stderr:\n" + result.stderr)
            elif result.stderr:
                error_lines = result.stderr.strip().split('\n')
                if error_lines:
                    print(f"[ERROR] {error_lines[-1]}")
            return None, 0, 0
            
        if not encoded_sample.exists():
            if not DEBUG:
                print(" FAILED (no output)")
            print(f"[WARN] Encoding produced no output file: {encoded_sample}")
            return None, 0, 0
        
        if not DEBUG:
            print(" OK", end="", flush=True)

        # Compute VMAF encoded vs original sample
        if not DEBUG:
            print(", VMAF...", end="", flush=True)
        
        # Verify both files exist before VMAF computation
        if not sample_clip.exists():
            if not DEBUG:
                print(" FAILED (missing reference)")
            print(f"[WARN] Reference clip missing for VMAF: {sample_clip}")
            return None, 0, 0
            
        if not encoded_sample.exists():
            if not DEBUG:
                print(" FAILED (missing encoded)")
            print(f"[WARN] Encoded sample missing for VMAF: {encoded_sample}")
            return None, 0, 0
        
        # Check file sizes to ensure they're not empty/corrupted
        ref_size = sample_clip.stat().st_size
        enc_size = encoded_sample.stat().st_size
        if ref_size == 0:
            if not DEBUG:
                print(" FAILED (empty reference)")
            print(f"[WARN] Reference clip is empty: {sample_clip}")
            return None, 0, 0
        if enc_size == 0:
            if not DEBUG:
                print(" FAILED (empty encoded)")
            print(f"[WARN] Encoded sample is empty: {encoded_sample}")
            return None, 0, 0
        
        vmaf_score = compute_vmaf_score(sample_clip, encoded_sample, n_threads=vmaf_threads)
        
        if not DEBUG:
            if vmaf_score is not None:
                print(f" {vmaf_score:.1f}")
            else:
                print(" FAILED")
        
        encoded_size = encoded_sample.stat().st_size
        sample_orig_size = sample_clip.stat().st_size if sample_clip.exists() else 0
        return vmaf_score, encoded_size, sample_orig_size

    finally:
        if not DEBUG:  # keep for inspection when debugging
            try:
                if not skip_cleanup and sample_clip.exists():
                    sample_clip.unlink()
                    TEMP_FILES.discard(str(sample_clip))
                if encoded_sample.exists():
                    encoded_sample.unlink()
                    if not use_clip_as_sample:
                        TEMP_FILES.discard(str(encoded_sample))
            except:
                pass


def evaluate_qp_on_clips(clips: List[Path], qp: int, encoder: str, encoder_type: str,
                        sample_duration: int, vmaf_threads: int, 
                        preserve_hdr_metadata: bool) -> dict | None:
    """Test a QP on multiple clips and return aggregated results."""
    vmaf_scores: list[float] = []
    total_clip_original_size = 0
    total_encoded_size = 0
    worst_vmaf_drop = 0.0
    
    # Progress bar per QP evaluation (quiet if only one clip)
    pbar = tqdm(total=len(clips), desc=f"QP{qp} clips", position=1, leave=False) if len(clips) > 1 else None
    try:
        for idx, clip in enumerate(clips):
            if pbar:
                pbar.set_description(f"QP{qp} clips - Encoding clip {idx+1}/{len(clips)}")
            else:
                print(f"  -> Encoding QP{qp} clip {idx+1}/{len(clips)}...")
            
            vmaf_score, encoded_size, clip_orig_size = test_qp_on_sample(
                clip, qp, encoder, encoder_type, sample_duration=sample_duration, 
                vmaf_threads=vmaf_threads, preserve_hdr_metadata=preserve_hdr_metadata,
                use_clip_as_sample=True
            )
            
            if pbar:
                if vmaf_score is not None:
                    pbar.set_description(f"QP{qp} clips - Clip {idx+1}: VMAF {vmaf_score:.1f}")
                else:
                    pbar.set_description(f"QP{qp} clips - Clip {idx+1}: VMAF failed")
            else:
                if vmaf_score is not None:
                    print(f"  OK Clip {idx+1}: VMAF {vmaf_score:.1f}")
                else:
                    print(f"  ✗ Clip {idx+1}: VMAF failed")
            
            if vmaf_score is not None:
                vmaf_scores.append(vmaf_score)
                total_encoded_size += encoded_size
                total_clip_original_size += clip_orig_size
                
                baseline_score = 100.0
                drop = baseline_score - vmaf_score
                worst_vmaf_drop = max(worst_vmaf_drop, drop)
            
            if pbar: 
                pbar.update(1)
    finally:
        if pbar: 
            pbar.close()
    
    if not vmaf_scores:
        return None
        
    mean_vmaf = sum(vmaf_scores) / len(vmaf_scores)
    min_vmaf = min(vmaf_scores)
    
    return {
        'mean_vmaf': mean_vmaf,
        'min_vmaf': min_vmaf,
        'passes_quality': True,  # Will be determined by caller
        'samples_used': len(vmaf_scores),
        'worst_drop': worst_vmaf_drop,
        'total_encoded_size': total_encoded_size,
        'total_clip_original_size': total_clip_original_size
    }


def find_optimal_qp(files: list[Path], encoder: str, encoder_type: str, 
                   candidate_qps: Optional[list[int]] = None, 
                   vmaf_target: float = 95.0, 
                   vmaf_min_threshold: float = 93.0,
                   sample_duration: int = 60,
                   min_qp_limit: int = 10,
                   max_qp_limit: int = 45,
                   vmaf_threads: int = 0,
                   preserve_hdr_metadata: bool = True) -> int:
    """Find optimal QP using VMAF scoring across sample files"""
    # Warn about hardware encoder inefficiency for pre-encoding
    warn_hardware_encoder_inefficiency(encoder_type, "QP optimization")
    
    if candidate_qps is None:
        candidate_qps = [16, 18, 20, 22, 24]
        
    print(f"[INFO] Testing QP values {candidate_qps} using VMAF scoring...")
    
    # Test each QP on each sample file
    qp_results = {}
    
    with tqdm(total=len(candidate_qps), desc="Testing QPs", position=0) as qp_pbar:
        for qp in candidate_qps:
            vmaf_scores = []
            total_sample_original_size = 0
            total_encoded_size = 0
            total_full_original_size = 0
            extrapolated_encoded_total = 0.0
            
            print(f"\n[QP {qp}] Testing on {len(files)} samples...")
            with tqdm(total=len(files), desc=f"QP {qp} samples", position=1, leave=False) as file_pbar:
                for file in files:
                    vmaf_score, encoded_size, sample_orig_size = test_qp_on_sample(
                        file, qp, encoder, encoder_type, 
                        sample_duration=sample_duration, vmaf_threads=vmaf_threads, 
                        preserve_hdr_metadata=preserve_hdr_metadata
                    )
                    if vmaf_score is not None:
                        vmaf_scores.append(vmaf_score)
                        total_encoded_size += encoded_size
                        total_sample_original_size += sample_orig_size
                        full_size = file.stat().st_size if file.exists() else 0
                        total_full_original_size += full_size
                        if sample_orig_size > 0 and full_size > 0:
                            # Scale encoded size by ratio of full file size to sample original size
                            extrapolated_encoded_total += (encoded_size / sample_orig_size) * full_size
                    file_pbar.update(1)
            
            if vmaf_scores:
                mean_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                min_vmaf = min(vmaf_scores)
                compression_ratio = (total_encoded_size / total_sample_original_size) * 100 if total_sample_original_size > 0 else 0
                extrapolated_full_pct = (extrapolated_encoded_total / total_full_original_size) * 100 if total_full_original_size > 0 else 0
                
                qp_results[qp] = {
                    'mean_vmaf': mean_vmaf,
                    'min_vmaf': min_vmaf,
                    'compression_ratio': compression_ratio,
                    'extrapolated_full_pct': extrapolated_full_pct,
                    'sample_count': len(vmaf_scores)
                }
            
            qp_pbar.update(1)
    
    # Print results table and select optimal QP
    print(f"\n{'QP':<3} {'MeanVMAF':<9} {'MinVMAF':<8} {'Sample%':<8} {'EstFull%':<9} {'Decision':<9} Notes")
    print("-" * 70)

    def print_row(qp: int):
        r = qp_results[qp]
        mean_vmaf = r['mean_vmaf']
        min_vmaf = r['min_vmaf']
        sample_pct = r['compression_ratio']
        full_pct = r.get('extrapolated_full_pct', 0)
        meets_target = mean_vmaf >= vmaf_target and min_vmaf >= vmaf_min_threshold
        decision = "PASS" if meets_target else "FAIL"
        notes: list[str] = []
        if sample_pct > 100 or full_pct > 100:
            notes.append("INEFFICIENT >100%")
        note_str = ";".join(notes)
        print(f"{qp:<3} {mean_vmaf:<9.2f} {min_vmaf:<8.2f} {sample_pct:<8.1f} {full_pct:<9.1f} {decision:<9} {note_str}")
        return meets_target

    selected_qp = max(candidate_qps)
    any_pass = False
    for qp in sorted(candidate_qps):
        if qp in qp_results:
            meets = print_row(qp)
            if meets:
                selected_qp = qp
                any_pass = True

    print(f"\n[SELECTED] QP {selected_qp} (target VMAF >= {vmaf_target}, min >= {vmaf_min_threshold})")
    return selected_qp


def adaptive_qp_search_per_file(file: Path, encoder: str, encoder_type: str,
                                samples: int = 6,
                                initial_qp: int = 24,
                                initial_step: float = 4.0,
                                min_step: float = 1.0,
                                vmaf_target: float = 95.0,
                                vmaf_min_threshold: float = 93.0,
                                sample_duration: int = 120,
                                min_qp_limit: int = 10,
                                max_qp_limit: int = 45,
                                max_iterations: int = 25,
                                vmaf_threads: int = 0,
                                preserve_hdr_metadata: bool = True,
                                use_gradient_descent: bool = True) -> tuple[int, dict | None]:
    """
    Adaptive QP search for a single file using multiple random clips.
    
    Args:
        use_gradient_descent: If True, uses 2D gradient descent optimization.
                            If False, uses traditional binary search approach.
    
    Returns:
        Tuple of (optimal_qp, result_dict)
    """
    
    # Use gradient descent if requested
    if use_gradient_descent:
        return gradient_descent_qp_search(
            file=file,
            encoder=encoder,
            encoder_type=encoder_type,
            samples=samples,
            initial_qp=initial_qp,
            vmaf_target=vmaf_target,
            vmaf_min_threshold=vmaf_min_threshold,
            sample_duration=sample_duration,
            min_qp_limit=min_qp_limit,
            max_qp_limit=max_qp_limit,
            vmaf_threads=vmaf_threads,
            preserve_hdr_metadata=preserve_hdr_metadata
        )
    
    print(f"[ADAPTIVE] Starting QP optimization for {file.name} at QP {initial_qp}")
    
    # Extract random clips
    clips = extract_random_clips(file, samples, sample_duration)
    if not clips:
        print(f"[ERROR] Could not extract any clips from {file.name}")
        return initial_qp, None
    
    print(f"[INFO] Extracted {len(clips)} clips for QP optimization")
    
    # Adaptive search logic
    qp_results: dict[int, dict] = {}
    tested_qps: set[int] = set()
    current_qp = max(min_qp_limit, min(max_qp_limit, initial_qp))
    direction: int | None = None
    last_good_qp: int | None = None
    best_efficient_qp: int | None = None
    direction_changes = 0
    step = float(initial_step)
    min_step = max(0.5, float(min_step))
    
    try:
        with tqdm(total=max_iterations, desc=f"QP Search {file.name[:40]}", position=0, leave=False) as adapt_bar:
            for it in range(1, max_iterations + 1):
                if current_qp in tested_qps:
                    break
                    
                tested_qps.add(current_qp)
                
                # Update progress bar
                adapt_bar.set_description(f"QP Search {file.name[:35]} - Testing QP{current_qp}")
                adapt_bar.refresh()
                
                result = evaluate_qp_on_clips(clips, current_qp, encoder, encoder_type,
                                            sample_duration, vmaf_threads, preserve_hdr_metadata)
                
                if result is None:
                    adapt_bar.write(f"[WARN] No VMAF scores at QP {current_qp} for {file.name}")
                    break
                    
                qp_results[current_qp] = result
                
                # Update with results
                vmaf = result['mean_vmaf']
                passes_quality = vmaf >= vmaf_target and result['min_vmaf'] >= vmaf_min_threshold
                result['passes_quality'] = passes_quality
                
                # Estimate size efficiency
                if result['total_clip_original_size'] > 0:
                    size_ratio = result['total_encoded_size'] / result['total_clip_original_size']
                    full_file_size = file.stat().st_size if file.exists() else 0
                    estimated_full_size = size_ratio * full_file_size
                    size_pct = (estimated_full_size / full_file_size) * 100 if full_file_size > 0 else 100
                    result['size_pct'] = size_pct
                    result['inefficient'] = size_pct > 100
                else:
                    result['size_pct'] = 100
                    result['inefficient'] = False
                
                adapt_bar.set_description(f"QP Search {file.name[:30]} - QP{current_qp}: VMAF {vmaf:.1f}")
                
                # Update best QPs
                if passes_quality:
                    last_good_qp = current_qp
                    if not result['inefficient']:
                        best_efficient_qp = current_qp
                
                # Determine next direction
                should_increase = (passes_quality and not result['inefficient']) or result['inefficient']
                should_decrease = not passes_quality
                
                if should_increase and should_decrease:
                    break  # Conflicting signals
                elif should_increase:
                    new_direction = 1
                elif should_decrease:
                    new_direction = -1
                else:
                    break  # Acceptable result
                
                # Handle direction changes and step size
                if direction is not None and direction != new_direction:
                    direction_changes += 1
                    step = max(min_step, step / 2.0)
                
                direction = new_direction
                
                # Calculate next QP
                qp_delta = max(1, round(step))
                next_qp = current_qp + (qp_delta * direction)
                next_qp = max(min_qp_limit, min(max_qp_limit, next_qp))
                
                if next_qp == current_qp or (direction_changes >= 2 and qp_delta <= min_step):
                    break
                    
                current_qp = next_qp
                adapt_bar.update(1)
    
    finally:
        # Clean up clips
        if not DEBUG:
            for clip in clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
    
    # Select final QP
    final_qp = best_efficient_qp or last_good_qp or current_qp
    
    if final_qp in qp_results:
        result = qp_results[final_qp]
        print(f"[RESULT] {file.name}: QP {final_qp} mean {result['mean_vmaf']:.2f}, min {result['min_vmaf']:.2f}, size {result.get('size_pct', 100):.1f}%")
        return final_qp, result
    else:
        print(f"[RESULT] {file.name}: QP {final_qp} (fallback)")
        return final_qp, None


def gradient_descent_qp_search(file: Path, encoder: str, encoder_type: str,
                              samples: int = 6,
                              initial_qp: int = 24,
                              vmaf_target: float = 95.0,
                              vmaf_min_threshold: float = 93.0,
                              sample_duration: int = 120,
                              min_qp_limit: int = 10,
                              max_qp_limit: int = 45,
                              vmaf_threads: int = 0,
                              preserve_hdr_metadata: bool = True,
                              learning_rate: float = 0.1,
                              momentum: float = 0.9,
                              tolerance: float = 0.5,
                              max_iterations: int = 15) -> tuple[int, dict | None]:
    """
    2D Gradient Descent QP optimization using QP and VMAF as dimensions.
    
    The algorithm treats this as an optimization problem where:
    - We want to minimize |vmaf_actual - vmaf_target| (quality error)
    - We want to maximize compression (minimize file size / maximize QP)
    
    Uses adaptive learning rate and momentum for faster convergence.
    """
    
    print(f"[GRADIENT-DESCENT] Starting 2D gradient descent QP optimization for {file.name}")
    
    # Extract random clips
    clips = extract_random_clips(file, samples, sample_duration)
    if not clips:
        print(f"[ERROR] Could not extract any clips from {file.name}")
        return initial_qp, None
    
    print(f"[INFO] Extracted {len(clips)} clips for gradient descent optimization")
    
    # Initialize gradient descent parameters
    qp = float(initial_qp)  # Allow fractional QP for gradient computation
    velocity_qp = 0.0  # Momentum term
    
    # Store evaluation history
    history: List[tuple[int, float, float, float]] = []  # (qp_int, vmaf, size_pct, loss)
    best_qp = initial_qp
    best_result = None
    
    print(f"[GRADIENT-DESCENT] Target VMAF: {vmaf_target:.1f}, Min VMAF: {vmaf_min_threshold:.1f}")
    print(f"[GRADIENT-DESCENT] Learning rate: {learning_rate:.3f}, Momentum: {momentum:.2f}, Tolerance: {tolerance:.1f}")
    
    try:
        with tqdm(total=max_iterations, desc=f"GD Search {file.name[:35]}", position=0, leave=False) as pbar:
            for iteration in range(max_iterations):
                # Convert to integer QP for evaluation
                qp_int = max(min_qp_limit, min(max_qp_limit, int(round(qp))))
                
                pbar.set_description(f"GD Search {file.name[:30]} - QP{qp_int} (iter {iteration+1})")
                
                # Evaluate current QP
                result = evaluate_qp_on_clips(clips, qp_int, encoder, encoder_type,
                                            sample_duration, vmaf_threads, preserve_hdr_metadata)
                if result is None:
                    print(f"[WARN] Failed to evaluate QP {qp_int}")
                    break
                
                vmaf = result['mean_vmaf']
                
                # Estimate size percentage
                if result['total_clip_original_size'] > 0:
                    size_ratio = result['total_encoded_size'] / result['total_clip_original_size']
                    full_file_size = file.stat().st_size if file.exists() else 0
                    estimated_full_size = size_ratio * full_file_size
                    size_pct = (estimated_full_size / full_file_size) * 100 if full_file_size > 0 else 100
                else:
                    size_pct = 100
                
                # Calculate loss function
                vmaf_error = abs(vmaf - vmaf_target)
                size_penalty = max(0, size_pct - 100) * 0.1  # Penalize size >100%
                compression_bonus = max(0, (100 - size_pct) * 0.02)  # Reward compression
                loss = vmaf_error + size_penalty - compression_bonus
                
                # Store in history
                history.append((qp_int, vmaf, size_pct, loss))
                
                # Update best result if quality thresholds are met
                passes_quality = vmaf >= vmaf_target and result['min_vmaf'] >= vmaf_min_threshold
                if passes_quality and (best_result is None or loss < best_result.get('loss', float('inf'))):
                    best_qp = qp_int
                    best_result = result.copy()
                    best_result['loss'] = loss
                
                pbar.set_description(f"GD Search {file.name[:25]} - QP{qp_int}: VMAF {vmaf:.1f}, Loss {loss:.2f}")
                
                # Check convergence
                if vmaf_error <= tolerance:
                    print(f"[GRADIENT-DESCENT] Converged: VMAF {vmaf:.2f} within tolerance {tolerance}")
                    break
                
                # Compute gradients using finite differences
                gradient_qp = 0.0
                
                if len(history) >= 2:
                    # Use previous point for gradient estimation
                    prev_qp, prev_vmaf, prev_size, prev_loss = history[-2]
                    curr_qp, curr_vmaf, curr_size, curr_loss = history[-1]
                    
                    if curr_qp != prev_qp:
                        # dLoss/dQP
                        gradient_qp = (curr_loss - prev_loss) / (curr_qp - prev_qp)
                else:
                    # First iteration: use sign of VMAF error to determine direction
                    if vmaf < vmaf_target:
                        # Need higher quality -> DECREASE QP (negative direction)
                        gradient_qp = -1.0
                    else:
                        # Quality is good -> try higher QP for more compression
                        gradient_qp = 1.0
                
                # Update velocity with momentum
                velocity_qp = momentum * velocity_qp - learning_rate * gradient_qp
                
                # Update QP
                qp += velocity_qp
                
                # Clamp to bounds
                qp = max(min_qp_limit, min(max_qp_limit, qp))
                
                if DEBUG:
                    print(f"[GD-DEBUG] Iter {iteration+1}: QP {qp_int} -> {qp:.2f}, VMAF {vmaf:.2f}, "
                          f"Loss {loss:.3f}, Grad {gradient_qp:.3f}")
                
                pbar.update(1)
                
                # Early stopping conditions
                if len(history) >= 3:
                    last_qps = [h[0] for h in history[-3:]]
                    last_losses = [h[3] for h in history[-3:]]
                    
                    # Check if QP is stuck at same value
                    if len(set(last_qps)) == 1:
                        print(f"[GRADIENT-DESCENT] Early stop: QP converged at {qp_int}")
                        break
                    
                    # Check if loss has plateaued
                    if len(set([round(l, 2) for l in last_losses])) == 1:
                        print(f"[GRADIENT-DESCENT] Early stop: Loss plateaued at {loss:.2f}")
                        break
    
    finally:
        # Clean up clips
        if not DEBUG:
            for clip in clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
    
    # Return best result
    if best_result:
        final_qp = best_qp
        result = best_result
        print(f"[GRADIENT-DESCENT] {file.name}: QP {final_qp} mean {result['mean_vmaf']:.2f}, min {result['min_vmaf']:.2f}")
        return final_qp, result
    else:
        # Fallback to best evaluated QP
        if history:
            best_by_quality = min(history, key=lambda x: abs(x[1] - vmaf_target))
            final_qp = best_by_quality[0]
            print(f"[GRADIENT-DESCENT] {file.name}: QP {final_qp} (fallback from gradient descent)")
            return final_qp, None
        else:
            print(f"[GRADIENT-DESCENT] {file.name}: QP {initial_qp} (no evaluations completed)")
            return initial_qp, None
