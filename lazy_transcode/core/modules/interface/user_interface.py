"""
User interface module for lazy_transcode.

This module handles user interaction including:
- Confirmation prompts
- Verification workflows
- Progress display
"""

import sys
from pathlib import Path
from typing import List

from ..system.system_utils import format_size, run_command
from ..analysis.media_utils import get_duration_sec, compute_vmaf_score, should_skip_codec


def prompt_user_confirmation(message: str, auto_yes: bool = False) -> bool:
    """Prompt user for confirmation."""
    if auto_yes:
        print(f"{message} [auto-yes]")
        return True
        
    while True:
        response = input(f"{message} [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def verify_and_prompt_transcode(files: List[Path], optimal_qp: int, encoder: str, encoder_type: str, 
                               args, preserve_hdr: bool) -> bool:
    """Verify quality and size thresholds, then prompt user for confirmation."""
    
    # Calculate total original size for display
    total_original_size = sum(f.stat().st_size for f in files if f.exists())
    
    print(f"\n[VERIFY] Quality & size verification for {len(files)} files (total: {format_size(total_original_size)})...")
    
    # Sample a few files for quality/size estimates
    verify_sample_count = min(3, len(files))
    verify_files = files[:verify_sample_count]
    
    vmaf_scores = []
    size_ratios = []
    
    for f in verify_files:
        # Quick VMAF test on middle sample
        duration = get_duration_sec(f)
        if duration is None:
            continue
            
        # Test on small sample for verification
        sample_duration = min(60, duration)
        start_time = max(0, (duration - sample_duration) / 2)
        
        sample_file = f.with_name(f"{f.stem}.verify_sample{f.suffix}")
        encoded_sample = f.with_name(f"{f.stem}.verify_encoded{f.suffix}")
        
        try:
            # Extract sample
            import subprocess
            extract_cmd = [
                "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                "-ss", str(start_time), "-i", str(f), "-t", str(sample_duration),
                "-c", "copy", str(sample_file)
            ]
            result = run_command(extract_cmd)
            
            if result.returncode == 0 and sample_file.exists():
                # Encode sample
                from ..processing.transcoding_engine import build_encode_cmd
                encode_cmd = build_encode_cmd(sample_file, encoded_sample, encoder, 
                                            encoder_type, optimal_qp, preserve_hdr)
                result = run_command(encode_cmd)
                
                if result.returncode == 0 and encoded_sample.exists():
                    # Compute VMAF
                    vmaf_score = compute_vmaf_score(sample_file, encoded_sample, 
                                                  n_threads=args.vmaf_threads)
                    if vmaf_score is not None:
                        vmaf_scores.append(vmaf_score)
                        
                        # Size ratio
                        orig_size = sample_file.stat().st_size
                        enc_size = encoded_sample.stat().st_size
                        if orig_size > 0:
                            size_ratios.append(enc_size / orig_size)
                            
        finally:
            # Cleanup verification files
            for temp_file in [sample_file, encoded_sample]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
    
    # Display verification results
    if vmaf_scores:
        avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
        min_vmaf = min(vmaf_scores)
        print(f"[VERIFY] Quality preview (QP {optimal_qp}): VMAF {avg_vmaf:.1f} avg, {min_vmaf:.1f} min")
        
        if size_ratios:
            avg_size_ratio = sum(size_ratios) / len(size_ratios)
            estimated_total_size = total_original_size * avg_size_ratio
            size_pct = avg_size_ratio * 100
            
            print(f"[VERIFY] Size preview: {size_pct:.1f}% of original ({format_size(int(estimated_total_size))})")
            
            # Quality checks
            meets_target = avg_vmaf >= args.vmaf_target
            meets_min = min_vmaf >= args.vmaf_min
            saves_space = size_pct <= args.max_size_pct
            
            # Decision logic
            quality_ok = meets_target and meets_min
            size_ok = saves_space or args.force_overwrite
            
            if quality_ok and size_ok:
                action = "proceed" if not args.force_overwrite else "proceed (forced)"
                print(f"[VERIFY] ✓ Quality and size checks passed - ready to {action}")
                
                if not args.auto_yes:
                    return prompt_user_confirmation(
                        f"Transcode {len(files)} files with QP {optimal_qp}?", 
                        args.auto_yes
                    )
                else:
                    return True
                    
            else:
                # Check if within prompting band
                vmaf_deficit = args.vmaf_target - avg_vmaf
                min_deficit = args.vmaf_min - min_vmaf
                worst_deficit = max(vmaf_deficit, min_deficit)
                
                within_prompt_band = worst_deficit <= args.quality_prompt_band
                
                if within_prompt_band and not args.force_overwrite:
                    print(f"[VERIFY] ⚠ Quality marginally below target (gap: {worst_deficit:.1f})")
                    return prompt_user_confirmation(
                        f"Proceed anyway with slightly lower quality?",
                        args.auto_yes
                    )
                else:
                    if not quality_ok:
                        print(f"[VERIFY] ✗ Quality check failed:")
                        print(f"  Target: {args.vmaf_target:.1f} (got {avg_vmaf:.1f}) {'✗' if not meets_target else '✓'}")
                        print(f"  Minimum: {args.vmaf_min:.1f} (got {min_vmaf:.1f}) {'✗' if not meets_min else '✓'}")
                    if not size_ok:
                        print(f"[VERIFY] ✗ Size check failed: {size_pct:.1f}% > {args.max_size_pct:.1f}% limit")
                    
                    if args.force_overwrite:
                        print(f"[VERIFY] Proceeding anyway due to --force-overwrite")
                        return True
                    else:
                        print(f"[VERIFY] Aborting. Use --force-overwrite to proceed anyway.")
                        return False
        else:
            print(f"[VERIFY] Could not verify quality - proceeding with caution")
            return prompt_user_confirmation(
                f"Unable to verify quality. Proceed with QP {optimal_qp}?",
                args.auto_yes
            )
    else:
        print(f"[VERIFY] No verification samples available - proceeding")
        return True


def display_encoder_info(encoder: str, encoder_type: str, cpu_forced: bool = False):
    """Display encoder information and optimization details."""
    print(f"[INFO] Using encoder: {encoder} ({encoder_type})")
    
    if cpu_forced:
        import os
        cpu_count = os.cpu_count() or 8
        print(f"[CPU] MAXIMUM optimization for {cpu_count} cores:")
        print(f"      • {cpu_count} threads (100% utilization)")  
        print(f"      • {min(12, max(3, cpu_count // 2))} frame-threads (aggressive pipeline)")
        print(f"      • {min(6, max(2, cpu_count // 4))} pools (NUMA-aware distribution)")
        print(f"      • {min(8, max(2, cpu_count // 4))} lookahead-threads (analysis parallelism)")
        print(f"      • Advanced: pmode, pme, b-adapt=2, subme=7 (CPU intensive)")


def display_vbr_results_summary(vbr_results: dict, total_files: int):
    """Display VBR optimization results summary."""
    if vbr_results:
        print(f"\n[VBR] Optimization complete for {len(vbr_results)}/{total_files} files:")
        total_bitrate = sum(r['bitrate'] for r in vbr_results.values())
        avg_bitrate = total_bitrate / len(vbr_results)
        avg_vmaf = sum(r['vmaf_score'] for r in vbr_results.values()) / len(vbr_results)
        
        print(f"  Average bitrate: {avg_bitrate:.0f} kbps")
        print(f"  Average VMAF: {avg_vmaf:.2f}")
    else:
        print(f"[VBR] No files successfully optimized")


def display_qp_optimization_summary(file_qp_map: dict, files: List[Path]):
    """Display QP optimization results summary."""
    if file_qp_map:
        all_qps = list(file_qp_map.values())
        optimal_qp = sorted(all_qps)[len(all_qps)//2] if all_qps else 24
        print(f"[INFO] Per-file optimization complete. QP range: {min(all_qps)}-{max(all_qps)}, median: {optimal_qp}")
        return optimal_qp
    else:
        print(f"[INFO] No QP optimization results available")
        return 24  # Default fallback
