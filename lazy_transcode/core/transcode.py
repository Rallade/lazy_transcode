import os, sys, shlex, signal, atexit, shutil, random, subprocess, argparse, re, time, threading
import asyncio
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:  # fallback minimal stub
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, position=None, leave=True):
            self.iterable = iterable or range(total or 0)
        def __iter__(self):
            for x in self.iterable:
                yield x
        def update(self, n=1):
            pass
        def close(self):
            pass
        def set_postfix(self, **kw):
            pass
        def write(self, s):
            print(s)
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
from functools import lru_cache
import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Optional, List
import queue

# Import modular components
from .modules.encoder_config import EncoderConfigBuilder
from .modules.vmaf_evaluator import VMAfEvaluator
from .modules.file_manager import FileManager

# --- config defaults ---
EXTS = [".mkv", ".mp4", ".mov", ".ts"]
SAMPLE_COUNT_DEFAULT = 6

# VMAF auto-QP configuration
CANDIDATE_QPS = [16, 18, 20, 22, 24]
VMAF_TARGET = 95.0
VMAF_MIN_THRESHOLD = 93.0

TEMP_FILES: set[str] = set()
# Stores per-original baseline (self) VMAF computed on the sampled clip (original vs itself)
BASELINE_VMAF: dict[str, float] = {}
DEBUG = False  # set by --debug

def get_next_transcoded_dir(base_path: Path) -> Path:
    """Find the next available Transcoded_X directory name to avoid overwriting existing folders."""
    base_name = "Transcoded"
    candidate_dir = base_path / base_name
    
    # If Transcoded doesn't exist, use it
    if not candidate_dir.exists():
        return candidate_dir
    
    # Find next available Transcoded_X
    counter = 2
    while True:
        candidate_dir = base_path / f"{base_name}_{counter}"
        if not candidate_dir.exists():
            return candidate_dir
        counter += 1

def _cleanup():
    for f in list(TEMP_FILES):
        try:
            if os.path.exists(f):
                os.remove(f)
                print(f"[CLEANUP] removed {f}")
        except Exception:
            pass
        finally:
            TEMP_FILES.discard(f)

atexit.register(_cleanup)
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGABRT):
    signal.signal(sig, lambda s, f: sys.exit(1))

def run_logged(cmd: list[str], **popen_kwargs) -> subprocess.CompletedProcess:
    print("[CMD] " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, **popen_kwargs)

def format_size(bytes_size: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def start_cpu_monitor(duration_seconds: int = 0) -> tuple[threading.Thread, threading.Event]:
    """Start CPU monitoring in background. Returns (thread, stop_event)"""
    stop_event = threading.Event()
    
    def monitor_cpu():
        try:
            import psutil
            print(f"[CPU-MONITOR] Starting CPU monitoring (cores: {psutil.cpu_count()})")
            start_time = time.time()
            while not stop_event.is_set():
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
                active_cores = sum(1 for c in cpu_per_core if c > 10)
                print(f"[CPU] {cpu_percent:5.1f}% total, {active_cores}/16 cores active (>10%)")
                if duration_seconds > 0 and time.time() - start_time > duration_seconds:
                    break
        except ImportError:
            print("[CPU-MONITOR] psutil not available, using basic Windows monitoring")
            start_time = time.time()
            while not stop_event.is_set():
                try:
                    # Windows-specific CPU monitoring using wmi
                    result = subprocess.run(['powershell', '-Command', 
                        'Get-Counter "\\Processor(_Total)\\% Processor Time" -SampleInterval 1 -MaxSamples 1 | Select -ExpandProperty CounterSamples | Select -ExpandProperty CookedValue'], 
                        capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        cpu_usage = float(result.stdout.strip())
                        print(f"[CPU] {cpu_usage:5.1f}% total")
                    time.sleep(2)
                except Exception:
                    time.sleep(3)
                    if DEBUG:
                        print("[CPU] Basic monitoring active")
                
                if duration_seconds > 0 and time.time() - start_time > duration_seconds:
                    break
        except Exception as e:
            if DEBUG:
                print(f"[CPU-MONITOR] Error: {e}")
    
    thread = threading.Thread(target=monitor_cpu, daemon=True)
    thread.start()
    return thread, stop_event

@lru_cache(maxsize=1024)
def ffprobe_field(file: Path, key: str) -> str | None:
    """Get field from video stream using ffprobe - delegates to module version."""
    from .modules.encoder_config import ffprobe_field as probe_field
    return probe_field(file, key)

@lru_cache(maxsize=4096)
def get_duration_sec(file: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nk=1:nw=1", str(file)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return float(out) if out else 0.0
    except subprocess.CalledProcessError:
        return 0.0

def get_video_codec(file: Path) -> str | None:
    """Get the video codec name from a file"""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name",
           "-of", "default=nk=1:nw=1", str(file)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out if out and out != "unknown" else None
    except subprocess.CalledProcessError:
        return None

def should_skip_codec(codec: str | None) -> bool:
    """Check if a codec should be skipped (already efficient)"""
    if not codec:
        return False
    
    # Modern efficient codecs that shouldn't be re-encoded
    efficient_codecs = {
        'hevc', 'h265',           # HEVC/H.265
        'av1',                    # AV1
        'vp9',                    # VP9 (also quite efficient)
    }
    
    return codec.lower() in efficient_codecs

def detect_hevc_encoder() -> tuple[str, str]:
    """Detect available HEVC encoder. Returns (encoder_name, encoder_type)"""
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], 
                              capture_output=True, text=True, check=True)
        encoders = result.stdout.lower()
        
        # Check in order of preference: AMD -> NVIDIA -> Intel -> Software
        if "hevc_amf" in encoders:
            return "hevc_amf", "hardware"
        elif "hevc_nvenc" in encoders:
            return "hevc_nvenc", "hardware" 
        elif "hevc_qsv" in encoders:
            return "hevc_qsv", "hardware"
        else:
            return "libx265", "software"
    except subprocess.CalledProcessError:
        # Fallback if ffmpeg not available or error
        return "libx265", "software"

@lru_cache(maxsize=1024)
def _extract_hdr_metadata(file: Path) -> tuple[str | None, str | None]:
    """Extract static HDR10 metadata (master_display, max_cll) if present.
    Returns (master_display, max_cll) strings usable directly with ffmpeg -master_display / -max_cll.
    """
    # Import from module to avoid duplication
    from .modules.encoder_config import _extract_hdr_metadata as extract_hdr
    return extract_hdr(file)

HDR_METADATA_ANNOUNCED: set[str] = set()

def build_encode_cmd(infile: Path, outfile: Path, encoder: str, encoder_type: str, qp: int, preserve_hdr_metadata: bool = True) -> list[str]:
    """Build encoding command using EncoderConfigBuilder."""
    builder = EncoderConfigBuilder()
    
    # Determine encoder type for threading
    if encoder_type == "hardware":
        threads = 4
    else:
        threads = os.cpu_count() or 16
    
    # Get video dimensions (not used by new builder but kept for compatibility)
    width, height = 1920, 1080  # Default fallback
    
    cmd = builder.build_standard_encode_cmd(
        str(infile), str(outfile), encoder, "medium", qp,
        width, height, threads=threads, 
        preserve_hdr=preserve_hdr_metadata, debug=DEBUG
    )
    
    # Handle HDR metadata announcement for hardware encoders
    if preserve_hdr_metadata and encoder_type == "hardware":
        from .modules.encoder_config import ffprobe_field, _extract_hdr_metadata
        pixfmt = ffprobe_field(infile, "pix_fmt")
        is10 = bool(pixfmt and ("10le" in pixfmt or "p10" in pixfmt))
        if is10:
            master_display, max_cll = _extract_hdr_metadata(infile)
            if (master_display or max_cll) and str(infile) not in HDR_METADATA_ANNOUNCED:
                print(f"[HDR] Preserving static HDR10 metadata for {infile.name}: "
                      f"master_display={'yes' if master_display else 'no'}, max_cll={'yes' if max_cll else 'no'}")
                HDR_METADATA_ANNOUNCED.add(str(infile))
    
    return cmd

def build_vbr_encode_cmd(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                        bitrate_kbps: int, preset: str = "medium", bf: int = 3, 
                        refs: int = 3, use_advanced_features: bool = True, 
                        preserve_hdr_metadata: bool = True) -> list[str]:
    """Build VBR encoding command using EncoderConfigBuilder."""
    builder = EncoderConfigBuilder()
    
    # Determine threading based on encoder type
    if encoder_type == "hardware":
        threads = 4
    else:
        threads = os.cpu_count() or 16
    
    # Get video dimensions for proper aspect ratio handling
    width, height = 1920, 1080  # Default fallback
    
    return builder.build_vbr_encode_cmd(
        str(infile), str(outfile), encoder, preset, bitrate_kbps,
        bf, refs, width, height, threads=threads,
        preserve_hdr=preserve_hdr_metadata, debug=DEBUG
    )

def encode_vbr_with_progress(infile: Path, outfile: Path, encoder: str, encoder_type: str, 
                           bitrate_kbps: int, preset: str = "medium", bf: int = 3, 
                           refs: int = 3, preserve_hdr_metadata: bool = True) -> bool:
    """Encode a file using VBR with progress tracking."""
    dur = get_duration_sec(infile)
    if dur <= 0:
        print(f"[WARN] could not get duration for {infile}")
    TEMP_FILES.add(str(outfile))

    cmd = build_vbr_encode_cmd(infile, outfile, encoder, encoder_type, bitrate_kbps, 
                              preset, bf, refs, preserve_hdr_metadata=preserve_hdr_metadata)
    
    # capture progress (stdout) + errors (stderr)
    if DEBUG:
        print("[VBR-ENCODE] " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Use tqdm for individual file progress
    with tqdm(total=100, desc=f"VBR Encoding {infile.name[:25]} ({bitrate_kbps}k)", unit="%", 
             position=1, leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}") as pbar:
        last_pct = 0
        if proc.stdout:
            for line in proc.stdout:
                # Handle out_time_ms, out_time_us, or out_time robustly
                if any(line.startswith(prefix) for prefix in ["out_time_ms=", "out_time_us=", "out_time="]) and dur > 0:
                    try:
                        time_str = line.split("=",1)[1].strip()
                        if time_str == "N/A":
                            continue
                        
                        # Parse time based on format
                        if line.startswith("out_time_ms="):
                            sec = int(time_str) / 1_000_000  # microseconds to seconds
                        elif line.startswith("out_time_us="):
                            sec = int(time_str) / 1_000_000  # microseconds to seconds
                        elif line.startswith("out_time="):
                            # Handle time format like "00:01:23.45"
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
                        # Skip invalid progress lines
                        continue

    proc.wait()
    if proc.returncode == 0:
        # Keep tracking until caller finalizes (swap/delete) to allow cleanup if interrupted mid-swap
        return True
    else:
        err = proc.stderr.read() if proc.stderr else ""
        print(f"[ERROR] VBR encoding failed for {infile}\n{err}")
        return False

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
    """Top-down bisection search to find minimum bitrate meeting VMAF target."""
    
    low, high = min_br, max_br
    best_result = None
    original_bounds = (min_br, max_br)
    bounds_hit = False
    
    # Start from upper bound and work downward to find minimum acceptable bitrate
    for iteration in range(max_iterations):
        # For first iteration, start at high bound; then use traditional bisection
        if iteration == 0:
            current_bitrate = high
        else:
            current_bitrate = (low + high) // 2
            
        print(f"    Bitrate {current_bitrate}kbps [{low}-{high}]")
        
        # Test this bitrate with current parameters
        vmaf_score = _test_vbr_encoding(
            infile, encoder, encoder_type, current_bitrate,
            clip_positions, clip_duration, preset, bf, refs
        )
        
        if vmaf_score is None:
            print(f"    Encoding failed at {current_bitrate}kbps")
            high = current_bitrate - 1
            continue
            
        vmaf_diff = vmaf_score - target_vmaf
        print(f"    VMAF: {vmaf_score:.2f} (target: {target_vmaf:.1f})")
        
        # For VBR optimization, we want minimum bitrate that MEETS the target
        # Only accept solutions at or above (target - tolerance)
        if vmaf_diff >= -vmaf_tolerance:
            # Acceptable quality - this is a valid solution, try to go lower
            best_result = {
                'success': True,
                'bitrate': current_bitrate,
                'preset': preset,
                'bf': bf,
                'refs': refs,
                'vmaf_score': vmaf_score,
                'bounds_hit': False  # We found a solution
            }
            print(f"    ✓ Valid solution found, trying lower...")
            high = current_bitrate - 1  # Try to find even lower bitrate
            
        elif vmaf_diff < -vmaf_tolerance:  # Quality too low - need higher bitrate
            # Check if we're hitting upper bounds
            if current_bitrate >= original_bounds[1] * 0.95:
                print(f"    [BOUNDS] Hitting upper bound at {current_bitrate}kbps")
                bounds_hit = True
                
                # Early abandonment: progressive thresholds based on how much we've expanded
                # This saves time on files that clearly can't reach target quality
                abandon_threshold = 15.0  # Default for first bounds hit
                if expand_factor > 0:  # We've already tried expanding bounds
                    # Get more aggressive with each expansion
                    # expand_factor 1: 12 points, expand_factor 2: 9 points, expand_factor 3: 6 points
                    abandon_threshold = max(6.0, 15.0 - (expand_factor * 3.0))
                
                if vmaf_diff < -abandon_threshold:
                    print(f"    [ABANDON] VMAF {vmaf_score:.2f} is {abs(vmaf_diff):.1f} points below target")
                    print(f"    [ABANDON] Threshold: {abandon_threshold:.1f} points - file likely cannot reach target quality")
                    return {
                        'success': False,
                        'bounds_hit': True,
                        'abandoned': True,
                        'vmaf_gap': abs(vmaf_diff),
                        'abandon_threshold': abandon_threshold
                    }
                
            low = current_bitrate + 1  # Increase minimum - this will lower the bound for next iteration
        else:  # Quality too high (above target + tolerance) - can reduce bitrate further
            high = current_bitrate - 1  # Reduce maximum
            
        if low > high:
            break
    
    # Return result with bounds info
    if best_result:
        return best_result
    else:
        return {
            'success': False,
            'bounds_hit': bounds_hit
        }

def _test_vbr_encoding(infile: Path, encoder: str, encoder_type: str, bitrate_kbps: int,
                      clip_positions: list[int], clip_duration: int,
                      preset: str, bf: int, refs: int) -> float | None:
    """Test VBR encoding at specific settings and return VMAF score."""
    
    # Use VMAfEvaluator for centralized VBR evaluation
    evaluator = VMAfEvaluator(debug=DEBUG, temp_files=TEMP_FILES, baseline_vmaf=BASELINE_VMAF)
    
    # Create VBR encoder command builder
    def vbr_encode_cmd_builder(clip_ref, clip_enc, enc, enc_type, bitrate_val, preserve_hdr):
        return build_vbr_encode_cmd(
            clip_ref, clip_enc, enc, enc_type,
            bitrate_val, preset, bf, refs,
            use_advanced_features=True, preserve_hdr_metadata=preserve_hdr
        )
    
    # Convert int positions to float for VMAfEvaluator
    float_positions = [float(pos) for pos in clip_positions]
    
    average_vmaf = evaluator.test_vbr_settings(
        infile, encoder, encoder_type, bitrate_kbps,
        preset, bf, refs, float_positions, float(clip_duration),
        vbr_encode_cmd_builder
    )
    
    return average_vmaf

def get_vbr_clip_positions(duration_seconds: int, num_clips: int = 2) -> list[int]:
    """Get clip start positions for VBR testing."""
    if duration_seconds < 60:
        return [10]  # Single clip for short files
    
    # For longer files, sample from different parts
    positions = []
    segment_size = duration_seconds // (num_clips + 1)
    
    for i in range(1, num_clips + 1):
        pos = i * segment_size
        positions.append(max(30, min(pos, duration_seconds - 60)))  # Ensure valid range
        
    return positions

def encode_with_progress(infile: Path, outfile: Path, encoder: str, encoder_type: str, qp: int, preserve_hdr_metadata: bool = True) -> bool:
    dur = get_duration_sec(infile)
    if dur <= 0:
        print(f"[WARN] could not get duration for {infile}")
    TEMP_FILES.add(str(outfile))

    cmd = build_encode_cmd(infile, outfile, encoder, encoder_type, qp, preserve_hdr_metadata=preserve_hdr_metadata)
    # capture progress (stdout) + errors (stderr)
    if DEBUG:
        print("[ENCODE] " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Use tqdm for individual file progress
    with tqdm(total=100, desc=f"Encoding {infile.name[:30]}", unit="%", 
             position=1, leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}") as pbar:
        last_pct = 0
        if proc.stdout:
            for line in proc.stdout:
                # Handle out_time_ms, out_time_us, or out_time robustly
                if any(line.startswith(prefix) for prefix in ["out_time_ms=", "out_time_us=", "out_time="]) and dur > 0:
                    try:
                        time_str = line.split("=",1)[1].strip()
                        if time_str == "N/A":
                            continue
                        
                        # Parse time based on format
                        if line.startswith("out_time_ms="):
                            sec = int(time_str) / 1_000_000  # microseconds to seconds
                        elif line.startswith("out_time_us="):
                            sec = int(time_str) / 1_000_000  # microseconds to seconds
                        elif line.startswith("out_time="):
                            # Handle time format like "00:01:23.45"
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
                        # Skip invalid progress lines
                        continue

    proc.wait()
    if proc.returncode == 0:
        # Keep tracking until caller finalizes (swap/delete) to allow cleanup if interrupted mid-swap
        return True
    else:
        err = proc.stderr.read() if proc.stderr else ""
        print(f"[ERROR] ffmpeg failed for {infile}\n{err}")
        return False

def compute_vmaf_score(reference: Path, distorted: Path, n_threads: int = 0, enable_cpu_monitoring: bool = False) -> float | None:
    """Compute VMAF score between reference (original) and distorted (encoded) video.
    n_threads: 0 lets libvmaf decide (auto). >0 sets explicit thread count.
    Retries once without thread option if initial invocation fails (for compatibility).
    
    PREPROCESSING BOTTLENECK SOLUTION: If videos have different resolutions,
    this function will pre-scale the reference to match the distorted resolution,
    eliminating the single-threaded scaling bottleneck in the VMAF pipeline.
    """
    def _get_video_resolution(video_path: Path) -> tuple[int, int] | None:
        """Get video resolution using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", 
                "-of", "csv=s=x:p=0", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                width, height = map(int, result.stdout.strip().split('x'))
                return width, height
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Resolution detection failed for {video_path}: {e}")
        return None

    def _create_scaled_reference(ref_path: Path, target_resolution: tuple[int, int]) -> Path | None:
        """Create a temporary scaled version of the reference video"""
        try:
            target_width, target_height = target_resolution
            temp_scaled = Path(f"temp_scaled_{ref_path.stem}_{target_width}x{target_height}.mp4")
            TEMP_FILES.add(str(temp_scaled))
            
            scale_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(ref_path),
                "-vf", f"scale={target_width}:{target_height}",
                "-c:v", "libx264", "-crf", "18",  # High quality scaling
                "-preset", "fast",  # Balance quality vs speed
                str(temp_scaled)
            ]
            
            if DEBUG:
                print(f"[VMAF] Creating scaled reference: {target_width}x{target_height}")
            
            result = subprocess.run(scale_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and temp_scaled.exists():
                return temp_scaled
            else:
                print(f"[WARN] Failed to scale reference video: {result.stderr}")
                TEMP_FILES.discard(str(temp_scaled))
                return None
                
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Scaling failed: {e}")
            return None

    def _run(ref_path: Path, dist_path: Path, thr: int) -> tuple[int, str]:
        # Enhanced libvmaf options for better threading performance
        opts = []
        if thr and thr > 0:
            opts.append(f"n_threads={thr}")
            # Add subsample option for better threading with high thread counts
            if thr >= 4:
                opts.append("n_subsample=1")  # Process every frame for accuracy
        
        # Correct libvmaf syntax with options (avoid log_path to prevent Windows path issues)
        opt_str = f"={':'.join(opts)}" if opts else ""
        filter_graph = f"[0:v][1:v]libvmaf{opt_str}"
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
            "-i", str(dist_path),   # distorted first
            "-i", str(ref_path),    # reference second
            "-lavfi", filter_graph,
            "-f", "null", "-"
        ]
        if DEBUG:
            print("[VMAF-CMD] " + " ".join(shlex.quote(c) for c in cmd))
            print(f"[VMAF-THREADS] Using {thr if thr > 0 else 'auto'} threads with enhanced options")
        
        # Start CPU monitoring if requested
        cpu_monitor = None
        cpu_stop_event = None
        if enable_cpu_monitoring:
            cpu_monitor, cpu_stop_event = start_cpu_monitor()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Stop CPU monitoring
        if cpu_monitor and cpu_stop_event:
            cpu_stop_event.set()
            cpu_monitor.join(timeout=1)
        
        return result.returncode, result.stderr or ""

    # Validate input files exist and are accessible
    if not reference.exists():
        print(f"[WARN] VMAF reference file missing: {reference}")
        return None
    if not distorted.exists():
        print(f"[WARN] VMAF distorted file missing: {distorted}")
        return None

    # SMART PREPROCESSING: Only scale if absolutely necessary
    ref_resolution = _get_video_resolution(reference)
    dist_resolution = _get_video_resolution(distorted)
    
    scaled_reference = None
    actual_reference = reference
    
    if ref_resolution and dist_resolution and ref_resolution != dist_resolution:
        if DEBUG:
            print(f"[VMAF] Resolution mismatch: {ref_resolution[0]}x{ref_resolution[1]} → {dist_resolution[0]}x{dist_resolution[1]}")
        
        # Calculate resolution difference to decide if scaling is worth it
        ref_pixels = ref_resolution[0] * ref_resolution[1]
        dist_pixels = dist_resolution[0] * dist_resolution[1] 
        pixel_ratio = abs(ref_pixels - dist_pixels) / ref_pixels
        
        if pixel_ratio > 0.1:  # More than 10% pixel difference
            if DEBUG:
                print(f"[VMAF] Pre-scaling reference for optimal threading ({pixel_ratio:.1%} difference)")
            scaled_reference = _create_scaled_reference(reference, dist_resolution)
            if scaled_reference:
                actual_reference = scaled_reference
            else:
                if DEBUG:
                    print(f"[VMAF] Pre-scaling failed, using FFmpeg scaling")
        elif DEBUG:
            print(f"[VMAF] Minor resolution difference ({pixel_ratio:.1%}), using FFmpeg scaling")
    elif DEBUG and ref_resolution and dist_resolution:
        print(f"[VMAF] Same resolution ({ref_resolution[0]}x{ref_resolution[1]}), optimal threading")

    try:
        rc, stderr_text = _run(actual_reference, distorted, n_threads)
        if rc != 0 and n_threads > 0:
            # Retry with auto threads (0) if explicit thread count rejected
            print(f"[WARN] VMAF failed with n_threads={n_threads}, retrying with auto (0)")
            rc, stderr_text = _run(actual_reference, distorted, 0)
        if rc != 0:
            # Show more context on failure
            last_lines = stderr_text.splitlines()[-3:] if stderr_text else ['unknown error']
            error_msg = ' | '.join(last_lines)
            print(f"[WARN] VMAF computation failed: {error_msg}")
            if DEBUG and stderr_text:
                print("[DEBUG] Full VMAF stderr:")
                for l in stderr_text.splitlines()[-10:]:
                    print("  " + l)
            return None
        
        # Parse VMAF score from stderr - try multiple patterns
        vmaf_patterns = [
            'VMAF score:',
            'Global VMAF score:',
            'aggregate VMAF:',
            'mean VMAF:'
        ]
        
        vmaf_score = None
        if stderr_text:
            for pattern in vmaf_patterns:
                for line in reversed(stderr_text.splitlines()):
                    if pattern in line:
                        score_match = re.search(r'([0-9]+\.[0-9]+)', line)
                        if score_match:
                            vmaf_score = float(score_match.group(1))
                            break
                if vmaf_score is not None:
                    break
        
        if vmaf_score is None:
            print("[WARN] Could not parse VMAF score from output")
            if DEBUG and stderr_text:
                print("[DEBUG] VMAF stderr for parsing:")
                for l in stderr_text.splitlines()[-5:]:
                    print("  " + l)
        
        return vmaf_score
        
    except Exception as e:
        print(f"[ERR] VMAF computation error: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None
    finally:
        # Clean up scaled reference if created
            if scaled_reference and scaled_reference.exists():
                try:
                    scaled_reference.unlink()
                    TEMP_FILES.discard(str(scaled_reference))
                    if DEBUG:
                        print(f"[VMAF] Cleaned up temporary scaled reference")
                except Exception as e:
                    if DEBUG:
                        print(f"[DEBUG] Failed to cleanup scaled reference: {e}")
def compute_vmaf_score_multiprocess(reference: Path, distorted: Path, n_threads: int = 8) -> float | None:
    """
    Alternative VMAF implementation using segment-based parallel processing.
    This can better utilize multiple cores by splitting the video into segments.
    Falls back to standard libvmaf if segmentation fails.
    """
    if n_threads <= 1:
        return compute_vmaf_score(reference, distorted, n_threads=n_threads)
    
    try:
        # For very long videos, segment-based processing can be more efficient
        # Get video duration first
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(reference)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            # Fall back to standard method if probe fails
            return compute_vmaf_score(reference, distorted, n_threads=n_threads)
            
        try:
            duration = float(result.stdout.strip())
        except (ValueError, AttributeError):
            return compute_vmaf_score(reference, distorted, n_threads=n_threads)
        
        # Only use segment-based processing for longer videos (>5 minutes)
        # and when we have sufficient threads
        if duration < 300 or n_threads < 4:
            return compute_vmaf_score(reference, distorted, n_threads=n_threads)
        
        # For now, fall back to enhanced standard method
        # Segment-based processing is complex and may not provide consistent results
        return compute_vmaf_score(reference, distorted, n_threads=n_threads)
        
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Multiprocess VMAF fallback: {e}")
        return compute_vmaf_score(reference, distorted, n_threads=n_threads)

@dataclass
class TranscodeJob:
    """Represents a transcoding job with its verification"""
    file: Path
    temp_file: Path
    backup_file: Path
    encode_future: Optional[concurrent.futures.Future] = None
    encode_success: bool = False
    vmaf_score: Optional[float] = None
    completed: bool = False

@dataclass
class StagedFile:
    """Represents a file staged for processing"""
    original_path: Path
    staged_path: Path
    size_bytes: int
    transfer_future: Optional[concurrent.futures.Future] = None
    ready: bool = False

class AsyncFileStager:
    """
    Async file transfer queue that maintains a buffer of ready-to-process files.
    Pre-stages files from network/slow drives to local temp storage for faster processing.
    """
    
    def __init__(self, buffer_size: int = 3, temp_dir: Optional[Path] = None):
        self.buffer_size = buffer_size
        self.temp_dir = temp_dir or Path.cwd() / "temp_transcode"
        self.staged_files: queue.Queue[StagedFile] = queue.Queue()
        self.pending_transfers: List[StagedFile] = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="FileStager")
        self.active = True
        self._lock = threading.Lock()
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        print(f"[STAGING] Temp directory: {self.temp_dir}")
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _copy_file_with_progress(self, src: Path, dst: Path, size_bytes: Optional[int] = None) -> bool:
        """Copy file with progress tracking using multiple optimized methods"""
        try:
            # Use cached size if provided, otherwise get it (but avoid repeated calls)
            if size_bytes is None:
                size_bytes = src.stat().st_size
            
            print(f"[STAGING] Transferring {src.name} ({format_size(size_bytes)})...")
            
            # Method 1: Try PowerShell native copy (often fastest for Windows network drives)
            if os.name == 'nt':
                try:
                    cmd = [
                        "powershell", "-Command",
                        f"Copy-Item -Path '{src}' -Destination '{dst}' -Force"
                    ]
                    
                    if DEBUG:
                        print(f"[STAGING-CMD] {' '.join(cmd)}")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                    
                    if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                        print(f"[STAGING] ✓ Ready: {src.name} (PowerShell copy)")
                        return True
                    else:
                        if DEBUG:
                            print(f"[STAGING] PowerShell copy failed (code {result.returncode}): {result.stderr}")
                except Exception as e:
                    if DEBUG:
                        print(f"[STAGING] PowerShell copy exception: {e}")
            
            # Method 2: Try FFmpeg copy (you mentioned it works at 500 Mbps)
            try:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src),
                    "-c", "copy",  # Stream copy (no re-encoding)
                    "-avoid_negative_ts", "make_zero",
                    str(dst)
                ]
                
                if DEBUG:
                    print(f"[STAGING-CMD] {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                    print(f"[STAGING] ✓ Ready: {src.name} (FFmpeg copy)")
                    return True
                else:
                    if DEBUG:
                        print(f"[STAGING] FFmpeg copy failed (code {result.returncode}): {result.stderr}")
            except Exception as e:
                if DEBUG:
                    print(f"[STAGING] FFmpeg copy exception: {e}")
            
            # Method 3: Try optimized robocopy with maximum performance flags
            if os.name == 'nt':
                try:
                    cmd = [
                        "robocopy", 
                        str(src.parent), str(dst.parent), src.name,
                        "/J",          # Unbuffered I/O for large files
                        "/MT:32",      # Maximum multi-threading (32 threads)
                        "/R:1", "/W:1", # Minimal retries
                        "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS"  # No logging
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                    if result.returncode in [0, 1]:  # 0=no copy needed, 1=files copied
                        print(f"[STAGING] ✓ Ready: {src.name} (robocopy)")
                        return True
                except Exception as e:
                    if DEBUG:
                        print(f"[STAGING] Robocopy exception: {e}")
            
            print(f"[STAGING] ✗ All transfer methods failed: {src.name}")
            return False
                
        except Exception as e:
            print(f"[STAGING] ✗ Transfer error for {src.name}: {e}")
            return False
    
    def stage_file(self, file_path: Path) -> StagedFile:
        """Queue a file for staging and return StagedFile object"""
        staged_path = self.temp_dir / file_path.name
        # Get size once and cache it to avoid repeated network access
        size_bytes = file_path.stat().st_size
        
        staged_file = StagedFile(
            original_path=file_path,
            staged_path=staged_path,
            size_bytes=size_bytes
        )
        
        # Submit transfer task with cached size to avoid repeated network access
        future = self.executor.submit(self._copy_file_with_progress, file_path, staged_path, size_bytes)
        staged_file.transfer_future = future
        
        with self._lock:
            self.pending_transfers.append(staged_file)
        
        return staged_file
    
    def get_ready_file(self, timeout: float = 300) -> Optional[StagedFile]:
        """Get next ready file, blocking until available or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout and self.active:
            # Check if any pending transfers completed
            with self._lock:
                completed = []
                for staged_file in self.pending_transfers[:]:
                    if staged_file.transfer_future and staged_file.transfer_future.done():
                        try:
                            success = staged_file.transfer_future.result()
                            if success:
                                staged_file.ready = True
                                self.staged_files.put(staged_file)
                                print(f"[STAGING] Buffer: {self.staged_files.qsize()}/{self.buffer_size} files ready")
                            completed.append(staged_file)
                        except Exception as e:
                            print(f"[STAGING] Transfer failed: {e}")
                            completed.append(staged_file)
                
                # Remove completed transfers
                for staged_file in completed:
                    self.pending_transfers.remove(staged_file)
            
            # Try to get a ready file
            try:
                return self.staged_files.get_nowait()
            except queue.Empty:
                time.sleep(0.5)  # Brief wait before checking again
        
        return None
    
    def cleanup_staged_file(self, staged_file: StagedFile):
        """Clean up a staged file after processing"""
        try:
            if staged_file.staged_path.exists():
                staged_file.staged_path.unlink()
                print(f"[STAGING] Cleaned up {staged_file.staged_path.name}")
        except Exception as e:
            print(f"[STAGING] Cleanup error: {e}")
    
    def maintain_buffer(self, file_queue: List[Path]):
        """Maintain buffer of staged files from remaining queue"""
        with self._lock:
            total_pending = len(self.pending_transfers)
            ready_count = self.staged_files.qsize()
            
            # Calculate how many more transfers we should start
            slots_needed = self.buffer_size - (total_pending + ready_count)
            
            if slots_needed > 0 and file_queue:
                files_to_stage = file_queue[:slots_needed]
                for file_path in files_to_stage:
                    self.stage_file(file_path)
                    file_queue.remove(file_path)
                    
                print(f"[STAGING] Queued {len(files_to_stage)} more files for transfer")
    
    def cleanup(self):
        """Clean up temp directory and executor"""
        self.active = False
        
        # Wait for pending transfers
        with self._lock:
            for staged_file in self.pending_transfers:
                if staged_file.transfer_future:
                    try:
                        staged_file.transfer_future.result(timeout=10)
                    except:
                        pass
        
        # Clean up temp directory
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"[STAGING] Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"[STAGING] Cleanup warning: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)

class ParallelTranscoder:
    """Parallel transcoder that pipelines encoding (GPU) and VMAF verification (CPU)"""
    
    def __init__(self, vmaf_threads: int = 8, use_experimental_vmaf: bool = False):
        """
        Initialize ParallelTranscoder with VMAF threading support.
        
        Args:
            vmaf_threads: Number of threads to use for VMAF computation (default 8)
            use_experimental_vmaf: Use experimental multiprocess VMAF implementation
        """
        # Lock for file operations (move, delete)
        self.file_lock = threading.Lock()
        self.vmaf_threads = vmaf_threads
        self.use_experimental_vmaf = use_experimental_vmaf
        
    def encode_file(self, job: TranscodeJob) -> bool:
        """Encode a single file. Returns True on success."""
        print(f"[ENCODE] Starting {job.file.name}")
        
        if job.temp_file.exists():
            try: 
                job.temp_file.unlink()
            except: 
                pass
                
        # This calls the existing encode_with_progress function
        success = encode_with_progress(
            job.file, job.temp_file, self.encoder, self.encoder_type, 
            self.qp, preserve_hdr_metadata=self.preserve_hdr_metadata
        )
        
        if success and job.temp_file.exists():
            print(f"[ENCODE] Completed {job.file.name}")
            return True
        else:
            print(f"[ENCODE] Failed {job.file.name}")
            if job.temp_file.exists():
                try: 
                    job.temp_file.unlink()
                except: 
                    pass
            return False
    
    def verify_quality(self, job: TranscodeJob, vmaf_threads: int = 8, use_experimental: bool = False) -> Optional[float]:
        """Run VMAF verification on encoded file. Returns VMAF score or None."""
        if not job.temp_file.exists():
            return None
            
        method = "experimental" if use_experimental else "standard"
        print(f"[VMAF] Verifying {job.file.name} ({vmaf_threads} threads, {method})")
        
        # Use enhanced compute_vmaf_score function with proper threading
        if use_experimental:
            vmaf_score = compute_vmaf_score_multiprocess(job.file, job.temp_file, n_threads=vmaf_threads)
        else:
            vmaf_score = compute_vmaf_score(job.file, job.temp_file, n_threads=vmaf_threads)
        
        if vmaf_score is not None:
            print(f"[VMAF] {job.file.name}: {vmaf_score:.2f}")
        else:
            print(f"[VMAF] Failed to compute VMAF for {job.file.name}")
            
        return vmaf_score
    
    def finalize_job(self, job: TranscodeJob) -> bool:
        """Atomically replace original with transcoded version if quality is acceptable."""
        if not job.encode_success or not job.temp_file.exists():
            print(f"[SKIP] {job.file.name} - encoding failed")
            return False
            
        # Check VMAF quality
        if job.vmaf_score is not None:
            if job.vmaf_score < self.vmaf_min:
                print(f"[SKIP] {job.file.name} - VMAF {job.vmaf_score:.2f} below minimum {self.vmaf_min}")
                try: 
                    job.temp_file.unlink()
                except: 
                    pass
                return False
        else:
            print(f"[WARN] {job.file.name} - no VMAF score, proceeding anyway")
        
        # Atomic swap with file lock
        with self.file_lock:
            try:
                if job.backup_file.exists():
                    job.backup_file.unlink()
                
                # Atomic-ish swap: original -> backup, temp -> original
                job.file.rename(job.backup_file)
                job.temp_file.rename(job.file)
                
                # Remove backup
                try: 
                    job.backup_file.unlink()
                except: 
                    pass
                    
                print(f"[SUCCESS] Replaced {job.file.name}")
                return True
                
            except Exception as e:
                print(f"[ERROR] Failed to replace {job.file.name}: {e}")
                # Try to restore from backup
                if job.backup_file.exists() and not job.file.exists():
                    try:
                        job.backup_file.rename(job.file)
                    except:
                        pass
                return False
    
    def process_files(self, files: List[Path]) -> int:
        """
        Process files with pipelined encoding and VMAF verification.
        Returns number of successfully processed files.
        """
        if not files:
            return 0
            
        jobs = []
        for f in files:
            job = TranscodeJob(
                file=f,
                temp_file=f.with_name(f.stem + ".transcode" + f.suffix),
                backup_file=f.with_name(f.stem + ".bak" + f.suffix)
            )
            jobs.append(job)
        
        success_count = 0
        
        with tqdm(total=len(files), desc="Transcoding Files", position=0) as overall:
            
            # Process jobs in pipeline fashion
            for i, job in enumerate(jobs):
                
                # Update progress with current file
                overall.set_description(f"Transcoding Files - Encoding {job.file.name[:30]}...")
                
                # Start encoding current file
                job.encode_success = self.encode_file(job)
                
                # If encoding succeeded, do VMAF verification immediately
                if job.encode_success:
                    overall.set_description(f"Transcoding Files - VMAF check {job.file.name[:25]}...")
                    job.vmaf_score = self.verify_quality(job, self.vmaf_threads, self.use_experimental_vmaf)
                    
                    # Finalize the job immediately
                    if self.finalize_job(job):
                        success_count += 1
                        overall.set_description(f"Transcoding Files - Success: {job.file.name[:25]}")
                    else:
                        overall.set_description(f"Transcoding Files - Failed quality: {job.file.name[:20]}")
                    job.completed = True
                else:
                    overall.set_description(f"Transcoding Files - Encode failed: {job.file.name[:20]}")
                
                overall.update(1)
        
        return success_count

    def process_files_with_qp_map(self, files: List[Path], qp_map: dict, encoder: str, encoder_type: str, 
                                  vmaf_target: float, vmaf_min: float, preserve_hdr: bool) -> int:
        """
        Process files with per-file QP mapping in parallel.
        Returns number of successfully processed files.
        """
        if not files:
            return 0
            
        jobs = []
        for f in files:
            job = TranscodeJob(
                file=f,
                temp_file=f.with_name(f.stem + ".transcode" + f.suffix),
                backup_file=f.with_name(f.stem + ".bak" + f.suffix)
            )
            jobs.append(job)
        
        success_count = 0
        
        with tqdm(total=len(files), desc="Transcoding with Optimized QPs", position=0) as overall:
            
            # Process jobs in pipeline fashion
            for i, job in enumerate(jobs):
                file_qp = qp_map.get(job.file, 24)  # fallback to QP 24
                
                # Update progress with current file and QP
                overall.set_description(f"Transcoding QP{file_qp} - Encoding {job.file.name[:25]}...")
                print(f"[ENCODE] Starting {job.file.name} with QP {file_qp}")
                
                # Start encoding current file with its specific QP
                if job.temp_file.exists():
                    try: 
                        job.temp_file.unlink()
                    except: 
                        pass
                        
                job.encode_success = encode_with_progress(
                    job.file, job.temp_file, encoder, encoder_type, 
                    file_qp, preserve_hdr_metadata=preserve_hdr
                )
                
                if job.encode_success:
                    print(f"[ENCODE] Success: {job.file.name}")
                    overall.set_description(f"Transcoding QP{file_qp} - VMAF check {job.file.name[:20]}...")
                else:
                    print(f"[ENCODE] Failed: {job.file.name}")
                    overall.set_description(f"Transcoding QP{file_qp} - Encode failed {job.file.name[:15]}")
                
                # If encoding succeeded, do VMAF verification immediately with threading
                if job.encode_success:
                    vmaf_func = compute_vmaf_score_multiprocess if self.use_experimental_vmaf else compute_vmaf_score
                    job.vmaf_score = vmaf_func(job.file, job.temp_file, n_threads=self.vmaf_threads)
                    method_str = f"({self.vmaf_threads} threads, {'experimental' if self.use_experimental_vmaf else 'standard'})"
                    if job.vmaf_score is not None:
                        print(f"[VMAF] {job.file.name}: {job.vmaf_score:.2f} {method_str}")
                        overall.set_description(f"Transcoding QP{file_qp} - VMAF {job.vmaf_score:.1f} {job.file.name[:15]}")
                    else:
                        print(f"[VMAF] Failed to compute VMAF for {job.file.name}")
                        overall.set_description(f"Transcoding QP{file_qp} - VMAF failed {job.file.name[:15]}")
                    
                    # Finalize the job immediately
                    if self.finalize_job_with_vmaf_check(job, vmaf_min):
                        success_count += 1
                        overall.set_description(f"Transcoding QP{file_qp} - Success: {job.file.name[:20]}")
                    else:
                        overall.set_description(f"Transcoding QP{file_qp} - Failed quality: {job.file.name[:15]}")
                    job.completed = True
                
                overall.update(1)
        
        return success_count
    
    def finalize_job_with_vmaf_check(self, job: TranscodeJob, vmaf_min: float) -> bool:
        """Atomically replace original with transcoded version if quality is acceptable."""
        if not job.encode_success or not job.temp_file.exists():
            print(f"[SKIP] {job.file.name} - encoding failed")
            return False
            
        # Check VMAF quality
        if job.vmaf_score is not None:
            if job.vmaf_score < vmaf_min:
                print(f"[SKIP] {job.file.name} - VMAF {job.vmaf_score:.2f} below minimum {vmaf_min}")
                try: 
                    job.temp_file.unlink()
                except: 
                    pass
                return False
        else:
            print(f"[WARN] {job.file.name} - no VMAF score, proceeding anyway")
        
        # Atomic swap with file lock
        with self.file_lock:
            try:
                if job.backup_file.exists():
                    job.backup_file.unlink()
                
                # Atomic-ish swap: original -> backup, temp -> original
                job.file.rename(job.backup_file)
                job.temp_file.rename(job.file)
                
                # Remove backup
                try: 
                    job.backup_file.unlink()
                except: 
                    pass
                
                print(f"[SUCCESS] {job.file.name} transcoded successfully")
                return True
                
            except Exception as e:
                print(f"[ERROR] Failed to finalize {job.file.name}: {e}")
                try:
                    # Restore original if possible
                    if job.backup_file.exists():
                        job.backup_file.rename(job.file)
                    if job.temp_file.exists():
                        job.temp_file.unlink()
                except:
                    pass
                return False

def test_qp_on_sample(file: Path, qp: int, encoder: str, encoder_type: str, sample_duration: int = 60, vmaf_threads: int = 8, preserve_hdr_metadata: bool = True, use_clip_as_sample: bool = False) -> tuple[float | None, int, int]:
    """Test a QP on a sample clip and return (VMAF score, encoded sample size, original sample size)"""
    # Use VMAfEvaluator for centralized evaluation logic
    evaluator = VMAfEvaluator(debug=DEBUG, temp_files=TEMP_FILES, baseline_vmaf=BASELINE_VMAF)
    
    # Create encoder command builder function
    def encode_cmd_builder(sample_clip, encoded_sample, enc, enc_type, qp_val, preserve_hdr):
        return build_encode_cmd(sample_clip, encoded_sample, enc, enc_type, qp_val, preserve_hdr_metadata=preserve_hdr)
    
    result = evaluator.test_qp_on_sample(
        file, qp, encoder, encoder_type, sample_duration, vmaf_threads,
        preserve_hdr_metadata, use_clip_as_sample, encode_cmd_builder
    )
    
    return result.vmaf_score, result.encoded_size, result.reference_size

def find_optimal_qp(files: list[Path], encoder: str, encoder_type: str, 
                   candidate_qps: list[int] = None, 
                   vmaf_target: float = 95.0, 
                   vmaf_min_threshold: float = 93.0,
                   sample_duration: int = 60,
                   min_qp_limit: int = 10,
                   max_qp_limit: int = 45,
                   vmaf_threads: int = 0,
                   preserve_hdr_metadata: bool = True) -> int:
    """Find optimal QP using VMAF scoring across sample files"""
    if candidate_qps is None:
        candidate_qps = CANDIDATE_QPS
        
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
                    vmaf_score, encoded_size, sample_orig_size = test_qp_on_sample(file, qp, encoder, encoder_type, sample_duration=sample_duration, vmaf_threads=vmaf_threads, preserve_hdr_metadata=preserve_hdr_metadata)
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
    
    # Print initial results table
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

    # If no QP passed quality thresholds, attempt downward expansion (lower QP => higher quality)
    if not any_pass:
        print("\n[ADAPT] No candidate QP met quality thresholds. Expanding downward for higher quality...")
        base_step = candidate_qps[-1] - candidate_qps[-2] if len(candidate_qps) >= 2 else 2
        step = max(1, base_step)
        current_qp = min(candidate_qps) - step
        while current_qp >= min_qp_limit:
            print(f"[ADAPT-DOWN] Testing lower QP {current_qp} (step {step}) for quality recovery...")
            vmaf_scores = []
            total_sample_original_size = 0
            total_encoded_size = 0
            total_full_original_size = 0
            extrapolated_encoded_total = 0.0
            for file in files:
                vmaf_score, encoded_size, sample_orig_size = test_qp_on_sample(file, current_qp, encoder, encoder_type, sample_duration=sample_duration, vmaf_threads=vmaf_threads, preserve_hdr_metadata=preserve_hdr_metadata)
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                    total_encoded_size += encoded_size
                    total_sample_original_size += sample_orig_size
                    full_size = file.stat().st_size if file.exists() else 0
                    total_full_original_size += full_size
                    if sample_orig_size > 0 and full_size > 0:
                        extrapolated_encoded_total += (encoded_size / sample_orig_size) * full_size
            if vmaf_scores:
                mean_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                min_vmaf = min(vmaf_scores)
                compression_ratio = (total_encoded_size / total_sample_original_size) * 100 if total_sample_original_size > 0 else 0
                extrapolated_full_pct = (extrapolated_encoded_total / total_full_original_size) * 100 if total_full_original_size > 0 else 0
                qp_results[current_qp] = {
                    'mean_vmaf': mean_vmaf,
                    'min_vmaf': min_vmaf,
                    'compression_ratio': compression_ratio,
                    'extrapolated_full_pct': extrapolated_full_pct,
                    'sample_count': len(vmaf_scores)
                }
                meets = print_row(current_qp)
                if mean_vmaf >= vmaf_target and min_vmaf >= vmaf_min_threshold:
                    selected_qp = current_qp
                    any_pass = True
                    print("[ADAPT-DOWN] Found acceptable quality.")
                    break
            current_qp -= step
        if not any_pass:
            print("[NOTICE] Even lowest tested QPs did not meet quality thresholds; proceeding with lowest QP (highest quality) anyway.")
            selected_qp = min(candidate_qps + ([current_qp] if current_qp >= min_qp_limit else []))

    # Adaptive expansion if still inefficient
    def is_inefficient(qp: int) -> bool:
        res = qp_results.get(qp)
        if not res:
            return False
        return res['compression_ratio'] > 100 or res.get('extrapolated_full_pct', 0) > 100

    if any_pass and is_inefficient(selected_qp):
        print("\n[ADAPT] Selected QP is inefficient (>100%). Attempting adaptive expansion...")
        base_step = candidate_qps[-1] - candidate_qps[-2] if len(candidate_qps) >= 2 else 2
        step = max(1, base_step)
        current_qp = max(candidate_qps) + step
        max_allowed_qp = max_qp_limit  # configurable upper bound
        while current_qp <= max_allowed_qp:
            print(f"[ADAPT] Testing higher QP {current_qp} (step {step})...")
            # Evaluate this QP (reuse logic from earlier loop)
            vmaf_scores = []
            total_sample_original_size = 0
            total_encoded_size = 0
            total_full_original_size = 0
            extrapolated_encoded_total = 0.0
            for file in files:
                vmaf_score, encoded_size, sample_orig_size = test_qp_on_sample(file, current_qp, encoder, encoder_type, sample_duration=sample_duration, vmaf_threads=vmaf_threads, preserve_hdr_metadata=preserve_hdr_metadata)
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                    total_encoded_size += encoded_size
                    total_sample_original_size += sample_orig_size
                    full_size = file.stat().st_size if file.exists() else 0
                    total_full_original_size += full_size
                    if sample_orig_size > 0 and full_size > 0:
                        extrapolated_encoded_total += (encoded_size / sample_orig_size) * full_size
            if vmaf_scores:
                mean_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                min_vmaf = min(vmaf_scores)
                compression_ratio = (total_encoded_size / total_sample_original_size) * 100 if total_sample_original_size > 0 else 0
                extrapolated_full_pct = (extrapolated_encoded_total / total_full_original_size) * 100 if total_full_original_size > 0 else 0
                qp_results[current_qp] = {
                    'mean_vmaf': mean_vmaf,
                    'min_vmaf': min_vmaf,
                    'compression_ratio': compression_ratio,
                    'extrapolated_full_pct': extrapolated_full_pct,
                    'sample_count': len(vmaf_scores)
                }
                meets = print_row(current_qp)
                if meets:
                    selected_qp = current_qp
                    if not is_inefficient(current_qp):
                        print("[ADAPT] Achieved efficient QP.")
                        break
                    # Still inefficient but passes quality: increase step (like adaptive LR growth)
                    step = min(step * 2, 8)
                    current_qp += step
                    continue
                else:
                    # Quality failed; reduce step (like learning rate decay)
                    step = max(1, step // 2)
                    if step == 1:
                        print("[ADAPT] Quality threshold failed at smallest step; stopping expansion.")
                        break
                    current_qp += step
                    continue
            else:
                print(f"[ADAPT] No VMAF scores at QP {current_qp}; stopping.")
                break

    print(f"\n[SELECTED] QP {selected_qp} (target VMAF >= {vmaf_target}, min >= {vmaf_min_threshold})")
    sel_res = qp_results.get(selected_qp)
    if sel_res and (sel_res['compression_ratio'] > 100 or sel_res.get('extrapolated_full_pct', 0) > 100):
        print("[NOTICE] Final selected QP still inefficient (>100%). Consider allowing even higher QP or different encoding strategy.")
    return selected_qp

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
                              disable_auto_scale: bool = False,
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
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode == 0 and clip_path.exists():
                clips.append(clip_path)
            else:
                print(f"[WARN] Failed to extract clip {i} from {file.name}")
                if clip_path in TEMP_FILES:
                    TEMP_FILES.discard(str(clip_path))
        
        return clips
    
    def evaluate_qp_on_clips(clips: List[Path], qp: int, qp_cache: dict = None) -> dict | None:
        """Test a QP on multiple clips and return aggregated results with early failure detection."""
        if qp_cache is None:
            qp_cache = {}
        
        # Check memory cache first to avoid redundant encoding within same run
        if qp in qp_cache:
            print(f"[MEMORY-CACHE] Using cached result for QP{qp}")
            return qp_cache[qp].copy()
        
        vmaf_scores: list[float] = []
        total_clip_original_size = 0
        total_encoded_size = 0
        worst_vmaf_drop = 0.0
        
        # Early failure threshold: if VMAF is this far below target, likely hopeless
        early_fail_threshold = vmaf_target - 20.0
        
        # Progress bar per QP evaluation (quiet if only one clip)
        pbar = tqdm(total=len(clips), desc=f"QP{qp} clips", position=2, leave=False) if len(clips) > 1 else None
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
                    
                    # Early failure detection: if first clip shows hopeless case, skip remaining clips
                    if idx == 0 and vmaf_score < early_fail_threshold and qp <= (min_qp_limit + 2):
                        if pbar:
                            pbar.set_description(f"QP{qp} clips - Early fail: VMAF {vmaf_score:.1f} << {vmaf_target:.1f}")
                        else:
                            print(f"  [EARLY-FAIL] VMAF {vmaf_score:.1f} too far below target {vmaf_target:.1f} at low QP{qp}")
                        
                        # Skip remaining clips for this hopeless QP
                        if pbar:
                            pbar.update(len(clips) - idx - 1)  # Fast-forward progress bar
                        break
                
                if pbar: pbar.update(1)
        finally:
            if pbar: pbar.close()
        
        if not vmaf_scores:
            return None
            
        mean_vmaf = sum(vmaf_scores) / len(vmaf_scores)
        min_vmaf = min(vmaf_scores)
        
        # Estimate full file size based on clip compression ratio
        full_file_size = file.stat().st_size if file.exists() else 0
        if total_clip_original_size > 0 and full_file_size > 0:
            compression_ratio = total_encoded_size / total_clip_original_size
            estimated_full_encoded_size = compression_ratio * full_file_size
            size_pct = (estimated_full_encoded_size / full_file_size) * 100
        else:
            size_pct = 100
            
        result = {
            'mean_vmaf': mean_vmaf,
            'min_vmaf': min_vmaf,
            'size_pct': size_pct,
            'passes_quality': mean_vmaf >= vmaf_target and min_vmaf >= vmaf_min_threshold,
            'samples_used': len(vmaf_scores),
            'worst_drop': worst_vmaf_drop
        }
        
        # Store result in memory cache for current run only (no disk cache files)
        if qp_cache is not None:
            qp_cache[qp] = result.copy()
        
        return result
    
    print(f"[GRADIENT-DESCENT] Starting 2D gradient descent QP optimization for {file.name}")
    
    # Auto-scale clips based on file duration
    duration = get_duration_sec(file)
    samples_was_explicitly_set = '--samples' in sys.argv
    
    if duration > 0 and not samples_was_explicitly_set and not disable_auto_scale:
        auto_clips = max(3, min(samples * 2, int(duration / 1200)))  # 1200s = 20min
        if auto_clips != samples:
            print(f"[INFO] Auto-scaling clips from {samples} to {auto_clips} based on {duration/60:.1f}min duration")
            samples = auto_clips
    
    # Extract random clips
    clips = extract_random_clips(file, samples, sample_duration)
    if not clips:
        print(f"[ERROR] Could not extract any clips from {file.name}")
        return initial_qp, None
    
    print(f"[INFO] Extracted {len(clips)} clips for gradient descent optimization")
    
    # Fast initial assessment: test lowest QP on just one clip to detect hopeless cases early
    print(f"[GRADIENT-DESCENT] Fast initial assessment at QP{min_qp_limit}...")
    first_clip = clips[0]
    initial_vmaf, _, _ = test_qp_on_sample(
        first_clip, min_qp_limit, encoder, encoder_type, 
        sample_duration=sample_duration, vmaf_threads=vmaf_threads, 
        preserve_hdr_metadata=preserve_hdr_metadata, use_clip_as_sample=True
    )
    
    if initial_vmaf is not None and initial_vmaf < (vmaf_target - 25.0):
        print(f"[GRADIENT-DESCENT] Early termination: VMAF {initial_vmaf:.1f} at lowest QP{min_qp_limit} is {vmaf_target - initial_vmaf:.1f} points below target")
        print(f"[GRADIENT-DESCENT] {file.name}: QP {min_qp_limit} (early fail - content cannot reach target quality)")
        # Clean up clips
        if not DEBUG:
            for clip in clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
        return min_qp_limit, None
    else:
        print(f"[GRADIENT-DESCENT] Initial assessment passed: VMAF {initial_vmaf:.1f} at QP{min_qp_limit}")
    
    # Initialize gradient descent parameters with adaptive settings
    qp = float(initial_qp)  # Allow fractional QP for gradient computation
    velocity_qp = 0.0  # Momentum term
    
    # Store evaluation history and cache QP results to avoid re-encoding
    history: List[tuple[int, float, float, float]] = []  # (qp_int, vmaf, size_pct, loss)
    qp_cache = {}  # Cache results to avoid re-encoding same QP
    best_qp = initial_qp
    best_result = None
    
    # Pre-populate cache with initial assessment if we did QP at min_qp_limit
    if initial_vmaf is not None and initial_qp == min_qp_limit:
        # We already tested min_qp_limit, so add this to cache to avoid re-encoding
        quick_result = {
            'mean_vmaf': initial_vmaf,
            'min_vmaf': initial_vmaf,
            'size_pct': 95.0,  # rough estimate
            'passes_quality': initial_vmaf >= vmaf_target and initial_vmaf >= vmaf_min_threshold,
            'samples_used': 1,
            'worst_drop': 100.0 - initial_vmaf
        }
        qp_cache[min_qp_limit] = quick_result
    
    # Adaptive parameters for faster convergence
    adaptive_learning_rate = learning_rate * 2.0  # Start with higher learning rate
    consecutive_oscillations = 0
    last_qp_direction = 0
    stagnation_count = 0
    
    print(f"[GRADIENT-DESCENT] Target VMAF: {vmaf_target:.1f}, Min VMAF: {vmaf_min_threshold:.1f}")
    print(f"[GRADIENT-DESCENT] Adaptive learning rate: {adaptive_learning_rate:.3f}, Momentum: {momentum:.2f}, Tolerance: {tolerance:.1f}")
    
    try:
        with tqdm(total=max_iterations, desc=f"GD Search {file.name[:35]}", position=0, leave=False) as pbar:
            for iteration in range(max_iterations):
                # Convert to integer QP for evaluation (hardware encoders need integer QP)
                qp_int = max(min_qp_limit, min(max_qp_limit, int(round(qp))))
                
                pbar.set_description(f"GD Search {file.name[:30]} - QP{qp_int} (iter {iteration+1})")
                
                # Evaluate current QP using cache
                result = evaluate_qp_on_clips(clips, qp_int, qp_cache)
                if result is None:
                    print(f"[WARN] Failed to evaluate QP {qp_int}")
                    break
                
                vmaf = result['mean_vmaf']
                size_pct = result['size_pct']
                
                # Calculate loss function (minimize VMAF error, prefer higher compression)
                vmaf_error = abs(vmaf - vmaf_target)
                size_penalty = max(0, size_pct - 100) * 0.1  # Penalize size >100%
                compression_bonus = max(0, (100 - size_pct) * 0.02)  # Reward compression
                loss = vmaf_error + size_penalty - compression_bonus
                
                # Store in history
                history.append((qp_int, vmaf, size_pct, loss))
                
                # Update best result if quality thresholds are met
                if result['passes_quality'] and (best_result is None or loss < best_result.get('loss', float('inf'))):
                    best_qp = qp_int
                    best_result = result.copy()
                    best_result['loss'] = loss
                
                pbar.set_description(f"GD Search {file.name[:25]} - QP{qp_int}: VMAF {vmaf:.1f}, Loss {loss:.2f}")
                
                # Fast convergence check
                if vmaf_error <= tolerance:
                    print(f"[GRADIENT-DESCENT] Converged: VMAF {vmaf:.2f} within tolerance {tolerance}")
                    break
                
                # Detect oscillation patterns for early termination
                if len(history) >= 4:
                    last_qps = [h[0] for h in history[-4:]]
                    # Check for oscillation between two QP values
                    if len(set(last_qps)) == 2 and last_qps[-1] == last_qps[-3]:
                        # Pick the better one and stop
                        recent_results = [(h[0], h[3]) for h in history[-4:]]  # (qp, loss)
                        best_recent = min(recent_results, key=lambda x: x[1])
                        print(f"[GRADIENT-DESCENT] Oscillation detected, stopping at QP{best_recent[0]}")
                        # Update to the best oscillating QP
                        for h in reversed(history):
                            if h[0] == best_recent[0]:
                                best_qp = h[0]
                                best_result = {'mean_vmaf': h[1], 'size_pct': h[2], 'passes_quality': True}
                                best_result['loss'] = h[3]
                                break
                        break
                
                # Adaptive gradient computation
                gradient_qp = 0.0
                current_direction = 0
                
                if len(history) >= 2:
                    # Use previous point for gradient estimation
                    prev_qp, prev_vmaf, prev_size, prev_loss = history[-2]
                    curr_qp, curr_vmaf, curr_size, curr_loss = history[-1]
                    
                    if curr_qp != prev_qp:
                        # dLoss/dQP - more aggressive gradient
                        gradient_qp = (curr_loss - prev_loss) / (curr_qp - prev_qp)
                        current_direction = 1 if (curr_qp - prev_qp) > 0 else -1
                        
                        # Detect direction changes for oscillation tracking
                        if last_qp_direction != 0 and current_direction != last_qp_direction:
                            consecutive_oscillations += 1
                        else:
                            consecutive_oscillations = 0
                        last_qp_direction = current_direction
                else:
                    # First iteration: more aggressive initial step
                    if vmaf < vmaf_target:
                        # Need higher quality -> DECREASE QP aggressively
                        gradient_qp = -2.0  # More aggressive initial step
                    else:
                        # Quality is good -> try higher QP for more compression
                        gradient_qp = 1.5   # More aggressive step
                
                # Dynamic learning rate adjustment
                if consecutive_oscillations >= 2:
                    # Reduce learning rate when oscillating
                    adaptive_learning_rate *= 0.5
                    consecutive_oscillations = 0
                    print(f"[GRADIENT-DESCENT] Reducing learning rate to {adaptive_learning_rate:.3f} due to oscillation")
                elif len(history) >= 3:
                    # Adjust based on loss progression
                    recent_losses = [h[3] for h in history[-3:]]
                    if recent_losses[-1] < recent_losses[-2]:  # Improving
                        adaptive_learning_rate *= 1.1  # Increase learning rate
                    elif recent_losses[-1] > recent_losses[-2] * 1.1:  # Getting worse
                        adaptive_learning_rate *= 0.7  # Decrease learning rate
                
                # Update velocity with momentum
                velocity_qp = momentum * velocity_qp - adaptive_learning_rate * gradient_qp
                
                # Update QP with larger steps for faster convergence
                qp += velocity_qp
                
                # Clamp to bounds
                qp = max(min_qp_limit, min(max_qp_limit, qp))
                
                if DEBUG:
                    print(f"[GD-DEBUG] Iter {iteration+1}: QP {qp_int} -> {qp:.2f}, VMAF {vmaf:.2f}, "
                          f"Loss {loss:.3f}, Grad {gradient_qp:.3f}, LR {adaptive_learning_rate:.4f}")
                
                pbar.update(1)
                
                # Early stopping for stagnation
                if len(history) >= 3:
                    last_qps = [h[0] for h in history[-3:]]
                    last_vmaf_errors = [abs(h[1] - vmaf_target) for h in history[-3:]]
                    
                    # Check if QP is stuck at same value
                    if len(set(last_qps)) == 1:
                        stagnation_count += 1
                        if stagnation_count >= 2:
                            print(f"[GRADIENT-DESCENT] Early stop: QP converged at {qp_int}")
                            break
                    else:
                        stagnation_count = 0
                    
                    # Check if VMAF error is barely improving
                    if len(last_vmaf_errors) >= 3 and all(abs(last_vmaf_errors[i] - last_vmaf_errors[i+1]) < 0.1 for i in range(2)):
                        print(f"[GRADIENT-DESCENT] Early stop: VMAF error improvement minimal")
                        break
                
                # Ensure QP makes meaningful progress by enforcing minimum step size
                qp_change = abs(qp - history[-1][0])
                if len(history) >= 2 and qp_change < 0.5:
                    # Force a larger step if we're making tiny progress
                    direction = -1 if vmaf < vmaf_target else 1  # Negative = decrease QP for higher quality
                    qp = history[-1][0] + direction * max(1.0, abs(velocity_qp))
                    qp = max(min_qp_limit, min(max_qp_limit, qp))
                    if DEBUG:
                        print(f"[GD-DEBUG] Forced larger step: QP {history[-1][0]} -> {qp:.2f}")
    
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
        rel_loss = 100 - result['mean_vmaf']
        worst_drop = result.get('worst_drop', 0)
        print(f"[GRADIENT-DESCENT] {file.name}: QP {final_qp} mean {result['mean_vmaf']:.2f} "
              f"(loss {rel_loss:.2f}), min {result['min_vmaf']:.2f}, worst drop {worst_drop:.2f}, "
              f"size {result['size_pct']:.1f}% (samples {result.get('samples_used', '?')})")
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


def adaptive_qp_search_per_file(file: Path, encoder: str, encoder_type: str,
                                samples: int = 6,  # number of 120-second clips
                                initial_qp: int = 24,
                                initial_step: float = 4.0,
                                min_step: float = 1.0,
                                vmaf_target: float = 95.0,
                                vmaf_min_threshold: float = 93.0,
                                sample_duration: int = 120,  # default 120 seconds (overridable with --sample-duration)
                                min_qp_limit: int = 10,
                                max_qp_limit: int = 45,
                                max_iterations: int = 25,
                                vmaf_threads: int = 0,
                                quick_sample_duration: int = 0,
                                quick_iters: int = 0,
                                scale_steps_by_vmaf: bool = False,
                                step_scale: float = 0.5,
                                max_dynamic_step: float = 8.0,
                                preserve_hdr_metadata: bool = True,
                                disable_auto_scale: bool = False,
                                use_gradient_descent: bool = True) -> tuple[int, dict | None]:
    """
    Adaptive QP search for a single file using multiple random 120-second clips.
    
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
            preserve_hdr_metadata=preserve_hdr_metadata,
            disable_auto_scale=disable_auto_scale
        )
    
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
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode == 0 and clip_path.exists():
                clips.append(clip_path)
            else:
                print(f"[WARN] Failed to extract clip {i} from {file.name}")
                if clip_path in TEMP_FILES:
                    TEMP_FILES.discard(str(clip_path))
        
        return clips

    def evaluate_qp_on_clips(clips: List[Path], qp: int) -> dict | None:
        """Test a QP on multiple clips and return aggregated results with early rejection."""
        # Verify all clips exist before starting
        missing_clips = [clip for clip in clips if not clip.exists()]
        if missing_clips:
            print(f"[ERROR] Missing clips before QP {qp} evaluation:")
            for clip in missing_clips:
                print(f"  - {clip}")
            return None
        
        vmaf_scores: list[float] = []
        total_clip_original_size = 0
        total_encoded_size = 0
        early_fail = False
        worst_vmaf_drop = 0.0  # Track largest quality drop from baseline
        
        # Progress bar per QP evaluation (quiet if only one clip)
        pbar = tqdm(total=len(clips), desc=f"QP{qp} clips", position=1, leave=False) if len(clips) > 1 else None
        try:
            for idx, clip in enumerate(clips):
                if pbar:
                    pbar.set_description(f"QP{qp} clips - Encoding clip {idx+1}/{len(clips)}")
                else:
                    # For single clip, print immediate feedback
                    print(f"  -> Encoding QP{qp} clip {idx+1}/{len(clips)}...")
                
                vmaf_score, encoded_size, clip_orig_size = test_qp_on_sample(
                    clip, qp, encoder, encoder_type, sample_duration=sample_duration, 
                    vmaf_threads=vmaf_threads, preserve_hdr_metadata=preserve_hdr_metadata,
                    use_clip_as_sample=True
                )
                
                if vmaf_score is None:
                    # Additional diagnosis for failed encoding
                    if not clip.exists():
                        print(f"[ERROR] Clip file missing during QP {qp} test: {clip}")
                    else:
                        print(f"[DEBUG] Clip exists but encoding failed: {clip} ({clip.stat().st_size} bytes)")
                
                if pbar:
                    if vmaf_score is not None:
                        pbar.set_description(f"QP{qp} clips - Clip {idx+1}: VMAF {vmaf_score:.1f}")
                        pbar.refresh()
                    else:
                        pbar.set_description(f"QP{qp} clips - Clip {idx+1}: VMAF failed")
                        pbar.refresh()
                else:
                    # For single clip, print result immediately
                    if vmaf_score is not None:
                        print(f"  OK Clip {idx+1}: VMAF {vmaf_score:.1f}")
                    else:
                        print(f"  ✗ Clip {idx+1}: VMAF failed")
                
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                    total_encoded_size += encoded_size
                    total_clip_original_size += clip_orig_size
                    
                    # Track worst case relative to baseline (~100)
                    baseline_score = 100.0  # Theoretical perfect score
                    drop = baseline_score - vmaf_score
                    worst_vmaf_drop = max(worst_vmaf_drop, drop)
                    
                    # Early rejection: if first (or any) clip catastrophically fails, abort rest
                    if idx == 0 and vmaf_score < (vmaf_min_threshold - 5):
                        early_fail = True
                        break
                else:
                    # Treat missing score as failure
                    if idx == 0:
                        early_fail = True
                        break
                if pbar: pbar.update(1)
        finally:
            if pbar: pbar.close()
        
        if not vmaf_scores:
            return None
            
        mean_vmaf = sum(vmaf_scores) / len(vmaf_scores)
        min_vmaf = min(vmaf_scores)
        
        # Estimate full file size based on clip compression ratio
        full_file_size = file.stat().st_size if file.exists() else 0
        if total_clip_original_size > 0 and full_file_size > 0:
            compression_ratio = total_encoded_size / total_clip_original_size
            estimated_full_encoded_size = compression_ratio * full_file_size
            size_pct = (estimated_full_encoded_size / full_file_size) * 100
        else:
            size_pct = 100
            
        return {
            'mean_vmaf': mean_vmaf,
            'min_vmaf': min_vmaf,
            'size_pct': size_pct,
            'passes_quality': mean_vmaf >= vmaf_target and min_vmaf >= vmaf_min_threshold,
            'inefficient': size_pct > 100,
            'early_fail': early_fail,
            'samples_used': len(vmaf_scores),
            'worst_drop': worst_vmaf_drop
        }

    print(f"[INFO] Analyzing {file.name} using {samples} random 120-second clips")
    
    # Auto-scale clips based on file duration for better coverage
    duration = get_duration_sec(file)
    # Only auto-scale clips if user didn't specify a specific number and didn't disable auto-scaling
    # Check if samples was explicitly set by user vs using default
    samples_was_explicitly_set = '--samples' in sys.argv
    
    if duration > 0 and not samples_was_explicitly_set and not disable_auto_scale:
        # Scale clips: 1 per 20 minutes, min 3, max samples*2
        auto_clips = max(3, min(samples * 2, int(duration / 1200)))  # 1200s = 20min
        if auto_clips != samples:
            print(f"[INFO] Auto-scaling clips from {samples} to {auto_clips} based on {duration/60:.1f}min duration (use --no-auto-scale-samples to disable)")
            samples = auto_clips
    elif (samples_was_explicitly_set or disable_auto_scale) and duration > 0:
        auto_clips = max(3, min(samples * 2, int(duration / 1200)))
        if auto_clips != samples:
            reason = "user specified --samples" if samples_was_explicitly_set else "--no-auto-scale-samples enabled"
            print(f"[INFO] Using {samples} samples ({reason}), auto-scaling would suggest {auto_clips} based on {duration/60:.1f}min duration")
    
    # Extract random clips
    clips = extract_random_clips(file, samples, sample_duration)
    if not clips:
        print(f"[ERROR] Could not extract any clips from {file.name}")
        return initial_qp
    
    print(f"[INFO] Extracted {len(clips)} clips for QP optimization")
    
    # Fast initial assessment: test lowest QP on just one clip to detect hopeless cases early
    print(f"[ADAPTIVE] Fast initial assessment at QP{min_qp_limit}...")
    first_clip = clips[0]
    initial_vmaf, _, _ = test_qp_on_sample(
        first_clip, min_qp_limit, encoder, encoder_type, 
        sample_duration=sample_duration, vmaf_threads=vmaf_threads, 
        preserve_hdr_metadata=preserve_hdr_metadata, use_clip_as_sample=True
    )
    
    if initial_vmaf is not None and initial_vmaf < (vmaf_target - 25.0):
        print(f"[ADAPTIVE] Early termination: VMAF {initial_vmaf:.1f} at lowest QP{min_qp_limit} is {vmaf_target - initial_vmaf:.1f} points below target")
        # Create minimal result for early failure
        early_fail_result = {
            'mean_vmaf': initial_vmaf,
            'min_vmaf': initial_vmaf,
            'size_pct': 95.0,  # rough estimate
            'passes_quality': False,
            'inefficient': False,
            'early_fail': True,
            'samples_used': 1,
            'worst_drop': vmaf_target - initial_vmaf
        }
        print(f"[RESULT] {file.name}: QP {min_qp_limit} mean {initial_vmaf:.2f} (loss {vmaf_target - initial_vmaf:.2f}), min {initial_vmaf:.2f}, worst drop {vmaf_target - initial_vmaf:.2f}, size 95.0% (samples 1)")
        
        # Clean up clips
        if not DEBUG:
            for clip in clips:
                try:
                    if clip.exists():
                        clip.unlink()
                    TEMP_FILES.discard(str(clip))
                except:
                    pass
        return min_qp_limit, early_fail_result
    else:
        print(f"[ADAPTIVE] Initial assessment passed: VMAF {initial_vmaf:.1f} at QP{min_qp_limit}")
    
    # Run adaptive QP search using the clips (similar logic to the existing function)
    qp_results: dict[int, dict] = {}
    tested_qps: set[int] = set()
    current_qp = max(min_qp_limit, min(max_qp_limit, initial_qp))
    direction: int | None = None
    last_good_qp: int | None = None
    best_efficient_qp: int | None = None
    direction_changes = 0
    step = float(initial_step)
    min_step = max(0.5, float(min_step))
    
    print(f"[ADAPTIVE] Starting QP optimization for {file.name} at QP {current_qp}")
    
    try:
        with tqdm(total=max_iterations, desc=f"QP Search {file.name[:40]}", position=0, leave=False) as adapt_bar:
            for it in range(1, max_iterations + 1):
                if current_qp in tested_qps:
                    break
                    
                tested_qps.add(current_qp)
                sample_dur = quick_sample_duration if quick_sample_duration > 0 and it <= quick_iters else sample_duration
                
                # Update progress bar with current QP being tested
                adapt_bar.set_description(f"QP Search {file.name[:35]} - Testing QP{current_qp}")
                adapt_bar.refresh()  # Force immediate update
                
                result = evaluate_qp_on_clips(clips, current_qp)
                
                if result is None:
                    adapt_bar.write(f"[WARN] No VMAF scores at QP {current_qp} for {file.name}")
                    break
                    
                qp_results[current_qp] = result
                
                # Update with results
                vmaf = result['mean_vmaf']
                size_pct = result['size_pct']
                adapt_bar.set_description(f"QP Search {file.name[:30]} - QP{current_qp}: VMAF {vmaf:.1f}, Size {size_pct:.0f}%")
                adapt_bar.refresh()
                
                # Update best QPs
                if result['passes_quality']:
                    last_good_qp = current_qp
                    if not result['inefficient']:
                        best_efficient_qp = current_qp
                
                # Determine next QP (similar to existing adaptive logic)
                catastrophic = result.get('early_fail', False)
                should_increase = (result['passes_quality'] and not result['inefficient']) or result['inefficient']
                should_decrease = (not result['passes_quality']) or catastrophic
                
                if should_increase and should_decrease:
                    break  # Conflicting signals, stop
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
                # If catastrophic failure, jump more aggressively downward
                if catastrophic:
                    step = max(step, 4.0)
                
                # Dynamic step scaling if enabled
                if scale_steps_by_vmaf and result['mean_vmaf'] > 0:
                    vmaf_surplus = result['mean_vmaf'] - vmaf_target
                    dynamic_step = abs(vmaf_surplus) * step_scale
                    step = min(max_dynamic_step, max(min_step, dynamic_step))
                
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
        rel_loss = 100 - result['mean_vmaf']
        worst_drop = result.get('worst_drop', 0)
        print(f"[RESULT] {file.name}: QP {final_qp} mean {result['mean_vmaf']:.2f} (loss {rel_loss:.2f}), min {result['min_vmaf']:.2f}, worst drop {worst_drop:.2f}, size {result['size_pct']:.1f}% (samples {result.get('samples_used', '?')})")
        return final_qp, result
    else:
        print(f"[RESULT] {file.name}: QP {final_qp} (fallback)")
        return final_qp, None
    """Adaptive, iterative QP search without predefined discrete list.
        Strategy:
            - Start at initial_qp. Evaluate quality + size efficiency.
            - If quality passes, try raising QP (more compression) to find boundary while retaining targets.
            - If inefficient (>100% size estimate), also raise QP to gain compression.
            - If quality fails, lower QP and reduce step (like learning rate decay on direction change).
            - On direction changes, halve the current step size down to (but not below) min_step.
            - Stop when effective integer step (after rounding) == min_step and we have changed direction at least twice (boundary found), or bounds reached.
            - Prefer highest passing efficient QP (size<=100%). Fallback to highest passing QP, else last evaluated.
        Notes:
            - Hardware/software HEVC encoders accept only integer QP values; fractional step is approximated by rounding to nearest integer delta.
    """
    def evaluate(qp: int, sample_dur: int):
        vmaf_scores = []
        total_sample_original_size = 0
        total_encoded_size = 0
        total_full_original_size = 0
        extrapolated_encoded_total = 0.0
        with tqdm(total=len(files), desc=f"QP {qp} eval", position=1, leave=False) as file_pbar:
            for f in files:
                vmaf_score, encoded_size, sample_orig_size = test_qp_on_sample(
                    f, qp, encoder, encoder_type, sample_duration=sample_dur, vmaf_threads=vmaf_threads, preserve_hdr_metadata=preserve_hdr_metadata)
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                    total_encoded_size += encoded_size
                    total_sample_original_size += sample_orig_size
                    full_size = f.stat().st_size if f.exists() else 0
                    total_full_original_size += full_size
                    if sample_orig_size > 0 and full_size > 0:
                        extrapolated_encoded_total += (encoded_size / sample_orig_size) * full_size
                file_pbar.update(1)
        if not vmaf_scores:
            return None
        mean_vmaf = sum(vmaf_scores)/len(vmaf_scores)
        min_vmaf = min(vmaf_scores)
        compression_ratio = (total_encoded_size/total_sample_original_size)*100 if total_sample_original_size>0 else 0
        extrapolated_full_pct = (extrapolated_encoded_total/total_full_original_size)*100 if total_full_original_size>0 else 0
        size_pct = extrapolated_full_pct if extrapolated_full_pct>0 else compression_ratio
        return {
            'mean_vmaf': mean_vmaf,
            'min_vmaf': min_vmaf,
            'sample_pct': compression_ratio,
            'full_pct': extrapolated_full_pct,
            'size_pct': size_pct,
            'passes_quality': mean_vmaf >= vmaf_target and min_vmaf >= vmaf_min_threshold,
            'inefficient': size_pct > 100
        }

    qp_results: dict[int, dict] = {}
    tested_qps: set[int] = set()
    current_qp = max(min_qp_limit, min(max_qp_limit, initial_qp))
    direction: int | None = None  # +1 increase QP, -1 decrease QP
    last_good_qp: int | None = None
    best_efficient_qp: int | None = None
    direction_changes = 0
    step = float(initial_step)
    min_step = max(0.5, float(min_step))  # guard minimum (0.5 still maps to 1 QP delta)
    # Track previous iteration stats for early-stop heuristics
    last_move_direction: int | None = None  # direction taken to arrive at current_qp
    prev_size_pct: float | None = None

    if scale_steps_by_vmaf:
        print(f"[ADAPTIVE] Starting automatic quality tuning at QP {current_qp} with VMAF-scaled steps (base initial {initial_step:g}, min {min_step:g}, scale {step_scale:g} QP per VMAF, max dynamic {max_dynamic_step:g}) within range {min_qp_limit}-{max_qp_limit}.")
    else:
        print(f"[ADAPTIVE] Starting automatic quality tuning at QP {current_qp} (initial step {step:g}, min step {min_step:g}) within allowed range {min_qp_limit}-{max_qp_limit}.")
    print("[INFO] VMAF = Video quality score 0-100 (higher is better). Target average >= "
          f"{vmaf_target}, minimum per clip >= {vmaf_min_threshold}. 'QP' is the compression level (higher = smaller file, lower quality).")

    with tqdm(total=max_iterations, desc="Adaptive QP", position=0) as adapt_bar:
        for it in range(1, max_iterations+1):
            if current_qp in qp_results:
                res = qp_results[current_qp]
            else:
                # Decide sample duration for this iteration (quick-pass if configured)
                iter_sample_dur = sample_duration
                if quick_sample_duration and quick_iters and it <= quick_iters:
                    iter_sample_dur = min(sample_duration, quick_sample_duration)
                res = evaluate(current_qp, iter_sample_dur)
                if res is None:
                    print(f"[ADAPTIVE] No VMAF data at QP {current_qp}, stopping.")
                    break
                qp_results[current_qp] = res
            tested_qps.add(current_qp)

            passes = res['passes_quality']
            inefficient = res['inefficient']
            if passes:
                last_good_qp = current_qp
                if not inefficient:
                    if (best_efficient_qp is None) or current_qp > best_efficient_qp:
                        best_efficient_qp = current_qp
            dir_char = {None:'-', 1:'↑', -1:'↓'}[direction]
            # Narrative style iteration report
            size_pct = res['size_pct']
            savings = max(0.0, 100 - size_pct) if size_pct > 0 else 0.0
            quality_status = "PASSED" if passes else "FAILED"
            eff_status = "efficient" if not inefficient else "inefficient (larger than original)"
            quick_note = " (quick short sample)" if (quick_sample_duration and quick_iters and it <= quick_iters) else ""
            direction_phrase = {
                None: "(initial point)",
                1: "Trying a higher QP next to seek smaller size.",
                -1: "Quality dropped; will try a lower QP next."
            }[direction]
            baseline_vals = [BASELINE_VMAF.get(str(f)) for f in files if BASELINE_VMAF.get(str(f)) is not None]
            baseline_avg = (sum(baseline_vals)/len(baseline_vals)) if baseline_vals else 100.0
            print(
                f"Iteration {it}: QP {current_qp}{quick_note}. "
                f"Average quality {res['mean_vmaf']:.2f} vs original baseline {baseline_avg:.2f}, lowest clip {res['min_vmaf']:.2f}. "
                f"Estimated output size {size_pct:.1f}% of original (~{savings:.1f}% smaller). "
                f"Quality {quality_status}; file size is {eff_status}. {direction_phrase}"
            )
            # Interim summary of best so far
            if last_good_qp is not None:
                print(f"  Best passing QP so far: {last_good_qp}. Highest passing & efficient QP: {best_efficient_qp if best_efficient_qp is not None else 'not yet'}.")

            # Early-stop: if we arrived here by decreasing QP (seeking higher quality) but size percentage grew further above 100%,
            # additional decreases will only inflate size. Abort downward exploration.
            if last_move_direction == -1 and prev_size_pct is not None:
                if res['size_pct'] >= prev_size_pct and res['size_pct'] > 100:
                    print("[EARLY-STOP] Lowering QP increased size further (>100%). Stopping further downward search.")
                    break

            # Decide next move
            next_direction = 1 if passes else -1
            if passes and inefficient:
                next_direction = 1

            if direction is not None and next_direction != direction:
                direction_changes += 1
                if not scale_steps_by_vmaf and step > min_step:
                    step = max(min_step, step / 2.0)
            direction = next_direction

            # Dynamic step scaling based on quality surplus/deficit relative to targets
            if scale_steps_by_vmaf:
                if passes:
                    surplus_mean = res['mean_vmaf'] - vmaf_target
                    surplus_min = res['min_vmaf'] - vmaf_min_threshold
                    # Use the smaller surplus (closer to dropping below threshold) to be conservative
                    surplus = min(surplus_mean, surplus_min)
                    # If inefficient and barely surplus, still want to move upward at least min_step
                    magnitude = max(0.0, surplus)
                else:
                    deficit_mean = vmaf_target - res['mean_vmaf']
                    deficit_min = vmaf_min_threshold - res['min_vmaf']
                    # Use the larger deficit (worst miss) to correct aggressively
                    deficit = max(deficit_mean, deficit_min)
                    magnitude = max(0.0, deficit)
                # Convert VMAF magnitude to QP step (linear heuristic)
                dyn_step = magnitude * step_scale
                # If quality passes but we are inefficient (>100%) and surplus is tiny, still bump at least min_step
                if passes and inefficient and dyn_step < min_step:
                    dyn_step = min_step
                step = max(min_step, min(max_dynamic_step, dyn_step if dyn_step > 0 else min_step))

            # Determine integer QP delta from (possibly fractional) step
            int_delta = int(round(step))
            if int_delta < 1:
                int_delta = 1
            proposed_qp = current_qp + (int_delta * direction)
            # Bounds check
            if proposed_qp > max_qp_limit or proposed_qp < min_qp_limit:
                break

            # Stopping condition: effective integer step reached min_step (rounded) & boundary explored
            if int(round(step)) <= int(round(min_step)) and direction_changes >= 2 and last_good_qp is not None:
                # Before stopping, probe immediate neighbors around boundary to avoid skipping (e.g., QP 23)
                neighbor_order = []
                # Prefer testing higher QP (more compression) first
                if last_good_qp + 1 <= max_qp_limit:
                    neighbor_order.append(last_good_qp + 1)
                if last_good_qp - 1 >= min_qp_limit:
                    neighbor_order.append(last_good_qp - 1)
                for nqp in neighbor_order:
                    if nqp not in tested_qps:
                        nres = evaluate(nqp, sample_duration)  # always use full sample duration for final refinement
                        if nres:
                            qp_results[nqp] = nres
                            tested_qps.add(nqp)
                            passes_n = nres['passes_quality']
                            inefficient_n = nres['inefficient']
                            if passes_n:
                                last_good_qp = nqp
                                if not inefficient_n and (best_efficient_qp is None or nqp > best_efficient_qp):
                                    best_efficient_qp = nqp
                            size_pct_n = nres.get('size_pct', 0)
                            savings_n = max(0.0, 100 - size_pct_n) if size_pct_n > 0 else 0.0
                            print(
                                f"[REFINE] Neighbor QP {nqp}: mean {nres['mean_vmaf']:.2f}, min {nres['min_vmaf']:.2f}, size {size_pct_n:.1f}% (~{savings_n:.1f}% smaller) - "
                                f"{'PASS' if passes_n else 'FAIL'} {'efficient' if not inefficient_n else 'inefficient'}")
                break

            current_qp = proposed_qp
            last_move_direction = direction  # direction used to move to new QP
            prev_size_pct = res['size_pct']
            adapt_bar.update(1)

    # Decide final
    final_qp = best_efficient_qp or last_good_qp or current_qp
    print(f"[ADAPTIVE] Chosen QP {final_qp} {'(best efficient match)' if final_qp==best_efficient_qp and best_efficient_qp is not None else '(best quality match)'}")
    if final_qp in qp_results:
        fr = qp_results[final_qp]
        size_pct = fr['size_pct']
        savings = max(0.0, 100 - size_pct) if size_pct > 0 else 0.0
        baseline_vals = [BASELINE_VMAF.get(str(f)) for f in files if BASELINE_VMAF.get(str(f)) is not None]
        baseline_avg = (sum(baseline_vals)/len(baseline_vals)) if baseline_vals else 100.0
        print(
            f"[RESULT] QP {final_qp} keeps average quality {fr['mean_vmaf']:.2f} compared to original baseline {baseline_avg:.2f} (target >= {vmaf_target}), lowest clip {fr['min_vmaf']:.2f} (minimum >= {vmaf_min_threshold}), "
            f"with estimated output size {size_pct:.1f}% of the originals (~{savings:.1f}% smaller)."
        )
    return final_qp

def prompt_user_confirmation(message: str, auto_yes: bool = False) -> bool:
    """Prompt user for Y/N confirmation. Returns True if user confirms, False otherwise."""
    if auto_yes:
        print(f"{message} [AUTO-YES]")
        return True
    
    response = input(f"{message} (Y/N) ").strip().lower()
    return response in ("y", "yes")

def verify_and_prompt_transcode(files: list[Path], optimal_qp: int, encoder: str, encoder_type: str, 
                               args, preserve_hdr: bool = True) -> bool:
    """
    Verify quality & size thresholds and prompt user for transcode confirmation.
    Returns True if user wants to proceed, False otherwise.
    This is the same logic used in the main transcode script.
    """
    sample_files = files[: min(args.samples if hasattr(args, 'samples') else 1, len(files))]
    
    # Verify quality + size thresholds on chosen QP before overwriting originals
    if sample_files:
        print("[VERIFY] Re-checking quality & size at chosen QP before overwriting originals...")
        verify_scores = []
        worst_score = None
        total_mean = 0.0
        total_encoded_size = 0
        total_sample_original_size = 0
        for sf in sample_files:
            vmaf_score, enc_sz, orig_sz = test_qp_on_sample(sf, optimal_qp, encoder, encoder_type, 
                                                          sample_duration=args.sample_duration, 
                                                          vmaf_threads=args.vmaf_threads, 
                                                          preserve_hdr_metadata=preserve_hdr)
            if vmaf_score is not None:
                verify_scores.append(vmaf_score)
                total_mean += vmaf_score
                if worst_score is None or vmaf_score < worst_score:
                    worst_score = vmaf_score
            if enc_sz and orig_sz:
                total_encoded_size += enc_sz
                total_sample_original_size += orig_sz
        if verify_scores:
            mean_vmaf_final = total_mean / len(verify_scores)
            min_vmaf_final = worst_score
            size_pct = (total_encoded_size / total_sample_original_size) * 100 if total_sample_original_size > 0 else 0.0
            savings = max(0.0, 100 - size_pct) if size_pct > 0 else 0.0
            print(f"[VERIFY] Mean VMAF {mean_vmaf_final:.2f} (require > {args.vmaf_target}), worst sample {min_vmaf_final:.2f} (require > {args.vmaf_min}); estimated size {size_pct:.1f}% of original (~{savings:.1f}% smaller) (require <= {args.max_size_pct}%).")
            passes_quality = (mean_vmaf_final > args.vmaf_target) and (min_vmaf_final > args.vmaf_min)
            passes_size = size_pct <= args.max_size_pct if size_pct > 0 else True
            passes_final = passes_quality and passes_size
            if not passes_quality:
                print("[VERIFY] Quality requirement failed.")
            if not passes_size:
                print("[VERIFY] Size saving requirement failed (no meaningful reduction).")
            if not passes_final and not getattr(args, 'force_overwrite', False):
                borderline_quality = (not passes_quality and
                                      mean_vmaf_final >= (args.vmaf_target - getattr(args, 'quality_prompt_band', 1.0)) and
                                      min_vmaf_final >= (args.vmaf_min - getattr(args, 'quality_prompt_band', 1.0)))
                if borderline_quality:
                    resp = input(f"[PROMPT] Quality within {getattr(args, 'quality_prompt_band', 1.0)} VMAF of target (mean {mean_vmaf_final:.2f} vs {args.vmaf_target}, worst {min_vmaf_final:.2f} vs {args.vmaf_min}). Proceed anyway? (Y/N) ").strip().lower()
                    if resp in ("y","yes"):
                        passes_final = True
                        print("[PROMPT] User accepted borderline quality; continuing.")
                if not passes_final:
                    print("[ABORT] Thresholds not satisfied; originals will NOT be overwritten. Use --force-overwrite to bypass or adjust --vmaf-target/--vmaf-min/--max-size-pct/--quality-prompt-band.")
                    return False
            if passes_final:
                print("[VERIFY] All thresholds satisfied; proceeding with full transcode.")
            elif getattr(args, 'force_overwrite', False):
                print("[WARN] Proceeding despite failing threshold(s) due to --force-overwrite.")
        else:
            print("[WARN] Unable to verify quality (no VMAF scores in verification step). Aborting overwrite for safety.")
            return False
    else:
        print("[INFO] No sample files available for verification; skipping overwrite for safety (set --samples > 0 or use --force-overwrite).")
        if not getattr(args, 'force_overwrite', False):
            return False
        print("[WARN] Proceeding without verification due to --force-overwrite.")
    
    return prompt_user_confirmation("\nProceed to FULL overwrite transcode?", getattr(args, 'auto_yes', False))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=".", help="Folder to scan (supports UNC)")
    ap.add_argument("--mode", choices=["qp", "vbr"], default="qp", 
                    help="Optimization mode: 'qp' for QP search (default), 'vbr' for VBR bitrate search")
    ap.add_argument("--exts", default=",".join(e.strip(".") for e in EXTS),
                    help="Comma-separated extensions (e.g. mkv,mp4,mov,ts)")
    ap.add_argument("--samples", type=int, default=SAMPLE_COUNT_DEFAULT,
                    help="How many files to sample for QP optimization")
    ap.add_argument("--no-auto-scale-samples", action="store_true", 
                    help="Disable automatic sample count scaling based on video duration")
    ap.add_argument("--qp-range", default=None,
                    help="Comma-separated QP values to test (optional; if omitted, adaptive QP search is used)")
    ap.add_argument("--use-qp", type=int, default=None,
                    help="Skip QP optimization and use this specific QP value directly")
    ap.add_argument("--vmaf-target", type=float, default=VMAF_TARGET,
                    help="Target mean VMAF score")
    ap.add_argument("--vmaf-min", type=float, default=VMAF_MIN_THRESHOLD,
                    help="Minimum acceptable VMAF score for any clip")
    # VBR mode parameters
    ap.add_argument("--vmaf-tol", type=float, default=1,
                    help="VMAF tolerance for VBR mode (default 0.5)")
    ap.add_argument("--vbr-clips", type=int, default=2,
                    help="Number of clips for VBR VMAF testing (default 2)")
    ap.add_argument("--vbr-clip-duration", type=int, default=30,
                    help="Duration in seconds for each VBR test clip (default 25)")
    ap.add_argument("--vbr-max-trials", type=int, default=6,
                    help="Maximum trials for VBR parameter optimization (default 6)")
    # Default sample duration now 600s (10 minutes) for more stable VMAF statistics
    ap.add_argument("--sample-duration", type=int, default=600,
                    help="Seconds per sample clip for QP testing (default 600)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only perform QP optimization analysis; skip actual transcoding")
    ap.add_argument("--skip-transcode", action="store_true",
                    help="Alias for --dry-run (skip actual transcoding)")
    ap.add_argument("--force-overwrite", action="store_true",
                    help="Proceed even if quality or size thresholds are not met (override safety guard)")
    ap.add_argument("--max-size-pct", type=float, default=100.0,
                    help="Require estimated encoded size to be <= this percent of original before overwriting (default 100). Set >100 to disable size saving requirement.")
    ap.add_argument("--quality-prompt-band", type=float, default=1.0,
                    help="If mean/worst VMAF are within this margin BELOW thresholds, interactively prompt instead of auto-abort (default 1.0 -> '95ish').")
    ap.add_argument("--min-qp", type=int, default=10, help="Lower bound QP for adaptive search")
    ap.add_argument("--max-qp", type=int, default=45, help="Upper bound QP for adaptive search")
    ap.add_argument("--initial-step", type=float, default=4.0, help="Initial QP step size for adaptive search (can be fractional; encoder still uses integer QP)")
    ap.add_argument("--min-step", type=float, default=1.0, help="Minimum QP step size before stopping refinement (>=1 effective integer delta)")
    ap.add_argument("--scale-steps-by-vmaf", action="store_true", help="Scale adaptive QP step size proportionally to VMAF surplus/deficit (dynamic steps)")
    ap.add_argument("--step-scale", type=float, default=0.5, help="QP step per 1 VMAF point surplus/deficit when --scale-steps-by-vmaf is enabled (default 0.5)")
    ap.add_argument("--max-dynamic-step", type=float, default=8.0, help="Upper bound on dynamically scaled QP step when --scale-steps-by-vmaf is enabled")
    # Default VMAF threads = use more threads for better performance (testing showed single-core limitation)
    detected_cores = os.cpu_count() or 0
    # Use more aggressive threading - up to 12 threads for VMAF
    optimal_vmaf_threads = min(12, detected_cores)  # Increased from 8 to 12 for better performance
    ap.add_argument("--vmaf-threads", type=int, default=optimal_vmaf_threads,
                    help=f"Threads for libvmaf (default = {optimal_vmaf_threads}; 0 lets libvmaf auto decide; 12+ may help with threading issues)")
    ap.add_argument("--experimental-vmaf", action="store_true", 
                    help="Use experimental multiprocess VMAF for better threading on long videos (may be less stable)")
    ap.add_argument("--quick-sample-duration", type=int, default=0,
                    help="If >0, use this shorter sample duration for first N adaptive iterations (see --quick-iters)")
    ap.add_argument("--quick-iters", type=int, default=0,
                    help="Number of initial adaptive iterations to use quick sample duration")
    ap.add_argument("--no-hdr-metadata", action="store_true", help="Disable preservation of HDR10 static metadata (master_display / max_cll)")
    ap.add_argument("--gradient-descent", action="store_true", help="Use 2D gradient descent optimization instead of traditional binary search for QP optimization")
    ap.add_argument("--debug", action="store_true", help="Verbose debugging: keep samples, show ffmpeg & VMAF commands and stderr")
    ap.add_argument("--cpu-monitor", action="store_true", help="Monitor CPU usage during VMAF operations")
    ap.add_argument("--cpu", action="store_true", help="Force CPU/software encoding (libx265) instead of hardware acceleration")
    ap.add_argument("--auto-yes", action="store_true", help="Automatically answer 'yes' to confirmation prompts")
    ap.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (use sequential encoding + VMAF verification)")
    ap.add_argument("--non-destructive", action="store_true", help="Save transcoded files to 'Transcoded' subdirectory instead of replacing originals")
    ap.add_argument("--staging", action="store_true", help="Enable async file staging for network drive optimization (maintains buffer of 3 files in temp_transcode/)")
    ap.add_argument("--limit", type=int, help="Limit processing to the first N files (useful for testing)")
    args = ap.parse_args()

    # Inform user about automatic defaults if not explicitly overridden
    if "--vmaf-threads" not in sys.argv and args.vmaf_threads == optimal_vmaf_threads:
        print(f"[INFO] libvmaf threads defaulting to optimal count: {optimal_vmaf_threads} (testing shows diminishing returns beyond 8)")
    if "--sample-duration" not in sys.argv and args.sample_duration == 600:
        print("[INFO] sample-duration defaulting to 600s (override with --sample-duration)")

    # Determine candidate QPs
    preserve_hdr = not args.no_hdr_metadata
    global DEBUG
    DEBUG = args.debug
    if DEBUG:
        print("[DEBUG] Debug mode active")
    if args.qp_range:
        candidate_qps = [int(qp.strip()) for qp in args.qp_range.split(",") if qp.strip()]
    else:
        # Auto adaptive default scaffold (will expand up/down)
        candidate_qps = [16, 20, 24, 28, 32]

    base = Path(args.path)
    if not base.exists():
        print(f"[ERR] path not found: {base}")
        sys.exit(1)

    # Detect encoder
    if args.cpu:
        cpu_count = os.cpu_count() or 8
        encoder, encoder_type = "libx265", "software"
        print(f"[INFO] Using encoder: {encoder} ({encoder_type}) - forced by --cpu flag")
        print(f"[CPU] MAXIMUM optimization for {cpu_count} cores:")
        print(f"      • {cpu_count} threads (100% utilization)")  
        print(f"      • {min(12, max(3, cpu_count // 2))} frame-threads (aggressive pipeline)")
        print(f"      • {min(6, max(2, cpu_count // 4))} pools (NUMA-aware distribution)")
        print(f"      • {min(8, max(2, cpu_count // 4))} lookahead-threads (analysis parallelism)")
        print(f"      • Advanced: pmode, pme, b-adapt=2, subme=7 (CPU intensive)")
    else:
        encoder, encoder_type = detect_hevc_encoder()
        print(f"[INFO] Using encoder: {encoder} ({encoder_type})")

    # Use FileManager for centralized file discovery and processing
    file_manager = FileManager(debug=DEBUG, temp_files=TEMP_FILES)
    
    # Perform startup cleanup
    removed_count = file_manager.startup_scavenge(base)
    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} stale temp file(s) from previous run.")
    
    # Discover and filter files - handle direct file path or directory
    try:
        if base.is_file():
            # Direct file path provided
            if base.suffix.lower() in [ext.lower() for ext in EXTS]:
                print(f"[INFO] Processing single file: {base.name}")
                files = [base]
                # For non-destructive mode, use the file's parent directory
                output_base = base.parent
            else:
                print(f"[ERR] Unsupported file type: {base.suffix}")
                sys.exit(1)
        else:
            # Directory path - use FileManager for discovery
            discovery_result = file_manager.process_files_with_codec_filtering(base, args.exts)
            files = discovery_result.files_to_transcode
            output_base = base
            
            if not files:
                if discovery_result.skipped_files:
                    print("[INFO] no files need transcoding (all already use efficient codecs)")
                else:
                    print(f"[INFO] no matching files in {base}")
                return
        
        # Apply limit if specified
        if args.limit and len(files) > args.limit:
            original_count = len(files)
            files = files[:args.limit]
            print(f"[INFO] Limited to first {args.limit} files (out of {original_count} total)")
            
    except ValueError as e:
        print(f"[ERR] {e}")
        sys.exit(1)

    # --- VBR mode handling ---
    if args.mode == "vbr":
        print(f"[VBR] Starting VBR optimization mode (target VMAF: {args.vmaf_target}±{args.vmaf_tol})")
        
        # Initialize async file stager if enabled
        stager = None
        remaining_files = None
        if args.staging:
            # Use current working directory for temp staging, not the target path
            local_temp_dir = Path.cwd() / "temp_transcode"
            stager = AsyncFileStager(buffer_size=3, temp_dir=local_temp_dir)
            remaining_files = files.copy()  # Working queue of files to process
            
            # Stage first file and wait for completion before starting processing
            if remaining_files:
                first_file = remaining_files.pop(0)
                print(f"[STAGING] Prioritizing first file: {first_file.name}")
                first_staged = stager.stage_file(first_file)
                
                # Wait for first file to complete transfer
                if first_staged.transfer_future:
                    print(f"[STAGING] Waiting for first file to complete transfer...")
                    first_staged.transfer_future.result()  # Block until complete
                    first_staged.ready = True
                    stager.staged_files.put(first_staged)
                    print(f"[STAGING] ✓ First file ready: {first_file.name}")
                
                # Start staging remaining files for buffer (async - don't wait)
                initial_staging_count = min(2, len(remaining_files))  # 2 more for 3 total
                for _ in range(initial_staging_count):
                    if remaining_files:
                        file_to_stage = remaining_files.pop(0)
                        stager.stage_file(file_to_stage)
                
                print(f"[STAGING] Started background staging of {initial_staging_count} additional files")
                print(f"[STAGING] Starting VBR processing immediately with first file ready")
            else:
                print(f"[STAGING] No files to stage")
        
        vbr_results = {}
        
        if args.staging and stager:
            # Staged processing mode
            with tqdm(total=len(files), desc="VBR Optimization (Staged)", position=0) as vbr_progress:
                processed_count = 0
                
                while processed_count < len(files):
                    # Get next ready file from staging buffer
                    staged_file = stager.get_ready_file(timeout=300)
                    if not staged_file:
                        print(f"[STAGING] Timeout waiting for file transfer")
                        break
                    
                    # Use staged file for processing
                    file = staged_file.original_path
                    staged_path = staged_file.staged_path
                    
                    vbr_progress.set_description(f"VBR Optimization - {file.name[:40]}...")
                    
                    # Maintain buffer by staging next file
                    if remaining_files:
                        stager.maintain_buffer(remaining_files)
                    
                    # Get file duration and clip positions for VBR testing (use staged file)
                    duration = get_duration_sec(staged_path)
                    if duration is None:
                        print(f"[VBR-SKIP] {file.name} - Cannot determine duration")
                        stager.cleanup_staged_file(staged_file)
                        processed_count += 1
                        vbr_progress.update(1)
                        continue
                    
                    # Auto-scale VBR clips based on file duration (same logic as QP mode)
                    vbr_clips = args.vbr_clips
                    vbr_clips_was_explicitly_set = '--vbr-clips' in sys.argv
                    
                    if duration > 0 and not vbr_clips_was_explicitly_set:
                        auto_clips = max(3, min(vbr_clips * 2, int(duration / 1200)))  # 1200s = 20min
                        if auto_clips != vbr_clips:
                            print(f"[VBR-INFO] Auto-scaling clips from {vbr_clips} to {auto_clips} based on {duration/60:.1f}min duration")
                            vbr_clips = auto_clips
                        
                    clip_positions = get_vbr_clip_positions(int(duration), vbr_clips)
                    
                    # Run VBR optimization on staged file (but store result with original file)
                    result = optimize_encoder_settings_vbr(
                        staged_path, encoder, encoder_type,
                        target_vmaf=args.vmaf_target,
                        vmaf_tolerance=args.vmaf_tol,
                        clip_positions=clip_positions,
                        clip_duration=args.vbr_clip_duration,
                        max_trials=args.vbr_max_trials
                    )
                    
                    if result['success']:
                        vbr_results[file] = result  # Store with original file as key
                        print(f"[VBR-SUCCESS] {file.name}: {result['bitrate']}kbps, "
                              f"VMAF {result['vmaf_score']:.2f}, "
                              f"{result['preset']}, bf={result['bf']}, refs={result['refs']}")
                    else:
                        if result.get('abandoned'):
                            vmaf_gap = result.get('vmaf_gap', 0)
                            print(f"[VBR-ABANDON] {file.name}: Target unreachable (VMAF gap: {vmaf_gap:.1f} points)")
                        else:
                            print(f"[VBR-FAILED] {file.name}: Could not find suitable VBR settings")
                    
                    # Clean up staged file after processing
                    stager.cleanup_staged_file(staged_file)
                    processed_count += 1
                    vbr_progress.update(1)
            
            # Cleanup stager
            stager.cleanup()
        else:
            # Standard processing mode (no staging)
            with tqdm(total=len(files), desc="VBR Optimization", position=0) as vbr_progress:
                for file in files:
                    vbr_progress.set_description(f"VBR Optimization - {file.name[:40]}...")
                    
                    # Get file duration and clip positions for VBR testing
                    duration = get_duration_sec(file)
                    if duration is None:
                        print(f"[VBR-SKIP] {file.name} - Cannot determine duration")
                        vbr_progress.update(1)
                        continue
                    
                    # Auto-scale VBR clips based on file duration (same logic as QP mode)
                    vbr_clips = args.vbr_clips
                    vbr_clips_was_explicitly_set = '--vbr-clips' in sys.argv
                    
                    if duration > 0 and not vbr_clips_was_explicitly_set:
                        auto_clips = max(3, min(vbr_clips * 2, int(duration / 1200)))  # 1200s = 20min
                        if auto_clips != vbr_clips:
                            print(f"[VBR-INFO] Auto-scaling clips from {vbr_clips} to {auto_clips} based on {duration/60:.1f}min duration")
                            vbr_clips = auto_clips
                        
                    clip_positions = get_vbr_clip_positions(int(duration), vbr_clips)
                    
                    # Run VBR optimization
                    result = optimize_encoder_settings_vbr(
                        file, encoder, encoder_type,
                        target_vmaf=args.vmaf_target,
                        vmaf_tolerance=args.vmaf_tol,
                        clip_positions=clip_positions,
                        clip_duration=args.vbr_clip_duration,
                        max_trials=args.vbr_max_trials
                    )
                    
                    if result['success']:
                        vbr_results[file] = result
                        print(f"[VBR-SUCCESS] {file.name}: {result['bitrate']}kbps, "
                              f"VMAF {result['vmaf_score']:.2f}, "
                              f"{result['preset']}, bf={result['bf']}, refs={result['refs']}")
                    else:
                        if result.get('abandoned'):
                            vmaf_gap = result.get('vmaf_gap', 0)
                            print(f"[VBR-ABANDON] {file.name}: Target unreachable (VMAF gap: {vmaf_gap:.1f} points)")
                        else:
                            print(f"[VBR-FAILED] {file.name}: Could not find suitable VBR settings")
                        
                    vbr_progress.update(1)
        
        # Print VBR results summary
        if vbr_results:
            print(f"\n[VBR] Optimization complete for {len(vbr_results)}/{len(files)} files:")
            total_bitrate = sum(r['bitrate'] for r in vbr_results.values())
            avg_bitrate = total_bitrate / len(vbr_results)
            avg_vmaf = sum(r['vmaf_score'] for r in vbr_results.values()) / len(vbr_results)
            
            print(f"  Average bitrate: {avg_bitrate:.0f} kbps")
            print(f"  Average VMAF: {avg_vmaf:.2f}")
            
            if not (args.dry_run or args.skip_transcode):
                print(f"[VBR] Starting VBR transcoding with optimized settings...")
                
                # Create Transcoded directory if non-destructive mode
                transcoded_dir = None
                if args.non_destructive:
                    transcoded_dir = get_next_transcoded_dir(output_base)
                    transcoded_dir.mkdir(exist_ok=True)
                    print(f"[INFO] Non-destructive mode: saving to {transcoded_dir}")
                
                success_count = 0
                with tqdm(total=len(vbr_results), desc="VBR Transcoding", position=0) as overall:
                    for file, result in vbr_results.items():
                        print(f"[VBR-ENCODE] Transcoding {file.name} at {result['bitrate']}kbps (VMAF {result['vmaf_score']:.2f})")
                        
                        if args.non_destructive:
                            # Non-destructive: save to Transcoded subdirectory
                            final_output = transcoded_dir / file.name
                            tmp = transcoded_dir / (file.stem + ".transcode" + file.suffix)
                        else:
                            # Destructive: replace original
                            final_output = file
                            tmp = file.with_name(file.stem + ".transcode" + file.suffix)
                            bak = file.with_name(file.stem + ".bak" + file.suffix)

                        if tmp.exists(): 
                            try: tmp.unlink()
                            except: pass
                        
                        # Use VBR-specific encoding with optimized settings
                        ok = encode_vbr_with_progress(
                            file, tmp, encoder, encoder_type,
                            result['bitrate'], result['preset'], result['bf'], result['refs'],
                            preserve_hdr_metadata=preserve_hdr
                        )
                        
                        if ok and tmp.exists():
                            # CRITICAL: Final VMAF validation on full transcoded file
                            print(f"  [VERIFY] Running final VMAF check on full transcoded file...")
                            final_vmaf = compute_vmaf_score(file, tmp, n_threads=args.vmaf_threads)
                            
                            if final_vmaf is not None:
                                vmaf_diff = abs(final_vmaf - args.vmaf_target)
                                within_tolerance = vmaf_diff <= args.vmaf_tol
                                
                                if within_tolerance:
                                    print(f"  ✓ Final VMAF validation passed: {final_vmaf:.2f} (target: {args.vmaf_target}±{args.vmaf_tol})")
                                    
                                    # Quality validated - proceed with file placement
                                    if args.non_destructive:
                                        # Non-destructive: simply move temp to final location
                                        if final_output.exists():
                                            final_output.unlink()  # Remove existing transcoded version
                                        shutil.move(str(tmp), str(final_output))
                                        print(f"  ✓ Saved verified transcode to Transcoded/{file.name}")
                                        success_count += 1
                                    else:
                                        # Destructive: atomic swap with backup
                                        if bak.exists():
                                            try: bak.unlink()
                                            except: pass
                                        shutil.move(str(file), str(bak))
                                        shutil.move(str(tmp), str(file))
                                        try: bak.unlink()
                                        except: pass
                                        print(f"  ✓ Verified and replaced {file.name}")
                                        success_count += 1
                                else:
                                    print(f"  ✗ Final VMAF validation FAILED: {final_vmaf:.2f} (expected: {args.vmaf_target}±{args.vmaf_tol})")
                                    print(f"    Difference: {vmaf_diff:.2f} > tolerance {args.vmaf_tol}")
                                    print(f"    Discarding transcoded file and keeping original intact")
                                    try:
                                        tmp.unlink()  # Remove failed transcode
                                    except:
                                        pass
                            else:
                                print(f"  ✗ Final VMAF calculation failed - discarding transcode")
                                try:
                                    tmp.unlink()  # Remove unverifiable transcode
                                except:
                                    pass
                        else:
                            print(f"  ✗ Failed to transcode {file.name}")
                            
                        overall.update(1)
                
                print(f"[VBR] Transcoding complete: {success_count}/{len(vbr_results)} files successful.")
            else:
                print(f"[VBR] Dry run mode - skipping actual transcoding")
        else:
            print(f"[VBR] No files successfully optimized")
            
        return  # Exit after VBR mode
    
    # --- QP optimization step ---
    if args.use_qp:
        # Use predetermined QP for all files
        optimal_qp = args.use_qp
        print(f"[INFO] Using predetermined QP {optimal_qp} (skipping optimization)")
        file_qp_map = {f: optimal_qp for f in files}
    else:
        # Per-file QP optimization using random clips
        print(f"[INFO] Starting per-file QP optimization using {args.samples} random 120-second clips per file")
        file_qp_map = {}
        optimal_qps_history = []  # Track optimal QPs for adaptive starting points
        
        with tqdm(total=len(files), desc="File QP Optimization", position=0) as file_progress:
            for file in files:
                # Update progress bar with current file
                file_progress.set_description(f"File QP Optimization - Analyzing {file.name[:45]}...")
                
                # Calculate adaptive initial QP based on previous files
                if optimal_qps_history:
                    initial_qp = round(sum(optimal_qps_history) / len(optimal_qps_history))
                    initial_qp = max(10, min(45, initial_qp))  # Clamp to valid range
                    if DEBUG:
                        print(f"[ADAPTIVE-START] Using QP {initial_qp} as starting point (average of {len(optimal_qps_history)} previous files: {optimal_qps_history})")
                else:
                    initial_qp = 24  # Default for first file
                    if DEBUG:
                        print(f"[ADAPTIVE-START] Using default QP {initial_qp} for first file")
                
                if args.qp_range:
                    # Use grid search approach (not yet adapted for per-file)
                    provided = [int(q.strip()) for q in args.qp_range.split(',') if q.strip()]
                    provided = sorted(set(provided))
                    median_idx = len(provided)//2
                    file_qp = provided[median_idx] if provided else 24
                    print(f"[INFO] Using median QP {file_qp} for {file.name} (qp_range not yet supported per-file)")
                else:
                    # Use adaptive search per file
                    file_qp, qp_result = adaptive_qp_search_per_file(
                        file,
                        encoder,
                        encoder_type,
                        samples=args.samples,
                        initial_qp=initial_qp,
                        initial_step=args.initial_step,
                        min_step=args.min_step,
                        vmaf_target=args.vmaf_target,
                        vmaf_min_threshold=args.vmaf_min,
                        sample_duration=args.sample_duration,
                        min_qp_limit=args.min_qp,
                        max_qp_limit=args.max_qp,
                        vmaf_threads=args.vmaf_threads,
                        quick_sample_duration=args.quick_sample_duration,
                        quick_iters=args.quick_iters,
                        scale_steps_by_vmaf=args.scale_steps_by_vmaf,
                        step_scale=args.step_scale,
                        max_dynamic_step=args.max_dynamic_step,
                        preserve_hdr_metadata=preserve_hdr,
                        disable_auto_scale=args.no_auto_scale_samples,
                        use_gradient_descent=args.gradient_descent
                    )
                
                file_qp_map[file] = file_qp
                
                # Only add to history if QP wasn't constrained by bounds (for better adaptive learning)
                # Exception: if QP is at bounds but still meets VMAF target, it's a valid result to learn from
                is_at_min_bound = (file_qp <= args.min_qp)
                is_at_max_bound = (file_qp >= args.max_qp)
                is_bound_constrained = is_at_min_bound or is_at_max_bound
                
                # Check if VMAF target was met despite being at bounds
                if qp_result:
                    file_vmaf = qp_result['mean_vmaf']
                    meets_vmaf_target = file_vmaf >= args.vmaf_target
                else:
                    meets_vmaf_target = False
                
                if not is_bound_constrained or meets_vmaf_target:
                    optimal_qps_history.append(file_qp)  # Track for next file's adaptive starting point
                    if DEBUG:
                        status = "meets target despite bounds" if (is_bound_constrained and meets_vmaf_target) else "not bound-constrained"
                        print(f"[ADAPTIVE-LEARN] Added QP {file_qp} to history ({status})")
                elif DEBUG:
                    vmaf_info = f", VMAF {file_vmaf:.1f}" if qp_result else ""
                    print(f"[ADAPTIVE-LEARN] Skipped QP {file_qp} (bound-constrained: min={args.min_qp}, max={args.max_qp}{vmaf_info})")
                
                # Update progress bar with result
                file_progress.set_description(f"File QP Optimization - {file.name[:35]}: QP{file_qp}")
                file_progress.update(1)
        
        # For compatibility, use the median QP as "optimal_qp" for verification
        all_qps = list(file_qp_map.values())
        optimal_qp = sorted(all_qps)[len(all_qps)//2] if all_qps else 24
        print(f"[INFO] Per-file optimization complete. QP range: {min(all_qps)}-{max(all_qps)}, median: {optimal_qp}")

    if args.dry_run or args.skip_transcode:
        print("[DRY-RUN] Skipping full transcode.")
        return

    # Verify quality + size thresholds and get user confirmation
    if not verify_and_prompt_transcode(files, optimal_qp, encoder, encoder_type, args, preserve_hdr):
        return

    # --- full transcode ---
    print(f"[INFO] starting full transcode with per-file optimized QPs...")
    
    if not args.no_parallel:
        print(f"[INFO] Using parallel processing (disable with --no-parallel)")
        transcoder = ParallelTranscoder(vmaf_threads=args.vmaf_threads, use_experimental_vmaf=args.experimental_vmaf)
        success_count = transcoder.process_files_with_qp_map(files, file_qp_map, encoder, encoder_type, args.vmaf_target, args.vmaf_min, preserve_hdr)
        print(f"[DONE] Parallel processing completed: {success_count}/{len(files)} files successful.")
    else:
        # Sequential processing with per-file QPs
        print("[INFO] Using sequential processing with per-file QPs")
        success_count = 0
        
        # Create Transcoded directory if non-destructive mode
        transcoded_dir = None
        if args.non_destructive:
            transcoded_dir = get_next_transcoded_dir(output_base)
            transcoded_dir.mkdir(exist_ok=True)
            print(f"[INFO] Non-destructive mode: saving to {transcoded_dir}")
        
        with tqdm(total=len(files), desc="Overall Transcode", position=0) as overall:
            for f in files:
                file_qp = file_qp_map.get(f, optimal_qp)
                print(f"[INFO] Transcoding {f.name} with QP {file_qp}")
                
                if args.non_destructive:
                    # Non-destructive: save to Transcoded subdirectory
                    final_output = transcoded_dir / f.name
                    tmp = transcoded_dir / (f.stem + ".transcode" + f.suffix)
                else:
                    # Destructive: replace original
                    final_output = f
                    tmp = f.with_name(f.stem + ".transcode" + f.suffix)
                    bak = f.with_name(f.stem + ".bak" + f.suffix)

                if tmp.exists(): 
                    try: tmp.unlink()
                    except: pass
                    
                ok = encode_with_progress(f, tmp, encoder, encoder_type, file_qp, preserve_hdr_metadata=preserve_hdr)
                if ok and tmp.exists():
                    # CRITICAL: Final VMAF validation on full transcoded file
                    print(f"  [VERIFY] Running final VMAF check on full transcoded file...")
                    final_vmaf = compute_vmaf_score(f, tmp, n_threads=args.vmaf_threads)
                    
                    if final_vmaf is not None:
                        within_target = final_vmaf >= args.vmaf_target
                        within_min = final_vmaf >= args.vmaf_min
                        
                        if within_target and within_min:
                            print(f"  ✓ Final VMAF validation passed: {final_vmaf:.2f} (target: ≥{args.vmaf_target}, min: ≥{args.vmaf_min})")
                            
                            # Quality validated - proceed with file placement
                            if args.non_destructive:
                                # Non-destructive: simply move temp to final location
                                if final_output.exists():
                                    final_output.unlink()  # Remove existing transcoded version
                                shutil.move(str(tmp), str(final_output))
                                print(f"  ✓ Saved verified transcode to Transcoded/{f.name}")
                                success_count += 1
                            else:
                                # Destructive: atomic swap with backup
                                if bak.exists():
                                    try: bak.unlink()
                                    except: pass
                                shutil.move(str(f), str(bak))
                                shutil.move(str(tmp), str(f))
                                try: bak.unlink()
                                except: pass
                                print(f"  ✓ Verified and replaced {f.name}")
                                success_count += 1
                        else:
                            print(f"  ✗ Final VMAF validation FAILED: {final_vmaf:.2f}")
                            print(f"    Target: ≥{args.vmaf_target} {'✗ FAIL' if not within_target else '✓'}")
                            print(f"    Minimum: ≥{args.vmaf_min} {'✗ FAIL' if not within_min else '✓'}")
                            print(f"    Discarding transcoded file and keeping original intact")
                            try:
                                tmp.unlink()  # Remove failed transcode
                            except:
                                pass
                    else:
                        print(f"  ✗ Final VMAF calculation failed - discarding transcode")
                        try:
                            tmp.unlink()  # Remove unverifiable transcode
                        except:
                            pass
                else:
                    if tmp.exists():
                        try: tmp.unlink()
                        except: pass
                    TEMP_FILES.discard(str(tmp))
                    print(f"  Failed to transcode {f.name}")
                overall.update(1)
        
        print(f"[DONE] Sequential processing completed: {success_count}/{len(files)} files successful.")

        print("[DONE] all files processed.")

if __name__ == "__main__":
    main()