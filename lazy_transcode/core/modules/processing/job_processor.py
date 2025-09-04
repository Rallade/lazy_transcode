"""
Job processing module for lazy_transcode.

This module handles parallel transcoding operations including:
- TranscodeJob dataclass for job configuration
- AsyncFileStager for network drive optimization
- ParallelTranscoder for concurrent processing
"""

import asyncio
import shutil
import queue
import threading
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Any
from tqdm import tqdm

from ..system.system_utils import TEMP_FILES, DEBUG
from ..analysis.media_utils import compute_vmaf_score
from .transcoding_engine import transcode_file_qp


@dataclass
class TranscodeJob:
    """Job configuration for transcoding operations."""
    input_file: Path
    output_file: Path
    encoder: str
    encoder_type: str
    qp: int
    preserve_hdr_metadata: bool = True


@dataclass 
class StagedFile:
    """Represents a file that has been staged for processing."""
    original_path: Path
    staged_path: Path
    ready: bool = False
    transfer_future: Optional[Any] = None  # Can be asyncio.Future or concurrent.futures.Future


class AsyncFileStager:
    """
    Asynchronous file staging system for optimizing network drive operations.
    
    Maintains a buffer of files transferred to local storage for faster processing.
    """
    
    def __init__(self, buffer_size: int = 3, temp_dir: Optional[Path] = None):
        self.buffer_size = buffer_size
        self.temp_dir = temp_dir or Path.cwd() / "temp_transcode"
        self.staged_files: queue.Queue[StagedFile] = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.temp_dir.mkdir(exist_ok=True)
        
    def stage_file(self, file_path: Path) -> StagedFile:
        """Stage a file for processing."""
        staged_path = self.temp_dir / file_path.name
        staged_file = StagedFile(file_path, staged_path)
        
        # Start async transfer
        future = self.executor.submit(self._copy_file, file_path, staged_path)
        staged_file.transfer_future = future
        
        return staged_file
        
    def _copy_file(self, src: Path, dst: Path) -> bool:
        """Copy file from source to destination."""
        try:
            if dst.exists():
                dst.unlink()
            shutil.copy2(str(src), str(dst))
            TEMP_FILES.add(str(dst))
            return True
        except Exception as e:
            if DEBUG:
                print(f"[STAGING-ERROR] Failed to copy {src} to {dst}: {e}")
            return False
            
    def get_ready_file(self, timeout: int = 300) -> Optional[StagedFile]:
        """Get next ready file from staging buffer."""
        try:
            return self.staged_files.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def maintain_buffer(self, remaining_files: List[Path]):
        """Maintain staging buffer by adding next file if space available."""
        if len(remaining_files) > 0 and self.staged_files.qsize() < self.buffer_size:
            next_file = remaining_files.pop(0)
            staged = self.stage_file(next_file)
            # Add to queue when ready
            self.executor.submit(self._wait_and_queue, staged)
            
    def _wait_and_queue(self, staged_file: StagedFile):
        """Wait for file transfer to complete and add to ready queue."""
        if staged_file.transfer_future:
            success = staged_file.transfer_future.result()
            if success:
                staged_file.ready = True
                self.staged_files.put(staged_file)
                
    def cleanup_staged_file(self, staged_file: StagedFile):
        """Clean up a staged file after processing."""
        try:
            if staged_file.staged_path.exists():
                staged_file.staged_path.unlink()
                TEMP_FILES.discard(str(staged_file.staged_path))
        except Exception as e:
            if DEBUG:
                print(f"[STAGING-CLEANUP] Failed to remove {staged_file.staged_path}: {e}")
                
    def cleanup(self):
        """Clean up staging system."""
        self.executor.shutdown(wait=True)
        try:
            if self.temp_dir.exists():
                shutil.rmtree(str(self.temp_dir))
        except Exception as e:
            if DEBUG:
                print(f"[STAGING-CLEANUP] Failed to remove temp dir {self.temp_dir}: {e}")


class ParallelTranscoder:
    """
    Parallel transcoding processor with concurrent encoding and VMAF verification.
    """
    
    def __init__(self, vmaf_threads: int = 8, use_experimental_vmaf: bool = False):
        self.vmaf_threads = vmaf_threads
        self.use_experimental_vmaf = use_experimental_vmaf
        
    def process_files_with_qp_map(self, files: List[Path], file_qp_map: dict[Path, int], 
                                 encoder: str, encoder_type: str, vmaf_target: float, 
                                 vmaf_min: float, preserve_hdr_metadata: bool = True) -> int:
        """Process files with per-file QP mapping using parallel execution."""
        
        # Create transcode jobs
        jobs = []
        for file in files:
            qp = file_qp_map.get(file, 24)  # Default fallback
            tmp_output = file.with_name(file.stem + ".transcode" + file.suffix)
            
            job = TranscodeJob(
                input_file=file,
                output_file=tmp_output, 
                encoder=encoder,
                encoder_type=encoder_type,
                qp=qp,
                preserve_hdr_metadata=preserve_hdr_metadata
            )
            jobs.append(job)
            
        success_count = 0
        
        # Process jobs in parallel
        max_workers = min(4, len(jobs))  # Limit concurrent transcodes
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(jobs), desc="Parallel Transcode", position=0) as progress:
                
                # Submit all jobs
                future_to_job = {
                    executor.submit(self._process_job, job, vmaf_target, vmaf_min): job 
                    for job in jobs
                }
                
                # Process completed jobs
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                            print(f"  ✓ {job.input_file.name} completed successfully")
                        else:
                            print(f"  ✗ {job.input_file.name} failed processing")
                    except Exception as e:
                        print(f"  ✗ {job.input_file.name} failed with exception: {e}")
                        
                    progress.update(1)
                    
        return success_count
        
    def _process_job(self, job: TranscodeJob, vmaf_target: float, vmaf_min: float) -> bool:
        """Process a single transcode job."""
        try:
            # Clean up any existing temp file
            if job.output_file.exists():
                job.output_file.unlink()
                
            # Perform transcoding
            success = transcode_file_qp(
                job.input_file, job.output_file, job.encoder, job.encoder_type,
                job.qp, job.preserve_hdr_metadata
            )
            
            if not success or not job.output_file.exists():
                return False
                
            # VMAF verification
            vmaf_score = compute_vmaf_score(job.input_file, job.output_file, 
                                          n_threads=self.vmaf_threads)
            
            if vmaf_score is None:
                print(f"  ✗ VMAF verification failed for {job.input_file.name}")
                if job.output_file.exists():
                    job.output_file.unlink()
                return False
                
            # Check quality thresholds
            meets_target = vmaf_score >= vmaf_target
            meets_min = vmaf_score >= vmaf_min
            
            if meets_target and meets_min:
                # Quality validated - replace original atomically
                backup_file = job.input_file.with_name(job.input_file.stem + ".bak" + job.input_file.suffix)
                
                # Atomic swap
                if backup_file.exists():
                    backup_file.unlink()
                shutil.move(str(job.input_file), str(backup_file))
                shutil.move(str(job.output_file), str(job.input_file))
                
                # Remove backup
                try:
                    backup_file.unlink()
                except:
                    pass
                    
                print(f"  ✓ {job.input_file.name}: VMAF {vmaf_score:.2f} (QP {job.qp})")
                return True
            else:
                print(f"  ✗ {job.input_file.name}: VMAF {vmaf_score:.2f} below threshold")
                if job.output_file.exists():
                    job.output_file.unlink()
                return False
                
        except Exception as e:
            print(f"  ✗ {job.input_file.name}: Exception during processing: {e}")
            # Clean up temp file
            if job.output_file.exists():
                try:
                    job.output_file.unlink()
                except:
                    pass
            return False
