"""
Gradient-Based VBR Optimization Module

Implementation of advanced gradient-based optimization techniques for VBR encoding,
based on research from "Comparative Analysis of Gradient-Based Optimization 
Techniques" (Asadi et al., 2024).

This module provides:
- BFGS-inspired quasi-Newton optimization for bitrate search
- Numerical gradient descent with adaptive step sizes
- Conjugate gradient methods 
- Intelligent initial point selection
- Multi-dimensional parameter optimization

Research Citation:
Asadi, S., et al. (2024). "Comparative Analysis of Gradient-Based Optimization
Techniques Using Multidimensional Surface 3D Visualizations and Initial Point
Sensitivity." arXiv:2409.04470v3
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field

from .system_utils import DEBUG, run_command, format_size
from .media_utils import get_video_dimensions, get_duration_sec
from .vmaf_evaluator import VMAfEvaluator
from ...utils.logging import get_logger

logger = get_logger("gradient_optimizer")


@dataclass
class OptimizationResult:
    """Result of gradient-based optimization."""
    success: bool
    bitrate: Optional[int] = None
    vmaf_score: Optional[float] = None
    preset: Optional[str] = None
    bf: Optional[int] = None
    refs: Optional[int] = None
    filesize: Optional[int] = None
    iterations: int = 0
    convergence_time: float = 0.0
    method_used: str = "unknown"
    optimization_path: Optional[List[Dict[str, Any]]] = field(default_factory=list)


@dataclass
class OptimizationPoint:
    """Single optimization point for tracking convergence."""
    bitrate: int
    vmaf_score: float
    preset: str = "medium"
    bf: int = 3
    refs: int = 3
    error: float = float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bitrate': self.bitrate,
            'vmaf_score': self.vmaf_score,
            'preset': self.preset,
            'bf': self.bf,
            'refs': self.refs,
            'error': self.error
        }


class GradientVBROptimizer:
    """
    Advanced gradient-based VBR optimizer using multiple optimization strategies.
    
    Implements simplified versions of optimization methods from Asadi et al. (2024):
    - Steepest Descent: Simple first-order method
    - Quasi-Newton: BFGS-inspired approximation
    - Conjugate Gradient: Direction optimization
    - Adaptive Step Sizes: Intelligent learning rate adjustment
    """
    
    def __init__(self, debug: bool = False):
        """Initialize gradient-based optimizer."""
        self.debug = debug
        self.vmaf_evaluator = VMAfEvaluator(debug=debug)
        self.optimization_history: List[OptimizationPoint] = []
        
        # Optimization parameters
        self.max_iterations = 50
        self.tolerance = 1e-6
        self.step_size_tolerance = 1e-8
        self.vmaf_tolerance = 0.5
        
        # Gradient approximation parameters
        self.gradient_epsilon = 100  # kbps step for numerical gradient
        self.initial_step_size = 500  # Initial step size in kbps
        self.step_decay = 0.8  # Step size decay factor
        self.min_step_size = 50  # Minimum step size
        
        # Content-adaptive parameters
        self.content_complexity = 1.0
        self.initial_point_strategy = "intelligent"
        
    def optimize_vbr_gradient_descent(self, infile: Path, outfile: Path, target_vmaf: float,
                                    encoder: str, encoder_type: str, preserve_hdr: bool = False,
                                    **kwargs) -> OptimizationResult:
        """
        Steepest Descent optimization for VBR bitrate.
        
        Simple gradient descent with numerical differentiation and adaptive step sizes.
        """
        start_time = time.time()
        logger.info(f"[Gradient Descent] Optimizing {infile.name} for VMAF {target_vmaf:.1f}")
        
        # Initialize
        self.optimization_history.clear()
        current_bitrate = self._calculate_initial_point(infile, target_vmaf)
        step_size = self.initial_step_size
        
        best_point = None
        best_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Evaluate current point
            vmaf_result = self._evaluate_vmaf_at_bitrate(
                infile, outfile, current_bitrate, encoder, encoder_type, preserve_hdr
            )
            
            if not vmaf_result.success or vmaf_result.vmaf_score is None:
                logger.error(f"[Gradient Descent] Failed to evaluate VMAF at {current_bitrate}kbps")
                break
            
            # Track current point
            current_error = abs(vmaf_result.vmaf_score - target_vmaf)
            point = OptimizationPoint(
                bitrate=current_bitrate,
                vmaf_score=vmaf_result.vmaf_score,
                error=current_error
            )
            self.optimization_history.append(point)
            
            if self.debug:
                logger.debug(f"[Gradient Descent] Iter {iteration+1}: {current_bitrate}kbps, "
                           f"VMAF {vmaf_result.vmaf_score:.2f}, Error {current_error:.2f}")
            
            # Check if this is our best point so far
            if current_error < best_error:
                best_error = current_error
                best_point = point
            
            # Check convergence
            if current_error <= self.vmaf_tolerance:
                logger.info(f"[Gradient Descent] ✓ Converged at iteration {iteration+1}")
                break
            
            # Calculate numerical gradient
            gradient = self._calculate_numerical_gradient(
                infile, outfile, current_bitrate, target_vmaf,
                encoder, encoder_type, preserve_hdr
            )
            
            if abs(gradient) < self.tolerance:
                logger.info(f"[Gradient Descent] ✓ Gradient tolerance reached at iteration {iteration+1}")
                break
            
            # Update bitrate using gradient descent
            direction = -gradient  # Move in direction of decreasing error
            new_bitrate = current_bitrate + step_size * direction
            
            # Bounds checking
            new_bitrate = max(500, min(25000, int(new_bitrate)))
            
            # Adaptive step size
            if abs(new_bitrate - current_bitrate) < self.min_step_size:
                step_size = max(self.min_step_size, step_size * self.step_decay)
            
            current_bitrate = new_bitrate
            
            # Decay step size for convergence
            step_size *= self.step_decay
            
            if step_size < self.min_step_size:
                logger.info(f"[Gradient Descent] ✓ Minimum step size reached at iteration {iteration+1}")
                break
        
        optimization_time = time.time() - start_time
        
        if best_point and best_error <= self.vmaf_tolerance:
            logger.info(f"[Gradient Descent] ✓ Optimal: {best_point.bitrate}kbps, "
                       f"VMAF {best_point.vmaf_score:.2f} ({optimization_time:.1f}s)")
            
            return OptimizationResult(
                success=True,
                bitrate=best_point.bitrate,
                vmaf_score=best_point.vmaf_score,
                preset="medium",
                bf=3, refs=3,
                iterations=len(self.optimization_history),
                convergence_time=optimization_time,
                method_used="Gradient Descent",
                optimization_path=[p.to_dict() for p in self.optimization_history]
            )
        
        # Even if we didn't converge within tolerance, return the best result if it's reasonable
        if best_point and best_error <= 5.0:  # Accept up to 5 VMAF points error
            logger.info(f"[Gradient Descent] ✓ Best effort: {best_point.bitrate}kbps, "
                       f"VMAF {best_point.vmaf_score:.2f}, error {best_error:.1f} ({optimization_time:.1f}s)")
            
            return OptimizationResult(
                success=True,
                bitrate=best_point.bitrate,
                vmaf_score=best_point.vmaf_score,
                preset="medium",
                bf=3, refs=3,
                iterations=len(self.optimization_history),
                convergence_time=optimization_time,
                method_used="Gradient Descent",
                optimization_path=[p.to_dict() for p in self.optimization_history]
            )
        
        logger.info(f"[Gradient Descent] ✗ Failed to converge within acceptable tolerance")
        if best_point:
            logger.info(f"[Gradient Descent] Best found: {best_point.bitrate}kbps, VMAF {best_point.vmaf_score:.2f}, error {best_error:.1f}")
        return OptimizationResult(
            success=False,
            method_used="Gradient Descent",
            iterations=len(self.optimization_history),
            convergence_time=optimization_time,
            optimization_path=[p.to_dict() for p in self.optimization_history]
        )
    
    def optimize_vbr_quasi_newton(self, infile: Path, outfile: Path, target_vmaf: float,
                                encoder: str, encoder_type: str, preserve_hdr: bool = False,
                                **kwargs) -> OptimizationResult:
        """
        BFGS-inspired quasi-Newton optimization for VBR bitrate.
        
        Simplified implementation that approximates BFGS using gradient history
        and adaptive Hessian approximation.
        """
        start_time = time.time()
        logger.info(f"[Quasi-Newton] Optimizing {infile.name} for VMAF {target_vmaf:.1f}")
        
        # Initialize
        self.optimization_history.clear()
        current_bitrate = self._calculate_initial_point(infile, target_vmaf)
        
        # Quasi-Newton state
        previous_gradient = None
        hessian_approx = 1.0  # Scalar approximation of inverse Hessian
        step_size = self.initial_step_size
        
        best_point = None
        best_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Evaluate current point
            vmaf_result = self._evaluate_vmaf_at_bitrate(
                infile, outfile, current_bitrate, encoder, encoder_type, preserve_hdr
            )
            
            if not vmaf_result.success or vmaf_result.vmaf_score is None:
                logger.error(f"[Quasi-Newton] Failed to evaluate VMAF at {current_bitrate}kbps")
                break
            
            # Track current point
            current_error = abs(vmaf_result.vmaf_score - target_vmaf)
            point = OptimizationPoint(
                bitrate=current_bitrate,
                vmaf_score=vmaf_result.vmaf_score,
                error=current_error
            )
            self.optimization_history.append(point)
            
            if self.debug:
                logger.debug(f"[Quasi-Newton] Iter {iteration+1}: {current_bitrate}kbps, "
                           f"VMAF {vmaf_result.vmaf_score:.2f}, Error {current_error:.2f}")
            
            # Check if this is our best point so far
            if current_error < best_error:
                best_error = current_error
                best_point = point
            
            # Check convergence
            if current_error <= self.vmaf_tolerance:
                logger.info(f"[Quasi-Newton] ✓ Converged at iteration {iteration+1}")
                break
            
            # Calculate gradient
            current_gradient = self._calculate_numerical_gradient(
                infile, outfile, current_bitrate, target_vmaf,
                encoder, encoder_type, preserve_hdr
            )
            
            if abs(current_gradient) < self.tolerance:
                logger.info(f"[Quasi-Newton] ✓ Gradient tolerance reached at iteration {iteration+1}")
                break
            
            # Update Hessian approximation using BFGS-style update
            if previous_gradient is not None and iteration > 0:
                gradient_diff = current_gradient - previous_gradient
                if abs(gradient_diff) > 1e-8:
                    # Simple scalar BFGS approximation
                    hessian_approx = max(0.1, min(10.0, abs(gradient_diff) / step_size))
            
            # Quasi-Newton step: x_new = x - H^(-1) * grad
            direction = -current_gradient / hessian_approx
            new_bitrate = current_bitrate + direction
            
            # Bounds checking
            new_bitrate = max(500, min(25000, int(new_bitrate)))
            
            # Update for next iteration
            previous_gradient = current_gradient
            step_size = abs(new_bitrate - current_bitrate)
            current_bitrate = new_bitrate
        
        optimization_time = time.time() - start_time
        
        if best_point and best_error <= self.vmaf_tolerance:
            logger.info(f"[Quasi-Newton] ✓ Optimal: {best_point.bitrate}kbps, "
                       f"VMAF {best_point.vmaf_score:.2f} ({optimization_time:.1f}s)")
            
            return OptimizationResult(
                success=True,
                bitrate=best_point.bitrate,
                vmaf_score=best_point.vmaf_score,
                preset="medium",
                bf=3, refs=3,
                iterations=len(self.optimization_history),
                convergence_time=optimization_time,
                method_used="Quasi-Newton",
                optimization_path=[p.to_dict() for p in self.optimization_history]
            )
        
        # Accept reasonable results within 5 VMAF points
        if best_point and best_error <= 5.0:
            logger.info(f"[Quasi-Newton] ✓ Best effort: {best_point.bitrate}kbps, "
                       f"VMAF {best_point.vmaf_score:.2f}, error {best_error:.1f} ({optimization_time:.1f}s)")
            
            return OptimizationResult(
                success=True,
                bitrate=best_point.bitrate,
                vmaf_score=best_point.vmaf_score,
                preset="medium",
                bf=3, refs=3,
                iterations=len(self.optimization_history),
                convergence_time=optimization_time,
                method_used="Quasi-Newton",
                optimization_path=[p.to_dict() for p in self.optimization_history]
            )
        
        logger.info(f"[Quasi-Newton] ✗ Failed to converge within acceptable tolerance")
        if best_point:
            logger.info(f"[Quasi-Newton] Best found: {best_point.bitrate}kbps, VMAF {best_point.vmaf_score:.2f}, error {best_error:.1f}")
        return OptimizationResult(
            success=False,
            method_used="Quasi-Newton",
            iterations=len(self.optimization_history),
            convergence_time=optimization_time,
            optimization_path=[p.to_dict() for p in self.optimization_history]
        )
    
    def optimize_vbr_conjugate_gradient(self, infile: Path, outfile: Path, target_vmaf: float,
                                       encoder: str, encoder_type: str, preserve_hdr: bool = False,
                                       **kwargs) -> OptimizationResult:
        """
        Conjugate Gradient optimization for VBR bitrate.
        
        From research: "takes a more direct path and reaches the local minimum 
        faster than the Steepest Descent method"
        """
        start_time = time.time()
        logger.info(f"[Conjugate Gradient] Optimizing {infile.name} for VMAF {target_vmaf:.1f}")
        
        # Initialize
        self.optimization_history.clear()
        current_bitrate = self._calculate_initial_point(infile, target_vmaf)
        
        # Conjugate gradient state
        previous_gradient = None
        search_direction = None
        beta = 0.0
        
        best_point = None
        best_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Evaluate current point
            vmaf_result = self._evaluate_vmaf_at_bitrate(
                infile, outfile, current_bitrate, encoder, encoder_type, preserve_hdr
            )
            
            if not vmaf_result.success or vmaf_result.vmaf_score is None:
                logger.error(f"[Conjugate Gradient] Failed to evaluate VMAF at {current_bitrate}kbps")
                break
            
            # Track current point
            current_error = abs(vmaf_result.vmaf_score - target_vmaf)
            point = OptimizationPoint(
                bitrate=current_bitrate,
                vmaf_score=vmaf_result.vmaf_score,
                error=current_error
            )
            self.optimization_history.append(point)
            
            if self.debug:
                logger.debug(f"[Conjugate Gradient] Iter {iteration+1}: {current_bitrate}kbps, "
                           f"VMAF {vmaf_result.vmaf_score:.2f}, Error {current_error:.2f}")
            
            # Check if this is our best point so far
            if current_error < best_error:
                best_error = current_error
                best_point = point
            
            # Check convergence
            if current_error <= self.vmaf_tolerance:
                logger.info(f"[Conjugate Gradient] ✓ Converged at iteration {iteration+1}")
                break
            
            # Calculate gradient
            current_gradient = self._calculate_numerical_gradient(
                infile, outfile, current_bitrate, target_vmaf,
                encoder, encoder_type, preserve_hdr
            )
            
            if abs(current_gradient) < self.tolerance:
                logger.info(f"[Conjugate Gradient] ✓ Gradient tolerance reached at iteration {iteration+1}")
                break
            
            # Update search direction using Polak-Ribière formula
            if previous_gradient is not None and search_direction is not None:
                # Polak-Ribière beta calculation
                gradient_diff = current_gradient - previous_gradient
                beta = max(0, (current_gradient * gradient_diff) / (previous_gradient * previous_gradient))
                # Reset direction if beta is too large (restart condition)
                if beta > 1.0:
                    beta = 0.0
            else:
                beta = 0.0
            
            # Update search direction: d = -gradient + beta * previous_direction
            if search_direction is None:
                search_direction = -current_gradient
            else:
                search_direction = -current_gradient + beta * search_direction
            
            # Line search step size (simple adaptive approach)
            step_size = min(self.initial_step_size, abs(1000 / (current_gradient + 1e-8)))
            
            # Update bitrate
            new_bitrate = current_bitrate + step_size * search_direction
            
            # Bounds checking
            new_bitrate = max(500, min(25000, int(new_bitrate)))
            
            # Update for next iteration
            previous_gradient = current_gradient
            current_bitrate = new_bitrate
        
        optimization_time = time.time() - start_time
        
        if best_point and best_error <= self.vmaf_tolerance:
            logger.info(f"[Conjugate Gradient] ✓ Optimal: {best_point.bitrate}kbps, "
                       f"VMAF {best_point.vmaf_score:.2f} ({optimization_time:.1f}s)")
            
            return OptimizationResult(
                success=True,
                bitrate=best_point.bitrate,
                vmaf_score=best_point.vmaf_score,
                preset="medium",
                bf=3, refs=3,
                iterations=len(self.optimization_history),
                convergence_time=optimization_time,
                method_used="Conjugate Gradient",
                optimization_path=[p.to_dict() for p in self.optimization_history]
            )
        
        # Accept reasonable results within 5 VMAF points  
        if best_point and best_error <= 5.0:
            logger.info(f"[Conjugate Gradient] ✓ Best effort: {best_point.bitrate}kbps, "
                       f"VMAF {best_point.vmaf_score:.2f}, error {best_error:.1f} ({optimization_time:.1f}s)")
            
            return OptimizationResult(
                success=True,
                bitrate=best_point.bitrate,
                vmaf_score=best_point.vmaf_score,
                preset="medium",
                bf=3, refs=3,
                iterations=len(self.optimization_history),
                convergence_time=optimization_time,
                method_used="Conjugate Gradient",
                optimization_path=[p.to_dict() for p in self.optimization_history]
            )
        
        logger.info(f"[Conjugate Gradient] ✗ Failed to converge within acceptable tolerance")
        if best_point:
            logger.info(f"[Conjugate Gradient] Best found: {best_point.bitrate}kbps, VMAF {best_point.vmaf_score:.2f}, error {best_error:.1f}")
        return OptimizationResult(
            success=False,
            method_used="Conjugate Gradient",
            iterations=len(self.optimization_history),
            convergence_time=optimization_time,
            optimization_path=[p.to_dict() for p in self.optimization_history]
        )
    
    def _calculate_numerical_gradient(self, infile: Path, outfile: Path, bitrate: int,
                                    target_vmaf: float, encoder: str, encoder_type: str,
                                    preserve_hdr: bool = False) -> float:
        """
        Calculate numerical gradient of VMAF error with respect to bitrate.
        
        Uses central difference approximation: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        """
        try:
            h = self.gradient_epsilon  # Step size for numerical differentiation
            
            # Evaluate at bitrate + h
            result_plus = self._evaluate_vmaf_at_bitrate(
                infile, outfile, bitrate + h, encoder, encoder_type, preserve_hdr
            )
            
            # Evaluate at bitrate - h  
            result_minus = self._evaluate_vmaf_at_bitrate(
                infile, outfile, bitrate - h, encoder, encoder_type, preserve_hdr
            )
            
            if (not result_plus.success or not result_minus.success or
                result_plus.vmaf_score is None or result_minus.vmaf_score is None):
                return 0.0  # No gradient information available
            
            # Calculate errors
            error_plus = abs(result_plus.vmaf_score - target_vmaf)
            error_minus = abs(result_minus.vmaf_score - target_vmaf)
            
            # Central difference gradient
            gradient = (error_plus - error_minus) / (2 * h)
            
            return gradient
            
        except Exception as e:
            if self.debug:
                logger.debug(f"Gradient calculation failed: {e}")
            return 0.0
    
    def _calculate_initial_point(self, infile: Path, target_vmaf: float) -> int:
        """
        Calculate intelligent initial point based on content analysis.
        
        From research: "critical role that initial point selection plays in 
        optimizing optimization outcomes"
        """
        try:
            # Get video properties
            width, height = get_video_dimensions(infile)
            duration = get_duration_sec(infile)
            
            if not width or not height:
                return 4000  # Conservative fallback
            
            # Calculate complexity metrics
            pixel_count = width * height
            complexity_factor = self._estimate_content_complexity(infile)
            
            # VMAF-based initial estimation
            # Higher VMAF targets need higher bitrates, relationship is roughly logarithmic
            vmaf_factor = math.log(target_vmaf / 30.0) if target_vmaf > 30 else 0.1
            
            # Resolution-based base bitrate (empirical relationship)
            if pixel_count <= 720 * 480:  # SD
                base_bitrate = 1500
            elif pixel_count <= 1280 * 720:  # 720p
                base_bitrate = 2500
            elif pixel_count <= 1920 * 1080:  # 1080p
                base_bitrate = 4000
            else:  # 4K+
                base_bitrate = 8000
            
            # Apply content complexity and VMAF scaling
            estimated_bitrate = int(base_bitrate * complexity_factor * (1.0 + vmaf_factor))
            
            # Bounds check
            estimated_bitrate = max(500, min(25000, estimated_bitrate))
            
            if self.debug:
                logger.debug(f"Initial point calculation: {width}x{height}, complexity={complexity_factor:.2f}, "
                           f"vmaf_factor={vmaf_factor:.2f}, estimated={estimated_bitrate}kbps")
            
            return estimated_bitrate
            
        except Exception as e:
            logger.debug(f"Failed to calculate initial point: {e}")
            return 4000  # Safe fallback
    
    def _estimate_content_complexity(self, infile: Path) -> float:
        """
        Estimate content complexity for adaptive optimization.
        
        Simple heuristic based on file size vs duration.
        More complex analysis could use SI/TI metrics from the research.
        """
        try:
            file_size_mb = infile.stat().st_size / (1024 * 1024)
            duration = get_duration_sec(infile)
            
            if duration <= 0:
                return 1.0
            
            # Mbps as complexity indicator
            mbps = file_size_mb * 8 / duration / 60
            
            # Normalize: 1.0 = average complexity, higher = more complex
            if mbps < 5:
                return 0.7  # Low complexity
            elif mbps < 15:
                return 1.0  # Normal complexity
            else:
                return 1.3  # High complexity
                
        except:
            return 1.0  # Default
    
    def _evaluate_vmaf_at_bitrate(self, infile: Path, outfile: Path, bitrate: int,
                                encoder: str, encoder_type: str, 
                                preserve_hdr: bool = False) -> Any:
        """Evaluate VMAF score at specific bitrate using fast clip-based approach."""
        try:
            from .media_utils import compute_vmaf_score
            from .system_utils import TEMP_FILES
            from types import SimpleNamespace
            import tempfile
            
            # Get video duration for clip extraction
            duration = get_duration_sec(infile)
            if duration <= 0:
                return SimpleNamespace(success=False, vmaf_score=None)
            
            # Use a short 10-second clip from 25% into the video for speed
            clip_start = max(10, int(duration * 0.25))  # Skip first 25% or 10s minimum
            clip_duration = min(10, int(duration - clip_start - 5))  # 10s clip for speed
            
            if clip_duration < 5:  # Too short for meaningful VMAF
                return SimpleNamespace(success=False, vmaf_score=None)
            
            # Create temporary files using simple approach
            temp_dir = Path(tempfile.gettempdir())
            clip_path = temp_dir / f"gradient_clip_{bitrate}k_{time.time()}.mkv"
            encoded_path = temp_dir / f"gradient_enc_{bitrate}k_{time.time()}.mkv"
            
            TEMP_FILES.add(str(clip_path))
            TEMP_FILES.add(str(encoded_path))
            
            try:
                # Extract clip using direct ffmpeg (faster than existing function)
                extract_cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                    "-ss", str(clip_start), "-i", str(infile),
                    "-t", str(clip_duration), "-c", "copy", str(clip_path)
                ]
                
                extract_result = run_command(extract_cmd, timeout=60)
                if extract_result.returncode != 0 or not clip_path.exists():
                    if self.debug:
                        logger.debug(f"Failed to extract clip: {extract_result.stderr}")
                    return SimpleNamespace(success=False, vmaf_score=None)
                
                # Encode the clip at specified bitrate
                success = self._encode_vbr_test(clip_path, encoded_path, bitrate, encoder, encoder_type, preserve_hdr)
                
                if success and encoded_path.exists():
                    # Compute VMAF score on the clip
                    vmaf_score = compute_vmaf_score(clip_path, encoded_path)
                    filesize = encoded_path.stat().st_size
                    
                    if self.debug:
                        logger.debug(f"Clip test at {bitrate}kbps: VMAF {vmaf_score:.2f} ({clip_duration}s clip)")
                    
                    return SimpleNamespace(
                        success=True,
                        vmaf_score=vmaf_score,
                        bitrate=bitrate,
                        filesize=filesize
                    )
                else:
                    if self.debug:
                        logger.debug(f"Failed to encode clip at {bitrate}kbps")
                    return SimpleNamespace(success=False, vmaf_score=None)
                    
            finally:
                # Cleanup temporary files
                for temp_file in [clip_path, encoded_path]:
                    if temp_file and temp_file.exists():
                        try:
                            temp_file.unlink()
                            TEMP_FILES.discard(str(temp_file))
                        except:
                            pass
                    
        except Exception as e:
            if self.debug:
                logger.debug(f"Failed to evaluate VMAF at {bitrate}kbps: {e}")
            from types import SimpleNamespace
            return SimpleNamespace(success=False, vmaf_score=None)
    
    def _encode_vbr_test(self, infile: Path, outfile: Path, bitrate_kbps: int,
                        encoder: str, encoder_type: str, preserve_hdr: bool = False) -> bool:
        """Encode a test clip at specified VBR bitrate."""
        try:
            # Import VBR encoding function from existing module
            from .vbr_optimizer import build_vbr_encode_cmd
            
            # Build the encoding command
            cmd = build_vbr_encode_cmd(
                infile, outfile, encoder, encoder_type,
                bitrate_kbps, int(bitrate_kbps * 0.8),  # Max and min bitrate
                preserve_hdr_metadata=preserve_hdr
            )
            
            if not DEBUG:
                cmd.extend(["-loglevel", "error"])
            
            if self.debug:
                logger.debug(f"VBR test encode: {bitrate_kbps}kbps")
            
            # Run encoding
            result = run_command(cmd, timeout=120)  # 2 minute timeout for clips
            success = result.returncode == 0 and outfile.exists()
            
            if not success and self.debug:
                logger.debug(f"VBR encode failed: {result.stderr}")
            
            return success
            
        except ImportError:
            # Fallback - build command manually
            return self._encode_vbr_manual(infile, outfile, bitrate_kbps, encoder, encoder_type, preserve_hdr)
        except Exception as e:
            if self.debug:
                logger.debug(f"VBR encode error: {e}")
            return False
    
    def _encode_vbr_manual(self, infile: Path, outfile: Path, bitrate_kbps: int,
                          encoder: str, encoder_type: str, preserve_hdr: bool = False) -> bool:
        """Manual VBR encoding implementation as fallback."""
        try:
            cmd = ["ffmpeg", "-i", str(infile)]
            
            # Video encoding parameters
            if encoder_type == "hardware":
                if encoder == "h264_nvenc":
                    cmd.extend(["-c:v", "h264_nvenc", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k"])
                elif encoder == "hevc_nvenc":
                    cmd.extend(["-c:v", "hevc_nvenc", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k"])
                elif encoder == "h264_amf":
                    cmd.extend(["-c:v", "h264_amf", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k"])
                elif encoder == "hevc_amf":
                    cmd.extend(["-c:v", "hevc_amf", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k"])
                else:
                    cmd.extend(["-c:v", encoder, "-b:v", f"{bitrate_kbps}k"])
            else:
                # Software encoder
                if encoder == "libx264":
                    cmd.extend(["-c:v", "libx264", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k", "-preset", "medium"])
                elif encoder == "libx265":
                    cmd.extend(["-c:v", "libx265", "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k", "-preset", "medium"])
                else:
                    cmd.extend(["-c:v", encoder, "-b:v", f"{bitrate_kbps}k"])
            
            # Audio copy
            cmd.extend(["-c:a", "copy"])
            
            # HDR preservation if requested
            if preserve_hdr:
                cmd.extend(["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"])
            
            # Output
            cmd.extend(["-y", str(outfile)])
            
            if not DEBUG:
                cmd.extend(["-loglevel", "error"])
            
            if self.debug:
                logger.debug(f"Manual VBR encode: {bitrate_kbps}kbps")
            
            result = run_command(cmd, timeout=120)
            return result.returncode == 0 and outfile.exists()
            
        except Exception as e:
            if self.debug:
                logger.debug(f"Manual VBR encode error: {e}")
            return False


def compare_optimization_methods(infile: Path, outfile: Path, target_vmaf: float,
                               encoder: str, encoder_type: str, preserve_hdr: bool = False,
                               methods: Optional[List[str]] = None) -> Dict[str, OptimizationResult]:
    """
    Compare multiple optimization methods and return results.
    
    Args:
        methods: List of methods to compare. Options:
                ["gradient-descent", "quasi-newton", "conjugate-gradient", "bisection"]
    """
    if methods is None:
        methods = ["gradient-descent", "quasi-newton", "conjugate-gradient"]
    
    results = {}
    optimizer = GradientVBROptimizer(debug=DEBUG)
    
    logger.info(f"Comparing optimization methods for {infile.name}")
    logger.info(f"Target VMAF: {target_vmaf:.1f}, Methods: {', '.join(methods)}")
    
    for method in methods:
        logger.info(f"\n--- Testing {method.upper()} Method ---")
        
        try:
            if method == "gradient-descent":
                result = optimizer.optimize_vbr_gradient_descent(
                    infile, outfile, target_vmaf, encoder, encoder_type, preserve_hdr
                )
            elif method == "quasi-newton":
                result = optimizer.optimize_vbr_quasi_newton(
                    infile, outfile, target_vmaf, encoder, encoder_type, preserve_hdr
                )
            elif method == "conjugate-gradient":
                result = optimizer.optimize_vbr_conjugate_gradient(
                    infile, outfile, target_vmaf, encoder, encoder_type, preserve_hdr
                )
            elif method == "bisection":
                # Use existing bisection method for comparison
                from .vbr_optimizer import optimize_encoder_settings_vbr
                vbr_result_dict = optimize_encoder_settings_vbr(
                    infile, encoder, encoder_type, target_vmaf, 1.0,
                    clip_positions=[0], clip_duration=10  # Sample parameters
                )
                
                # Convert to our result format
                result = OptimizationResult(
                    success=vbr_result_dict.get('success', False),
                    bitrate=vbr_result_dict.get('bitrate'),
                    vmaf_score=vbr_result_dict.get('vmaf_score'),
                    preset=vbr_result_dict.get('preset'),
                    bf=vbr_result_dict.get('bf'),
                    refs=vbr_result_dict.get('refs'),
                    filesize=vbr_result_dict.get('filesize'),
                    iterations=0,  # Not tracked by bisection
                    convergence_time=0.0,  # Not tracked by bisection
                    method_used="Bisection Search"
                )
            else:
                logger.debug(f"Unknown method: {method}")
                continue
            
            results[method] = result
            
            # Print comparison metrics
            if result.success:
                logger.info(f"{method.upper()} Result: {result.bitrate}kbps, "
                           f"VMAF {result.vmaf_score:.2f}, "
                           f"{result.iterations} iterations, "
                           f"{result.convergence_time:.1f}s")
            else:
                logger.debug(f"{method.upper()} failed to converge")
                
        except Exception as e:
            logger.error(f"Error testing {method}: {e}")
            results[method] = OptimizationResult(success=False, method_used=method)
    
    # Summary comparison
    successful_results = {k: v for k, v in results.items() if v.success}
    
    if successful_results:
        logger.info(f"\n--- Optimization Method Comparison ---")
        logger.info(f"Successful methods: {len(successful_results)}/{len(methods)}")
        
        # Find fastest convergence
        fastest = min(successful_results.items(), key=lambda x: x[1].convergence_time)
        logger.info(f"Fastest convergence: {fastest[0].upper()} ({fastest[1].convergence_time:.1f}s)")
        
        # Find most iterations
        most_thorough = max(successful_results.items(), key=lambda x: x[1].iterations)
        logger.info(f"Most thorough: {most_thorough[0].upper()} ({most_thorough[1].iterations} iterations)")
        
        # Find best accuracy (closest to target VMAF)
        if all(r.vmaf_score is not None for r in successful_results.values()):
            best_accuracy = min(successful_results.items(), 
                              key=lambda x: abs(x[1].vmaf_score - target_vmaf))
            logger.info(f"Best accuracy: {best_accuracy[0].upper()} "
                       f"(VMAF {best_accuracy[1].vmaf_score:.2f}, error {abs(best_accuracy[1].vmaf_score - target_vmaf):.2f})")
    
    return results
