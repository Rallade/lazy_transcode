"""
Animation content detection and auto-tune functionality for lazy_transcode.

This module provides automatic detection of animated content and applies
the appropriate encoder tune settings for optimal compression.
"""

from pathlib import Path
from typing import Optional, Tuple
from ..analysis.content_analyzer import get_content_analyzer
from ..analysis.media_utils import get_video_dimensions


def detect_animation_content(input_file: Path, sample_duration: int = 30) -> bool:
    """
    Detect if content is likely animated based on content analysis.
    
    Args:
        input_file: Path to input video file
        sample_duration: Duration of sample to analyze in seconds
        
    Returns:
        True if content appears to be animated, False otherwise
    """
    try:
        # Use existing content analyzer
        analyzer = get_content_analyzer()
        complexity = analyzer.analyze_content_complexity(input_file, sample_duration)
        
        # Animation typically has:
        # - Low spatial information (flat colors, simple gradients)
        # - Variable temporal information (can be high during action scenes)
        # - Content analyzer already identifies "simple animation" pattern
        
        # Check if content analyzer detected simple animation pattern
        if (complexity.spatial_info < 20 and complexity.temporal_info > 40):
            return True
            
        # Additional heuristics for animation detection:
        # - Very low spatial info often indicates flat animated content
        if complexity.spatial_info < 15:
            return True
            
        # Check for anime/animation keywords in filename
        filename = input_file.name.lower()
        animation_keywords = [
            'anime', 'animation', 'cartoon', 'animated'
        ]
        
        # Episode/series indicators that suggest TV anime
        series_indicators = ['ep', 'episode', 's01', 's02', 's03', 's04', 's05']
        
        has_animation_keyword = any(keyword in filename for keyword in animation_keywords)
        has_series_indicator = any(indicator in filename for indicator in series_indicators)
        
        # If filename has animation keywords OR (series indicators AND low-moderate spatial info)
        if has_animation_keyword:
            return True
        elif has_series_indicator and complexity.spatial_info < 30:
            return True
        
        return False
        
    except Exception:
        # If analysis fails, fall back to filename detection only
        filename = input_file.name.lower()
        animation_keywords = ['anime', 'animation', 'cartoon', 'animated']
        return any(keyword in filename for keyword in animation_keywords)


def get_optimal_tune_for_content(input_file: Path, encoder: str, 
                                sample_duration: int = 30) -> Optional[str]:
    """
    Determine optimal tune setting based on content analysis.
    
    Args:
        input_file: Path to input video file
        encoder: Encoder name (e.g., 'libx264', 'libx265')
        sample_duration: Duration of sample to analyze in seconds
        
    Returns:
        Recommended tune setting or None if no specific tune recommended
    """
    # Only software encoders support tune
    if encoder not in ['libx264', 'libx265']:
        return None
        
    try:
        # Check if content is animated
        if detect_animation_content(input_file, sample_duration):
            return 'animation'
            
        # Could add more content-based tune detection here:
        # - Film grain detection -> 'grain'
        # - Still image detection -> 'stillimage'
        # - For now, default to None (let encoder use default)
        
        return None
        
    except Exception:
        # If detection fails, don't apply any tune
        return None


def get_animation_optimized_settings(input_file: Path, encoder: str) -> dict:
    """
    Get recommended encoder settings optimized for animation content.
    
    Args:
        input_file: Path to input video file  
        encoder: Encoder name
        
    Returns:
        Dictionary of recommended settings for animation encoding
    """
    settings = {}
    
    try:
        # Detect if content is animated
        is_animation = detect_animation_content(input_file)
        
        if is_animation:
            # Animation-specific optimizations
            settings['tune'] = 'animation'
            
            # For animation, we can often use:
            # - Faster presets (animation compresses well)
            # - Lower reference frames (less motion prediction needed)
            # - Different B-frame settings
            
            if encoder in ['libx264', 'libx265']:
                # Use faster preset for animation (still good quality)
                settings['suggested_preset'] = 'fast'
                # Reduce reference frames for animation
                settings['refs'] = 2
                # Optimize B-frames for animation
                settings['bf'] = 2
                
        return settings
        
    except Exception:
        return {}
