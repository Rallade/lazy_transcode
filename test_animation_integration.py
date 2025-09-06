#!/usr/bin/env python3
"""
Comprehensive test for animation tune integration with VBR optimization.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock the dependencies that might not be available
with patch('lazy_transcode.core.modules.analysis.media_utils.ffprobe_field', return_value="yuv420p"):
    with patch('lazy_transcode.core.modules.analysis.media_utils.get_video_dimensions', return_value=(1920, 1080)):
        with patch('lazy_transcode.core.modules.analysis.media_utils.get_duration_sec', return_value=1440.0):
            with patch('pathlib.Path.exists', return_value=True):
                # Import after mocking
                from lazy_transcode.core.modules.optimization.vbr_optimizer import build_vbr_encode_cmd
                from lazy_transcode.core.modules.analysis.animation_detector import (
                    detect_animation_content, get_optimal_tune_for_content, get_animation_optimized_settings
                )


def test_animation_detection():
    """Test animation content detection."""
    print("Testing animation content detection...")
    
    # Test with anime filename
    anime_file = Path("Attack.on.Titan.S01E01.1080p.mkv")
    
    # Mock content analyzer to return animation-like metrics
    mock_complexity = MagicMock()
    mock_complexity.spatial_info = 18  # Low spatial info (flat animation colors)
    mock_complexity.temporal_info = 45  # High temporal info (motion)
    
    with patch('lazy_transcode.core.modules.analysis.animation_detector.get_content_analyzer') as mock_analyzer:
        mock_analyzer.return_value.analyze_content_complexity.return_value = mock_complexity
        
        is_animation = detect_animation_content(anime_file)
        print(f"Anime file detection: {is_animation}")
        
        if is_animation:
            print("✓ Correctly detected anime content")
        else:
            print("✗ Failed to detect anime content")
    
    # Test with regular movie filename
    movie_file = Path("The.Matrix.1999.1080p.mkv")
    mock_complexity.spatial_info = 35  # Higher spatial info (live action detail)
    mock_complexity.temporal_info = 25  # Moderate temporal info
    
    with patch('lazy_transcode.core.modules.analysis.animation_detector.get_content_analyzer') as mock_analyzer:
        mock_analyzer.return_value.analyze_content_complexity.return_value = mock_complexity
        
        is_animation = detect_animation_content(movie_file)
        print(f"Movie file detection: {is_animation}")
        
        if not is_animation:
            print("✓ Correctly detected non-anime content")
        else:
            print("✗ Incorrectly detected movie as anime")


def test_optimal_tune_detection():
    """Test optimal tune detection for different content types."""
    print("\nTesting optimal tune detection...")
    
    # Test anime content with libx265
    anime_file = Path("Demon.Slayer.EP01.1080p.mkv")
    
    # Mock animation detection to return True
    with patch('lazy_transcode.core.modules.analysis.animation_detector.detect_animation_content', return_value=True):
        tune = get_optimal_tune_for_content(anime_file, 'libx265')
        print(f"Anime + libx265 tune: {tune}")
        
        if tune == 'animation':
            print("✓ Correctly recommended animation tune for anime content")
        else:
            print("✗ Failed to recommend animation tune for anime content")
    
    # Test non-anime content
    movie_file = Path("Inception.2010.1080p.mkv")
    
    with patch('lazy_transcode.core.modules.analysis.animation_detector.detect_animation_content', return_value=False):
        tune = get_optimal_tune_for_content(movie_file, 'libx265')
        print(f"Movie + libx265 tune: {tune}")
        
        if tune is None:
            print("✓ Correctly recommended no specific tune for movie content")
        else:
            print("✗ Incorrectly recommended tune for movie content")
    
    # Test hardware encoder (should not get tune)
    with patch('lazy_transcode.core.modules.analysis.animation_detector.detect_animation_content', return_value=True):
        tune = get_optimal_tune_for_content(anime_file, 'hevc_nvenc')
        print(f"Anime + nvenc tune: {tune}")
        
        if tune is None:
            print("✓ Correctly skipped tune for hardware encoder")
        else:
            print("✗ Incorrectly provided tune for hardware encoder")


def test_vbr_animation_integration():
    """Test VBR optimizer integration with animation tune."""
    print("\nTesting VBR optimizer animation integration...")
    
    anime_file = Path("Your.Name.2016.1080p.mkv")
    output_file = Path("output.mkv")
    
    # Mock all the required functions
    with patch('lazy_transcode.core.modules.optimization.vbr_optimizer.get_video_dimensions', return_value=(1920, 1080)):
        with patch('lazy_transcode.core.modules.optimization.vbr_optimizer.DEBUG', False):
            # Mock animation detection to return animation tune
            with patch('lazy_transcode.core.modules.analysis.animation_detector.get_optimal_tune_for_content', return_value='animation'):
                
                # Test with libx265 (software encoder)
                cmd = build_vbr_encode_cmd(
                    anime_file, output_file, 
                    encoder='libx265',
                    encoder_type='software',
                    max_bitrate=5000,
                    avg_bitrate=4000,
                    preset='medium'
                )
                
                cmd_str = ' '.join(cmd)
                print(f"VBR command: {cmd_str}")
                
                if '-tune animation' in cmd_str:
                    print("✓ Animation tune correctly integrated in VBR command")
                else:
                    print("✗ Animation tune missing from VBR command")
                
                # Test with hardware encoder (should not include tune)
                cmd_hw = build_vbr_encode_cmd(
                    anime_file, output_file,
                    encoder='hevc_nvenc', 
                    encoder_type='hardware',
                    max_bitrate=5000,
                    avg_bitrate=4000,
                    preset='medium'
                )
                
                cmd_hw_str = ' '.join(cmd_hw)
                print(f"Hardware VBR command: {cmd_hw_str}")
                
                if '-tune' not in cmd_hw_str:
                    print("✓ Tune correctly skipped for hardware encoder")
                else:
                    print("✗ Tune incorrectly added for hardware encoder")


def test_animation_optimized_settings():
    """Test animation-optimized encoding settings."""
    print("\nTesting animation-optimized settings...")
    
    anime_file = Path("Studio.Ghibli.Movie.1080p.mkv")
    
    # Mock animation detection
    with patch('lazy_transcode.core.modules.analysis.animation_detector.detect_animation_content', return_value=True):
        settings = get_animation_optimized_settings(anime_file, 'libx265')
        print(f"Animation settings: {settings}")
        
        expected_keys = ['tune', 'suggested_preset', 'refs', 'bf']
        
        for key in expected_keys:
            if key in settings:
                print(f"✓ Settings include {key}: {settings[key]}")
            else:
                print(f"✗ Settings missing {key}")
        
        if settings.get('tune') == 'animation':
            print("✓ Correct animation tune in optimized settings")
        else:
            print("✗ Wrong or missing animation tune in optimized settings")


if __name__ == "__main__":
    print("=== Animation Tune Integration Tests ===")
    test_animation_detection()
    test_optimal_tune_detection()
    test_vbr_animation_integration()
    test_animation_optimized_settings()
    print("\n=== Tests Complete ===")
