"""
Unit tests for animation tune functionality in encoder configuration.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder


class TestAnimationTune(unittest.TestCase):
    """Test animation tune functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = EncoderConfigBuilder()

    def test_animation_tune_libx264(self):
        """Test animation tune with libx264 encoder."""
        self.builder.reset()
        self.builder.set_base_config("input.mkv", "output.mkv")
        self.builder.set_encoder("libx264", preset="medium", crf=23, tune="animation")
        
        cmd = self.builder.build_command()
        cmd_str = ' '.join(cmd)
        
        self.assertIn('-tune animation', cmd_str)
        self.assertIn('-c:v libx264', cmd_str)

    def test_animation_tune_libx265(self):
        """Test animation tune with libx265 encoder."""
        self.builder.reset()
        self.builder.set_base_config("input.mkv", "output.mkv")
        self.builder.set_encoder("libx265", preset="medium", crf=23, tune="animation")
        
        cmd = self.builder.build_command()
        cmd_str = ' '.join(cmd)
        
        self.assertIn('-tune animation', cmd_str)
        self.assertIn('-c:v libx265', cmd_str)

    def test_animation_tune_skipped_hardware_encoders(self):
        """Test that tune is skipped for hardware encoders."""
        hardware_encoders = ['hevc_nvenc', 'h264_nvenc', 'hevc_qsv', 'hevc_amf']
        
        for encoder in hardware_encoders:
            with self.subTest(encoder=encoder):
                self.builder.reset()
                self.builder.set_base_config("input.mkv", "output.mkv")
                self.builder.set_encoder(encoder, preset="medium", crf=23, tune="animation")
                
                cmd = self.builder.build_command()
                cmd_str = ' '.join(cmd)
                
                self.assertNotIn('-tune', cmd_str, f"Tune should be skipped for {encoder}")

    def test_different_tune_options(self):
        """Test different tune options work correctly."""
        tune_options = ['film', 'animation', 'grain', 'stillimage', 'psnr', 'ssim']
        
        for tune in tune_options:
            with self.subTest(tune=tune):
                self.builder.reset()
                self.builder.set_base_config("input.mkv", "output.mkv")
                self.builder.set_encoder("libx264", preset="medium", crf=23, tune=tune)
                
                cmd = self.builder.build_command()
                cmd_str = ' '.join(cmd)
                
                self.assertIn(f'-tune {tune}', cmd_str)

    @patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field')
    @patch('lazy_transcode.core.modules.analysis.media_utils.get_video_dimensions')
    def test_vbr_encode_cmd_with_tune(self, mock_dimensions, mock_ffprobe):
        """Test VBR encoding command includes tune parameter."""
        mock_ffprobe.return_value = "yuv420p"
        mock_dimensions.return_value = (1920, 1080)
        
        with patch('pathlib.Path.exists', return_value=True):
            cmd = self.builder.build_vbr_encode_cmd(
                "input.mkv", "output.mkv", "libx265", "medium", 5000,
                bf=3, refs=3, width=1920, height=1080, tune="animation"
            )
        
        cmd_str = ' '.join(cmd)
        self.assertIn('-tune animation', cmd_str)

    @patch('lazy_transcode.core.modules.config.encoder_config.ffprobe_field')
    def test_standard_encode_cmd_with_tune(self, mock_ffprobe):
        """Test standard encoding command includes tune parameter.""" 
        mock_ffprobe.return_value = "yuv420p"
        
        with patch('pathlib.Path.exists', return_value=True):
            cmd = self.builder.build_standard_encode_cmd(
                "input.mkv", "output.mkv", "libx265", "medium", 23,
                width=1920, height=1080, tune="animation"
            )
        
        cmd_str = ' '.join(cmd)
        self.assertIn('-tune animation', cmd_str)

    def test_no_tune_when_none_specified(self):
        """Test that no tune parameter is added when None specified."""
        self.builder.reset()
        self.builder.set_base_config("input.mkv", "output.mkv")
        self.builder.set_encoder("libx264", preset="medium", crf=23, tune=None)
        
        cmd = self.builder.build_command()
        cmd_str = ' '.join(cmd)
        
        self.assertNotIn('-tune', cmd_str)


class TestAnimationDetection(unittest.TestCase):
    """Test animation content detection functionality."""

    @patch('lazy_transcode.core.modules.analysis.animation_detector.get_content_analyzer')
    def test_animation_detection_by_metrics(self, mock_analyzer):
        """Test animation detection based on content metrics."""
        from lazy_transcode.core.modules.analysis.animation_detector import detect_animation_content
        
        # Mock content analyzer for animation-like content
        mock_complexity = MagicMock()
        mock_complexity.spatial_info = 18  # Low spatial info
        mock_complexity.temporal_info = 45  # High temporal info
        mock_analyzer.return_value.analyze_content_complexity.return_value = mock_complexity
        
        anime_file = Path("test_anime.mkv")
        is_animation = detect_animation_content(anime_file)
        
        self.assertTrue(is_animation)

    @patch('lazy_transcode.core.modules.analysis.animation_detector.get_content_analyzer')
    def test_animation_detection_by_filename(self, mock_analyzer):
        """Test animation detection based on filename keywords."""
        from lazy_transcode.core.modules.analysis.animation_detector import detect_animation_content
        
        # Mock content analyzer for borderline content
        mock_complexity = MagicMock()
        mock_complexity.spatial_info = 25  # Moderate spatial info
        mock_complexity.temporal_info = 30  # Moderate temporal info
        mock_analyzer.return_value.analyze_content_complexity.return_value = mock_complexity
        
        # Filename suggests anime with series indicator
        anime_file = Path("Attack.on.Titan.S01E01.mkv")
        is_animation = detect_animation_content(anime_file)
        
        self.assertTrue(is_animation)

    def test_optimal_tune_for_software_encoders(self):
        """Test optimal tune detection for software encoders."""
        from lazy_transcode.core.modules.analysis.animation_detector import get_optimal_tune_for_content
        
        with patch('lazy_transcode.core.modules.analysis.animation_detector.detect_animation_content', return_value=True):
            tune = get_optimal_tune_for_content(Path("anime.mkv"), "libx265")
            self.assertEqual(tune, "animation")
            
            tune = get_optimal_tune_for_content(Path("anime.mkv"), "libx264")
            self.assertEqual(tune, "animation")

    def test_no_tune_for_hardware_encoders(self):
        """Test that hardware encoders don't get tune recommendations."""
        from lazy_transcode.core.modules.analysis.animation_detector import get_optimal_tune_for_content
        
        with patch('lazy_transcode.core.modules.analysis.animation_detector.detect_animation_content', return_value=True):
            tune = get_optimal_tune_for_content(Path("anime.mkv"), "hevc_nvenc")
            self.assertIsNone(tune)


if __name__ == '__main__':
    unittest.main()
