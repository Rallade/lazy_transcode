#!/usr/bin/env python3
"""
Quick test for animation tune functionality.
"""

from pathlib import Path
from lazy_transcode.core.modules.config.encoder_config import EncoderConfigBuilder

def test_animation_tune():
    """Test animation tune parameter in EncoderConfigBuilder."""
    print("Testing animation tune functionality...")
    
    # Create builder
    builder = EncoderConfigBuilder()
    
    # Test with libx264 (should include tune)
    print("\n1. Testing libx264 with animation tune:")
    builder.reset()
    builder.set_base_config("input.mkv", "output.mkv")
    builder.set_encoder("libx264", preset="medium", crf=23, tune="animation")
    cmd = builder.build_command()
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    
    if '-tune animation' in cmd_str:
        print("✓ Animation tune correctly added for libx264")
    else:
        print("✗ Animation tune missing for libx264")
    
    # Test with libx265 (should include tune)
    print("\n2. Testing libx265 with animation tune:")
    builder.reset()
    builder.set_base_config("input.mkv", "output.mkv")
    builder.set_encoder("libx265", preset="medium", crf=23, tune="animation")
    cmd = builder.build_command()
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    
    if '-tune animation' in cmd_str:
        print("✓ Animation tune correctly added for libx265")
    else:
        print("✗ Animation tune missing for libx265")
    
    # Test with nvenc (should NOT include tune)
    print("\n3. Testing nvenc (should skip tune):")
    builder.reset()
    builder.set_base_config("input.mkv", "output.mkv")
    builder.set_encoder("h264_nvenc", preset="medium", crf=23, tune="animation")
    cmd = builder.build_command()
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    
    if '-tune' not in cmd_str:
        print("✓ Tune correctly skipped for nvenc encoder")
    else:
        print("✗ Tune incorrectly added for nvenc encoder")
    
    # Test different tune options for libx264
    print("\n4. Testing different tune options:")
    for tune_option in ['film', 'animation', 'grain', 'stillimage']:
        builder.reset()
        builder.set_base_config("input.mkv", "output.mkv")
        builder.set_encoder("libx264", preset="medium", crf=23, tune=tune_option)
        cmd = builder.build_command()
        cmd_str = ' '.join(cmd)
        
        if f'-tune {tune_option}' in cmd_str:
            print(f"✓ Tune '{tune_option}' correctly added")
        else:
            print(f"✗ Tune '{tune_option}' missing")

if __name__ == "__main__":
    test_animation_tune()
