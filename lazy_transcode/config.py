"""Configuration management for lazy-transcode."""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env in current directory, then in package directory
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent / ".env",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                env_path = candidate
                break
    
    env_vars = {}
    
    if env_path and env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
    
    return env_vars


def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables and .env file."""
    env_vars = load_env_file()
    
    config = {
        'test_root': env_vars.get('test_root', os.getenv('TEST_ROOT')),
        'vmaf_target': float(env_vars.get('vmaf_target', os.getenv('VMAF_TARGET', '95.0'))),
        'vmaf_threads': int(env_vars.get('vmaf_threads', os.getenv('VMAF_THREADS', str(min(8, os.cpu_count() or 8))))),
        'sample_duration': int(env_vars.get('sample_duration', os.getenv('SAMPLE_DURATION', '120'))),
        'debug': env_vars.get('debug', os.getenv('DEBUG', 'false')).lower() in ('true', '1', 'yes'),
    }
    
    return config


def get_test_root() -> Optional[Path]:
    """Get the test root directory from configuration."""
    config = get_config()
    test_root = config.get('test_root')
    
    if test_root:
        path = Path(test_root)
        if path.exists() and path.is_dir():
            return path
        else:
            print(f"[WARNING] test_root path does not exist: {test_root}")
    
    return None
