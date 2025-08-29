"""
Lazy Transcode - A smart video transcoding utility with VMAF-based quality optimization.
"""

__version__ = "1.0.0"
__author__ = "Rallade"
__email__ = "rallade@hotmail.com"

# Import configuration utilities
from .config import get_config, get_test_root, load_env_file

# Core functionality will be imported on-demand to avoid import issues
def get_transcode_functions():
    """Get transcode functions (imported on-demand)."""
    try:
        from .core.transcode import (
            verify_and_prompt_transcode,
            prompt_user_confirmation,
            detect_hevc_encoder,
            get_video_codec,
            should_skip_codec,
            adaptive_qp_search_per_file,
            format_size,
        )
        return {
            'verify_and_prompt_transcode': verify_and_prompt_transcode,
            'prompt_user_confirmation': prompt_user_confirmation,
            'detect_hevc_encoder': detect_hevc_encoder,
            'get_video_codec': get_video_codec,
            'should_skip_codec': should_skip_codec,
            'adaptive_qp_search_per_file': adaptive_qp_search_per_file,
            'format_size': format_size,
        }
    except ImportError:
        return {}

def get_manager_functions():
    """Get manager functions (imported on-demand)."""
    try:
        from .core.manager import (
            analyze_folder_direct,
            discover_subdirs,
            build_forward_args,
        )
        return {
            'analyze_folder_direct': analyze_folder_direct,
            'discover_subdirs': discover_subdirs,
            'build_forward_args': build_forward_args,
        }
    except ImportError:
        return {}

__all__ = [
    "get_config",
    "get_test_root", 
    "load_env_file",
    "get_transcode_functions",
    "get_manager_functions",
]
