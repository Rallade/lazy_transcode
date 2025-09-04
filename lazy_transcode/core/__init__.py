"""Core transcode and manager modules.

Expose legacy expected symbols for backward compatibility with tests
that import lazy_transcode.core.transcode.*
"""

# Re-export refactored modules under expected names
from .modules.processing import transcoding_engine as transcode  # type: ignore
from .main import main
from .modules.optimization.vbr_optimizer import calculate_intelligent_vbr_bounds
from .modules.analysis.media_utils import get_video_codec

__all__ = ["transcode", "main", "calculate_intelligent_vbr_bounds", "get_video_codec"]
