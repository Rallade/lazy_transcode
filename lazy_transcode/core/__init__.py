"""Core transcode and manager modules.

Expose legacy expected symbols for backward compatibility with tests
that import lazy_transcode.core.transcode.*
"""

# Re-export refactored modules under expected names
from .modules import transcoding_engine as transcode  # type: ignore
__all__ = ["transcode"]
