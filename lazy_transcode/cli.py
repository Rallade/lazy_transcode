"""CLI entry points for lazy-transcode package."""

import sys
import os
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main_transcode():
    """Entry point for lazy-transcode command."""
    from lazy_transcode.core.transcode import main
    main()

def main_manager():
    """Entry point for lazy-transcode-manager command."""
    from lazy_transcode.core.manager import main
    main()

if __name__ == "__main__":
    # If called directly, determine which command to run based on script name
    script_name = Path(sys.argv[0]).stem
    if "manager" in script_name:
        main_manager()
    else:
        main_transcode()
