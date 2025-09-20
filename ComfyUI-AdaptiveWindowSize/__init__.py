"""
ComfyUI-AdaptiveWindowSize

Adaptive window size nodes for WanVideo that automatically adjust frame window sizes
to minimize waste frames and improve video alignment.

Author: AI Assistant
Version: 1.0.0
"""

import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .adaptive_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    # Export the mappings for ComfyUI
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

    print("🎯 ComfyUI-AdaptiveWindowSize: Successfully loaded adaptive window size nodes")
    print("   Available nodes:")
    for key, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"   - {display_name} ({key})")

except ImportError as e:
    print(f"❌ ComfyUI-AdaptiveWindowSize: Failed to import nodes: {e}")
    print("   Make sure ComfyUI-WanVideoWrapper is installed and accessible")

    # Provide empty mappings to prevent ComfyUI errors
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Optional: Version information
__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Adaptive window size nodes for WanVideo with automatic frame alignment"

# Web extension metadata (if needed for future web UI integration)
WEB_DIRECTORY = "./web"

# Optional: Add any initialization code here
def initialize():
    """Initialize the extension"""
    pass

# Call initialization
initialize()