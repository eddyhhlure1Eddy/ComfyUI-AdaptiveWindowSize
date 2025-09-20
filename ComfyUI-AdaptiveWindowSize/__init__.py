"""
ComfyUI-AdaptiveWindowSize

Adaptive window size nodes for WanVideo that automatically adjust frame window sizes
to minimize waste frames and improve video alignment.

Author: eddy
Version: 1.0.0
"""

import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .adaptive_nodes import NODE_CLASS_MAPPINGS as ADAPTIVE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ADAPTIVE_DISPLAY_MAPPINGS

    # Import enhanced face crop nodes
    try:
        from .enhanced_face_crop import NODE_CLASS_MAPPINGS as FACE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FACE_DISPLAY_MAPPINGS

        # Combine all mappings
        NODE_CLASS_MAPPINGS = {**ADAPTIVE_MAPPINGS, **FACE_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**ADAPTIVE_DISPLAY_MAPPINGS, **FACE_DISPLAY_MAPPINGS}

        print("🎯 ComfyUI-AdaptiveWindowSize: Successfully loaded all enhanced nodes")

    except ImportError as e:
        print(f"⚠️  Enhanced face crop nodes not available: {e}")
        # Fall back to adaptive nodes only
        NODE_CLASS_MAPPINGS = ADAPTIVE_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS = ADAPTIVE_DISPLAY_MAPPINGS

    # Export the mappings for ComfyUI
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

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
__author__ = "eddy"
__description__ = "Adaptive window size nodes for WanVideo with automatic frame alignment"

# Web extension metadata (if needed for future web UI integration)
WEB_DIRECTORY = "./web"

# Optional: Add any initialization code here
def initialize():
    """Initialize the extension"""
    pass

# Call initialization
initialize()