"""
Configuration file for Productivity Monitoring System
Edit these values to customize the detection behavior.
"""

# ============================================================================
# FACE PRESENCE DETECTION SETTINGS
# ============================================================================

# Time (in seconds) before marking user as "away" when face is not detected
FACE_PRESENT_TO_AWAY_TIME = 2.0

# Time (in seconds) before marking user as "present" when face is detected
FACE_AWAY_TO_PRESENT_TIME = 0.7


# ============================================================================
# HEAD POSE DETECTION SETTINGS
# ============================================================================

# Maximum deviation from baseline (in degrees) for yaw (left/right head turn)
HEAD_YAW_THRESHOLD = 20

# Maximum deviation from baseline (in degrees) for pitch (up/down head tilt)
HEAD_PITCH_THRESHOLD = 20

# Time (in seconds) before marking as "distracted" when head moves away
HEAD_ATTENTIVE_TO_DISTRACTED_TIME = 1.0

# Time (in seconds) before marking as "attentive" when head returns
HEAD_DISTRACTED_TO_ATTENTIVE_TIME = 0.5


# ============================================================================
# EYE GAZE DETECTION SETTINGS
# ============================================================================

# Smoothing factor for eye gaze (0.0 to 1.0)
# Higher value = more smoothing, slower response
# Lower value = less smoothing, faster response
EYE_SMOOTHING_FACTOR = 0.85

# Threshold for eye down score
# More negative = looking down/away
# Typical range: -0.3 to 0.3
EYE_DISTRACTION_THRESHOLD = -0.18


# ============================================================================
# CAMERA SETTINGS
# ============================================================================

# Camera index (0 = default camera, 1 = secondary camera, etc.)
CAMERA_INDEX = 0

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Path to the face detection model
FACE_DETECTION_MODEL_PATH = "blaze_face_short_range.tflite"


# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Show detailed status panel by default
SHOW_DEBUG_INFO = True

# Mirror the video feed (recommended for user-facing camera)
MIRROR_VIDEO = True


# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Enable logging to files
ENABLE_LOGGING = True

# Log file names
DEBUG_LOG_FILE = "microscope.log"
STATS_LOG_FILE = "dashboard.log"

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"


# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# MediaPipe Face Mesh settings
FACE_MESH_MAX_FACES = 1
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# Face Detection settings
FACE_DETECTION_MIN_CONFIDENCE = 0.5


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def load_preset(preset_name: str) -> dict:
    """
    Load a preset configuration.
    
    Presets:
        - "strict": Very sensitive, catches brief distractions
        - "balanced": Default settings
        - "relaxed": More forgiving, ignores brief glances away
    """
    presets = {
        "strict": {
            "FACE_PRESENT_TO_AWAY_TIME": 1.0,
            "HEAD_YAW_THRESHOLD": 15,
            "HEAD_PITCH_THRESHOLD": 15,
            "HEAD_ATTENTIVE_TO_DISTRACTED_TIME": 0.5,
            "EYE_DISTRACTION_THRESHOLD": -0.12,
        },
        "balanced": {
            "FACE_PRESENT_TO_AWAY_TIME": 2.0,
            "HEAD_YAW_THRESHOLD": 20,
            "HEAD_PITCH_THRESHOLD": 20,
            "HEAD_ATTENTIVE_TO_DISTRACTED_TIME": 1.0,
            "EYE_DISTRACTION_THRESHOLD": -0.18,
        },
        "relaxed": {
            "FACE_PRESENT_TO_AWAY_TIME": 3.0,
            "HEAD_YAW_THRESHOLD": 25,
            "HEAD_PITCH_THRESHOLD": 25,
            "HEAD_ATTENTIVE_TO_DISTRACTED_TIME": 1.5,
            "EYE_DISTRACTION_THRESHOLD": -0.25,
        }
    }
    
    return presets.get(preset_name, presets["balanced"])


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Using a preset
--------------------------
from config import load_preset

settings = load_preset("strict")
# Apply settings to your detectors

Example 2: Custom settings
--------------------------
import config

config.HEAD_YAW_THRESHOLD = 25  # More lenient head turning
config.EYE_SMOOTHING_FACTOR = 0.9  # More smoothing

Example 3: Different camera
--------------------------
config.CAMERA_INDEX = 1  # Use secondary camera
config.CAMERA_WIDTH = 640
config.CAMERA_HEIGHT = 480
"""
