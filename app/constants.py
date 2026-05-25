"""
Application-wide constants and enumerations.

Defines all enum types and magic constants used across modules
to ensure consistent behavior and avoid hard-coded values.
"""

from enum import Enum, auto


# ──────────────────────────────────────────────
# Tracking
# ──────────────────────────────────────────────

class TrackingMode(Enum):
    """Input tracking method."""
    HAND = auto()
    COLOR = auto()


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────

class BrushType(Enum):
    """Available brush types for the drawing engine."""
    PEN = auto()
    SOFT_BRUSH = auto()
    MARKER = auto()
    HIGHLIGHTER = auto()
    PENCIL = auto()
    SPRAY = auto()
    CALLIGRAPHY = auto()
    ERASER = auto()


class DrawingTool(Enum):
    """Shape and freehand drawing tools."""
    FREEHAND = auto()
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    POLYGON = auto()
    ARROW = auto()
    ERASER = auto()


class BlendMode(Enum):
    """Layer blending modes."""
    NORMAL = auto()
    MULTIPLY = auto()
    SCREEN = auto()
    OVERLAY = auto()
    SOFT_LIGHT = auto()


# ──────────────────────────────────────────────
# Canvas
# ──────────────────────────────────────────────

class CanvasBackground(Enum):
    """Canvas background presets."""
    WHITE = auto()
    BLACK = auto()
    TRANSPARENT = auto()
    CUSTOM = auto()


# ──────────────────────────────────────────────
# Application State
# ──────────────────────────────────────────────

class AppState(Enum):
    """Top-level application state."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


# ──────────────────────────────────────────────
# Drawing State
# ──────────────────────────────────────────────

class DrawingState(Enum):
    """Current drawing interaction state."""
    IDLE = auto()
    READY = auto()
    DRAWING = auto()
    ERASING = auto()
    PAUSED = auto()


# ──────────────────────────────────────────────
# Numeric Constants
# ──────────────────────────────────────────────

# Camera defaults
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30

# Processing
MAX_FRAME_QUEUE_SIZE = 2
TRACKING_RESOLUTION_WIDTH = 640
TRACKING_RESOLUTION_HEIGHT = 480

# Drawing defaults
DEFAULT_BRUSH_SIZE = 4
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 100
DEFAULT_BRUSH_OPACITY = 1.0
DEFAULT_BRUSH_COLOR_BGR = (50, 50, 255)  # Red in BGR

# Smoothing
DEFAULT_KALMAN_PROCESS_NOISE = 0.03
DEFAULT_KALMAN_MEASUREMENT_NOISE = 1.0
DEFAULT_EMA_ALPHA = 0.45
DEFAULT_DEADZONE_THRESHOLD = 2.5

# Gesture thresholds
PINCH_DISTANCE_THRESHOLD = 0.05  # Normalized distance
GESTURE_HOLD_DURATION_MS = 800   # For clear canvas confirmation
GESTURE_DEBOUNCE_MS = 300        # Debounce for single-shot gestures

# Performance
TARGET_FPS = 30
MIN_ACCEPTABLE_FPS = 20
FPS_AVERAGING_WINDOW = 30

# UI
TOOLBAR_ICON_SIZE = 32
STATUS_BAR_HEIGHT = 28
