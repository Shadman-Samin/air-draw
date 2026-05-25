"""
Global Event Hub / Signal Broker.

Uses PyQt6 signals to decouple components (tracking, drawing engine, UI)
and allow thread-safe communications.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from gestures.gesture_types import GestureState
from tracking.tracking_result import TrackingResult


class EventHub(QObject):
    """
    Central hub for thread-safe event routing via PyQt Signals.

    Avoids direct coupling between the UI, the capture thread,
    and background processing components.
    """

    # --- Video & Pipeline Signals ---
    # Emitted when a frame has been completely processed: (q_image, tracking_result, gesture_state)
    frame_processed = pyqtSignal(QImage, object, object)
    
    # Process FPS reporting
    fps_updated = pyqtSignal(float)

    # --- Tool & Brush Settings Signals ---
    # Emitted when the user alters brush color: (bgr_color)
    color_changed = pyqtSignal(tuple)
    
    # Emitted when brush size changes: (size_px)
    brush_size_changed = pyqtSignal(int)
    
    # Emitted when active brush changes: (brush_name)
    brush_type_changed = pyqtSignal(str)
    
    # Emitted when eraser size changes: (size_px)
    eraser_size_changed = pyqtSignal(int)
    
    # Emitted when tracking mode changes: ("hand", "color")
    tracking_mode_changed = pyqtSignal(str)

    # --- Canvas & History Signals ---
    # Undo / Redo availability
    undo_available = pyqtSignal(bool)
    redo_available = pyqtSignal(bool)
    
    # Core stack operations
    canvas_cleared = pyqtSignal()
    layer_stack_changed = pyqtSignal()

    # --- System Status ---
    # Messages to render in status bars or logs: (message, duration_ms)
    status_message = pyqtSignal(str, int)


# Global singleton instance
events = EventHub()
