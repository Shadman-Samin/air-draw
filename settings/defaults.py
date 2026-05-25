"""
Default application configurations.

Provides a structured dictionary of all tunable parameters in the system,
categorized by component (tracking, gesture, filtering, drawing, ui).
"""

from __future__ import annotations

DEFAULT_SETTINGS = {
    # Camera & Capture
    "camera": {
        "device_index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "mirror": True,
        "auto_exposure": True,
    },
    
    # Hand Tracking
    "hand_tracking": {
        "max_num_hands": 1,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
        "model_complexity": 1,
    },
    
    # Color Tracking
    "color_tracking": {
        "hsv_lower": [160, 100, 100],  # Default: Vibrant Red/Pink
        "hsv_upper": [180, 255, 255],
        "min_contour_area": 500,       # Min size of marker in pixels
        "max_contour_area": 50000,
        "marker_color_bgr": [0, 0, 255],
    },
    
    # Smoothing / Filtering Chain
    "filtering": {
        "active_filters": ["kalman", "ema", "deadzone"],
        "kalman": {
            "process_noise": 0.03,      # Q - dynamic movement uncertainty (lower = smoother/more lag)
            "measurement_noise": 1.5,   # R - sensor noise (higher = smoother)
        },
        "ema": {
            "alpha": 0.35,              # Coefficient for EMA filter (lower = smoother/more lag)
        },
        "deadzone": {
            "threshold": 2.2,           # Pixels of movement required to break stasis
        }
    },
    
    # Gesture Recognition
    "gesture": {
        "pinch_threshold": 0.04,        # Ratio of hand distance for drawing gesture
        "erase_threshold": 0.06,        # Distance for erase gesture (open hand)
        "hold_duration_s": 0.5,         # Hold duration for button hover clicks
        "fist_threshold": 0.08,         # Ratio of finger to palm distance for fist (clear canvas)
    },
    
    # Drawing & Brushes
    "drawing": {
        "active_brush": "pen",
        "brush_size": 6,                # In pixels
        "brush_opacity": 1.0,           # [0.0, 1.0]
        "brush_color_bgr": [50, 50, 255], # Bright Red
        "eraser_size": 28,              # Erasing diameter
    },
    
    # UI/UX Preferences
    "ui": {
        "theme": "dark",
        "show_fps": True,
        "show_landmarks": True,
        "show_gesture_overlay": True,
        "brush_preview_opacity": 0.6,
    }
}
