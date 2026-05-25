"""
Gesture type definitions and gesture state container.

Defines all recognized gestures and provides a structured
output for the gesture recognition pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from tracking.tracking_result import Point2D


class GestureType(Enum):
    """Recognized hand gestures for application control."""

    NONE = auto()          # No hand detected
    DRAW = auto()          # Index finger up only → draw
    CURSOR = auto()        # Index + middle up → cursor / move
    PAUSE = auto()         # Open palm (all fingers) → pause drawing
    ERASE = auto()         # Closed fist → eraser mode
    UNDO = auto()          # Three fingers → undo (single-shot)
    REDO = auto()          # Four fingers → redo (single-shot)
    PINCH_COLOR = auto()   # Thumb + index pinch → color change
    PINCH_SIZE = auto()    # Thumb + middle pinch → brush size
    CLEAR = auto()         # Palm hold (sustained) → clear canvas


@dataclass(slots=True)
class GestureState:
    """
    Output of the gesture recognition system.

    Contains the recognized gesture, the drawing point (fingertip),
    and metadata about detection confidence and hand presence.
    """
    gesture: GestureType = GestureType.NONE
    fingertip: Optional[Point2D] = None
    hand_detected: bool = False
    confidence: float = 0.0
    finger_count: int = 0

    # Additional data for pinch gestures
    pinch_distance: float = 0.0

    @property
    def is_drawing(self) -> bool:
        """Whether the current gesture indicates active drawing."""
        return self.gesture == GestureType.DRAW and self.fingertip is not None

    @property
    def is_erasing(self) -> bool:
        """Whether the current gesture indicates erasing."""
        return self.gesture == GestureType.ERASE

    @property
    def is_single_shot(self) -> bool:
        """Whether this gesture should fire once (not continuously)."""
        return self.gesture in (GestureType.UNDO, GestureType.REDO, GestureType.CLEAR)
