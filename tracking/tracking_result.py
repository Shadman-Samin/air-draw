"""
Data classes for tracking output.

Provides a unified representation of tracking results regardless
of whether the input comes from MediaPipe hand tracking or
HSV color tracking. All coordinates are in pixel space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Point2D:
    """A 2D point in pixel coordinates."""
    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        """Euclidean distance to another point."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def lerp(self, other: Point2D, t: float) -> Point2D:
        """Linear interpolation toward another point."""
        return Point2D(
            x=self.x + (other.x - self.x) * t,
            y=self.y + (other.y - self.y) * t,
        )

    def as_int_tuple(self) -> tuple[int, int]:
        """Return (x, y) as integer tuple for OpenCV functions."""
        return (int(round(self.x)), int(round(self.y)))

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass(slots=True)
class NormalizedPoint:
    """A 2D point in normalized [0, 1] coordinates."""
    x: float
    y: float
    z: float = 0.0

    def to_pixel(self, width: int, height: int) -> Point2D:
        """Convert to pixel coordinates given frame dimensions."""
        return Point2D(x=self.x * width, y=self.y * height)


@dataclass(slots=True)
class HandLandmarks:
    """
    Detected hand landmarks from MediaPipe.

    Contains 21 landmark points in pixel coordinates,
    handedness label, and detection confidence.
    """
    landmarks: list[Point2D] = field(default_factory=list)
    handedness: str = "Right"  # "Left" or "Right"
    confidence: float = 0.0

    # Key landmark indices (MediaPipe convention)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    @property
    def index_tip(self) -> Optional[Point2D]:
        """Get index fingertip position."""
        if len(self.landmarks) > self.INDEX_TIP:
            return self.landmarks[self.INDEX_TIP]
        return None

    @property
    def thumb_tip(self) -> Optional[Point2D]:
        """Get thumb tip position."""
        if len(self.landmarks) > self.THUMB_TIP:
            return self.landmarks[self.THUMB_TIP]
        return None

    @property
    def middle_tip(self) -> Optional[Point2D]:
        """Get middle fingertip position."""
        if len(self.landmarks) > self.MIDDLE_TIP:
            return self.landmarks[self.MIDDLE_TIP]
        return None

    @property
    def wrist(self) -> Optional[Point2D]:
        """Get wrist position."""
        if len(self.landmarks) > self.WRIST:
            return self.landmarks[self.WRIST]
        return None

    def get_landmark(self, index: int) -> Optional[Point2D]:
        """Safely get a landmark by index."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None


@dataclass(slots=True)
class TrackingResult:
    """
    Unified tracking result from any tracking mode.

    Contains detected hands, timestamp, and frame metadata.
    """
    hands: list[HandLandmarks] = field(default_factory=list)
    timestamp_ms: int = 0
    frame_width: int = 0
    frame_height: int = 0

    @property
    def primary_hand(self) -> Optional[HandLandmarks]:
        """Get the first detected hand (primary drawing hand)."""
        return self.hands[0] if self.hands else None

    @property
    def secondary_hand(self) -> Optional[HandLandmarks]:
        """Get the second detected hand (control hand)."""
        return self.hands[1] if len(self.hands) > 1 else None

    @property
    def has_hands(self) -> bool:
        """Whether any hands were detected."""
        return len(self.hands) > 0

    @property
    def primary_fingertip(self) -> Optional[Point2D]:
        """Get the index fingertip of the primary hand."""
        hand = self.primary_hand
        if hand:
            return hand.index_tip
        return None
