"""
Deadzone filter for eliminating micro-jitter when stationary.

Suppresses tiny movements below a pixel threshold, preventing
the cursor from jittering when the hand is held still.
Only updates the output position if the input moves beyond
the deadzone radius from the last accepted position.
"""

from __future__ import annotations

from filters.base_filter import BaseFilter
from tracking.tracking_result import Point2D


class DeadzoneFilter(BaseFilter):
    """
    Hysteresis / deadzone filter.

    Outputs the last accepted position until the input moves
    beyond a configurable threshold distance (in pixels).
    """

    def __init__(self, threshold: float = 2.5):
        """
        Args:
            threshold: Minimum movement distance (pixels) to update output.
        """
        self._threshold = max(0.0, threshold)
        self._last_accepted: Point2D | None = None

    def process(self, point: Point2D) -> Point2D:
        """
        Apply deadzone filtering.

        Returns the previous accepted position if the new point
        is within the threshold distance.
        """
        if self._last_accepted is None:
            self._last_accepted = Point2D(x=point.x, y=point.y)
            return self._last_accepted

        distance = self._last_accepted.distance_to(point)

        if distance >= self._threshold:
            self._last_accepted = Point2D(x=point.x, y=point.y)

        return Point2D(x=self._last_accepted.x, y=self._last_accepted.y)

    def reset(self) -> None:
        """Reset filter state."""
        self._last_accepted = None

    @property
    def is_initialized(self) -> bool:
        return self._last_accepted is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update deadzone threshold at runtime."""
        self._threshold = max(0.0, value)
