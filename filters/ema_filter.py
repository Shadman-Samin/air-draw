"""
Exponential Moving Average filter for 2D point smoothing.

Applies EMA independently to X and Y coordinates:
  smoothed = alpha * current + (1 - alpha) * previous_smoothed

Higher alpha → more responsive (less smoothing).
Lower alpha → smoother (more lag).
"""

from __future__ import annotations

from filters.base_filter import BaseFilter
from tracking.tracking_result import Point2D


class ExponentialMovingAverage(BaseFilter):
    """
    EMA-based smoothing filter.

    Best used after a Kalman filter to provide an additional
    layer of high-frequency noise reduction.
    """

    def __init__(self, alpha: float = 0.45):
        """
        Args:
            alpha: Smoothing factor in [0, 1].
                Higher = more responsive, lower = smoother.
        """
        self._alpha = max(0.01, min(1.0, alpha))
        self._smoothed: Point2D | None = None

    def process(self, point: Point2D) -> Point2D:
        """Apply EMA smoothing to the input point."""
        if self._smoothed is None:
            self._smoothed = Point2D(x=point.x, y=point.y)
            return self._smoothed

        self._smoothed = Point2D(
            x=self._alpha * point.x + (1 - self._alpha) * self._smoothed.x,
            y=self._alpha * point.y + (1 - self._alpha) * self._smoothed.y,
        )
        return Point2D(x=self._smoothed.x, y=self._smoothed.y)

    def reset(self) -> None:
        """Reset filter state."""
        self._smoothed = None

    @property
    def is_initialized(self) -> bool:
        return self._smoothed is not None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Update smoothing factor at runtime."""
        self._alpha = max(0.01, min(1.0, value))
