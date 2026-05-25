"""
Abstract base class for smoothing filters.

All smoothing filters (Kalman, EMA, Moving Average, Deadzone)
implement this interface for composability in the FilterChain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tracking.tracking_result import Point2D


class BaseFilter(ABC):
    """
    Abstract smoothing filter.

    Filters process 2D points sequentially, maintaining internal state
    to produce smoothed output. They must be resettable for when
    tracking is lost and reacquired.
    """

    @abstractmethod
    def process(self, point: Point2D) -> Point2D:
        """
        Apply filter to a raw input point.

        Args:
            point: Noisy input point from tracking.

        Returns:
            Smoothed output point.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset filter state.

        Called when tracking is lost so the filter doesn't
        interpolate between disconnected tracking sessions.
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Whether the filter has received at least one point."""
        ...
