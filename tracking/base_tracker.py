"""
Abstract base class for all tracking implementations.

Provides the interface that HandTracker and ColorTracker must implement,
ensuring consistent behavior regardless of tracking mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tracking.tracking_result import TrackingResult


class BaseTracker(ABC):
    """
    Abstract tracker interface.

    All concrete trackers (hand, color) must implement `process_frame`
    and manage their own lifecycle via `start` / `stop`.
    """

    def __init__(self):
        self._is_running: bool = False

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> TrackingResult:
        """
        Process a single video frame and return tracking results.

        Args:
            frame: BGR image from the camera (already horizontally flipped).
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            TrackingResult with detected hand landmarks or tracked points.
        """
        ...

    def start(self) -> None:
        """Initialize tracker resources."""
        self._is_running = True

    def stop(self) -> None:
        """Release tracker resources."""
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running
