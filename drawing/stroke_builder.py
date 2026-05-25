"""
Stroke builder — accumulates points into stroke objects.

Handles the incremental construction of strokes, computing
velocity between consecutive points and providing the
finished stroke when drawing ends.
"""

from __future__ import annotations

import time
from typing import Optional

from drawing.stroke import Stroke, StrokePoint
from tracking.tracking_result import Point2D


class StrokeBuilder:
    """
    Incrementally builds a Stroke from incoming points.

    Usage:
        builder.start(first_point)
        builder.add_point(next_point)  # repeat
        stroke = builder.finish()      # get complete stroke
    """

    def __init__(self):
        self._current_stroke: Stroke | None = None
        self._last_point: Point2D | None = None
        self._last_time: float = 0.0

    @property
    def is_active(self) -> bool:
        """Whether a stroke is currently being built."""
        return self._current_stroke is not None

    @property
    def last_point(self) -> Optional[Point2D]:
        """Last point added to the current stroke."""
        return self._last_point

    def start(self, point: Point2D) -> None:
        """
        Begin a new stroke at the given point.

        Args:
            point: Starting position for the stroke.
        """
        now = time.monotonic()
        self._current_stroke = Stroke(
            timestamp_start=now,
        )
        stroke_point = StrokePoint(
            position=Point2D(x=point.x, y=point.y),
            timestamp=now,
            velocity=0.0,
        )
        self._current_stroke.points.append(stroke_point)
        self._last_point = point
        self._last_time = now

    def add_point(self, point: Point2D) -> Optional[StrokePoint]:
        """
        Add a point to the current stroke.

        Computes velocity from the previous point for pressure simulation.

        Args:
            point: New position to add.

        Returns:
            The created StrokePoint, or None if no active stroke.
        """
        if self._current_stroke is None:
            return None

        now = time.monotonic()
        dt = now - self._last_time

        # Compute velocity (pixels per second)
        velocity = 0.0
        if dt > 0 and self._last_point is not None:
            distance = self._last_point.distance_to(point)
            velocity = distance / dt

        stroke_point = StrokePoint(
            position=Point2D(x=point.x, y=point.y),
            timestamp=now,
            velocity=velocity,
        )
        self._current_stroke.points.append(stroke_point)
        self._last_point = point
        self._last_time = now

        return stroke_point

    def finish(self) -> Optional[Stroke]:
        """
        Finish the current stroke and return it.

        Returns:
            The completed Stroke, or None if no active stroke.
        """
        if self._current_stroke is None:
            return None

        self._current_stroke.timestamp_end = time.monotonic()
        stroke = self._current_stroke

        # Reset state
        self._current_stroke = None
        self._last_point = None
        self._last_time = 0.0

        return stroke

    def cancel(self) -> None:
        """Cancel the current stroke without returning it."""
        self._current_stroke = None
        self._last_point = None
        self._last_time = 0.0
