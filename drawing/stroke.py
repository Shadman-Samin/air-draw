"""
Stroke data model.

Represents a single continuous drawing stroke as a sequence
of time-stamped 2D points with associated brush metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tracking.tracking_result import Point2D


@dataclass
class StrokePoint:
    """A single point in a stroke with timestamp and optional velocity."""
    position: Point2D
    timestamp: float  # monotonic time
    velocity: float = 0.0  # pixels per second


@dataclass
class Stroke:
    """
    A complete drawing stroke.

    Contains the sequence of points, brush settings at time of drawing,
    and bounding box for efficient region operations.
    """
    points: list[StrokePoint] = field(default_factory=list)
    color: tuple[int, int, int] = (0, 0, 255)
    size: int = 4
    opacity: float = 1.0
    brush_type: str = "pen"
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def duration_ms(self) -> float:
        """Duration of the stroke in milliseconds."""
        if len(self.points) < 2:
            return 0.0
        return (self.timestamp_end - self.timestamp_start) * 1000

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """
        Axis-aligned bounding box (x, y, w, h).

        Padded by brush size for correct region captures.
        """
        if not self.points:
            return (0, 0, 0, 0)

        xs = [p.position.x for p in self.points]
        ys = [p.position.y for p in self.points]

        pad = self.size + 2
        x_min = int(min(xs)) - pad
        y_min = int(min(ys)) - pad
        x_max = int(max(xs)) + pad
        y_max = int(max(ys)) + pad

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    @property
    def average_velocity(self) -> float:
        """Average velocity across all points."""
        if len(self.points) < 2:
            return 0.0
        velocities = [p.velocity for p in self.points if p.velocity > 0]
        return sum(velocities) / len(velocities) if velocities else 0.0

    def get_positions(self) -> list[Point2D]:
        """Extract just the position data."""
        return [p.position for p in self.points]
