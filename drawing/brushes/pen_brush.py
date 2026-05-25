"""
Pen brush — hard-edged circular brush.

The simplest brush type: constant-size circular stamps
with full opacity and hard edges. Uses cv2.circle with
LINE_AA for anti-aliased rendering.
"""

from __future__ import annotations

from canvas.layer import Layer
from drawing.brush_engine import BrushEngine
from tracking.tracking_result import Point2D


class PenBrush:
    """
    Hard-edged pen brush.

    Produces clean, consistent lines similar to a ballpoint pen.
    No soft edges, no texture — just clean anti-aliased circles.
    """

    def __init__(self):
        self._engine = BrushEngine()

    def draw_segment(
        self,
        layer: Layer,
        p1: Point2D,
        p2: Point2D,
        color: tuple[int, int, int],
        size: int,
        opacity: float = 1.0,
    ) -> None:
        """Draw a pen stroke segment between two points."""
        self._engine.draw_segment(layer, p1, p2, color, size, opacity)

    @property
    def name(self) -> str:
        return "Pen"

    @property
    def description(self) -> str:
        return "Hard-edged pen for clean, precise lines"
