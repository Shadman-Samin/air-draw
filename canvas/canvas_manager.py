"""
Canvas manager — orchestrates drawing input, state transitions, and canvas operations.

This is the central controller that connects gesture recognition output
to the drawing engine and manages the drawing state machine:

  IDLE → DRAWING → IDLE
  IDLE → ERASING → IDLE
"""

from __future__ import annotations

import logging

import numpy as np

from app.constants import (
    DEFAULT_BRUSH_COLOR_BGR,
    DEFAULT_BRUSH_SIZE,
    DrawingState,
)
from canvas.layer_stack import LayerStack
from drawing.brush_engine import BrushEngine
from drawing.stroke import Stroke
from drawing.stroke_builder import StrokeBuilder
from gestures.gesture_types import GestureState, GestureType
from tracking.tracking_result import Point2D

logger = logging.getLogger(__name__)


class CanvasManager:
    """
    Central canvas controller.

    Manages the drawing state machine, translates gesture input
    into drawing operations on the active layer, and provides
    canvas-level operations (clear, undo, redo).
    """

    def __init__(self, width: int, height: int):
        """
        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self._width = width
        self._height = height

        # Layer system
        self.layer_stack = LayerStack(width, height)

        # Drawing engine
        self._brush_engine = BrushEngine()
        self._stroke_builder = StrokeBuilder()

        # Drawing state
        self._state = DrawingState.IDLE
        self._current_stroke: Stroke | None = None

        # Brush settings
        self._brush_size = DEFAULT_BRUSH_SIZE
        self._brush_color = DEFAULT_BRUSH_COLOR_BGR
        self._brush_opacity = 1.0
        self._eraser_size = 20

        # Undo/Redo history (simple snapshot-based for Phase 1)
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_undo = 30

        # Stroke counter for logging
        self._stroke_count = 0

    @property
    def state(self) -> DrawingState:
        return self._state

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @brush_size.setter
    def brush_size(self, value: int) -> None:
        self._brush_size = max(1, min(100, value))

    @property
    def brush_color(self) -> tuple[int, int, int]:
        return self._brush_color

    @brush_color.setter
    def brush_color(self, value: tuple[int, int, int]) -> None:
        self._brush_color = value

    @property
    def brush_opacity(self) -> float:
        return self._brush_opacity

    @brush_opacity.setter
    def brush_opacity(self, value: float) -> None:
        self._brush_opacity = max(0.0, min(1.0, value))

    @property
    def eraser_size(self) -> int:
        return self._eraser_size

    @eraser_size.setter
    def eraser_size(self, value: int) -> None:
        self._eraser_size = max(1, min(100, value))

    def handle_input(
        self,
        smooth_point: Point2D | None,
        gesture_state: GestureState,
    ) -> None:
        """
        Process smoothed input point and gesture state.

        This is the main entry point called every frame by the pipeline.
        Manages state transitions and delegates to drawing operations.

        Args:
            smooth_point: Smoothed cursor position, or None if no hand.
            gesture_state: Current gesture recognition result.
        """
        gesture = gesture_state.gesture

        # ── Handle single-shot gestures ──
        if gesture == GestureType.UNDO:
            self._end_stroke_if_active()
            self.undo()
            return

        if gesture == GestureType.REDO:
            self._end_stroke_if_active()
            self.redo()
            return

        if gesture == GestureType.CLEAR:
            self._end_stroke_if_active()
            self.clear_canvas()
            return

        # ── Drawing state machine ──
        if gesture == GestureType.DRAW and smooth_point is not None:
            if self._state != DrawingState.DRAWING:
                self._start_stroke(smooth_point)
            else:
                self._continue_stroke(smooth_point)

        elif gesture == GestureType.ERASE and smooth_point is not None:
            if self._state != DrawingState.ERASING:
                self._end_stroke_if_active()
                self._state = DrawingState.ERASING
            self._erase_at(smooth_point)

        else:
            # Any other gesture ends the current stroke
            self._end_stroke_if_active()
            if gesture == GestureType.NONE:
                self._state = DrawingState.IDLE
            else:
                self._state = DrawingState.READY

    def _start_stroke(self, point: Point2D) -> None:
        """Begin a new drawing stroke."""
        self._save_undo_state()

        self._state = DrawingState.DRAWING
        self._stroke_builder.start(point)
        self._stroke_count += 1

        logger.debug("Stroke #%d started at (%.1f, %.1f)",
                      self._stroke_count, point.x, point.y)

    def _continue_stroke(self, point: Point2D) -> None:
        """Add a point to the current stroke and render."""
        layer = self.layer_stack.active_layer
        if layer is None or layer.locked:
            return

        prev_point = self._stroke_builder.last_point
        self._stroke_builder.add_point(point)

        if prev_point is not None:
            self._brush_engine.draw_segment(
                layer=layer,
                p1=prev_point,
                p2=point,
                color=self._brush_color,
                size=self._brush_size,
                opacity=self._brush_opacity,
            )

    def _end_stroke_if_active(self) -> None:
        """End the current stroke if one is active."""
        if self._state == DrawingState.DRAWING:
            stroke = self._stroke_builder.finish()
            if stroke is not None:
                logger.debug(
                    "Stroke #%d ended with %d points",
                    self._stroke_count, len(stroke.points),
                )
            self._state = DrawingState.READY

    def _erase_at(self, point: Point2D) -> None:
        """Erase at the given position on the active layer."""
        layer = self.layer_stack.active_layer
        if layer is None or layer.locked:
            return

        self._brush_engine.erase_at(
            layer=layer,
            center=point,
            radius=self._eraser_size,
        )

    def _save_undo_state(self) -> None:
        """Save current active layer state for undo."""
        layer = self.layer_stack.active_layer
        if layer is None:
            return

        self._undo_stack.append(layer.get_snapshot())
        self._redo_stack.clear()

        # Limit undo history
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """
        Undo the last drawing operation.

        Returns:
            True if undo was successful.
        """
        if not self._undo_stack:
            logger.debug("Nothing to undo")
            return False

        layer = self.layer_stack.active_layer
        if layer is None:
            return False

        # Save current state for redo
        self._redo_stack.append(layer.get_snapshot())

        # Restore previous state
        snapshot = self._undo_stack.pop()
        layer.restore_snapshot(snapshot)

        logger.debug("Undo successful (%d remaining)", len(self._undo_stack))
        return True

    def redo(self) -> bool:
        """
        Redo the last undone operation.

        Returns:
            True if redo was successful.
        """
        if not self._redo_stack:
            logger.debug("Nothing to redo")
            return False

        layer = self.layer_stack.active_layer
        if layer is None:
            return False

        # Save current state for undo
        self._undo_stack.append(layer.get_snapshot())

        # Restore redo state
        snapshot = self._redo_stack.pop()
        layer.restore_snapshot(snapshot)

        logger.debug("Redo successful (%d remaining)", len(self._redo_stack))
        return True

    def clear_canvas(self) -> None:
        """Clear all layers."""
        self._save_undo_state()
        self.layer_stack.clear_all()
        logger.info("Canvas cleared")

    def resize(self, width: int, height: int) -> None:
        """
        Resize the canvas.

        Note: This discards the current undo history.
        """
        self._width = width
        self._height = height
        self.layer_stack = LayerStack(width, height)
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.info("Canvas resized to %d×%d", width, height)
