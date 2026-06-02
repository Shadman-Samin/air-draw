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
    DrawingTool,
)
from canvas.layer_stack import LayerStack
from drawing.brush_engine import BrushEngine
from drawing.shape_renderer import ShapeRenderer
from drawing.stroke import Stroke
from drawing.stroke_builder import StrokeBuilder
from gestures.gesture_types import GestureState, GestureType
from tracking.tracking_result import Point2D

logger = logging.getLogger(__name__)


class _HandSession:
    """Per-hand drawing state machine session."""

    __slots__ = (
        "state", "current_stroke", "stroke_builder",
        "shape_anchor", "shape_last_point",
        "shape_has_preview",
    )

    def __init__(self):
        self.state = DrawingState.IDLE
        self.current_stroke: Stroke | None = None
        self.stroke_builder = StrokeBuilder()
        self.shape_anchor: Point2D | None = None
        self.shape_last_point: Point2D | None = None
        self.shape_has_preview: bool = False


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
        self._shape_renderer = ShapeRenderer()

        # Per-hand drawing state
        self._sessions: dict[int, _HandSession] = {}

        # Drawing tool (shared — only primary hand uses shapes)
        self._drawing_tool = DrawingTool.FREEHAND

        # Brush settings (shared)
        self._brush_size = DEFAULT_BRUSH_SIZE
        self._brush_color = DEFAULT_BRUSH_COLOR_BGR
        self._brush_opacity = 1.0
        self._eraser_size = 20
        self._secondary_brush_color = DEFAULT_BRUSH_COLOR_BGR

        # Undo/Redo history (simple snapshot-based for Phase 1)
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_undo = 30

        # Stroke counter for logging
        self._stroke_count = 0

    def _hand(self, hand_id: int) -> _HandSession:
        if hand_id not in self._sessions:
            self._sessions[hand_id] = _HandSession()
        return self._sessions[hand_id]

    @property
    def state(self) -> DrawingState:
        return self._hand(0).state

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

    @property
    def drawing_tool(self) -> DrawingTool:
        return self._drawing_tool

    @drawing_tool.setter
    def drawing_tool(self, tool: DrawingTool | str) -> None:
        if isinstance(tool, str):
            _map = {
                "freehand": DrawingTool.FREEHAND,
                "line": DrawingTool.LINE,
                "rectangle": DrawingTool.RECTANGLE,
                "circle": DrawingTool.CIRCLE,
                "arrow": DrawingTool.ARROW,
            }
            tool = _map.get(tool.lower(), DrawingTool.FREEHAND)
        self._drawing_tool = tool
        for s in self._sessions.values():
            self._cancel_shape_preview(s)

    @property
    def secondary_brush_color(self) -> tuple[int, int, int]:
        return self._secondary_brush_color

    @secondary_brush_color.setter
    def secondary_brush_color(self, value: tuple[int, int, int]) -> None:
        self._secondary_brush_color = value

    @property
    def shape_preview(self) -> tuple[Point2D, Point2D, str] | None:
        s = self._hand(0)
        if (
            s.state == DrawingState.DRAWING
            and self._drawing_tool != DrawingTool.FREEHAND
            and s.shape_anchor is not None
            and s.shape_last_point is not None
            and s.shape_has_preview
        ):
            return (
                s.shape_anchor,
                s.shape_last_point,
                self._drawing_tool.name.lower(),
            )
        return None

    def handle_input(
        self,
        smooth_point: Point2D | None,
        gesture_state: GestureState,
        brush_color: tuple[int, int, int] | None = None,
        hand_id: int = 0,
    ) -> None:
        """
        Process smoothed input point and gesture state.

        This is the main entry point called every frame by the pipeline.
        Manages state transitions and delegates to drawing operations.

        Args:
            smooth_point: Smoothed cursor position, or None if no hand.
            gesture_state: Current gesture recognition result.
            brush_color: Optional override (e.g. secondary hand color).
            hand_id: Which hand this input belongs to (0 = primary).
        """
        active_color = brush_color if brush_color is not None else self._brush_color
        gesture = gesture_state.gesture
        s = self._hand(hand_id)

        # ── Handle single-shot gestures (primary hand only) ──
        if hand_id == 0:
            if gesture == GestureType.UNDO:
                self._finalize_shape_if_active(s)
                self._end_stroke_if_active(s)
                self.undo()
                return

            if gesture == GestureType.REDO:
                self._finalize_shape_if_active(s)
                self._end_stroke_if_active(s)
                self.redo()
                return

            if gesture == GestureType.CLEAR:
                self._finalize_shape_if_active(s)
                self._end_stroke_if_active(s)
                self.clear_canvas()
                return

        # ── Drawing state machine ──
        if gesture == GestureType.DRAW and smooth_point is not None:
            if self._drawing_tool != DrawingTool.FREEHAND and hand_id == 0:
                self._handle_shape_draw(smooth_point, active_color, s)
            elif s.state != DrawingState.DRAWING:
                self._start_stroke(smooth_point, s)
            else:
                self._continue_stroke(smooth_point, s)

        elif gesture == GestureType.ERASE and smooth_point is not None:
            if s.state != DrawingState.ERASING:
                self._end_stroke_if_active(s)
                s.state = DrawingState.ERASING
            self._erase_at(smooth_point)

        else:
            self._finalize_shape_if_active(s)
            self._end_stroke_if_active(s)
            if gesture == GestureType.NONE:
                s.state = DrawingState.IDLE
            else:
                s.state = DrawingState.READY

    def _start_stroke(self, point: Point2D, s: _HandSession) -> None:
        """Begin a new drawing stroke."""
        self._save_undo_state()

        s.state = DrawingState.DRAWING
        s.stroke_builder.start(point)
        self._stroke_count += 1

        logger.debug("Stroke #%d started at (%.1f, %.1f)",
                      self._stroke_count, point.x, point.y)

    def _continue_stroke(self, point: Point2D, s: _HandSession) -> None:
        """Add a point to the current stroke and render."""
        layer = self.layer_stack.active_layer
        if layer is None or layer.locked:
            return

        prev_point = s.stroke_builder.last_point
        s.stroke_builder.add_point(point)

        if prev_point is not None:
            self._brush_engine.draw_segment(
                layer=layer,
                p1=prev_point,
                p2=point,
                color=self._brush_color,
                size=self._brush_size,
                opacity=self._brush_opacity,
            )

    def _handle_shape_draw(self, point: Point2D, color: tuple[int, int, int], s: _HandSession) -> None:
        layer = self.layer_stack.active_layer
        if layer is None or layer.locked:
            return

        if s.state != DrawingState.DRAWING:
            self._save_undo_state()
            s.shape_anchor = point
            s.shape_last_point = point
            s.shape_has_preview = False
            s.state = DrawingState.DRAWING
            return

        if s.shape_anchor is None:
            return

        if s.shape_anchor.distance_to(point) < 3.0:
            return

        # Shape is NOT drawn on the layer during preview — the HUD
        # overlay renders it via draw_frame_preview. This avoids
        # erasing concurrent strokes from other hands.
        s.shape_last_point = point
        s.shape_has_preview = True

    def _finalize_shape_if_active(self, s: _HandSession) -> None:
        if (
            s.state != DrawingState.DRAWING
            or self._drawing_tool == DrawingTool.FREEHAND
            or s.shape_anchor is None
        ):
            return

        if s.shape_has_preview:
            layer = self.layer_stack.active_layer
            if layer is not None and not layer.locked:
                tool_name = self._drawing_tool.name.lower()
                self._shape_renderer.draw_shape(
                    layer, tool_name, s.shape_anchor, s.shape_last_point,
                    self._brush_color, self._brush_size, self._brush_opacity,
                )
        else:
            self._cancel_shape_preview(s, pop_undo=True)

        self._clear_shape_session(s)
        s.state = DrawingState.READY

    def _clear_shape_session(self, s: _HandSession) -> None:
        s.shape_anchor = None
        s.shape_last_point = None
        s.shape_has_preview = False

    def _cancel_shape_preview(self, s: _HandSession, pop_undo: bool = False) -> None:
        if pop_undo and self._undo_stack:
            self._undo_stack.pop()
        self._clear_shape_session(s)

    def _end_stroke_if_active(self, s: _HandSession) -> None:
        """End the current freehand stroke if one is active."""
        if s.state != DrawingState.DRAWING:
            return
        if self._drawing_tool != DrawingTool.FREEHAND:
            return

        stroke = s.stroke_builder.finish()
        if stroke is not None:
            logger.debug(
                "Stroke #%d ended with %d points",
                self._stroke_count, len(stroke.points),
            )
        s.state = DrawingState.READY

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

    def export_bgr(self, background_bgr: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Export flattened canvas as opaque BGR image."""
        flat = self.layer_stack.flatten()
        composite = flat.data
        h, w = composite.shape[:2]
        result = np.full((h, w, 3), background_bgr, dtype=np.uint8)
        alpha = composite[:, :, 3:4].astype(np.float32) / 255.0
        fg = composite[:, :, :3].astype(np.float32)
        blended = (fg * alpha + result.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return blended

    def save_to_file(self, filepath: str, background_bgr: tuple[int, int, int] = (255, 255, 255)) -> bool:
        """Save canvas to PNG/JPG/BMP."""
        import cv2

        image = self.export_bgr(background_bgr)
        ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else "png"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in ("jpg", "jpeg") else []
        if ext == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        return bool(cv2.imwrite(filepath, image, params))

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
        self._sessions.clear()
        logger.info("Canvas resized to %d×%d", width, height)
