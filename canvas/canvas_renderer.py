"""
Canvas renderer — composites layers and overlays onto camera frames.

Handles the final stage of the rendering pipeline:
1. Composite all visible layers (bottom-up alpha blending)
2. Overlay the canvas composite onto the camera frame
3. Draw UI elements (cursor, gesture indicator, HUD)
"""

from __future__ import annotations

import cv2
import numpy as np

from canvas.layer_stack import LayerStack
from gestures.gesture_types import GestureState, GestureType
from tracking.tracking_result import Point2D


class CanvasRenderer:
    """
    Composites canvas layers and overlays them on camera frames.

    Maintains a cached composite that is only recomputed when
    layers are modified (dirty flag optimization).
    """

    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

        # Cached composite of all layers (BGRA)
        self._composite_cache: np.ndarray | None = None

    def composite_layers(self, layer_stack: LayerStack) -> np.ndarray:
        """
        Composite all visible layers into a single BGRA image.

        Uses cached result if no layers have been modified.

        Args:
            layer_stack: The layer stack to composite.

        Returns:
            BGRA composite image.
        """
        if not layer_stack.any_dirty and self._composite_cache is not None:
            return self._composite_cache

        # Determine if we can do partial compositing
        can_do_partial = (
            self._composite_cache is not None
            and self._composite_cache.shape[:2] == (self._height, self._width)
        )

        if can_do_partial:
            # Check if any dirty layer has None for its dirty_rect (full dirty)
            for layer in layer_stack.layers:
                if layer.visible and layer.is_dirty and layer.dirty_rect is None:
                    can_do_partial = False
                    break

        if can_do_partial:
            # Find the bounding box union of all dirty rects of visible layers
            x_min, y_min = self._width, self._height
            x_max, y_max = 0, 0
            has_dirty_regions = False

            for layer in layer_stack.layers:
                if layer.visible and layer.is_dirty:
                    r = layer.dirty_rect
                    if r is not None:
                        rx, ry, rw, rh = r
                        rx = max(0, min(rx, self._width))
                        ry = max(0, min(ry, self._height))
                        rw = max(0, min(rw, self._width - rx))
                        rh = max(0, min(rh, self._height - ry))
                        if rw > 0 and rh > 0:
                            x_min = min(x_min, rx)
                            y_min = min(y_min, ry)
                            x_max = max(x_max, rx + rw)
                            y_max = max(y_max, ry + rh)
                            has_dirty_regions = True

            if has_dirty_regions and x_min < x_max and y_min < y_max:
                # Clear only the dirty ROI in the cached composite
                self._composite_cache[y_min:y_max, x_min:x_max] = 0

                # Composite only the ROI from all visible layers
                for layer in layer_stack.layers:
                    if not layer.visible:
                        continue
                    
                    dst_roi = self._composite_cache[y_min:y_max, x_min:x_max]
                    src_roi = layer.data[y_min:y_max, x_min:x_max]
                    LayerStack._alpha_composite_inplace(
                        dst_roi, src_roi, layer.opacity,
                    )
                
                # Clear dirty flags on all layers
                for layer in layer_stack.layers:
                    layer.clear_dirty()
                
                return self._composite_cache

        # Fallback: full recomposition
        composite = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8,
        )

        # Bottom-up compositing
        for layer in layer_stack.layers:
            if not layer.visible:
                continue

            LayerStack._alpha_composite_inplace(
                composite, layer.data, layer.opacity,
            )
            layer.clear_dirty()

        self._composite_cache = composite
        return composite

    def render_frame(
        self,
        camera_frame: np.ndarray,
        layer_stack: LayerStack,
        gesture_state: GestureState | None = None,
        smooth_point: Point2D | None = None,
        brush_size: int = 4,
        brush_color: tuple[int, int, int] = (50, 50, 255),
        show_cursor: bool = True,
        whiteboard_mode: bool = False,
        whiteboard_bg: tuple[int, int, int] = (255, 255, 255),
        show_grid: bool = True,
    ) -> np.ndarray:
        """
        Produce the final display frame.

        Composites canvas layers onto the camera frame and draws
        UI overlays (cursor, gesture indicator).

        Args:
            camera_frame: BGR camera frame (already mirrored).
            layer_stack: Canvas layer stack.
            gesture_state: Current gesture state for UI indicators.
            smooth_point: Smoothed cursor position.
            brush_size: Current brush size for cursor display.
            brush_color: Current brush color (BGR).
            show_cursor: Whether to draw the cursor indicator.

        Returns:
            BGR frame ready for display.
        """
        h, w = camera_frame.shape[:2]

        # Ensure canvas matches frame dimensions
        if w != self._width or h != self._height:
            self._width = w
            self._height = h
            self._composite_cache = None

        canvas = self.composite_layers(layer_stack)

        if whiteboard_mode:
            display = np.full((h, w, 3), whiteboard_bg, dtype=np.uint8)
            if show_grid:
                self._draw_grid(display)
            display = self._overlay_on_frame(display, canvas)
        else:
            display = self._overlay_on_frame(camera_frame, canvas)

        # Draw UI elements
        if show_cursor and gesture_state is not None:
            self._draw_cursor(display, gesture_state, smooth_point, brush_size, brush_color)
            self._draw_gesture_indicator(display, gesture_state)

        return display

    def _overlay_on_frame(
        self,
        frame_bgr: np.ndarray,
        canvas_bgra: np.ndarray,
    ) -> np.ndarray:
        """
        Overlay BGRA canvas onto BGR camera frame.

        Only processes pixels where canvas alpha > 0 for efficiency.
        Optimized with fast integer blending.
        """
        h, w = frame_bgr.shape[:2]
        ch, cw = canvas_bgra.shape[:2]

        # Resize canvas if dimensions don't match
        if cw != w or ch != h:
            canvas_bgra = cv2.resize(canvas_bgra, (w, h))

        # Extract alpha channel
        alpha = canvas_bgra[:, :, 3]

        # Fast path: if canvas is entirely transparent, return camera frame
        if cv2.countNonZero(alpha) == 0:
            return frame_bgr.copy()

        # Find the active bounding box (ROI) of non-zero alpha pixels
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return frame_bgr.copy()

        rx, ry, rw, rh = cv2.boundingRect(coords)
        rx = max(0, min(rx, w))
        ry = max(0, min(ry, h))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))

        # Slice ROI regions for alpha, canvas, and camera frame
        alpha_roi = alpha[ry:ry+rh, rx:rx+rw]
        canvas_bgr_roi = canvas_bgra[ry:ry+rh, rx:rx+rw, :3]
        frame_bgr_roi = frame_bgr[ry:ry+rh, rx:rx+rw]

        # Fast integer blending to avoid expensive float conversions and allocations
        alpha_3ch = alpha_roi[:, :, np.newaxis].astype(np.uint16)
        canvas_roi_16 = canvas_bgr_roi.astype(np.uint16)
        frame_roi_16 = frame_bgr_roi.astype(np.uint16)

        # Blend: (canvas * alpha + frame * (255 - alpha)) // 255
        blended_roi = (canvas_roi_16 * alpha_3ch + frame_roi_16 * (255 - alpha_3ch)) // 255

        # Paste blended ROI back into a copy of the camera frame
        result = frame_bgr.copy()
        result[ry:ry+rh, rx:rx+rw] = blended_roi.astype(np.uint8)
        return result

    def _draw_cursor(
        self,
        frame: np.ndarray,
        gesture_state: GestureState,
        smooth_point: Point2D | None,
        brush_size: int,
        brush_color: tuple[int, int, int],
    ) -> None:
        """Draw cursor indicator based on current gesture."""
        if smooth_point is None:
            return

        cx, cy = smooth_point.as_int_tuple()
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        if not (0 <= cx < w and 0 <= cy < h):
            return

        gesture = gesture_state.gesture

        if gesture == GestureType.DRAW:
            # Drawing cursor: filled circle with brush color + white outline
            radius = max(brush_size // 2, 2)
            cv2.circle(frame, (cx, cy), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, brush_color, -1, cv2.LINE_AA)

        elif gesture == GestureType.ERASE:
            # Eraser cursor: white circle outline
            radius = max(brush_size, 10)
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius, (180, 180, 180), 1, cv2.LINE_AA)

        elif gesture == GestureType.CURSOR:
            # Cursor mode: small crosshair
            size = 12
            cv2.line(frame, (cx - size, cy), (cx + size, cy),
                     (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - size), (cx, cy + size),
                     (0, 255, 0), 1, cv2.LINE_AA)

        elif gesture == GestureType.PAUSE:
            # Pause indicator: double vertical bars
            cv2.rectangle(frame, (cx - 10, cy - 12), (cx - 4, cy + 12),
                          (0, 200, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (cx + 4, cy - 12), (cx + 10, cy + 12),
                          (0, 200, 255), -1, cv2.LINE_AA)

        else:
            # Default: small dot
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)

    def _draw_gesture_indicator(
        self,
        frame: np.ndarray,
        gesture_state: GestureState,
    ) -> None:
        """Draw gesture label in the top-left corner."""
        gesture = gesture_state.gesture
        if gesture == GestureType.NONE:
            return

        labels = {
            GestureType.DRAW: ("DRAW", (50, 50, 255)),
            GestureType.CURSOR: ("CURSOR", (0, 255, 0)),
            GestureType.PAUSE: ("PAUSE", (0, 200, 255)),
            GestureType.ERASE: ("ERASE", (180, 180, 180)),
            GestureType.UNDO: ("UNDO", (255, 200, 0)),
            GestureType.REDO: ("REDO", (255, 200, 0)),
            GestureType.CLEAR: ("CLEAR!", (0, 0, 255)),
        }

        label, color = labels.get(gesture, ("", (255, 255, 255)))
        if label:
            # Background rectangle
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
            )
            cv2.rectangle(
                frame, (8, 8), (20 + tw, 18 + th),
                (0, 0, 0), -1,
            )
            cv2.rectangle(
                frame, (8, 8), (20 + tw, 18 + th),
                color, 1,
            )
            cv2.putText(
                frame, label, (14, 14 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

    def invalidate_cache(self) -> None:
        """Force recomposite on next render."""
        self._composite_cache = None

    @staticmethod
    def _draw_grid(frame: np.ndarray, spacing: int = 40) -> None:
        """Draw subtle grid lines on whiteboard background."""
        h, w = frame.shape[:2]
        grid_color = (230, 230, 230)
        for x in range(0, w, spacing):
            cv2.line(frame, (x, 0), (x, h), grid_color, 1, cv2.LINE_AA)
        for y in range(0, h, spacing):
            cv2.line(frame, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)
