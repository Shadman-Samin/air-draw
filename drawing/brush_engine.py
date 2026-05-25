"""
Brush engine — stamp-based rendering for professional drawing quality.

Instead of using cv2.line() (which produces hard, aliased lines),
the brush engine interpolates points along the stroke path and
stamps brush tips at each position with proper alpha compositing.

This approach enables:
- Anti-aliased drawing
- Variable opacity
- Soft edges
- Pressure-sensitive width
- Different brush textures
"""

from __future__ import annotations

import cv2
import numpy as np

from canvas.layer import Layer
from tracking.tracking_result import Point2D


class BrushEngine:
    """
    Stamp-based brush rendering engine.

    Renders stroke segments by interpolating between two points
    and stamping a circular brush tip at regular intervals.
    Supports anti-aliased rendering via cv2.LINE_AA.
    """

    # Stamp spacing as fraction of brush size (lower = smoother, higher = faster)
    STAMP_SPACING = 0.25

    # Minimum distance between stamps in pixels
    MIN_STAMP_DISTANCE = 1.0

    def draw_segment(
        self,
        layer: Layer,
        p1: Point2D,
        p2: Point2D,
        color: tuple[int, int, int],
        size: int,
        opacity: float = 1.0,
    ) -> None:
        """
        Draw a stroke segment between two points on the given layer.

        Interpolates intermediate points and stamps the brush tip
        at each position.

        Args:
            layer: Target BGRA layer.
            p1: Start point.
            p2: End point.
            color: BGR color tuple.
            size: Brush diameter in pixels.
            opacity: Brush opacity [0.0, 1.0].
        """
        if layer.locked:
            return

        # Compute spacing between stamps
        spacing = max(self.MIN_STAMP_DISTANCE, size * self.STAMP_SPACING)
        distance = p1.distance_to(p2)

        if distance < 0.5:
            # Points are essentially the same — stamp once
            self._stamp(layer, p2, color, size, opacity)
            return

        # Number of stamps needed
        num_stamps = max(1, int(distance / spacing))

        for i in range(num_stamps + 1):
            t = i / max(num_stamps, 1)
            x = p1.x + (p2.x - p1.x) * t
            y = p1.y + (p2.y - p1.y) * t
            self._stamp(layer, Point2D(x, y), color, size, opacity)

    def _stamp(
        self,
        layer: Layer,
        center: Point2D,
        color: tuple[int, int, int],
        size: int,
        opacity: float,
    ) -> None:
        """
        Stamp a single brush tip onto the layer.

        Uses cv2.circle with LINE_AA for anti-aliased rendering.
        Handles alpha compositing for opacity < 1.0.
        """
        cx, cy = center.as_int_tuple()
        radius = max(size // 2, 1)

        # Bounds check
        h, w = layer.data.shape[:2]
        if cx < -radius or cx >= w + radius or cy < -radius or cy >= h + radius:
            return

        # Compute affected region for dirty tracking
        x_min = max(0, cx - radius - 1)
        y_min = max(0, cy - radius - 1)
        x_max = min(w, cx + radius + 2)
        y_max = min(h, cy + radius + 2)

        if opacity >= 0.99:
            # Fast path: full opacity — draw directly on BGRA image
            bgra_color = (int(color[0]), int(color[1]), int(color[2]), 255)
            cv2.circle(
                layer.data,
                (cx, cy), radius, bgra_color, -1, cv2.LINE_AA,
            )
        else:
            # Semi-transparent: use a temporary buffer and blend
            # Create stamp mask only for the affected region (bounding box)
            h_roi = y_max - y_min
            w_roi = x_max - x_min
            if h_roi <= 0 or w_roi <= 0:
                return

            region_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
            cv2.circle(
                region_mask,
                (cx - x_min, cy - y_min),
                radius,
                255,
                -1,
                cv2.LINE_AA,
            )

            if not np.any(region_mask):
                return

            region = layer.data[y_min:y_max, x_min:x_max]
            mask_f = region_mask.astype(np.float32) / 255.0 * opacity

            # Blend color
            for c in range(3):
                region[:, :, c] = (
                    color[c] * mask_f + region[:, :, c] * (1.0 - mask_f)
                ).astype(np.uint8)

            # Blend alpha
            existing_alpha = region[:, :, 3].astype(np.float32) / 255.0
            new_alpha = mask_f + existing_alpha * (1.0 - mask_f)
            region[:, :, 3] = (new_alpha * 255).astype(np.uint8)

        layer.mark_dirty((x_min, y_min, x_max - x_min, y_max - y_min))

    def erase_at(
        self,
        layer: Layer,
        center: Point2D,
        radius: int = 20,
    ) -> None:
        """
        Erase at the given position by setting alpha to 0.

        Args:
            layer: Target BGRA layer.
            center: Erase center position.
            radius: Eraser radius in pixels.
        """
        if layer.locked:
            return

        cx, cy = center.as_int_tuple()
        h, w = layer.data.shape[:2]

        # Bounds check
        if cx < -radius or cx >= w + radius or cy < -radius or cy >= h + radius:
            return

        # Set pixels to transparent
        cv2.circle(layer.data, (cx, cy), radius, (0, 0, 0, 0), -1, cv2.LINE_AA)

        x_min = max(0, cx - radius - 1)
        y_min = max(0, cy - radius - 1)
        x_max = min(w, cx + radius + 2)
        y_max = min(h, cy + radius + 2)
        layer.mark_dirty((x_min, y_min, x_max - x_min, y_max - y_min))

    def draw_line(
        self,
        layer: Layer,
        p1: Point2D,
        p2: Point2D,
        color: tuple[int, int, int],
        size: int,
        opacity: float = 1.0,
    ) -> None:
        """
        Draw a straight line between two points.

        Uses stamp-based rendering for consistency with freehand strokes.
        """
        self.draw_segment(layer, p1, p2, color, size, opacity)
