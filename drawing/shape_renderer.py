"""
Shape drawing utilities for geometric tools.

Renders lines, rectangles, circles, and arrows onto BGRA layers
with anti-aliased OpenCV primitives. Provides on-frame preview helpers.
"""

from __future__ import annotations

import cv2
import numpy as np

from canvas.layer import Layer
from tracking.tracking_result import Point2D


class ShapeRenderer:
    """Draws geometric shapes between two anchor points."""

    @staticmethod
    def _clamp_point(x: int, y: int, w: int, h: int) -> tuple[int, int]:
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    @staticmethod
    def _dirty_rect(
        x1: int, y1: int, x2: int, y2: int, thickness: int, w: int, h: int,
    ) -> tuple[int, int, int, int]:
        pad = thickness + 4
        rx = max(0, min(x1, x2) - pad)
        ry = max(0, min(y1, y2) - pad)
        x2p = min(w, max(x1, x2) + pad)
        y2p = min(h, max(y1, y2) + pad)
        return rx, ry, min(w - rx, x2p - rx), min(h - ry, y2p - ry)

    @staticmethod
    def draw_shape(
        layer: Layer,
        tool: str,
        start: Point2D,
        end: Point2D,
        color: tuple[int, int, int],
        size: int,
        opacity: float = 1.0,
        filled: bool = False,
    ) -> None:
        """Draw a shape from start to end on the BGRA layer."""
        if layer.locked:
            return

        thickness = max(1, size // 2)
        alpha = int(max(0.0, min(1.0, opacity)) * 255)
        bgra = (int(color[0]), int(color[1]), int(color[2]), alpha)

        h, w = layer.data.shape[:2]
        x1, y1 = ShapeRenderer._clamp_point(*start.as_int_tuple(), w, h)
        x2, y2 = ShapeRenderer._clamp_point(*end.as_int_tuple(), w, h)

        tool_lower = tool.lower()

        if tool_lower == "line":
            cv2.line(layer.data, (x1, y1), (x2, y2), bgra, thickness, cv2.LINE_AA)
        elif tool_lower == "rectangle":
            fill = -1 if filled else thickness
            cv2.rectangle(layer.data, (x1, y1), (x2, y2), bgra, fill, cv2.LINE_AA)
        elif tool_lower == "circle":
            radius = max(int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5), 1)
            fill = -1 if filled else thickness
            cv2.circle(layer.data, (x1, y1), radius, bgra, fill, cv2.LINE_AA)
        elif tool_lower == "arrow":
            cv2.arrowedLine(
                layer.data, (x1, y1), (x2, y2), bgra, thickness,
                tipLength=0.2, line_type=cv2.LINE_AA,
            )
        else:
            cv2.line(layer.data, (x1, y1), (x2, y2), bgra, thickness, cv2.LINE_AA)

        layer.mark_dirty(ShapeRenderer._dirty_rect(x1, y1, x2, y2, thickness, w, h))

    @staticmethod
    def draw_frame_preview(
        frame: np.ndarray,
        tool: str,
        start: Point2D,
        end: Point2D,
        color: tuple[int, int, int] = (255, 200, 50),
    ) -> None:
        """
        Draw a dashed shape guide on the live camera frame (HUD helper).

        Does not modify the canvas layer — preview only.
        """
        h, w = frame.shape[:2]
        x1, y1 = ShapeRenderer._clamp_point(*start.as_int_tuple(), w, h)
        x2, y2 = ShapeRenderer._clamp_point(*end.as_int_tuple(), w, h)
        guide = (int(color[0]), int(color[1]), int(color[2]))
        white = (255, 255, 255)

        tool_lower = tool.lower()
        if tool_lower == "line":
            ShapeRenderer._dashed_line(frame, (x1, y1), (x2, y2), guide)
        elif tool_lower == "arrow":
            ShapeRenderer._dashed_line(frame, (x1, y1), (x2, y2), guide)
            cv2.arrowedLine(
                frame, (x1, y1), (x2, y2), guide, 2,
                tipLength=0.15, line_type=cv2.LINE_AA,
            )
        elif tool_lower == "rectangle":
            ShapeRenderer._dashed_rect(frame, (x1, y1), (x2, y2), guide)
        elif tool_lower == "circle":
            radius = max(int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5), 1)
            cv2.circle(frame, (x1, y1), radius, white, 1, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), radius, guide, 1, cv2.LINE_AA)
        else:
            ShapeRenderer._dashed_line(frame, (x1, y1), (x2, y2), guide)

        cv2.circle(frame, (x1, y1), 5, white, 1, cv2.LINE_AA)
        cv2.circle(frame, (x1, y1), 4, guide, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2, y2), 5, white, 1, cv2.LINE_AA)
        cv2.circle(frame, (x2, y2), 4, guide, -1, cv2.LINE_AA)

    @staticmethod
    def _dashed_line(
        frame: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        dash_len: int = 10,
        gap_len: int = 6,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dist = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        step = dash_len + gap_len
        pos = 0.0
        while pos < dist:
            end_pos = min(pos + dash_len, dist)
            sx, sy = int(x1 + dx * pos), int(y1 + dy * pos)
            ex, ey = int(x1 + dx * end_pos), int(y1 + dy * end_pos)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2, cv2.LINE_AA)
            pos += step

    @staticmethod
    def _dashed_rect(
        frame: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        for i in range(4):
            ShapeRenderer._dashed_line(frame, corners[i], corners[i + 1], color)
