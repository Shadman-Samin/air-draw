"""
Single canvas layer backed by a BGRA NumPy array.

Each layer is an independent drawing surface with opacity,
visibility, and lock controls. Layers use BGRA format
(Blue, Green, Red, Alpha) for efficient OpenCV compositing.
"""

from __future__ import annotations

import numpy as np

from app.constants import BlendMode


class Layer:
    """
    A single drawing layer.

    The data array is BGRA uint8 with shape (height, width, 4).
    Alpha channel controls per-pixel opacity.
    """

    def __init__(
        self,
        width: int,
        height: int,
        name: str = "Layer",
        fill_color: tuple[int, int, int, int] | None = None,
    ):
        """
        Args:
            width: Layer width in pixels.
            height: Layer height in pixels.
            name: Human-readable layer name.
            fill_color: Optional BGRA fill color. Default is transparent.
        """
        self._width = width
        self._height = height
        self.name = name
        self.visible = True
        self.locked = False
        self.opacity = 1.0  # Layer-level opacity [0.0, 1.0]
        self.blend_mode = BlendMode.NORMAL

        # Initialize as transparent by default
        if fill_color is not None:
            self.data = np.full(
                (height, width, 4), fill_color, dtype=np.uint8,
            )
        else:
            self.data = np.zeros((height, width, 4), dtype=np.uint8)

        # Dirty region tracking for optimized compositing
        self._dirty = True
        self._dirty_rect: tuple[int, int, int, int] | None = None  # (x, y, w, h)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width)"""
        return (self._height, self._width)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def dirty_rect(self) -> tuple[int, int, int, int] | None:
        """Get the dirty rectangle tracking modifications (x, y, w, h)."""
        return self._dirty_rect

    def mark_dirty(
        self,
        rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        """
        Mark the layer as modified.

        Args:
            rect: Optional dirty rectangle (x, y, w, h).
                  If None, entire layer is marked dirty.
        """
        self._dirty = True
        if rect is not None:
            if self._dirty_rect is None:
                self._dirty_rect = rect
            else:
                # Expand dirty rect to encompass both
                x1 = min(self._dirty_rect[0], rect[0])
                y1 = min(self._dirty_rect[1], rect[1])
                x2 = max(
                    self._dirty_rect[0] + self._dirty_rect[2],
                    rect[0] + rect[2],
                )
                y2 = max(
                    self._dirty_rect[1] + self._dirty_rect[3],
                    rect[1] + rect[3],
                )
                self._dirty_rect = (x1, y1, x2 - x1, y2 - y1)
        else:
            self._dirty_rect = None

    def clear_dirty(self) -> None:
        """Mark the layer as clean after compositing."""
        self._dirty = False
        self._dirty_rect = None

    def clear(self) -> None:
        """Clear the layer to fully transparent."""
        self.data[:] = 0
        self.mark_dirty()

    def get_snapshot(self) -> np.ndarray:
        """Get a copy of the layer data for undo/redo."""
        return self.data.copy()

    def restore_snapshot(self, snapshot: np.ndarray) -> None:
        """Restore layer data from a snapshot."""
        np.copyto(self.data, snapshot)
        self.mark_dirty()

    def get_region(
        self,
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """Get a copy of a rectangular region."""
        x = max(0, x)
        y = max(0, y)
        w = min(w, self._width - x)
        h = min(h, self._height - y)
        return self.data[y:y+h, x:x+w].copy()

    def set_region(
        self,
        x: int, y: int,
        region: np.ndarray,
    ) -> None:
        """Paste a region back onto the layer."""
        rh, rw = region.shape[:2]
        x = max(0, x)
        y = max(0, y)
        rw = min(rw, self._width - x)
        rh = min(rh, self._height - y)
        self.data[y:y+rh, x:x+rw] = region[:rh, :rw]
        self.mark_dirty((x, y, rw, rh))

    def __repr__(self) -> str:
        state = []
        if not self.visible:
            state.append("hidden")
        if self.locked:
            state.append("locked")
        if self.opacity < 1.0:
            state.append(f"opacity={self.opacity:.0%}")
        extra = f" ({', '.join(state)})" if state else ""
        return f"Layer('{self.name}', {self._width}×{self._height}{extra})"
