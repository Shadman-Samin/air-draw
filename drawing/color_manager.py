"""
Color manager — palette, recent colors, and favorites.

Manages the current drawing color and provides preset color
palettes for quick selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Preset color palette (BGR format for OpenCV)
PRESET_COLORS_BGR: list[tuple[int, int, int]] = [
    # Row 1: Primary & bright
    (50, 50, 255),     # Red
    (0, 100, 255),     # Orange
    (0, 200, 255),     # Yellow
    (0, 200, 0),       # Green
    (255, 200, 0),     # Cyan
    (255, 100, 0),     # Blue
    (255, 0, 100),     # Purple
    (200, 0, 200),     # Magenta

    # Row 2: Pastel & muted
    (150, 150, 255),   # Light red
    (100, 180, 255),   # Light orange
    (100, 220, 255),   # Light yellow
    (100, 220, 100),   # Light green
    (255, 220, 100),   # Light cyan
    (255, 180, 100),   # Light blue
    (255, 100, 180),   # Light purple
    (220, 100, 220),   # Light magenta

    # Row 3: Grayscale
    (255, 255, 255),   # White
    (200, 200, 200),   # Light gray
    (150, 150, 150),   # Medium gray
    (100, 100, 100),   # Dark gray
    (50, 50, 50),      # Charcoal
    (0, 0, 0),         # Black

    # Row 4: Earth tones
    (30, 70, 140),     # Brown
    (50, 120, 180),    # Tan
]


class ColorManager:
    """
    Manages drawing colors: current color, palette, recent colors, favorites.
    """

    MAX_RECENT = 12
    MAX_FAVORITES = 16

    def __init__(self):
        self._current_color: tuple[int, int, int] = (50, 50, 255)  # Default red
        self._recent_colors: list[tuple[int, int, int]] = []
        self._favorite_colors: list[tuple[int, int, int]] = []
        self._palette = list(PRESET_COLORS_BGR)

    @property
    def current_color(self) -> tuple[int, int, int]:
        """Current drawing color (BGR)."""
        return self._current_color

    @current_color.setter
    def current_color(self, color: tuple[int, int, int]) -> None:
        """Set current color and add to recent."""
        old = self._current_color
        self._current_color = color
        if old != color:
            self._add_to_recent(old)

    @property
    def palette(self) -> list[tuple[int, int, int]]:
        """Preset color palette."""
        return self._palette

    @property
    def recent_colors(self) -> list[tuple[int, int, int]]:
        """Recently used colors (most recent first)."""
        return self._recent_colors

    @property
    def favorite_colors(self) -> list[tuple[int, int, int]]:
        """User-favorited colors."""
        return self._favorite_colors

    def _add_to_recent(self, color: tuple[int, int, int]) -> None:
        """Add a color to the recent list (deduplicating)."""
        if color in self._recent_colors:
            self._recent_colors.remove(color)
        self._recent_colors.insert(0, color)
        if len(self._recent_colors) > self.MAX_RECENT:
            self._recent_colors.pop()

    def add_favorite(self, color: tuple[int, int, int]) -> None:
        """Add a color to favorites."""
        if color not in self._favorite_colors:
            self._favorite_colors.append(color)
            if len(self._favorite_colors) > self.MAX_FAVORITES:
                self._favorite_colors.pop(0)

    def remove_favorite(self, color: tuple[int, int, int]) -> None:
        """Remove a color from favorites."""
        if color in self._favorite_colors:
            self._favorite_colors.remove(color)

    def cycle_next(self) -> tuple[int, int, int]:
        """Cycle to the next color in the palette."""
        try:
            idx = self._palette.index(self._current_color)
            idx = (idx + 1) % len(self._palette)
        except ValueError:
            idx = 0
        self.current_color = self._palette[idx]
        return self._current_color

    def cycle_prev(self) -> tuple[int, int, int]:
        """Cycle to the previous color in the palette."""
        try:
            idx = self._palette.index(self._current_color)
            idx = (idx - 1) % len(self._palette)
        except ValueError:
            idx = 0
        self.current_color = self._palette[idx]
        return self._current_color

    @staticmethod
    def bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Convert BGR to RGB."""
        return (color[2], color[1], color[0])

    @staticmethod
    def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Convert RGB to BGR."""
        return (color[2], color[1], color[0])

    @staticmethod
    def bgr_to_hex(color: tuple[int, int, int]) -> str:
        """Convert BGR to hex string (#RRGGBB)."""
        return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

    @staticmethod
    def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
        """Convert hex string (#RRGGBB) to BGR."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
