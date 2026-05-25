"""
Layer stack management.

Manages an ordered collection of layers with operations for
adding, removing, reordering, and selecting the active layer.
"""

from __future__ import annotations

import logging
from typing import Optional

from canvas.layer import Layer

logger = logging.getLogger(__name__)


class LayerStack:
    """
    Ordered stack of canvas layers.

    Layers are ordered bottom-to-top: index 0 is the bottom layer,
    higher indices are drawn on top. The active layer receives
    drawing input.
    """

    def __init__(self, width: int, height: int):
        """
        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self._width = width
        self._height = height
        self._layers: list[Layer] = []
        self._active_index: int = -1

        # Create default layer
        self.add_layer("Layer 1")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def layers(self) -> list[Layer]:
        """All layers in bottom-to-top order."""
        return self._layers

    @property
    def active_layer(self) -> Optional[Layer]:
        """Currently active layer for drawing."""
        if 0 <= self._active_index < len(self._layers):
            return self._layers[self._active_index]
        return None

    @property
    def active_index(self) -> int:
        return self._active_index

    @active_index.setter
    def active_index(self, index: int) -> None:
        """Set the active layer by index."""
        if 0 <= index < len(self._layers):
            self._active_index = index
        else:
            logger.warning("Invalid layer index: %d (have %d layers)", index, len(self._layers))

    @property
    def count(self) -> int:
        return len(self._layers)

    @property
    def any_dirty(self) -> bool:
        """Whether any layer has been modified since last composite."""
        return any(layer.is_dirty for layer in self._layers)

    def add_layer(
        self,
        name: str | None = None,
        index: int | None = None,
    ) -> Layer:
        """
        Add a new transparent layer.

        Args:
            name: Layer name. Auto-generated if None.
            index: Position to insert. Appends to top if None.

        Returns:
            The newly created layer.
        """
        if name is None:
            name = f"Layer {len(self._layers) + 1}"

        layer = Layer(self._width, self._height, name=name)

        if index is not None:
            idx = max(0, min(index, len(self._layers)))
            self._layers.insert(idx, layer)
            # Adjust active index if inserting below
            if idx <= self._active_index:
                self._active_index += 1
        else:
            self._layers.append(layer)

        # Activate the new layer
        self._active_index = self._layers.index(layer)

        logger.debug("Added layer '%s' at index %d", name, self._active_index)
        return layer

    def remove_layer(self, index: int) -> Optional[Layer]:
        """
        Remove a layer by index.

        Cannot remove the last remaining layer.

        Returns:
            The removed layer, or None if removal was rejected.
        """
        if len(self._layers) <= 1:
            logger.warning("Cannot remove the last layer")
            return None

        if not (0 <= index < len(self._layers)):
            logger.warning("Invalid layer index: %d", index)
            return None

        removed = self._layers.pop(index)

        # Adjust active index
        if self._active_index >= len(self._layers):
            self._active_index = len(self._layers) - 1
        elif self._active_index > index:
            self._active_index -= 1

        logger.debug("Removed layer '%s'", removed.name)
        return removed

    def move_layer(self, from_index: int, to_index: int) -> bool:
        """
        Move a layer from one position to another.

        Returns:
            True if the move was successful.
        """
        if not (0 <= from_index < len(self._layers)):
            return False
        if not (0 <= to_index < len(self._layers)):
            return False
        if from_index == to_index:
            return True

        layer = self._layers.pop(from_index)
        self._layers.insert(to_index, layer)

        # Update active index to follow the active layer
        if self._active_index == from_index:
            self._active_index = to_index
        elif from_index < self._active_index <= to_index:
            self._active_index -= 1
        elif to_index <= self._active_index < from_index:
            self._active_index += 1

        return True

    def merge_down(self, index: int) -> bool:
        """
        Merge a layer into the one below it.

        Returns:
            True if the merge was successful.
        """
        if index <= 0 or index >= len(self._layers):
            return False

        top = self._layers[index]
        bottom = self._layers[index - 1]

        if bottom.locked:
            return False

        # Alpha composite top onto bottom
        self._alpha_composite_inplace(bottom.data, top.data, top.opacity)
        bottom.mark_dirty()

        # Remove the top layer
        self._layers.pop(index)
        if self._active_index >= index:
            self._active_index = max(0, self._active_index - 1)

        return True

    def clear_all(self) -> None:
        """Clear all layers to transparent."""
        for layer in self._layers:
            layer.clear()

    def flatten(self) -> Layer:
        """
        Flatten all visible layers into a single layer.

        Returns a new Layer with the composited result.
        Does not modify the existing stack.
        """
        result = Layer(self._width, self._height, name="Flattened")
        for layer in self._layers:
            if layer.visible:
                self._alpha_composite_inplace(
                    result.data, layer.data, layer.opacity,
                )
        return result

    @staticmethod
    def _alpha_composite_inplace(
        dst: 'np.ndarray',
        src: 'np.ndarray',
        src_opacity: float = 1.0,
    ) -> None:
        """
        Alpha-composite src onto dst in-place.

        Uses standard Porter-Duff "over" operation.
        """
        import numpy as np

        # Extract alpha channels as float [0, 1]
        src_alpha = src[:, :, 3:4].astype(np.float32) / 255.0 * src_opacity
        dst_alpha = dst[:, :, 3:4].astype(np.float32) / 255.0

        # Compute output alpha
        out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)

        # Avoid division by zero
        safe_alpha = np.maximum(out_alpha, 1e-6)

        # Composite RGB channels
        dst[:, :, :3] = (
            (src[:, :, :3].astype(np.float32) * src_alpha +
             dst[:, :, :3].astype(np.float32) * dst_alpha * (1.0 - src_alpha))
            / safe_alpha
        ).astype(np.uint8)

        # Set output alpha
        dst[:, :, 3:4] = (out_alpha * 255.0).astype(np.uint8)
