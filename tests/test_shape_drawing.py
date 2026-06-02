import unittest
import numpy as np

from app.constants import DrawingTool
from canvas.canvas_manager import CanvasManager
from gestures.gesture_types import GestureState, GestureType
from tracking.tracking_result import Point2D


class TestShapeDrawing(unittest.TestCase):
    def test_shape_draw_and_commit(self):
        canvas = CanvasManager(640, 480)
        canvas.drawing_tool = DrawingTool.LINE
        layer = canvas.layer_stack.active_layer
        empty = layer.get_snapshot()

        draw = GestureState(gesture=GestureType.DRAW)
        canvas.handle_input(Point2D(50, 50), draw)
        canvas.handle_input(Point2D(150, 150), draw)
        canvas.handle_input(Point2D(150, 150), GestureState(gesture=GestureType.PAUSE))

        self.assertFalse(np.array_equal(empty, layer.get_snapshot()))


if __name__ == "__main__":
    unittest.main()
