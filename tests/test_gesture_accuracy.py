import unittest

from gestures.hand_state import HandState
from tracking.tracking_result import HandLandmarks, Point2D


class TestGestureAccuracy(unittest.TestCase):
    def setUp(self):
        self.hand_state = HandState()

    def _create_hand(self, extended_fingers: dict[str, bool]) -> HandLandmarks:
        lm = [Point2D(0, 0) for _ in range(21)]

        lm[0] = Point2D(50.0, 100.0)
        lm[5] = Point2D(40.0, 50.0)
        lm[9] = Point2D(47.0, 48.0)
        lm[13] = Point2D(53.0, 49.0)
        lm[17] = Point2D(60.0, 52.0)

        lm[6] = Point2D(40.0, 35.0)
        lm[7] = Point2D(40.0, 25.0)
        lm[8] = Point2D(40.0, 15.0) if extended_fingers.get("index", True) else Point2D(40.0, 60.0)

        lm[10] = Point2D(47.0, 32.0)
        lm[11] = Point2D(47.0, 22.0)
        lm[12] = Point2D(47.0, 12.0) if extended_fingers.get("middle", True) else Point2D(47.0, 58.0)

        lm[14] = Point2D(53.0, 33.0)
        lm[15] = Point2D(53.0, 23.0)
        lm[16] = Point2D(53.0, 13.0) if extended_fingers.get("ring", True) else Point2D(53.0, 59.0)

        lm[18] = Point2D(60.0, 38.0)
        lm[19] = Point2D(60.0, 28.0)
        lm[20] = Point2D(60.0, 18.0) if extended_fingers.get("pinky", True) else Point2D(60.0, 62.0)

        lm[1] = Point2D(47.0, 85.0)
        lm[2] = Point2D(42.0, 75.0)
        lm[3] = Point2D(35.0, 70.0)
        if extended_fingers.get("thumb", True):
            lm[4] = Point2D(40.0, 68.0)  # tip.x > ip.x for Right hand
        else:
            lm[4] = Point2D(28.0, 72.0)  # tip.x < ip.x when folded

        return HandLandmarks(landmarks=lm, handedness="Right", confidence=0.9)

    def test_finger_extensions_upright(self):
        hand_open = self._create_hand({
            "thumb": True, "index": True, "middle": True, "ring": True, "pinky": True,
        })
        states = self.hand_state.get_finger_states(hand_open)
        self.assertTrue(states.thumb)
        self.assertTrue(states.index)
        self.assertTrue(states.middle)
        self.assertTrue(states.ring)
        self.assertTrue(states.pinky)

        hand_index = self._create_hand({
            "thumb": False, "index": True, "middle": False, "ring": False, "pinky": False,
        })
        states = self.hand_state.get_finger_states(hand_index)
        self.assertFalse(states.thumb)
        self.assertTrue(states.index)
        self.assertFalse(states.middle)
        self.assertFalse(states.ring)
        self.assertFalse(states.pinky)


if __name__ == "__main__":
    unittest.main()
