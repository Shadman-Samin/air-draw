import unittest
import math
from tracking.tracking_result import Point2D, HandLandmarks
from gestures.hand_state import HandState


class TestGestureAccuracy(unittest.TestCase):
    def setUp(self):
        self.hand_state = HandState()

    def _rotate_point(self, pt: Point2D, angle_rad: float, center: Point2D) -> Point2D:
        # Rotate point pt around center
        dx = pt.x - center.x
        dy = pt.y - center.y
        rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        return Point2D(center.x + rx, center.y + ry)

    def _create_hand(self, extended_fingers: dict[str, bool]) -> HandLandmarks:
        # Create a standard upright right hand landmarks
        lm = [Point2D(0, 0) for _ in range(21)]
        
        # Wrist
        lm[0] = Point2D(50.0, 100.0)
        
        # Pinky MCP (17), Index MCP (5) to define lateral direction
        # Right hand: Pinky is on the right, Index is on the left
        lm[5] = Point2D(40.0, 50.0)    # Index MCP
        lm[9] = Point2D(47.0, 48.0)    # Middle MCP
        lm[13] = Point2D(53.0, 49.0)   # Ring MCP
        lm[17] = Point2D(60.0, 52.0)   # Pinky MCP
        
        # Index finger
        lm[6] = Point2D(40.0, 35.0)    # PIP
        lm[7] = Point2D(40.0, 25.0)    # DIP
        if extended_fingers.get("index", True):
            lm[8] = Point2D(40.0, 15.0) # TIP (extended)
        else:
            lm[8] = Point2D(40.0, 60.0) # TIP (folded)

        # Middle finger
        lm[10] = Point2D(47.0, 32.0)
        lm[11] = Point2D(47.0, 22.0)
        if extended_fingers.get("middle", True):
            lm[12] = Point2D(47.0, 12.0)
        else:
            lm[12] = Point2D(47.0, 58.0)

        # Ring finger
        lm[14] = Point2D(53.0, 33.0)
        lm[15] = Point2D(53.0, 23.0)
        if extended_fingers.get("ring", True):
            lm[16] = Point2D(53.0, 13.0)
        else:
            lm[16] = Point2D(53.0, 59.0)

        # Pinky
        lm[18] = Point2D(60.0, 38.0)
        lm[19] = Point2D(60.0, 28.0)
        if extended_fingers.get("pinky", True):
            lm[20] = Point2D(60.0, 18.0)
        else:
            lm[20] = Point2D(60.0, 62.0)

        # Thumb
        # Thumb MCP (2), IP (3), TIP (4)
        lm[1] = Point2D(47.0, 85.0)
        lm[2] = Point2D(42.0, 75.0)    # MCP
        lm[3] = Point2D(35.0, 70.0)    # IP
        if extended_fingers.get("thumb", True):
            lm[4] = Point2D(25.0, 68.0) # TIP (extended leftwards)
        else:
            lm[4] = Point2D(48.0, 72.0) # TIP (folded rightwards)

        return HandLandmarks(landmarks=lm, handedness="Right", confidence=0.9)

    def test_finger_extensions_upright(self):
        # 1. All extended
        hand_open = self._create_hand({"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True})
        states = self.hand_state.get_finger_states(hand_open)
        self.assertTrue(states.thumb)
        self.assertTrue(states.index)
        self.assertTrue(states.middle)
        self.assertTrue(states.ring)
        self.assertTrue(states.pinky)

        # 2. Only Index extended (draw gesture)
        hand_index = self._create_hand({"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False})
        states = self.hand_state.get_finger_states(hand_index)
        self.assertFalse(states.thumb)
        self.assertTrue(states.index)
        self.assertFalse(states.middle)
        self.assertFalse(states.ring)
        self.assertFalse(states.pinky)

    def test_rotation_invariance(self):
        # Test rotating the hand by various angles preserves correct finger states
        test_angles = [math.pi / 4, math.pi / 2, math.pi, -math.pi / 2]
        
        for angle in test_angles:
            # Create a hand with only Index extended
            hand = self._create_hand({"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False})
            
            # Rotate all landmarks around the wrist (landmark 0)
            center = hand.landmarks[0]
            rotated_landmarks = [self._rotate_point(pt, angle, center) for pt in hand.landmarks]
            rotated_hand = HandLandmarks(landmarks=rotated_landmarks, handedness="Right", confidence=0.9)
            
            states = self.hand_state.get_finger_states(rotated_hand)
            with self.subTest(angle=angle):
                self.assertFalse(states.thumb, f"Thumb should be folded at angle {angle}")
                self.assertTrue(states.index, f"Index should be extended at angle {angle}")
                self.assertFalse(states.middle, f"Middle should be folded at angle {angle}")
                self.assertFalse(states.ring, f"Ring should be folded at angle {angle}")
                self.assertFalse(states.pinky, f"Pinky should be folded at angle {angle}")


if __name__ == "__main__":
    unittest.main()
