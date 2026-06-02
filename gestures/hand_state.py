"""
Hand state analysis for gesture detection.

Analyzes MediaPipe hand landmarks to determine which fingers
are extended or curled, and detects pinch gestures between
specific finger pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from tracking.tracking_result import HandLandmarks, Point2D


@dataclass(slots=True)
class FingerStates:
    """Boolean state for each finger (True = extended)."""
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    @property
    def count(self) -> int:
        """Number of extended fingers."""
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


class HandState:
    """
    Analyzes hand landmarks to determine finger extension states
    and detect pinch gestures.

    Finger detection strategy:
    - For index, middle, ring, pinky: tip.y < pip.y means extended
      (in screen coords where Y increases downward)
    - For thumb: uses X-axis comparison, handedness-aware
    """

    def get_finger_states(
        self,
        hand: HandLandmarks,
    ) -> FingerStates:
        """
        Determine which fingers are extended.

        Args:
            hand: Hand landmarks with at least 21 points.

        Returns:
            FingerStates with boolean for each finger.
        """
        if len(hand.landmarks) < 21:
            return FingerStates()

        lm = hand.landmarks

        # ── Thumb ──
        # Compare thumb tip X vs thumb IP X
        # For right hand: tip.x > ip.x means extended
        # For left hand: tip.x < ip.x means extended
        # (frame is already mirrored, so MediaPipe "Right" = user's right)
        thumb_tip = lm[HandLandmarks.THUMB_TIP]
        thumb_ip = lm[HandLandmarks.THUMB_IP]
        thumb_mcp = lm[HandLandmarks.THUMB_MCP]

        if hand.handedness == "Right":
            thumb_extended = thumb_tip.x > thumb_ip.x
        else:
            thumb_extended = thumb_tip.x < thumb_ip.x

        # ── Index finger ──
        index_tip = lm[HandLandmarks.INDEX_TIP]
        index_pip = lm[HandLandmarks.INDEX_PIP]
        index_extended = index_tip.y < index_pip.y

        # ── Middle finger ──
        middle_tip = lm[HandLandmarks.MIDDLE_TIP]
        middle_pip = lm[HandLandmarks.MIDDLE_PIP]
        middle_extended = middle_tip.y < middle_pip.y

        # ── Ring finger ──
        ring_tip = lm[HandLandmarks.RING_TIP]
        ring_pip = lm[HandLandmarks.RING_PIP]
        ring_extended = ring_tip.y < ring_pip.y

        # ── Pinky ──
        pinky_tip = lm[HandLandmarks.PINKY_TIP]
        pinky_pip = lm[HandLandmarks.PINKY_PIP]
        pinky_extended = pinky_tip.y < pinky_pip.y

        return FingerStates(
            thumb=thumb_extended,
            index=index_extended,
            middle=middle_extended,
            ring=ring_extended,
            pinky=pinky_extended,
        )

    def get_pinch_distance(
        self,
        hand: HandLandmarks,
        finger_a_tip: int,
        finger_b_tip: int,
    ) -> float:
        """
        Calculate normalized distance between two fingertips.

        Normalizes by the hand size (wrist to middle MCP distance)
        so pinch detection is scale-invariant.

        Args:
            hand: Hand landmarks.
            finger_a_tip: Landmark index of first fingertip.
            finger_b_tip: Landmark index of second fingertip.

        Returns:
            Normalized distance (0 = touching, ~1 = far apart).
        """
        if len(hand.landmarks) < 21:
            return 1.0

        tip_a = hand.landmarks[finger_a_tip]
        tip_b = hand.landmarks[finger_b_tip]

        # Normalize by hand size for scale invariance
        wrist = hand.landmarks[HandLandmarks.WRIST]
        middle_mcp = hand.landmarks[HandLandmarks.MIDDLE_MCP]
        hand_size = wrist.distance_to(middle_mcp)

        if hand_size < 1.0:
            return 1.0

        return tip_a.distance_to(tip_b) / hand_size

    def get_thumb_index_pinch(self, hand: HandLandmarks) -> float:
        """Normalized pinch distance between thumb and index."""
        return self.get_pinch_distance(
            hand,
            HandLandmarks.THUMB_TIP,
            HandLandmarks.INDEX_TIP,
        )

    def get_thumb_middle_pinch(self, hand: HandLandmarks) -> float:
        """Normalized pinch distance between thumb and middle finger."""
        return self.get_pinch_distance(
            hand,
            HandLandmarks.THUMB_TIP,
            HandLandmarks.MIDDLE_TIP,
        )
