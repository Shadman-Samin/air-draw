"""
Multi-hand input controller.

Assigns independent smoothing filters and gesture recognizers per hand
so two hands can draw or control the canvas simultaneously.
"""

from __future__ import annotations

from filters.filter_chain import FilterChain
from gestures.gesture_recognizer import GestureRecognizer
from gestures.gesture_types import GestureState, GestureType
from tracking.tracking_result import HandLandmarks, Point2D, TrackingResult


class MultiHandController:
    """
    Processes multiple detected hands with per-hand filter chains.

    Primary hand (index 0): full gesture set (draw, erase, undo, etc.).
    Secondary hand (index 1+): simplified — index up draws, fist erases.
    """

    MAX_HANDS = 2

    def __init__(
        self,
        kalman_process_noise: float = 0.02,
        kalman_measurement_noise: float = 1.2,
        ema_alpha: float = 0.4,
        deadzone_threshold: float = 2.0,
    ):
        self._kalman_process = kalman_process_noise
        self._kalman_measurement = kalman_measurement_noise
        self._ema_alpha = ema_alpha
        self._deadzone = deadzone_threshold
        self._filter_chains: list[FilterChain] = []
        self._gesture_recognizers: list[GestureRecognizer] = []
        self._ensure_capacity(self.MAX_HANDS)

    def _ensure_capacity(self, count: int) -> None:
        while len(self._filter_chains) < count:
            self._filter_chains.append(FilterChain(
                kalman_process_noise=self._kalman_process,
                kalman_measurement_noise=self._kalman_measurement,
                ema_alpha=self._ema_alpha,
                deadzone_threshold=self._deadzone,
            ))
            self._gesture_recognizers.append(GestureRecognizer())

    def reset_all(self) -> None:
        """Reset all per-hand filters when tracking is lost."""
        for chain in self._filter_chains:
            chain.reset()

    def process(
        self,
        tracking_result: TrackingResult,
        use_full_gestures: bool = True,
    ) -> list[tuple[int, Point2D | None, GestureState]]:
        """
        Process all hands and return smoothed points with gesture states.

        Returns:
            List of (hand_index, smoothed_point, gesture_state) tuples.
        """
        outputs: list[tuple[int, Point2D | None, GestureState]] = []

        if not tracking_result.has_hands:
            self.reset_all()
            return outputs

        hands = tracking_result.hands[: self.MAX_HANDS]
        self._ensure_capacity(len(hands))

        for idx, hand in enumerate(hands):
            raw_pt = hand.index_tip
            if raw_pt is None:
                self._filter_chains[idx].reset()
                outputs.append((idx, None, GestureState(gesture=GestureType.NONE)))
                continue

            smoothed = self._filter_chains[idx].process(raw_pt)

            if idx == 0 and use_full_gestures:
                single_result = TrackingResult(
                    hands=[hand],
                    timestamp_ms=tracking_result.timestamp_ms,
                    frame_width=tracking_result.frame_width,
                    frame_height=tracking_result.frame_height,
                )
                gesture_state = self._gesture_recognizers[idx].update(single_result)
            else:
                gesture_state = self._classify_secondary_hand(hand)

            outputs.append((idx, smoothed, gesture_state))

        # Reset unused chains
        for idx in range(len(hands), len(self._filter_chains)):
            self._filter_chains[idx].reset()

        return outputs

    def _classify_secondary_hand(self, hand: HandLandmarks) -> GestureState:
        """Simplified gestures for the non-primary hand."""
        from gestures.hand_state import HandState

        state = HandState()
        fingers = state.get_finger_states(hand)
        if fingers.count == 0:
            return GestureState(gesture=GestureType.ERASE, hand_detected=True)
        if fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky:
            return GestureState(gesture=GestureType.DRAW, hand_detected=True)
        return GestureState(gesture=GestureType.CURSOR, hand_detected=True)
