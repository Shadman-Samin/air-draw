"""
Gesture recognition engine using a state machine approach.

Maps finger states to gesture types and handles debouncing
for single-shot gestures (undo, redo, clear). Provides stable
gesture output by requiring consistent detection across frames.
"""

from __future__ import annotations

import logging
import time

from app.constants import GESTURE_DEBOUNCE_MS, GESTURE_HOLD_DURATION_MS, PINCH_DISTANCE_THRESHOLD
from gestures.gesture_types import GestureState, GestureType
from gestures.hand_state import FingerStates, HandState
from tracking.tracking_result import HandLandmarks, Point2D, TrackingResult

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """
    State machine-based gesture recognition.

    Takes tracking results and produces stable, debounced gesture states.
    Single-shot gestures (undo, redo) fire once and require the gesture
    to be released before firing again.
    """

    # Minimum frames of consistent gesture before accepting
    STABILITY_FRAMES = 3

    def __init__(self):
        self._hand_state = HandState()

        # State tracking
        self._current_gesture = GestureType.NONE
        self._gesture_frame_count = 0
        self._stable_gesture = GestureType.NONE

        # Single-shot debouncing
        self._single_shot_fired: dict[GestureType, bool] = {
            GestureType.UNDO: False,
            GestureType.REDO: False,
            GestureType.CLEAR: False,
        }

        # Palm hold timing for clear gesture
        self._palm_start_time: float = 0.0

    def update(self, tracking_result: TrackingResult) -> GestureState:
        """
        Process tracking result and return current gesture state.

        Args:
            tracking_result: Latest tracking data from the tracker.

        Returns:
            GestureState with recognized gesture and metadata.
        """
        if not tracking_result.has_hands:
            self._reset_state()
            return GestureState(
                gesture=GestureType.NONE,
                hand_detected=False,
            )

        hand = tracking_result.primary_hand
        if hand is None or len(hand.landmarks) < 21:
            self._reset_state()
            return GestureState(gesture=GestureType.NONE, hand_detected=False)

        # Analyze finger states
        finger_states = self._hand_state.get_finger_states(hand)
        raw_gesture = self._classify_gesture(hand, finger_states)

        # Stabilize: require consistent gesture for N frames
        if raw_gesture == self._current_gesture:
            self._gesture_frame_count += 1
        else:
            self._current_gesture = raw_gesture
            self._gesture_frame_count = 1

        if self._gesture_frame_count >= self.STABILITY_FRAMES:
            self._stable_gesture = self._current_gesture

        # Handle single-shot gestures
        output_gesture = self._handle_debounce(self._stable_gesture)

        # Get fingertip for drawing
        fingertip = hand.index_tip

        # Pinch distance for size/color controls
        pinch_dist = 0.0
        if output_gesture in (GestureType.PINCH_COLOR, GestureType.PINCH_SIZE):
            pinch_dist = self._hand_state.get_thumb_index_pinch(hand)

        return GestureState(
            gesture=output_gesture,
            fingertip=fingertip,
            hand_detected=True,
            confidence=hand.confidence,
            finger_count=finger_states.count,
            pinch_distance=pinch_dist,
        )

    def _classify_gesture(
        self,
        hand: HandLandmarks,
        fingers: FingerStates,
    ) -> GestureType:
        """
        Map finger states to a gesture type.

        Priority order handles ambiguous finger combinations.
        """
        thumb, index, middle, ring, pinky = fingers.as_tuple()
        count = fingers.count

        # ── Closed fist: no fingers extended ──
        if count == 0:
            return GestureType.ERASE

        # ── Single index finger: draw ──
        if index and not middle and not ring and not pinky:
            # Check for thumb+index pinch
            pinch_dist = self._hand_state.get_thumb_index_pinch(hand)
            if thumb and pinch_dist < PINCH_DISTANCE_THRESHOLD:
                return GestureType.PINCH_COLOR
            return GestureType.DRAW

        # ── Index + middle (peace sign): cursor mode ──
        if index and middle and not ring and not pinky:
            # Check for thumb+middle pinch
            if thumb:
                pinch_dist = self._hand_state.get_thumb_middle_pinch(hand)
                if pinch_dist < PINCH_DISTANCE_THRESHOLD:
                    return GestureType.PINCH_SIZE
            return GestureType.CURSOR

        # ── Three fingers (index + middle + ring): undo ──
        if index and middle and ring and not pinky:
            return GestureType.UNDO

        # ── Four fingers (index + middle + ring + pinky): redo ──
        if index and middle and ring and pinky and not thumb:
            return GestureType.REDO

        # ── All five fingers (open palm): pause / clear ──
        if count == 5:
            return self._handle_palm_gesture()

        return GestureType.CURSOR

    def _handle_palm_gesture(self) -> GestureType:
        """
        Distinguish between brief palm (pause) and sustained palm (clear).

        A brief open palm pauses drawing. Holding the palm for
        GESTURE_HOLD_DURATION_MS triggers a canvas clear.
        """
        now = time.monotonic()

        if self._palm_start_time == 0.0:
            self._palm_start_time = now

        elapsed_ms = (now - self._palm_start_time) * 1000

        if elapsed_ms >= GESTURE_HOLD_DURATION_MS:
            return GestureType.CLEAR
        return GestureType.PAUSE

    def _handle_debounce(self, gesture: GestureType) -> GestureType:
        """
        Handle single-shot gesture debouncing.

        Single-shot gestures (undo, redo, clear) fire once and then
        convert to CURSOR until the gesture is released.
        """
        # Reset palm timer if not palm gesture
        if gesture not in (GestureType.PAUSE, GestureType.CLEAR):
            self._palm_start_time = 0.0

        # Check if this is a single-shot gesture
        if gesture in self._single_shot_fired:
            if self._single_shot_fired[gesture]:
                # Already fired — suppress until released
                return GestureType.CURSOR
            else:
                # Fire once
                self._single_shot_fired[gesture] = True
                return gesture
        else:
            # Reset all single-shot flags when a non-single-shot gesture is active
            for key in self._single_shot_fired:
                self._single_shot_fired[key] = False
            return gesture

    def _reset_state(self) -> None:
        """Reset all gesture state when hand is lost."""
        self._current_gesture = GestureType.NONE
        self._gesture_frame_count = 0
        self._stable_gesture = GestureType.NONE
        self._palm_start_time = 0.0
        for key in self._single_shot_fired:
            self._single_shot_fired[key] = False
