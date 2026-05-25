"""
MediaPipe Tasks-based hand tracker.

Uses mediapipe.tasks.python.vision.HandLandmarker for real-time hand landmark detection.
Extracts 21 landmarks per hand, handedness, and confidence scores.
"""

from __future__ import annotations

import logging

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from tracking.base_tracker import BaseTracker
from tracking.tracking_result import (
    HandLandmarks,
    NormalizedPoint,
    Point2D,
    TrackingResult,
)

logger = logging.getLogger(__name__)


class HandTracker(BaseTracker):
    """
    Real-time hand tracking using MediaPipe Tasks HandLandmarker.

    Detects up to 2 hands simultaneously, providing 21 landmarks
    per hand in pixel coordinates.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Args:
            max_hands: Maximum number of hands to detect (1 or 2).
            detection_confidence: Minimum confidence for initial detection.
            tracking_confidence: Minimum confidence for frame-to-frame tracking.
            model_complexity: 0 = lite, 1 = full. Full is more accurate.
        """
        super().__init__()
        self._max_hands = max_hands
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence
        self._model_complexity = model_complexity
        self._landmarker: vision.HandLandmarker | None = None

    def start(self) -> None:
        """Initialize the MediaPipe Hands detector."""
        super().start()
        
        # Configure HandLandmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        
        logger.info(
            "HandTracker started (max_hands=%d, det_conf=%.2f, track_conf=%.2f)",
            self._max_hands,
            self._detection_confidence,
            self._tracking_confidence,
        )

    def stop(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        super().stop()
        logger.info("HandTracker stopped")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> TrackingResult:
        """
        Process a BGR frame and detect hand landmarks.

        The frame should already be horizontally flipped (mirrored)
        for intuitive user interaction.

        Args:
            frame: BGR image from the camera.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            TrackingResult containing detected hand landmarks.
        """
        if self._landmarker is None:
            return TrackingResult(timestamp_ms=timestamp_ms)

        h, w = frame.shape[:2]

        # MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert OpenCV frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform synchronous detection for video running mode
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        tracking_result = TrackingResult(
            timestamp_ms=timestamp_ms,
            frame_width=w,
            frame_height=h,
        )

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness_list in zip(
                results.hand_landmarks,
                results.handedness,
            ):
                # Extract handedness and confidence
                classification = handedness_list[0]
                hand_label = classification.category_name  # "Left" or "Right"
                confidence = classification.score

                # Convert normalized landmarks to pixel coordinates
                pixel_landmarks: list[Point2D] = []
                for lm in hand_landmarks:
                    point = NormalizedPoint(x=lm.x, y=lm.y, z=lm.z)
                    pixel_landmarks.append(point.to_pixel(w, h))

                hand = HandLandmarks(
                    landmarks=pixel_landmarks,
                    handedness=hand_label,
                    confidence=confidence,
                )
                tracking_result.hands.append(hand)

        return tracking_result
