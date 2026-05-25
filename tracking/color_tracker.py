"""
HSV color marker tracking implementation.

Identifies and tracks the centroid of a user-defined colored object/marker
in the frame. Returns a virtual HandLandmarks object to maintain compatibility
with the unified tracking pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracking.base_tracker import BaseTracker
from tracking.tracking_result import HandLandmarks, Point2D, TrackingResult


class ColorTracker(BaseTracker):
    """
    Tracks a specific color range in HSV space.

    Processes frames using range thresholding, morphological smoothing,
    and contour detection to find the largest matching object.
    Maps the detected centroid to a virtual index fingertip for unified pipeline compatibility.
    """

    def __init__(
        self,
        hsv_lower: tuple[int, int, int] = (160, 100, 100),
        hsv_upper: tuple[int, int, int] = (180, 255, 255),
        min_contour_area: float = 500.0,
        max_contour_area: float = 50000.0,
    ):
        super().__init__()
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area

        # Morphological operation kernel
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update_range(self, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> None:
        """Update active HSV thresholds."""
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> TrackingResult:
        """
        Processes BGR frame, thresholds in HSV space, and computes centroid of largest marker.
        """
        h, w = frame.shape[:2]
        result = TrackingResult(
            timestamp_ms=timestamp_ms,
            frame_width=w,
            frame_height=h,
        )

        if not self._is_running:
            return result

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only target colors
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Apply morphological operations to filter out small noises
        mask = cv2.erode(mask, self._morph_kernel, iterations=1)
        mask = cv2.dilate(mask, self._morph_kernel, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = None
        max_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

        if largest_contour is not None:
            # Get centroid using image moments
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroid = Point2D(x=cx, y=cy)

                # Construct a virtual HandLandmarks instance where
                # key tips (index, thumb, middle, wrist) are all at the centroid.
                # This guarantees compatibility with the drawing/gesture modules.
                landmarks = [centroid for _ in range(21)]
                virtual_hand = HandLandmarks(
                    landmarks=landmarks,
                    handedness="Right",
                    confidence=1.0,
                )
                result.hands.append(virtual_hand)

        return result
