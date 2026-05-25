"""
Kalman Filter for 2D point smoothing.

Uses OpenCV's KalmanFilter with a constant-velocity model:
  State:       [x, y, dx, dy]
  Measurement: [x, y]

Provides optimal noise reduction while preserving responsiveness
for real-time hand tracking applications.
"""

from __future__ import annotations

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from tracking.tracking_result import Point2D


class KalmanFilter2D(BaseFilter):
    """
    2D Kalman filter for smoothing tracked point positions.

    Uses a constant-velocity motion model. Tunable via process noise
    (how much the filter trusts the model) and measurement noise
    (how much the filter trusts the raw input).

    Lower process noise → smoother but more laggy.
    Higher measurement noise → smoother but more laggy.
    """

    def __init__(
        self,
        process_noise: float = 0.03,
        measurement_noise: float = 1.0,
    ):
        """
        Args:
            process_noise: Process noise covariance scalar.
                Controls how much the filter follows raw input.
            measurement_noise: Measurement noise covariance scalar.
                Controls how much to trust raw measurements.
        """
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._kf: cv2.KalmanFilter | None = None
        self._initialized = False

    def _init_filter(self, initial_point: Point2D) -> None:
        """Initialize the Kalman filter with the first measurement."""
        # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
        kf = cv2.KalmanFilter(4, 2)

        # Measurement matrix: we observe [x, y]
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # State transition matrix: constant velocity model
        # x' = x + vx, y' = y + vy, vx' = vx, vy' = vy
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Process noise covariance
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self._process_noise

        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self._measurement_noise

        # Initialize state with the first measurement
        kf.statePre = np.array(
            [[initial_point.x], [initial_point.y], [0], [0]],
            dtype=np.float32,
        )
        kf.statePost = kf.statePre.copy()

        # Error covariance
        kf.errorCovPre = np.eye(4, dtype=np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        self._kf = kf
        self._initialized = True

    def process(self, point: Point2D) -> Point2D:
        """
        Filter a raw tracked point through the Kalman filter.

        On the first call, initializes the filter with the given point.
        On subsequent calls, predicts and corrects using the measurement.
        """
        if not self._initialized:
            self._init_filter(point)
            return point

        assert self._kf is not None

        # Predict next state
        self._kf.predict()

        # Correct with measurement
        measurement = np.array(
            [[np.float32(point.x)], [np.float32(point.y)]],
        )
        estimated = self._kf.correct(measurement)

        return Point2D(
            x=float(estimated[0, 0]),
            y=float(estimated[1, 0]),
        )

    def reset(self) -> None:
        """Reset the Kalman filter state."""
        self._kf = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def set_noise(
        self,
        process_noise: float | None = None,
        measurement_noise: float | None = None,
    ) -> None:
        """
        Update noise parameters at runtime.

        Takes effect immediately on the active filter instance.
        """
        if process_noise is not None:
            self._process_noise = process_noise
            if self._kf is not None:
                self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        if measurement_noise is not None:
            self._measurement_noise = measurement_noise
            if self._kf is not None:
                self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
