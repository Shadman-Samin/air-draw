"""
Composable filter chain for point smoothing.

Chains multiple smoothing filters in sequence:
  Raw Input → Kalman → EMA → Deadzone → Smooth Output

Each filter can be independently configured or replaced.
The chain resets all filters simultaneously when tracking is lost.
"""

from __future__ import annotations

from app.constants import (
    DEFAULT_DEADZONE_THRESHOLD,
    DEFAULT_EMA_ALPHA,
    DEFAULT_KALMAN_MEASUREMENT_NOISE,
    DEFAULT_KALMAN_PROCESS_NOISE,
)
from filters.base_filter import BaseFilter
from filters.deadzone_filter import DeadzoneFilter
from filters.ema_filter import ExponentialMovingAverage
from filters.kalman_filter import KalmanFilter2D
from tracking.tracking_result import Point2D


class FilterChain:
    """
    Sequential pipeline of smoothing filters.

    Default chain: Kalman → EMA → Deadzone

    The Kalman filter handles the bulk of noise reduction,
    EMA smooths out residual high-frequency noise, and
    the deadzone eliminates micro-jitter at rest.
    """

    def __init__(
        self,
        kalman_process_noise: float = DEFAULT_KALMAN_PROCESS_NOISE,
        kalman_measurement_noise: float = DEFAULT_KALMAN_MEASUREMENT_NOISE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        deadzone_threshold: float = DEFAULT_DEADZONE_THRESHOLD,
    ):
        self._filters: list[BaseFilter] = [
            KalmanFilter2D(
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise,
            ),
            ExponentialMovingAverage(alpha=ema_alpha),
            DeadzoneFilter(threshold=deadzone_threshold),
        ]

    def process(self, point: Point2D) -> Point2D:
        """
        Run a point through all filters in sequence.

        Args:
            point: Raw tracked point.

        Returns:
            Smoothed point after all filters.
        """
        result = point
        for f in self._filters:
            result = f.process(result)
        return result

    def reset(self) -> None:
        """Reset all filters. Call when tracking is lost."""
        for f in self._filters:
            f.reset()

    @property
    def filters(self) -> list[BaseFilter]:
        """Access individual filters for runtime tuning."""
        return self._filters

    @property
    def kalman(self) -> KalmanFilter2D:
        """Direct access to the Kalman filter for tuning."""
        return self._filters[0]  # type: ignore[return-value]

    @property
    def ema(self) -> ExponentialMovingAverage:
        """Direct access to the EMA filter for tuning."""
        return self._filters[1]  # type: ignore[return-value]

    @property
    def deadzone(self) -> DeadzoneFilter:
        """Direct access to the deadzone filter for tuning."""
        return self._filters[2]  # type: ignore[return-value]

    @property
    def is_initialized(self) -> bool:
        """Whether the first filter in the chain has been initialized."""
        return self._filters[0].is_initialized if self._filters else False
