"""Constant-velocity Kalman tracking for two ordered profile breakpoints.

The filter is deliberately measurement-agnostic: it predicts where the two
physical endpoints should be searched, orders a measured endpoint pair, and
quantifies whether that pair is statistically plausible.  It never fabricates
an endpoint measurement and therefore cannot hide a missed detection from the
calibration solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DualEndpointKalmanConfig:
    initial_position_std_m: float = 0.0005
    initial_velocity_std_m_s: float = 0.05
    process_acceleration_std_m_s2: float = 1.0
    measurement_std_floor_m: float = 0.00008
    mahalanobis_threshold: float = 9.21
    assignment_ambiguity_ratio: float = 0.05
    maximum_endpoint_speed_m_s: float = 0.25
    minimum_dt_s: float = 1.0e-4
    maximum_dt_s: float = 0.20

    def __post_init__(self) -> None:
        positive = (
            "initial_position_std_m",
            "initial_velocity_std_m_s",
            "process_acceleration_std_m_s2",
            "measurement_std_floor_m",
            "mahalanobis_threshold",
            "maximum_endpoint_speed_m_s",
            "minimum_dt_s",
            "maximum_dt_s",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_dt_s < self.minimum_dt_s:
            raise ValueError("maximum_dt_s must not be smaller than minimum_dt_s")
        if not 0.0 <= self.assignment_ambiguity_ratio < 1.0:
            raise ValueError("assignment_ambiguity_ratio must be in [0, 1)")


class DualEndpointKalmanTracker:
    """Jointly track ``e1/e2`` in sensor X-Z coordinates.

    State order is ``[x1, z1, x2, z2, vx1, vz1, vx2, vz2]``.  The two
    endpoints share one timestamp but retain separate position covariance and
    identity.  Pair-level geometric validation remains the responsibility of
    the breakpoint detector.
    """

    def __init__(self, config: DualEndpointKalmanConfig | None = None) -> None:
        self.config = config or DualEndpointKalmanConfig()
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.timestamp_s: float | None = None
        self.missed_frames = 0
        self.missed_frames_by_endpoint = np.zeros(2, dtype=int)

    @property
    def initialized(self) -> bool:
        return self.state is not None and self.covariance is not None

    def reset(self, endpoints: np.ndarray, timestamp_s: float | None = None) -> None:
        endpoints = np.asarray(endpoints, dtype=float).reshape(2, 3)
        if not np.all(np.isfinite(endpoints[:, (0, 2)])):
            raise ValueError("endpoint initialization must be finite")
        self.state = np.zeros(8, dtype=float)
        self.state[:4] = endpoints[:, (0, 2)].reshape(4)
        position_variance = self.config.initial_position_std_m**2
        velocity_variance = self.config.initial_velocity_std_m_s**2
        self.covariance = np.diag(
            [position_variance] * 4 + [velocity_variance] * 4
        )
        self.timestamp_s = None if timestamp_s is None else float(timestamp_s)
        self.missed_frames = 0
        self.missed_frames_by_endpoint = np.zeros(2, dtype=int)

    def _limit_endpoint_speeds(self) -> None:
        """Prevent one noisy association from creating an unbounded coast."""
        if not self.initialized:
            return
        velocities = self.state[4:].reshape(2, 2)
        speeds = np.linalg.norm(velocities, axis=1)
        limit = self.config.maximum_endpoint_speed_m_s
        for endpoint, speed in enumerate(speeds):
            if speed > limit:
                velocities[endpoint] *= limit / max(float(speed), 1.0e-15)
        self.state[4:] = velocities.reshape(4)

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt
        q = self.config.process_acceleration_std_m_s2**2
        process = np.zeros((8, 8), dtype=float)
        position_variance = 0.25 * dt**4 * q
        cross_covariance = 0.5 * dt**3 * q
        velocity_variance = dt**2 * q
        for coordinate in range(4):
            velocity = coordinate + 4
            process[coordinate, coordinate] = position_variance
            process[coordinate, velocity] = cross_covariance
            process[velocity, coordinate] = cross_covariance
            process[velocity, velocity] = velocity_variance
        return transition, process

    def predict(self, timestamp_s: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("dual endpoint Kalman tracker is not initialized")
        timestamp = float(timestamp_s)
        if self.timestamp_s is None:
            self.timestamp_s = timestamp
            return self.endpoints()
        raw_dt = timestamp - self.timestamp_s
        # Repeated or out-of-order sensor stamps contain no forward-time
        # information. Advancing by an artificial minimum dt would accumulate
        # displacement while the physical time has not advanced.
        if raw_dt <= 0.0:
            return self.endpoints()
        dt = float(
            np.clip(
                raw_dt,
                self.config.minimum_dt_s,
                self.config.maximum_dt_s,
            )
        )
        transition, process = self._transition(dt)
        self.state = transition @ self.state
        self._limit_endpoint_speeds()
        self.covariance = (
            transition @ self.covariance @ transition.T + process
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.timestamp_s = timestamp
        return self.endpoints()

    def endpoints(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("dual endpoint Kalman tracker is not initialized")
        xz = self.state[:4].reshape(2, 2)
        return np.column_stack((xz[:, 0], np.zeros(2), xz[:, 1]))

    def endpoint_position_covariance(self, endpoint: int) -> np.ndarray:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be zero or one")
        if not self.initialized:
            raise RuntimeError("dual endpoint Kalman tracker is not initialized")
        start = 2 * endpoint
        return self.covariance[start : start + 2, start : start + 2].copy()

    def search_radius(
        self,
        *,
        minimum_m: float,
        maximum_m: float,
        sigma_multiplier: float = 3.0,
    ) -> float:
        """Return the larger of the two endpoint-specific search radii."""
        return float(
            np.max(
                self.endpoint_search_radii(
                    minimum_m=minimum_m,
                    maximum_m=maximum_m,
                    sigma_multiplier=sigma_multiplier,
                )
            )
        )

    def endpoint_search_radii(
        self,
        *,
        minimum_m: float,
        maximum_m: float,
        sigma_multiplier: float = 3.0,
    ) -> np.ndarray:
        """Return independent covariance-driven gates for ``e1`` and ``e2``."""
        if minimum_m <= 0.0 or maximum_m < minimum_m:
            raise ValueError("invalid search-radius limits")
        radii: list[float] = []
        for endpoint in (0, 1):
            eigenvalues = np.linalg.eigvalsh(
                self.endpoint_position_covariance(endpoint)
            )
            sigma = float(
                np.sqrt(max(float(np.max(eigenvalues)), 0.0))
            )
            radii.append(
                float(
                    np.clip(
                        sigma_multiplier * sigma,
                        minimum_m,
                        maximum_m,
                    )
                )
            )
        return np.asarray(radii, dtype=float)

    def _measurement_covariance(self, sigma_m: float) -> np.ndarray:
        sigma = max(float(sigma_m), self.config.measurement_std_floor_m)
        return np.eye(4) * sigma**2

    def order_measurement(
        self, endpoints: np.ndarray, *, measurement_sigma_m: float
    ) -> tuple[np.ndarray, tuple[float, float]] | None:
        """Assign a measured pair to persistent e1/e2 with chi-square gates."""
        if not self.initialized:
            raise RuntimeError("dual endpoint Kalman tracker is not initialized")
        endpoints = np.asarray(endpoints, dtype=float).reshape(2, 3)
        measured_options = (endpoints, endpoints[::-1])
        predicted = self.state[:4].reshape(2, 2)
        measurement_variance = max(
            float(measurement_sigma_m), self.config.measurement_std_floor_m
        ) ** 2

        option_costs: list[tuple[float, tuple[float, float], np.ndarray]] = []
        for option in measured_options:
            measured = option[:, (0, 2)]
            distances: list[float] = []
            for endpoint in (0, 1):
                innovation = measured[endpoint] - predicted[endpoint]
                innovation_covariance = (
                    self.endpoint_position_covariance(endpoint)
                    + np.eye(2) * measurement_variance
                )
                distance = float(
                    innovation.T
                    @ np.linalg.solve(innovation_covariance, innovation)
                )
                distances.append(distance)
            option_costs.append(
                (sum(distances), (distances[0], distances[1]), option.copy())
            )
        option_costs.sort(key=lambda item: item[0])
        best, second = option_costs
        scale = max(abs(best[0]), abs(second[0]), 1.0e-12)
        if abs(best[0] - second[0]) / scale < self.config.assignment_ambiguity_ratio:
            return None
        if max(best[1]) > self.config.mahalanobis_threshold:
            return None
        return best[2], best[1]

    def select_endpoint_candidate(
        self,
        endpoint: int,
        candidates: np.ndarray,
        *,
        measurement_sigma_m: float,
    ) -> tuple[np.ndarray, float] | None:
        """Select one local physical-breakpoint candidate for ``e1`` or ``e2``.

        This is deliberately an endpoint-level operation.  It is used only
        when a complete fitted breakpoint pair is unavailable, so one visible
        physical edge can still constrain its own Kalman state while the
        other endpoint coasts.  Calibration output remains pair-only.
        """
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be zero or one")
        if not self.initialized:
            raise RuntimeError("dual endpoint Kalman tracker is not initialized")
        values = np.asarray(candidates, dtype=float)
        if values.size == 0:
            return None
        values = values.reshape(-1, 3)
        values = values[np.all(np.isfinite(values[:, (0, 2)]), axis=1)]
        if len(values) == 0:
            return None
        predicted = self.state[2 * endpoint : 2 * endpoint + 2]
        measurement_variance = max(
            float(measurement_sigma_m), self.config.measurement_std_floor_m
        ) ** 2
        innovation_covariance = (
            self.endpoint_position_covariance(endpoint)
            + np.eye(2) * measurement_variance
        )
        inverse = np.linalg.inv(innovation_covariance)
        scored: list[tuple[float, np.ndarray]] = []
        for candidate in values:
            innovation = candidate[[0, 2]] - predicted
            distance = float(innovation.T @ inverse @ innovation)
            if distance <= self.config.mahalanobis_threshold:
                scored.append((distance, candidate.copy()))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        return scored[0][1], scored[0][0]

    def update(self, endpoints: np.ndarray, *, measurement_sigma_m: float) -> None:
        endpoints = np.asarray(endpoints, dtype=float).reshape(2, 3)
        self.update_partial(
            {0: endpoints[0], 1: endpoints[1]},
            measurement_sigma_m=measurement_sigma_m,
        )

    def update_partial(
        self,
        measurements: dict[int, np.ndarray],
        *,
        measurement_sigma_m: float,
    ) -> None:
        """Update any visible subset of the two physical breakpoints.

        Missing endpoints are not fabricated: their state is only propagated
        by :meth:`predict`, and a per-endpoint missed-frame counter expands
        the subsequent search.  A frame with no measurements must instead use
        :meth:`mark_missed`.
        """
        if not self.initialized:
            raise RuntimeError("dual endpoint Kalman tracker is not initialized")
        if not measurements:
            raise ValueError("partial update needs at least one endpoint")
        endpoints = sorted(int(endpoint) for endpoint in measurements)
        if any(endpoint not in (0, 1) for endpoint in endpoints):
            raise ValueError("endpoint keys must be zero or one")
        measurement_values: list[float] = []
        observed_coordinates: list[int] = []
        for endpoint in endpoints:
            value = np.asarray(measurements[endpoint], dtype=float).reshape(3)
            if not np.all(np.isfinite(value[[0, 2]])):
                raise ValueError("partial endpoint measurements must be finite")
            measurement_values.extend(value[[0, 2]].tolist())
            observed_coordinates.extend((2 * endpoint, 2 * endpoint + 1))
        measurement = np.asarray(measurement_values, dtype=float)
        observation = np.zeros((len(observed_coordinates), 8), dtype=float)
        for row, coordinate in enumerate(observed_coordinates):
            observation[row, coordinate] = 1.0
        sigma = max(
            float(measurement_sigma_m), self.config.measurement_std_floor_m
        )
        measurement_covariance = np.eye(len(measurement)) * sigma**2
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T
            + measurement_covariance
        )
        gain = (
            self.covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        self._limit_endpoint_speeds()
        identity = np.eye(8)
        joseph = identity - gain @ observation
        self.covariance = (
            joseph @ self.covariance @ joseph.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        observed = set(endpoints)
        for endpoint in (0, 1):
            if endpoint in observed:
                self.missed_frames_by_endpoint[endpoint] = 0
            else:
                self.missed_frames_by_endpoint[endpoint] += 1
        self.missed_frames = int(np.max(self.missed_frames_by_endpoint))

    def mark_missed(self, endpoint: int | None = None) -> None:
        if self.initialized:
            if endpoint is None:
                self.missed_frames_by_endpoint += 1
            elif endpoint in (0, 1):
                self.missed_frames_by_endpoint[endpoint] += 1
            else:
                raise ValueError("endpoint must be zero, one or None")
            self.missed_frames = int(np.max(self.missed_frames_by_endpoint))
