"""Validated data models shared by calibration, NBV and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import make_transform


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} is not orthonormal")
    if np.linalg.det(array) < 0.0:
        raise ValueError(f"{name} must be right-handed")
    return array


@dataclass(frozen=True)
class FlangePose:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _rotation(self.rotation, "rotation"))
        object.__setattr__(self, "translation", _vector(self.translation, "translation"))

    @property
    def transform(self) -> np.ndarray:
        return make_transform(self.rotation, self.translation)


@dataclass(frozen=True)
class Measurement:
    profile_points: np.ndarray
    endpoint_u: np.ndarray
    endpoint_v: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.profile_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("profile_points must have shape (N, 3)")
        if len(points) < 2:
            raise ValueError("a bilateral measurement needs at least two profile points")
        object.__setattr__(self, "profile_points", points)
        object.__setattr__(self, "endpoint_u", _vector(self.endpoint_u, "endpoint_u"))
        object.__setattr__(self, "endpoint_v", _vector(self.endpoint_v, "endpoint_v"))

    def as_solver_dict(self) -> dict[str, Any]:
        return {
            "p_S_plane": self.profile_points,
            "p_S_e1": self.endpoint_u,
            "p_S_e2": self.endpoint_v,
            "valid_e1": True,
            "valid_e2": True,
        }


@dataclass(frozen=True)
class BoardModel:
    corner: np.ndarray
    rotation: np.ndarray
    length_u: float
    length_v: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "corner", _vector(self.corner, "corner"))
        object.__setattr__(self, "rotation", _rotation(self.rotation, "rotation"))
        if self.length_u <= 0.0 or self.length_v <= 0.0:
            raise ValueError("board dimensions must be positive")

    @property
    def u(self) -> np.ndarray:
        return self.rotation[:, 0]

    @property
    def v(self) -> np.ndarray:
        return self.rotation[:, 1]

    @property
    def normal(self) -> np.ndarray:
        return self.rotation[:, 2]

    @property
    def corners(self) -> np.ndarray:
        return np.array(
            [
                self.corner,
                self.corner + self.length_u * self.u,
                self.corner + self.length_u * self.u + self.length_v * self.v,
                self.corner + self.length_v * self.v,
            ]
        )


@dataclass(frozen=True)
class TrapezoidDomain:
    """A calibrated polygon in the Gocator ``X_S-Z_S`` measurement plane."""

    z_near: float
    z_far: float
    x_left_near: float
    x_left_far: float
    x_right_near: float
    x_right_far: float

    def __post_init__(self) -> None:
        if self.z_far <= self.z_near:
            raise ValueError("z_far must be greater than z_near")
        if self.x_left_near >= self.x_right_near:
            raise ValueError("near trapezoid width must be positive")
        if self.x_left_far >= self.x_right_far:
            raise ValueError("far trapezoid width must be positive")

    def x_limits(self, z: float) -> tuple[float, float]:
        fraction = (float(z) - self.z_near) / (self.z_far - self.z_near)
        left = (1.0 - fraction) * self.x_left_near + fraction * self.x_left_far
        right = (1.0 - fraction) * self.x_right_near + fraction * self.x_right_far
        return float(left), float(right)

    def margin(self, point_sensor: np.ndarray) -> float:
        x, _, z = _vector(point_sensor, "point_sensor")
        left, right = self.x_limits(z)
        return float(min(z - self.z_near, self.z_far - z, x - left, right - x))

    def contains(self, point_sensor: np.ndarray) -> bool:
        return self.margin(point_sensor) >= 0.0


@dataclass(frozen=True)
class SensorROI:
    """Outer valid and inner planning-safe Gocator measurement domains.

    ``min_range``/``max_range``/``half_fov_deg`` remain as a compatibility
    input.  New configurations should provide explicit, possibly asymmetric,
    hard and safe trapezoids in the same metric sensor coordinates as profiles.
    """

    min_range: float = 0.27
    max_range: float = 0.82
    half_fov_deg: float = 15.0
    safe_margin: float = 0.01
    hard_domain: TrapezoidDomain | None = None
    safe_domain: TrapezoidDomain | None = None

    def __post_init__(self) -> None:
        if self.hard_domain is None:
            tangent = float(np.tan(np.deg2rad(self.half_fov_deg)))
            hard = TrapezoidDomain(
                self.min_range,
                self.max_range,
                -self.min_range * tangent,
                -self.max_range * tangent,
                self.min_range * tangent,
                self.max_range * tangent,
            )
            object.__setattr__(self, "hard_domain", hard)
        if self.safe_domain is None:
            hard = self.hard_domain
            assert hard is not None
            safe = TrapezoidDomain(
                hard.z_near + self.safe_margin,
                hard.z_far - self.safe_margin,
                hard.x_left_near + self.safe_margin,
                hard.x_left_far + self.safe_margin,
                hard.x_right_near - self.safe_margin,
                hard.x_right_far - self.safe_margin,
            )
            object.__setattr__(self, "safe_domain", safe)

    def contains(self, point_sensor: np.ndarray, *, safe: bool = True) -> bool:
        return self.margin(point_sensor, safe=safe) >= 0.0

    def margin(self, point_sensor: np.ndarray, *, safe: bool = True) -> float:
        domain = self.safe_domain if safe else self.hard_domain
        assert domain is not None
        return domain.margin(point_sensor)

    def hard_margin(self, point_sensor: np.ndarray) -> float:
        return self.margin(point_sensor, safe=False)


@dataclass(frozen=True)
class CalibrationEstimate:
    handeye_rotation: np.ndarray
    handeye_translation: np.ndarray
    board: BoardModel
    x9: np.ndarray
    covariance_x9: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "handeye_rotation", _rotation(self.handeye_rotation, "handeye_rotation")
        )
        object.__setattr__(
            self, "handeye_translation", _vector(self.handeye_translation, "handeye_translation")
        )
        x9 = np.asarray(self.x9, dtype=float)
        if x9.shape != (9,):
            raise ValueError("x9 must have shape (9,)")
        object.__setattr__(self, "x9", x9)

    @property
    def handeye_transform(self) -> np.ndarray:
        return make_transform(self.handeye_rotation, self.handeye_translation)


@dataclass(frozen=True)
class SolverDiagnostics:
    singular_values: np.ndarray
    rank: int
    condition_number: float
    residual_variance: float
    effective_handeye_information: np.ndarray
    weakest_direction: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    estimate: CalibrationEstimate
    cost: float
    converged: bool
    message: str
    evaluations: int
    diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    a: float
    b: float
    alpha: float
    psi: float
    working_distance: float
    branch: int
    sensor_transform_nominal: np.ndarray
    flange_transform_command: np.ndarray
    virtual_measurement: Measurement | None = None
    nominal_margin: float = float("-inf")


@dataclass(frozen=True)
class Prediction:
    valid: bool
    reason: str
    measurement: Measurement | None = None
    edge_labels: tuple[str, str] | None = None
    roi_margin: float = float("-inf")
    edge_margin: float = float("-inf")
    profile_length: float = 0.0
    intersection_margin: float = float("-inf")


@dataclass
class CandidateScore:
    candidate: Candidate
    prediction: Prediction
    valid_probability: float
    information_gain: float
    minimum_eigenvalue: float
    metadata: dict[str, Any] = field(default_factory=dict)
