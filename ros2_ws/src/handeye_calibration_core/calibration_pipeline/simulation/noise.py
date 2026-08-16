"""Configurable, reproducible noise models for the Gazebo calibration sensor."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from ..geometry import make_transform, so3_exp


@dataclass(frozen=True)
class SimulationNoiseConfig:
    random_seed: int = 20260728
    profile_gaussian_std_m: float = 0.000055
    endpoint_gaussian_std_m: float = 0.000080
    robot_translation_std_m: float = 0.000030
    robot_rotation_std_deg: float = 0.003
    board_flatness_rms_m: float = 0.000030
    sync_delay_mean_s: float = 0.0
    sync_jitter_std_s: float = 0.001
    point_outlier_probability: float = 0.002
    point_outlier_std_m: float = 0.0005
    endpoint_outlier_probability: float = 0.001
    endpoint_outlier_std_m: float = 0.0005
    point_dropout_probability: float = 0.01
    frame_dropout_probability: float = 0.001
    endpoint_dropout_probability: float = 0.002

    def __post_init__(self) -> None:
        nonnegative = (
            "profile_gaussian_std_m",
            "endpoint_gaussian_std_m",
            "robot_translation_std_m",
            "robot_rotation_std_deg",
            "board_flatness_rms_m",
            "sync_delay_mean_s",
            "sync_jitter_std_s",
            "point_outlier_std_m",
            "endpoint_outlier_std_m",
        )
        probabilities = (
            "point_outlier_probability",
            "endpoint_outlier_probability",
            "point_dropout_probability",
            "frame_dropout_probability",
            "endpoint_dropout_probability",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in probabilities:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


class JointSnapshotBuffer:
    """Select an encoder snapshot delayed from the current measurement time."""

    def __init__(self, maximum_size: int = 2000) -> None:
        if maximum_size < 2:
            raise ValueError("maximum_size must be at least two")
        self._items: deque[tuple[int, np.ndarray]] = deque(maxlen=maximum_size)

    def append(self, monotonic_ns: int, joints: np.ndarray) -> None:
        self._items.append(
            (int(monotonic_ns), np.asarray(joints, dtype=float).copy())
        )

    def delayed(self, now_ns: int, delay_s: float) -> np.ndarray | None:
        if not self._items:
            return None
        target = int(now_ns - max(float(delay_s), 0.0) * 1e9)
        # History is ordered. Choose the newest sample not later than target;
        # if the requested delay predates the buffer, use its oldest sample.
        for stamp, joints in reversed(self._items):
            if stamp <= target:
                return joints.copy()
        return self._items[0][1].copy()


class SimulationNoiseModel:
    """Apply physical model mismatch and sensor extraction noise."""

    def __init__(self, config: SimulationNoiseConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(
            None if config.random_seed < 0 else config.random_seed
        )
        self._flatness_coefficients = self.rng.normal(size=6)
        grid = np.linspace(0.0, 1.0, 41)
        xi, eta = np.meshgrid(grid, grid, indexing="ij")
        raw = self._flatness_raw(xi.reshape(-1), eta.reshape(-1))
        self._flatness_scale = max(float(np.sqrt(np.mean(raw**2))), 1e-12)

    def sample_sync_delay_s(self) -> float:
        delay = self.config.sync_delay_mean_s + self.rng.normal(
            0.0, self.config.sync_jitter_std_s
        )
        return max(float(delay), 0.0)

    def sample_frame_dropout(self) -> bool:
        return bool(
            self.rng.random() < self.config.frame_dropout_probability
        )

    def perturb_flange(self, flange_transform: np.ndarray) -> np.ndarray:
        transform = np.asarray(flange_transform, dtype=float)
        translation = transform[:3, 3] + self.rng.normal(
            0.0, self.config.robot_translation_std_m, 3
        )
        rotation_vector = self.rng.normal(
            0.0, np.deg2rad(self.config.robot_rotation_std_deg), 3
        )
        rotation = transform[:3, :3] @ so3_exp(rotation_vector)
        return make_transform(rotation, translation)

    def _flatness_raw(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        c = self._flatness_coefficients
        basis = np.column_stack(
            (
                np.sin(np.pi * xi),
                np.sin(np.pi * eta),
                np.sin(np.pi * xi) * np.sin(np.pi * eta),
                np.sin(2.0 * np.pi * xi + 0.31) * np.sin(np.pi * eta),
                np.sin(np.pi * xi) * np.sin(2.0 * np.pi * eta - 0.47),
                (xi - 0.5) * (eta - 0.5),
            )
        )
        return basis @ c

    def flatness_height(
        self,
        points_base: np.ndarray,
        *,
        corner: np.ndarray,
        board_u: np.ndarray,
        board_v: np.ndarray,
        width: float,
        height: float,
    ) -> np.ndarray:
        points = np.asarray(points_base, dtype=float).reshape(-1, 3)
        delta = points - np.asarray(corner, dtype=float)
        xi = (delta @ np.asarray(board_u, dtype=float)) / float(width)
        eta = (delta @ np.asarray(board_v, dtype=float)) / float(height)
        raw = self._flatness_raw(xi, eta)
        return self.config.board_flatness_rms_m * raw / self._flatness_scale

    def deform_points_in_laser_plane(
        self,
        points_base: np.ndarray,
        *,
        laser_normal: np.ndarray,
        board_normal: np.ndarray,
        corner: np.ndarray,
        board_u: np.ndarray,
        board_v: np.ndarray,
        width: float,
        height: float,
    ) -> np.ndarray:
        """Approximate a fixed non-planar plate while retaining laser coplanarity."""
        points = np.asarray(points_base, dtype=float).reshape(-1, 3)
        if len(points) == 0 or self.config.board_flatness_rms_m == 0.0:
            return points.copy()
        laser_normal = np.asarray(laser_normal, dtype=float)
        board_normal = np.asarray(board_normal, dtype=float)
        direction = board_normal - laser_normal * float(
            laser_normal @ board_normal
        )
        denominator = float(board_normal @ direction)
        if abs(denominator) < 1e-10:
            return points.copy()
        heights = self.flatness_height(
            points,
            corner=corner,
            board_u=board_u,
            board_v=board_v,
            width=width,
            height=height,
        )
        return points + (heights / denominator)[:, None] * direction[None, :]

    def deform_bounded_scanline(
        self,
        points_base: np.ndarray,
        boundary_endpoints_base: np.ndarray,
        *,
        boundary_labels: list[str] | tuple[str, ...],
        laser_normal: np.ndarray,
        board_normal: np.ndarray,
        corner: np.ndarray,
        board_u: np.ndarray,
        board_v: np.ndarray,
        width: float,
        height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp a bilateral scanline and re-solve its physical boundaries.

        Directly warping the two flat-board endpoints moves them tangentially
        away from their physical edges.  That creates a contradictory datum:
        the detector labels a point as an edge endpoint although the simulated
        point no longer lies on that edge.  Here the flat scanline parameter is
        extended and each boundary intersection is solved again after applying
        the fixed height field.  The returned endpoint order follows
        ``boundary_endpoints_base``.
        """
        points = np.asarray(points_base, dtype=float).reshape(-1, 3)
        endpoints = np.asarray(boundary_endpoints_base, dtype=float).reshape(-1, 3)
        labels = tuple(str(label) for label in boundary_labels)
        if (
            len(points) < 2
            or len(endpoints) != 2
            or len(labels) != 2
            or set(labels) != {"e1", "e2"}
            or self.config.board_flatness_rms_m == 0.0
        ):
            deformed = self.deform_points_in_laser_plane(
                points,
                laser_normal=laser_normal,
                board_normal=board_normal,
                corner=corner,
                board_u=board_u,
                board_v=board_v,
                width=width,
                height=height,
            )
            endpoint_deformed = self.deform_points_in_laser_plane(
                endpoints,
                laser_normal=laser_normal,
                board_normal=board_normal,
                corner=corner,
                board_u=board_u,
                board_v=board_v,
                width=width,
                height=height,
            )
            return deformed, endpoint_deformed

        laser_normal = np.asarray(laser_normal, dtype=float)
        normal = np.asarray(board_normal, dtype=float)
        board_u = np.asarray(board_u, dtype=float)
        board_v = np.asarray(board_v, dtype=float)
        corner = np.asarray(corner, dtype=float)
        direction = normal - laser_normal * float(laser_normal @ normal)
        denominator = float(normal @ direction)
        if abs(denominator) < 1e-10:
            return points.copy(), endpoints.copy()
        line_direction = points[-1] - points[0]
        line_norm = float(np.linalg.norm(line_direction))
        if line_norm < 1e-10:
            return points.copy(), endpoints.copy()
        line_direction /= line_norm
        origin = points[0]

        def warped(parameter: float) -> tuple[np.ndarray, np.ndarray]:
            flat_point = origin + float(parameter) * line_direction
            height_value = float(
                self.flatness_height(
                    flat_point[None, :],
                    corner=corner,
                    board_u=board_u,
                    board_v=board_v,
                    width=width,
                    height=height,
                )[0]
            )
            # Fixed-point evaluation makes the height depend on the actual
            # tangent-plane coordinates rather than the original flat point.
            for _ in range(5):
                point = flat_point + (height_value / denominator) * direction
                projected = point - normal * float(normal @ (point - corner))
                updated = float(
                    self.flatness_height(
                        projected[None, :],
                        corner=corner,
                        board_u=board_u,
                        board_v=board_v,
                        width=width,
                        height=height,
                    )[0]
                )
                if abs(updated - height_value) < 1e-12:
                    height_value = updated
                    break
                height_value = updated
            point = flat_point + (height_value / denominator) * direction
            projected = point - normal * float(normal @ (point - corner))
            return point, projected

        def boundary_value(parameter: float, label: str, endpoint: np.ndarray) -> float:
            _, projected = warped(parameter)
            if label == "e1":
                coordinate = float((projected - corner) @ board_v)
                endpoint_coordinate = float((endpoint - corner) @ board_v)
                target = 0.0 if endpoint_coordinate < 0.5 * height else float(height)
            else:
                coordinate = float((projected - corner) @ board_u)
                endpoint_coordinate = float((endpoint - corner) @ board_u)
                target = 0.0 if endpoint_coordinate < 0.5 * width else float(width)
            return coordinate - target

        def solve_boundary(endpoint: np.ndarray, label: str) -> float | None:
            center = float((endpoint - origin) @ line_direction)
            span = max(0.005, 20.0 * self.config.board_flatness_rms_m)
            for _ in range(4):
                lower = center - span
                upper = center + span
                f_lower = boundary_value(lower, label, endpoint)
                f_upper = boundary_value(upper, label, endpoint)
                if f_lower == 0.0:
                    return lower
                if f_upper == 0.0:
                    return upper
                if f_lower * f_upper < 0.0:
                    for _ in range(50):
                        middle = 0.5 * (lower + upper)
                        f_middle = boundary_value(middle, label, endpoint)
                        if abs(f_middle) < 1e-12:
                            return middle
                        if f_lower * f_middle <= 0.0:
                            upper = middle
                        else:
                            lower = middle
                            f_lower = f_middle
                    return 0.5 * (lower + upper)
                span *= 2.0
            return None

        roots = [
            solve_boundary(endpoint, label)
            for endpoint, label in zip(endpoints, labels)
        ]
        if any(root is None for root in roots):
            deformed = self.deform_points_in_laser_plane(
                points,
                laser_normal=laser_normal,
                board_normal=normal,
                corner=corner,
                board_u=board_u,
                board_v=board_v,
                width=width,
                height=height,
            )
            endpoint_deformed = self.deform_points_in_laser_plane(
                endpoints,
                laser_normal=laser_normal,
                board_normal=normal,
                corner=corner,
                board_u=board_u,
                board_v=board_v,
                width=width,
                height=height,
            )
            return deformed, endpoint_deformed
        root_values = np.asarray(roots, dtype=float)
        parameters = np.linspace(
            float(np.min(root_values)),
            float(np.max(root_values)),
            len(points),
        )
        flat_points = origin[None, :] + parameters[:, None] * line_direction[None, :]
        height_values = self.flatness_height(
            flat_points,
            corner=corner,
            board_u=board_u,
            board_v=board_v,
            width=width,
            height=height,
        )
        for _ in range(5):
            warped_points = (
                flat_points
                + (height_values / denominator)[:, None] * direction[None, :]
            )
            signed_height = (warped_points - corner[None, :]) @ normal
            projected = warped_points - signed_height[:, None] * normal[None, :]
            updated = self.flatness_height(
                projected,
                corner=corner,
                board_u=board_u,
                board_v=board_v,
                width=width,
                height=height,
            )
            if float(np.max(np.abs(updated - height_values))) < 1e-12:
                height_values = updated
                break
            height_values = updated
        warped_points = (
            flat_points
            + (height_values / denominator)[:, None] * direction[None, :]
        )
        root_points = np.asarray([warped(value)[0] for value in root_values])
        return warped_points, root_points

    def corrupt_profile(
        self, points_sensor: np.ndarray, *, frame_dropped: bool = False
    ) -> np.ndarray:
        points = np.asarray(points_sensor, dtype=float).reshape(-1, 3).copy()
        if frame_dropped:
            return np.zeros((0, 3))
        if len(points) == 0:
            return points
        points[:, (0, 2)] += self.rng.normal(
            0.0,
            self.config.profile_gaussian_std_m,
            (len(points), 2),
        )
        outliers = (
            self.rng.random(len(points))
            < self.config.point_outlier_probability
        )
        if np.any(outliers):
            outlier_indices = np.flatnonzero(outliers)
            points[np.ix_(outlier_indices, (0, 2))] += self.rng.normal(
                0.0,
                self.config.point_outlier_std_m,
                (len(outlier_indices), 2),
            )
        keep = (
            self.rng.random(len(points))
            >= self.config.point_dropout_probability
        )
        points = points[keep]
        points[:, 1] = 0.0
        return points

    def corrupt_endpoint(
        self,
        endpoint_sensor: np.ndarray,
        *,
        frame_dropped: bool = False,
    ) -> tuple[np.ndarray, bool]:
        point = np.asarray(endpoint_sensor, dtype=float).copy()
        if (
            frame_dropped
            or self.rng.random()
            < self.config.endpoint_dropout_probability
        ):
            return point, False
        point[[0, 2]] += self.rng.normal(
            0.0, self.config.endpoint_gaussian_std_m, 2
        )
        if self.rng.random() < self.config.endpoint_outlier_probability:
            point[[0, 2]] += self.rng.normal(
                0.0, self.config.endpoint_outlier_std_m, 2
            )
        point[1] = 0.0
        return point, True
