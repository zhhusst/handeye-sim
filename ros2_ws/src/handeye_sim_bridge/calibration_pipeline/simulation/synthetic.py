"""Deterministic synthetic bilateral observations for tests and the CLI demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import invert_transform, make_transform, so3_exp
from ..models import BoardModel, FlangePose, Measurement, SensorROI
from ..nbv.candidate_generator import _sensor_transform
from ..nbv.profile_predictor import predict_profile


@dataclass(frozen=True)
class SyntheticScene:
    handeye_rotation: np.ndarray
    handeye_translation: np.ndarray
    board: BoardModel
    roi: SensorROI

    @property
    def handeye_transform(self) -> np.ndarray:
        return make_transform(self.handeye_rotation, self.handeye_translation)


def generate_seed_dataset(
    scene: SyntheticScene,
    *,
    count: int = 8,
    seed: int = 7,
    noise_std: float = 0.0,
) -> tuple[list[FlangePose], list[Measurement]]:
    rng = np.random.default_rng(seed)
    poses: list[FlangePose] = []
    measurements: list[Measurement] = []
    attempts = 0
    while len(poses) < count and attempts < count * 200:
        attempts += 1
        point_u = scene.board.corner + rng.uniform(0.08, scene.board.length_u - 0.08) * scene.board.u
        point_v = scene.board.corner + rng.uniform(0.08, scene.board.length_v - 0.08) * scene.board.v
        sensor = _sensor_transform(
            point_u,
            point_v,
            scene.board.normal,
            np.deg2rad(rng.uniform(22.0, 52.0)),
            np.deg2rad(rng.uniform(-12.0, 12.0)),
            rng.uniform(0.4, 0.65),
            int(rng.choice((-1, 1))),
        )
        if sensor is None:
            continue
        prediction = predict_profile(sensor, scene.board, scene.roi, edge_safe_margin=0.03)
        if not prediction.valid or prediction.measurement is None:
            continue
        flange = sensor @ invert_transform(scene.handeye_transform)
        measurement = prediction.measurement
        if noise_std > 0.0:
            measurement = Measurement(
                measurement.profile_points + rng.normal(0.0, noise_std, measurement.profile_points.shape),
                measurement.endpoint_u + rng.normal(0.0, noise_std, 3),
                measurement.endpoint_v + rng.normal(0.0, noise_std, 3),
            )
        poses.append(FlangePose(flange[:3, :3], flange[:3, 3]))
        measurements.append(measurement)
    if len(poses) != count:
        raise RuntimeError(f"could generate only {len(poses)} of {count} requested poses")
    return poses, measurements


def default_scene() -> SyntheticScene:
    return SyntheticScene(
        handeye_rotation=so3_exp(np.deg2rad(np.array([2.0, -1.0, 3.0]))),
        handeye_translation=np.array([-0.011579, -0.004621, 0.359284]),
        board=BoardModel(
            corner=np.array([0.7, 0.0, 0.25]),
            rotation=np.eye(3),
            length_u=0.4,
            length_v=0.5,
        ),
        roi=SensorROI(min_range=0.27, max_range=0.82, half_fov_deg=15.0, safe_margin=0.005),
    )
