"""Command-line smoke demo for the ROS-independent calibration core."""

from __future__ import annotations

import argparse

import numpy as np

from .geometry import rotation_distance_deg, so3_exp
from .pipeline import ActiveCalibrationPipeline
from .simulation.synthetic import default_scene, generate_seed_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=8)
    parser.add_argument("--noise-mm", type=float, default=0.0)
    args = parser.parse_args()

    scene = default_scene()
    poses, measurements = generate_seed_dataset(
        scene, count=args.poses, noise_std=args.noise_mm / 1000.0
    )
    nominal_rotation = scene.handeye_rotation @ so3_exp(np.deg2rad(np.array([1.0, -2.0, 1.5])))
    nominal_translation = scene.handeye_translation + np.array([0.008, -0.005, 0.006])
    pipeline = ActiveCalibrationPipeline(
        nominal_rotation,
        nominal_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=args.poses,
    )
    for pose, measurement in zip(poses, measurements):
        pipeline.append_seed(pose, measurement)
    result = pipeline.initialize()
    rotation_error = rotation_distance_deg(
        result.estimate.handeye_rotation, scene.handeye_rotation
    )
    translation_error = (
        np.linalg.norm(result.estimate.handeye_translation - scene.handeye_translation) * 1000.0
    )
    print(f"converged={result.converged} rank={result.diagnostics.rank}")
    print(f"cost={result.cost:.6e}")
    print(f"rotation_error_deg={rotation_error:.6f}")
    print(f"translation_error_mm={translation_error:.6f}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
