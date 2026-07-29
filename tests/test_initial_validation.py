from dataclasses import replace

import numpy as np

from calibration_pipeline.dataset_io import SeedObservationGroup
from calibration_pipeline.geometry import so3_exp
from calibration_pipeline.initial_validation import (
    bootstrap_initial_stability,
    inflate_handeye_covariance_from_stability,
)
from calibration_pipeline.models import FlangePose
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset
from calibration_pipeline.solvers import TwelveDofV2Solver


def _groups_with_duplicate_frames():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=6)
    return scene, tuple(
        SeedObservationGroup(
            f"seed_{index}",
            (pose, pose),
            (measurement, measurement),
        )
        for index, (pose, measurement) in enumerate(zip(poses, measurements))
    )


def test_identical_multiframe_seeds_have_zero_bootstrap_spread():
    scene, groups = _groups_with_duplicate_frames()
    solver = TwelveDofV2Solver()
    poses = [pose for group in groups for pose in group.poses]
    measurements = [
        measurement for group in groups for measurement in group.measurements
    ]
    result = solver.solve(
        poses,
        measurements,
        scene.handeye_rotation,
        scene.handeye_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    report = bootstrap_initial_stability(
        groups,
        solver,
        result,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
        trials=3,
    )
    assert report.available
    assert report.accepted
    assert report.rotation_p95_deg < 1e-5
    assert report.translation_p95_m < 1e-8


def test_bootstrap_gate_rejects_repeatably_shifted_solutions():
    scene, groups = _groups_with_duplicate_frames()
    solver = TwelveDofV2Solver()
    poses = [pose for group in groups for pose in group.poses]
    measurements = [
        measurement for group in groups for measurement in group.measurements
    ]
    reference = solver.solve(
        poses,
        measurements,
        scene.handeye_rotation,
        scene.handeye_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    shifted_estimate = replace(
        reference.estimate,
        handeye_rotation=(
            reference.estimate.handeye_rotation
            @ so3_exp(np.deg2rad(np.array([2.0, 0.0, 0.0])))
        ),
        handeye_translation=(
            reference.estimate.handeye_translation
            + np.array([0.01, 0.0, 0.0])
        ),
    )

    class ShiftedSolver:
        def solve(self, *_args, **_kwargs):
            return replace(reference, estimate=shifted_estimate)

    report = bootstrap_initial_stability(
        groups,
        ShiftedSolver(),
        reference,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
        trials=3,
        maximum_rotation_p95_deg=1.0,
        maximum_translation_p95_m=0.005,
    )
    assert not report.accepted
    assert "rotation p95" in report.reason
    assert "translation p95" in report.reason

    inflated = inflate_handeye_covariance_from_stability(reference, report)
    assert (
        inflated.estimate.covariance_x9[0, 0]
        >= (np.deg2rad(report.rotation_p95_deg) / 1.96) ** 2
    )
    assert (
        inflated.estimate.covariance_x9[3, 3]
        >= (report.translation_p95_m / 1.96) ** 2
    )
