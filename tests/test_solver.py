import numpy as np

from calibration_pipeline.geometry import rotation_distance_deg, so3_exp
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset
from calibration_pipeline.solvers import TwelveDofV2Solver
from calibration_pipeline.v2_backend.information import (
    covariance_from_jacobian,
    scaled_jacobian,
)


def test_twelve_dof_v2_recovers_noise_free_scene():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=8)
    nominal_rotation = scene.handeye_rotation @ so3_exp(
        np.deg2rad(np.array([1.0, -2.0, 1.5]))
    )
    nominal_translation = scene.handeye_translation + np.array([0.008, -0.005, 0.006])
    result = TwelveDofV2Solver().solve(
        poses,
        measurements,
        nominal_rotation,
        nominal_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    assert result.converged
    assert result.diagnostics.rank == 9
    assert rotation_distance_deg(
        result.estimate.handeye_rotation, scene.handeye_rotation
    ) < 1e-5
    assert np.linalg.norm(
        result.estimate.handeye_translation - scene.handeye_translation
    ) < 1e-7
    assert np.linalg.norm(result.estimate.board.corner - scene.board.corner) < 1e-7
    endpoint_u_offsets = []
    endpoint_v_offsets = []
    for pose, measurement in zip(poses, measurements):
        sensor_rotation = pose.rotation @ result.estimate.handeye_rotation
        sensor_translation = (
            pose.translation + pose.rotation @ result.estimate.handeye_translation
        )
        endpoint_u_offsets.append(
            sensor_rotation @ measurement.endpoint_u
            + sensor_translation
            - result.estimate.board.corner
        )
        endpoint_v_offsets.append(
            sensor_rotation @ measurement.endpoint_v
            + sensor_translation
            - result.estimate.board.corner
        )
    assert np.median(np.asarray(endpoint_u_offsets) @ result.estimate.board.u) > 0.0
    assert np.median(np.asarray(endpoint_v_offsets) @ result.estimate.board.v) > 0.0


def test_solver_rejects_too_few_poses():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=3)
    try:
        TwelveDofV2Solver().solve(
            poses,
            measurements,
            scene.handeye_rotation,
            scene.handeye_translation,
            board_dimensions=(0.4, 0.5),
        )
    except ValueError as error:
        assert "at least four" in str(error)
    else:
        raise AssertionError("solver accepted an underconstrained dataset")


def test_dimensionless_jacobian_is_invariant_to_metre_or_millimetre_units():
    rng = np.random.default_rng(4)
    jacobian_m = rng.normal(size=(30, 9))
    scale_m = np.array([0.2] * 3 + [0.1] * 3 + [0.2] * 3)
    jacobian_mm = jacobian_m.copy()
    jacobian_mm[:, 3:6] /= 1000.0
    scale_mm = scale_m.copy()
    scale_mm[3:6] *= 1000.0
    assert np.allclose(
        scaled_jacobian(jacobian_m, scale_m),
        scaled_jacobian(jacobian_mm, scale_mm),
    )


def test_projected_corner_is_counted_in_residual_variance_dof():
    jacobian = np.eye(9)
    jacobian = np.vstack((jacobian, np.ones((6, 9))))
    residual = np.ones(15) * 0.1
    _, variance = covariance_from_jacobian(
        jacobian,
        residual,
        fitted_nuisance_parameters=3,
    )
    # 15 residuals - 9 nonlinear states - 3 projected corner coordinates.
    assert np.isclose(variance, float(residual @ residual) / 3.0)
