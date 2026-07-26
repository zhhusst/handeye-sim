import numpy as np

from calibration_pipeline.geometry import rotation_distance_deg, so3_exp
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset
from calibration_pipeline.solvers import TwelveDofV2Solver


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
