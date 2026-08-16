from dataclasses import replace

import numpy as np

from calibration_pipeline.geometry import rotation_distance_deg, so3_exp
from calibration_pipeline.models import FlangePose
from calibration_pipeline.dataset_io import result_payload
from calibration_pipeline.research import SharedShapeHandEyeSolver, SurfaceBasis
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset
from calibration_pipeline.solvers import TwelveDofV2Solver
from calibration_pipeline.nbv import generate_candidates, predict_candidate, score_candidates


def test_surface_bases_do_not_duplicate_constant_or_linear_plane_modes():
    grid = np.linspace(0.0, 1.0, 81)
    xi, eta = np.meshgrid(grid, grid, indexing="ij")
    xi = xi.reshape(-1)
    eta = eta.reshape(-1)
    plane = np.column_stack((np.ones_like(xi), xi, eta))
    for basis in (SurfaceBasis("matched"), SurfaceBasis("legendre", degree=4)):
        augmented = np.column_stack((plane, basis.evaluate(xi, eta)))
        assert np.linalg.matrix_rank(augmented) == augmented.shape[1]


def test_shared_shape_solver_preserves_noise_free_handeye_solution():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=10, seed=13)
    nominal_rotation = scene.handeye_rotation @ so3_exp(
        np.deg2rad(np.array([1.0, -2.0, 1.5]))
    )
    nominal_translation = scene.handeye_translation + np.array([0.008, -0.005, 0.006])
    flat = TwelveDofV2Solver().solve(
        poses,
        measurements,
        nominal_rotation,
        nominal_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    result = SharedShapeHandEyeSolver(SurfaceBasis("legendre", degree=3)).solve(
        poses,
        measurements,
        flat,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    assert result.converged
    assert rotation_distance_deg(result.handeye_rotation, scene.handeye_rotation) < 1e-5
    assert np.linalg.norm(result.handeye_translation - scene.handeye_translation) < 1e-7
    assert np.linalg.norm(result.shape_coefficients) < 1e-7


def test_production_v2_shared_mode_preserves_noise_free_solution_and_state():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=8, seed=17)
    solver = TwelveDofV2Solver(
        surface_model="shared",
        surface_basis_kind="legendre",
        surface_degree=4,
    )
    result = solver.solve(
        poses,
        measurements,
        scene.handeye_rotation,
        scene.handeye_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )

    assert result.converged
    assert result.estimate.surface_model == "shared"
    assert result.diagnostics.rank == len(result.estimate.state)
    assert result.diagnostics.prior_augmented_rank == len(
        result.estimate.state
    )
    assert (
        result.diagnostics.condition_number
        > result.diagnostics.prior_augmented_condition_number
    )
    assert result.diagnostics.state_information.shape == (
        len(result.estimate.state),
        len(result.estimate.state),
    )
    assert rotation_distance_deg(
        result.estimate.handeye_rotation, scene.handeye_rotation
    ) < 1e-5
    assert np.linalg.norm(
        result.estimate.handeye_translation - scene.handeye_translation
    ) < 1e-7
    assert result.diagnostics.surface_rms_m < 1e-7
    payload = result_payload(result)
    assert payload["schema_version"] == 2
    assert payload["surface"]["model"] == "shared"
    assert len(payload["surface"]["coefficients_m"]) == 12
    assert payload["diagnostics"]["data_only"]["rank"] == 24
    assert payload["diagnostics"]["prior_augmented"]["rank"] == 24
    assert (
        payload["diagnostics"]["data_only"]["condition_number"]
        > payload["diagnostics"]["prior_augmented"]["condition_number"]
    )


def test_shared_shape_is_used_by_future_profile_and_information_score():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=8, seed=19)
    solver = TwelveDofV2Solver(surface_model="shared", surface_degree=4)
    result = solver.solve(
        poses,
        measurements,
        scene.handeye_rotation,
        scene.handeye_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    state = result.estimate.state.copy()
    state[12] = 0.0004
    estimate = replace(
        result.estimate,
        state=state,
        shape_coefficients=state[12:],
    )
    shaped_result = replace(result, estimate=estimate)
    candidates = generate_candidates(
        estimate,
        roi=scene.roi,
        edge_samples=2,
        alphas_deg=(35.0,),
        psis_deg=(0.0,),
        working_distances=(0.55,),
    )
    predictions = [
        (candidate, predict_candidate(candidate, estimate, scene.roi))
        for candidate in candidates
    ]
    candidate, prediction = next(
        item for item in predictions if item[1].valid
    )
    assert prediction.measurement is not None
    sensor_transform = candidate.flange_transform_command @ estimate.handeye_transform
    points_base = (
        sensor_transform[:3, :3]
        @ prediction.measurement.profile_points.T
    ).T + sensor_transform[:3, 3]
    delta = points_base - estimate.board.corner
    xi = (delta @ estimate.board.u) / estimate.board.length_u
    eta = (delta @ estimate.board.v) / estimate.board.length_v
    expected_height = solver.surface_basis.height(
        xi, eta, estimate.shape_coefficients
    )
    assert np.max(
        np.abs(delta @ estimate.board.normal - expected_height)
    ) < 1e-8

    scored = score_candidates(
        [candidate],
        shaped_result,
        poses,
        measurements,
        scene.roi,
        minimum_valid_probability=0.0,
        solver=solver,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    assert scored
    assert np.isfinite(scored[0].information_gain)

    flange = candidate.flange_transform_command
    rolling = solver.solve(
        poses + [FlangePose(flange[:3, :3], flange[:3, 3])],
        measurements + [prediction.measurement],
        estimate.handeye_rotation,
        estimate.handeye_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
        initial_board_rotation=estimate.board.rotation,
        initial_estimate=estimate,
    )
    assert rolling.converged
    assert rolling.estimate.surface_model == "shared"
    assert len(rolling.estimate.state) == len(estimate.state)
