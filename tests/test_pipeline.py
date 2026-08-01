from dataclasses import replace

import numpy as np
import pytest

from calibration_pipeline.geometry import so3_exp, so3_log
from calibration_pipeline.models import (
    FlangePose,
    Measurement,
    SensorROI,
    TrapezoidDomain,
)
from calibration_pipeline.nbv import generate_candidates, predict_candidate
from calibration_pipeline.pipeline import ActiveCalibrationPipeline, PipelineStage
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset
from handeye_sim_bridge.active_calibration_sim_node import (
    effective_measurement_timeout_s,
)


def test_candidate_budget_covers_the_full_grid_instead_of_prefix_only():
    values = list(range(101))
    subset = ActiveCalibrationPipeline._uniform_candidate_subset(values, 9)
    assert len(subset) == 9
    assert subset[0] == 0
    assert subset[-1] == 100
    assert len({value // 25 for value in subset}) >= 4


def test_measurement_timeout_scales_with_large_batches():
    assert effective_measurement_timeout_s(5.0, 5, 2.0) == 5.0
    assert effective_measurement_timeout_s(5.0, 20, 2.0) == 11.0


def test_pipeline_reaches_active_nbv_after_required_seeds():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=6)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=6,
    )
    for pose, measurement in zip(poses, measurements):
        pipeline.append_seed(pose, measurement)
    assert pipeline.stage is PipelineStage.INITIALIZE_12DOF_V2
    result = pipeline.initialize()
    assert result.converged
    assert pipeline.stage is PipelineStage.ACTIVE_NBV


def test_seed_batch_adds_many_observations_but_one_physical_seed():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=8)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=2,
    )
    pipeline.append_seed_batch(poses[:4], measurements[:4])
    assert pipeline.seed_count == 1
    assert len(pipeline.poses) == 4
    assert pipeline.stage is PipelineStage.COLLECT_SEEDS
    pipeline.append_seed_batch(poses[4:], measurements[4:])
    assert pipeline.seed_count == 2
    assert len(pipeline.poses) == 8
    assert pipeline.stage is PipelineStage.INITIALIZE_12DOF_V2


def test_failed_nbv_trial_does_not_mutate_dataset():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=6)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=6,
        maximum_update_rotation_deg=1e-12,
        maximum_update_translation_m=1e-12,
        maximum_board_rotation_deg=1e-12,
    )
    for pose, measurement in zip(poses, measurements):
        pipeline.append_seed(pose, measurement)
    result = pipeline.initialize()
    candidate = generate_candidates(
        result.estimate,
        roi=scene.roi,
        edge_samples=2,
        alphas_deg=(35.0,),
        psis_deg=(0.0,),
        working_distances=(0.55,),
    )[0]
    prediction = predict_candidate(candidate, result.estimate, scene.roi)
    assert prediction.measurement is not None
    measurement = prediction.measurement
    perturbed = Measurement(
        measurement.profile_points + np.array([0.0, 0.0, 2e-4]),
        measurement.endpoint_u + np.array([1e-4, 0.0, 0.0]),
        measurement.endpoint_v + np.array([-1e-4, 0.0, 0.0]),
    )
    flange = candidate.flange_transform_command
    before = (len(pipeline.poses), len(pipeline.measurements), pipeline.nbv_count)
    with pytest.raises(RuntimeError, match="transactional update jump"):
        pipeline.append_nbv(
            FlangePose(flange[:3, :3], flange[:3, 3]),
            perturbed,
        )
    assert (len(pipeline.poses), len(pipeline.measurements), pipeline.nbv_count) == before


def test_first_nbv_may_use_a_wider_correction_bound_only_once():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=6)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=6,
        maximum_update_rotation_deg=5.0,
        initial_maximum_update_rotation_deg=10.0,
    )
    for pose, measurement in zip(poses, measurements):
        pipeline.append_seed(pose, measurement)
    initialized = pipeline.initialize()

    class FixedSolver:
        def __init__(self, result):
            self.result = result

        def solve(self, *_args, **_kwargs):
            return self.result

    first_estimate = replace(
        initialized.estimate,
        handeye_rotation=(
            initialized.estimate.handeye_rotation
            @ so3_exp(np.deg2rad(np.array([8.0, 0.0, 0.0])))
        ),
    )
    fixed_solver = FixedSolver(replace(initialized, estimate=first_estimate))
    pipeline.solver = fixed_solver
    pipeline.append_nbv(poses[0], measurements[0])
    assert pipeline.nbv_count == 1

    second_estimate = replace(
        first_estimate,
        handeye_rotation=(
            first_estimate.handeye_rotation
            @ so3_exp(np.deg2rad(np.array([8.0, 0.0, 0.0])))
        ),
    )
    fixed_solver.result = replace(initialized, estimate=second_estimate)
    with pytest.raises(RuntimeError, match="rolling update"):
        pipeline.append_nbv(poses[0], measurements[0])
    assert pipeline.nbv_count == 1


def test_real_nbv_may_cross_safe_boundary_but_must_remain_in_hard_domain():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=6)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=6,
    )
    for pose, measurement in zip(poses, measurements):
        pipeline.append_seed(pose, measurement)
    result = pipeline.initialize()
    pipeline.roi = SensorROI(
        hard_domain=TrapezoidDomain(0.2, 0.8, -0.2, -0.3, 0.1, 0.25),
        safe_domain=TrapezoidDomain(0.3, 0.7, -0.15, -0.22, 0.07, 0.18),
    )
    endpoint_u = np.array([0.08, 0.0, 0.3])
    endpoint_v = np.array([0.0, 0.0, 0.4])
    assert not pipeline.roi.contains(endpoint_u)
    assert pipeline.roi.contains(endpoint_u, safe=False)

    class FixedSolver:
        def solve(self, *_args, **_kwargs):
            return result

    pipeline.solver = FixedSolver()
    accepted = pipeline.append_nbv(
        poses[0],
        Measurement(
            np.vstack((endpoint_u, endpoint_v)),
            endpoint_u,
            endpoint_v,
        ),
        candidate_id="observed_candidate",
    )
    assert accepted is result
    assert pipeline.nbv_count == 1
    assert "observed_candidate" in pipeline.failed_candidates


def test_synchronized_nbv_batch_counts_as_one_physical_pose():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=8)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=6,
    )
    for pose, measurement in zip(poses[:6], measurements[:6]):
        pipeline.append_seed(pose, measurement)
    pipeline.initialize()

    result = pipeline.append_nbv_batch(
        [poses[6], poses[7]],
        [measurements[6], measurements[7]],
        candidate_id="candidate_batch",
    )

    assert result.converged
    assert pipeline.nbv_count == 1
    assert len(pipeline.poses) == 8
    assert "candidate_batch" in pipeline.failed_candidates


def test_held_out_geometry_is_diagnostic_and_latest_result_is_retained():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=6)
    pipeline = ActiveCalibrationPipeline(
        scene.handeye_rotation,
        scene.handeye_translation,
        (scene.board.length_u, scene.board.length_v),
        roi=scene.roi,
        minimum_seed_poses=6,
        maximum_update_rotation_deg=10.0,
        maximum_update_translation_m=0.1,
    )
    for pose, measurement in zip(poses, measurements):
        pipeline.append_seed(pose, measurement)
        pipeline.append_validation_observation(pose, measurement)
    initialized = pipeline.initialize()
    initial_score = pipeline.current_validation_metrics.score_m

    bad_rotation = initialized.estimate.handeye_rotation @ so3_exp(
        np.deg2rad(np.array([2.0, 0.0, 0.0]))
    )
    bad_translation = (
        initialized.estimate.handeye_translation
        + np.array([0.002, -0.001, 0.001])
    )
    bad_x9 = initialized.estimate.x9.copy()
    bad_x9[:3] = so3_log(bad_rotation)
    bad_x9[3:6] = bad_translation
    bad_estimate = replace(
        initialized.estimate,
        handeye_rotation=bad_rotation,
        handeye_translation=bad_translation,
        x9=bad_x9,
    )

    class FixedSolver:
        def solve(self, *_args, **_kwargs):
            return replace(initialized, estimate=bad_estimate)

    pipeline.solver = FixedSolver()
    pipeline.append_nbv(poses[0], measurements[0])

    assert pipeline.current_validation_metrics.score_m > initial_score
    assert np.allclose(
        pipeline.result.estimate.handeye_rotation,
        bad_rotation,
    )
