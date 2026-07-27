import numpy as np
import pytest

from calibration_pipeline.models import (
    FlangePose,
    Measurement,
    SensorROI,
    TrapezoidDomain,
)
from calibration_pipeline.nbv import generate_candidates, predict_candidate
from calibration_pipeline.pipeline import ActiveCalibrationPipeline, PipelineStage
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset


def test_candidate_budget_covers_the_full_grid_instead_of_prefix_only():
    values = list(range(101))
    subset = ActiveCalibrationPipeline._uniform_candidate_subset(values, 9)
    assert len(subset) == 9
    assert subset[0] == 0
    assert subset[-1] == 100
    assert len({value // 25 for value in subset}) >= 4


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
