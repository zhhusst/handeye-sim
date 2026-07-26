from calibration_pipeline.pipeline import ActiveCalibrationPipeline, PipelineStage
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset


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
