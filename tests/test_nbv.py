import numpy as np

from calibration_pipeline.models import CalibrationEstimate
from calibration_pipeline.nbv import generate_candidates, predict_candidate, score_candidates
from calibration_pipeline.simulation.synthetic import default_scene, generate_seed_dataset
from calibration_pipeline.solvers import TwelveDofV2Solver


def _calibrated_fixture():
    scene = default_scene()
    poses, measurements = generate_seed_dataset(scene, count=8)
    result = TwelveDofV2Solver().solve(
        poses,
        measurements,
        scene.handeye_rotation,
        scene.handeye_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    return scene, poses, measurements, result


def test_generated_candidate_hits_only_target_adjacent_edges():
    scene, _, _, result = _calibrated_fixture()
    candidates = generate_candidates(
        result.estimate,
        edge_samples=2,
        alphas_deg=(35.0,),
        psis_deg=(0.0,),
        working_distances=(0.55,),
    )
    predictions = [predict_candidate(candidate, result.estimate, scene.roi) for candidate in candidates]
    valid = [prediction for prediction in predictions if prediction.valid]
    assert valid
    assert all(prediction.edge_labels == ("u0", "v0") or set(prediction.edge_labels) == {"u0", "v0"} for prediction in valid)


def test_information_scoring_recomputes_augmented_varproj_model():
    scene, poses, measurements, result = _calibrated_fixture()
    candidates = generate_candidates(
        result.estimate,
        edge_samples=2,
        alphas_deg=(35.0,),
        psis_deg=(0.0,),
        working_distances=(0.55,),
    )
    # Avoid an unrealistically singular zero covariance in the exact synthetic case.
    estimate = CalibrationEstimate(
        result.estimate.handeye_rotation,
        result.estimate.handeye_translation,
        result.estimate.board,
        result.estimate.x9,
        np.eye(9) * 1e-10,
    )
    synthetic_result = type("Result", (), {"estimate": estimate})()
    scored = score_candidates(
        candidates,
        synthetic_result,
        poses,
        measurements,
        scene.roi,
        maximum_candidates=4,
        minimum_valid_probability=0.5,
    )
    assert scored
    assert np.isfinite(scored[0].information_gain)
