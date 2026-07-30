import numpy as np

from calibration_pipeline.geometry import transform_point
from calibration_pipeline.models import CalibrationEstimate, SensorROI, TrapezoidDomain
from calibration_pipeline.nbv import (
    StopPolicy,
    generate_candidates,
    predict_candidate,
    score_candidates,
)
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


def test_candidate_matches_v5_closed_form_virtual_endpoints():
    scene, _, _, result = _calibrated_fixture()
    candidate = generate_candidates(
        result.estimate,
        roi=scene.roi,
        edge_samples=2,
        alphas_deg=(35.0,),
        psis_deg=(12.0,),
        working_distances=(0.55,),
    )[0]
    measurement = candidate.virtual_measurement
    assert measurement is not None
    profile_length = np.hypot(candidate.a, candidate.b)
    psi = candidate.psi
    expected_u = np.array(
        [
            -0.5 * profile_length * np.cos(psi),
            0.0,
            candidate.working_distance + 0.5 * profile_length * np.sin(psi),
        ]
    )
    expected_v = np.array(
        [
            0.5 * profile_length * np.cos(psi),
            0.0,
            candidate.working_distance - 0.5 * profile_length * np.sin(psi),
        ]
    )
    assert np.allclose(measurement.endpoint_u, expected_u, atol=1e-10)
    assert np.allclose(measurement.endpoint_v, expected_v, atol=1e-10)
    assert np.allclose(
        transform_point(candidate.sensor_transform_nominal, measurement.endpoint_u),
        result.estimate.board.corner + candidate.a * result.estimate.board.u,
        atol=1e-10,
    )


def test_asymmetric_safe_trapezoid_uses_metric_sensor_coordinates():
    roi = SensorROI(
        hard_domain=TrapezoidDomain(0.2, 0.8, -0.2, -0.3, 0.1, 0.25),
        safe_domain=TrapezoidDomain(0.3, 0.7, -0.15, -0.22, 0.07, 0.18),
    )
    assert roi.contains(np.array([0.06, 0.0, 0.3]))
    assert not roi.contains(np.array([0.08, 0.0, 0.3]))
    assert roi.contains(np.array([0.08, 0.0, 0.3]), safe=False)


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


def test_virtual_information_matches_configured_measurement_batch():
    scene, poses, measurements, result = _calibrated_fixture()
    candidates = generate_candidates(
        result.estimate,
        edge_samples=2,
        alphas_deg=(35.0,),
        psis_deg=(0.0,),
        working_distances=(0.55,),
    )
    estimate = CalibrationEstimate(
        result.estimate.handeye_rotation,
        result.estimate.handeye_translation,
        result.estimate.board,
        result.estimate.x9,
        np.eye(9) * 1e-10,
    )
    synthetic_result = type("Result", (), {"estimate": estimate})()
    one = score_candidates(
        candidates[:1],
        synthetic_result,
        poses,
        measurements,
        scene.roi,
        minimum_valid_probability=0.5,
        virtual_batch_size=1,
    )
    five = score_candidates(
        candidates[:1],
        synthetic_result,
        poses,
        measurements,
        scene.roi,
        minimum_valid_probability=0.5,
        virtual_batch_size=5,
    )
    assert one and five
    assert five[0].information_gain > one[0].information_gain


def test_stop_policy_accepts_physical_covariance_target_without_low_gain():
    policy = StopPolicy(
        minimum_nbv_poses=1,
        consecutive_low_gain_limit=3,
        information_gain_threshold=1e-6,
        minimum_effective_eigenvalue=1e-8,
    )
    stop, _ = policy.evaluate(
        total_poses=7,
        nbv_poses=1,
        effective_rank=6,
        best_information_gain=10.0,
        minimum_effective_eigenvalue=1.0,
        handeye_covariance=None,
    )
    assert not stop
    stop, reason = policy.evaluate(
        total_poses=7,
        nbv_poses=1,
        effective_rank=6,
        best_information_gain=10.0,
        minimum_effective_eigenvalue=1.0,
        handeye_covariance=np.diag(
            [np.deg2rad(0.01) ** 2] * 3 + [1e-4**2] * 3
        ),
    )
    assert stop
    assert reason == "hand-eye uncertainty target reached"


def test_stop_policy_accepts_saturated_gain_without_covariance_target():
    policy = StopPolicy(
        minimum_nbv_poses=1,
        consecutive_low_gain_limit=2,
        information_gain_threshold=1.0,
        minimum_effective_eigenvalue=1e-8,
    )
    for _ in range(2):
        stop, reason = policy.evaluate(
            total_poses=7,
            nbv_poses=1,
            effective_rank=6,
            best_information_gain=0.0,
            minimum_effective_eigenvalue=1.0,
            handeye_covariance=None,
        )
    assert stop
    assert reason == "information gain saturated"


def test_stop_policy_accepts_truth_independent_validation_plateau():
    policy = StopPolicy(
        minimum_nbv_poses=3,
        maximum_total_poses=20,
        minimum_effective_eigenvalue=1e-8,
    )
    stop, reason = policy.evaluate(
        total_poses=9,
        nbv_poses=3,
        effective_rank=6,
        best_information_gain=1.0,
        minimum_effective_eigenvalue=1.0,
        handeye_covariance=None,
        validation_plateaued=True,
    )
    assert stop
    assert reason == "held-out validation score plateaued"
