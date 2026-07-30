import numpy as np

from calibration_pipeline.perception import (
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)
from calibration_pipeline.simulation import compute_fov_plate_scanline


def make_profile(
    start=(-0.04, 0.34),
    stop=(0.06, 0.47),
    *,
    count=401,
    noise_std=0.0,
    seed=3,
):
    xz = np.linspace(np.asarray(start), np.asarray(stop), count)
    if noise_std:
        xz += np.random.default_rng(seed).normal(
            0.0, noise_std, size=xz.shape
        )
    return np.column_stack((xz[:, 0], np.zeros(count), xz[:, 1]))


def test_detector_recovers_ideal_profile_boundaries():
    profile = make_profile()
    detector = ProfileEndpointDetector()

    result = detector.detect(profile)

    assert result is not None
    pitch = np.linalg.norm(profile[1] - profile[0])
    # The detector estimates the physical sample-cell boundary, half a sample
    # beyond the first and last valid return.
    expected_first = profile[0] - 0.5 * (profile[1] - profile[0])
    expected_second = profile[-1] + 0.5 * (profile[-1] - profile[-2])
    assert np.linalg.norm(result.first - expected_first) < pitch * 0.05
    assert np.linalg.norm(result.second - expected_second) < pitch * 0.05
    assert result.support_count == len(profile)
    assert result.confidence > 0.95


def test_detector_rejects_sparse_outliers_and_random_dropouts():
    rng = np.random.default_rng(19)
    clean = make_profile(noise_std=5.5e-5)
    keep = rng.random(len(clean)) > 0.08
    profile = clean[keep]
    outlier_indices = rng.choice(len(profile), size=5, replace=False)
    profile[outlier_indices, 2] += rng.normal(0.0, 0.003, size=5)
    detector = ProfileEndpointDetector()

    result = detector.detect(profile)

    assert result is not None
    assert result.support_count >= 0.9 * len(profile)
    assert np.linalg.norm(result.first - clean[0]) < 0.001
    assert np.linalg.norm(result.second - clean[-1]) < 0.001
    assert result.residual_rms_m < 0.0002


def test_detector_selects_supported_board_segment_across_depth_gaps():
    rng = np.random.default_rng(23)
    prefix = make_profile(
        (-0.09, 0.58), (-0.075, 0.58), count=25, noise_std=4e-5, seed=5
    )
    board = make_profile(noise_std=5.5e-5, seed=7)
    suffix = make_profile(
        (0.075, 0.62), (0.09, 0.62), count=25, noise_std=4e-5, seed=9
    )
    profile = np.vstack((prefix, board, suffix))
    profile += rng.normal(0.0, 5e-6, size=profile.shape)
    detector = ProfileEndpointDetector()

    result = detector.detect(profile)

    assert result is not None
    assert result.support_count > 350
    assert np.linalg.norm(result.first - board[0]) < 0.001
    assert np.linalg.norm(result.second - board[-1]) < 0.001


def test_detector_reports_reason_for_insufficient_profile():
    detector = ProfileEndpointDetector()

    assert detector.detect(np.zeros((5, 3))) is None
    assert detector.last_rejection_reason == "insufficient_finite_points"


def test_detector_configuration_rejects_invalid_ranges():
    try:
        EndpointDetectionConfig(
            minimum_segment_length_m=0.2,
            maximum_segment_length_m=0.1,
        )
    except ValueError as error:
        assert "maximum_segment_length_m" in str(error)
    else:
        raise AssertionError("invalid detector configuration was accepted")


def test_detector_recovers_hidden_simulation_endpoints_without_truth_input():
    rotation_sensor_base = np.array(
        [
            [-0.366, 0.817, -0.446],
            [0.815, 0.513, 0.270],
            [0.450, -0.265, -0.853],
        ]
    )
    result = compute_fov_plate_scanline(
        rotation_sensor_base=rotation_sensor_base,
        translation_sensor_base=np.array([0.884, -0.057, 0.520]),
        corner=np.array([0.7, 0.0, 0.25]),
        normal=np.array([0.0, 0.0, 1.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        width=0.4,
        height=0.5,
    )
    detector = ProfileEndpointDetector(
        EndpointDetectionConfig(maximum_segment_length_m=0.8)
    )

    detection = detector.detect(result["scan_pts_S"])

    assert detection is not None
    truth = [point for _, point in result["endpoints_S"]]
    assert len(truth) == 2
    direct = (
        np.linalg.norm(detection.first - truth[0])
        + np.linalg.norm(detection.second - truth[1])
    )
    swapped = (
        np.linalg.norm(detection.first - truth[1])
        + np.linalg.norm(detection.second - truth[0])
    )
    assert min(direct, swapped) / 2.0 < 0.0005
