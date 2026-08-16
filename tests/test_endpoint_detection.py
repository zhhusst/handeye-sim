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


def test_detector_uses_local_tangents_for_a_gently_curved_profile():
    parameter = np.linspace(0.0, 0.20, 801)
    x = -0.05 + parameter
    z = (
        0.40
        + 0.35 * parameter
        + 0.0005 * np.sin(np.pi * parameter / 0.20)
        + 0.0002 * np.sin(2.0 * np.pi * parameter / 0.20)
    )
    profile = np.column_stack((x, np.zeros_like(x), z))
    detector = ProfileEndpointDetector(
        EndpointDetectionConfig(maximum_segment_length_m=0.8)
    )

    result = detector.detect(profile)

    assert result is not None
    pitch = np.linalg.norm(profile[1] - profile[0])
    first_direction = (profile[1] - profile[0]) / np.linalg.norm(
        profile[1] - profile[0]
    )
    second_direction = (profile[-1] - profile[-2]) / np.linalg.norm(
        profile[-1] - profile[-2]
    )
    expected_first = profile[0] - 0.5 * pitch * first_direction
    expected_second = profile[-1] + 0.5 * pitch * second_direction
    assert np.linalg.norm(result.first - expected_first) < 3e-5
    assert np.linalg.norm(result.second - expected_second) < 3e-5


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


def _target_surface_config():
    return EndpointDetectionConfig(
        minimum_segment_length_m=0.10,
        maximum_segment_length_m=0.14,
        maximum_residual_rms_m=0.00015,
        smoothing_window=5,
        local_fit_window=16,
        angle_change_threshold_deg=10.0,
        height_jump_threshold_m=0.0002,
        breakpoint_cluster_points=12,
    )


def test_detector_extracts_thin_plate_between_small_parallel_height_steps():
    rng = np.random.default_rng(20260814)
    x = np.linspace(-0.15, 0.15, 1201)
    z = 0.30 + 0.15 * x
    on_plate = (x >= -0.06) & (x <= 0.06)
    z[on_plate] -= 0.0008
    z += rng.normal(0.0, 0.00003, len(z))
    profile = np.column_stack((x, np.zeros(len(x)), z))
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect(profile)

    assert result is not None
    assert result.selection_mode == "change_point"
    assert result.breakpoint_count == 2
    assert len(result.surface_points) == result.support_count
    assert np.min(result.surface_points[:, 0]) > -0.061
    assert np.max(result.surface_points[:, 0]) < 0.061
    assert abs(result.first[0] + 0.06) < 0.0005
    assert abs(result.second[0] - 0.06) < 0.0005


def test_detector_rejects_thick_side_walls_and_selects_top_surface():
    rng = np.random.default_rng(8)

    def segment(x0, z0, x1, z1, count, *, endpoint=False):
        x = np.linspace(x0, x1, count, endpoint=endpoint)
        z = np.linspace(z0, z1, count, endpoint=endpoint)
        return np.column_stack((x, np.zeros(count), z))

    profile = np.vstack(
        (
            segment(-0.15, 0.30, -0.06, 0.30, 360),
            segment(-0.06, 0.30, -0.06, 0.285, 60),
            segment(-0.06, 0.285, 0.06, 0.285, 480),
            segment(0.06, 0.285, 0.06, 0.30, 60),
            segment(0.06, 0.30, 0.15, 0.30, 361, endpoint=True),
        )
    )
    profile[:, (0, 2)] += rng.normal(
        0.0, 0.00002, (len(profile), 2)
    )
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect(profile)

    assert result is not None
    assert result.selection_mode == "change_point"
    assert len(result.surface_points) > 400
    assert np.ptp(result.surface_points[:, 0]) > 0.11
    assert np.ptp(result.surface_points[:, 2]) < 0.0002
    assert abs(result.first[0] + 0.06) < 0.001
    assert abs(result.second[0] - 0.06) < 0.001


def test_guided_detector_selects_target_instead_of_longer_parallel_table():
    rng = np.random.default_rng(91)
    table_left_x = np.linspace(-0.18, -0.05, 520, endpoint=False)
    target_x = np.linspace(-0.05, 0.05, 400, endpoint=False)
    table_right_x = np.linspace(0.05, 0.18, 521)
    profile_x = np.concatenate((table_left_x, target_x, table_right_x))
    profile_z = np.concatenate(
        (
            np.full(len(table_left_x), 0.3000),
            np.full(len(target_x), 0.2992),
            np.full(len(table_right_x), 0.3000),
        )
    )
    profile_z += rng.normal(0.0, 0.000025, len(profile_z))
    profile = np.column_stack(
        (profile_x, np.zeros(len(profile_x)), profile_z)
    )
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect_guided(
        profile,
        np.array([-0.05, 0.0, 0.2992]),
        np.array([0.05, 0.0, 0.2992]),
        normal_gate_m=0.00045,
        endpoint_gate_m=0.008,
        maximum_angle_difference_deg=8.0,
        selection_mode="guided_align",
    )

    assert result is not None
    assert result.selection_mode == "guided_align"
    assert abs(result.first[0] + 0.05) < 0.001
    assert abs(result.second[0] - 0.05) < 0.001
    assert np.max(result.surface_points[:, 2]) < 0.2995


def test_guided_detector_does_not_jump_to_unrelated_line_when_target_is_lost():
    x = np.linspace(-0.18, 0.18, 1201)
    table = np.column_stack(
        (x, np.zeros(len(x)), np.full(len(x), 0.3000))
    )
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect_guided(
        table,
        np.array([-0.05, 0.0, 0.2900]),
        np.array([0.05, 0.0, 0.2900]),
        normal_gate_m=0.001,
        endpoint_gate_m=0.01,
        selection_mode="guided_track",
    )

    assert result is None
    assert detector.last_rejection_reason in {
        "guided_roi_has_too_few_points",
        "guided_target_not_found",
    }


def test_wide_guided_recovery_resplits_top_surface_and_side_walls():
    """A wider recall ROI must not merge several physical surfaces."""

    def segment(x0, z0, x1, z1, count, *, endpoint=False):
        x = np.linspace(x0, x1, count, endpoint=endpoint)
        z = np.linspace(z0, z1, count, endpoint=endpoint)
        return np.column_stack((x, np.zeros(count), z))

    profile = np.vstack(
        (
            segment(-0.15, 0.30, -0.06, 0.30, 360),
            segment(-0.06, 0.30, -0.06, 0.285, 60),
            segment(-0.06, 0.285, 0.06, 0.285, 480),
            segment(0.06, 0.285, 0.06, 0.30, 60),
            segment(0.06, 0.30, 0.15, 0.30, 361, endpoint=True),
        )
    )
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect_guided(
        profile,
        np.array([-0.06, 0.0, 0.285]),
        np.array([0.06, 0.0, 0.285]),
        # This deliberately includes both side walls and nearby table points.
        normal_gate_m=0.020,
        endpoint_gate_m=0.040,
        maximum_angle_difference_deg=35.0,
        selection_mode="local_reacquire",
    )

    assert result is not None
    assert result.selection_mode == "local_reacquire"
    assert abs(result.first[0] + 0.06) < 0.001
    assert abs(result.second[0] - 0.06) < 0.001
    # At most one acquisition cell from each adjoining wall is retained at
    # the fitted physical boundary; the wall itself is not merged.
    assert np.ptp(result.surface_points[:, 2]) < 0.0003


def test_temporal_breakpoint_pair_tracks_shifted_top_between_physical_edges():
    rng = np.random.default_rng(41)

    def segment(x0, z0, x1, z1, count, *, endpoint=False):
        x = np.linspace(x0, x1, count, endpoint=endpoint)
        z = np.linspace(z0, z1, count, endpoint=endpoint)
        return np.column_stack((x, np.zeros(count), z))

    shift = np.array([0.0018, 0.0, -0.0012])
    profile = np.vstack(
        (
            segment(-0.15, 0.30, -0.06, 0.30, 360),
            segment(-0.06, 0.30, -0.06, 0.285, 60),
            segment(-0.06, 0.285, 0.06, 0.285, 480),
            segment(0.06, 0.285, 0.06, 0.30, 60),
            segment(0.06, 0.30, 0.15, 0.30, 361, endpoint=True),
        )
    )
    profile += shift
    profile[:, (0, 2)] += rng.normal(0.0, 0.00002, (len(profile), 2))
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect_temporal_breakpoint_pair(
        profile,
        np.array([-0.06, 0.0, 0.285]) + shift,
        np.array([0.06, 0.0, 0.285]) + shift,
        endpoint_gate_m=0.004,
        normal_gate_m=0.002,
        maximum_angle_difference_deg=10.0,
    )

    assert result is not None
    assert result.selection_mode == "seed_temporal_track"
    assert abs(result.first[0] - (-0.06 + shift[0])) < 0.001
    assert abs(result.second[0] - (0.06 + shift[0])) < 0.001
    assert np.ptp(result.surface_points[:, 2]) < 0.00035


def test_temporal_breakpoint_pair_does_not_use_roi_crop_as_fake_endpoints():
    profile = make_profile((-0.15, 0.30), (0.15, 0.30), count=1201)
    detector = ProfileEndpointDetector(_target_surface_config())

    result = detector.detect_temporal_breakpoint_pair(
        profile,
        np.array([-0.05, 0.0, 0.30]),
        np.array([0.05, 0.0, 0.30]),
        endpoint_gate_m=0.005,
        normal_gate_m=0.002,
    )

    assert result is None
    assert detector.last_rejection_reason == "tracked_breakpoint_pair_not_found"


def test_temporal_pair_spans_internal_false_change_point_between_real_edges():
    def segment(x0, z0, x1, z1, count, *, endpoint=False):
        x = np.linspace(x0, x1, count, endpoint=endpoint)
        z = np.linspace(z0, z1, count, endpoint=endpoint)
        return np.column_stack((x, np.zeros(count), z))

    profile = np.vstack(
        (
            segment(-0.15, 0.30, -0.06, 0.30, 360),
            segment(-0.06, 0.30, -0.06, 0.285, 60),
            segment(-0.06, 0.285, 0.06, 0.285, 480),
            segment(0.06, 0.285, 0.06, 0.30, 60),
            segment(0.06, 0.30, 0.15, 0.30, 361, endpoint=True),
        )
    )
    detector = ProfileEndpointDetector(_target_surface_config())
    # The extra index 660 imitates a weak response inside the target surface.
    # An adjacency-only implementation incorrectly returns a short half plate.
    detector._change_points = lambda _component: [360, 420, 660, 900, 960]

    result = detector.detect_temporal_breakpoint_pair(
        profile,
        np.array([-0.06, 0.0, 0.285]),
        np.array([0.06, 0.0, 0.285]),
        endpoint_gate_m=0.004,
        normal_gate_m=0.002,
        maximum_angle_difference_deg=10.0,
    )

    assert result is not None
    assert abs(result.first[0] + 0.06) < 0.001
    assert abs(result.second[0] - 0.06) < 0.001
    assert result.segment_length_m > 0.119


def test_guided_pair_spans_weak_internal_changes_between_real_edges():
    """ALIGN must evaluate the full physical chord, not only adjacent ripples."""

    def segment(x0, z0, x1, z1, count, *, endpoint=False):
        x = np.linspace(x0, x1, count, endpoint=endpoint)
        z = np.linspace(z0, z1, count, endpoint=endpoint)
        return np.column_stack((x, np.zeros(count), z))

    profile = np.vstack(
        (
            segment(-0.15, 0.30, -0.06, 0.30, 360),
            segment(-0.06, 0.30, -0.06, 0.285, 60),
            segment(-0.06, 0.285, 0.06, 0.285, 480),
            segment(0.06, 0.285, 0.06, 0.30, 60),
            segment(0.06, 0.30, 0.15, 0.30, 361, endpoint=True),
        )
    )
    detector = ProfileEndpointDetector(_target_surface_config())
    # Several weak responses lie between the two real edges.  The old ALIGN
    # implementation considered only adjacent spans and could never propose
    # the complete top surface.
    detector._change_points = lambda component: [
        len(component) // 5,
        len(component) // 2,
        4 * len(component) // 5,
    ]

    result = detector.detect_guided(
        profile,
        np.array([-0.055, 0.0, 0.285]),
        np.array([0.072, 0.0, 0.285]),
        normal_gate_m=0.003,
        endpoint_gate_m=0.016,
        maximum_angle_difference_deg=12.0,
        selection_mode="guided_align",
    )

    assert result is not None
    assert abs(result.first[0] + 0.06) < 0.001
    assert abs(result.second[0] - 0.06) < 0.001
    assert result.segment_length_m > 0.119


def test_detector_accepts_a_measured_twelve_millimetre_surface_chord():
    profile = make_profile(
        (-0.006, 0.30), (0.006, 0.30), count=121, noise_std=2.0e-5
    )
    detector = ProfileEndpointDetector(
        EndpointDetectionConfig(
            minimum_segment_length_m=0.010,
            maximum_segment_length_m=0.020,
            maximum_residual_rms_m=0.0002,
        )
    )

    result = detector.detect(profile)

    assert result is not None
    assert 0.0115 < result.segment_length_m < 0.0125


def test_temporal_candidates_keep_one_visible_physical_breakpoint():
    def segment(x0, z0, x1, z1, count, *, endpoint=False):
        x = np.linspace(x0, x1, count, endpoint=endpoint)
        z = np.linspace(z0, z1, count, endpoint=endpoint)
        return np.column_stack((x, np.zeros(count), z))

    # The left table/top transition is present at x=-60 mm, while the profile
    # ends before the expected right transition at x=+60 mm.
    profile = np.vstack(
        (
            segment(-0.15, 0.30, -0.06, 0.30, 360),
            segment(-0.06, 0.30, -0.06, 0.285, 60),
            segment(-0.06, 0.285, 0.00, 0.285, 240, endpoint=True),
        )
    )
    detector = ProfileEndpointDetector(_target_surface_config())

    first, second = detector.temporal_breakpoint_candidates(
        profile,
        np.array(
            [
                [-0.06, 0.0, 0.285],
                [0.06, 0.0, 0.285],
            ]
        ),
        endpoint_gate_m=(0.004, 0.004),
    )

    assert len(first) >= 1
    assert np.min(np.linalg.norm(first[:, (0, 2)] - [-0.06, 0.285], axis=1)) < 0.001
    assert len(second) == 0
