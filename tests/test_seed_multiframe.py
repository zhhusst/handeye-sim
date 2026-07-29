import numpy as np

from calibration_pipeline.seed_collection import robust_endpoint_inliers


def test_robust_endpoint_batch_rejects_a_joint_endpoint_outlier():
    rng = np.random.default_rng(23)
    endpoint_u = np.tile(np.array([-0.02, 0.0, 0.42]), (20, 1))
    endpoint_v = np.tile(np.array([0.03, 0.0, 0.50]), (20, 1))
    endpoint_u[:, (0, 2)] += rng.normal(0.0, 2e-4, (20, 2))
    endpoint_v[:, (0, 2)] += rng.normal(0.0, 2e-4, (20, 2))
    endpoint_u[-1, 0] += 0.01

    inliers, diagnostics = robust_endpoint_inliers(
        endpoint_u, endpoint_v, mad_multiplier=3.5
    )

    assert not inliers[-1]
    assert diagnostics.inlier_count >= 18
    assert np.linalg.norm(
        diagnostics.median_u - np.array([-0.02, 0.0, 0.42])
    ) < 2e-4


def test_identical_endpoint_batch_is_not_rejected_by_zero_mad():
    endpoint_u = np.tile(np.array([-0.02, 0.0, 0.42]), (5, 1))
    endpoint_v = np.tile(np.array([0.03, 0.0, 0.50]), (5, 1))
    inliers, diagnostics = robust_endpoint_inliers(endpoint_u, endpoint_v)
    assert inliers.all()
    assert diagnostics.inlier_count == 5
