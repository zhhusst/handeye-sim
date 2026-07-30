import json

import numpy as np

from calibration_pipeline.dataset_io import (
    aggregate_seed_group,
    load_seed_dataset,
    load_seed_dataset_grouped,
    split_stationary_group,
)


def _frame(offset: float) -> dict:
    return {
        "R_BF": np.eye(3).tolist(),
        "t_BF": [offset, 0.0, 0.0],
        "profile_points_S": [[0.0, 0.0, 0.4], [0.1, 0.0, 0.5]],
        "endpoint_u_S": [0.0, 0.0, 0.4],
        "endpoint_v_S": [0.1, 0.0, 0.5],
    }


def test_schema_v3_preserves_physical_groups_and_flattens_compatibly(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "seeds": [
                    {"label": "reference", "frames": [_frame(0.0), _frame(0.1)]},
                    {"label": "rx_positive", "frames": [_frame(0.2)]},
                ],
            }
        )
    )

    grouped = load_seed_dataset_grouped(path)
    poses, measurements = load_seed_dataset(path)

    assert grouped.physical_seed_count == 2
    assert grouped.observation_count == 3
    assert [group.label for group in grouped.groups] == [
        "reference",
        "rx_positive",
    ]
    assert len(poses) == len(measurements) == 3

    pose, measurement = aggregate_seed_group(grouped.groups[0])
    assert np.allclose(pose.translation, [0.05, 0.0, 0.0])
    assert len(measurement.profile_points) == 4


def test_stationary_group_split_reserves_a_validation_frame(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "seeds": [
                    {
                        "label": "reference",
                        "frames": [_frame(0.0), _frame(0.1), _frame(0.2)],
                    }
                ],
            }
        )
    )
    group = load_seed_dataset_grouped(path).groups[0]
    fit, validation = split_stationary_group(
        group, validation_frame_count=1
    )
    assert validation is not None
    assert len(fit.poses) == 2
    assert len(validation.poses) == 1
    assert validation.poses[0].translation[0] == 0.1
