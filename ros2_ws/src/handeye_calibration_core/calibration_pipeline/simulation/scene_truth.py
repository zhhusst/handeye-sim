"""Simulation-only hand-eye truth, read from the calibration scene YAML."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import yaml


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
)


def load_handeye_truth(config_path: str | Path = DEFAULT_CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """Return ^F T_S from the single simulation truth in calibration.yaml."""
    parameters = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    truth = parameters["/**"]["ros__parameters"]["evaluation"]
    if not truth.get("truth_available", False):
        raise ValueError("simulation evaluation.truth_available must be true")
    rotation = np.asarray(truth["handeye_rotation"], dtype=float).reshape(3, 3)
    translation = np.asarray(truth["handeye_translation_m"], dtype=float).reshape(3)
    if (
        not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("simulation hand-eye truth must be a finite SE(3) transform")
    return rotation, translation


HAND_EYE_ROTATION, HAND_EYE_TRANSLATION = load_handeye_truth()
HAND_EYE_RPY_RAD = Rotation.from_matrix(HAND_EYE_ROTATION).as_euler("xyz")
