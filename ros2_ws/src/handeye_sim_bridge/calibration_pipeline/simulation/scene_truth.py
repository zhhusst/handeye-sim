"""Ground truth used only by the repository's Gazebo simulation and evaluation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


HAND_EYE_RPY_RAD = np.array([0.485145, 0.160648, -1.509479])
HAND_EYE_ROTATION = Rotation.from_euler("xyz", HAND_EYE_RPY_RAD).as_matrix()
HAND_EYE_TRANSLATION = np.array([-0.011579, -0.004621, 0.359284])
