"""FANUC M-20iD/25 model support shared by simulation and real backends."""

from .fanuc_kinematic import (
    JOINT_LIMITS_DEG,
    forward_kinematics_urdf,
    inverse_kinematics_numeric,
)

__all__ = [
    "JOINT_LIMITS_DEG",
    "forward_kinematics_urdf",
    "inverse_kinematics_numeric",
]
