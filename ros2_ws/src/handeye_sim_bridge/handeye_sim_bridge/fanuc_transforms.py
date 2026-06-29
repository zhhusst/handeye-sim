"""
fanuc_transforms.py — FANUC WPR 格式与旋转矩阵转换

从 welding workspace 的 robot_control/transforms.py 移植,
保持接口兼容。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_matrix_fanuc(pose):
    """FANUC (x,y,z,w,p,r) -> 4x4 矩阵.  WPR=ZYX Euler"""
    t = pose[:3]
    rot = R.from_euler("ZYX", [pose[5], pose[4], pose[3]], degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = t
    return T


def matrix_to_pose_fanuc(T):
    """4x4 -> FANUC (x,y,z,w,p,r)"""
    t = T[:3, 3]
    zyx = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)
    return np.array([t[0], t[1], t[2], zyx[2], zyx[1], zyx[0]])


def wpr_to_quat_wxyz(pose_xyz_wpr):
    """(x,y,z,w,p,r) -> 四元数 (w,x,y,z)"""
    rot = R.from_euler("ZYX", [pose_xyz_wpr[5], pose_xyz_wpr[4], pose_xyz_wpr[3]], degrees=True)
    q = rot.as_quat()  # (x,y,z,w)
    return np.array([q[3], q[0], q[1], q[2]])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def matrix_to_quat(T):
    """4x4 -> 四元数 (w,x,y,z) + 平移 (x,y,z)"""
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()  # (x,y,z,w)
    q_wxyz = np.array([q[3], q[0], q[1], q[2]])
    return q_wxyz, T[:3, 3]
