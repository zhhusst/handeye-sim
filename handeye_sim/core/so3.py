#!/usr/bin/env python3
"""
core/so3.py — SO(3) 基础工具（代码库唯一来源）

所有旋转相关函数集中在此，其他模块不再重复实现。
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """反对称矩阵 [v]_× ∈ so(3)"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """轴角 → SO(3) 旋转矩阵: R = exp([w]_×)"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) → 轴角: w = log(R)"""
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(tr)
    if abs(theta) < 1e-12:
        return np.zeros(3)
    return theta / (2 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def so3_expm(w: np.ndarray) -> np.ndarray:
    """so(3) 矩阵指数: exp([w]_×). 返回 3×3 旋转矩阵 (等同于 so3_exp)"""
    return so3_exp(w)


def dexpm(w: np.ndarray) -> np.ndarray:
    """SO(3) 右 Jacobian (Dexp)

    将轴角增量 δw 映射到切空间增量 δφ:
      R(w + δw) ≈ R(w) exp(Dexp(w) δw)
      δφ = Dexp(w) δw

    Dexp(w) = I - (1-cosθ)/θ² [w]_× + (θ-sinθ)/θ³ [w]_×²
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = skew(w / theta)
    a = (1 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta * theta * theta)
    return np.eye(3) - a * theta * K + b * theta * theta * (K @ K)


def dexpm_inv(w: np.ndarray) -> np.ndarray:
    """SO(3) 右 Jacobian 逆

    Dexp(w)^(-1) = I + (1/2)[w]_× + (1/θ² - (1+cosθ)/(2θ sinθ)) [w]_×²
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = skew(w / theta)
    c = (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * theta * K + (1.0/(theta*theta) - c) * theta * theta * (K @ K)


def rpy_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """ZYX 欧拉角 (度) → 旋转矩阵 R = Rx(rx)·Ry(ry)·Rz(rz)"""
    rx, ry, rz = np.deg2rad(rx_deg), np.deg2rad(ry_deg), np.deg2rad(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def rot_x(deg: float) -> np.ndarray:
    """绕 x 轴旋转 (度)"""
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    """绕 y 轴旋转 (度)"""
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    """绕 z 轴旋转 (度)"""
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """两个旋转矩阵之间的角度误差 (度)"""
    dR = R_est.T @ R_gt
    tr = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return np.rad2deg(np.arccos(tr))


def translation_error_mm(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """平移误差 (mm)"""
    return np.linalg.norm(t_est - t_gt) * 1000


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """两向量夹角 (度)"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.rad2deg(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))
