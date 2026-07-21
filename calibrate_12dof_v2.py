
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12-DOF hand-eye calibration v2

核心改进
1. 使用最小独立标量残差：
   plane: n^T(p-C)
   edge1 in-plane: v^T(p_e1-C)
   edge2 in-plane: u^T(p_e2-C)
   endpoint plane: n^T(p_e1-C), n^T(p_e2-C)

2. 自动平衡密集平面点与稀疏端点：
   w_plane = N_pose / N_plane_points

3. 对角点 C 做变量投影：
   给定 [R_he, t_he, R_pl] 后，C 由线性最小二乘解析求解，
   非线性优化变量由 12 维降为 9 维，但最终仍输出完整 12-DOF 结果。

依赖:
    numpy
    scipy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=float,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(3)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3) + skew(w)

    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < 1e-10:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2],
             R[0, 2] - R[2, 0],
             R[1, 0] - R[0, 1]],
            dtype=float,
        )

    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2],
         R[0, 2] - R[2, 0],
         R[1, 0] - R[0, 1]],
        dtype=float,
    )


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    dR = np.asarray(R_est).T @ np.asarray(R_gt)
    return float(np.rad2deg(np.linalg.norm(so3_log(dR))))


@dataclass
class PoseMeasurement:
    R_i: np.ndarray
    t_i: np.ndarray
    plane_points: np.ndarray
    p_e1: np.ndarray
    p_e2: np.ndarray


@dataclass
class CalibrationResult:
    R_he: np.ndarray
    t_he: np.ndarray
    R_pl: np.ndarray
    C: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str
    w_plane: float


def load_dataset(path: str | Path) -> tuple[list[PoseMeasurement], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    measurements: list[PoseMeasurement] = []
    for item in raw["poses"]:
        if not item.get("valid_e1", False) or not item.get("valid_e2", False):
            continue

        measurements.append(
            PoseMeasurement(
                R_i=np.asarray(item["R_i"], dtype=float),
                t_i=np.asarray(item["t_i"], dtype=float),
                plane_points=np.asarray(item["scan_pts_S"], dtype=float),
                p_e1=np.asarray(item["p_S_e1"], dtype=float),
                p_e2=np.asarray(item["p_S_e2"], dtype=float),
            )
        )

    if len(measurements) < 3:
        raise ValueError("至少需要3个含两条有效边端点的位姿。")

    return measurements, raw.get("scene", {})


def _fit_direction(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("拟合方向至少需要2个点。")

    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    d = vh[0]
    return d / np.linalg.norm(d)


def initialize_plane_rotation(
    measurements: list[PoseMeasurement],
    R_he_nominal: np.ndarray,
    t_he_nominal: np.ndarray,
) -> np.ndarray:
    """
    用当前手眼名义值把两组端点投到基坐标系，
    再由两条边的PCA方向初始化 R_pl=[u v n]。
    """
    e1_base: list[np.ndarray] = []
    e2_base: list[np.ndarray] = []

    for m in measurements:
        R_bs = m.R_i @ R_he_nominal
        t_bs = m.t_i + m.R_i @ t_he_nominal
        e1_base.append(R_bs @ m.p_e1 + t_bs)
        e2_base.append(R_bs @ m.p_e2 + t_bs)

    u = _fit_direction(np.asarray(e1_base))
    v_raw = _fit_direction(np.asarray(e2_base))

    # Gram-Schmidt，保证正交。
    v = v_raw - u * float(u @ v_raw)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        raise ValueError("两条边方向接近共线，无法初始化平板姿态。")
    v /= v_norm

    n = np.cross(u, v)
    n /= np.linalg.norm(n)

    R_pl = np.column_stack([u, v, n])
    if np.linalg.det(R_pl) < 0.0:
        v = -v
        n = np.cross(u, v)
        n /= np.linalg.norm(n)
        R_pl = np.column_stack([u, v, n])

    return R_pl


def automatic_plane_weight(measurements: list[PoseMeasurement]) -> float:
    """
    每帧的密集扫描点共享同一机器人位姿误差，不能把所有点完全视为独立观测。
    将整个平面点块的总权重归一到位姿数量：
        w_plane * N_plane = N_pose
    """
    n_pose = len(measurements)
    n_plane = sum(len(m.plane_points) for m in measurements)
    if n_plane <= 0:
        raise ValueError("没有平面扫描点。")
    return float(n_pose / n_plane)


def build_linear_corner_system(
    x9: np.ndarray,
    measurements: list[PoseMeasurement],
    w_plane: float,
    w_edge: float = 1.0,
    w_endpoint_plane: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造 A C ≈ b。

    对固定的 [R_he, t_he, R_pl]，所有残差关于 C 都是线性的。
    """
    x9 = np.asarray(x9, dtype=float).reshape(9)
    R_he = so3_exp(x9[0:3])
    t_he = x9[3:6]
    R_pl = so3_exp(x9[6:9])

    u, v, n = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    swp = np.sqrt(w_plane)
    swe = np.sqrt(w_edge)
    swn = np.sqrt(w_endpoint_plane)

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for m in measurements:
        R_bs = m.R_i @ R_he
        t_bs = m.t_i + m.R_i @ t_he

        # 密集平面点：n^T(p-C)=0
        for q in m.plane_points:
            p = R_bs @ q + t_bs
            rows.append(swp * n)
            rhs.append(float(swp * (n @ p)))

        p1 = R_bs @ m.p_e1 + t_bs
        p2 = R_bs @ m.p_e2 + t_bs

        # 两条边的独立面内约束。
        rows.append(swe * v)
        rhs.append(float(swe * (v @ p1)))

        rows.append(swe * u)
        rhs.append(float(swe * (u @ p2)))

        # 两个端点也必须位于平面内。
        rows.append(swn * n)
        rhs.append(float(swn * (n @ p1)))

        rows.append(swn * n)
        rhs.append(float(swn * (n @ p2)))

    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    return A, b


def solve_corner(
    x9: np.ndarray,
    measurements: list[PoseMeasurement],
    w_plane: float,
    w_edge: float = 1.0,
    w_endpoint_plane: float = 1.0,
) -> np.ndarray:
    A, b = build_linear_corner_system(
        x9,
        measurements,
        w_plane=w_plane,
        w_edge=w_edge,
        w_endpoint_plane=w_endpoint_plane,
    )
    C, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise ValueError("角点线性系统秩不足，当前位姿或边方向退化。")
    return C


def variable_projection_residual(
    x9: np.ndarray,
    measurements: list[PoseMeasurement],
    w_plane: float,
    w_edge: float = 1.0,
    w_endpoint_plane: float = 1.0,
) -> np.ndarray:
    A, b = build_linear_corner_system(
        x9,
        measurements,
        w_plane=w_plane,
        w_edge=w_edge,
        w_endpoint_plane=w_endpoint_plane,
    )
    C, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        # 返回大残差而不是让优化器崩溃。
        return np.full_like(b, 1e3)
    return A @ C - b


def solve_12dof_v2(
    measurements: list[PoseMeasurement],
    R_he_init: np.ndarray,
    t_he_init: np.ndarray,
    R_pl_init: np.ndarray | None = None,
    w_plane: float | None = None,
    w_edge: float = 1.0,
    w_endpoint_plane: float = 1.0,
    max_nfev: int = 5000,
) -> CalibrationResult:
    """
    返回完整 [R_he, t_he, R_pl, C]，但只对前9维做非线性优化。
    """
    R_he_init = np.asarray(R_he_init, dtype=float).reshape(3, 3)
    t_he_init = np.asarray(t_he_init, dtype=float).reshape(3)

    if R_pl_init is None:
        R_pl_init = initialize_plane_rotation(
            measurements, R_he_init, t_he_init
        )

    if w_plane is None:
        w_plane = automatic_plane_weight(measurements)

    x0 = np.concatenate(
        [so3_log(R_he_init), t_he_init, so3_log(R_pl_init)]
    )

    fun = lambda x: variable_projection_residual(
        x,
        measurements,
        w_plane=w_plane,
        w_edge=w_edge,
        w_endpoint_plane=w_endpoint_plane,
    )

    result = least_squares(
        fun,
        x0,
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )

    C = solve_corner(
        result.x,
        measurements,
        w_plane=w_plane,
        w_edge=w_edge,
        w_endpoint_plane=w_endpoint_plane,
    )

    return CalibrationResult(
        R_he=so3_exp(result.x[0:3]),
        t_he=result.x[3:6].copy(),
        R_pl=so3_exp(result.x[6:9]),
        C=C,
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        w_plane=float(w_plane),
    )


def demo(args: argparse.Namespace) -> None:
    measurements, scene = load_dataset(args.data)

    if "R_he_gt" not in scene or "t_he_gt" not in scene:
        raise ValueError("演示模式需要JSON中的手眼真值用于生成测试初值和评价。")

    R_gt = np.asarray(scene["R_he_gt"], dtype=float)
    t_gt = np.asarray(scene["t_he_gt"], dtype=float)

    rng = np.random.default_rng(args.seed)
    delta_rot = np.deg2rad(args.rot_sigma_deg / 3.0) * rng.normal(size=3)
    R_init = R_gt @ so3_exp(delta_rot)
    t_init = t_gt + (args.trans_sigma_mm / 1000.0) * rng.normal(size=3)

    result = solve_12dof_v2(
        measurements,
        R_he_init=R_init,
        t_he_init=t_init,
    )

    r_err = rotation_error_deg(result.R_he, R_gt)
    t_err = 1000.0 * np.linalg.norm(result.t_he - t_gt)

    print("=== 12-DOF v2 ===")
    print(f"success       : {result.success}")
    print(f"nfev          : {result.nfev}")
    print(f"w_plane(auto) : {result.w_plane:.12f}")
    print(f"R_err         : {r_err:.9f} deg")
    print(f"t_err         : {t_err:.9f} mm")
    print(f"cost          : {result.cost:.12e}")
    print(f"C             : {result.C}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/manual_v4.json",
        help="manual_v4.json路径",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rot-sigma-deg", type=float, default=5.0)
    parser.add_argument("--trans-sigma-mm", type=float, default=12.0)
    return parser


if __name__ == "__main__":
    demo(build_parser().parse_args())
