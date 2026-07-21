#!/usr/bin/env python3
"""
solvers/cross_12dof_v2.py — 12-DOF v2: 变量投影 + 标量残差

核心改进:
  1. 变量投影: C 由线性最小二乘解析求解, 非线性优化仅 9 维 [w_he, t_he, w_pl]
  2. 标量残差: v^T(p_e1-C), u^T(p_e2-C) 替代 cross-product, 每帧 4 个独立标量
  3. 自动权重: w_plane = N_pose / N_plane_points

依赖: numpy, scipy
"""

import numpy as np
from scipy.optimize import least_squares

from handeye_sim.core.so3 import so3_exp, so3_log, skew
from handeye_sim.core.types import CalibResult


def _fit_direction(points):
    """PCA 拟合 3D 点集方向"""
    if len(points) < 2:
        raise ValueError("至少需要2个点拟合方向")
    pts = np.asarray(points)
    _, _, vh = np.linalg.svd(pts - np.mean(pts, axis=0), full_matrices=False)
    d = vh[0]
    return d / np.linalg.norm(d)


def init_R_pl_from_endpoints(meas_list, R_he_nom, t_he_nom):
    """用名义手眼投影端点 → PCA 初始化 R_pl = [u v n]"""
    e1_base, e2_base = [], []
    for m in meas_list:
        R_i, t_i = m['R_i'], m['t_i']
        R_bs = R_i @ R_he_nom
        t_bs = t_i + R_i @ t_he_nom
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            e1_base.append(R_bs @ np.asarray(m['p_S_e1']) + t_bs)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            e2_base.append(R_bs @ np.asarray(m['p_S_e2']) + t_bs)

    u = _fit_direction(e1_base)
    v_raw = _fit_direction(e2_base)

    # Gram-Schmidt 正交化
    v = v_raw - u * float(u @ v_raw)
    v /= np.linalg.norm(v)
    n = np.cross(u, v); n /= np.linalg.norm(n)
    v = np.cross(n, u)  # 保证右手系

    R_pl = np.column_stack([u, v, n])
    if np.linalg.det(R_pl) < 0:
        v = -v; n = np.cross(u, v); n /= np.linalg.norm(n)
        R_pl = np.column_stack([u, v, n])
    return R_pl


def _build_linear_C_system(x9, poses, meas, w_plane, w_edge=1.0, w_ep=1.0):
    """构造 A·C ≈ b, 给定 [w_he, t_he, w_pl] 后 C 是线性的"""
    R_he = so3_exp(x9[0:3])
    t_he = x9[3:6]
    R_pl = so3_exp(x9[6:9])
    u, v, n = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    swp, swe, swn = np.sqrt(w_plane), np.sqrt(w_edge), np.sqrt(w_ep)
    rows, rhs = [], []

    for (R_i, t_i), m in zip(poses, meas):
        R_bs = np.asarray(R_i) @ R_he
        t_bs = np.asarray(t_i) + np.asarray(R_i) @ t_he

        # 平面点: n^T(p - C) = 0  →  n^T·C = n^T·p
        for q in m.get('p_S_plane', []):
            p = R_bs @ np.asarray(q) + t_bs
            rows.append(swp * n)
            rhs.append(float(swp * (n @ p)))

        # 边端点
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p1 = R_bs @ np.asarray(m['p_S_e1']) + t_bs
            rows.append(swe * v);   rhs.append(float(swe * (v @ p1)))   # v^T·(p1-C)=0
            rows.append(swn * n);   rhs.append(float(swn * (n @ p1)))   # n^T·(p1-C)=0

        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p2 = R_bs @ np.asarray(m['p_S_e2']) + t_bs
            rows.append(swe * u);   rhs.append(float(swe * (u @ p2)))   # u^T·(p2-C)=0
            rows.append(swn * n);   rhs.append(float(swn * (n @ p2)))   # n^T·(p2-C)=0

    return np.array(rows), np.array(rhs)


def _solve_C_linear(x9, poses, meas, w_plane, w_edge=1.0, w_ep=1.0):
    """线性求解最优 C"""
    A, b = _build_linear_C_system(x9, poses, meas, w_plane, w_edge, w_ep)
    C, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return np.zeros(3), False
    return C, True


def _varproj_residual(x9, poses, meas, w_plane, w_edge=1.0, w_ep=1.0):
    """变量投影残差: r = A·C_opt(x9) - b"""
    A, b = _build_linear_C_system(x9, poses, meas, w_plane, w_edge, w_ep)
    C, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        return np.full_like(b, 1e3)
    return A @ C - b


def calibrate_12dof_v2(poses, meas, R_he_nom=None, t_he_nom=None,
                        solver_cfg=None, seed=42) -> CalibResult:
    """12-DOF v2: 变量投影标定

    Args:
        poses: [(R_i, t_i), ...]
        meas:  [{valid_e1, p_S_e1, valid_e2, p_S_e2, p_S_plane}, ...]
        R_he_nom, t_he_nom: 名义手眼初值
        solver_cfg: dict, 来自 config.yaml solvers.dof12v2
    """
    cfg = solver_cfg or {}
    max_nfev = cfg.get('max_nfev', 5000)
    _ftol = cfg.get('ftol', 1e-13)
    _xtol = cfg.get('xtol', 1e-13)
    _gtol = cfg.get('gtol', 1e-13)
    w_edge = cfg.get('w_edge', 1.0)
    w_ep = cfg.get('w_ep', 1.0)
    R0 = np.asarray(R_he_nom) if R_he_nom is not None else np.eye(3)
    t0 = np.asarray(t_he_nom) if t_he_nom is not None else np.zeros(3)

    # 只保留同时含 e1/e2 的帧
    valid = []
    for m in meas:
        ok = m.get('valid_e1') and m.get('p_S_e1') is not None and \
             m.get('valid_e2') and m.get('p_S_e2') is not None
        valid.append(ok)

    # 过滤: 至少需要 3 帧
    poses_f = [p for p, v in zip(poses, valid) if v]
    meas_f = [m for m, v in zip(meas, valid) if v]
    if len(poses_f) < 3:
        return CalibResult(method='12dof-v2', converged=False,
                           R_he=np.eye(3), t_he=np.zeros(3), cost=float('inf'),
                           diagnostics={'error': 'need >= 3 poses with both edges'})

    # 自动权重
    n_pose = len(poses_f)
    n_plane = sum(len(m.get('p_S_plane', [])) for m in meas_f)
    w_plane = n_pose / max(n_plane, 1)

    # 初始化 R_pl
    meas_list = [{'R_i': p[0], 't_i': p[1], **m} for p, m in zip(poses_f, meas_f)]
    R_pl_init = init_R_pl_from_endpoints(meas_list, R0, t0)

    x0 = np.concatenate([so3_log(R0), t0, so3_log(R_pl_init)])

    # SciPy trust-region 优化
    result = least_squares(
        lambda x: _varproj_residual(x, poses_f, meas_f, w_plane, w_edge, w_ep),
        x0, method='trf', x_scale='jac', max_nfev=max_nfev,
        ftol=_ftol, xtol=_xtol, gtol=_gtol,
    )

    # 最终 C
    C, ok = _solve_C_linear(result.x, poses_f, meas_f, w_plane, w_edge, w_ep)
    R_he = so3_exp(result.x[0:3])
    t_he = result.x[3:6]
    R_pl = so3_exp(result.x[6:9])

    return CalibResult(
        method='12dof-v2',
        R_he=R_he, t_he=t_he, R_pl=R_pl, C=C,
        cost=float(result.cost),
        diagnostics={'nfev': int(result.nfev), 'w_plane': float(w_plane),
                     'n_poses_used': n_pose},
    )
