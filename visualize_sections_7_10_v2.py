#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线激光双边角点主动手眼标定：第7—10节交互式可视化

对应《线激光双边角点主动手眼标定_完整方法原理_v2.md》：

第7节：局部双边角点模型 + 人工安全梯形
第8节：eta=(a,b,alpha,psi,h,s) -> T_PS -> T_BS -> T_BF
第9节：固定法兰指令下的名义未来轮廓预测
第10节：固定同一法兰指令下的联合不确定性有效概率

依赖：
    pip install numpy matplotlib

运行：
    python visualize_sections_7_10_v2.py

保存初始界面预览：
    python visualize_sections_7_10_v2.py --save preview.png

说明：
1. 本程序使用合成的平板位姿、手眼关系和历史双边观测，仅用于解释几何原理。
2. 第10节中，角点 C 会针对每个联合状态样本，使用合成历史数据重新进行
   线性变量投影求解，而不是直接随意扰动。
3. 不包含真实机器人 IK、关节限位、碰撞与真实 Gocator 光学误差。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# --save 模式必须在导入 pyplot 前切换后端。
_SAVE_MODE = "--save" in sys.argv
if _SAVE_MODE:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# 中文字体
# =============================================================================

def find_cjk_font() -> tuple[Optional[FontProperties], Optional[str]]:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                pass
            return FontProperties(fname=str(path)), str(path)

    families = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
    ]
    for family in families:
        try:
            path = font_manager.findfont(
                FontProperties(family=family),
                fallback_to_default=False,
            )
        except Exception:
            continue
        if path and Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
            return FontProperties(fname=path), path
    return None, None


CJK_FONT, CJK_FONT_PATH = find_cjk_font()
if CJK_FONT is not None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [CJK_FONT.get_name(), "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    print(
        "[字体提示] 未检测到中文字体。Ubuntu/Debian 可安装：\n"
        "  sudo apt update && sudo apt install fonts-noto-cjk\n"
        "  rm -rf ~/.cache/matplotlib"
    )

plt.rcParams["axes.unicode_minus"] = False


def apply_cjk_font(fig) -> None:
    if CJK_FONT is None:
        return
    for artist in fig.findobj(match=Text):
        artist.set_fontproperties(CJK_FONT)


# =============================================================================
# 基础几何
# =============================================================================

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    axis = w / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    return rot_x(rx) @ rot_y(ry) @ rot_z(rz)


def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def inv_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        return T[:3, :3] @ points + T[:3, 3]
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("零向量不能归一化")
    return v / n


# =============================================================================
# 模型数据结构
# =============================================================================

@dataclass(frozen=True)
class CandidateParameters:
    a: float
    b: float
    alpha: float
    psi: float
    h: float
    s: int


@dataclass
class CandidateGeometry:
    eta: CandidateParameters
    A_P: np.ndarray
    B_P: np.ndarray
    M_P: np.ndarray
    d_line_P: np.ndarray
    laser_normal_P: np.ndarray
    T_PS: np.ndarray
    T_BS_target: np.ndarray
    T_BF_command: np.ndarray
    A_S_direct: np.ndarray
    B_S_direct: np.ndarray


@dataclass
class HistoricalMeasurement:
    T_BF: np.ndarray
    plane_points_S: np.ndarray
    e1_S: np.ndarray
    e2_S: np.ndarray


@dataclass
class Prediction:
    valid: bool
    reason: str
    a_intersection: float
    b_intersection: float
    e1_B: Optional[np.ndarray]
    e2_B: Optional[np.ndarray]
    e1_S: Optional[np.ndarray]
    e2_S: Optional[np.ndarray]
    samples_S: Optional[np.ndarray]
    safe_margin: float


@dataclass
class SampleOutcome:
    valid: bool
    reason: str
    a_intersection: float
    b_intersection: float
    e1_S: Optional[np.ndarray]
    e2_S: Optional[np.ndarray]
    C_B: np.ndarray


@dataclass(frozen=True)
class TrapezoidDomain:
    z_near: float
    z_far: float
    x_left_near: float
    x_right_near: float
    x_left_far: float
    x_right_far: float

    def x_bounds(self, z: float) -> tuple[float, float]:
        if self.z_far <= self.z_near:
            raise ValueError("z_far 必须大于 z_near")
        tau = (z - self.z_near) / (self.z_far - self.z_near)
        xl = (1.0 - tau) * self.x_left_near + tau * self.x_left_far
        xr = (1.0 - tau) * self.x_right_near + tau * self.x_right_far
        return float(xl), float(xr)

    def contains(self, p_S: np.ndarray) -> bool:
        x, _, z = np.asarray(p_S, dtype=float)
        if z < self.z_near or z > self.z_far:
            return False
        xl, xr = self.x_bounds(z)
        return xl <= x <= xr

    def margin(self, p_S: np.ndarray) -> float:
        x, _, z = np.asarray(p_S, dtype=float)
        if z < self.z_near or z > self.z_far:
            return min(z - self.z_near, self.z_far - z)
        xl, xr = self.x_bounds(z)
        return min(x - xl, xr - x, z - self.z_near, self.z_far - z)

    def vertices_xz(self) -> np.ndarray:
        return np.array([
            [self.x_left_near, self.z_near],
            [self.x_right_near, self.z_near],
            [self.x_right_far, self.z_far],
            [self.x_left_far, self.z_far],
        ])


@dataclass
class DemoModel:
    T_BP: np.ndarray
    X_FS_nominal: np.ndarray
    hard_domain: TrapezoidDomain
    safe_domain: TrapezoidDomain
    a_min: float = 0.045
    a_max: float = 0.220
    b_min: float = 0.045
    b_max: float = 0.220


# =============================================================================
# 第8节：eta -> T_PS -> T_BS -> T_BF
# =============================================================================

def construct_candidate(
    eta: CandidateParameters,
    model: DemoModel,
) -> CandidateGeometry:
    A_P = np.array([eta.a, 0.0, 0.0])
    B_P = np.array([0.0, eta.b, 0.0])
    M_P = 0.5 * (A_P + B_P)

    rho = np.linalg.norm(B_P - A_P)
    if rho < 1e-9:
        raise ValueError("a,b 不能同时接近零")

    d_line = (B_P - A_P) / rho
    n_P = np.array([0.0, 0.0, 1.0])
    q_P = normalize(np.cross(n_P, d_line))

    laser_normal = (
        np.cos(eta.alpha) * n_P
        + eta.s * np.sin(eta.alpha) * q_P
    )
    laser_normal = normalize(laser_normal)

    x0 = d_line
    yS = laser_normal
    z0 = normalize(np.cross(x0, yS))

    xS = normalize(np.cos(eta.psi) * x0 + np.sin(eta.psi) * z0)
    zS = normalize(-np.sin(eta.psi) * x0 + np.cos(eta.psi) * z0)

    # 数值正交化，确保 R_PS 是右手旋转矩阵。
    yS = normalize(np.cross(zS, xS))
    zS = normalize(np.cross(xS, yS))
    R_PS = np.column_stack([xS, yS, zS])

    t_PS = M_P - eta.h * zS
    T_PS = make_pose(R_PS, t_PS)
    T_BS = model.T_BP @ T_PS
    T_BF = T_BS @ inv_pose(model.X_FS_nominal)

    A_S = transform_points(inv_pose(T_PS), A_P)
    B_S = transform_points(inv_pose(T_PS), B_P)

    return CandidateGeometry(
        eta=eta,
        A_P=A_P,
        B_P=B_P,
        M_P=M_P,
        d_line_P=d_line,
        laser_normal_P=laser_normal,
        T_PS=T_PS,
        T_BS_target=T_BS,
        T_BF_command=T_BF,
        A_S_direct=A_S,
        B_S_direct=B_S,
    )


# =============================================================================
# 第9节：固定法兰指令下的名义预测
# =============================================================================

def predict_from_fixed_flange(
    T_BF_command: np.ndarray,
    X_FS: np.ndarray,
    T_BP: np.ndarray,
    model: DemoModel,
    n_samples: int = 25,
) -> Prediction:
    T_BS = T_BF_command @ X_FS
    R_BS = T_BS[:3, :3]
    t_BS = T_BS[:3, 3]

    R_BP = T_BP[:3, :3]
    C_B = T_BP[:3, 3]
    u_B, v_B = R_BP[:, 0], R_BP[:, 1]

    m_B = R_BS[:, 1]  # y_S 是激光平面法向
    numerator = float(m_B @ (t_BS - C_B))
    denom_u = float(m_B @ u_B)
    denom_v = float(m_B @ v_B)

    if abs(denom_u) < 1e-7 or abs(denom_v) < 1e-7:
        return Prediction(
            False, "激光平面与某条目标边近似平行",
            np.nan, np.nan, None, None, None, None, None, -np.inf
        )

    a_int = numerator / denom_u
    b_int = numerator / denom_v

    if not (model.a_min <= a_int <= model.a_max):
        return Prediction(
            False, "E_u 交点越出可信边段",
            a_int, b_int, None, None, None, None, None, -np.inf
        )
    if not (model.b_min <= b_int <= model.b_max):
        return Prediction(
            False, "E_v 交点越出可信边段",
            a_int, b_int, None, None, None, None, None, -np.inf
        )

    e1_B = C_B + a_int * u_B
    e2_B = C_B + b_int * v_B
    T_SB = inv_pose(T_BS)
    e1_S = transform_points(T_SB, e1_B)
    e2_S = transform_points(T_SB, e2_B)

    if abs(e1_S[1]) > 1e-6 or abs(e2_S[1]) > 1e-6:
        return Prediction(
            False, "端点未落在 y_S=0 激光平面内",
            a_int, b_int, e1_B, e2_B, e1_S, e2_S, None, -np.inf
        )

    if not model.safe_domain.contains(e1_S) or not model.safe_domain.contains(e2_S):
        margin = min(model.safe_domain.margin(e1_S), model.safe_domain.margin(e2_S))
        return Prediction(
            False, "一个或两个端点越出人工安全梯形",
            a_int, b_int, e1_B, e2_B, e1_S, e2_S, None, margin
        )

    lam = np.linspace(0.0, 1.0, n_samples)[:, None]
    samples_S = (1.0 - lam) * e1_S + lam * e2_S
    margin = min(model.safe_domain.margin(e1_S), model.safe_domain.margin(e2_S))
    return Prediction(
        True, "名义双边观测有效",
        a_int, b_int, e1_B, e2_B, e1_S, e2_S, samples_S, margin
    )


# =============================================================================
# 合成历史观测 + 变量投影角点
# =============================================================================

def generate_history(model: DemoModel) -> list[HistoricalMeasurement]:
    deg = np.deg2rad
    seeds = [
        CandidateParameters(0.10, 0.09, deg(25), deg(-18), 0.43, +1),
        CandidateParameters(0.13, 0.10, deg(35), deg(-8), 0.45, -1),
        CandidateParameters(0.09, 0.14, deg(45), deg(0), 0.46, +1),
        CandidateParameters(0.15, 0.08, deg(30), deg(12), 0.42, -1),
        CandidateParameters(0.10, 0.16, deg(55), deg(22), 0.48, +1),
        CandidateParameters(0.14, 0.13, deg(40), deg(32), 0.44, -1),
    ]

    history: list[HistoricalMeasurement] = []
    for eta in seeds:
        cand = construct_candidate(eta, model)
        e1 = cand.A_S_direct
        e2 = cand.B_S_direct
        lam = np.linspace(0.0, 1.0, 21)[:, None]
        points = (1.0 - lam) * e1 + lam * e2
        history.append(HistoricalMeasurement(
            T_BF=cand.T_BF_command.copy(),
            plane_points_S=points,
            e1_S=e1.copy(),
            e2_S=e2.copy(),
        ))
    return history


def solve_corner_variable_projection(
    X_FS: np.ndarray,
    R_BP: np.ndarray,
    history: list[HistoricalMeasurement],
) -> np.ndarray:
    u_B, v_B, n_B = R_BP[:, 0], R_BP[:, 1], R_BP[:, 2]
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for meas in history:
        T_BS = meas.T_BF @ X_FS
        points_B = transform_points(T_BS, meas.plane_points_S)
        e1_B = transform_points(T_BS, meas.e1_S)
        e2_B = transform_points(T_BS, meas.e2_S)

        # 每帧平面点总权重近似一致。
        w_plane = 1.0 / max(len(points_B), 1)
        sqrt_wp = np.sqrt(w_plane)
        for p in points_B:
            rows.append(sqrt_wp * n_B)
            rhs.append(float(sqrt_wp * n_B @ p))

        # e1 属于 u 边：v^T(e1-C)=0，且在平面内。
        rows.extend([v_B, n_B])
        rhs.extend([float(v_B @ e1_B), float(n_B @ e1_B)])

        # e2 属于 v 边：u^T(e2-C)=0，且在平面内。
        rows.extend([u_B, n_B])
        rhs.extend([float(u_B @ e2_B), float(n_B @ e2_B)])

    A = np.vstack(rows)
    b = np.asarray(rhs)
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C


# =============================================================================
# 第10节：联合状态样本
# =============================================================================

def evaluate_joint_samples(
    candidate: CandidateGeometry,
    model: DemoModel,
    history: list[HistoricalMeasurement],
    base_noise: np.ndarray,
    uncertainty_scale: float,
) -> list[SampleOutcome]:
    outcomes: list[SampleOutcome] = []

    R_he0 = model.X_FS_nominal[:3, :3]
    t_he0 = model.X_FS_nominal[:3, 3]
    R_BP0 = model.T_BP[:3, :3]

    # 演示性标准差。正式系统应由 Sigma_x9 决定。
    sigma_rot_he = np.deg2rad(1.2) * uncertainty_scale
    sigma_t_he = 0.004 * uncertainty_scale
    sigma_rot_pl = np.deg2rad(1.0) * uncertainty_scale

    for z in base_noise:
        dwh = z[:3] * sigma_rot_he
        dth = z[3:6] * sigma_t_he
        dwp = z[6:9] * sigma_rot_pl

        R_he = R_he0 @ so3_exp(dwh)
        t_he = t_he0 + dth
        X_sample = make_pose(R_he, t_he)

        R_BP = R_BP0 @ so3_exp(dwp)
        C_B = solve_corner_variable_projection(X_sample, R_BP, history)
        T_BP_sample = make_pose(R_BP, C_B)

        pred = predict_from_fixed_flange(
            candidate.T_BF_command,
            X_sample,
            T_BP_sample,
            model,
        )
        outcomes.append(SampleOutcome(
            valid=pred.valid,
            reason=pred.reason,
            a_intersection=pred.a_intersection,
            b_intersection=pred.b_intersection,
            e1_S=pred.e1_S,
            e2_S=pred.e2_S,
            C_B=C_B,
        ))
    return outcomes


# =============================================================================
# 绘图工具
# =============================================================================

def draw_trapezoid_2d(ax, domain: TrapezoidDomain, label: str, linestyle: str) -> None:
    v = domain.vertices_xz()
    closed = np.vstack([v, v[0]])
    ax.plot(closed[:, 0], closed[:, 1], linestyle=linestyle, linewidth=1.8, label=label)


def trapezoid_vertices_S(domain: TrapezoidDomain) -> np.ndarray:
    xz = domain.vertices_xz()
    return np.column_stack([xz[:, 0], np.zeros(4), xz[:, 1]])


def plot_frame_3d(ax, T: np.ndarray, scale: float, prefix: str) -> None:
    origin = T[:3, 3]
    R = T[:3, :3]
    styles = ["-", "--", ":"]
    labels = [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
    for i in range(3):
        end = origin + scale * R[:, i]
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            linestyle=styles[i],
            linewidth=2.0,
        )
        ax.text(end[0], end[1], end[2], labels[i], fontsize=8)


def set_3d_equal(ax, points: np.ndarray, pad: float = 0.03) -> None:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = 0.5 * (pmin + pmax)
    radius = max(np.max(pmax - pmin) * 0.55, 0.05) + pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


# =============================================================================
# 交互式界面
# =============================================================================

class Visualizer:
    def __init__(self) -> None:
        deg = np.deg2rad

        # 合成的平板基坐标位姿。
        R_BP = euler_xyz(deg(8), deg(-16), deg(20))
        C_B = np.array([0.62, 0.04, 0.28])
        T_BP = make_pose(R_BP, C_B)

        # 合成的当前手眼估计。
        R_FS = euler_xyz(deg(4), deg(-7), deg(9))
        t_FS = np.array([0.075, -0.018, 0.115])
        X_FS = make_pose(R_FS, t_FS)

        hard = TrapezoidDomain(
            z_near=0.20,
            z_far=0.75,
            x_left_near=-0.105,
            x_right_near=0.115,
            x_left_far=-0.240,
            x_right_far=0.225,
        )
        safe = TrapezoidDomain(
            z_near=0.255,
            z_far=0.690,
            x_left_near=-0.078,
            x_right_near=0.084,
            x_left_far=-0.195,
            x_right_far=0.180,
        )
        self.model = DemoModel(T_BP, X_FS, hard, safe)
        self.history = generate_history(self.model)

        rng = np.random.default_rng(7)
        self.base_noise = rng.standard_normal((61, 9))

        self.fig = plt.figure(figsize=(15.5, 10.0))
        self.fig.subplots_adjust(
            left=0.055, right=0.975, top=0.925, bottom=0.245,
            wspace=0.25, hspace=0.28
        )
        gs = self.fig.add_gridspec(2, 2)
        self.ax_local = self.fig.add_subplot(gs[0, 0])
        self.ax_3d = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_nominal = self.fig.add_subplot(gs[1, 0])
        self.ax_unc = self.fig.add_subplot(gs[1, 1])

        self.fig.suptitle(
            "第7—10节：局部双边模型 → 候选位姿 → 名义预测 → 鲁棒有效性",
            fontsize=15,
        )
        self.fig.text(
            0.5, 0.218,
            r"$\eta=(a,b,\alpha,\psi,h,s)"
            r"\ \rightarrow\ ^PT_S"
            r"\ \rightarrow\ ^BT_S"
            r"\ \rightarrow\ ^BT_F"
            r"\ \rightarrow\ \hat z_c"
            r"\ \rightarrow\ P_{\mathrm{valid}}$",
            ha="center", va="center", fontsize=12,
        )
        self.fig.text(
            0.5, 0.190,
            "a、b：两条边上的断点位置；α、s：激光平面方向；"
            "ψ：轮廓在 X-Z 平面内的倾斜；h：轮廓中点深度。",
            ha="center", va="center", fontsize=9,
        )

        self.s_value = +1
        self._build_controls()
        self.update(None)

        if CJK_FONT_PATH:
            print(f"[字体] 使用：{CJK_FONT_PATH}")

    def _build_controls(self) -> None:
        slider_specs = [
            ("a / m", 0.050, 0.210, 0.115),
            ("b / m", 0.050, 0.210, 0.105),
            ("α / °", 8.0, 75.0, 34.0),
            ("ψ / °", -50.0, 50.0, 8.0),
            ("h / m", 0.280, 0.640, 0.450),
            ("不确定性倍率", 0.0, 3.0, 1.0),
        ]

        positions = [
            [0.070, 0.155, 0.245, 0.022],
            [0.385, 0.155, 0.245, 0.022],
            [0.700, 0.155, 0.245, 0.022],
            [0.070, 0.105, 0.245, 0.022],
            [0.385, 0.105, 0.245, 0.022],
            [0.700, 0.105, 0.245, 0.022],
        ]
        self.sliders: list[Slider] = []
        for spec, pos in zip(slider_specs, positions):
            ax = self.fig.add_axes(pos)
            slider = Slider(ax, spec[0], spec[1], spec[2], valinit=spec[3])
            slider.on_changed(self.update)
            self.sliders.append(slider)

        radio_ax = self.fig.add_axes([0.070, 0.018, 0.105, 0.060])
        self.radio = RadioButtons(radio_ax, ("s = +1", "s = -1"), active=0)
        self.radio.on_clicked(self._on_s_change)

        button_defs = [
            ("安全候选", [0.225, 0.030, 0.105, 0.036], self.preset_safe),
            ("梯形边缘风险", [0.350, 0.030, 0.125, 0.036], self.preset_trapezoid_risk),
            ("边段风险", [0.495, 0.030, 0.105, 0.036], self.preset_edge_risk),
            ("大倾角", [0.620, 0.030, 0.095, 0.036], self.preset_large_alpha),
            ("复位", [0.735, 0.030, 0.080, 0.036], self.preset_reset),
        ]
        self.buttons = []
        for label, pos, callback in button_defs:
            ax = self.fig.add_axes(pos)
            button = Button(ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def current_eta(self) -> CandidateParameters:
        a, b, alpha_deg, psi_deg, h, _ = [s.val for s in self.sliders]
        return CandidateParameters(
            a=float(a),
            b=float(b),
            alpha=np.deg2rad(alpha_deg),
            psi=np.deg2rad(psi_deg),
            h=float(h),
            s=self.s_value,
        )

    def _on_s_change(self, label: str) -> None:
        self.s_value = +1 if "+1" in label else -1
        self.update(None)

    def _set_values(self, values: tuple[float, float, float, float, float, float]) -> None:
        for slider, value in zip(self.sliders, values):
            slider.set_val(value)

    def preset_safe(self, _event) -> None:
        self._set_values((0.115, 0.105, 34.0, 8.0, 0.450, 0.8))

    def preset_trapezoid_risk(self, _event) -> None:
        self._set_values((0.190, 0.175, 40.0, 42.0, 0.315, 1.6))

    def preset_edge_risk(self, _event) -> None:
        self._set_values((0.052, 0.205, 28.0, -24.0, 0.455, 1.5))

    def preset_large_alpha(self, _event) -> None:
        self._set_values((0.120, 0.110, 68.0, 5.0, 0.455, 1.0))

    def preset_reset(self, _event) -> None:
        self._set_values((0.115, 0.105, 34.0, 8.0, 0.450, 1.0))
        self.radio.set_active(0)

    def update(self, _value) -> None:
        eta = self.current_eta()
        try:
            candidate = construct_candidate(eta, self.model)
        except Exception as exc:
            self.fig.suptitle(f"候选构造失败：{exc}")
            self.fig.canvas.draw_idle()
            return

        nominal = predict_from_fixed_flange(
            candidate.T_BF_command,
            self.model.X_FS_nominal,
            self.model.T_BP,
            self.model,
        )
        uncertainty_scale = float(self.sliders[5].val)
        outcomes = evaluate_joint_samples(
            candidate,
            self.model,
            self.history,
            self.base_noise,
            uncertainty_scale,
        )

        self._draw_module7(candidate)
        self._draw_module8(candidate)
        self._draw_module9(candidate, nominal)
        self._draw_module10(candidate, outcomes)

        apply_cjk_font(self.fig)
        self.fig.canvas.draw_idle()

    def _draw_module7(self, candidate: CandidateGeometry) -> None:
        ax = self.ax_local
        ax.clear()
        m = self.model

        ax.plot(
            [m.a_min, m.a_max], [0, 0],
            linewidth=5, label=r"$E_u$可信边段"
        )
        ax.plot(
            [0, 0], [m.b_min, m.b_max],
            linewidth=5, label=r"$E_v$可信边段"
        )
        ax.plot(
            [0, 0.24], [0, 0],
            linestyle="--", linewidth=1.0, label="u轴延长线"
        )
        ax.plot(
            [0, 0], [0, 0.24],
            linestyle=":", linewidth=1.0, label="v轴延长线"
        )

        A, B = candidate.A_P, candidate.B_P
        ax.plot([A[0], B[0]], [A[1], B[1]], linewidth=2.2, label="目标轮廓 AB")
        ax.scatter([0], [0], marker="s", s=55, label="固定角点 C")
        ax.scatter([A[0]], [A[1]], marker="o", s=70, label="A：e1")
        ax.scatter([B[0]], [B[1]], marker="^", s=70, label="B：e2")

        ax.annotate(f"a={candidate.eta.a:.3f} m", (A[0], A[1]), xytext=(8, 10),
                    textcoords="offset points")
        ax.annotate(f"b={candidate.eta.b:.3f} m", (B[0], B[1]), xytext=(8, 10),
                    textcoords="offset points")

        ax.set_title("模块7：局部双边角点模型")
        ax.set_xlabel(r"平板局部坐标 $\xi$（沿 $u$）/ m")
        ax.set_ylabel(r"平板局部坐标 $\eta_P$（沿 $v$）/ m")
        ax.set_xlim(-0.025, 0.245)
        ax.set_ylim(-0.025, 0.245)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02, 0.98,
            "只维护同一角点及两条固定边；\n不需要恢复整个矩形平板。",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", alpha=0.10),
        )

    def _draw_module8(self, candidate: CandidateGeometry) -> None:
        ax = self.ax_3d
        ax.clear()

        A, B, M = candidate.A_P, candidate.B_P, candidate.M_P
        m = self.model

        # 局部平板小区域，仅用于展示角点附近。
        patch = np.array([
            [0.0, 0.0, 0.0],
            [0.23, 0.0, 0.0],
            [0.23, 0.23, 0.0],
            [0.0, 0.23, 0.0],
        ])
        collection = Poly3DCollection([patch], alpha=0.08)
        ax.add_collection3d(collection)

        ax.plot([0, 0.23], [0, 0], [0, 0], linestyle="--", linewidth=1.2)
        ax.plot([0, 0], [0, 0.23], [0, 0], linestyle=":", linewidth=1.2)
        ax.plot(
            [m.a_min, m.a_max], [0, 0], [0, 0],
            linewidth=4.5, label=r"$E_u$"
        )
        ax.plot(
            [0, 0], [m.b_min, m.b_max], [0, 0],
            linewidth=4.5, label=r"$E_v$"
        )
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]],
                linewidth=2.4, label="目标交线 AB")
        ax.scatter(*A, marker="o", s=45)
        ax.scatter(*B, marker="^", s=45)
        ax.scatter(*M, marker="x", s=45)

        # 人工硬/安全梯形从 S 系变换到 P 系。
        hard_P = transform_points(candidate.T_PS, trapezoid_vertices_S(m.hard_domain))
        safe_P = transform_points(candidate.T_PS, trapezoid_vertices_S(m.safe_domain))

        hard_closed = np.vstack([hard_P, hard_P[0]])
        safe_closed = np.vstack([safe_P, safe_P[0]])
        ax.plot(hard_closed[:, 0], hard_closed[:, 1], hard_closed[:, 2],
                linestyle="--", linewidth=1.5, label="人工硬梯形")
        ax.plot(safe_closed[:, 0], safe_closed[:, 1], safe_closed[:, 2],
                linestyle="-", linewidth=2.0, label="人工安全梯形")

        sensor_origin = candidate.T_PS[:3, 3]
        plot_frame_3d(ax, candidate.T_PS, 0.055, "S:")
        ax.scatter(*sensor_origin, marker="s", s=35, label="传感器原点")

        # 激光面法向。
        normal_end = M + 0.07 * candidate.laser_normal_P
        ax.plot([M[0], normal_end[0]], [M[1], normal_end[1]], [M[2], normal_end[2]],
                linewidth=2.0)
        ax.text(*normal_end, "激光面法向", fontsize=8)

        all_points = np.vstack([patch, hard_P, safe_P, sensor_origin[None, :], A[None, :], B[None, :]])
        set_3d_equal(ax, all_points)
        ax.set_xlabel("P:x / m")
        ax.set_ylabel("P:y / m")
        ax.set_zlabel("P:z / m")
        ax.set_title("模块8：η 唯一构造传感器—平板相对位姿")
        ax.view_init(elev=24, azim=-55)
        ax.legend(loc="upper left", fontsize=7)
        ax.text2D(
            0.02, 0.02,
            r"$a,b$定交点；$\alpha,s$定激光面；"
            "\n"
            r"$\psi$定平面内姿态；$h$定中点深度。",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", alpha=0.10),
        )

    def _draw_module9(self, candidate: CandidateGeometry, nominal: Prediction) -> None:
        ax = self.ax_nominal
        ax.clear()
        draw_trapezoid_2d(ax, self.model.hard_domain, "人工硬梯形", "--")
        draw_trapezoid_2d(ax, self.model.safe_domain, "人工安全梯形", "-")

        if nominal.e1_S is not None and nominal.e2_S is not None:
            p1, p2 = nominal.e1_S, nominal.e2_S
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]],
                    linewidth=2.2, label="名义预测轮廓")
            ax.scatter([p1[0]], [p1[2]], marker="o", s=65, label="e1")
            ax.scatter([p2[0]], [p2[2]], marker="^", s=65, label="e2")
            ax.scatter(
                [0.5 * (p1[0] + p2[0])],
                [0.5 * (p1[2] + p2[2])],
                marker="x", s=50, label="轮廓中点",
            )

        direct_err = (
            np.linalg.norm(candidate.A_S_direct - nominal.e1_S)
            + np.linalg.norm(candidate.B_S_direct - nominal.e2_S)
            if nominal.e1_S is not None and nominal.e2_S is not None
            else np.nan
        )

        ax.set_title("模块9：固定法兰指令下的名义未来轮廓")
        ax.set_xlabel(r"$x_S$ / m")
        ax.set_ylabel(r"$z_S$ / m")
        ax.set_xlim(-0.27, 0.27)
        ax.set_ylim(0.16, 0.78)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

        status = "有效" if nominal.valid else "无效"
        text = (
            f"名义状态：{status}\n"
            f"原因：{nominal.reason}\n"
            f"求交 a={nominal.a_intersection:.4f} m\n"
            f"求交 b={nominal.b_intersection:.4f} m\n"
            f"安全余量={nominal.safe_margin:.4f} m\n"
            f"构造—重算一致性误差={direct_err:.2e}"
        )
        ax.text(
            0.98, 0.98, text,
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", alpha=0.10),
            fontsize=9,
        )

    def _draw_module10(
        self,
        candidate: CandidateGeometry,
        outcomes: list[SampleOutcome],
    ) -> None:
        ax = self.ax_unc
        ax.clear()
        draw_trapezoid_2d(ax, self.model.hard_domain, "人工硬梯形", "--")
        draw_trapezoid_2d(ax, self.model.safe_domain, "人工安全梯形", "-")

        valid_count = sum(int(o.valid) for o in outcomes)
        p_valid = valid_count / max(len(outcomes), 1)

        # 用线型和标记区分有效/无效，不依赖指定颜色。
        shown_valid = False
        shown_invalid = False
        for idx, out in enumerate(outcomes):
            if out.e1_S is None or out.e2_S is None:
                continue
            p1, p2 = out.e1_S, out.e2_S
            if out.valid:
                label = "有效样本" if not shown_valid else None
                shown_valid = True
                ax.plot(
                    [p1[0], p2[0]], [p1[2], p2[2]],
                    linestyle="-", linewidth=0.8, alpha=0.22, label=label
                )
                ax.scatter([p1[0], p2[0]], [p1[2], p2[2]],
                           marker="o", s=9, alpha=0.30)
            else:
                label = "无效样本" if not shown_invalid else None
                shown_invalid = True
                ax.plot(
                    [p1[0], p2[0]], [p1[2], p2[2]],
                    linestyle=":", linewidth=0.9, alpha=0.35, label=label
                )
                ax.scatter([p1[0], p2[0]], [p1[2], p2[2]],
                           marker="x", s=18, alpha=0.55)

        # 名义目标轮廓作为粗线。
        p1, p2 = candidate.A_S_direct, candidate.B_S_direct
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]],
                linewidth=2.8, label="名义目标")

        reason_counts: dict[str, int] = {}
        for out in outcomes:
            key = out.reason
            reason_counts[key] = reason_counts.get(key, 0) + 1
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))
        reason_text = "\n".join(
            f"{name}: {count}" for name, count in sorted_reasons[:5]
        )

        ax.set_title("模块10：同一法兰指令下的联合不确定性传播")
        ax.set_xlabel(r"$x_S$ / m")
        ax.set_ylabel(r"$z_S$ / m")
        ax.set_xlim(-0.27, 0.27)
        ax.set_ylim(0.16, 0.78)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

        ax.text(
            0.98, 0.98,
            f"$P_{{valid}}$={p_valid:.3f}\n"
            f"有效 {valid_count}/{len(outcomes)}\n\n"
            f"失败/状态统计：\n{reason_text}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", alpha=0.10),
            fontsize=9,
        )
        ax.text(
            0.02, 0.02,
            "候选法兰指令始终固定；\n"
            "每个样本只改变手眼和平板状态，\n"
            "并重新变量投影求解角点 C。",
            transform=ax.transAxes, va="bottom",
            bbox=dict(boxstyle="round", alpha=0.08),
            fontsize=8,
        )

    def save(self, path: str) -> None:
        apply_cjk_font(self.fig)
        self.fig.savefig(path, dpi=170, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None, help="保存静态预览并退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = Visualizer()
    if args.save:
        app.save(args.save)
        print(f"已保存预览：{args.save}")
        return
    plt.show()


if __name__ == "__main__":
    main()
