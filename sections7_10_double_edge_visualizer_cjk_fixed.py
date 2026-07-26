#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sections 7-10 geometry visualizer for the double-edge active hand-eye method.

This script visualizes four ideas:
  7) Recover a finite rectangular board from C, u, v, L, W.
  8) Select A on E_u and B on E_v, then construct a laser plane through AB.
  9) Predict the physical profile by intersecting laser plane -> board plane -> finite rectangle,
     then check edge identity and sensor ROI.
 10) Keep one commanded flange pose fixed, perturb the 9-D reduced state, recompute C by
     variable projection from a synthetic seed dataset, and estimate P_valid.

Run:
    python finite_board_double_edge_demo.py

Save an initial screenshot in a headless environment:
    python finite_board_double_edge_demo.py --save demo.png

Dependencies:
    numpy, matplotlib
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button


def _find_cjk_font() -> tuple[FontProperties | None, str | None]:
    """Find a local font that can render Simplified Chinese.

    The returned FontProperties is applied explicitly to every Matplotlib Text
    object, including Slider and Button labels. This avoids the common case in
    which rcParams is set correctly but TkAgg widgets still fall back to
    DejaVu Sans.
    """
    candidate_paths = [
        # Ubuntu/Debian: sudo apt install fonts-noto-cjk
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        # Common Linux alternatives
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]

    for raw_path in candidate_paths:
        path = Path(raw_path)
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                pass
            return FontProperties(fname=str(path)), str(path)

    candidate_families = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "AR PL UMing CN",
        "AR PL SungtiL GB",
        "AR PL KaitiM GB",
    ]
    for family in candidate_families:
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


_CJK_FONT_PROP, _CJK_FONT_PATH = _find_cjk_font()

if _CJK_FONT_PROP is not None:
    _CJK_FONT_NAME = _CJK_FONT_PROP.get_name()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        _CJK_FONT_NAME,
        "DejaVu Sans",
    ]
else:
    _CJK_FONT_NAME = None
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    print(
        "\n[字体提示] 没有检测到可显示中文的字体。\n"
        "Ubuntu/Debian 可执行：\n"
        "  sudo apt update && sudo apt install fonts-noto-cjk\n"
        "安装后重新运行本程序。\n"
    )

plt.rcParams["axes.unicode_minus"] = False


def _apply_cjk_font(fig) -> None:
    """Apply the selected CJK font to all existing text objects in a figure."""
    if _CJK_FONT_PROP is None:
        return
    for artist in fig.findobj(match=Text):
        artist.set_fontproperties(_CJK_FONT_PROP)

EPS = 1e-10


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    axis = w / theta
    K = skew(axis)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def normalize(v: np.ndarray, name: str = "vector") -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise ValueError(f"{name} norm is too small")
    return v / n


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    one_point = p.ndim == 1
    p = p.reshape(-1, 3)
    out = (T[:3, :3] @ p.T).T + T[:3, 3]
    return out[0] if one_point else out


@dataclass
class BoardState:
    C: np.ndarray
    R_pl: np.ndarray  # columns [u, v, n]
    L: float
    W: float

    @property
    def u(self) -> np.ndarray:
        return self.R_pl[:, 0]

    @property
    def v(self) -> np.ndarray:
        return self.R_pl[:, 1]

    @property
    def n(self) -> np.ndarray:
        return self.R_pl[:, 2]

    def point(self, xi: float, eta: float) -> np.ndarray:
        return self.C + xi * self.u + eta * self.v

    def corners(self) -> np.ndarray:
        return np.vstack([
            self.point(0.0, 0.0),
            self.point(self.L, 0.0),
            self.point(self.L, self.W),
            self.point(0.0, self.W),
        ])


@dataclass
class ROI:
    x_min: float = -0.30
    x_max: float = 0.30
    z_min: float = 0.12
    z_max: float = 0.82


@dataclass
class PredictedProfile:
    valid: bool
    reason: str
    endpoints_B: Optional[np.ndarray] = None
    endpoints_uv: Optional[np.ndarray] = None
    endpoints_S: Optional[np.ndarray] = None
    samples_S: Optional[np.ndarray] = None
    edge_ids: tuple[str, str] = ("", "")
    profile_length: float = 0.0
    roi_margin: float = -np.inf


@dataclass
class Measurement:
    T_BF: np.ndarray
    plane_points_S: np.ndarray
    p_e1_S: np.ndarray
    p_e2_S: np.ndarray


@dataclass
class CandidateGeometry:
    A: np.ndarray
    B: np.ndarray
    d_line: np.ndarray
    m_laser: np.ndarray
    T_BS: np.ndarray
    T_BF_cmd: np.ndarray


EDGE_EU = "E_u: eta=0"
EDGE_EV = "E_v: xi=0"
EDGE_U_FAR = "far: eta=W"
EDGE_V_FAR = "far: xi=L"


def board_uv(board: BoardState, p_B: np.ndarray) -> np.ndarray:
    delta = np.asarray(p_B, dtype=float).reshape(3) - board.C
    return np.array([board.u @ delta, board.v @ delta], dtype=float)


def plane_plane_intersection(
    n1: np.ndarray,
    p1: np.ndarray,
    n2: np.ndarray,
    p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n1 = normalize(n1, "plane normal 1")
    n2 = normalize(n2, "plane normal 2")
    direction = np.cross(n1, n2)
    dn = float(np.linalg.norm(direction))
    if dn < 1e-8:
        raise ValueError("Laser plane and board plane are parallel or coincident")
    direction /= dn
    A = np.vstack([n1, n2, direction])
    b = np.array([n1 @ p1, n2 @ p2, 0.0], dtype=float)
    point = np.linalg.solve(A, b)
    return point, direction


def clip_infinite_line_to_rectangle(
    p0: np.ndarray,
    d: np.ndarray,
    L: float,
    W: float,
) -> Optional[np.ndarray]:
    """Clip p(s)=p0+s*d to [0,L] x [0,W]. Return two UV endpoints."""
    p0 = np.asarray(p0, dtype=float).reshape(2)
    d = np.asarray(d, dtype=float).reshape(2)
    s_lo, s_hi = -np.inf, np.inf
    for value, slope, lower, upper in (
        (p0[0], d[0], 0.0, L),
        (p0[1], d[1], 0.0, W),
    ):
        if abs(slope) < EPS:
            if value < lower - 1e-9 or value > upper + 1e-9:
                return None
            continue
        s1 = (lower - value) / slope
        s2 = (upper - value) / slope
        low_i, high_i = min(s1, s2), max(s1, s2)
        s_lo = max(s_lo, low_i)
        s_hi = min(s_hi, high_i)
        if s_lo > s_hi + 1e-10:
            return None
    if not np.isfinite(s_lo) or not np.isfinite(s_hi):
        return None
    return np.vstack([p0 + s_lo * d, p0 + s_hi * d])


def classify_edge(uv: np.ndarray, L: float, W: float, tol: float = 1e-6) -> str:
    xi, eta = map(float, uv)
    distances = {
        EDGE_EU: abs(eta),
        EDGE_EV: abs(xi),
        EDGE_U_FAR: abs(eta - W),
        EDGE_V_FAR: abs(xi - L),
    }
    edge = min(distances, key=distances.get)
    if distances[edge] > tol:
        return "unknown"
    return edge


def construct_candidate(
    board: BoardState,
    a: float,
    b: float,
    alpha_deg: float,
    psi_deg: float,
    h: float,
    sign: int,
    T_FS_nominal: np.ndarray,
) -> CandidateGeometry:
    A = board.point(a, 0.0)
    B = board.point(0.0, b)
    d_line = normalize(B - A, "candidate line direction")

    q = normalize(np.cross(board.n, d_line), "q = n cross d")
    alpha = math.radians(alpha_deg)
    m_laser = normalize(math.cos(alpha) * board.n + math.sin(alpha) * q, "laser plane normal")

    y_axis = normalize(float(sign) * m_laser, "sensor y axis")
    x0 = d_line
    z0 = normalize(np.cross(x0, y_axis), "sensor z0 axis")

    psi = math.radians(psi_deg)
    x_axis = normalize(math.cos(psi) * x0 - math.sin(psi) * z0, "sensor x axis")
    z_axis = normalize(math.sin(psi) * x0 + math.cos(psi) * z0, "sensor z axis")
    R_BS = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(R_BS) < 0.0:
        raise ValueError("Constructed sensor frame is not right-handed")

    midpoint = 0.5 * (A + B)
    t_BS = midpoint - h * z_axis
    T_BS = make_transform(R_BS, t_BS)
    T_BF_cmd = T_BS @ invert_transform(T_FS_nominal)
    return CandidateGeometry(A, B, d_line, m_laser, T_BS, T_BF_cmd)


def predict_profile(board: BoardState, T_BS: np.ndarray, roi: ROI) -> PredictedProfile:
    R_BS = T_BS[:3, :3]
    t_BS = T_BS[:3, 3]
    laser_normal_B = R_BS[:, 1]  # y_S=0

    try:
        line_point_B, line_direction_B = plane_plane_intersection(
            board.n, board.C, laser_normal_B, t_BS
        )
    except ValueError as exc:
        return PredictedProfile(False, str(exc))

    p0_uv = board_uv(board, line_point_B)
    d_uv = np.array([board.u @ line_direction_B, board.v @ line_direction_B], dtype=float)
    clipped_uv = clip_infinite_line_to_rectangle(p0_uv, d_uv, board.L, board.W)
    if clipped_uv is None:
        return PredictedProfile(False, "The infinite intersection line does not cross the finite board")

    endpoints_B = np.vstack([board.point(*uv) for uv in clipped_uv])
    edge_ids = (
        classify_edge(clipped_uv[0], board.L, board.W),
        classify_edge(clipped_uv[1], board.L, board.W),
    )

    expected_pair = {EDGE_EU, EDGE_EV}
    if set(edge_ids) != expected_pair:
        return PredictedProfile(
            False,
            f"Physical endpoints switched to edges: {edge_ids}",
            endpoints_B=endpoints_B,
            endpoints_uv=clipped_uv,
            edge_ids=edge_ids,
        )

    # Reorder: endpoint 0 is e1 on E_u; endpoint 1 is e2 on E_v.
    if edge_ids[0] == EDGE_EV:
        endpoints_B = endpoints_B[::-1].copy()
        clipped_uv = clipped_uv[::-1].copy()
        edge_ids = (edge_ids[1], edge_ids[0])

    T_SB = invert_transform(T_BS)
    endpoints_S = transform_points(T_SB, endpoints_B)
    x = endpoints_S[:, 0]
    y = endpoints_S[:, 1]
    z = endpoints_S[:, 2]

    if np.max(np.abs(y)) > 1e-6:
        return PredictedProfile(
            False,
            "Numerical inconsistency: endpoints are not on y_S=0",
            endpoints_B=endpoints_B,
            endpoints_uv=clipped_uv,
            endpoints_S=endpoints_S,
            edge_ids=edge_ids,
        )

    margins = np.concatenate([
        x - roi.x_min,
        roi.x_max - x,
        z - roi.z_min,
        roi.z_max - z,
    ])
    roi_margin = float(np.min(margins))
    if roi_margin < 0.0:
        return PredictedProfile(
            False,
            "Both physical edge points are not fully inside the sensor ROI",
            endpoints_B=endpoints_B,
            endpoints_uv=clipped_uv,
            endpoints_S=endpoints_S,
            edge_ids=edge_ids,
            profile_length=float(np.linalg.norm(endpoints_B[1] - endpoints_B[0])),
            roi_margin=roi_margin,
        )

    ts = np.linspace(0.0, 1.0, 40)
    samples_B = endpoints_B[0][None, :] * (1.0 - ts[:, None]) + endpoints_B[1][None, :] * ts[:, None]
    samples_S = transform_points(T_SB, samples_B)
    length = float(np.linalg.norm(endpoints_B[1] - endpoints_B[0]))
    if length < 0.04:
        return PredictedProfile(
            False,
            "Profile segment is too short",
            endpoints_B=endpoints_B,
            endpoints_uv=clipped_uv,
            endpoints_S=endpoints_S,
            samples_S=samples_S,
            edge_ids=edge_ids,
            profile_length=length,
            roi_margin=roi_margin,
        )

    return PredictedProfile(
        True,
        "Nominal double-edge profile is valid",
        endpoints_B=endpoints_B,
        endpoints_uv=clipped_uv,
        endpoints_S=endpoints_S,
        samples_S=samples_S,
        edge_ids=edge_ids,
        profile_length=length,
        roi_margin=roi_margin,
    )


def solve_corner_variable_projection(
    measurements: Iterable[Measurement],
    R_he: np.ndarray,
    t_he: np.ndarray,
    R_pl: np.ndarray,
) -> np.ndarray:
    """Recompute C from all fixed seed measurements for one perturbed 9-D state."""
    u, v, n = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    for measurement in measurements:
        R_BF = measurement.T_BF[:3, :3]
        t_BF = measurement.T_BF[:3, 3]

        pts_S = measurement.plane_points_S
        pts_B = (R_BF @ (R_he @ pts_S.T + t_he[:, None]) + t_BF[:, None]).T
        plane_weight = 1.0 / math.sqrt(max(len(pts_B), 1))
        for p_B in pts_B:
            A_rows.append(plane_weight * n)
            b_rows.append(float(plane_weight * (n @ p_B)))

        e1_B = R_BF @ (R_he @ measurement.p_e1_S + t_he) + t_BF
        e2_B = R_BF @ (R_he @ measurement.p_e2_S + t_he) + t_BF

        # e1 lies on the u-edge, e2 lies on the v-edge.
        A_rows.extend([v, u, n, n])
        b_rows.extend([
            float(v @ e1_B),
            float(u @ e2_B),
            float(n @ e1_B),
            float(n @ e2_B),
        ])

    A = np.vstack(A_rows)
    b = np.asarray(b_rows, dtype=float)
    C, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise RuntimeError("Corner variable-projection system is rank deficient")
    return C


def make_nominal_problem() -> tuple[BoardState, np.ndarray, ROI, list[Measurement]]:
    R_pl = so3_exp(np.deg2rad(np.array([8.0, -11.0, 6.0])))
    C = np.array([0.12, -0.08, 0.20], dtype=float)
    board = BoardState(C=C, R_pl=R_pl, L=0.50, W=0.40)

    R_FS = so3_exp(np.deg2rad(np.array([12.0, -8.0, 18.0])))
    t_FS = np.array([0.045, -0.018, 0.115], dtype=float)
    T_FS = make_transform(R_FS, t_FS)
    roi = ROI()

    # Synthetic already-collected double-edge seed measurements.
    specs = [
        (0.12, 0.10, 36.0, -15.0, 0.42, 1),
        (0.30, 0.11, 44.0, 10.0, 0.46, 1),
        (0.14, 0.29, 52.0, -8.0, 0.48, 1),
        (0.34, 0.27, 58.0, 16.0, 0.52, 1),
        (0.22, 0.18, 47.0, 28.0, 0.45, 1),
        (0.38, 0.15, 62.0, -25.0, 0.50, 1),
    ]
    measurements: list[Measurement] = []
    for a, b, alpha, psi, h, sign in specs:
        candidate = construct_candidate(board, a, b, alpha, psi, h, sign, T_FS)
        profile = predict_profile(board, candidate.T_BS, roi)
        if not profile.valid or profile.samples_S is None or profile.endpoints_S is None:
            raise RuntimeError(f"Synthetic seed profile invalid: {profile.reason}")
        measurements.append(
            Measurement(
                T_BF=candidate.T_BF_cmd,
                plane_points_S=profile.samples_S[::2].copy(),
                p_e1_S=profile.endpoints_S[0].copy(),
                p_e2_S=profile.endpoints_S[1].copy(),
            )
        )
    return board, T_FS, roi, measurements


def axis_sigma_points(scale: float) -> list[np.ndarray]:
    """Center + +/- 2 sigma along each of the 9 reduced-state axes."""
    base_std = np.concatenate([
        np.deg2rad([1.0, 1.0, 1.0]),
        np.array([0.005, 0.005, 0.008]),
        np.deg2rad([0.7, 0.7, 0.7]),
    ])
    gamma = 2.0
    sigma = scale * base_std
    points = [np.zeros(9)]
    for i in range(9):
        delta = np.zeros(9)
        delta[i] = gamma * sigma[i]
        points.append(delta.copy())
        points.append(-delta.copy())
    return points


def uncertainty_predictions(
    board_nom: BoardState,
    T_FS_nom: np.ndarray,
    candidate: CandidateGeometry,
    measurements: list[Measurement],
    roi: ROI,
    scale: float,
) -> list[tuple[BoardState, np.ndarray, PredictedProfile, np.ndarray]]:
    results = []
    R_he_nom = T_FS_nom[:3, :3]
    t_he_nom = T_FS_nom[:3, 3]

    for delta in axis_sigma_points(scale):
        R_he_s = R_he_nom @ so3_exp(delta[:3])
        t_he_s = t_he_nom + delta[3:6]
        R_pl_s = board_nom.R_pl @ so3_exp(delta[6:9])
        C_s = solve_corner_variable_projection(measurements, R_he_s, t_he_s, R_pl_s)
        board_s = BoardState(C=C_s, R_pl=R_pl_s, L=board_nom.L, W=board_nom.W)

        T_FS_s = make_transform(R_he_s, t_he_s)
        # Critical rule from Section 10: commanded flange pose remains fixed.
        T_BS_actual_s = candidate.T_BF_cmd @ T_FS_s
        profile_s = predict_profile(board_s, T_BS_actual_s, roi)
        results.append((board_s, T_BS_actual_s, profile_s, delta))
    return results


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * np.max(maxs - mins), 0.18)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


class GeometryDemo:
    def __init__(self) -> None:
        self.board, self.T_FS_nom, self.roi, self.measurements = make_nominal_problem()
        self.margin = 0.04

        self.fig = plt.figure(figsize=(14, 10.5))
        gs = self.fig.add_gridspec(2, 2)
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_uv = self.fig.add_subplot(gs[0, 1])
        self.ax_profile = self.fig.add_subplot(gs[1, 0])
        self.ax_text = self.fig.add_subplot(gs[1, 1])
        self.fig.subplots_adjust(bottom=0.29, wspace=0.22, hspace=0.29)

        self._build_controls()
        self.update(None)
        _apply_cjk_font(self.fig)

    def _build_controls(self) -> None:
        left = 0.13
        width = 0.58
        height = 0.022
        rows = [0.245, 0.210, 0.175, 0.140, 0.105, 0.070]

        self.s_a = Slider(self.fig.add_axes([left, rows[0], width, height]),
                          "a on E_u / m", self.margin, self.board.L - self.margin, valinit=0.22)
        self.s_b = Slider(self.fig.add_axes([left, rows[1], width, height]),
                          "b on E_v / m", self.margin, self.board.W - self.margin, valinit=0.18)
        self.s_alpha = Slider(self.fig.add_axes([left, rows[2], width, height]),
                              "alpha / deg", 5.0, 78.0, valinit=45.0)
        self.s_psi = Slider(self.fig.add_axes([left, rows[3], width, height]),
                            "psi / deg", -45.0, 45.0, valinit=5.0)
        self.s_h = Slider(self.fig.add_axes([left, rows[4], width, height]),
                          "working distance h / m", 0.26, 0.72, valinit=0.46)
        self.s_unc = Slider(self.fig.add_axes([left, rows[5], width, height]),
                            "uncertainty scale", 0.0, 4.0, valinit=1.0)

        for slider in (self.s_a, self.s_b, self.s_alpha, self.s_psi, self.s_h, self.s_unc):
            slider.on_changed(self.update)

        ax_safe = self.fig.add_axes([0.76, 0.205, 0.10, 0.04])
        ax_risky = self.fig.add_axes([0.87, 0.205, 0.10, 0.04])
        ax_roi = self.fig.add_axes([0.76, 0.145, 0.10, 0.04])
        ax_reset = self.fig.add_axes([0.87, 0.145, 0.10, 0.04])
        self.b_safe = Button(ax_safe, "安全候选")
        self.b_risky = Button(ax_risky, "边缘风险")
        self.b_roi = Button(ax_roi, "ROI风险")
        self.b_reset = Button(ax_reset, "重置")
        self.b_safe.on_clicked(self.preset_safe)
        self.b_risky.on_clicked(self.preset_risky)
        self.b_roi.on_clicked(self.preset_roi)
        self.b_reset.on_clicked(self.preset_reset)

    def _set_sliders(self, *, a: float, b: float, alpha: float, psi: float, h: float, unc: float) -> None:
        self.s_a.set_val(a)
        self.s_b.set_val(b)
        self.s_alpha.set_val(alpha)
        self.s_psi.set_val(psi)
        self.s_h.set_val(h)
        self.s_unc.set_val(unc)

    def preset_safe(self, _event) -> None:
        self._set_sliders(a=0.24, b=0.19, alpha=48.0, psi=5.0, h=0.46, unc=0.7)

    def preset_risky(self, _event) -> None:
        self._set_sliders(a=0.050, b=0.350, alpha=35.0, psi=0.0, h=0.43, unc=2.5)

    def preset_roi(self, _event) -> None:
        self._set_sliders(a=0.450, b=0.350, alpha=58.0, psi=40.0, h=0.29, unc=2.0)

    def preset_reset(self, _event) -> None:
        self._set_sliders(a=0.22, b=0.18, alpha=45.0, psi=5.0, h=0.46, unc=1.0)

    def update(self, _value) -> None:
        a = float(self.s_a.val)
        b = float(self.s_b.val)
        alpha = float(self.s_alpha.val)
        psi = float(self.s_psi.val)
        h = float(self.s_h.val)
        unc = float(self.s_unc.val)

        candidate = construct_candidate(
            self.board, a, b, alpha, psi, h, 1, self.T_FS_nom
        )
        nominal = predict_profile(self.board, candidate.T_BS, self.roi)
        samples = uncertainty_predictions(
            self.board, self.T_FS_nom, candidate, self.measurements, self.roi, unc
        )
        self.draw(candidate, nominal, samples)
        # Titles, labels and annotations are recreated on every update, so the
        # explicit font must also be reapplied on every redraw.
        _apply_cjk_font(self.fig)
        self.fig.canvas.draw_idle()

    def draw(self, candidate: CandidateGeometry, nominal: PredictedProfile,
             samples: list[tuple[BoardState, np.ndarray, PredictedProfile, np.ndarray]]) -> None:
        self._draw_3d(candidate)
        self._draw_uv(candidate, nominal, samples)
        self._draw_profile(nominal, samples)
        self._draw_text(candidate, nominal, samples)

    def _draw_3d(self, candidate: CandidateGeometry) -> None:
        ax = self.ax_3d
        ax.clear()
        corners = self.board.corners()
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], linewidth=1.8)

        U, V = np.meshgrid(np.linspace(0, self.board.L, 2), np.linspace(0, self.board.W, 2))
        P = self.board.C[None, None, :] + U[..., None] * self.board.u + V[..., None] * self.board.v
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=0.14)

        ax.scatter(*self.board.C, s=42)
        ax.text(*(self.board.C + 0.018 * self.board.n), "C")
        ax.text(*self.board.point(self.board.L * 0.55, 0.0), "E_u")
        ax.text(*self.board.point(0.0, self.board.W * 0.55), "E_v")

        A, B = candidate.A, candidate.B
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], linewidth=2.2)
        ax.scatter(*A, s=42)
        ax.scatter(*B, s=42)
        ax.text(*(A + 0.014 * self.board.n), "A")
        ax.text(*(B + 0.014 * self.board.n), "B")

        midpoint = 0.5 * (A + B)
        plane_side = normalize(np.cross(candidate.m_laser, candidate.d_line), "laser plane side")
        ss, rr = np.meshgrid(np.linspace(-0.34, 0.34, 2), np.linspace(-0.20, 0.20, 2))
        LP = midpoint[None, None, :] + ss[..., None] * candidate.d_line + rr[..., None] * plane_side
        ax.plot_surface(LP[..., 0], LP[..., 1], LP[..., 2], alpha=0.12)

        R = candidate.T_BS[:3, :3]
        t = candidate.T_BS[:3, 3]
        ax.scatter(*t, marker="s", s=38)
        for i, label in enumerate(("x_S", "y_S", "z_S")):
            vec = 0.09 * R[:, i]
            ax.quiver(t[0], t[1], t[2], vec[0], vec[1], vec[2], arrow_length_ratio=0.15)
            ax.text(*(t + vec), label, fontsize=8)
        ax.plot([t[0], midpoint[0]], [t[1], midpoint[1]], [t[2], midpoint[2]], linestyle="--")

        ax.set_title("7+8：有限平板、A/B目标点、候选激光平面与传感器位姿")
        ax.set_xlabel("B-X / m")
        ax.set_ylabel("B-Y / m")
        ax.set_zlabel("B-Z / m")
        set_axes_equal_3d(ax, np.vstack([corners, t, A, B]))
        ax.view_init(elev=25, azim=-58)

    def _draw_uv(self, candidate: CandidateGeometry, nominal: PredictedProfile,
                 samples: list[tuple[BoardState, np.ndarray, PredictedProfile, np.ndarray]]) -> None:
        ax = self.ax_uv
        ax.clear()
        L, W = self.board.L, self.board.W
        rect = np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], dtype=float)
        ax.plot(rect[:, 0], rect[:, 1], linewidth=1.8)
        ax.text(0.5 * L, -0.028, "E_u: eta=0", ha="center")
        ax.text(-0.018, 0.5 * W, "E_v: xi=0", va="center", rotation=90)
        ax.text(0.5 * L, W + 0.018, "far edge: eta=W", ha="center", fontsize=8)
        ax.text(L + 0.012, 0.5 * W, "far edge: xi=L", va="center", rotation=90, fontsize=8)

        Auv = board_uv(self.board, candidate.A)
        Buv = board_uv(self.board, candidate.B)
        direction = normalize(np.array([Buv[0] - Auv[0], Buv[1] - Auv[1], 0.0]))[:2]
        t = np.linspace(-0.5, 0.9, 2)
        ext = Auv[None, :] + t[:, None] * direction[None, :]
        ax.plot(ext[:, 0], ext[:, 1], linestyle="--", alpha=0.7)
        ax.plot([Auv[0], Buv[0]], [Auv[1], Buv[1]], linewidth=2.2)
        ax.scatter([Auv[0], Buv[0]], [Auv[1], Buv[1]], s=38)
        ax.text(Auv[0], Auv[1] + 0.015, "A")
        ax.text(Buv[0] + 0.008, Buv[1], "B")

        # Overlay the finite-board intersections predicted by all state samples.
        for board_s, _T_BS_s, profile_s, _delta in samples:
            corners_s_B = board_s.corners()
            corners_s_uv_nom = np.vstack([board_uv(self.board, p) for p in corners_s_B])
            loop_s = np.vstack([corners_s_uv_nom, corners_s_uv_nom[0]])
            ax.plot(loop_s[:, 0], loop_s[:, 1], alpha=0.10)
            if profile_s.endpoints_B is not None:
                uv_nom = np.vstack([board_uv(self.board, p) for p in profile_s.endpoints_B])
                if profile_s.valid:
                    ax.plot(uv_nom[:, 0], uv_nom[:, 1], alpha=0.38)
                else:
                    ax.plot(uv_nom[:, 0], uv_nom[:, 1], linestyle=":", alpha=0.50)

        if nominal.endpoints_uv is not None:
            ax.scatter(nominal.endpoints_uv[:, 0], nominal.endpoints_uv[:, 1], marker="o", s=45)

        ax.set_title("9+10：先与有限矩形求交，再识别边；细线为状态不确定性样本")
        ax.set_xlabel("xi along u / m")
        ax.set_ylabel("eta along v / m")
        ax.set_xlim(-0.07, L + 0.08)
        ax.set_ylim(-0.06, W + 0.07)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    def _draw_profile(self, nominal: PredictedProfile,
                      samples: list[tuple[BoardState, np.ndarray, PredictedProfile, np.ndarray]]) -> None:
        ax = self.ax_profile
        ax.clear()
        roi = self.roi
        ax.add_patch(Rectangle((roi.x_min, roi.z_min), roi.x_max - roi.x_min,
                               roi.z_max - roi.z_min, fill=False, linewidth=1.8))
        ax.text(roi.x_min, roi.z_max + 0.02, "sensor ROI in x_S-z_S")

        for _board_s, _T_BS_s, profile_s, _delta in samples:
            if profile_s.endpoints_S is None:
                continue
            ep = profile_s.endpoints_S
            if profile_s.valid:
                ax.plot(ep[:, 0], ep[:, 2], alpha=0.30)
                ax.scatter(ep[:, 0], ep[:, 2], marker=".", s=16, alpha=0.40)
            else:
                ax.plot(ep[:, 0], ep[:, 2], linestyle=":", alpha=0.45)
                ax.scatter(ep[:, 0], ep[:, 2], marker="x", s=24, alpha=0.55)

        if nominal.samples_S is not None:
            ax.plot(nominal.samples_S[:, 0], nominal.samples_S[:, 2], linewidth=2.2)
        if nominal.endpoints_S is not None:
            ep = nominal.endpoints_S
            ax.scatter(ep[:, 0], ep[:, 2], s=52)
            ax.text(ep[0, 0], ep[0, 2], "e1")
            ax.text(ep[1, 0], ep[1, 2], "e2")

        ax.set_title("9：物理边缘端点变换到传感器系后，最后检查ROI")
        ax.set_xlabel("x_S / m")
        ax.set_ylabel("z_S / m")
        ax.set_xlim(roi.x_min - 0.08, roi.x_max + 0.08)
        ax.set_ylim(max(0.0, roi.z_min - 0.10), roi.z_max + 0.10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    def _draw_text(self, candidate: CandidateGeometry, nominal: PredictedProfile,
                   samples: list[tuple[BoardState, np.ndarray, PredictedProfile, np.ndarray]]) -> None:
        ax = self.ax_text
        ax.clear()
        ax.axis("off")

        valid_count = sum(int(item[2].valid) for item in samples)
        total = len(samples)
        p_valid = valid_count / max(total, 1)
        edge_fail = sum(
            (not item[2].valid) and ("switched" in item[2].reason)
            for item in samples
        )
        roi_fail = sum(
            (not item[2].valid) and ("ROI" in item[2].reason)
            for item in samples
        )
        other_fail = total - valid_count - edge_fail - roi_fail

        text = (
            "四个部分的对应关系\n\n"
            "7  用 C、u、v、L、W 把无限平面恢复成有限矩形。\n"
            "8  在 E_u、E_v 上选 A、B；先确定合法线 AB，再构造激光平面和传感器位姿。\n"
            "9  激光平面∩平板平面→无限线；再与有限矩形求交→物理断点；最后检查边身份和ROI。\n"
            "10 固定同一个法兰指令；扰动9维状态；每个样本用种子数据重新变量投影求 C；重复第9步。\n\n"
            f"名义判定：{'有效' if nominal.valid else '无效'}\n"
            f"原因：{nominal.reason}\n"
            f"边身份：{nominal.edge_ids}\n"
            f"轮廓长度：{nominal.profile_length*1000:.1f} mm\n"
            f"ROI最小余量：{nominal.roi_margin*1000:.1f} mm\n\n"
            f"不确定性样本：{valid_count}/{total} 有效\n"
            f"P_valid ≈ {p_valid:.3f}\n"
            f"边切换失败：{edge_fail}\n"
            f"ROI失败：{roi_fail}\n"
            f"其他失败：{other_fail}\n\n"
            "验证边界：几何求交、固定法兰指令、C的变量投影重算已实现；"
            "IK、碰撞、反射、曝光和真实断点检测未模拟。"
        )
        ax.text(0.01, 0.98, text, va="top", ha="left", fontsize=9.0, wrap=True)
        ax.set_title("当前参数下的判定与方法解释")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Sections 7-10 of the double-edge active hand-eye method")
    parser.add_argument("--save", type=Path, default=None, help="Save the initial figure instead of opening a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if _CJK_FONT_PATH is not None:
        print(f"[字体] 使用中文字体: {_CJK_FONT_PATH}")
    app = GeometryDemo()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        app.fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved preview to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
