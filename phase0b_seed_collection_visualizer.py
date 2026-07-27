#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0b：预设多轴旋转 + 无标定双边特征平移伺服
交互式原理可视化

对应：
《线激光双边角点主动手眼标定_完整方法原理_v5.md》
第5节“Phase 0b：基于预设多轴旋转和无标定双边特征伺服的自动种子采集”。

依赖：
    pip install numpy matplotlib

运行：
    python phase0b_seed_collection_visualizer.py

保存初始预览：
    python phase0b_seed_collection_visualizer.py --save phase0b_preview.png

重要说明：
1. 这是“控制层简化仿真”，用于解释第5节的逻辑，不是完整机器人/传感器物理仿真。
2. 程序故意不使用手眼关系。旋转造成的轮廓漂移，以及法兰平移对 x_mid/z_mid
   的影响，被抽象为未知、随姿态缓慢变化的局部输入—输出关系。
3. 该程序包含多个故障预设，可用于发现当前方案可能存在的薄弱环节。
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SAVE_MODE = "--save" in sys.argv
if _SAVE_MODE:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =============================================================================
# 中文字体
# =============================================================================

def find_cjk_font() -> Tuple[Optional[FontProperties], Optional[str]]:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for raw in candidates:
        p = Path(raw)
        if p.exists():
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:
                pass
            return FontProperties(fname=str(p)), str(p)

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
        "[字体提示] 未检测到中文字体。Ubuntu/Debian可执行：\n"
        "sudo apt update && sudo apt install fonts-noto-cjk\n"
        "rm -rf ~/.cache/matplotlib"
    )
plt.rcParams["axes.unicode_minus"] = False


def apply_cjk_font(fig) -> None:
    if CJK_FONT is None:
        return
    for artist in fig.findobj(match=Text):
        artist.set_fontproperties(CJK_FONT)


# =============================================================================
# 旋转工具
# =============================================================================

def rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def relative_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    v = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(v)))


# =============================================================================
# 人工安全梯形
# =============================================================================

@dataclass(frozen=True)
class TrapezoidDomain:
    z_near: float = 260.0
    z_far: float = 650.0
    x_left_near: float = -75.0
    x_right_near: float = 75.0
    x_left_far: float = -175.0
    x_right_far: float = 175.0

    def x_bounds(self, z: float) -> Tuple[float, float]:
        tau = (z - self.z_near) / (self.z_far - self.z_near)
        xl = (1.0 - tau) * self.x_left_near + tau * self.x_left_far
        xr = (1.0 - tau) * self.x_right_near + tau * self.x_right_far
        return float(xl), float(xr)

    def margin(self, point_xz: np.ndarray) -> float:
        x, z = np.asarray(point_xz, dtype=float)
        xl, xr = self.x_bounds(z)
        return float(min(z - self.z_near, self.z_far - z, x - xl, xr - x))

    def contains(self, point_xz: np.ndarray) -> bool:
        return self.margin(point_xz) >= 0.0

    def vertices(self) -> np.ndarray:
        return np.array([
            [self.x_left_near, self.z_near],
            [self.x_right_near, self.z_near],
            [self.x_right_far, self.z_far],
            [self.x_left_far, self.z_far],
        ])


# =============================================================================
# 仿真状态
# =============================================================================

BRANCHES = ["R0", "Rx+", "Rx-", "Ry+", "Ry-", "Rx+Ry+"]
AXES = ["F-X", "F-Y", "F-Z"]


@dataclass
class ProfileState:
    x_mid: float = 0.0
    z_mid: float = 445.0
    length: float = 155.0
    tilt_deg: float = 8.0

    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        phi = np.deg2rad(self.tilt_deg)
        d = np.array([np.cos(phi), np.sin(phi)])
        p1 = np.array([self.x_mid, self.z_mid]) - 0.5 * self.length * d
        p2 = np.array([self.x_mid, self.z_mid]) + 0.5 * self.length * d
        return p1, p2


@dataclass
class SimSnapshot:
    R_current: np.ndarray
    branch_progress_deg: float
    combo_stage: int
    profile: ProfileState
    g_hat: np.ndarray
    selected_axis: Optional[int]
    step_deg: float
    previous_labeled_endpoints: Optional[Tuple[np.ndarray, np.ndarray]]
    stable_count: int


@dataclass
class SimulationState:
    branch: str = "Rx+"
    R_reference: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_current: np.ndarray = field(default_factory=lambda: np.eye(3))
    branch_progress_deg: float = 0.0
    combo_stage: int = 0

    profile: ProfileState = field(default_factory=ProfileState)
    reference_profile: ProfileState = field(default_factory=ProfileState)

    g_hat: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    probe_safe: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))
    selected_axis: Optional[int] = None
    probe_done: bool = False

    last_valid: Optional[SimSnapshot] = None
    previous_labeled_endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = None
    stable_count: int = 0
    seed_saved: bool = False
    saved_branches: List[str] = field(default_factory=lambda: ["R0"])

    event_index: int = 0
    hist_event: List[int] = field(default_factory=list)
    hist_x_mid: List[float] = field(default_factory=list)
    hist_z_dev: List[float] = field(default_factory=list)
    hist_margin: List[float] = field(default_factory=list)
    hist_action: List[str] = field(default_factory=list)

    message: str = "人工给出第一个双边稳定位姿；当前处于安全参考状态。"
    controller_state: str = "SAVE_REFERENCE"
    last_identity_costs: Tuple[float, float] = (np.nan, np.nan)
    identity_ambiguous: bool = False


# =============================================================================
# 控制层简化模型
# =============================================================================

class SeedCollectionModel:
    def __init__(self) -> None:
        self.domain = TrapezoidDomain()
        self.margin_required = 12.0
        self.length_min = 70.0
        self.length_max = 250.0
        self.x_tolerance = 2.0
        self.stable_required = 2
        self.g_min = 0.06
        self.rng = np.random.default_rng(8)

        self.state = SimulationState()
        self.state.stable_count = self.stable_required
        e1, e2 = self.state.profile.endpoints()
        self.state.previous_labeled_endpoints = (e1.copy(), e2.copy())
        self.state.last_valid = self.make_snapshot()
        self.record("reference")

    # -------------------------------------------------------------------------
    # 场景参数
    # -------------------------------------------------------------------------

    def scenario_name(self, scenario_id: int) -> str:
        return [
            "正常局部模型",
            "明显Z漂移",
            "平移灵敏度过弱",
            "灵敏度随姿态变号",
            "端点身份容易模糊",
        ][scenario_id]

    def rotation_drift_per_deg(
        self, axis: str, sign: float, scenario_id: int
    ) -> Tuple[float, float, float, float]:
        """
        返回每旋转1°对：
        x_mid(mm), z_mid(mm), tilt(deg), length(mm)
        的控制层漂移。
        """
        if axis == "X":
            base = np.array([2.6, 0.75, 0.42, -0.25])
        else:
            base = np.array([-1.9, 1.15, -0.30, 0.18])

        base[0] *= sign
        base[2] *= sign

        if scenario_id == 1:  # Z漂移
            base[1] *= 3.8
        elif scenario_id == 4:  # 身份模糊：长度逐渐缩短
            base[3] -= 2.1
        return tuple(float(v) for v in base)

    def true_sensitivity(
        self, total_angle_deg: float, scenario_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        g_true: 法兰局部平移(mm) -> x_mid变化(mm)
        h_true: 法兰局部平移(mm) -> z_mid变化(mm)
        """
        theta = float(total_angle_deg)
        g = np.array([
            0.24 + 0.003 * theta,
            -0.90 + 0.04 * np.sin(np.deg2rad(5.0 * theta)),
            0.14 - 0.002 * theta,
        ])
        hz = np.array([0.10, 0.18, -0.72])

        if scenario_id == 2:
            g = np.array([0.025, -0.035, 0.018])
        elif scenario_id == 3:
            # F-Y轴在约7°附近穿过零点并变号。
            g[1] = -0.85 + 0.125 * theta
        elif scenario_id == 1:
            hz = np.array([0.20, 0.42, -0.88])

        return g, hz

    # -------------------------------------------------------------------------
    # 状态与测量
    # -------------------------------------------------------------------------

    def make_snapshot(self) -> SimSnapshot:
        s = self.state
        return SimSnapshot(
            R_current=s.R_current.copy(),
            branch_progress_deg=float(s.branch_progress_deg),
            combo_stage=int(s.combo_stage),
            profile=copy.deepcopy(s.profile),
            g_hat=s.g_hat.copy(),
            selected_axis=s.selected_axis,
            step_deg=0.0,
            previous_labeled_endpoints=(
                None
                if s.previous_labeled_endpoints is None
                else (
                    s.previous_labeled_endpoints[0].copy(),
                    s.previous_labeled_endpoints[1].copy(),
                )
            ),
            stable_count=int(s.stable_count),
        )

    def restore_snapshot(self, snap: SimSnapshot) -> None:
        s = self.state
        s.R_current = snap.R_current.copy()
        s.branch_progress_deg = snap.branch_progress_deg
        s.combo_stage = snap.combo_stage
        s.profile = copy.deepcopy(snap.profile)
        s.g_hat = snap.g_hat.copy()
        s.selected_axis = snap.selected_axis
        s.probe_done = bool(np.any(np.isfinite(s.g_hat)))
        s.previous_labeled_endpoints = (
            None
            if snap.previous_labeled_endpoints is None
            else (
                snap.previous_labeled_endpoints[0].copy(),
                snap.previous_labeled_endpoints[1].copy(),
            )
        )
        s.stable_count = snap.stable_count

    def profile_metrics(self) -> Dict[str, float]:
        e1, e2 = self.state.profile.endpoints()
        margin = min(self.domain.margin(e1), self.domain.margin(e2))
        return {
            "x_mid": self.state.profile.x_mid,
            "z_mid": self.state.profile.z_mid,
            "length": self.state.profile.length,
            "margin": margin,
        }

    def safety_ok(self) -> bool:
        m = self.profile_metrics()
        e1, e2 = self.state.profile.endpoints()
        return (
            self.domain.contains(e1)
            and self.domain.contains(e2)
            and m["margin"] >= self.margin_required
            and self.length_min <= m["length"] <= self.length_max
        )

    def target_total_deg(self, target_angle: float) -> float:
        return 2.0 * target_angle if self.state.branch == "Rx+Ry+" else target_angle

    def branch_axis_and_sign(
        self, target_angle: float
    ) -> Tuple[Optional[str], float]:
        branch = self.state.branch
        if branch == "R0":
            return None, 0.0
        if branch == "Rx+":
            return "X", +1.0
        if branch == "Rx-":
            return "X", -1.0
        if branch == "Ry+":
            return "Y", +1.0
        if branch == "Ry-":
            return "Y", -1.0
        if branch == "Rx+Ry+":
            if self.state.branch_progress_deg < target_angle - 1e-9:
                self.state.combo_stage = 0
                return "X", +1.0
            self.state.combo_stage = 1
            return "Y", +1.0
        raise ValueError(f"未知分支：{branch}")

    def target_rotation(self, target_angle: float) -> np.ndarray:
        a = np.deg2rad(target_angle)
        b = self.state.branch
        if b == "R0":
            return self.state.R_reference.copy()
        if b == "Rx+":
            return self.state.R_reference @ rot_x(+a)
        if b == "Rx-":
            return self.state.R_reference @ rot_x(-a)
        if b == "Ry+":
            return self.state.R_reference @ rot_y(+a)
        if b == "Ry-":
            return self.state.R_reference @ rot_y(-a)
        return self.state.R_reference @ rot_x(+a) @ rot_y(+a)

    def record(self, action: str) -> None:
        s = self.state
        m = self.profile_metrics()
        s.event_index += 1
        s.hist_event.append(s.event_index)
        s.hist_x_mid.append(m["x_mid"])
        s.hist_z_dev.append(m["z_mid"] - s.reference_profile.z_mid)
        s.hist_margin.append(m["margin"])
        s.hist_action.append(action)

    def noisy_unlabeled_endpoints(
        self, noise_std: float, scenario_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        e1, e2 = self.state.profile.endpoints()
        q1 = e1 + self.rng.normal(0.0, noise_std, size=2)
        q2 = e2 + self.rng.normal(0.0, noise_std, size=2)

        if scenario_id == 4 and self.state.profile.length < 95.0:
            if self.rng.random() < 0.50:
                q1, q2 = q2, q1
        return q1, q2

    def track_identity(
        self, q1: np.ndarray, q2: np.ndarray, ambiguity_threshold: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        s = self.state
        if s.previous_labeled_endpoints is None:
            e1, e2 = (q1, q2) if q1[0] <= q2[0] else (q2, q1)
            s.previous_labeled_endpoints = (e1.copy(), e2.copy())
            s.last_identity_costs = (0.0, np.inf)
            s.identity_ambiguous = False
            return e1, e2

        p1, p2 = s.previous_labeled_endpoints
        D1 = float(np.sum((q1 - p1) ** 2) + np.sum((q2 - p2) ** 2))
        D2 = float(np.sum((q2 - p1) ** 2) + np.sum((q1 - p2) ** 2))
        s.last_identity_costs = (D1, D2)
        s.identity_ambiguous = abs(D1 - D2) < ambiguity_threshold

        if s.identity_ambiguous:
            return None, None

        if D1 <= D2:
            e1, e2 = q1, q2
        else:
            e1, e2 = q2, q1
        s.previous_labeled_endpoints = (e1.copy(), e2.copy())
        return e1, e2

    # -------------------------------------------------------------------------
    # 分支与动作
    # -------------------------------------------------------------------------

    def select_branch(self, branch: str) -> None:
        s = self.state
        s.branch = branch
        self.return_reference(keep_message=False)
        s.controller_state = "SELECT_ROTATION_TARGET"
        s.message = f"已返回参考位姿，准备执行分支 {branch}。"
        self.record("select branch")

    def return_reference(self, keep_message: bool = True) -> None:
        s = self.state
        s.R_current = s.R_reference.copy()
        s.branch_progress_deg = 0.0
        s.combo_stage = 0
        s.profile = copy.deepcopy(s.reference_profile)
        s.g_hat[:] = np.nan
        s.probe_safe[:] = False
        s.selected_axis = None
        s.probe_done = False
        s.stable_count = self.stable_required
        s.seed_saved = False
        s.identity_ambiguous = False
        e1, e2 = s.profile.endpoints()
        s.previous_labeled_endpoints = (e1.copy(), e2.copy())
        s.last_valid = self.make_snapshot()
        s.controller_state = "SAVE_REFERENCE"
        if keep_message:
            s.message = "已返回R0参考位姿。返回过程不保存为新种子。"
        self.record("return R0")

    def rotate_one_step(
        self,
        step_deg: float,
        target_angle: float,
        noise_std: float,
        scenario_id: int,
        ambiguity_threshold: float,
    ) -> bool:
        s = self.state

        if s.branch == "R0":
            s.message = "R0是参考种子，不需要继续旋转。"
            return False

        if (
            abs(s.profile.x_mid) > self.x_tolerance
            or not self.safety_ok()
            or s.stable_count < self.stable_required
        ):
            s.controller_state = "WAIT_SERVO"
            s.message = (
                "当前轮廓尚未重新稳定，禁止继续旋转。"
                "应先完成平移伺服和安全检查。"
            )
            return False

        total_target = self.target_total_deg(target_angle)
        remaining = total_target - s.branch_progress_deg
        if remaining <= 1e-9:
            s.message = "当前分支已经达到目标旋转量。"
            return False

        step = min(step_deg, remaining)
        axis, sign = self.branch_axis_and_sign(target_angle)
        if axis is None:
            return False

        s.last_valid = self.make_snapshot()

        # 右乘：绕当前法兰局部轴旋转。
        dR = rot_x(np.deg2rad(sign * step)) if axis == "X" else rot_y(np.deg2rad(sign * step))
        s.R_current = s.R_current @ dR
        s.branch_progress_deg += step

        dx, dz, dphi, dlen = self.rotation_drift_per_deg(
            axis, sign, scenario_id
        )
        s.profile.x_mid += dx * step
        s.profile.z_mid += dz * step
        s.profile.tilt_deg += dphi * step
        s.profile.length += dlen * step

        s.g_hat[:] = np.nan
        s.probe_safe[:] = False
        s.selected_axis = None
        s.probe_done = False
        s.stable_count = 0

        q1, q2 = self.noisy_unlabeled_endpoints(noise_std, scenario_id)
        e1, e2 = self.track_identity(q1, q2, ambiguity_threshold)

        s.controller_state = "READ_PROFILE"
        if e1 is None or e2 is None:
            s.message = (
                "旋转后端点身份匹配模糊：|D1-D2|过小。"
                "当前帧不应保存，也不应继续旋转。"
            )
        elif not self.safety_ok():
            s.message = (
                "旋转后双边进入风险区或轮廓长度异常。"
                "按文档应立即回退last_valid_pose并减小旋转步长。"
            )
        else:
            s.message = (
                f"已绕法兰局部{axis}轴旋转{sign * step:+.1f}°。"
                "旋转已固定，下一步应试探平移轴并使x_mid→0。"
            )
        self.record(f"rotate {axis}{sign:+.0f}")
        return True

    def probe_axes(
        self,
        probe_mm: float,
        noise_std: float,
        scenario_id: int,
    ) -> bool:
        s = self.state
        if s.identity_ambiguous or not self.safety_ok():
            s.controller_state = "PROBE_BLOCKED"
            s.message = "当前双边或身份不可靠，不能进行平移试探。"
            return False

        g_true, h_true = self.true_sensitivity(
            s.branch_progress_deg, scenario_id
        )
        g_est = np.full(3, np.nan)
        safe = np.zeros(3, dtype=bool)
        base_profile = copy.deepcopy(s.profile)

        for j in range(3):
            dx = g_true[j] * probe_mm + self.rng.normal(0.0, noise_std)
            dz = h_true[j] * probe_mm
            trial = copy.deepcopy(base_profile)
            trial.x_mid += dx
            trial.z_mid += dz
            e1, e2 = trial.endpoints()
            margin = min(self.domain.margin(e1), self.domain.margin(e2))
            safe[j] = (
                self.domain.contains(e1)
                and self.domain.contains(e2)
                and margin >= 0.5 * self.margin_required
                and self.length_min <= trial.length <= self.length_max
            )
            if safe[j]:
                g_est[j] = dx / probe_mm

        s.g_hat = g_est
        s.probe_safe = safe
        candidates = [
            j
            for j in range(3)
            if safe[j] and np.isfinite(g_est[j]) and abs(g_est[j]) >= self.g_min
        ]

        if not candidates:
            s.selected_axis = None
            s.probe_done = False
            s.controller_state = "PROBE_FAILED"
            s.message = (
                "没有找到既安全又具有足够|g_hat|的平移轴。"
                "此时直接使用Δq=-λx_mid/g_hat会数值不稳定。"
            )
            self.record("probe failed")
            return False

        s.selected_axis = max(candidates, key=lambda j: abs(g_est[j]))
        s.probe_done = True
        s.controller_state = "TRANSLATION_SERVO"
        s.message = (
            f"试探完成，选择{AXES[s.selected_axis]}轴："
            f"g_hat={g_est[s.selected_axis]:+.3f}。"
            "所有试探均从同一有效位姿出发并返回原位。"
        )
        self.record("probe axes")
        return True

    def servo_one_step(
        self,
        gain: float,
        q_max: float,
        noise_std: float,
        update_rate: float,
        scenario_id: int,
        ambiguity_threshold: float,
    ) -> bool:
        s = self.state
        if not s.probe_done or s.selected_axis is None:
            s.message = "尚未完成安全平移试探，不能执行伺服。"
            return False

        j = s.selected_axis
        g_est = s.g_hat[j]
        if not np.isfinite(g_est) or abs(g_est) < self.g_min:
            s.controller_state = "SERVO_UNSTABLE"
            s.message = "当前g_hat过小，除法控制会放大噪声，应重新选轴或扩展特征。"
            return False

        x_before = s.profile.x_mid
        dq = float(np.clip(-gain * x_before / g_est, -q_max, q_max))
        if abs(dq) < 1e-6:
            dq = 0.0

        g_true, h_true = self.true_sensitivity(
            s.branch_progress_deg, scenario_id
        )
        s.profile.x_mid += g_true[j] * dq + self.rng.normal(0.0, noise_std)
        s.profile.z_mid += h_true[j] * dq

        q1, q2 = self.noisy_unlabeled_endpoints(noise_std, scenario_id)
        e1, e2 = self.track_identity(q1, q2, ambiguity_threshold)

        if abs(dq) > 1e-9:
            g_measured = (s.profile.x_mid - x_before) / dq
            s.g_hat[j] = (
                (1.0 - update_rate) * s.g_hat[j]
                + update_rate * g_measured
            )

        if e1 is None or e2 is None:
            s.controller_state = "IDENTITY_AMBIGUOUS"
            s.message = "平移后端点身份变得模糊，当前帧不保存。"
            self.record("servo identity ambiguous")
            return False

        if not self.safety_ok():
            s.controller_state = "SAFETY_FAILED"
            s.message = (
                "平移伺服虽然在控制x_mid，但端点已触碰安全域或轮廓异常。"
                "这说明x_mid→0本身不足以保证双边安全。"
            )
            self.record("servo unsafe")
            return False

        if abs(s.profile.x_mid) <= self.x_tolerance:
            s.stable_count += 1
        else:
            s.stable_count = 0

        if s.stable_count >= self.stable_required:
            s.last_valid = self.make_snapshot()
            s.controller_state = "CHECK_FEATURE_SAFETY"
            s.message = (
                f"x_mid已收敛到{s.profile.x_mid:+.2f} mm，"
                f"连续稳定计数={s.stable_count}。"
                "当前状态可以作为下一次小步旋转的起点。"
            )
        else:
            s.controller_state = "TRANSLATION_SERVO"
            s.message = (
                f"沿{AXES[j]}移动Δq={dq:+.2f} mm；"
                f"x_mid={s.profile.x_mid:+.2f} mm，继续伺服。"
            )

        self.record(f"servo {AXES[j]}")
        return True

    def rollback(self, step_slider: Optional[Slider] = None) -> bool:
        s = self.state
        if s.last_valid is None:
            s.message = "没有可用的last_valid_pose。"
            return False
        self.restore_snapshot(s.last_valid)
        s.identity_ambiguous = False
        s.controller_state = "ROLLBACK_AND_REDUCE_STEP"
        s.message = "已回退到last_valid_pose。建议将旋转步长减半后重试。"
        if step_slider is not None:
            step_slider.set_val(max(step_slider.val * 0.5, step_slider.valmin))
        self.record("rollback")
        return True

    def save_seed(self, target_angle: float) -> bool:
        s = self.state
        total_target = self.target_total_deg(target_angle)
        reached = s.branch_progress_deg >= total_target - 1e-6
        if (
            reached
            and abs(s.profile.x_mid) <= self.x_tolerance
            and self.safety_ok()
            and s.stable_count >= self.stable_required
            and not s.identity_ambiguous
        ):
            if s.branch not in s.saved_branches:
                s.saved_branches.append(s.branch)
            s.seed_saved = True
            s.controller_state = "SAVE_SEED"
            s.message = (
                f"分支{s.branch}通过双边、安全域、稳定性和目标角度检查，"
                "已保存为种子。返回R0的途中不新增种子。"
            )
            self.record("save seed")
            return True

        s.message = (
            "当前状态尚不满足种子保存条件："
            "需同时达到目标角度、x_mid收敛、安全余量足够且连续稳定。"
        )
        return False

    def auto_current_branch(
        self,
        step_deg: float,
        target_angle: float,
        probe_mm: float,
        gain: float,
        q_max: float,
        noise_std: float,
        update_rate: float,
        scenario_id: int,
        ambiguity_threshold: float,
    ) -> None:
        s = self.state
        max_outer = 40
        max_servo = 18
        local_step = step_deg

        for _ in range(max_outer):
            total_target = self.target_total_deg(target_angle)

            if (
                s.branch_progress_deg >= total_target - 1e-6
                and abs(s.profile.x_mid) <= self.x_tolerance
                and self.safety_ok()
                and s.stable_count >= self.stable_required
            ):
                self.save_seed(target_angle)
                return

            ok = self.rotate_one_step(
                local_step,
                target_angle,
                noise_std,
                scenario_id,
                ambiguity_threshold,
            )
            if not ok:
                if s.identity_ambiguous or not self.safety_ok():
                    self.rollback()
                    local_step *= 0.5
                    if local_step < 0.4:
                        s.message = "自动执行失败：旋转步长已缩小到下限。"
                        return
                    continue
                return

            if s.identity_ambiguous or not self.safety_ok():
                self.rollback()
                local_step *= 0.5
                if local_step < 0.4:
                    s.message = "自动执行失败：无法在当前分支维持双边。"
                    return
                continue

            if not self.probe_axes(probe_mm, noise_std, scenario_id):
                self.rollback()
                local_step *= 0.5
                if local_step < 0.4:
                    s.message = "自动执行失败：没有稳定平移方向。"
                    return
                continue

            servo_ok = False
            for _ in range(max_servo):
                self.servo_one_step(
                    gain,
                    q_max,
                    noise_std,
                    update_rate,
                    scenario_id,
                    ambiguity_threshold,
                )
                if not self.safety_ok() or s.identity_ambiguous:
                    break
                if (
                    abs(s.profile.x_mid) <= self.x_tolerance
                    and s.stable_count >= self.stable_required
                ):
                    servo_ok = True
                    break

            if not servo_ok:
                self.rollback()
                local_step *= 0.5
                if local_step < 0.4:
                    s.message = (
                        "自动执行失败：内层平移伺服无法在安全约束下收敛。"
                    )
                    return

        s.message = "自动执行达到最大循环次数，未完成当前分支。"


# =============================================================================
# 绘图界面
# =============================================================================

class Visualizer:
    def __init__(self) -> None:
        self.model = SeedCollectionModel()

        self.fig = plt.figure(figsize=(17.0, 10.2))
        self.fig.subplots_adjust(
            left=0.045,
            right=0.985,
            top=0.915,
            bottom=0.235,
            wspace=0.28,
            hspace=0.34,
        )
        gs = self.fig.add_gridspec(2, 3)
        self.ax_plan = self.fig.add_subplot(gs[0, 0])
        self.ax_frame = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_profile = self.fig.add_subplot(gs[0, 2])
        self.ax_probe = self.fig.add_subplot(gs[1, 0])
        self.ax_history = self.fig.add_subplot(gs[1, 1])
        self.ax_status = self.fig.add_subplot(gs[1, 2])

        self.fig.suptitle(
            "Phase 0b可视化：预设局部轴旋转 + 无标定双边特征平移伺服",
            fontsize=15,
        )
        self.fig.text(
            0.5,
            0.203,
            "控制层简化仿真：不使用手眼关系；只演示“旋转制造姿态差异、"
            "轮廓反馈学习局部平移方向、安全约束决定是否继续”。",
            ha="center",
            fontsize=9,
        )

        self._build_controls()
        self.update()

        if CJK_FONT_PATH:
            print(f"[字体] 使用中文字体：{CJK_FONT_PATH}")

    # -------------------------------------------------------------------------
    # 控件
    # -------------------------------------------------------------------------

    def _build_controls(self) -> None:
        # 第一行滑块
        specs = [
            ("目标角 θ / °", 6.0, 20.0, 12.0),
            ("旋转步长 / °", 0.5, 4.0, 2.5),
            ("试探量 / mm", 0.5, 5.0, 2.0),
            ("伺服增益 λ", 0.15, 1.2, 0.65),
            ("平移限幅 / mm", 1.0, 20.0, 8.0),
        ]
        positions = [
            [0.050, 0.150, 0.165, 0.022],
            [0.245, 0.150, 0.165, 0.022],
            [0.440, 0.150, 0.165, 0.022],
            [0.635, 0.150, 0.165, 0.022],
            [0.830, 0.150, 0.135, 0.022],
        ]
        self.sliders: Dict[str, Slider] = {}
        keys = ["target", "step", "probe", "gain", "qmax"]
        for key, spec, pos in zip(keys, specs, positions):
            ax = self.fig.add_axes(pos)
            slider = Slider(ax, spec[0], spec[1], spec[2], valinit=spec[3])
            slider.on_changed(lambda _v: self.update())
            self.sliders[key] = slider

        # 第二行滑块
        specs2 = [
            ("测量噪声 / mm", 0.0, 3.0, 0.35),
            ("灵敏度更新率", 0.0, 1.0, 0.35),
            ("身份模糊阈值", 0.0, 2500.0, 180.0),
        ]
        positions2 = [
            [0.050, 0.105, 0.210, 0.022],
            [0.290, 0.105, 0.210, 0.022],
            [0.530, 0.105, 0.210, 0.022],
        ]
        keys2 = ["noise", "update", "ambiguity"]
        for key, spec, pos in zip(keys2, specs2, positions2):
            ax = self.fig.add_axes(pos)
            slider = Slider(ax, spec[0], spec[1], spec[2], valinit=spec[3])
            slider.on_changed(lambda _v: self.update())
            self.sliders[key] = slider

        # 分支选择
        ax_branch = self.fig.add_axes([0.765, 0.082, 0.095, 0.105])
        self.branch_radio = RadioButtons(
            ax_branch,
            BRANCHES,
            active=BRANCHES.index("Rx+"),
        )
        self.branch_radio.on_clicked(self._on_branch)

        # 场景选择
        scenario_labels = [
            "正常",
            "Z漂移",
            "弱灵敏度",
            "g变号",
            "身份模糊",
        ]
        ax_scenario = self.fig.add_axes([0.875, 0.082, 0.105, 0.105])
        self.scenario_radio = RadioButtons(
            ax_scenario, scenario_labels, active=0
        )
        self.scenario_radio.on_clicked(self._on_scenario)
        self.scenario_id = 0

        # 操作按钮
        buttons = [
            ("旋转一步", [0.050, 0.028, 0.105, 0.040], self._rotate),
            ("试探三轴", [0.170, 0.028, 0.105, 0.040], self._probe),
            ("伺服一步", [0.290, 0.028, 0.105, 0.040], self._servo),
            ("自动当前分支", [0.410, 0.028, 0.135, 0.040], self._auto),
            ("回退", [0.560, 0.028, 0.085, 0.040], self._rollback),
            ("保存种子", [0.660, 0.028, 0.105, 0.040], self._save_seed),
            ("返回R0", [0.780, 0.028, 0.090, 0.040], self._return),
            ("完全复位", [0.885, 0.028, 0.095, 0.040], self._reset),
        ]
        self.buttons: List[Button] = []
        for label, pos, callback in buttons:
            ax = self.fig.add_axes(pos)
            b = Button(ax, label)
            b.on_clicked(callback)
            self.buttons.append(b)

    def _values(self) -> Dict[str, float]:
        return {key: float(slider.val) for key, slider in self.sliders.items()}

    def _on_branch(self, label: str) -> None:
        self.model.select_branch(label)
        self.update()

    def _on_scenario(self, label: str) -> None:
        labels = ["正常", "Z漂移", "弱灵敏度", "g变号", "身份模糊"]
        self.scenario_id = labels.index(label)
        self.model.return_reference(keep_message=False)
        self.model.state.message = (
            f"已切换故障预设：{self.model.scenario_name(self.scenario_id)}。"
        )
        self.update()

    def _rotate(self, _event) -> None:
        v = self._values()
        self.model.rotate_one_step(
            v["step"],
            v["target"],
            v["noise"],
            self.scenario_id,
            v["ambiguity"],
        )
        self.update()

    def _probe(self, _event) -> None:
        v = self._values()
        self.model.probe_axes(
            v["probe"],
            v["noise"],
            self.scenario_id,
        )
        self.update()

    def _servo(self, _event) -> None:
        v = self._values()
        self.model.servo_one_step(
            v["gain"],
            v["qmax"],
            v["noise"],
            v["update"],
            self.scenario_id,
            v["ambiguity"],
        )
        self.update()

    def _auto(self, _event) -> None:
        v = self._values()
        self.model.auto_current_branch(
            v["step"],
            v["target"],
            v["probe"],
            v["gain"],
            v["qmax"],
            v["noise"],
            v["update"],
            self.scenario_id,
            v["ambiguity"],
        )
        self.update()

    def _rollback(self, _event) -> None:
        self.model.rollback(self.sliders["step"])
        self.update()

    def _save_seed(self, _event) -> None:
        self.model.save_seed(self.sliders["target"].val)
        self.update()

    def _return(self, _event) -> None:
        self.model.return_reference()
        self.update()

    def _reset(self, _event) -> None:
        branch = self.model.state.branch
        self.model = SeedCollectionModel()
        self.model.state.branch = branch
        self.branch_radio.set_active(BRANCHES.index(branch))
        self.scenario_id = 0
        self.scenario_radio.set_active(0)
        self.update()

    # -------------------------------------------------------------------------
    # 绘制
    # -------------------------------------------------------------------------

    def update(self) -> None:
        self._draw_plan()
        self._draw_frame()
        self._draw_profile()
        self._draw_probe()
        self._draw_history()
        self._draw_status()
        apply_cjk_font(self.fig)
        self.fig.canvas.draw_idle()

    def _draw_plan(self) -> None:
        ax = self.ax_plan
        ax.clear()
        s = self.model.state
        target = self.sliders["target"].val

        coords = {
            "R0": (0.0, 0.0),
            "Rx+": (1.0, 0.0),
            "Rx-": (-1.0, 0.0),
            "Ry+": (0.0, 1.0),
            "Ry-": (0.0, -1.0),
            "Rx+Ry+": (1.0, 1.0),
        }

        # 星形连接
        for b in ["Rx+", "Rx-", "Ry+", "Ry-"]:
            x0, y0 = coords["R0"]
            x1, y1 = coords[b]
            ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=1.0)
        ax.plot(
            [coords["R0"][0], coords["Rx+"][0], coords["Rx+Ry+"][0]],
            [coords["R0"][1], coords["Rx+"][1], coords["Rx+Ry+"][1]],
            linestyle=":",
            linewidth=1.4,
        )

        for b, (x, y) in coords.items():
            marker = "s" if b in s.saved_branches else "o"
            size = 95 if b == s.branch else 58
            ax.scatter([x], [y], marker=marker, s=size)
            ax.text(x + 0.05, y + 0.05, b, fontsize=9)

        total_target = self.model.target_total_deg(target)
        frac = 0.0 if total_target <= 0 else min(s.branch_progress_deg / total_target, 1.0)

        if s.branch != "R0":
            if s.branch == "Rx+Ry+":
                if frac <= 0.5:
                    local = frac / 0.5
                    p = np.array(coords["R0"]) * (1 - local) + np.array(coords["Rx+"]) * local
                else:
                    local = (frac - 0.5) / 0.5
                    p = np.array(coords["Rx+"]) * (1 - local) + np.array(coords["Rx+Ry+"]) * local
            else:
                p = np.array(coords["R0"]) * (1 - frac) + np.array(coords[s.branch]) * frac
            ax.scatter([p[0]], [p[1]], marker="x", s=120, label="当前进度")

        ax.set_title("外层：星形种子旋转计划")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.30, 1.35)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.text(
            0.02,
            0.02,
            f"当前分支：{s.branch}\n"
            f"累计进度：{s.branch_progress_deg:.1f}° / {total_target:.1f}°\n"
            f"已保存种子：{', '.join(s.saved_branches)}",
            transform=ax.transAxes,
            va="bottom",
            bbox=dict(boxstyle="round", alpha=0.10),
        )

    def _draw_frame(self) -> None:
        ax = self.ax_frame
        ax.clear()
        s = self.model.state
        target_R = self.model.target_rotation(self.sliders["target"].val)

        def draw_frame(R: np.ndarray, scale: float, prefix: str, linestyle: str) -> None:
            origin = np.zeros(3)
            for i, label in enumerate(["x", "y", "z"]):
                end = origin + scale * R[:, i]
                ax.plot(
                    [0, end[0]],
                    [0, end[1]],
                    [0, end[2]],
                    linestyle=linestyle,
                    linewidth=2.0,
                )
                ax.text(end[0], end[1], end[2], f"{prefix}{label}", fontsize=8)

        draw_frame(target_R, 0.92, "目标:", "--")
        draw_frame(s.R_current, 0.78, "当前:", "-")

        if s.selected_axis is not None:
            d = s.R_current[:, s.selected_axis]
            ax.quiver(
                0, 0, 0,
                d[0], d[1], d[2],
                length=1.15,
                normalize=True,
                arrow_length_ratio=0.12,
                linewidth=2.4,
            )
            ax.text(*(1.05 * d), f"伺服轴 {AXES[s.selected_axis]}", fontsize=9)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.15, 1.15)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("B-X")
        ax.set_ylabel("B-Y")
        ax.set_zlabel("B-Z")
        ax.set_title("当前法兰局部轴与目标朝向")
        ax.view_init(elev=23, azim=-48)
        ax.text2D(
            0.02,
            0.02,
            "右乘局部旋转：R_next=R·Rx/Ry\n"
            "旋转一步后，法兰姿态固定；\n"
            "内层控制只允许法兰平移。",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", alpha=0.10),
            fontsize=8,
        )

    def _draw_profile(self) -> None:
        ax = self.ax_profile
        ax.clear()
        s = self.model.state
        domain = self.model.domain

        vertices = domain.vertices()
        closed = np.vstack([vertices, vertices[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="人工安全梯形")

        e1, e2 = s.profile.endpoints()
        ax.plot([e1[0], e2[0]], [e1[1], e2[1]], linewidth=2.5, label="当前轮廓")
        ax.scatter([e1[0]], [e1[1]], marker="o", s=75, label="e1")
        ax.scatter([e2[0]], [e2[1]], marker="^", s=75, label="e2")
        ax.scatter([s.profile.x_mid], [s.profile.z_mid], marker="x", s=80, label="中点")

        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.annotate(
            f"x_mid={s.profile.x_mid:+.1f} mm",
            (s.profile.x_mid, s.profile.z_mid),
            xytext=(8, 12),
            textcoords="offset points",
        )

        m = self.model.profile_metrics()
        ax.set_title("传感器X-Z平面：双边特征与硬安全约束")
        ax.set_xlabel("x_S / mm")
        ax.set_ylabel("z_S / mm")
        ax.set_xlim(-205, 205)
        ax.set_ylim(225, 685)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        ax.text(
            0.98,
            0.98,
            f"x_mid={m['x_mid']:+.2f} mm\n"
            f"z_mid={m['z_mid']:.2f} mm\n"
            f"L_p={m['length']:.1f} mm\n"
            f"最小安全余量={m['margin']:.1f} mm\n"
            f"安全={'是' if self.model.safety_ok() else '否'}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.10),
        )

    def _draw_probe(self) -> None:
        ax = self.ax_probe
        ax.clear()
        s = self.model.state
        g_true, _ = self.model.true_sensitivity(
            s.branch_progress_deg, self.scenario_id
        )
        x = np.arange(3)
        width = 0.36
        ax.bar(x - width / 2, g_true, width=width, label="真实局部g（未知）")
        shown_hat = np.nan_to_num(s.g_hat, nan=0.0)
        ax.bar(x + width / 2, shown_hat, width=width, label="试探估计g_hat")

        for j in range(3):
            if not np.isfinite(s.g_hat[j]):
                ax.text(j + width / 2, 0.02, "未估计", ha="center", fontsize=8, rotation=90)
            elif not s.probe_safe[j]:
                ax.text(j + width / 2, s.g_hat[j], "不安全", ha="center", fontsize=8)

        if s.selected_axis is not None:
            ax.scatter(
                [s.selected_axis],
                [s.g_hat[s.selected_axis]],
                marker="*",
                s=160,
                label="自动选择轴",
            )

        ax.axhline(+self.model.g_min, linestyle=":", linewidth=1.0)
        ax.axhline(-self.model.g_min, linestyle=":", linewidth=1.0)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xticks(x, AXES)
        ax.set_ylabel("Δx_mid / Δq")
        ax.set_title("无标定试探：学习法兰平移到x_mid的局部灵敏度")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.02,
            "控制器看不到“真实g”；\n"
            "只根据小幅试探估计g_hat，\n"
            "并选安全且|g_hat|最大的轴。",
            transform=ax.transAxes,
            va="bottom",
            bbox=dict(boxstyle="round", alpha=0.08),
            fontsize=8,
        )

    def _draw_history(self) -> None:
        ax = self.ax_history
        ax.clear()
        s = self.model.state

        if s.hist_event:
            ax.plot(s.hist_event, s.hist_x_mid, marker="o", markersize=3, label="x_mid")
            ax.plot(s.hist_event, s.hist_z_dev, linestyle="--", label="z_mid-z_ref")
            ax.plot(s.hist_event, s.hist_margin, linestyle=":", label="安全余量")
        ax.axhline(0.0, linewidth=0.8)
        ax.axhline(+self.model.x_tolerance, linestyle="--", linewidth=0.8)
        ax.axhline(-self.model.x_tolerance, linestyle="--", linewidth=0.8)
        ax.axhline(self.model.margin_required, linestyle=":", linewidth=0.8)
        ax.set_xlabel("动作序号")
        ax.set_ylabel("mm")
        ax.set_title("闭环历史：x控制收敛不等于z和安全余量一定安全")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        if s.hist_action:
            ax.text(
                0.02,
                0.98,
                f"最近动作：{s.hist_action[-1]}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round", alpha=0.08),
                fontsize=8,
            )

    def _draw_status(self) -> None:
        ax = self.ax_status
        ax.clear()
        ax.axis("off")
        s = self.model.state
        m = self.model.profile_metrics()
        D1, D2 = s.last_identity_costs
        target = self.sliders["target"].val
        total_target = self.model.target_total_deg(target)
        current_to_target = relative_angle_deg(
            s.R_current, self.model.target_rotation(target)
        )

        selected = (
            "未选择"
            if s.selected_axis is None
            else f"{AXES[s.selected_axis]}，g_hat={s.g_hat[s.selected_axis]:+.3f}"
        )
        identity = (
            "模糊，禁止保存"
            if s.identity_ambiguous
            else "连续匹配有效"
        )

        text = (
            "当前状态\n"
            "────────────\n"
            f"状态机：{s.controller_state}\n"
            f"故障预设：{self.model.scenario_name(self.scenario_id)}\n"
            f"分支：{s.branch}\n"
            f"累计旋转：{s.branch_progress_deg:.1f}/{total_target:.1f}°\n"
            f"距目标姿态：{current_to_target:.2f}°\n"
            f"平移轴：{selected}\n"
            f"x_mid：{m['x_mid']:+.2f} mm\n"
            f"z_mid：{m['z_mid']:.2f} mm\n"
            f"安全余量：{m['margin']:.2f} mm\n"
            f"连续稳定：{s.stable_count}/{self.model.stable_required}\n"
            f"身份状态：{identity}\n"
            f"D1={D1:.1f}，D2={D2:.1f}\n\n"
            "文档逻辑\n"
            "────────────\n"
            "① 预设局部轴小步旋转\n"
            "② 立即读取真实双边轮廓\n"
            "③ 固定旋转，不再继续转\n"
            "④ 安全试探法兰局部平移轴\n"
            "⑤ 用Δq=-λx_mid/g_hat修正\n"
            "⑥ x_mid收敛且安全域满足\n"
            "⑦ 才允许下一小步旋转\n"
            "⑧ 丢边/身份模糊立即回退\n\n"
            "最近解释\n"
            "────────────\n"
            f"{s.message}"
        )
        ax.text(
            0.0,
            1.0,
            text,
            va="top",
            fontsize=9,
            linespacing=1.25,
        )

    def save(self, path: str) -> None:
        apply_cjk_font(self.fig)
        self.fig.savefig(path, dpi=170, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="保存静态预览并退出",
    )
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
