#!/usr/bin/env python3
"""
edge_visual_servo.py — 基于2D轮廓特征的边缘视觉伺服

核心思想:
  FOV三角形沿平板边缘自主推进，仅凭2D轮廓点（板面点+边缘断点）作为伺服信号。
  在推进过程中主动"扭动"（调制pitch/yaw）探索信息丰富的位姿，
  条件数好的位姿自动记录为标定数据。

自动化流程:
  人工初始化: 将FOV三角形穿过平板的一个边缘 → 系统自动:
    Phase 1 SEARCH_EDGE:    搜索边缘（无断点→单断点）
    Phase 2 ALONG_EDGE1:    沿边1推进 + 扭动采集
    Phase 3 APPROACH_CORNER: 检测角点逼近（双断点出现）
    Phase 4 TURN_CORNER:   绕过角点过渡到边2
    Phase 5 ALONG_EDGE2:    沿边2推进 + 扭动采集
    Phase 6 DONE:           采集完成

工作模式: 纯仿真（对接现有 compute_fov_plate_scanline）

依赖:
  Num2 现有模块: reproduction_scene, corner_scene
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ============================================================================
# 导入已有基础模块
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reproduction_scene import (
    compute_fov_plate_scanline, make_transform, rodrigues,
    rpy_to_matrix, generate_hand_eye_gt
)
from corner_scene import generate_corner_plane, so3_exp, so3_log


# ============================================================================
# 1. 伺服特征提取 — 从2D扫描线中提取控制信号
# ============================================================================

@dataclass
class ProfileFeatures:
    """从单帧扫描线中提取的全部伺服特征"""
    # 原始数据
    n_pts: int = 0                      # 板面点数
    has_intersection: bool = False
    
    # 板面点统计 (Sensor系)
    x_min: float = 0.0
    x_max: float = 0.0
    x_center: float = 0.0
    x_span: float = 0.0
    z_mean: float = 0.0
    z_std: float = 0.0
    
    # 端点信息
    has_e1: bool = False
    has_e2: bool = False
    e1_x: float = 0.0
    e1_z: float = 0.0
    e2_x: float = 0.0  
    e2_z: float = 0.0
    
    # 高级特征
    edge_gap: Optional[float] = None    # 双断点时两断点间距
    n_endpoints: int = 0                # 断点数量 (0, 1, 2)


def extract_profile_features(sl: dict) -> ProfileFeatures:
    """从 compute_fov_plate_scanline 的输出中提取伺服特征
    
    Args:
        sl: scanline result dict from compute_fov_plate_scanline
    
    Returns:
        ProfileFeatures instance
    """
    f = ProfileFeatures()
    
    if not sl['has_intersection']:
        f.has_intersection = False
        return f
    
    f.has_intersection = True
    
    # 板面点统计
    pts_S = sl['scan_pts_S']  # N×3
    f.n_pts = len(pts_S)
    
    if f.n_pts > 0:
        x = pts_S[:, 0]
        z = pts_S[:, 2]
        f.x_min = float(x.min())
        f.x_max = float(x.max())
        f.x_center = float((x.min() + x.max()) / 2)
        f.z_mean = float(z.mean())
        f.z_std = float(z.std())
        f.x_span = float(x.max() - x.min())
    
    # 端点信息
    endpoints_S = sl['endpoints_S']
    f.n_endpoints = len(endpoints_S)
    
    for et, pt in endpoints_S:
        if et == 'e1':
            f.has_e1 = True
            f.e1_x = pt[0]
            f.e1_z = pt[2]
        elif et == 'e2':
            f.has_e2 = True
            f.e2_x = pt[0]
            f.e2_z = pt[2]
    
    # 双断点间距
    if f.has_e1 and f.has_e2:
        f.edge_gap = abs(f.e1_x - f.e2_x)
    
    return f


# ============================================================================
# 2. 坐标框架工具 — 传感器系与基系之间的增量运动映射
# ============================================================================

def incremental_motion_in_sensor_frame(
    T_B_H: np.ndarray,       # 当前手部位姿 (4×4)
    T_S_H: np.ndarray,       # 手眼变换 (4×4), X_gt
    delta_sensor: np.ndarray # 6-DOF: [v_x, v_y, v_z, ω_x, ω_y, ω_z] 在传感器系
) -> np.ndarray:
    """在传感器系中施加增量运动，返回新的手部位姿
    
    传感器系中的小位移/旋转 → 映射到手系 → 更新 T_B_H
    
    Args:
        T_B_H: 当前手部位姿
        T_S_H: 手眼变换
        delta_sensor: 6自由度增量 [dx, dy, dz, droll, dpitch, dyaw] 在传感器系
    
    Returns:
        新的 T_B_H
    """
    # 传感器增量齐次变换
    dv = delta_sensor[:3]
    domega = delta_sensor[3:6]
    
    # 平移部分在传感器系
    t_S = dv
    
    # 旋转部分: 小角近似
    omega_norm = np.linalg.norm(domega)
    if omega_norm < 1e-12:
        R_delta_S = np.eye(3)
    else:
        axis = domega / omega_norm
        R_delta_S = rodrigues(axis, omega_norm)
    
    T_delta_S = make_transform(R_delta_S, t_S)
    
    # 变换到手系: T_delta_H = T_S_H @ T_delta_S @ inv(T_S_H)
    T_SH_inv = np.linalg.inv(T_S_H)
    T_delta_H = T_S_H @ T_delta_S @ T_SH_inv
    
    # 更新手部位姿
    T_B_H_new = T_B_H @ T_delta_H
    
    return T_B_H_new


# ============================================================================
# 3. 单边伺服控制器 — 保持FOV在边缘上的低层控制
# ============================================================================

@dataclass
class EdgeServoParams:
    """单边伺服控制器参数"""
    # 期望值
    target_e1_x: float = 0.0           # 期望e1断点在传感器x轴位置 (mm, 0=中心)
    target_z_mean: float = 0.50        # 期望板面点平均距离 (m)
    target_n_pts_min: int = 5          # 最少板面点数
    
    # PID增益 (特征→动作的线性映射)
    K_x: float = 0.0002                # e1_x偏差→沿v_B移动 (m/px)
    K_z: float = 0.001                 # z_mean偏差→沿n_B移动 (m/m)
    K_wiggle_amp: float = 0.5          # 适用于调节wiggle上层振幅
    
    # 安全限幅
    max_trans_step: float = 0.02       # 单步最大平移 (m)
    max_rot_step: float = 0.15         # 单步最大旋转 (rad ≈ 8.6°)


class EdgeServoController:
    """低层边缘伺服 — 仅用2D轮廓特征闭环保持FOV在边上"""
    
    def __init__(self, scene: dict, T_S_H: np.ndarray, params: EdgeServoParams = None):
        self.scene = scene
        self.C = scene['C']
        self.n_B = scene['n_B']
        self.u_B = scene['u_B']
        self.v_B = scene['v_B']
        self.pw = scene['w']
        self.ph = scene['h']
        self.T_S_H = T_S_H
        self.params = params or EdgeServoParams()
        
        # 伺服状态
        self.last_features = None
        self.step_count = 0
    
    def compute_servo_command(
        self, features: ProfileFeatures, target_e1_x: float = None
    ) -> np.ndarray:
        """根据当前轮廓特征计算传感器系中的增量运动命令
        
        Args:
            features: 当前帧的轮廓特征
            target_e1_x: 可选的e1目标位置，默认使用params.target_e1_x
            
        Returns:
            delta_sensor: [dx, dy, dz, droll, dpitch, dyaw] 在传感器系
        """
        self.last_features = features
        self.step_count += 1
        params = self.params
        
        if target_e1_x is None:
            target_e1_x = params.target_e1_x
        
        delta = np.zeros(6)
        
        if not features.has_intersection or features.n_pts < 1:
            # 无交线 — 沿传感器-Z方向后退（找平板）
            delta[2] = -0.01  # dz negative = move back in sensor z = away from plate
            return delta
        
        # 1. 横向微调: e1_x偏差 → 沿传感器X方向移动
        if features.has_e1:
            e1_err = features.e1_x - target_e1_x
            delta[0] = -params.K_x * e1_err  # 负反馈: 偏右则左移
        
        # 2. 纵向微调: z_mean偏差 → 沿传感器Z方向移动
        z_err = features.z_mean - params.target_z_mean
        delta[2] = -params.K_z * z_err  # 太远则靠近
        
        # 3. 限幅
        delta[:3] = np.clip(delta[:3], -params.max_trans_step, params.max_trans_step)
        delta[3:6] = np.clip(delta[3:6], -params.max_rot_step, params.max_rot_step)
        
        return delta
    
    def apply_servo(
        self, T_B_H: np.ndarray, features: ProfileFeatures, target_e1_x: float = None
    ) -> np.ndarray:
        """伺服一步: 计算增量并应用到机器人与，返回新位姿
        
        Args:
            T_B_H: 当前手部位姿
            features: 当前轮廓特征
            
        Returns:
            新的 T_B_H
        """
        delta = self.compute_servo_command(features, target_e1_x)
        return incremental_motion_in_sensor_frame(T_B_H, self.T_S_H, delta)


# ============================================================================
# 4. 主动探索器（扭动）— 叠加强制姿态调制
# ============================================================================

@dataclass
class ActiveExplorerParams:
    """主动探索参数"""
    # 主wiggle: 调制pitch (绕传感器X轴)
    pitch_amplitude: float = 0.35      # ±20° (rad)
    pitch_period: int = 8              # 步数每周期
    
    # 副wiggle: 调制yaw (绕传感器Z轴)
    yaw_amplitude: float = 0.20        # ±11.5° (rad)
    yaw_period: int = 12               # 步数每周期
    
    # 渐进推进
    forward_speed: float = 0.003       # 沿传感器X的微量前馈 (m/步)
    stagger_amplitude: float = 0.002   # 侧向抖动 (m)
    stagger_period: int = 5


class ActiveExplorer:
    """主动探索器: 在伺服基础上叠加姿态和位置调制
    
    两层架构:
      **基坐标系推进**: 沿边缘方向在基坐标系中直接移动(forward_speed)
      **传感器系姿态调制**: pitch/yaw 正弦波来改变传感器相对平板的视角
    
    这个设计解决了"传感器系前向运动不沿边缘"的根本问题。
    推进在已知的基坐标系边缘方向进行，伺服在传感器系做横向保持。
    """
    
    def __init__(self, params: ActiveExplorerParams = None,
                 u_B: np.ndarray = None, v_B: np.ndarray = None):
        self.params = params or ActiveExplorerParams()
        self.step_count = 0
        # 平板边缘方向 (基坐标系)
        self.u_B = u_B
        self.v_B = v_B
    
    def get_base_forward_delta(self, phase: str) -> np.ndarray:
        """返回基坐标系中的平移增量 (3D vector)
        
        沿边缘方向推进，方向根据相位自动选择。
        """
        p = self.params
        if phase == 'ALONG_EDGE1' and self.u_B is not None:
            # 边1: 沿 -u_B 向角点推进
            return -p.forward_speed * (self.u_B / np.linalg.norm(self.u_B))
        elif phase == 'ALONG_EDGE2' and self.v_B is not None:
            # 边2: 沿 -v_B 向角点推进
            return -p.forward_speed * (self.v_B / np.linalg.norm(self.v_B))
        return np.zeros(3)
    
    def get_sensor_wiggle_delta(self, phase: str) -> np.ndarray:
        """返回传感器系中的姿态调制增量 [0,0,0, droll, dpitch, dyaw]
        
        纯姿态调制，不影响平移。
        """
        p = self.params
        t = self.step_count
        
        if phase not in ('ALONG_EDGE1', 'ALONG_EDGE2'):
            return np.zeros(6)
        
        delta = np.zeros(6)
        pitch = p.pitch_amplitude * np.sin(2 * np.pi * t / p.pitch_period)
        yaw = p.yaw_amplitude * np.sin(2 * np.pi * t / p.yaw_period + 1.2)
        delta[4] = pitch
        delta[5] = yaw
        
        return delta
    
    def step(self, phase: str) -> tuple:
        """返回 (base_translation_delta, sensor_wiggle_delta)
        
        base_translation_delta: [dx, dy, dz] 在基坐标系
        sensor_wiggle_delta: [0,0,0, droll, dpitch, dyaw] 在传感器系
        """
        self.step_count += 1
        base_trans = self.get_base_forward_delta(phase)
        sensor_wiggle = self.get_sensor_wiggle_delta(phase)
        return base_trans, sensor_wiggle


# ============================================================================
# 5. 数据记录器 — 在信息丰富位姿记录标定数据
# ============================================================================

@dataclass
class CalibDataRecord:
    """单个标定数据记录"""
    T_B_H: np.ndarray      # 机器人手部位姿 (4×4)
    profile: dict          # 扫描线原始数据
    features: ProfileFeatures  # 提取的特征
    step: int              # 采集时的步数
    phase: str             # 采集时的阶段


class DataRecorder:
    """数据记录器: 基于信息度条件自动记录标定数据
    
    记录策略:
      - 不重复记录（相邻位姿差异太小不记）
      - 在wiggle的极值点记录（姿态变化最大时）
      - 确保覆盖边1和边2
    """
    
    def __init__(self, min_rot_diff: float = 0.08, min_trans_diff: float = 0.005):
        """
        Args:
            min_rot_diff: 最小旋转差异 (rad) 才记录新位姿
            min_trans_diff: 最小平移差异 (m) 才记录新位姿
        """
        self.records: List[CalibDataRecord] = []
        self.last_T_B_H = None
        self.min_rot_diff = min_rot_diff
        self.min_trans_diff = min_trans_diff
    
    def should_record(self, T_B_H: np.ndarray, features: ProfileFeatures) -> bool:
        """判断当前位姿是否值得记录"""
        # 必须有交线
        if not features.has_intersection or features.n_pts < 2:
            return False
        
        # 必须有至少一个断点
        if not features.has_e1 and not features.has_e2:
            return False
        
        # 第一个记录
        if self.last_T_B_H is None:
            return True
        
        # 检查与上一个记录的差异
        T_prev = self.last_T_B_H
        T_curr = T_B_H
        
        t_diff = np.linalg.norm(T_curr[:3, 3] - T_prev[:3, 3])
        
        R_diff_mat = T_curr[:3, :3] @ T_prev[:3, :3].T
        R_diff = np.arccos(np.clip((np.trace(R_diff_mat) - 1) / 2, -1, 1))
        
        return t_diff > self.min_trans_diff or R_diff > self.min_rot_diff
    
    def record(self, T_B_H: np.ndarray, sl: dict, features: ProfileFeatures,
               step: int, phase: str):
        """记录一个标定数据点"""
        rec = CalibDataRecord(
            T_B_H=T_B_H.copy(),
            profile=sl,
            features=features,
            step=step,
            phase=phase
        )
        self.records.append(rec)
        self.last_T_B_H = T_B_H.copy()
    
    @property
    def n_records(self) -> int:
        return len(self.records)
    
    def get_phase_records(self, phase: str) -> List[CalibDataRecord]:
        return [r for r in self.records if r.phase == phase]


# ============================================================================
# 6. 角点导航器 — 检测并绕过角点
# ============================================================================

@dataclass
class CornerNavParams:
    """角点导航参数 — 在角点处旋转90°从边1切到边2"""
    # 检测阈值
    both_edges_detect: float = 0.025    # 双断点时开始减速逼近 (25mm)
    corner_gap_min: float = 0.005        # 认为到达角点的最小断点间距 (5mm)
    turn_steps: int = 20                 # 90°旋转的步数
    
    # 旋转参数
    rotation_angle: float = np.pi / 2    # 90° 绕传感器Z轴旋转
    backoff_distance: float = 0.02       # 旋转前后退距离 (m)
    rotate_amplitude_ratio: float = 0.3  # 旋转时的wiggle衰减比


class CornerNavigator:
    """角点导航: 检测角点, 在角点处执行90°传感器旋转
    
    核心逻辑:
      Step 1: 检测双断点 → APPROACH_CORNER
      Step 2: 断点间距缩小到corner_gap_min → TURN_CORNER
      Step 3: 在TURN_CORNER中, 停止前进, 后退2cm, 绕传感器Z轴旋转90°
      Step 4: 旋转完成后 → ALONG_EDGE2
    """
    
    def __init__(self, scene: dict, params: CornerNavParams = None):
        self.scene = scene
        self.C = scene['C']
        self.u_B = scene['u_B']
        self.v_B = scene['v_B']
        self.params = params or CornerNavParams()
        
        self.corner_distance_history = []
        self.steps_both_edges = 0
    
    def detect_approach(self, features: ProfileFeatures) -> bool:
        """检测是否在逼近角点 (双断点且间距<阈值)"""
        if features.has_e1 and features.has_e2 and features.edge_gap is not None:
            if features.edge_gap < self.params.both_edges_detect:
                self.steps_both_edges += 1
                return self.steps_both_edges >= 3  # 连续3步确认
        else:
            self.steps_both_edges = max(0, self.steps_both_edges - 1)
        return False
    
    def detect_at_corner(self, features: ProfileFeatures) -> bool:
        """检测是否到达角点 (双断点且间距极小)"""
        if features.has_e1 and features.has_e2 and features.edge_gap is not None:
            return features.edge_gap < self.params.corner_gap_min
        return False
    
    def get_turn_delta(self, step_in_turn: int) -> tuple:
        """返回TURN_CORNER阶段的运动: (base_translation, sensor_rotation)
        
        策略: 
          - 前1/4步: 后退2cm (增加standoff防止脱离板面)
          - 中间: 绕传感器Z轴旋转90°
          - 后1/4步: 略微前进恢复standoff
        
        Returns:
            (base_translation_3d, sensor_rotation_6d)
        """
        p = self.params
        total = p.turn_steps
        t = min(step_in_turn / total, 1.0)
        
        # 后退: 在基坐标系沿-n_B方向
        backoff = np.zeros(3)
        if t < 0.25:  # 前1/4: 后退
            frac = t / 0.25
            backoff = -frac * p.backoff_distance * self.scene['n_B']
        elif t > 0.75:  # 后1/4: 恢复
            frac = (t - 0.75) / 0.25
            backoff = -(1 - frac) * p.backoff_distance * self.scene['n_B']
        
        # 旋转: 绕传感器Z轴
        rotation_progress = np.clip((t - 0.25) / 0.5, 0, 1)  # 在中间50%时间内完成旋转
        angle = rotation_progress * p.rotation_angle
        
        # 当前步的旋转增量 (不是总角度, 是当前步的增量)
        # 用正弦加速+减速曲线
        smooth = np.sin(np.pi * rotation_progress)  # 0→1→0
        step_angle = smooth * p.rotation_angle / (total * 0.5)
        
        sensor_rot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, step_angle])  # dyaw in sensor
        
        return backoff, sensor_rot


# ============================================================================
# 7. 顶层视觉伺服 — 状态机编排器
# ============================================================================

@dataclass
class EdgeVisualServoResult:
    """视觉伺服采集结果"""
    success: bool
    n_steps: int
    n_records: int
    records: List[CalibDataRecord]
    phase_history: List[str]
    final_pose: np.ndarray
    edge1_records: int = 0
    edge2_records: int = 0
    error_msg: str = ""


class EdgeVisualServo:
    """基于2D轮廓特征的边缘视觉伺服 — 顶层编排器
    
    全自动流程:
      1. SEARCH_EDGE:     搜索平板边缘
      2. ALONG_EDGE1:     沿边1推进 + 扭动采集
      3. APPROACH_CORNER: 逼近角点
      4. TURN_CORNER:    绕过角点
      5. ALONG_EDGE2:     沿边2推进 + 扭动采集
      6. DONE:            完成
      
    用法:
        servo = EdgeVisualServo(scene, T_S_H)
        result = servo.run(T_B_H_initial, max_steps=200)
    """
    
    # 阶段定义
    PHASES = [
        'SEARCH_EDGE',     # 0 — 找边缘
        'ALONG_EDGE1',     # 1 — 沿边1推进，朝向角点
        'APPROACH_CORNER', # 2 — 双断点出现，逼近角点
        'TURN_CORNER',     # 3 — 在角点处旋转90°，从边1切换到边2
        'ALONG_EDGE2',     # 4 — 沿边2推进
        'DONE',            # 5 — 完成
    ]
    
    def __init__(
        self,
        scene: dict,
        T_S_H: np.ndarray,
        edge_servo_params: EdgeServoParams = None,
        explorer_params: ActiveExplorerParams = None,
        corner_params: CornerNavParams = None,
    ):
        self.scene = scene
        self.T_S_H = T_S_H
        
        # 子系统
        self.edge_servo = EdgeServoController(scene, T_S_H, edge_servo_params)
        self.explorer = ActiveExplorer(explorer_params, u_B=scene['u_B'], v_B=scene['v_B'])
        self.recorder = DataRecorder()
        self.corner_nav = CornerNavigator(scene, corner_params)
        
        # 状态
        self.phase_idx = 0
        self.phase = self.PHASES[self.phase_idx]
        self.phase_history = []
        self.step = 0
        self.cross_step = 0
        self.stable_on_edge1_steps = 0  # 在边1上的稳定步数
        self.stable_on_edge2_steps = 0  # 在边2上的稳定步数
        self.max_stable_steps = 100     # 采集够就停
        
        # 可视化数据
        self.feature_history = []
        self.lost_counter = 0
    
    def get_scanline(self, T_B_H: np.ndarray) -> dict:
        """获取当前位姿下的扫描线"""
        T_B_S = T_B_H @ self.T_S_H
        R_BS = T_B_S[:3, :3]
        t_BS = T_B_S[:3, 3]
        
        return compute_fov_plate_scanline(
            R_BS, t_BS,
            self.scene['C'],
            self.scene['n_B'],
            self.scene['u_B'],
            self.scene['v_B'],
            self.scene['w'],
            self.scene['h'],
        )
    
    def phase_transition(self, features: ProfileFeatures, T_B_H: np.ndarray):
        """状态转移逻辑"""
        current = self.PHASES[self.phase_idx]
        
        # ---- 全局: 交线丢失恢复 ----
        # 如果在非搜索阶段连续5步无有效交线, 回退到搜索
        if current != 'SEARCH_EDGE' and not features.has_intersection:
            self.lost_counter += 1
            if self.lost_counter > 5:
                self._enter_phase('SEARCH_EDGE')
                self.lost_counter = 0
        elif features.has_intersection:
            self.lost_counter = 0
        
        if current == 'SEARCH_EDGE':
            # 搜索阶段: 直到看到一个断点
            if features.has_e1 or features.has_e2:
                self._enter_phase('ALONG_EDGE1')
        
        elif current == 'ALONG_EDGE1':
            # 沿边1推进: 检测角点逼近
            self.stable_on_edge1_steps += 1
            
            if self.corner_nav.detect_approach(features):
                self._enter_phase('APPROACH_CORNER')
        
        elif current == 'APPROACH_CORNER':
            # 逼近角点: 确认到达角点
            if self.corner_nav.detect_at_corner(features):
                self._enter_phase('TURN_CORNER')
                self.cross_step = 0
        
        elif current == 'TURN_CORNER':
            # 在角点处旋转
            self.cross_step += 1
            if self.cross_step >= self.corner_nav.params.turn_steps:
                self._enter_phase('ALONG_EDGE2')
                # 更新explorer的相位: 边2的边缘方向
                self.explorer = ActiveExplorer(ActiveExplorerParams(
                    pitch_amplitude=0.25, pitch_period=7,
                    yaw_amplitude=0.18, yaw_period=9,
                    forward_speed=self.explorer.params.forward_speed
                ), u_B=self.scene['u_B'], v_B=self.scene['v_B'])
        
        elif current == 'ALONG_EDGE2':
            # 沿边2推进
            self.stable_on_edge2_steps += 1
            if self.stable_on_edge2_steps > self.max_stable_steps:
                self._enter_phase('DONE')
        
        elif current == 'DONE':
            pass
    
    def _enter_phase(self, phase: str):
        """进入新阶段"""
        self.phase = phase
        self.phase_idx = self.PHASES.index(phase)
        self.phase_history.append(phase)
    
    def compute_servo_step(
        self, T_B_H: np.ndarray, features: ProfileFeatures
    ) -> tuple:
        """计算单步伺服动作
        
        Returns:
            (base_translation, sensor_delta)
            base_translation: [dx, dy, dz] 在基坐标系
            sensor_delta: [0,0,0, droll, dpitch, dyaw] 在传感器系
        """
        phase = self.phase
        
        # 1. 伺服基本动作: 保持FOV在边上 (传感器系)
        if phase in ('ALONG_EDGE1', 'APPROACH_CORNER', 'TURN_CORNER', 'ALONG_EDGE2'):
            if phase in ('ALONG_EDGE1', 'APPROACH_CORNER'):
                target_x = 0.005
            else:
                target_x = -0.005
            delta_servo = self.edge_servo.compute_servo_command(features, target_x)
        
        elif phase == 'SEARCH_EDGE':
            delta_servo = np.array([0.0, 0.0, -0.008, 0.0, 0.0, 0.0])
        else:
            delta_servo = np.zeros(6)
        
        # 2. 活跃探索: 基坐标系推进 + 传感器系wiggle
        base_trans, sensor_wiggle = self.explorer.step(phase)
        
        # 3. 角点过渡特殊处理
        if phase == 'TURN_CORNER':
            # 在角点处: 后退+旋转, 不从explorer拿推进
            base_turn, sensor_turn = self.corner_nav.get_turn_delta(self.cross_step)
            base_trans = base_turn  # 覆盖explorer的推进
            # 旋转增量加到sensor_delta里
        else:
            sensor_turn = np.zeros(6)
        
        # 4. 逼近角点时减速
        if phase == 'APPROACH_CORNER':
            delta_servo[:3] *= 0.3
            base_trans *= 0.3
        
        # 合成传感器系增量
        sensor_delta = np.zeros(6)
        sensor_delta[:3] = delta_servo[:3] * 0.5  # 伺服修正(横向)
        sensor_delta[3:6] = delta_servo[3:6] + sensor_wiggle[3:6]
        if phase == 'TURN_CORNER':
            sensor_delta[3:6] += sensor_turn[3:6]  # 叠加旋转
        
        # 限幅
        max_r = 0.25
        sensor_delta[3:6] = np.clip(sensor_delta[3:6], -max_r, max_r)
        sensor_delta[:3] = np.clip(sensor_delta[:3], -0.01, 0.01)
        
        return base_trans, sensor_delta
    
    def compute_servo_step_old(
        self, T_B_H: np.ndarray, features: ProfileFeatures
    ) -> np.ndarray:
        """旧版: 纯传感器系增量 (保留用于对比)"""
        delta = np.zeros(6)
        phase = self.phase
        
        if phase in ('ALONG_EDGE1', 'APPROACH_CORNER', 'TURN_CORNER', 'ALONG_EDGE2'):
            if phase in ('ALONG_EDGE1', 'APPROACH_CORNER'):
                target_x = 0.005
            else:
                target_x = -0.005
            delta = self.edge_servo.compute_servo_command(features, target_x)
        
        elif phase == 'SEARCH_EDGE':
            delta = np.array([0.0, 0.0, -0.008, 0.0, 0.0, 0.0])
        
        # 限幅
        delta[:3] = np.clip(delta[:3], -0.025, 0.025)
        delta[3:6] = np.clip(delta[3:6], -0.25, 0.25)
        return delta
    
    def run(
        self, T_B_H_initial: np.ndarray, max_steps: int = 300
    ) -> EdgeVisualServoResult:
        """运行视觉伺服采集流程
        
        Args:
            T_B_H_initial: 初始手部位姿 (4×4)
            max_steps: 最大伺服步数
            
        Returns:
            EdgeVisualServoResult
        """
        T_B_H = T_B_H_initial.copy()
        
        for step in range(max_steps):
            self.step = step
            
            # 1. 获取当前扫描线
            sl = self.get_scanline(T_B_H)
            features = extract_profile_features(sl)
            self.feature_history.append(features)
            
            # 2. 状态转移
            self.phase_transition(features, T_B_H)
            
            # 3. 检查是否完成
            if self.phase == 'DONE':
                break
            
            # 4. 计算伺服动作 (基坐标系平移 + 传感器系增量)
            base_trans, sensor_delta = self.compute_servo_step(T_B_H, features)
            
            # 5. 应用: 基坐标系平移 + 传感器系姿态调整
            # 基坐标系平移直接加到手部位姿的平移分量
            T_B_H_new = T_B_H.copy()
            T_B_H_new[:3, 3] += base_trans
            # 传感器系增量 (主要是姿态调制)
            T_B_H_new = incremental_motion_in_sensor_frame(
                T_B_H_new, self.T_S_H, sensor_delta)
            T_B_H = T_B_H_new
            
            # 6. 数据记录
            if self.recorder.should_record(T_B_H, features) and features.has_intersection:
                # 仅在信息丰富的位姿记录
                rec_features = extract_profile_features(sl)
                self.recorder.record(T_B_H, sl, rec_features, step, self.phase)
            
            # 7. 安全: 如果连续很多步无有效交线, 扩大搜索
            if step > 10 and self.phase == 'SEARCH_EDGE':
                # 退后更多
                pass  # servo already handles this
        
        # 汇总结果
        result = EdgeVisualServoResult(
            success=(len(self.recorder.records) >= 6),
            n_steps=step + 1,
            n_records=len(self.recorder.records),
            records=self.recorder.records,
            phase_history=self.phase_history,
            final_pose=T_B_H,
            error_msg=""
        )
        
        # 统计各边记录数
        result.edge1_records = len(self.recorder.get_phase_records('ALONG_EDGE1'))
        result.edge2_records = len(self.recorder.get_phase_records('ALONG_EDGE2'))
        
        return result


# ============================================================================
# 8. 可视化 — 伺服过程可视化
# ============================================================================

def visualize_trajectory(
    result: EdgeVisualServoResult,
    scene: dict,
    save_path: str = None
):
    """可视化伺服轨迹和采集点"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Edge Visual Servo — 2D Profile Based Navigation', fontsize=14)
    
    # 从记录中提取信息
    records = result.records
    n_records = len(records)
    
    if n_records == 0:
        plt.figtext(0.5, 0.5, 'No records to visualize', ha='center', fontsize=12)
        plt.savefig(save_path) if save_path else None
        plt.show()
        return
    
    # --- 图1: 采集点在板面上的投影 ---
    ax1 = axes[0, 0]
    C = scene['C']
    u_B = scene['u_B']
    v_B = scene['v_B']
    pw = scene['w']
    ph = scene['h']
    
    # 绘制平板
    corners = [
        C,
        C + pw * u_B,
        C + pw * u_B + ph * v_B,
        C + ph * v_B,
        C
    ]
    plate_pts = np.array([[p[0] for p in corners], [p[1] for p in corners], [p[2] for p in corners]])
    ax1.plot(plate_pts[0], plate_pts[1], 'k-', alpha=0.3, label='Plate')
    
    # 采集点
    if records:
        rec_positions = np.array([r.T_B_H[:3, 3] for r in records])
        ax1.scatter(rec_positions[:, 0], rec_positions[:, 1], 
                    c=range(len(rec_positions)), cmap='viridis', s=30, alpha=0.8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Robot Hand Positions ({n_records} records)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # --- 图2: 断点e1_x随步数变化 ---
    ax2 = axes[0, 1]
    all_features = result.phase_history
    
    # 提取特征历史 - 从record里取
    if records:
        steps = [r.step for r in records]
        e1_x_vals = [r.features.e1_x if r.features.has_e1 else np.nan for r in records]
        ax2.plot(steps, e1_x_vals, 'b-o', markersize=4, label='e1_x')
        
        e2_x_vals = [r.features.e2_x if r.features.has_e2 else np.nan for r in records]
        ax2.plot(steps, e2_x_vals, 'r-s', markersize=4, label='e2_x')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Endpoint x (m)')
    ax2.set_title('Endpoint X Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- 图3: z_mean 变化 ---
    ax3 = axes[0, 2]
    if records:
        z_means = [r.features.z_mean for r in records]
        ax3.plot(steps, z_means, 'g-o', markersize=4)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('z_mean (m)')
    ax3.set_title('Mean Standoff Distance')
    ax3.grid(True, alpha=0.3)
    
    # --- 图4: 相位变化 ---
    ax4 = axes[1, 0]
    phases = EdgeVisualServo.PHASES
    phase_colors = ['gray', 'blue', 'orange', 'green', 'red', 'black']
    
    if records:
        phase_nums = [phases.index(r.phase) for r in records]
        ax4.scatter(steps, phase_nums, c=phase_nums, cmap='tab10', s=50)
        ax4.set_yticks(range(len(phases)))
        ax4.set_yticklabels(phases)
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Phase')
    ax4.set_title('Phase Progression')
    ax4.grid(True, alpha=0.3)
    
    # --- 图5: 位置3D ---
    ax5 = axes[1, 1]
    if records:
        positions = np.array([r.T_B_H[:3, 3] for r in records])
        ax5.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=3, alpha=0.7)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z') if hasattr(ax5, 'set_zlabel') else None
    ax5.set_title('3D Trajectory')
    
    # --- 图6: 信息统计 ---
    ax6 = axes[1, 2]
    ax6.axis('off')
    info = (
        f"Results Summary\n"
        f"{'='*20}\n"
        f"Steps: {result.n_steps}\n"
        f"Records: {result.n_records}\n"
        f"Edge1 records: {result.edge1_records}\n"
        f"Edge2 records: {result.edge2_records}\n"
        f"Phase sequence: {' → '.join(result.phase_history)}\n"
        f"Success: {result.success}"
    )
    ax6.text(0.1, 0.5, info, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()


# ============================================================================
# 9. B模式: 更强力的探索 — 随机游走+伺服保持
# ============================================================================

@dataclass
class RandomExplorerParams:
    """随机探索参数 — 比正弦波更激进的探索"""
    max_pitch: float = 0.52       # ±30°
    max_yaw: float = 0.35         # ±20°
    change_interval: int = 4      # 每N步变化一次方向
    dwell_on_edge: bool = True    # 是否确保一直有边缘可见


class RandomExplorer(ActiveExplorer):
    """随机探索器: 随机改变pitch/yaw, 更大的探索范围"""
    
    def __init__(self, params=None, u_B=None, v_B=None):
        # 如果params不是RandomExplorerParams实例, 使用默认
        if params is None:
            params = RandomExplorerParams()
        # 调用父类init
        super().__init__(params, u_B, v_B)
        self.random_params = params
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.interp_steps = 0
    
    def get_sensor_wiggle_delta(self, phase: str) -> np.ndarray:
        p = self.random_params
        t = self.step_count
        
        if phase not in ('ALONG_EDGE1', 'ALONG_EDGE2'):
            return np.zeros(6)
        
        # 每 change_interval 步随机换一次方向
        if t % p.change_interval == 0:
            self.target_pitch = np.random.uniform(-p.max_pitch, p.max_pitch)
            self.target_yaw = np.random.uniform(-p.max_yaw, p.max_yaw)
            self.interp_steps = 0
        
        # 线性插值到目标
        alpha = min(self.interp_steps / p.change_interval, 1.0)
        self.interp_steps += 1
        
        pitch = self.current_pitch + alpha * (self.target_pitch - self.current_pitch)
        yaw = self.current_yaw + alpha * (self.target_yaw - self.current_yaw)
        
        delta = np.zeros(6)
        delta[4] = pitch * 0.3
        delta[5] = yaw * 0.3
        
        return delta


# ============================================================================
# 10. 演示入口 — 完整采集演示
# ============================================================================

def demo(search_mode: str = 'sinusoidal', max_steps: int = 200, seed: int = 42):
    """完整视觉伺服采集演示
    
    Args:
        search_mode: 'sinusoidal' — 正弦调制探索 (默认)
                     'random' — 随机探索 (更激进)
        max_steps: 最大伺服步数
        seed: 随机种子
    """
    rng = np.random.default_rng(seed)
    
    # 1. 生成场景
    print("=" * 60)
    print("Edge Visual Servo Demo")
    print("=" * 60)
    
    C, n_B, u_B, v_B, d_1, d_2, w_m, h_m = generate_corner_plane(
        rng, plate_w=400, plate_h=500, alpha=np.pi/2)
    w_m, h_m = 0.4, 0.5  # 统一用米
    
    scene = {
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'w': w_m, 'h': h_m
    }
    
    print(f"Plate size: {w_m:.2f}m × {h_m:.2f}m")
    print(f"Corner C: [{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}]")
    print(f"Normal n_B: [{n_B[0]:.3f}, {n_B[1]:.3f}, {n_B[2]:.3f}]")
    
    # 2. 生成手眼真值
    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]
    print(f"Hand-eye: R={so3_log(R_he)}°, t={t_he}")
    
    # 3. 生成初始位姿 — FOV在边1上（沿u_B方向，在板边缘）
    # 目标位姿: 传感器在边1稍微入板的位置
    from nbv_edge_plane import _build_R_edge
    
    # 沿边1的一个位姿: 从角点C沿u_B方向0.2m，稍偏v_B方向0.02m入板
    target_on_edge1 = C + 0.20 * u_B + 0.015 * v_B  # 略微切入板内
    
    # 构建一个能看到边1的传感器朝向
    R_S = _build_R_edge(pitch_deg=-15, yaw_deg=5, x_align=v_B,
                         n_B=n_B, u_B=u_B, v_B=v_B)
    standoff = 0.50
    sensor_pos = target_on_edge1 + standoff * n_B
    
    # 反推手部位姿
    t_B_S = sensor_pos
    R_B_S = R_S
    T_B_S = make_transform(R_B_S, t_B_S)
    T_S_H = X_gt
    T_B_H_init = T_B_S @ np.linalg.inv(T_S_H)
    
    # 验证初始位姿是否有有效扫描线
    sl_check = compute_fov_plate_scanline(
        R_B_S, t_B_S, C, n_B, u_B, v_B, w_m, h_m)
    features_check = extract_profile_features(sl_check)
    
    print(f"\nInitial pose check:")
    print(f"  Has intersection: {sl_check['has_intersection']}")
    print(f"  Plane points: {features_check.n_pts}")
    print(f"  Endpoints: e1={features_check.has_e1}, e2={features_check.has_e2}")
    if features_check.has_e1:
        print(f"  e1_x = {features_check.e1_x:.4f}m")
    if features_check.has_e2:
        print(f"  e2_x = {features_check.e2_x:.4f}m")
    
    if not sl_check['has_intersection']:
        print("\n⚠ Initial pose has no intersection with plate.")
        print("  Trying alternative initial placement...")
        # 尝试在板上多个位置找有效初始位姿
        for u_off in [0.30, 0.25, 0.15, 0.10]:
            for v_off in [0.03, 0.01, 0.005, -0.01]:
                for pitch in [-10, -15, -20]:
                    for yaw in [0, 5, 10]:
                        target = C + u_off * u_B + v_off * v_B
                        R_S_try = _build_R_edge(pitch, yaw, v_B, n_B, u_B, v_B)
                        sp = target + standoff * n_B
                        sl_try = compute_fov_plate_scanline(
                            R_S_try, sp, C, n_B, u_B, v_B, w_m, h_m)
                        if sl_try['has_intersection']:
                            R_B_S = R_S_try
                            t_B_S = sp
                            T_B_S = make_transform(R_B_S, t_B_S)
                            T_B_H_init = T_B_S @ np.linalg.inv(T_S_H)
                            sl_check = sl_try
                            features_check = extract_profile_features(sl_check)
                            print(f"  Found valid: u={u_off:.2f}, v={v_off:.03f}, pitch={pitch}, yaw={yaw}")
                            break
                    if sl_check['has_intersection']:
                        break
                if sl_check['has_intersection']:
                    break
            if sl_check['has_intersection']:
                break
    
    print("\n" + "=" * 60)
    print("Starting visual servo...")
    print("=" * 60)
    
    # 4. 创建伺服器并运行
    if search_mode == 'random':
        explorer = RandomExplorer()
    else:
        explorer = ActiveExplorer()
    
    servo = EdgeVisualServo(
        scene=scene,
        T_S_H=X_gt,
        explorer_params=explorer.params if search_mode == 'random' else None,
    )
    
    # 如果是random模式，替换explorer
    if search_mode == 'random':
        servo.explorer = explorer
    
    result = servo.run(T_B_H_init, max_steps=max_steps)
    
    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"Servo Result")
    print(f"{'='*60}")
    print(f"  Steps:     {result.n_steps}")
    print(f"  Records:   {result.n_records}")
    print(f"  Edge1:     {result.edge1_records}")
    print(f"  Edge2:     {result.edge2_records}")
    print(f"  Success:   {result.success}")
    print(f"  Phases:    {' → '.join(result.phase_history)}")
    
    if result.records:
        # 显示采集点的手眼估计质量
        print(f"\n  Recorded poses (hand position):")
        for i, rec in enumerate(result.records[:5]):
            pos = rec.T_B_H[:3, 3]
            print(f"    [{i}] step={rec.step}, phase={rec.phase}, "
                  f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        if len(result.records) > 5:
            print(f"    ... and {len(result.records) - 5} more")
    
    # 6. 可视化
    try:
        visualize_trajectory(result, scene)
    except Exception as e:
        print(f"Visualization error (non-fatal): {e}")
    
    return result


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Edge Visual Servo Demo')
    parser.add_argument('--mode', choices=['sinusoidal', 'random'], default='sinusoidal',
                       help='Exploration mode')
    parser.add_argument('--steps', type=int, default=200, help='Max servo steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    demo(search_mode=args.mode, max_steps=args.steps, seed=args.seed)
