#!/usr/bin/env python3
"""
core/types.py — 标定数据结构定义

统一所有模块使用的数据格式。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Pose:
    """法兰位姿: R_i ∈ SO(3), t_i ∈ R³ (在 base/世界坐标系)"""
    R: np.ndarray   # 3×3
    t: np.ndarray   # (3,)

    def SE3(self) -> np.ndarray:
        """4×4 齐次变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


@dataclass
class Measurement:
    """单次观测的测量数据

    传感器帧 S 下的坐标:
      - p_S_e1, p_S_e2: 激光线与标定板边的交点
      - scan_pts_S: 传感器帧下的整条轮廓线点云 [N, 3]
    """
    valid_e1: bool = False
    p_S_e1: Optional[np.ndarray] = None    # (3,) or None
    valid_e2: bool = False
    p_S_e2: Optional[np.ndarray] = None    # (3,) or None
    scan_pts_S: List[np.ndarray] = field(default_factory=list)  # list of (3,)

    def n_corners(self) -> int:
        """检测到的角点数量 (边交点)"""
        return int(self.valid_e1) + int(self.valid_e2)


@dataclass
class CalibRecord:
    """一次采集记录: 法兰位姿 + 测量 + (可选) 关节角"""
    pose: Pose
    meas: Measurement
    joints: Optional[np.ndarray] = None  # (6,) 弧度, J3_display, 噪声注入用


@dataclass
class SceneGT:
    """标定场景真值 (仿真环境已知)"""
    R_he: np.ndarray   # 3×3, 手眼旋转真值
    t_he: np.ndarray   # (3,), 手眼平移真值


@dataclass
class CalibResult:
    """标定结果"""
    method: str
    R_he: np.ndarray       # 3×3
    t_he: np.ndarray       # (3,)
    R_pl: Optional[np.ndarray] = None  # 3×3, 平板姿态 (9-DOF/12-DOF)
    C: Optional[np.ndarray] = None     # (3,), 角点位置 (12-DOF)
    cost: float = 0.0
    n_iter: int = 0
    converged: bool = True
    diagnostics: dict = field(default_factory=dict)

    def R_err_deg(self, gt: SceneGT) -> float:
        from handeye_sim.core.so3 import rotation_error_deg
        return rotation_error_deg(self.R_he, gt.R_he)

    def t_err_mm(self, gt: SceneGT) -> float:
        from handeye_sim.core.so3 import translation_error_mm
        return translation_error_mm(self.t_he, gt.t_he)

    def summary(self, gt: Optional[SceneGT] = None) -> str:
        s = f"[{self.method}] cost={self.cost:.3e}"
        if gt is not None:
            s += f"  R_err={self.R_err_deg(gt):.4f}°  t_err={self.t_err_mm(gt):.2f}mm"
        return s


@dataclass
class CalibData:
    """完整标定数据集"""
    records: List[CalibRecord] = field(default_factory=list)
    scene_gt: Optional[SceneGT] = None

    def n_poses(self) -> int:
        return len(self.records)

    def n_e1(self) -> int:
        return sum(1 for r in self.records if r.meas.valid_e1)

    def n_e2(self) -> int:
        return sum(1 for r in self.records if r.meas.valid_e2)

    def get_raw(self) -> Tuple[List[Tuple], List[dict]]:
        """导出为旧格式 (poses, meas) 供求解器使用"""
        poses = [(r.pose.R, r.pose.t) for r in self.records]
        meas = [{
            'valid_e1': r.meas.valid_e1,
            'p_S_e1': r.meas.p_S_e1,
            'valid_e2': r.meas.valid_e2,
            'p_S_e2': r.meas.p_S_e2,
            'p_S_plane': r.meas.scan_pts_S,
        } for r in self.records]
        return poses, meas
